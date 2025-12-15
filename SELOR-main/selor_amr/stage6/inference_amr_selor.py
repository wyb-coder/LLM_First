"""Stage6: Inference / batch prediction for AMR-SELOR.

用途：对已有三元组索引的 CSV 做推理并导出预测与解释；不包含 AMR 解析。
假设 per_sample_indices 与输入样本顺序一致，可用 --start_index 指定偏移。
"""
import argparse
import json
import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from selor_amr.stage4.amr_selor import (
    AMRSELOR,
    TripleDataset,
    build_triple_embedding,
    collate_fn,
    filter_empty_samples,
    load_texts,
    reset_seed,
)
from selor_utils.net import TripleConsequentEstimator


class InferTripleDataset(TripleDataset):
    def __init__(self, texts: List[str], indices: List[List[int]], tokenizer, max_len: int, max_triples: int):
        dummy_labels = [0] * len(texts)  # labels not used; placeholder
        super().__init__(texts, dummy_labels, indices, tokenizer, max_len, max_triples)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["raw_text"] = self.texts[idx]
        return item


def collate_fn_infer(batch):
    base = collate_fn(batch)
    base["raw_texts"] = [b["raw_text"] for b in batch]
    return base


def load_vocab(triples_dir: str) -> Optional[List[str]]:
    path = os.path.join(triples_dir, "global_triple_vocab.pkl")
    if os.path.exists(path):
        vocab = pickle.load(open(path, "rb"))
        assert isinstance(vocab, list), f"Expected vocab to be list, got {type(vocab)}"
        return vocab
    return None


def idx_to_triple_text(idx: int, vocab: Optional[List[str]]):
    if vocab is None or idx < 0 or (vocab and idx >= len(vocab)):
        return f"idx:{idx}"
    return vocab[idx]


@torch.no_grad()
def run_inference(model, loader, triple_emb_table, device, vocab=None):
    model.eval()
    preds = []
    probs = []
    texts = []
    explanations = []
    coverages = []
    sigma_means = []

    pbar = tqdm(loader, desc="Infer", unit="batch")
    for batch in pbar:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        class_prob, mu, sigma, coverage, select_probs = model(batch, triple_emb_table)
        pred = torch.argmax(class_prob, dim=-1)

        preds.extend(pred.cpu().numpy())
        probs.extend(class_prob.cpu().numpy())
        coverages.extend(coverage.squeeze(-1).cpu().numpy().tolist())
        sigma_means.extend(sigma.mean(dim=-1).cpu().numpy().tolist())
        texts.extend(batch["raw_texts"])

        triple_indices = batch["triple_indices"]
        triple_mask = batch["triple_mask"]
        select_probs_cpu = select_probs.cpu()
        for i in range(triple_indices.size(0)):
            T = triple_indices.size(1)
            chosen = torch.argmax(select_probs_cpu[i], dim=-1)
            expl = []
            for t_idx in chosen:
                if t_idx >= T:
                    continue
                global_idx = triple_indices[i, t_idx].item()
                if triple_mask[i, t_idx].item():
                    expl.append(idx_to_triple_text(global_idx, vocab))
            explanations.append(" AND ".join(expl))

    return {
        "pred": preds,
        "probs": probs,
        "text": texts,
        "explanation": explanations,
        "coverage": coverages,
        "sigma_mean": sigma_means,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--output_csv", type=str, default="result/amr_selor_infer/predictions.csv")
    p.add_argument("--csv_has_header", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--text_col", type=str, default="1")
    p.add_argument("--triples_dir", type=str, default="result/triples")
    p.add_argument("--train_embedding", type=str, default="result/embeddings/train_cls.pt")
    p.add_argument("--ce_path", type=str, default="result/ce_triple/ce_triple_best.pt")
    p.add_argument("--model_path", type=str, default="result/amr_selor/amr_selor_best.pt")
    p.add_argument("--antecedent_len", type=int, default=3)
    p.add_argument("--max_triples", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start_index", type=int, default=0, help="offset inside per_sample_indices")
    return p.parse_args()


def main():
    args = parse_args()
    reset_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    header = 0 if args.csv_has_header else None
    texts = load_texts(args.input_csv, header, args.text_col)

    per_sample = pickle.load(open(os.path.join(args.triples_dir, "per_sample_indices.pkl"), "rb"))
    true_matrix = sparse.load_npz(os.path.join(args.triples_dir, "true_matrix.npz"))
    vocab = load_vocab(args.triples_dir)

    start = args.start_index
    end = start + len(texts)
    if end > len(per_sample):
        raise ValueError(f"per_sample_indices size={len(per_sample)} is smaller than requested slice [{start}, {end})")
    indices = per_sample[start:end]

    texts, dummy_labels, indices = filter_empty_samples(texts, [0]*len(texts), indices, "infer")

    train_emb = torch.load(args.train_embedding)
    triple_emb_table = build_triple_embedding(true_matrix, train_emb).to(device)
    hidden_dim = triple_emb_table.shape[1]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    ds = InferTripleDataset(texts, indices, tokenizer, max_len=512, max_triples=args.max_triples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_infer)

    ce_state = torch.load(args.ce_path, map_location=device)
    num_classes = 2  # Inference不需要标签，假设二分类；若需多类请改为训练时类别数
    ce_model = TripleConsequentEstimator(num_classes=num_classes, hidden_dim=hidden_dim).to(device)
    ce_model.load_state_dict(ce_state, strict=False)

    model = AMRSELOR(
        ce_model=ce_model,
        hidden_dim=hidden_dim,
        antecedent_len=args.antecedent_len,
        n_data=len(texts),
        num_classes=num_classes,
    ).to(device)
    model.set_text_encoder(tokenizer, bert_model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    outputs = run_inference(model, loader, triple_emb_table, device, vocab)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame({
        "text": outputs["text"],
        "pred": outputs["pred"],
        "explanation": outputs["explanation"],
        "coverage": outputs["coverage"],
        "sigma_mean": outputs["sigma_mean"],
    })
    # 写出概率
    probs = outputs["probs"]
    if probs and isinstance(probs[0], (list, np.ndarray)) and len(probs[0]) == 2:
        df["prob_pos"] = [p[1] for p in probs]
    else:
        df["probs"] = [json.dumps(p) for p in probs]

    df.to_csv(args.output_csv, index=False)
    print(f"Saved inference results to {args.output_csv}")


if __name__ == "__main__":
    main()