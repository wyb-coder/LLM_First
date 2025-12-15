"""Stage5: Evaluation and explanation export for AMR-SELOR.

Loads the trained AMR-SELOR (Stage4) checkpoint and the frozen CE, runs
evaluation on the test split, and writes metrics + per-sample predictions
and explanations.

Dependencies: reuse key utilities from stage4 (dataset, emb builder, model).
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
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Reuse Stage4 components
from selor_amr.stage4.amr_selor import (
    AMRSELOR,
    TripleDataset,
    build_triple_embedding,
    collate_fn,
    filter_empty_samples,
    load_labels,
    load_texts,
    reset_seed,
)
from selor_utils.net import TripleConsequentEstimator


class EvalTripleDataset(TripleDataset):
    """TripleDataset with raw text preserved for export."""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["raw_text"] = self.texts[idx]
        return item


def collate_fn_eval(batch):
    base = collate_fn(batch)
    base["raw_texts"] = [b["raw_text"] for b in batch]
    return base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, default=None)
    parser.add_argument("--csv_has_header", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--text_col", type=str, default="1")
    parser.add_argument("--label_col", type=str, default="0")
    parser.add_argument("--label_offset", type=int, default=0)
    parser.add_argument("--triples_dir", type=str, default="result/triples")
    parser.add_argument("--train_embedding", type=str, default="result/embeddings/train_cls.pt")
    parser.add_argument("--ce_path", type=str, default="result/ce_triple/ce_triple_best.pt")
    parser.add_argument("--ce_config", type=str, default="result/ce_triple/ce_triple_config.pkl")
    parser.add_argument("--model_path", type=str, default="result/amr_selor/amr_selor_best.pt")
    parser.add_argument("--antecedent_len", type=int, default=3)
    parser.add_argument("--max_triples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="result/amr_selor_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_split", type=float, default=0.0, help="Split ratio from test set (optional)")
    return parser.parse_args()


def load_vocab(triples_dir: str) -> Optional[List[str]]:
    vocab_path = os.path.join(triples_dir, "global_triple_vocab.pkl")
    if os.path.exists(vocab_path):
        vocab = pickle.load(open(vocab_path, "rb"))
        assert isinstance(vocab, list), f"Expected vocab to be list, got {type(vocab)}"
        return vocab
    return None


def idx_to_triple_text(idx: int, vocab: Optional[List[str]]):
    if vocab is None:
        return f"idx:{idx}"
    if idx < 0 or idx >= len(vocab):
        return f"idx:{idx}"
    return vocab[idx]


@torch.no_grad()
def evaluate_and_collect(model, loader, triple_emb_table, device, num_classes, vocab=None):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_texts = []
    all_explanations = []
    all_coverages = []
    all_sigmas = []

    pbar = tqdm(loader, desc="Evaluating", unit="batch")
    for batch in pbar:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        class_prob, mu, sigma, coverage, select_probs = model(batch, triple_emb_table)
        labels = batch["labels"]
        loss = torch.nn.functional.nll_loss(torch.log(class_prob + 1e-8), labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        pred = torch.argmax(class_prob, dim=-1)
        correct += (pred == labels).sum().item()
        total += bs

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(class_prob.cpu().numpy())

        triple_indices = batch["triple_indices"]
        triple_mask = batch["triple_mask"]
        select_probs_cpu = select_probs.cpu()
        for i in range(bs):
            T = triple_indices.size(1)
            chosen = torch.argmax(select_probs_cpu[i], dim=-1)
            expl = []
            for j, t_idx in enumerate(chosen):
                if t_idx >= T:
                    continue
                global_idx = triple_indices[i, t_idx].item()
                if triple_mask[i, t_idx].item():
                    expl.append(idx_to_triple_text(global_idx, vocab))
            all_explanations.append(" AND ".join(expl))

        # Collect coverage and sigma for uncertainty analysis
        all_coverages.extend(coverage.cpu().numpy().flatten().tolist())
        all_sigmas.extend(sigma.cpu().numpy().tolist())

        all_texts.extend(batch["raw_texts"])

    avg_loss = total_loss / total
    acc = correct / total

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)

    report = classification_report(all_labels_np, all_preds_np, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]

    roc_auc = 0.0
    pr_auc = 0.0
    if num_classes == 2:
        try:
            roc_auc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
            pr_auc = average_precision_score(all_labels_np, all_probs_np[:, 1])
        except Exception:
            roc_auc, pr_auc = 0.0, 0.0
    else:
        try:
            roc_auc = roc_auc_score(all_labels_np, all_probs_np, multi_class="ovr", average="macro")
        except Exception:
            roc_auc = 0.0

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "samples": int(total),
    }

    outputs = {
        "text": all_texts,
        "label": all_labels_np.tolist(),
        "pred": all_preds_np.tolist(),
        "probs": all_probs_np.tolist(),
        "explanation": all_explanations,
        "coverage": all_coverages,
        "sigma": all_sigmas,
    }

    return metrics, outputs


def main():
    args = parse_args()
    reset_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    header = 0 if args.csv_has_header else None

    per_sample_path = os.path.join(args.triples_dir, "per_sample_indices.pkl")
    if not os.path.exists(per_sample_path):
        raise FileNotFoundError(f"Missing per_sample_indices.pkl at {per_sample_path}")
    per_sample = pickle.load(open(per_sample_path, "rb"))
    true_matrix = sparse.load_npz(os.path.join(args.triples_dir, "true_matrix.npz"))
    vocab = load_vocab(args.triples_dir)

    train_emb = torch.load(args.train_embedding)
    triple_emb_table = build_triple_embedding(true_matrix, train_emb).to(device)
    hidden_dim = triple_emb_table.shape[1]

    train_labels = load_labels(args.train_csv, header, args.label_col, args.label_offset)
    test_labels_all = load_labels(args.test_csv, header, args.label_col, args.label_offset)
    train_texts = load_texts(args.train_csv, header, args.text_col)
    test_texts_all = load_texts(args.test_csv, header, args.text_col)

    n_train = len(train_labels)
    n_test_all = len(test_labels_all)
    train_indices = per_sample[:n_train]
    test_indices_all = per_sample[n_train: n_train + n_test_all]

    if args.valid_split > 0:
        from sklearn.model_selection import train_test_split

        indices = list(range(n_test_all))
        test_idx, valid_idx = train_test_split(
            indices, test_size=args.valid_split, random_state=args.seed, stratify=test_labels_all
        )
        test_labels = [test_labels_all[i] for i in test_idx]
        test_texts = [test_texts_all[i] for i in test_idx]
        test_indices = [test_indices_all[i] for i in test_idx]
        valid_labels = [test_labels_all[i] for i in valid_idx]
        valid_texts = [test_texts_all[i] for i in valid_idx]
        valid_indices = [test_indices_all[i] for i in valid_idx]
        print(f"Split test set: {len(test_labels)} test, {len(valid_labels)} valid")
    else:
        test_labels, test_texts, test_indices = test_labels_all, test_texts_all, test_indices_all
        valid_labels, valid_texts, valid_indices = [], [], []

    train_texts, train_labels, train_indices = filter_empty_samples(train_texts, train_labels, train_indices, "train")
    test_texts, test_labels, test_indices = filter_empty_samples(test_texts, test_labels, test_indices, "test")
    if len(valid_labels) > 0:
        valid_texts, valid_labels, valid_indices = filter_empty_samples(valid_texts, valid_labels, valid_indices, "valid")

    num_classes = len(set(train_labels))

    def make_dataset(texts, labels, indices):
        return EvalTripleDataset(texts, labels, indices, tokenizer, max_len=512, max_triples=args.max_triples)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    train_ds = make_dataset(train_texts, train_labels, train_indices)
    test_ds = make_dataset(test_texts, test_labels, test_indices)
    valid_ds = make_dataset(valid_texts, valid_labels, valid_indices) if len(valid_labels) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_eval)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_eval)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_eval) if valid_ds else None

    ce_state = torch.load(args.ce_path, map_location=device)
    ce_model = TripleConsequentEstimator(num_classes=num_classes, hidden_dim=hidden_dim).to(device)
    ce_model.load_state_dict(ce_state, strict=False)

    model = AMRSELOR(
        ce_model=ce_model,
        hidden_dim=hidden_dim,
        antecedent_len=args.antecedent_len,
        n_data=len(train_labels),
        num_classes=num_classes,
    ).to(device)
    model.set_text_encoder(tokenizer, bert_model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    os.makedirs(args.save_dir, exist_ok=True)

    if valid_loader is not None:
        val_metrics, _ = evaluate_and_collect(model, valid_loader, triple_emb_table, device, num_classes, vocab)
        print("Valid metrics:", val_metrics)

    test_metrics, outputs = evaluate_and_collect(model, test_loader, triple_emb_table, device, num_classes, vocab)
    print("Test metrics:", test_metrics)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    df_out = pd.DataFrame({
        "text": outputs["text"],
        "label": outputs["label"],
        "pred": outputs["pred"],
        "explanation": outputs["explanation"],
    })
    if num_classes == 2:
        probs_pos = [p[1] for p in outputs["probs"]]
        df_out["prob_pos"] = probs_pos
    else:
        df_out["probs"] = [json.dumps(p) for p in outputs["probs"]]
    
    # Add coverage and sigma for uncertainty analysis
    df_out["coverage"] = outputs["coverage"]
    df_out["sigma"] = [json.dumps(s) for s in outputs["sigma"]]
    df_out.to_csv(os.path.join(args.save_dir, "predictions.csv"), index=False)
    print(f"Saved metrics and predictions to {args.save_dir}")


if __name__ == "__main__":
    main()