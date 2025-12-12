"""Train AMR-SELOR (Stage 4) with GRU+mask selector and frozen TripleConsequentEstimator.

【溯源说明】
改造自：selor.py（原 SELOR 主训练脚本）
原功能：加载已预训的 ConsequentEstimator，使用 AtomSelector 训练 AG+CE 联合模型，输出分类与解释。
本文件职能：
  - 载入 Stage1/2 的三元组池与嵌入（true_matrix @ CLS）
  - 载入 Stage3 预训的 TripleConsequentEstimator（冻结）
  - 采用 GRU+mask 选择三元组（指针/文本编码为可选增强，未启用）
  - 训练分类模型并生成三元组解释

基线差异：原子 → 三元组；动态候选池；嵌入查表来自 true_matrix@CLS；其他流程与 SELOR 保持一致（NLL 分类损失，Gumbel-Softmax 选择）。
"""
import argparse
import os
import pickle
import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from selor_utils.net import TripleConsequentEstimator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--csv_has_header", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--text_col", type=str, default="1")
    parser.add_argument("--label_col", type=str, default="0")
    parser.add_argument("--label_offset", type=int, default=0)
    parser.add_argument("--triples_dir", type=str, default="result/triples")
    parser.add_argument("--train_embedding", type=str, default="result/embeddings/train_cls.pt")
    parser.add_argument("--ce_path", type=str, default="result/ce_triple/ce_triple_best.pt")
    parser.add_argument("--ce_config", type=str, default="result/ce_triple/ce_triple_config.pkl")
    parser.add_argument("--antecedent_len", type=int, default=3)
    parser.add_argument("--max_triples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.95, help="Learning rate decay factor (like original SELOR)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="result/amr_selor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_split", type=float, default=0.5, help="Split ratio from test set for validation (like original SELOR)")
    return parser.parse_args()


def reset_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(path: str, header: Optional[int], col, offset: int) -> List[int]:
    df = pd.read_csv(path, header=header)
    col = int(col) if str(col).isdigit() else col
    labels = df[col].astype(int).tolist()
    if offset != 0:
        labels = [l - offset for l in labels]
    return labels


def load_texts(path: str, header: Optional[int], col) -> List[str]:
    df = pd.read_csv(path, header=header)
    col = int(col) if str(col).isdigit() else col
    return df[col].astype(str).tolist()


def build_triple_embedding(true_matrix: sparse.csr_matrix, train_embed: torch.Tensor) -> torch.Tensor:
    n_train = train_embed.shape[0]
    tm_train = true_matrix[:, :n_train].tocsr()
    counts = np.array(tm_train.sum(axis=1)).reshape(-1, 1) + 1e-8
    norm_tm = tm_train.multiply(1.0 / counts)
    triple_emb_np = norm_tm.dot(train_embed.cpu().numpy())
    return torch.tensor(triple_emb_np, dtype=torch.float32)


class TripleDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], sample_indices: List[List[int]], tokenizer, max_len: int, max_triples: int):
        self.texts = texts
        self.labels = labels
        self.sample_indices = sample_indices
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_triples = max_triples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        idxs = self.sample_indices[idx] if idx < len(self.sample_indices) else []
        idxs = idxs[: self.max_triples]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "triple_indices": torch.tensor(idxs, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    triple_lists = [b["triple_indices"] for b in batch]
    max_len = max(t.size(0) for t in triple_lists) if triple_lists else 0
    padded = []
    masks = []
    for t in triple_lists:
        pad_len = max_len - t.size(0)
        if pad_len > 0:
            t = torch.cat([t, torch.full((pad_len,), -1, dtype=torch.long)])
        mask = (t != -1)
        t = torch.clamp(t, min=0)
        padded.append(t)
        masks.append(mask)
    triple_indices = torch.stack(padded) if padded else torch.zeros((len(batch), 0), dtype=torch.long)
    triple_mask = torch.stack(masks) if masks else torch.zeros((len(batch), 0), dtype=torch.bool)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "triple_indices": triple_indices,
        "triple_mask": triple_mask,
    }


class GRUMaskedSelector(nn.Module):
    """GRU+mask selector adapted to dynamic triple pools (baseline)."""

    def __init__(self, hidden_dim: int, antecedent_len: int):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.antecedent_len = antecedent_len

    def forward(self, cls_emb: torch.Tensor, triple_emb: torch.Tensor, triple_mask: torch.Tensor, training: bool) -> torch.Tensor:
        # cls_emb: [B, H]; triple_emb: [B, T, H]; triple_mask: [B, T]
        B, T, H = triple_emb.shape
        cur_input = cls_emb.unsqueeze(0)
        cur_h = None
        probs = []
        for step in range(self.antecedent_len):
            if cur_h is not None:
                _, cur_h = self.gru(cur_input, cur_h)
            else:
                _, cur_h = self.gru(cur_input)
            h = self.dropout(cur_h[-1])  # [B, H]
            scores = torch.bmm(triple_emb, h.unsqueeze(-1)).squeeze(-1)  # [B, T]
            scores = scores.masked_fill(~triple_mask, float('-inf'))
            if training:
                prob = F.gumbel_softmax(scores, tau=1.0, hard=True, dim=-1)
            else:
                prob = torch.zeros_like(scores)
                idx = torch.argmax(scores, dim=-1)
                prob.scatter_(1, idx.unsqueeze(-1), 1.0)
            probs.append(prob)
            # selected embedding as next input
            sel = torch.bmm(prob.unsqueeze(1), triple_emb).squeeze(1)  # [B, H]
            cur_input = (cls_emb + sel).unsqueeze(0)
        return torch.stack(probs, dim=1)  # [B, L, T]


class AMRSELOR(nn.Module):
    def __init__(self, ce_model: nn.Module, hidden_dim: int, antecedent_len: int, n_data: int, num_classes: int):
        super().__init__()
        self.tokenizer = None
        self.bert = None
        self.selector = GRUMaskedSelector(hidden_dim, antecedent_len)
        self.ce_model = ce_model
        for p in self.ce_model.parameters():
            p.requires_grad = False
        self.antecedent_len = antecedent_len
        self.n_data = n_data
        self.num_classes = num_classes
        # Laplace smoothing parameter (like original SELOR)
        self.alpha = nn.Parameter(torch.ones(1))

    def set_text_encoder(self, tokenizer, bert_model):
        self.tokenizer = tokenizer
        self.bert = bert_model

    def forward(self, batch, triple_emb_table: torch.Tensor):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        triple_indices = batch["triple_indices"]
        triple_mask = batch["triple_mask"]

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :]

        # gather triple embeddings
        B, T = triple_indices.shape
        flat = triple_indices.view(-1)
        gathered = triple_emb_table[flat]
        triple_emb = gathered.view(B, T, -1)

        select_probs = self.selector(cls_emb, triple_emb, triple_mask, self.training)
        # expected embedding per step: select_probs [B, L, T], triple_emb [B, T, H]
        L = self.antecedent_len
        H = triple_emb.shape[-1]
        # Expand triple_emb to [B, L, T, H] then reshape to [B*L, T, H]
        triple_emb_expanded = triple_emb.unsqueeze(1).expand(B, L, T, H).reshape(B * L, T, H)
        # select_probs: [B, L, T] -> [B*L, 1, T]
        selected_emb = torch.bmm(select_probs.view(B * L, 1, T), triple_emb_expanded)
        selected_emb = selected_emb.view(B, L, -1)  # [B, L, H]

        mu, sigma, coverage = self.ce_model(selected_emb)
        
        # Laplace smoothing (like original SELOR)
        n = coverage * self.n_data  # [B, 1]
        smooth = self.alpha / (n + 1e-8)  # [B, 1]
        smooth = smooth.expand(-1, self.num_classes)  # [B, num_classes]
        class_prob = (mu + smooth) / (1 + self.num_classes * smooth)
        
        return class_prob, mu, sigma, coverage, select_probs


def train_epoch(model, loader, triple_emb_table, optimizer, device, n_classes):
    """Train for one epoch."""
    model.train()
    total = 0.0
    correct = 0
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        class_prob, mu, sigma, coverage, select_probs = model(batch, triple_emb_table)
        labels = batch["labels"]
        # Use class_prob (with Laplace smoothing) for loss
        loss = F.nll_loss(torch.log(class_prob + 1e-8), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        pred = torch.argmax(class_prob, dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, triple_emb_table, device, n_classes, return_metrics=False):
    """Evaluate model with rich metrics (like original SELOR)."""
    model.eval()
    total = 0.0
    correct = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            class_prob, mu, sigma, coverage, select_probs = model(batch, triple_emb_table)
            labels = batch["labels"]
            loss = F.nll_loss(torch.log(class_prob + 1e-8), labels)
            total_loss += loss.item() * labels.size(0)
            pred = torch.argmax(class_prob, dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(class_prob.cpu().numpy())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    if return_metrics:
        import numpy as np
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Classification report (Macro-F1)
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        macro_f1 = report["macro avg"]["f1-score"]
        
        # ROC-AUC and PR-AUC (for binary classification)
        if n_classes == 2:
            try:
                roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
                pr_auc = average_precision_score(all_labels, all_probs[:, 1])
            except:
                roc_auc, pr_auc = 0.0, 0.0
        else:
            # Multi-class: use macro average
            try:
                roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                pr_auc = 0.0  # Not directly applicable for multi-class
            except:
                roc_auc, pr_auc = 0.0, 0.0
        
        return avg_loss, accuracy, macro_f1, roc_auc, pr_auc
    
    return avg_loss, accuracy


def main():
    args = parse_args()
    reset_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    header = 0 if args.csv_has_header else None

    # Load triples and matrices
    per_sample = pickle.load(open(os.path.join(args.triples_dir, "per_sample_indices.pkl"), "rb"))
    true_matrix = sparse.load_npz(os.path.join(args.triples_dir, "true_matrix.npz"))

    # Load embeddings
    train_emb = torch.load(args.train_embedding)
    triple_emb_table = build_triple_embedding(true_matrix, train_emb).to(device)
    hidden_dim = triple_emb_table.shape[1]

    # Labels and texts
    train_labels = load_labels(args.train_csv, header, args.label_col, args.label_offset)
    test_labels_all = load_labels(args.test_csv, header, args.label_col, args.label_offset)
    train_texts = load_texts(args.train_csv, header, args.text_col)
    test_texts_all = load_texts(args.test_csv, header, args.text_col)

    # Slice per-sample indices
    n_train = len(train_labels)
    n_test_all = len(test_labels_all)
    train_indices = per_sample[:n_train]
    test_indices_all = per_sample[n_train: n_train + n_test_all]

    # Split test set into test and valid (like original SELOR: 50% each)
    if args.valid_split > 0:
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
        print(f"  Split test set: {len(test_labels)} test, {len(valid_labels)} valid")
    else:
        test_labels, test_texts, test_indices = test_labels_all, test_texts_all, test_indices_all
        valid_labels, valid_texts, valid_indices = [], [], []
        print(f"  No validation split, using all {len(test_labels)} samples for test")

    n_valid = len(valid_labels)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    train_ds = TripleDataset(train_texts, train_labels, train_indices, tokenizer, max_len=512, max_triples=args.max_triples)
    test_ds = TripleDataset(test_texts, test_labels, test_indices, tokenizer, max_len=512, max_triples=args.max_triples)
    valid_ds = TripleDataset(valid_texts, valid_labels, valid_indices, tokenizer, max_len=512, max_triples=args.max_triples) if n_valid > 0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if valid_ds else None

    # Load CE
    ce_state = torch.load(args.ce_path, map_location=device)
    ce_cfg = pickle.load(open(args.ce_config, "rb")) if os.path.exists(args.ce_config) else {}
    num_classes = len(set(train_labels))
    ce_model = TripleConsequentEstimator(num_classes=num_classes, hidden_dim=hidden_dim).to(device)
    ce_model.load_state_dict(ce_state, strict=False)

    # Create model with n_data and num_classes for Laplace smoothing
    model = AMRSELOR(
        ce_model=ce_model, 
        hidden_dim=hidden_dim, 
        antecedent_len=args.antecedent_len,
        n_data=n_train,
        num_classes=num_classes
    ).to(device)
    model.set_text_encoder(tokenizer, bert_model)

    # Optimizer and LR scheduler (like original SELOR)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "amr_selor_best.pt")

    print(f"\n{'='*80}")
    print(f"Training AMR-SELOR (like original SELOR)")
    print(f"  Train samples: {n_train}, Valid: {n_valid}, Test: {len(test_labels)}")
    print(f"  Epochs: {args.epochs}, LR: {args.learning_rate}, Gamma: {args.gamma}")
    print(f"{'='*80}\n")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, triple_emb_table, optimizer, device, num_classes)
        
        # Validate with rich metrics
        if valid_loader is not None:
            val_loss, val_acc, val_f1, val_roc, val_pr = evaluate(
                model, valid_loader, triple_emb_table, device, num_classes, return_metrics=True
            )
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)
            mark = "*" if improved else ""
            print(f"Epoch {epoch:2d} | Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f} "
                  f"ROC={val_roc:.4f} PR={val_pr:.4f} {mark}")
        else:
            torch.save(model.state_dict(), best_path)
            print(f"Epoch {epoch:2d} | Train Loss={train_loss:.4f} Acc={train_acc:.4f}")
        
        # Step LR scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if epoch % 5 == 0:
            print(f"  [LR adjusted to {current_lr:.2e}]")

    # Load best and test with rich metrics
    print(f"\n{'='*80}")
    print("Final Evaluation on Test Set")
    print(f"{'='*80}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_f1, test_roc, test_pr = evaluate(
        model, test_loader, triple_emb_table, device, num_classes, return_metrics=True
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")
    print(f"Test ROC-AUC: {test_roc:.4f}")
    print(f"Test PR-AUC: {test_pr:.4f}")

    # Save results
    with open(os.path.join(args.save_dir, "metrics.txt"), "w") as f:
        f.write(f"test_loss={test_loss:.4f}\n")
        f.write(f"test_acc={test_acc:.4f}\n")
        f.write(f"test_macro_f1={test_f1:.4f}\n")
        f.write(f"test_roc_auc={test_roc:.4f}\n")
        f.write(f"test_pr_auc={test_pr:.4f}\n")
    print(f"\nResults saved to {args.save_dir}/metrics.txt")


if __name__ == "__main__":
    main()