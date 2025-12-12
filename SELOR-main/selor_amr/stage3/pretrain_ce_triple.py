"""Pretrain TripleConsequentEstimator on AMR triples (Stage3).

【溯源说明】
改造自：pretrain_consequent_estimator.py（原 SELOR CE 预训练脚本）
原文件职能：
  - 采样原子组合（1-4个原子）
  - 统计每个组合的经验概率分布 p(y|α) 和覆盖率
  - 用 MSE 损失训练 ConsequentEstimator 回归经验概率
本文件职能：
  - 采样三元组组合
  - 统计每个组合的经验概率分布和覆盖率
  - 用 MSE 损失训练 TripleConsequentEstimator
核心改造：
  - 原子 → 三元组
  - 保持训练目标（回归）与损失函数（MSE）不变

v2.0: Fixed to match original SELOR approach:
  - Training target: empirical probability distribution (not discrete labels)
  - Loss: MSE (not NLLLoss)  
  - Metrics: Mu MAE, Mu RMSE, Coverage MAE, Argmax Accuracy
"""
import argparse
import os
import pickle
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Local imports
from selor_utils.net import TripleConsequentEstimator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="CSV with labels")
    parser.add_argument("--csv_has_header", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--label_col", type=str, default="0", help="Label column name or index")
    parser.add_argument("--triples_dir", type=str, default="result/triples", help="Stage1/2 output dir")
    parser.add_argument("--train_embedding", type=str, required=True, help="Torch tensor of train CLS embeddings")
    parser.add_argument("--antecedent_len", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of triple combinations to sample per length")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="result/ce_triple")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_offset", type=int, default=0, help="Subtract from labels (e.g., 1 for Yelp)")
    parser.add_argument("--valid_split", type=float, default=0.1, help="Fraction for validation (0 to disable)")
    parser.add_argument("--min_coverage", type=int, default=5, help="Min samples a combination must cover")
    return parser.parse_args()


def reset_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(triples_dir: str, train_csv: str, header, label_col: str, label_offset: int):
    """Load triples, true_matrix, and labels."""
    train_triples = pickle.load(open(os.path.join(triples_dir, "train_triples.pkl"), "rb"))
    per_sample = pickle.load(open(os.path.join(triples_dir, "per_sample_indices.pkl"), "rb"))
    true_matrix = sparse.load_npz(os.path.join(triples_dir, "true_matrix.npz"))
    
    n_train = len(train_triples)
    train_indices = per_sample[:n_train]
    
    # Load labels
    df = pd.read_csv(train_csv, header=header)
    col = int(label_col) if str(label_col).isdigit() else label_col
    labels = df[col].astype(int).tolist()
    if label_offset != 0:
        labels = [l - label_offset for l in labels]
    
    return train_indices, true_matrix, labels, n_train


def build_triple_embedding(true_matrix: sparse.csr_matrix, train_embed: torch.Tensor) -> torch.Tensor:
    """Compute triple embeddings via normalized true_matrix @ train_embed."""
    n_train = train_embed.shape[0]
    tm_train = true_matrix[:, :n_train].tocsr()
    counts = np.array(tm_train.sum(axis=1)).reshape(-1, 1) + 1e-8
    norm_tm = tm_train.multiply(1.0 / counts)
    triple_emb_np = norm_tm.dot(train_embed.cpu().numpy())
    return torch.tensor(triple_emb_np, dtype=torch.float32)


def precompute_triple_coverage(true_matrix: sparse.csr_matrix, n_train: int) -> Dict[int, set]:
    """Precompute which samples each triple covers (as sets for fast intersection)."""
    print("  Precomputing triple coverage sets...")
    num_triples = true_matrix.shape[0]
    coverage_sets = {}
    
    tm_train = true_matrix[:, :n_train].tocsr()
    for triple_idx in tqdm(range(num_triples), desc="  Building coverage sets"):
        row = tm_train.getrow(triple_idx)
        covered_samples = set(row.indices)
        if len(covered_samples) > 0:
            coverage_sets[triple_idx] = covered_samples
    
    print(f"  {len(coverage_sets)} triples have non-empty coverage")
    return coverage_sets


def compute_empirical_distribution_fast(
    combination: Tuple[int, ...],
    coverage_sets: Dict[int, set],
    labels: List[int],
    n_train: int,
    num_classes: int
) -> Tuple[np.ndarray, int]:
    """Fast version using precomputed coverage sets."""
    # Find samples that contain ALL triples (set intersection)
    covered = None
    for triple_idx in combination:
        if triple_idx not in coverage_sets:
            return np.ones(num_classes) / num_classes, 0
        if covered is None:
            covered = coverage_sets[triple_idx].copy()
        else:
            covered &= coverage_sets[triple_idx]
    
    n = len(covered) if covered else 0
    if n == 0:
        return np.ones(num_classes) / num_classes, 0
    
    # Count labels
    label_counts = np.zeros(num_classes)
    for idx in covered:
        label_counts[labels[idx]] += 1
    
    mu = label_counts / n
    return mu, n


def sample_combinations_from_samples(
    train_indices: List[List[int]],
    antecedent_len: int,
    num_samples: int,
    coverage_sets: Dict[int, set],
    labels: List[int],
    n_train: int,
    num_classes: int,
    min_coverage: int,
    seed: int
) -> List[Dict]:
    """Sample triple combinations FROM actual samples (guarantees coverage).
    
    Strategy: Instead of randomly combining triples (which rarely co-occur),
    we sample combinations from within each sample's triple set. This guarantees
    at least 1 sample coverage per combination.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Collect samples that have enough triples
    valid_sample_ids = []
    for sample_id, triple_list in enumerate(train_indices):
        if len(triple_list) >= antecedent_len:
            valid_sample_ids.append(sample_id)
    
    print(f"  {len(valid_sample_ids)} samples have >= {antecedent_len} triples")
    
    if len(valid_sample_ids) == 0:
        print("Warning: No samples have enough triples!")
        return []
    
    samples = []
    seen_combinations = set()
    attempts = 0
    max_attempts = num_samples * 50
    
    print(f"Sampling {num_samples} combinations of length {antecedent_len}...")
    pbar = tqdm(total=num_samples)
    
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Pick a random sample
        sample_id = random.choice(valid_sample_ids)
        triple_list = train_indices[sample_id]
        
        # Sample antecedent_len triples from this sample
        if len(triple_list) < antecedent_len:
            continue
        
        selected = random.sample(triple_list, antecedent_len)
        combination = tuple(sorted(selected))
        
        # Skip if already seen
        if combination in seen_combinations:
            continue
        seen_combinations.add(combination)
        
        # Compute empirical distribution
        mu, n = compute_empirical_distribution_fast(
            combination, coverage_sets, labels, n_train, num_classes
        )
        
        # Skip if coverage too low
        if n < min_coverage:
            continue
        
        samples.append({
            'combination': combination,
            'mu': mu,
            'n': n,
            'coverage': n / n_train
        })
        pbar.update(1)
    
    pbar.close()
    print(f"Sampled {len(samples)} valid combinations (attempts: {attempts})")
    
    # Statistics
    if samples:
        coverages = [s['n'] for s in samples]
        print(f"  Coverage stats: min={min(coverages)}, max={max(coverages)}, avg={np.mean(coverages):.1f}")
    
    return samples


class TripleCombinationDataset(Dataset):
    """Dataset of triple combinations with empirical probability targets."""
    
    def __init__(self, samples: List[Dict], triple_emb_table: torch.Tensor, antecedent_len: int):
        self.samples = samples
        self.triple_emb_table = triple_emb_table
        self.antecedent_len = antecedent_len
        self.hidden_dim = triple_emb_table.shape[1]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        combination = sample['combination']
        mu = sample['mu']
        coverage = sample['coverage']
        
        # Build embedding sequence
        emb_list = []
        for triple_idx in combination:
            if 0 <= triple_idx < len(self.triple_emb_table):
                emb_list.append(self.triple_emb_table[triple_idx])
            else:
                emb_list.append(torch.zeros(self.hidden_dim))
        
        # Pad if needed
        while len(emb_list) < self.antecedent_len:
            emb_list.append(torch.zeros(self.hidden_dim))
        
        emb_seq = torch.stack(emb_list, dim=0)
        mu_tensor = torch.tensor(mu, dtype=torch.float32)
        coverage_tensor = torch.tensor(coverage, dtype=torch.float32)
        
        return emb_seq, mu_tensor, coverage_tensor


def collate_fn(batch):
    embs, mus, coverages = zip(*batch)
    return (
        torch.stack(embs, dim=0),
        torch.stack(mus, dim=0),
        torch.stack(coverages, dim=0)
    )


def train_epoch(model, loader, optimizer, device, n_data: int):
    """Train one epoch using MSE loss on empirical probabilities."""
    model.train()
    total_mu_loss = 0.0
    total_coverage_loss = 0.0
    
    for emb_seq, mu_target, coverage_target in loader:
        emb_seq = emb_seq.to(device)
        mu_target = mu_target.to(device)
        coverage_target = coverage_target.to(device)
        
        mu_pred, sigma_pred, coverage_pred = model(emb_seq)
        
        # MSE loss on probability distribution (main objective)
        mu_loss = F.mse_loss(mu_pred, mu_target)
        
        # MSE loss on coverage (auxiliary objective)
        coverage_loss = F.mse_loss(coverage_pred, coverage_target)
        
        # Total loss
        loss = mu_loss + 0.1 * coverage_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_mu_loss += mu_loss.item() * emb_seq.size(0)
        total_coverage_loss += coverage_loss.item() * emb_seq.size(0)
    
    n = len(loader.dataset)
    return total_mu_loss / n, total_coverage_loss / n


def eval_epoch(model, loader, device, n_data: int) -> Dict[str, float]:
    """Evaluate with metrics matching original SELOR eval_pretrain.
    
    Original SELOR returns: avg_mu_err, avg_sigma_err, avg_coverage_err, f1_score
    We match these for direct comparison.
    """
    model.eval()
    
    all_mu_pred = []
    all_mu_target = []
    all_sigma_pred = []
    all_coverage_pred = []
    all_coverage_target = []
    
    with torch.no_grad():
        for emb_seq, mu_target, coverage_target in loader:
            emb_seq = emb_seq.to(device)
            mu_target = mu_target.to(device)
            coverage_target = coverage_target.to(device)
            
            mu_pred, sigma_pred, coverage_pred = model(emb_seq)
            
            all_mu_pred.append(mu_pred.cpu())
            all_mu_target.append(mu_target.cpu())
            all_sigma_pred.append(sigma_pred.cpu())
            all_coverage_pred.append(coverage_pred.cpu())
            all_coverage_target.append(coverage_target.cpu())
    
    mu_pred = torch.cat(all_mu_pred, dim=0)
    mu_target = torch.cat(all_mu_target, dim=0)
    sigma_pred = torch.cat(all_sigma_pred, dim=0)
    coverage_pred = torch.cat(all_coverage_pred, dim=0)
    coverage_target = torch.cat(all_coverage_target, dim=0)
    
    # === Original SELOR metrics (for comparison) ===
    # avg_mu_err: mean absolute error (L1) on mu
    avg_mu_err = torch.mean(torch.abs(mu_pred - mu_target)).item()
    
    # avg_sigma_err: for sigma, we don't have ground truth in our setup
    # Original SELOR compares to statistical sigma, we set to 0 for now
    avg_sigma_err = 0.0
    
    # avg_coverage_err: mean absolute error on coverage
    avg_coverage_err = torch.mean(torch.abs(coverage_pred - coverage_target)).item()
    
    # F1: argmax classification F1 (same as original SELOR)
    pred_class = torch.argmax(mu_pred, dim=-1)
    target_class = torch.argmax(mu_target, dim=-1)
    
    # Compute F1 manually (macro average)
    from sklearn.metrics import f1_score as sk_f1
    f1 = sk_f1(target_class.numpy(), pred_class.numpy(), average='macro', zero_division=0)
    
    return {
        'avg_mu_err': avg_mu_err,
        'avg_sigma_err': avg_sigma_err,
        'avg_coverage_err': avg_coverage_err,
        'f1': f1
    }


def main():
    args = parse_args()
    reset_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    header = 0 if args.csv_has_header else None
    
    print("=" * 60)
    print("Stage 3: Pretrain TripleConsequentEstimator (v2.0)")
    print("=" * 60)
    
    print("\n[1/5] Loading data...")
    train_indices, true_matrix, labels, n_train = load_data(
        args.triples_dir, args.train_csv, header, args.label_col, args.label_offset
    )
    num_classes = len(set(labels))
    print(f"  Samples: {n_train}, Classes: {num_classes}")
    
    print("\n[2/5] Loading embeddings and building triple embedding table...")
    train_emb = torch.load(args.train_embedding)
    triple_emb_table = build_triple_embedding(true_matrix, train_emb)
    hidden_dim = triple_emb_table.shape[1]
    print(f"  Triple embedding table: {triple_emb_table.shape}")
    
    print(f"\n[3/5] Sampling {args.num_samples} triple combinations...")
    # Precompute coverage sets for fast sampling
    coverage_sets = precompute_triple_coverage(true_matrix, n_train)
    
    all_samples = sample_combinations_from_samples(
        train_indices, args.antecedent_len, args.num_samples,
        coverage_sets, labels, n_train, num_classes, args.min_coverage, args.seed
    )
    
    if len(all_samples) < 100:
        print("Warning: Too few valid samples. Check min_coverage or antecedent_len.")
        return
    
    # Split into train/valid
    if args.valid_split > 0:
        split_idx = int(len(all_samples) * (1 - args.valid_split))
        train_samples = all_samples[:split_idx]
        valid_samples = all_samples[split_idx:]
        print(f"  Train: {len(train_samples)}, Valid: {len(valid_samples)}")
    else:
        train_samples = all_samples
        valid_samples = []
    
    print("\n[4/5] Creating datasets and dataloaders...")
    train_ds = TripleCombinationDataset(train_samples, triple_emb_table, args.antecedent_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    valid_loader = None
    if valid_samples:
        valid_ds = TripleCombinationDataset(valid_samples, triple_emb_table, args.antecedent_len)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("\n[5/5] Training...")
    model = TripleConsequentEstimator(num_classes=num_classes, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "ce_triple_best.pt")
    best_mu_mae = float("inf")
    
    # Print header matching original SELOR eval_pretrain output
    print(f"\n{'Epoch':<6} {'Mu_Loss':<10} {'Cov_Loss':<10} | {'avg_mu_err':<10} {'avg_cov_err':<10} {'F1':<10} {'Best'}")
    print("-" * 75)
    
    for epoch in range(1, args.epochs + 1):
        mu_loss, cov_loss = train_epoch(model, train_loader, optimizer, device, n_train)
        
        if valid_loader is not None:
            metrics = eval_epoch(model, valid_loader, device, n_train)
            improved = metrics['avg_mu_err'] < best_mu_mae
            if improved:
                best_mu_mae = metrics['avg_mu_err']
                torch.save(model.state_dict(), save_path)
            print(f"{epoch:<6} {mu_loss:<10.4f} {cov_loss:<10.4f} | {metrics['avg_mu_err']:<10.4f} {metrics['avg_coverage_err']:<10.4f} {metrics['f1']:<10.4f} {'*' if improved else ''}")
        else:
            torch.save(model.state_dict(), save_path)
            print(f"{epoch:<6} {mu_loss:<10.4f} {cov_loss:<10.4f}")
    
    # Save config
    config = {
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
        "antecedent_len": args.antecedent_len,
        "num_samples": args.num_samples,
        "min_coverage": args.min_coverage,
    }
    with open(os.path.join(args.save_dir, "ce_triple_config.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    print(f"\nSaved best model to {save_path}")
    print(f"Best validation Mu MAE: {best_mu_mae:.4f}")


if __name__ == "__main__":
    main()
