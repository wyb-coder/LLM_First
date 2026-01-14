"""Build triple pool, global vocab, and true_matrix from extracted triples.

【溯源说明】
改造自：build_atom_pool.py（后半段）+ extract_base_embedding.py（嵌入计算逻辑）
原文件职能：
  build_atom_pool.py:
    - 构建全局原子池、计算 true_matrix（原子-样本关系矩阵）
    - 基于词频过滤低频原子，生成固定维度的原子嵌入表
  extract_base_embedding.py:
    - 提取 BERT 基础嵌入（CLS）供后续使用
    - 计算样本级别的上下文嵌入
本文件职能：
  - 基于 stage1 提取的三元组构建全局三元组词表（频率过滤）
  - 生成 per_sample_indices（每样本的三元组索引列表，top-K 截断）
  - 构建稀疏 true_matrix [num_triples, n_samples]，用于基线嵌入计算
核心改造：
  - 原子池：固定全局词汇 → 动态三元组（每样本 top-K）
  - true_matrix：密集 numpy → 稀疏 scipy CSR（防止内存爆炸）
  - 频率过滤：新增 --min_freq 参数过滤罕见三元组
  - 卡方过滤：新增 --filter_method chi2 按情感区分度过滤
  - 嵌入计算：基线用 true_matrix @ CLS 均值（可选文本编码器移至待拓展）

Inputs: train/valid/test triples pickles produced by `extract_triples.py`.
Outputs (under --out_dir):
- global_triple_vocab.pkl : list[str], filtered by min_freq/chi2, ordered by score
- true_matrix.npz         : scipy sparse CSR matrix [num_triples, n_data]
- per_sample_indices.pkl  : list[List[int]] triple indices per sample (top-K applied)
- filter_stats.json       : filtering statistics for debugging

Key changes from original:
- Uses scipy.sparse.lil_matrix to avoid memory explosion
- Adds --min_freq to filter rare triples (crucial for large datasets)
- Adds --filter_method chi2/tfidf for sentiment-discriminative filtering

Usage (yelp with chi-square filtering):
    python build_triple_pool.py \\
        --triples_dir result/triples \\
        --labels_csv data/yelp_review_polarity_csv/train.csv \\
        --max_triples 50 \\
        --min_freq 5 \\
        --filter_method chi2 \\
        --top_k 5000 \\
        --out_dir result/triples
"""
import argparse
import json
import os
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
from scipy import sparse
from scipy.stats import chi2_contingency

TRIPLE_SEP = " \u0001 "  # Separator used in extract_triples


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_labels(csv_path: str, label_col: int = 0, has_header: bool = False) -> List[int]:
    """Load labels from CSV file."""
    import csv
    labels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) > label_col:
                try:
                    labels.append(int(row[label_col].strip()))
                except ValueError:
                    continue
    return labels


def build_vocab_with_freq_filter(
    triples: List[List[Tuple[str, str, str]]], min_freq: int = 1
) -> Tuple[List[str], Counter]:
    """Build vocab filtered by minimum frequency."""
    counter = Counter()
    for ts in triples:
        counter.update([TRIPLE_SEP.join(t) for t in ts])
    
    # Filter by min_freq and sort by frequency descending
    filtered = [(t, c) for t, c in counter.most_common() if c >= min_freq]
    vocab = [t for t, _ in filtered]
    return vocab, counter


def compute_chi2_scores(
    triples: List[List[Tuple[str, str, str]]],
    labels: List[int],
    min_freq: int = 5,
) -> Dict[str, Tuple[float, float, int, int, int]]:
    """
    Compute chi-square scores for each triple.
    
    Returns:
        Dict[triple_str, (chi2_score, p_value, pos_count, neg_count, total_count)]
    """
    # Convert labels to binary (assuming 1=neg, 2=pos for Yelp format)
    # Or 0=neg, 1=pos for standard format
    unique_labels = set(labels)
    if unique_labels == {1, 2}:
        # Yelp format: 1=negative, 2=positive
        binary_labels = [1 if l == 2 else 0 for l in labels]
    else:
        # Standard format: assume max label is positive
        max_label = max(unique_labels)
        binary_labels = [1 if l == max_label else 0 for l in labels]
    
    n_samples = len(triples)
    n_pos = sum(binary_labels)
    n_neg = n_samples - n_pos
    
    print(f"  Chi2 computation: {n_samples} samples, {n_pos} positive, {n_neg} negative")
    
    # Count triple occurrences per class
    triple_pos_count: Counter = Counter()  # triple -> count in positive class
    triple_neg_count: Counter = Counter()  # triple -> count in negative class
    
    for i, ts in enumerate(triples):
        label = binary_labels[i]
        for t in ts:
            key = TRIPLE_SEP.join(t)
            if label == 1:
                triple_pos_count[key] += 1
            else:
                triple_neg_count[key] += 1
    
    # Compute chi2 for each triple
    all_triples = set(triple_pos_count.keys()) | set(triple_neg_count.keys())
    chi2_scores: Dict[str, Tuple[float, float, int, int, int]] = {}
    
    for triple_str in all_triples:
        a = triple_pos_count.get(triple_str, 0)  # triple & positive
        b = triple_neg_count.get(triple_str, 0)  # triple & negative
        total = a + b
        
        # Skip rare triples (mitigation: min_freq threshold)
        if total < min_freq:
            continue
        
        c = n_pos - a  # no triple & positive
        d = n_neg - b  # no triple & negative
        
        # Contingency table: [[a, b], [c, d]]
        # a = triple & pos, b = triple & neg
        # c = no triple & pos, d = no triple & neg
        contingency = np.array([[a, b], [c, d]])
        
        try:
            chi2, p_value, _, _ = chi2_contingency(contingency, correction=True)
        except ValueError:
            # Handle edge cases (e.g., zero rows/cols)
            chi2, p_value = 0.0, 1.0
        
        chi2_scores[triple_str] = (chi2, p_value, a, b, total)
    
    return chi2_scores


def filter_by_chi2(
    chi2_scores: Dict[str, Tuple[float, float, int, int, int]],
    top_k: int = 5000,
    p_threshold: float = 0.05,
    top_k_per_class: bool = True,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Filter triples by chi-square significance and return top-K.
    
    Mitigation measures:
    1. p_threshold: Only keep triples with p < threshold (statistically significant)
    2. top_k: Limit total number of triples
    3. top_k_per_class: Balance positive and negative triples
    
    Returns:
        (filtered_vocab, filter_reasons)
    """
    filter_reasons: Dict[str, str] = {}
    
    # First filter by p-value significance
    significant = []
    for triple_str, (chi2, p_val, pos_cnt, neg_cnt, total) in chi2_scores.items():
        if p_val >= p_threshold:
            filter_reasons[triple_str] = f"p_value={p_val:.4f} >= {p_threshold}"
            continue
        # Determine if it's more associated with positive or negative
        # Higher pos_cnt relative to expected -> positive indicator
        # Higher neg_cnt relative to expected -> negative indicator
        is_positive_indicator = pos_cnt > neg_cnt
        significant.append((triple_str, chi2, p_val, pos_cnt, neg_cnt, total, is_positive_indicator))
    
    print(f"  After p-value filter (p < {p_threshold}): {len(significant)} triples")
    
    if top_k_per_class and top_k > 0:
        # Split by class indicator and take top_k/2 from each
        half_k = top_k // 2
        
        pos_triples = [(t, chi2) for t, chi2, p, pc, nc, tot, is_pos in significant if is_pos]
        neg_triples = [(t, chi2) for t, chi2, p, pc, nc, tot, is_pos in significant if not is_pos]
        
        # Sort by chi2 score descending
        pos_triples.sort(key=lambda x: -x[1])
        neg_triples.sort(key=lambda x: -x[1])
        
        # Take top half_k from each
        selected_pos = [t for t, _ in pos_triples[:half_k]]
        selected_neg = [t for t, _ in neg_triples[:half_k]]
        
        # Mark filtered ones
        for t, chi2 in pos_triples[half_k:]:
            filter_reasons[t] = f"chi2={chi2:.2f} below top-{half_k} positive"
        for t, chi2 in neg_triples[half_k:]:
            filter_reasons[t] = f"chi2={chi2:.2f} below top-{half_k} negative"
        
        filtered_vocab = selected_pos + selected_neg
        print(f"  Balanced selection: {len(selected_pos)} positive, {len(selected_neg)} negative")
    else:
        # Just take global top_k by chi2
        significant.sort(key=lambda x: -x[1])  # Sort by chi2 descending
        if top_k > 0:
            selected = significant[:top_k]
            for item in significant[top_k:]:
                filter_reasons[item[0]] = f"chi2={item[1]:.2f} below top-{top_k}"
        else:
            selected = significant
        filtered_vocab = [item[0] for item in selected]
    
    # Sort final vocab by chi2 score for consistency
    vocab_with_scores = [(t, chi2_scores[t][0]) for t in filtered_vocab]
    vocab_with_scores.sort(key=lambda x: -x[1])
    filtered_vocab = [t for t, _ in vocab_with_scores]
    
    return filtered_vocab, filter_reasons


def cap_sample_indices(
    triples: List[List[Tuple[str, str, str]]],
    vocab2idx: Dict[str, int],
    max_triples: int
) -> List[List[int]]:
    """Convert triples to indices, capping per sample."""
    per_sample: List[List[int]] = []
    for ts in triples:
        idxs = []
        for t in ts:
            key = TRIPLE_SEP.join(t)
            if key in vocab2idx:
                idxs.append(vocab2idx[key])
        if max_triples > 0:
            idxs = idxs[:max_triples]
        per_sample.append(idxs)
    return per_sample


def build_sparse_true_matrix(per_sample: List[List[int]], num_triples: int) -> sparse.csr_matrix:
    """Build sparse CSR matrix [num_triples, n_data]."""
    n_data = len(per_sample)
    # Use lil_matrix for efficient incremental construction
    mat = sparse.lil_matrix((num_triples, n_data), dtype=np.int8)
    
    for j, idxs in enumerate(per_sample):
        for idx in idxs:
            mat[idx, j] = 1
    
    # Convert to CSR for efficient storage and operations
    return mat.tocsr()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_dir", type=str, default="./saved_models/triples")
    parser.add_argument("--use_vocab", type=str, default=None, help="Optional existing vocab pkl")
    parser.add_argument("--max_triples", type=int, default=50, help="Per-sample cap (0=unlimited)")
    parser.add_argument("--min_freq", type=int, default=5, help="Min frequency to keep triple in vocab")
    parser.add_argument("--out_dir", type=str, default="./saved_models/triples")
    
    # Chi-square filtering arguments
    parser.add_argument("--filter_method", type=str, default="none", 
                        choices=["none", "chi2", "tfidf"],
                        help="Filtering method: none (freq only), chi2 (chi-square), tfidf")
    parser.add_argument("--labels_csv", type=str, default=None,
                        help="CSV file with labels (required for chi2 filtering)")
    parser.add_argument("--label_col", type=int, default=0, help="Label column index in CSV")
    parser.add_argument("--csv_has_header", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--top_k", type=int, default=5000, 
                        help="Keep top-K triples by chi2/tfidf score (0=unlimited)")
    parser.add_argument("--p_threshold", type=float, default=0.05,
                        help="P-value threshold for chi2 significance")
    parser.add_argument("--balanced", type=lambda x: str(x).lower() == "true", default=True,
                        help="Balance top-K between positive and negative indicators")
    
    args = parser.parse_args()

    train_path = os.path.join(args.triples_dir, "train_triples.pkl")
    test_path = os.path.join(args.triples_dir, "test_triples.pkl")
    valid_path = os.path.join(args.triples_dir, "valid_triples.pkl")

    print("=" * 60)
    print("Build Triple Pool with Filtering")
    print("=" * 60)
    
    print("\nLoading triples...")
    train_triples = load_pickle(train_path)
    test_triples = load_pickle(test_path)
    valid_triples = load_pickle(valid_path) if os.path.exists(valid_path) else []
    
    n_train = len(train_triples)
    n_test = len(test_triples)
    n_valid = len(valid_triples)

    all_triples = train_triples + valid_triples + test_triples
    print(f"Total samples: {len(all_triples)} (train={n_train}, valid={n_valid}, test={n_test})")

    # Statistics for logging
    filter_stats = {
        "n_train": n_train,
        "n_test": n_test,
        "n_valid": n_valid,
        "filter_method": args.filter_method,
        "min_freq": args.min_freq,
        "top_k": args.top_k,
        "p_threshold": args.p_threshold,
    }

    if args.use_vocab:
        vocab = load_pickle(args.use_vocab)
        print(f"Using existing vocab: {len(vocab)} triples")
        filter_stats["vocab_source"] = "existing"
        filter_stats["vocab_size"] = len(vocab)
    else:
        # Step 1: Build initial vocab with frequency filter
        print(f"\nStep 1: Building vocab with min_freq={args.min_freq}...")
        vocab, counter = build_vocab_with_freq_filter(all_triples, min_freq=args.min_freq)
        print(f"  Original unique triples: {len(counter)}")
        print(f"  After freq filter (>={args.min_freq}): {len(vocab)}")
        
        filter_stats["original_unique"] = len(counter)
        filter_stats["after_freq_filter"] = len(vocab)
        
        # Step 2: Apply chi2/tfidf filtering if requested
        if args.filter_method == "chi2":
            if not args.labels_csv:
                raise ValueError("--labels_csv is required for chi2 filtering")
            
            print(f"\nStep 2: Chi-square filtering...")
            print(f"  Loading labels from {args.labels_csv}...")
            labels = load_labels(args.labels_csv, args.label_col, args.csv_has_header)
            
            if len(labels) != n_train:
                print(f"  WARNING: Label count ({len(labels)}) != train samples ({n_train})")
                labels = labels[:n_train]  # Truncate if necessary
            
            print(f"  Computing chi2 scores for {len(vocab)} triples...")
            chi2_scores = compute_chi2_scores(train_triples, labels, min_freq=args.min_freq)
            print(f"  Chi2 computed for {len(chi2_scores)} triples (after min_freq filter)")
            
            print(f"  Filtering by p < {args.p_threshold} and top_k={args.top_k}...")
            vocab, filter_reasons = filter_by_chi2(
                chi2_scores, 
                top_k=args.top_k, 
                p_threshold=args.p_threshold,
                top_k_per_class=args.balanced
            )
            
            print(f"  Final vocab size: {len(vocab)}")
            
            # Log filter reasons summary
            reason_counts = Counter()
            for reason in filter_reasons.values():
                # Simplify reason for counting
                if "p_value" in reason:
                    reason_counts["p_value_too_high"] += 1
                elif "below top" in reason:
                    reason_counts["below_top_k"] += 1
                else:
                    reason_counts["other"] += 1
            
            print(f"  Filter reasons: {dict(reason_counts)}")
            filter_stats["chi2_computed"] = len(chi2_scores)
            filter_stats["filter_reasons"] = dict(reason_counts)
            filter_stats["after_chi2_filter"] = len(vocab)
            
        elif args.filter_method == "tfidf":
            print("\nStep 2: TF-IDF filtering (not yet implemented, using freq filter only)")
            # TODO: Implement TF-IDF filtering
            filter_stats["tfidf_note"] = "not_implemented"
        else:
            print("\nStep 2: No additional filtering (using freq filter only)")
    
    vocab2idx = {t: i for i, t in enumerate(vocab)}
    filter_stats["final_vocab_size"] = len(vocab)

    print(f"\nStep 3: Building per-sample indices...")
    per_sample_indices = cap_sample_indices(all_triples, vocab2idx, args.max_triples)
    
    # Stats
    non_empty = sum(1 for idxs in per_sample_indices if len(idxs) > 0)
    avg_triples = np.mean([len(idxs) for idxs in per_sample_indices])
    print(f"  Non-empty samples: {non_empty}/{len(per_sample_indices)} ({100*non_empty/len(per_sample_indices):.1f}%)")
    print(f"  Average triples per sample: {avg_triples:.2f}")
    
    filter_stats["non_empty_samples"] = non_empty
    filter_stats["avg_triples_per_sample"] = float(avg_triples)

    print(f"\nStep 4: Building sparse true_matrix...")
    true_matrix = build_sparse_true_matrix(per_sample_indices, num_triples=len(vocab))
    
    # Memory stats
    mem_mb = (true_matrix.data.nbytes + true_matrix.indices.nbytes + true_matrix.indptr.nbytes) / 1024 / 1024
    print(f"  Sparse matrix shape: {true_matrix.shape}, nnz: {true_matrix.nnz}, memory: {mem_mb:.1f} MB")
    
    filter_stats["true_matrix_shape"] = list(true_matrix.shape)
    filter_stats["true_matrix_nnz"] = int(true_matrix.nnz)
    filter_stats["true_matrix_memory_mb"] = float(mem_mb)

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    save_pickle(vocab, os.path.join(args.out_dir, "global_triple_vocab.pkl"))
    save_pickle(per_sample_indices, os.path.join(args.out_dir, "per_sample_indices.pkl"))
    sparse.save_npz(os.path.join(args.out_dir, "true_matrix.npz"), true_matrix)
    
    # Save filter stats for debugging
    with open(os.path.join(args.out_dir, "filter_stats.json"), "w", encoding="utf-8") as f:
        json.dump(filter_stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Saved to {args.out_dir}:")
    print(f"  - global_triple_vocab.pkl: {len(vocab)} triples")
    print(f"  - per_sample_indices.pkl: {len(per_sample_indices)} samples")
    print(f"  - true_matrix.npz: sparse {true_matrix.shape}")
    print(f"  - filter_stats.json: filtering statistics")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
