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
  - 嵌入计算：基线用 true_matrix @ CLS 均值（可选文本编码器移至待拓展）

Inputs: train/valid/test triples pickles produced by `extract_triples.py`.
Outputs (under --out_dir):
- global_triple_vocab.pkl : list[str], filtered by min_freq, ordered by freq
- true_matrix.npz         : scipy sparse CSR matrix [num_triples, n_data]
- per_sample_indices.pkl  : list[List[int]] triple indices per sample (top-K applied)

Key changes from original:
- Uses scipy.sparse.lil_matrix to avoid memory explosion
- Adds --min_freq to filter rare triples (crucial for large datasets)

Usage (yelp):
    python build_triple_pool.py \
        --triples_dir result/triples \
        --max_triples 50 \
        --min_freq 5 \
        --out_dir result/triples
"""
import argparse
import os
import pickle
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
from scipy import sparse

TRIPLE_SEP = " \u0001 "  # Separator used in extract_triples


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


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
    args = parser.parse_args()

    train_path = os.path.join(args.triples_dir, "train_triples.pkl")
    test_path = os.path.join(args.triples_dir, "test_triples.pkl")
    valid_path = os.path.join(args.triples_dir, "valid_triples.pkl")

    print("Loading triples...")
    train_triples = load_pickle(train_path)
    test_triples = load_pickle(test_path)
    valid_triples = load_pickle(valid_path) if os.path.exists(valid_path) else []

    all_triples = train_triples + valid_triples + test_triples
    print(f"Total samples: {len(all_triples)}")

    if args.use_vocab:
        vocab = load_pickle(args.use_vocab)
        print(f"Using existing vocab: {len(vocab)} triples")
    else:
        print(f"Building vocab with min_freq={args.min_freq}...")
        vocab, counter = build_vocab_with_freq_filter(all_triples, min_freq=args.min_freq)
        print(f"Original unique triples: {len(counter)}")
        print(f"After freq filter (>={args.min_freq}): {len(vocab)}")
    
    vocab2idx = {t: i for i, t in enumerate(vocab)}

    print("Building per-sample indices...")
    per_sample_indices = cap_sample_indices(all_triples, vocab2idx, args.max_triples)
    
    # Stats
    non_empty = sum(1 for idxs in per_sample_indices if len(idxs) > 0)
    avg_triples = np.mean([len(idxs) for idxs in per_sample_indices])
    print(f"Non-empty samples: {non_empty}/{len(per_sample_indices)} ({100*non_empty/len(per_sample_indices):.1f}%)")
    print(f"Average triples per sample: {avg_triples:.2f}")

    print("Building sparse true_matrix...")
    true_matrix = build_sparse_true_matrix(per_sample_indices, num_triples=len(vocab))
    
    # Memory stats
    mem_mb = (true_matrix.data.nbytes + true_matrix.indices.nbytes + true_matrix.indptr.nbytes) / 1024 / 1024
    print(f"Sparse matrix shape: {true_matrix.shape}, nnz: {true_matrix.nnz}, memory: {mem_mb:.1f} MB")

    os.makedirs(args.out_dir, exist_ok=True)
    save_pickle(vocab, os.path.join(args.out_dir, "global_triple_vocab.pkl"))
    save_pickle(per_sample_indices, os.path.join(args.out_dir, "per_sample_indices.pkl"))
    sparse.save_npz(os.path.join(args.out_dir, "true_matrix.npz"), true_matrix)

    print(f"\nSaved to {args.out_dir}:")
    print(f"  - global_triple_vocab.pkl: {len(vocab)} triples")
    print(f"  - per_sample_indices.pkl: {len(per_sample_indices)} samples")
    print(f"  - true_matrix.npz: sparse {true_matrix.shape}")


if __name__ == "__main__":
    main()
