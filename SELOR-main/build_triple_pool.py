"""Build triple pool, global vocab, and true_matrix from extracted triples.

Inputs: train/valid/test triples pickles produced by `extract_triples.py`.
Outputs (under --out_dir, default ./saved_models/triples):
- global_triple_vocab.pkl : list[str], ordered by freq (same as input if provided)
- true_matrix.npz         : sparse-ish numpy matrix [num_triples, n_data] of {0,1}
- per_sample_indices.pkl  : list[List[int]] triple indices per sample (top-K applied)

Usage (yelp defaults):
    python build_triple_pool.py \
        --triples_dir ./saved_models/triples \
        --max_triples 50

Note: This implementation uses dense numpy arrays for simplicity. For very large
corpuses, replace with scipy.sparse to save memory.
"""
import argparse
import os
import pickle
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np

TRIPLE_SEP = " \u0001 "  # Separator used in extract_triples


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def build_vocab(triples: List[List[Tuple[str, str, str]]]) -> List[str]:
    counter = Counter()
    for ts in triples:
        counter.update([TRIPLE_SEP.join(t) for t in ts])
    return [t for t, _ in counter.most_common()]


def cap_sample_indices(triples: List[List[Tuple[str, str, str]]], vocab2idx: Dict[str, int], max_triples: int) -> List[List[int]]:
    per_sample: List[List[int]] = []
    for ts in triples:
        idxs = [vocab2idx[TRIPLE_SEP.join(t)] for t in ts if TRIPLE_SEP.join(t) in vocab2idx]
        if max_triples > 0:
            idxs = idxs[:max_triples]
        per_sample.append(idxs)
    return per_sample


def build_true_matrix(per_sample: List[List[int]], num_triples: int) -> np.ndarray:
    n_data = len(per_sample)
    mat = np.zeros((num_triples, n_data), dtype=np.int8)
    for j, idxs in enumerate(per_sample):
        if not idxs:
            continue
        mat[idxs, j] = 1
    return mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_dir", type=str, default="./saved_models/triples")
    parser.add_argument("--use_vocab", type=str, default=None, help="Optional existing vocab pkl")
    parser.add_argument("--max_triples", type=int, default=50, help="Per-sample cap (0=unlimited)")
    parser.add_argument("--out_dir", type=str, default="./saved_models/triples")
    args = parser.parse_args()

    train_path = os.path.join(args.triples_dir, "train_triples.pkl")
    test_path = os.path.join(args.triples_dir, "test_triples.pkl")
    valid_path = os.path.join(args.triples_dir, "valid_triples.pkl")

    train_triples = load_pickle(train_path)
    test_triples = load_pickle(test_path)
    valid_triples = load_pickle(valid_path) if os.path.exists(valid_path) else []

    all_triples = train_triples + valid_triples + test_triples

    if args.use_vocab:
        vocab = load_pickle(args.use_vocab)
    else:
        vocab = build_vocab(all_triples)
    vocab2idx = {t: i for i, t in enumerate(vocab)}

    # Per-sample indices (cap applied)
    per_sample_indices = cap_sample_indices(all_triples, vocab2idx, args.max_triples)

    true_matrix = build_true_matrix(per_sample_indices, num_triples=len(vocab))

    os.makedirs(args.out_dir, exist_ok=True)
    save_pickle(vocab, os.path.join(args.out_dir, "global_triple_vocab.pkl"))
    save_pickle(per_sample_indices, os.path.join(args.out_dir, "per_sample_indices.pkl"))
    np.savez_compressed(os.path.join(args.out_dir, "true_matrix.npz"), true_matrix=true_matrix)

    print("Saved vocab/per-sample indices/true_matrix to", args.out_dir)
    print("Vocab size:", len(vocab), "Samples:", len(per_sample_indices))


if __name__ == "__main__":
    main()
