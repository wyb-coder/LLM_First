"""Extract AMR triples from pre-parsed CSV files.

This script assumes you already have AMR strings in CSV columns (e.g., `train_with_amr.csv`).
It decodes AMR in PENMAN format, extracts filtered triples, and saves:
- train/valid/test triples as pickle lists
- a global triple vocab (ordered by frequency)

Usage example (yelp default paths):
    python extract_triples.py \
        --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
        --test_csv  data/yelp_review_polarity_csv/test_with_amr.csv \
        --amr_col amr

Requirements: penman, pandas
"""
import argparse
import os
import pickle
from collections import Counter
from typing import List, Tuple, Dict

import pandas as pd
import penman


KeepRelations = {
    ":ARG0",
    ":ARG1",
    ":ARG2",
    ":mod",
    ":manner",
    ":domain",
    ":location",
    ":time",
    ":polarity",
}


def get_concept(graph: penman.Graph, var: str) -> str:
    """Resolve a variable to its concept string."""
    for triple in graph.triples:
        if triple[0] == var and triple[1] == ":instance":
            return str(triple[2])
    return str(var)


def normalize_role(role: str) -> str:
    """Normalize :*-of roles to forward direction (e.g., :ARG0-of -> :ARG0)."""
    return role[:-3] if role.endswith("-of") else role


def extract_triples_from_amr(amr_str: str) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    try:
        graph = penman.decode(amr_str)
    except Exception:
        return triples

    for src, role, tgt in graph.triples:
        role_norm = normalize_role(role)
        if role_norm not in KeepRelations:
            continue
        head = get_concept(graph, src)
        tail = get_concept(graph, tgt)
        triples.append((head, role_norm, tail))
    return triples


def read_csv_and_extract(path: str, amr_col: str) -> List[List[Tuple[str, str, str]]]:
    df = pd.read_csv(path)
    if amr_col not in df.columns:
        raise ValueError(f"Column {amr_col} not found in {path}")
    results: List[List[Tuple[str, str, str]]] = []
    for amr in df[amr_col].fillna(""):
        results.append(extract_triples_from_amr(amr))
    return results


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def build_global_vocab(all_triples: List[List[Tuple[str, str, str]]]) -> List[str]:
    counter = Counter()
    for triples in all_triples:
        counter.update([" \u0001 ".join(t) for t in triples])
    # Ordered by frequency desc
    vocab = [t for t, _ in counter.most_common()]
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--amr_col", type=str, default="amr")
    parser.add_argument("--out_dir", type=str, default="./saved_models/triples")
    args = parser.parse_args()

    splits: Dict[str, List[List[Tuple[str, str, str]]]] = {}
    splits["train"] = read_csv_and_extract(args.train_csv, args.amr_col)
    if args.valid_csv:
        splits["valid"] = read_csv_and_extract(args.valid_csv, args.amr_col)
    splits["test"] = read_csv_and_extract(args.test_csv, args.amr_col)

    for name, data in splits.items():
        save_pickle(data, os.path.join(args.out_dir, f"{name}_triples.pkl"))

    vocab = build_global_vocab(list(splits.values()))
    save_pickle(vocab, os.path.join(args.out_dir, "global_triple_vocab.pkl"))

    print("Saved triples to", args.out_dir)
    print({k: len(v) for k, v in splits.items()})
    print("Vocab size:", len(vocab))


if __name__ == "__main__":
    main()
