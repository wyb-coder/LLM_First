"""Extract triples from CSV files.

Two modes:
- AMR string column decoded with penman.
- Pre-extracted triple-list column (e.g., last column in *_with_amr.csv).

Outputs: train/valid/test triples pickles and a global triple vocab.

Example (pre-extracted triples, no header, last column index 2):
    python extract_triples.py \
        --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
        --test_csv  data/yelp_review_polarity_csv/test_with_amr.csv \
        --triples_col 2 \
        --csv_has_header False

Example (AMR column named `amr` with header):
    python extract_triples.py \
        --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
        --test_csv  data/yelp_review_polarity_csv/test_with_amr.csv \
        --amr_col amr \
        --csv_has_header True
"""
import argparse
import os
import pickle
import ast
from collections import Counter
from typing import List, Tuple, Dict, Any, Union

import pandas as pd


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


def maybe_decode_amr(amr_str: str):
    # Lazy import to avoid requiring penman when using pre-extracted triples.
    import penman
    return penman.decode(amr_str)


def get_concept(graph: Any, var: str) -> str:
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
        graph = maybe_decode_amr(amr_str)
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


def coerce_triple_list(cell: Union[str, List[Any]]) -> List[Tuple[str, str, str]]:
    """Convert a cell containing triples to List[Tuple]. Handles strified lists and python lists."""
    if isinstance(cell, float):  # NaN
        return []
    raw = cell
    if isinstance(cell, str):
        try:
            raw = ast.literal_eval(cell)
        except Exception:
            raw = []
    if not isinstance(raw, list):
        return []

    triples: List[Tuple[str, str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            head, rel, tail = item
        elif isinstance(item, str):
            parts = item.strip().split()
            if len(parts) != 3:
                continue
            head, rel, tail = parts
        else:
            continue
        triples.append((str(head), str(rel), str(tail)))
    return triples


def read_csv_and_extract(path: str, amr_col: str, header: Union[int, None]) -> List[List[Tuple[str, str, str]]]:
    df = pd.read_csv(path, header=header)
    if amr_col not in df.columns:
        raise ValueError(f"Column {amr_col} not found in {path}")
    results: List[List[Tuple[str, str, str]]] = []
    for amr in df[amr_col].fillna(""):
        results.append(extract_triples_from_amr(amr))
    return results


def read_csv_and_parse_list(path: str, triples_col: Union[str, int], header: Union[int, None]) -> List[List[Tuple[str, str, str]]]:
    df = pd.read_csv(path, header=header)
    if triples_col not in df.columns:
        raise ValueError(f"Column {triples_col} not found in {path}")
    results: List[List[Tuple[str, str, str]]] = []
    for cell in df[triples_col].fillna("[]"):
        results.append(coerce_triple_list(cell))
    return results


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def build_global_vocab(all_splits: List[List[List[Tuple[str, str, str]]]]) -> List[str]:
    """Build global vocab from all splits (train/valid/test).
    
    Args:
        all_splits: List of splits, each split is List[List[Tuple]] (samples -> triples)
    """
    counter = Counter()
    for split in all_splits:  # each split (train/test)
        for triples in split:  # each sample's triples
            counter.update([" \u0001 ".join(t) for t in triples])
    vocab = [t for t, _ in counter.most_common()]
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--amr_col", type=str, default="amr", help="AMR column name (AMR mode)")
    parser.add_argument("--triples_col", type=str, default=None, help="Triple-list column name or index (list mode)")
    parser.add_argument("--csv_has_header", type=lambda x: str(x).lower() == "true", default=False, help="Whether CSV has header row")
    parser.add_argument("--out_dir", type=str, default="./saved_models/triples")
    args = parser.parse_args()

    header = 0 if args.csv_has_header else None

    if args.triples_col is not None:
        col = int(args.triples_col) if str(args.triples_col).isdigit() else args.triples_col
        reader = lambda path: read_csv_and_parse_list(path, col, header)
    else:
        reader = lambda path: read_csv_and_extract(path, args.amr_col, header)

    splits: Dict[str, List[List[Tuple[str, str, str]]]] = {}
    splits["train"] = reader(args.train_csv)
    if args.valid_csv:
        splits["valid"] = reader(args.valid_csv)
    splits["test"] = reader(args.test_csv)

    for name, data in splits.items():
        save_pickle(data, os.path.join(args.out_dir, f"{name}_triples.pkl"))

    vocab = build_global_vocab(list(splits.values()))
    save_pickle(vocab, os.path.join(args.out_dir, "global_triple_vocab.pkl"))

    print("Saved triples to", args.out_dir)
    print({k: len(v) for k, v in splits.items()})
    print("Vocab size:", len(vocab))


if __name__ == "__main__":
    main()
