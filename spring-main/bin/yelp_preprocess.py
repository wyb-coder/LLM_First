"""Yelp dataset preprocessing: select top-K longest reviews per class.

从 Yelp 数据集中，按正负类各选取长度最长的 N 条数据。
保持原始 CSV 结构不变（列名、列顺序、无表头格式）。

Usage:
    python bin/yelp_preprocess.py \
        --input data/yelp_review_polarity_csv/train.csv \
        --output data/yelp_review_polarity_csv/train_top30k.csv \
        --per-class 30000

    # 测试集同理
    python bin/yelp_preprocess.py \
        --input data/yelp_review_polarity_csv/test.csv \
        --output data/yelp_review_polarity_csv/test_top10k.csv \
        --per-class 10000
"""
import argparse
import csv
import os
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select top-K longest reviews per class from Yelp dataset"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input CSV path (original Yelp format: label,text without header)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV path (same format as input)"
    )
    parser.add_argument(
        "--per-class", type=int, default=30000,
        help="Number of samples to keep per class (default: 30000)"
    )
    parser.add_argument(
        "--label-col", type=int, default=0,
        help="Column index for label (default: 0)"
    )
    parser.add_argument(
        "--text-col", type=int, default=1,
        help="Column index for text (default: 1)"
    )
    parser.add_argument(
        "--has-header", action="store_true",
        help="Set if input CSV has header row"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for tie-breaking (default: 42)"
    )
    return parser.parse_args()


def count_words(text: str) -> int:
    """Count words in text (simple split)."""
    return len(text.split())


def main():
    args = parse_args()
    
    # Read all rows
    rows_by_class: dict = {}  # {label: [(word_count, row_index, row), ...]}
    all_rows: List[List[str]] = []
    header = None
    
    print(f"Reading {args.input}...")
    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        
        if args.has_header:
            header = next(reader, None)
        
        for idx, row in enumerate(reader):
            if len(row) < 2:
                continue
            
            label = row[args.label_col].strip()
            text = row[args.text_col].strip()
            word_count = count_words(text)
            
            if label not in rows_by_class:
                rows_by_class[label] = []
            
            rows_by_class[label].append((word_count, idx, row))
            all_rows.append(row)
    
    print(f"Total rows read: {len(all_rows)}")
    print(f"Classes found: {list(rows_by_class.keys())}")
    for label, items in rows_by_class.items():
        print(f"  Class '{label}': {len(items)} samples")
    
    # Sort each class by word count (descending), then by original index (for stability)
    selected_rows: List[Tuple[int, List[str]]] = []  # [(original_idx, row), ...]
    
    for label, items in rows_by_class.items():
        # Sort by word_count descending, then by idx ascending (for tie-breaking)
        items_sorted = sorted(items, key=lambda x: (-x[0], x[1]))
        
        # Select top-K
        top_k = items_sorted[:args.per_class]
        
        print(f"  Class '{label}': selected {len(top_k)} samples")
        if top_k:
            word_counts = [x[0] for x in top_k]
            print(f"    Word count range: {min(word_counts)} - {max(word_counts)}")
        
        for word_count, orig_idx, row in top_k:
            selected_rows.append((orig_idx, row))
    
    # Sort by original index to maintain relative order
    selected_rows.sort(key=lambda x: x[0])
    
    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    print(f"\nWriting {len(selected_rows)} rows to {args.output}...")
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        
        if header:
            writer.writerow(header)
        
        for _, row in selected_rows:
            writer.writerow(row)
    
    print(f"Done! Output: {args.output}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Original samples: {len(all_rows)}")
    print(f"  Selected samples: {len(selected_rows)}")
    print(f"  Per-class limit:  {args.per_class}")
    print("=" * 60)


if __name__ == "__main__":
    main()
