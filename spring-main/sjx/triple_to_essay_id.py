from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_CSV = PROJECT_ROOT / 'data' / 'yelp_review_polarity_csv' / 'train.csv'
DEFAULT_TRIPLES_CSV = PROJECT_ROOT / 'outputs' / 'train_amr_triples_essay.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'outputs' / 'train_with_triples.csv'


def load_triples_map(path: Path) -> Dict[int, str]:
    df = pd.read_csv(path)
    required = {'id', 'triples_essay'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'缺少必要列: {missing}')
    df = df.dropna(subset=['id'])
    df['id'] = df['id'].astype(int)
    df['triples_essay'] = df['triples_essay'].fillna('')
    triples_map: Dict[int, str] = dict(zip(df['id'], df['triples_essay']))
    return triples_map


def augment_train_with_triples(
    train_csv: Path,
    triples_csv: Path,
    output_csv: Path,
    has_header: bool = False,
) -> None:
    triples_map = load_triples_map(triples_csv)
    total = 0
    matched = 0

    with train_csv.open('r', encoding='utf-8', newline='') as fin, \
            output_csv.open('w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        if has_header:
            header = next(reader, None)
            if header is not None:
                writer.writerow(header + ['triples_essay'])
        row_idx = 0
        for row in reader:
            triple_text = triples_map.get(row_idx, '')
            if triple_text:
                matched += 1
            writer.writerow(row + [triple_text])
            row_idx += 1
        total = row_idx

    print(
        f'已处理 {total} 条作文，其中 {matched} 条匹配到三元组；'
        f'结果已写入 {output_csv}'
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='将作文级三元组合并回 Yelp 训练集 CSV。')
    parser.add_argument('--train-csv', type=Path, default=DEFAULT_TRAIN_CSV, help='原始 train.csv 路径。')
    parser.add_argument('--triples-csv', type=Path, default=DEFAULT_TRIPLES_CSV, help='train_amr_triples_essay.csv 路径。')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='写入包含三元组的新 CSV。')
    parser.add_argument('--has-header', action='store_true', help='若 train.csv 首行是表头则开启。')
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    augment_train_with_triples(args.train_csv, args.triples_csv, args.output, args.has_header)


if __name__ == '__main__':
    main()