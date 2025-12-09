from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_CSV = PROJECT_ROOT / 'data' / 'yelp_review_polarity_csv' / 'train.csv'
DEFAULT_TRIPLES_CSV = PROJECT_ROOT / 'outputs' / 'train_amr_triples_essay.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'outputs' / 'train_with_triples.csv'


def load_triples_map(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path, dtype={'id': str})
    required = {'id', 'triples_essay'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'缺少必要列: {missing}')
    df = df.dropna(subset=['id'])
    df['triples_essay'] = df['triples_essay'].fillna('')
    triples_map: Dict[str, str] = dict(zip(df['id'], df['triples_essay']))
    return triples_map


def augment_train_with_triples(
    train_csv: Path,
    triples_csv: Path,
    output_csv: Path,
    has_header: bool = False,
    id_column: Optional[int] = None,
) -> None:
    triples_map = load_triples_map(triples_csv)
    total = 0
    matched = 0
    appended = 0

    temp_copy = output_csv.with_suffix(output_csv.suffix + '.tmpcopy')
    shutil.copyfile(train_csv, temp_copy)

    header_row: Optional[List[str]] = None

    with temp_copy.open('r', encoding='utf-8', newline='') as fin, \
            output_csv.open('w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        if has_header:
            header_row = next(reader, None)
            if header_row is not None:
                writer.writerow(header_row + ['triples_essay'])

        row_idx = 0
        for row in reader:
            if id_column is not None and 0 <= id_column < len(row):
                lookup_id = str(row[id_column])
            else:
                lookup_id = str(row_idx)
            triple_text = triples_map.get(lookup_id, '')
            if triple_text:
                matched += 1
            writer.writerow(row + [triple_text])
            appended += 1
            total += 1
            row_idx += 1

    filtered_output = output_csv.with_suffix(output_csv.suffix + '.filtered')
    removed = 0
    with output_csv.open('r', encoding='utf-8', newline='') as fin, \
            filtered_output.open('w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        first_row_consumed = False
        for row in reader:
            if has_header and not first_row_consumed:
                first_row_consumed = True
                if header_row is not None:
                    writer.writerow(row)
                    continue
            triples_value = row[-1] if row else ''
            if not triples_value.strip():
                removed += 1
                continue
            writer.writerow(row)

    filtered_output.replace(output_csv)
    try:
        temp_copy.unlink()
    except FileNotFoundError:
        pass

    print(
        f'已处理 {total} 条作文，其中 {matched} 条匹配到三元组；'
        f'追加列 {appended} 条，移除空三元组 {removed} 条；'
        f'结果已写入 {output_csv}'
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='将作文级三元组合并回 Yelp 训练集 CSV。')
    parser.add_argument('--train-csv', type=Path, default=DEFAULT_TRAIN_CSV, help='原始 train.csv 路径。')
    parser.add_argument('--triples-csv', type=Path, default=DEFAULT_TRIPLES_CSV, help='train_amr_triples_essay.csv 路径。')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='写入包含三元组的新 CSV。')
    parser.add_argument('--has-header', action='store_true', help='若 train.csv 首行是表头则开启。')
    parser.add_argument('--id-column', type=int, help='train.csv 中 review_id 所在列索引；若不指定则按行号匹配。')
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    augment_train_with_triples(args.train_csv, args.triples_csv, args.output, args.has_header, args.id_column)


if __name__ == '__main__':
    main()