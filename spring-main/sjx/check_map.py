from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def count_lines(path: Path) -> int:
    with path.open('r', encoding='utf-8') as handle:
        return sum(1 for _ in handle)


def load_train_row_count(path: Path, has_header: bool) -> int:
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        if has_header:
            next(reader, None)
        return sum(1 for _ in reader)


def validate_map_file(
    map_path: Path,
    train_csv: Path | None,
    train_has_header: bool,
    sentences_txt: Path | None,
) -> Tuple[Dict[int, int], List[str], List[str], Dict[str, int]]:
    essay_counts: Dict[int, int] = defaultdict(int)
    issues: List[str] = []
    warnings: List[str] = []
    stats = {
        'pairs': 0,
        'max_sent_idx': -1,
        'max_essay_id': -1,
    }

    seen_sent_indices = set()
    expected_sent_idx = 0

    with map_path.open('r', encoding='utf-8') as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                warnings.append(f'第 {line_num} 行为空，已跳过。')
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                issues.append(f'第 {line_num} 行缺少字段: {line!r}')
                continue
            try:
                sent_idx = int(parts[0])
                essay_id = int(parts[1])
            except ValueError:
                issues.append(f'第 {line_num} 行无法解析整数: {line!r}')
                continue

            if sent_idx in seen_sent_indices:
                issues.append(f'句子索引 {sent_idx} 重复出现在第 {line_num} 行。')
            else:
                seen_sent_indices.add(sent_idx)

            if sent_idx != expected_sent_idx:
                issues.append(
                    f'句子索引连续性错误：预期 {expected_sent_idx}，实际 {sent_idx} (行 {line_num})'
                )
                expected_sent_idx = sent_idx + 1
            else:
                expected_sent_idx += 1

            if essay_id < 0:
                issues.append(f'第 {line_num} 行出现负的作文索引 {essay_id}。')
                continue

            essay_counts[essay_id] += 1
            stats['pairs'] += 1
            stats['max_sent_idx'] = max(stats['max_sent_idx'], sent_idx)
            stats['max_essay_id'] = max(stats['max_essay_id'], essay_id)

    if sentences_txt is not None:
        sentence_lines = count_lines(sentences_txt)
        if sentence_lines != stats['pairs']:
            issues.append(
                'train.sent.txt 行数与 map.tsv 记录数不一致：'
                f"{sentence_lines} vs {stats['pairs']}"
            )

    if train_csv is not None:
        total_rows = load_train_row_count(train_csv, train_has_header)
        if stats['max_essay_id'] >= total_rows:
            issues.append(
                f"map.tsv 中的作文索引 {stats['max_essay_id']} 超出 train.csv 行数 {total_rows}。"
            )
        missing_ids = [idx for idx in range(total_rows) if essay_counts.get(idx, 0) == 0]
        if missing_ids:
            warnings.append(
                f'共有 {len(missing_ids)} 篇作文未在 map.tsv 中出现（可能是空作文）：{missing_ids[:20]}'
            )

    return essay_counts, issues, warnings, stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='校验 train.sent.txt.map.tsv 的完整性与覆盖情况。')
    parser.add_argument('--map-tsv', type=Path, default=Path('data') / 'yelp_review_polarity_csv' / 'train.sent.txt.map.tsv', help='句子索引到作文索引的映射文件。')
    parser.add_argument('--sentences', type=Path, default=Path('data') / 'yelp_review_polarity_csv' / 'train.sent.txt', help='句子拆分文本（用于核对行数），可选。')
    parser.add_argument('--train-csv', type=Path, default=Path('data') / 'yelp_review_polarity_csv' / 'train.csv', help='原始训练集 CSV，用于校验作文索引范围，可选。')
    parser.add_argument('--train-has-header', action='store_true', help='train.csv 是否包含表头。')
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    sentences_file = args.sentences if args.sentences.exists() else None
    if args.sentences and not args.sentences.exists():
        print(f'⚠️ 未找到句子文件 {args.sentences}，跳过行数核对。')
    train_file = args.train_csv if args.train_csv.exists() else None
    if args.train_csv and not args.train_csv.exists():
        print(f'⚠️ 未找到 train.csv ({args.train_csv})，跳过作文范围核对。')

    essay_counts, issues, warnings, stats = validate_map_file(
        map_path=args.map_tsv,
        train_csv=train_file,
        train_has_header=args.train_has_header,
        sentences_txt=sentences_file,
    )

    print('=== map.tsv 核查结果 ===')
    print(f"映射条目: {stats['pairs']} (句子索引 0..{stats['max_sent_idx']})")
    if train_file is not None:
        print(f"作文索引最大值: {stats['max_essay_id']} (train.csv 路径 {train_file})")
    else:
        print(f"作文索引最大值: {stats['max_essay_id']} (train.csv 未提供，无法进一步核对)")

    if issues:
        print(f'❌ 发现 {len(issues)} 个问题：')
        for msg in issues:
            print(f'  - {msg}')
    else:
        print('✅ 未发现结构性问题。')

    if warnings:
        print(f'⚠️ 额外提醒 ({len(warnings)} 条)：')
        for msg in warnings:
            print(f'  - {msg}')

    top_examples = sorted(essay_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    print('作文句子数 Top5（作文索引 -> 句子数量）:')
    for essay_id, count in top_examples:
        print(f'  - {essay_id}: {count}')


if __name__ == '__main__':
    main()
