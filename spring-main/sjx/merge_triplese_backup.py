from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AMR_CSV = PROJECT_ROOT / 'outputs' / 'train_amr_triples.csv'
DEFAULT_MAP_TSV = PROJECT_ROOT / 'data' / 'yelp_review_polarity_csv' / 'train.sent.txt.map.tsv'
DEFAULT_AMR_OUTPUT = PROJECT_ROOT / 'outputs' / 'train_amr_triples_essay.csv'


def merge_triples(prompt: str) -> None:
    """legacy: merge triples by prompt id list"""
    data_path = PROJECT_ROOT / 'data' / 'triple' / f'triples{prompt}.new.csv'
    data = pd.read_csv(data_path)
    ss = sorted(set(data['id'].tolist()))
    merge_triple = [[] for _ in range(len(ss))]
    num = 0
    id_list: List[int] = []
    for i in range(len(data)):
        iid = int(data['id'][i])
        t_i = data['triple'][i].split(',')
        merge_triple[num].extend(t_i)
        if i <= len(data) - 2 and int(data['id'][i + 1]) == iid:
            continue
        id_list.append(iid)
        num += 1

    triple_str = []
    for triples in merge_triple:
        s = ''
        for idx, item in enumerate(triples):
            s += (',' if idx else '') + item
        triple_str.append(s)

    out_dir = PROJECT_ROOT / 'data' / 'triple_essay'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'triple_essay{prompt}.new.csv'
    T = pd.DataFrame({'triples_essay': triple_str, 'id': id_list})
    T.to_csv(out_path, index=False, encoding='utf-8')


def _parse_triple_cell(cell) -> List[str]:
    if isinstance(cell, list):
        return [str(item) for item in cell if str(item).strip()]
    text = str(cell) if cell is not None else ''
    if not text or text == '[]':
        return []
    try:
        values = json.loads(text)
        if isinstance(values, (list, tuple)):
            return [str(item) for item in values if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [piece.strip() for piece in text.split(',') if piece.strip()]


def _source_sort_key(value: str) -> tuple:
    path = Path(value)
    stem = path.stem
    match = re.search(r'(\d+)$', stem)
    num = int(match.group(1)) if match else -1
    prefix = stem if not match else stem[: -len(match.group(1))]
    return (str(path.parent), prefix, num, stem)


def _resolve_source_path(source_root: Path, source_value: str) -> Path:
    candidate = Path(source_value)
    if not candidate.is_absolute():
        candidate = (source_root / source_value).resolve()
    return candidate


def _count_lines(path: Path) -> int:
    with path.open('r', encoding='utf-8') as handle:
        return sum(1 for _ in handle)


def _build_source_offsets(sources: Iterable[str], source_root: Path) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    current = 0
    for source in sorted(set(sources), key=_source_sort_key):
        resolved = _resolve_source_path(source_root, source)
        if not resolved.exists():
            raise FileNotFoundError(f'找不到源文件: {resolved}')
        offsets[source] = current
        current += _count_lines(resolved)
    return offsets


def _load_sentence_doc_map(map_path: Path) -> Dict[int, int]:
    df = pd.read_csv(map_path, sep='\t', header=None, usecols=[0, 1], names=['sent_idx', 'essay_id'])
    return dict(zip(df['sent_idx'].astype(int), df['essay_id'].astype(int)))


def merge_triples_from_amr(
    triples_csv: Path,
    map_tsv: Path,
    output_csv: Path,
    source_root: Path | None = None,
) -> None:
    df = pd.read_csv(triples_csv)
    required_cols = {'nsent', 'triples', 'source'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'缺少必要列: {missing}')

    total_rows = len(df)
    df = df.dropna(subset=['source', 'nsent', 'triples'])
    dropped_na = total_rows - len(df)
    df['triples_list'] = df['triples'].apply(_parse_triple_cell)
    empty_triples_mask = ~df['triples_list'].map(bool)
    dropped_empty = int(empty_triples_mask.sum())
    df = df[~empty_triples_mask]
    df['nsent'] = df['nsent'].astype(int)

    root = source_root or PROJECT_ROOT
    source_offsets = _build_source_offsets(df['source'].unique(), root)
    sentence_map = _load_sentence_doc_map(map_tsv)

    essay_triples: Dict[int, List[str]] = defaultdict(list)
    skipped_missing_source = 0
    skipped_missing_map = 0
    for row in df.itertuples(index=False):
        offset = source_offsets.get(row.source)
        if offset is None:
            skipped_missing_source += 1
            continue
        global_idx = offset + int(row.nsent)
        essay_id = sentence_map.get(global_idx)
        if essay_id is None:
            skipped_missing_map += 1
            continue
        essay_triples[int(essay_id)].extend(row.triples_list)

    records = [
        {'id': essay_id, 'triples_essay': json.dumps(triples, ensure_ascii=False)}
        for essay_id, triples in sorted(essay_triples.items())
        if triples
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False, encoding='utf-8')
    print(
        f'已合并 {len(df)} 句子，得到 {len(records)} 篇作文，写入 {output_csv}。'
    )
    print(
        '跳过统计: '
        f'空/缺失三元组 {dropped_na + dropped_empty} 条（缺列 {dropped_na} + 空三元组 {dropped_empty}），'
        f'源文件缺失 {skipped_missing_source} 条，'
        f'未在 map 中找到 {skipped_missing_map} 条。'
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Merge sentence triples into essay-level triples.')
    parser.add_argument('--amr-csv', type=Path, default=None, help='来自 amr_to_instance 的输出 CSV 路径。')
    parser.add_argument('--map-tsv', type=Path, default=DEFAULT_MAP_TSV, help='train.sent.txt.map.tsv 路径。')
    parser.add_argument('--output', type=Path, default=DEFAULT_AMR_OUTPUT, help='作文级三元组输出路径。')
    parser.add_argument('--source-root', type=Path, default=PROJECT_ROOT, help='source 列为相对路径时的根目录。')
    parser.add_argument('--legacy-prompt', type=str, help='如果仍需旧流程，提供 prompt 名称。')
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.legacy_prompt:
        merge_triples(args.legacy_prompt)
        return
    if not args.amr_csv:
        parser.error('请提供 --amr-csv 或使用 --legacy-prompt。')
    merge_triples_from_amr(args.amr_csv, args.map_tsv, args.output, args.source_root)


if __name__ == '__main__':
    main()


