from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_triple_cell(raw: str | None) -> List[str]:
	if raw is None:
		return []
	text = str(raw).strip()
	if not text or text == '[]':
		return []
	try:
		values = json.loads(text)
		if isinstance(values, list):
			return [str(item) for item in values if str(item).strip()]
	except json.JSONDecodeError:
		pass
	return [piece.strip() for piece in text.split(',') if piece.strip()]


def load_observed_triples(
	path: Path,
	has_header: bool,
	triples_field: str,
	triples_index: int,
) -> Tuple[Dict[int, List[str]], int]:
	observed: Dict[int, List[str]] = {}
	total_rows = 0
	with path.open('r', encoding='utf-8', newline='') as handle:
		if has_header:
			reader = csv.DictReader(handle)
			if not reader.fieldnames:
				raise ValueError('CSV 表头为空，无法定位列。')
			if triples_field not in reader.fieldnames:
				raise ValueError(
					f"列 '{triples_field}' 不存在，现有列: {reader.fieldnames}"
				)
			for idx, row in enumerate(reader):
				observed[idx] = parse_triple_cell(row.get(triples_field, ''))
			total_rows = idx + 1 if 'idx' in locals() else 0
		else:
			reader = csv.reader(handle)
			for idx, row in enumerate(reader):
				value = row[triples_index] if len(row) > triples_index else ''
				observed[idx] = parse_triple_cell(value)
			total_rows = idx + 1 if 'idx' in locals() else 0
	return observed, total_rows


def read_unique_sources(triples_csv: Path) -> List[str]:
	sources = set()
	with triples_csv.open('r', encoding='utf-8', newline='') as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			source = (row.get('source') or '').strip()
			if source:
				sources.add(source)
	return sorted(sources)


def compute_source_offsets(
	sources: Iterable[str],
	source_root: Path,
) -> Dict[str, int]:
	offsets: Dict[str, int] = {}
	current = 0
	for source in sources:
		path = (source_root / source).resolve()
		if not path.exists():
			raise FileNotFoundError(f'找不到源文件: {path}')
		with path.open('r', encoding='utf-8') as handle:
			line_count = sum(1 for _ in handle)
		offsets[source] = current
		current += line_count
	return offsets


def load_sentence_to_essay_map(map_path: Path) -> Dict[int, int]:
	mapping: Dict[int, int] = {}
	with map_path.open('r', encoding='utf-8') as handle:
		for line in handle:
			parts = line.strip().split('\t')
			if len(parts) < 2:
				continue
			sent_idx = int(parts[0])
			essay_id = int(parts[1])
			mapping[sent_idx] = essay_id
	return mapping


def build_expected_triples(
	triples_csv: Path,
	offsets: Dict[str, int],
	sentence_map: Dict[int, int],
) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
	expected: Dict[int, List[str]] = defaultdict(list)
	stats = {
		'total_sentences': 0,
		'empty_triples': 0,
		'missing_source': 0,
		'missing_map': 0,
	}
	with triples_csv.open('r', encoding='utf-8', newline='') as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			stats['total_sentences'] += 1
			triples = parse_triple_cell(row.get('triples'))
			if not triples:
				stats['empty_triples'] += 1
				continue
			source = (row.get('source') or '').strip()
			nsent_raw = row.get('nsent')
			if not source or nsent_raw is None:
				stats['missing_source'] += 1
				continue
			if source not in offsets:
				stats['missing_source'] += 1
				continue
			try:
				nsent = int(nsent_raw)
			except ValueError:
				stats['missing_source'] += 1
				continue
			global_idx = offsets[source] + nsent
			essay_id = sentence_map.get(global_idx)
			if essay_id is None:
				stats['missing_map'] += 1
				continue
			expected[essay_id].extend(triples)
	return expected, stats


def compare_triples(
	expected: Dict[int, List[str]],
	observed: Dict[int, List[str]],
	total_rows: int,
	max_report: int,
) -> Tuple[int, List[Tuple[int, List[str], List[str]]]]:
	mismatches: List[Tuple[int, List[str], List[str]]] = []
	mismatch_count = 0
	for essay_id in range(total_rows):
		exp = expected.get(essay_id, [])
		obs = observed.get(essay_id, [])
		if exp != obs:
			mismatch_count += 1
			if len(mismatches) < max_report:
				mismatches.append((essay_id, exp, obs))
	return mismatch_count, mismatches


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='校验三元组是否与正确的作文行对齐。')
	parser.add_argument('--train-csv', type=Path, default=Path('outputs') / 'train_with_triples.csv', help='含三元组列的 train_with_triples.csv 路径。')
	parser.add_argument('--train-has-header', action='store_true', help='train CSV 第一行是否为表头。')
	parser.add_argument('--train-triples-field', type=str, default='triples_essay', help='train CSV 中三元组列名（有表头时生效）。')
	parser.add_argument('--train-triples-index', type=int, default=2, help='train CSV 中三元组列索引（无表头时生效）。')
	parser.add_argument('--triples-csv', type=Path, default=Path('outputs') / 'train_amr_triples.csv', help='句子级三元组 CSV（amr_to_instance 输出）。')
	parser.add_argument('--map-tsv', type=Path, default=Path('data') / 'yelp_review_polarity_csv' / 'train.sent.txt.map.tsv', help='prepare_yelp_to_txt 生成的映射文件。')
	parser.add_argument('--source-root', type=Path, default=Path('.'), help='source 列相对路径的根目录。')
	parser.add_argument('--max-report', type=int, default=20, help='最多展示的映射差异条目数。')
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	observed_map, total_rows = load_observed_triples(
		path=args.train_csv,
		has_header=args.train_has_header,
		triples_field=args.train_triples_field,
		triples_index=args.train_triples_index,
	)
	print(f'已读取 train 文件，共 {total_rows} 行，其中 {len([v for v in observed_map.values() if v])} 行含三元组。')

	sources = read_unique_sources(args.triples_csv)
	offsets = compute_source_offsets(sources, args.source_root)
	sentence_map = load_sentence_to_essay_map(args.map_tsv)
	expected_map, stats = build_expected_triples(args.triples_csv, offsets, sentence_map)

	print(
		'句子级统计: '
		f"总记录 {stats['total_sentences']}, 三元组为空 {stats['empty_triples']}, "
		f"缺少源信息 {stats['missing_source']}, map 不存在 {stats['missing_map']}。"
	)

	mismatch_count, samples = compare_triples(
		expected=expected_map,
		observed=observed_map,
		total_rows=total_rows,
		max_report=args.max_report,
	)

	if mismatch_count == 0:
		print('✅ 所有作文的三元组与句子级聚合结果完全一致，映射未发现错位。')
		return

	print(f'⚠️ 共有 {mismatch_count} 行作文的三元组与预期不一致。示例：')
	for essay_id, expected, observed in samples:
		print('-' * 60)
		print(f'作文索引: {essay_id}')
		print(f'预期: {expected}')
		print(f'实际: {observed}')


if __name__ == '__main__':
	main()
