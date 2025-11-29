from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def reservoir_sample(
	rows: Sequence[Sequence[str]],
	sample_size: int,
	rng: random.Random,
) -> List[Tuple[int, List[str]]]:
	reservoir: List[Tuple[int, List[str]]] = []
	for idx, row in enumerate(rows):
		if len(reservoir) < sample_size:
			reservoir.append((idx, list(row)))
		else:
			replace_idx = rng.randint(0, idx)
			if replace_idx < sample_size:
				reservoir[replace_idx] = (idx, list(row))
	return reservoir


def sample_csv(
	input_path: Path,
	output_path: Path,
	sample_size: int,
	has_header: bool,
	seed: Optional[int],
) -> None:
	rng = random.Random(seed)
	with input_path.open('r', encoding='utf-8', newline='') as fin:
		reader = csv.reader(fin)
		header: Optional[List[str]] = None
		if has_header:
			header = next(reader, None)

		samples = reservoir_sample(reader, sample_size, rng)
		samples.sort(key=lambda item: item[0])

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open('w', encoding='utf-8', newline='') as fout:
		writer = csv.writer(fout)
		if header is not None:
			writer.writerow(['row_index', *header])
		else:
			writer.writerow(['row_index'])
		for row_idx, row in samples:
			writer.writerow([row_idx, *row])


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='从 train_with_triples 中随机抽样行。')
	parser.add_argument('--input', type=Path, default=Path('outputs') / 'train_with_triples.csv', help='源 CSV 路径。')
	parser.add_argument('--output', type=Path, default=Path('outputs') / 'check.csv', help='抽样结果输出路径。')
	parser.add_argument('--sample-size', type=int, default=100, help='抽样行数。')
	parser.add_argument('--has-header', action='store_true', help='如果 CSV 首行为表头则开启。')
	parser.add_argument('--seed', type=int, default=42, help='随机种子，方便复现。')
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	sample_csv(
		input_path=args.input,
		output_path=args.output,
		sample_size=args.sample_size,
		has_header=args.has_header,
		seed=args.seed,
	)
	print(f'已从 {args.input} 抽取 {args.sample_size} 行写入 {args.output}')


if __name__ == '__main__':
	main()
