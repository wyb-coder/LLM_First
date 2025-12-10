from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / 'outputs' / 'train_with_triples.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'outputs' / 'train_with_amr.csv'


def strip_first_column(input_csv: Path, output_csv: Path) -> None:
	removed = 0
	total = 0

	output_csv.parent.mkdir(parents=True, exist_ok=True)

	with input_csv.open('r', encoding='utf-8', newline='') as fin, \
			output_csv.open('w', encoding='utf-8', newline='') as fout:
		reader = csv.reader(fin)
		writer = csv.writer(fout)

		for row in reader:
			total += 1
			if row:
				removed += 1
				writer.writerow(row[1:])
			else:
				writer.writerow([])

	print(f'已处理 {total} 行，已移除首列，结果写入 {output_csv}')


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='从 train_with_triples.csv 中删除首列并输出 train_with_amr.csv。')
	parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='输入 CSV，默认 outputs/train_with_triples.csv。')
	parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='输出 CSV，默认 outputs/train_with_amr.csv。')
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	strip_first_column(args.input, args.output)


if __name__ == '__main__':
	main()
