
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def add_review_ids(
	input_csv: Path,
	output_csv: Path,
	start_index: int,
	id_field: str,
	has_header: bool,
) -> None:
	if start_index < 0:
		raise ValueError('start-index 必须是非负整数')

	num_rows = 0

	with input_csv.open('r', encoding='utf-8', newline='') as fin, \
			output_csv.open('w', encoding='utf-8', newline='') as fout:
		reader = csv.reader(fin)
		writer = csv.writer(fout)

		if has_header:
			header = next(reader, None)
			if header is not None:
				writer.writerow([id_field, *header])

		for idx, row in enumerate(reader, start=start_index):
			writer.writerow([idx, *row])
			num_rows += 1

	print(
		f'已为 {input_csv} 的 {num_rows} 条记录添加 {id_field} 字段，'
		f'写入 {output_csv}'
	)


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='为 Yelp train.csv 添加固定 Review_ID 列。')
	default_input = Path('data') / 'yelp_review_polarity_csv' / 'train.csv'
	default_output = default_input.with_name('train.with_id.csv')
	parser.add_argument('--input', type=Path, default=default_input, help='原始 train.csv 路径。')
	parser.add_argument('--output', type=Path, default=default_output, help='写入包含 Review_ID 的 CSV。')
	parser.add_argument('--start-index', type=int, default=0, help='Review_ID 起始值（默认 0）。')
	parser.add_argument('--field-name', type=str, default='review_id', help='新增列的列名（有表头时生效）。')
	parser.add_argument('--has-header', action='store_true', help='若输入 CSV 第一行是表头则开启。')
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	add_review_ids(args.input, args.output, args.start_index, args.field_name, args.has_header)


if __name__ == '__main__':
	main()
