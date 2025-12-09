import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict


RESULT_DIR = Path(__file__).resolve().parent.parent / 'result'


def sample_rows(source_csv: Path, count: int, seed: int) -> List[Dict[str, str]]:
	with source_csv.open('r', encoding='utf-8', newline='') as handle:
		reader = csv.DictReader(handle)
		rows = list(reader)
		if not rows:
			raise ValueError(f"{source_csv} 为空，无法抽样")
		rng = random.Random(seed)
		sample_size = min(count, len(rows))
		return rng.sample(rows, sample_size), reader.fieldnames or []


def write_rows(rows: List[Dict[str, str]], fieldnames: List[str], output_csv: Path) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open('w', encoding='utf-8', newline='') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow({name: row.get(name, '') for name in fieldnames})


def main() -> None:
	parser = argparse.ArgumentParser(description='随机抽取聚合 CSV 的若干行并导出')
	parser.add_argument('--csv', type=Path, required=True, help='聚合结果 CSV 路径')
	parser.add_argument('--count', type=int, default=200, help='抽样行数 (默认 200)')
	parser.add_argument('--output', type=Path, default=RESULT_DIR / 'test.csv', help='输出 CSV (默认 result/test.csv)')
	parser.add_argument('--seed', type=int, default=42, help='随机种子，保证复现')
	args = parser.parse_args()

	rows, fieldnames = sample_rows(args.csv, args.count, args.seed)
	if not fieldnames:
		fieldnames = list(rows[0].keys())
	write_rows(rows, fieldnames, args.output)
	print(f"已从 {args.csv} 抽取 {len(rows)} 条样本 -> {args.output}")


if __name__ == '__main__':
	main()
