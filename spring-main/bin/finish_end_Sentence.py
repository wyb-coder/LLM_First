import argparse
import ast
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple


RESULT_DIR = Path(__file__).resolve().parent.parent / 'result'
RESULT_TEMP_DIR = RESULT_DIR / 'temp'
DATA_TEMP_DIR_NAME = 'temp'
META_FIELDNAMES = ['review_id', 'sentence_id', 'sentence', 'amr']
AMR_NONE_TOKEN = 'AMR-None'


def parse_shard_open(expr: str) -> Tuple[int, List[int]]:
	try:
		values = ast.literal_eval(expr)
	except Exception as exc:
		raise argparse.ArgumentTypeError(f"无法解析 --shard-open：{expr}") from exc
	if not isinstance(values, (list, tuple)) or len(values) < 2:
		raise argparse.ArgumentTypeError("--shard-open 需要形如 [N, gpu1, ...] 的列表")
	try:
		num = int(values[0])
		gpus = [int(v) for v in values[1:]]
	except Exception as exc:
		raise argparse.ArgumentTypeError("--shard-open 只能包含整数") from exc
	if num <= 0:
		raise argparse.ArgumentTypeError("分片数量必须大于 0")
	if num != len(gpus):
		raise argparse.ArgumentTypeError("GPU 数量必须与分片数量一致")
	return num, gpus


def load_meta_rows(meta_path: Path) -> List[Dict[str, str]]:
	rows: List[Dict[str, str]] = []
	with meta_path.open('r', encoding='utf-8', newline='') as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			rows.append(row)
	return rows


def write_meta_rows(meta_path: Path, rows: List[Dict[str, str]]) -> None:
	meta_path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = meta_path.with_suffix(f"{meta_path.suffix}.tmp")
	with tmp_path.open('w', encoding='utf-8', newline='') as handle:
		writer = csv.DictWriter(handle, fieldnames=META_FIELDNAMES)
		writer.writeheader()
		for row in rows:
			writer.writerow({field: row.get(field, '') for field in META_FIELDNAMES})
		handle.flush()
		os.fsync(handle.fileno())
	tmp_path.replace(meta_path)


def collect_shard_infos(text_path: Path, num_shards: int) -> List[Dict[str, object]]:
	temp_dir = text_path.parent / DATA_TEMP_DIR_NAME
	result_temp_dir = RESULT_TEMP_DIR / text_path.stem
	if not temp_dir.exists():
		raise FileNotFoundError(f"缺少分片目录: {temp_dir}")
	if not result_temp_dir.exists():
		raise FileNotFoundError(f"缺少结果目录: {result_temp_dir}")

	shard_infos: List[Dict[str, object]] = []
	for shard_idx in range(num_shards):
		shard_name = f"{text_path.stem}{shard_idx + 1}{text_path.suffix}"
		shard_input = temp_dir / shard_name
		meta_path = result_temp_dir / f"{shard_name}.csv"
		if not shard_input.exists():
			raise FileNotFoundError(f"找不到输入分片: {shard_input}")
		if not meta_path.exists():
			raise FileNotFoundError(f"找不到中间 CSV: {meta_path}")
		rows = load_meta_rows(meta_path)
		pending = sum(1 for row in rows if not (row.get('amr') or '').strip())
		shard_infos.append({
			'name': shard_name,
			'meta_path': meta_path,
			'input_path': shard_input,
			'total': len(rows),
			'pending': pending,
			'rows': rows,
		})
	return shard_infos


def print_shard_status(shard_infos: List[Dict[str, object]], gpu_ids: List[int]) -> None:
	total_pending = 0
	print("分片状态预览：")
	for idx, info in enumerate(shard_infos):
		gpu_id = gpu_ids[idx]
		total_pending += info['pending']
		print(
			f"  Shard{idx + 1} -> GPU{gpu_id} | 全部 {info['total']} 条 | 待处理 {info['pending']} 条 | meta={info['meta_path']}"
		)
	print(f"总待处理句子：{total_pending}")


def fill_empty_amr(shard_infos: List[Dict[str, object]]) -> None:
	total_filled = 0
	for info in shard_infos:
		rows = info['rows']
		filled = 0
		for row in rows:
			if not (row.get('amr') or '').strip():
				row['amr'] = AMR_NONE_TOKEN
				filled += 1
		if filled:
			write_meta_rows(info['meta_path'], rows)
		print(f"{info['name']}: 填充 {filled} 条 AMR-None")
		total_filled += filled
	print(f"共写入 {total_filled} 条 AMR-None。")


def main() -> None:
	parser = argparse.ArgumentParser(description='填充未完成分片的 AMR-None')
	parser.add_argument('--text', type=Path, required=True, help='原始文本路径 (用于定位分片)')
	parser.add_argument('--shard-open', type=str, required=True, help='形如 [N, gpu1, ...] 的配置')
	args = parser.parse_args()

	num_shards, gpu_ids = parse_shard_open(args.shard_open)
	shard_infos = collect_shard_infos(args.text, num_shards)
	print_shard_status(shard_infos, gpu_ids)

	answer = input('是否开始将空 AMR 填充为 AMR-None? (Yes/No): ').strip().lower()
	if answer == 'yes':
		fill_empty_amr(shard_infos)
	else:
		print('已取消操作。')


if __name__ == '__main__':
	main()
