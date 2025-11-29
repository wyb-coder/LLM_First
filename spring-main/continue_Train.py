from __future__ import annotations

import argparse
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm

from bin.predict_amrs_from_plaintext import (
	RESULT_DIR,
	RESULT_TEMP_DIR,
	aggregate_shard_outputs,
	build_model_config,
	parse_shard_open,
	read_file_in_batches,
	split_text_file_for_shards,
)

from spring_amr.penman import encode
from spring_amr.utils import instantiate_model_and_tokenizer


def extract_nsent(lines: Iterable[str]) -> int:
	for line in lines:
		if line.startswith('# ::nsent'):
			parts = line.strip().split()
			if parts:
				try:
					return int(parts[-1])
				except ValueError as exc:
					raise ValueError(f"Invalid nsent line: {line.rstrip()}") from exc
	raise ValueError('Missing # ::nsent metadata in AMR block')


def iter_amr_blocks(text: str) -> Iterable[List[str]]:
	block: List[str] = []
	for line in text.splitlines():
		if line.strip():
			block.append(line)
		elif block:
			yield block
			block = []
	if block:
		yield block


def cleanup_amr_file(path: Path) -> Tuple[int, int]:
	path.parent.mkdir(parents=True, exist_ok=True)
	if not path.exists() or path.stat().st_size == 0:
		path.touch(exist_ok=True)
		return 0, 0

	content = path.read_text(encoding='utf-8')
	complete_blocks: List[str] = []
	last_nsent = -1

	for lines in iter_amr_blocks(content):
		try:
			nsent = extract_nsent(lines)
		except ValueError:
			break
		complete_blocks.append('\n'.join(lines))
		last_nsent = max(last_nsent, nsent)

	if complete_blocks:
		path.write_text('\n\n'.join(complete_blocks) + '\n\n', encoding='utf-8')
	else:
		path.write_text('', encoding='utf-8')

	resume_idx = last_nsent + 1 if last_nsent >= 0 else 0
	return resume_idx, resume_idx


def prepare_resume_tasks(shard_infos: List[Dict[str, object]]) -> List[Dict[str, object]]:
	prepared: List[Dict[str, object]] = []
	for info in shard_infos:
		updated = dict(info)
		output_path = Path(updated['result_path'])
		resume_idx, processed = cleanup_amr_file(output_path)
		updated['resume_index'] = resume_idx
		updated['processed'] = processed
		prepared.append(updated)
		print(f"Resume shard {output_path.name}: start from nsent {resume_idx}")
	return prepared
	for info in shard_infos:
		updated = dict(info)
		output_path = Path(updated['result_path'])
		resume_idx, processed = cleanup_amr_file(output_path)
		updated['resume_index'] = resume_idx
		updated['processed'] = processed
		prepared.append(updated)
		print(f"Resume shard {output_path.name}: start from nsent {resume_idx}")
	return prepared


def resume_prediction_tasks(
	tasks: List[Dict[str, object]],
	config: Dict[str, object],
	device_str: str,
	progress_callback,
) -> None:
	device = torch.device(device_str)
	model, tokenizer = instantiate_model_and_tokenizer(
		config['model_name'],
		dropout=0.0,
		attention_dropout=0.0,
		penman_linearization=config['penman_linearization'],
		use_pointer_tokens=config['use_pointer_tokens'],
	)
	state = torch.load(config['checkpoint'], map_location='cpu')['model']
	model.load_state_dict(state)
	model.to(device)
	model.eval()

	for task in tasks:
		input_path = Path(task['input_path'])
		output_path = Path(task['result_path'])
		resume_index = int(task.get('resume_index', 0))

		iterator, total_examples = read_file_in_batches(input_path, config['batch_size'])
		if resume_index >= total_examples:
			continue

		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open('a', encoding='utf-8') as output_stream:
			for batch in iterator:
				filtered = [sample for sample in batch if sample[0] >= resume_index]
				if not filtered:
					continue
				ids, sentences, _ = zip(*filtered)
				x, _ = tokenizer.batch_encode_sentences(sentences, device=device)
				with torch.no_grad():
					model.amr_mode = True
					generated = model.generate(
						**x,
						max_length=512,
						decoder_start_token_id=0,
						num_beams=config['beam_size'],
					)

				graphs = []
				for idx, sent, tokk in zip(ids, sentences, generated):
					graph, status, _meta = tokenizer.decode_amr(
						tokk.tolist(),
						restore_name_ops=config['restore_name_ops'],
					)
					if config['only_ok'] and ('OK' not in str(status)):
						continue
					graph.metadata['status'] = str(status)
					graph.metadata['source'] = str(input_path)
					graph.metadata['nsent'] = str(idx)
					graph.metadata['snt'] = sent
					graphs.append(graph)

				for graph in graphs:
					output_stream.write(encode(graph))
					output_stream.write('\n\n')

				if progress_callback:
					progress_callback(len(filtered))

	model.to('cpu')


def shard_worker(
	config: Dict[str, object],
	device_str: str,
	tasks: List[Dict[str, object]],
	shard_index: int,
	progress_queue,
	error_queue,
) -> None:
	try:
		def _progress(delta: int) -> None:
			if delta:
				progress_queue.put((shard_index, delta))

		resume_prediction_tasks(tasks, config, device_str, _progress)
		progress_queue.put((shard_index, 'done'))
	except Exception:
		error_queue.put((shard_index, traceback.format_exc()))
		progress_queue.put((shard_index, 'error'))


def orchestrate_resume(args, shard_config: Tuple[int, List[int]]) -> None:
	num_shards, gpu_ids = shard_config
	RESULT_DIR.mkdir(parents=True, exist_ok=True)
	RESULT_TEMP_DIR.mkdir(parents=True, exist_ok=True)

	split_map: Dict[str, List[Dict[str, object]]] = {}
	for text_path in args.texts:
		split_map[text_path] = prepare_resume_tasks(
			split_text_file_for_shards(text_path, num_shards)
		)

	tasks_per_shard: List[List[Dict[str, object]]] = []
	for shard_idx in range(num_shards):
		shard_tasks = []
		for text_path in args.texts:
			shard_tasks.append(split_map[text_path][shard_idx])
		tasks_per_shard.append(shard_tasks)

	totals = [sum(int(task['line_count']) for task in shard_tasks) for shard_tasks in tasks_per_shard]
	processed = [sum(int(task.get('processed', 0)) for task in shard_tasks) for shard_tasks in tasks_per_shard]

	manager = mp.Manager()
	progress_queue = manager.Queue()
	error_queue = manager.Queue()

	config = build_model_config(args)

	processes: List[mp.Process] = []
	for shard_idx, shard_tasks in enumerate(tasks_per_shard):
		device_id = gpu_ids[shard_idx]
		device_str = device_id if isinstance(device_id, str) and str(device_id).startswith('cuda') else f"cuda:{device_id}"
		proc = mp.Process(
			target=shard_worker,
			args=(config, device_str, shard_tasks, shard_idx, progress_queue, error_queue),
		)
		proc.start()
		processes.append(proc)

	bars: List[tqdm] = []
	finished = 0

	for shard_idx, total in enumerate(totals):
		total_value = max(total, 1)
		bar = tqdm(
			total=total_value,
			desc=f"Shard{shard_idx + 1}:GPU{gpu_ids[shard_idx]}",
			position=shard_idx,
			leave=True,
		)
		initial = min(processed[shard_idx], bar.total)
		if initial:
			bar.update(initial)
		if total == 0:
			bar.update(total_value)
		bars.append(bar)

	while finished < num_shards:
		shard_idx, payload = progress_queue.get()
		bar = bars[shard_idx]
		if payload == 'done':
			remaining = bar.total - bar.n
			if remaining > 0:
				bar.update(remaining)
			bar.close()
			finished += 1
		elif payload == 'error':
			remaining = bar.total - bar.n
			if remaining > 0:
				bar.update(remaining)
			bar.close()
			finished += 1
		else:
			bar.update(payload)

	for proc in processes:
		proc.join()

	errors = []
	while not error_queue.empty():
		errors.append(error_queue.get())

	if errors:
		raise RuntimeError('\n'.join(f"Shard {idx + 1} failed:\n{tb}" for idx, tb in errors))

	aggregated = []
	for original_path, shard_infos in split_map.items():
		aggregated.append(aggregate_shard_outputs(original_path, shard_infos))

	for path in aggregated:
		print(f"Aggregated AMR saved to {path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description='Resume AMR prediction from shard outputs.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument('--texts', type=str, required=True, nargs='+', help='Input text files used in the original run.')
	parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint to restore.')
	parser.add_argument('--model', type=str, default='facebook/bart-large', help='Model identifier.')
	parser.add_argument('--beam-size', type=int, default=1, help='Beam size.')
	parser.add_argument('--batch-size', type=int, default=1000, help='Approximate token budget per batch.')
	parser.add_argument('--penman-linearization', action='store_true', help='Use PENMAN linearization.')
	parser.add_argument('--use-pointer-tokens', action='store_true')
	parser.add_argument('--restore-name-ops', action='store_true')
	parser.add_argument('--device', type=str, default='cuda', help='Device string.')
	parser.add_argument('--only-ok', action='store_true')
	parser.add_argument('--shard-open', type=str, required=True, help='[N, gpu1, gpu2, ...] list describing active shards.')

	args = parser.parse_args()

	try:
		shard_config = parse_shard_open(args.shard_open)
	except ValueError as exc:
		parser.error(str(exc))

	orchestrate_resume(args, shard_config)


if __name__ == '__main__':
	main()
