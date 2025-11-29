from pathlib import Path
import ast
import multiprocessing as mp
import traceback

import torch
from tqdm import tqdm

from spring_amr.penman import encode
from spring_amr.utils import instantiate_model_and_tokenizer


RESULT_DIR = Path(__file__).resolve().parent.parent / 'result'
RESULT_TEMP_DIR = RESULT_DIR / 'temp'
DATA_TEMP_DIR_NAME = 'temp'


def read_file_in_batches(path, batch_size=1000, max_length=100):

    data = []
    idx = 0
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        n = len(line.split())
        if n > max_length:
            continue
        data.append((idx, line, n))
        idx += 1

    def _iterator(data):

        data = sorted(data, key=lambda x: x[2], reverse=True)

        maxn = 0
        batch = []

        for sample in data:
            idx, line, n = sample
            if n > batch_size:
                if batch:
                    yield batch
                    maxn = 0
                    batch = []
                yield [sample]
            else:
                curr_batch_size = maxn * len(batch)
                cand_batch_size = max(maxn, n) * (len(batch) + 1)

                if 0 < curr_batch_size <= batch_size and cand_batch_size > batch_size:
                    yield batch
                    maxn = 0
                    batch = []
                maxn = max(maxn, n)
                batch.append(sample)

        if batch:
            yield batch

    return _iterator(data), len(data)


def parse_shard_open(expr):
    if expr is None:
        return None
    if isinstance(expr, (list, tuple)):
        values = list(expr)
    else:
        try:
            values = ast.literal_eval(str(expr))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"无法解析 --shard-open 参数：{expr}") from exc
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("--shard-open 需要形如 [N, gpu1, gpu2, ...] 的列表")
    try:
        num = int(values[0])
        gpus = [int(v) for v in values[1:]]
    except Exception as exc:
        raise ValueError("--shard-open 参数必须是整数") from exc
    if num <= 0:
        raise ValueError("--shard-open 中的并行GPU数必须大于 0")
    if num != len(gpus):
        raise ValueError("--shard-open 的GPU数量与声明的并行数不一致")
    return num, gpus


def split_text_file_for_shards(path, num_shards):
    path = Path(path)
    temp_dir = path.parent / DATA_TEMP_DIR_NAME
    temp_dir.mkdir(parents=True, exist_ok=True)

    stem = path.stem
    suffix = path.suffix

    lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    total = len(lines)
    base = total // num_shards
    remainder = total % num_shards

    shard_infos = []
    start = 0

    result_temp_dir = RESULT_TEMP_DIR / stem
    result_temp_dir.mkdir(parents=True, exist_ok=True)

    for shard_idx in range(num_shards):
        extra = 1 if shard_idx < remainder else 0
        end = start + base + extra
        shard_lines = lines[start:end]
        shard_name = f"{stem}{shard_idx + 1}{suffix}"
        shard_path = temp_dir / shard_name
        content = '\n'.join(shard_lines)
        shard_path.write_text((content + '\n') if content else '', encoding='utf-8')
        shard_result_path = result_temp_dir / f"{shard_name}.amr"
        shard_infos.append({
            'input_path': shard_path,
            'result_path': shard_result_path,
            'line_count': len(shard_lines),
        })
        start = end

    return shard_infos


def build_model_config(args):
    return {
        'model_name': args.model,
        'checkpoint': args.checkpoint,
        'beam_size': args.beam_size,
        'batch_size': args.batch_size,
        'penman_linearization': args.penman_linearization,
        'use_pointer_tokens': args.use_pointer_tokens,
        'restore_name_ops': args.restore_name_ops,
        'only_ok': args.only_ok,
    }


def run_prediction_tasks(tasks, config, device_str, print_amr=True, show_progress=True, progress_callback=None):
    device = torch.device(device_str)
    model, tokenizer = instantiate_model_and_tokenizer(
        config['model_name'],
        dropout=0.,
        attention_dropout=0,
        penman_linearization=config['penman_linearization'],
        use_pointer_tokens=config['use_pointer_tokens'],
    )
    state = torch.load(config['checkpoint'], map_location='cpu')['model']
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    file_iterable = tasks
    if show_progress and len(tasks) > 1:
        file_iterable = tqdm(tasks, desc='Files:', leave=True)

    for task in file_iterable:
        input_path = Path(task['input_path'])
        output_path = Path(task['result_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        iterator, nsent = read_file_in_batches(input_path, config['batch_size'])

        bar = None
        if show_progress:
            bar = tqdm(desc=str(input_path), total=nsent, leave=False)

        with output_path.open('w', encoding='utf-8') as output_stream:
            for batch in iterator:
                if not batch:
                    continue
                ids, sentences, _ = zip(*batch)
                x, _ = tokenizer.batch_encode_sentences(sentences, device=device)
                with torch.no_grad():
                    model.amr_mode = True
                    out = model.generate(**x, max_length=512, decoder_start_token_id=0, num_beams=config['beam_size'])

                bgraphs = []
                for idx, sent, tokk in zip(ids, sentences, out):
                    graph, status, (lin, backr) = tokenizer.decode_amr(
                        tokk.tolist(),
                        restore_name_ops=config['restore_name_ops'],
                    )
                    if config['only_ok'] and ('OK' not in str(status)):
                        continue
                    graph.metadata['status'] = str(status)
                    graph.metadata['source'] = str(input_path)
                    graph.metadata['nsent'] = str(idx)
                    graph.metadata['snt'] = sent
                    bgraphs.append((idx, graph))

                for _, graph in bgraphs:
                    encoded_graph = encode(graph)
                    if print_amr:
                        print(encoded_graph)
                        print()
                    output_stream.write(encoded_graph)
                    output_stream.write('\n\n')

                if progress_callback:
                    progress_callback(len(sentences))
                if bar:
                    bar.update(len(sentences))

        if bar:
            bar.close()

    model.to('cpu')


def shard_worker(config, device_str, tasks, shard_index, progress_queue, error_queue):
    try:
        def _progress(count):
            if count:
                progress_queue.put((shard_index, count))

        run_prediction_tasks(
            tasks=tasks,
            config=config,
            device_str=device_str,
            print_amr=False,
            show_progress=False,
            progress_callback=_progress,
        )
        progress_queue.put((shard_index, 'done'))
    except Exception:  # pragma: no cover - defensive
        tb = traceback.format_exc()
        error_queue.put((shard_index, tb))
        progress_queue.put((shard_index, 'error'))


def aggregate_shard_outputs(original_path, shard_infos):
    original_path = Path(original_path)
    final_path = RESULT_DIR / f"{original_path.name}.amr"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open('w', encoding='utf-8') as fout:
        for info in shard_infos:
            shard_result = Path(info['result_path'])
            if shard_result.exists():
                content = shard_result.read_text(encoding='utf-8')
                if content:
                    fout.write(content)
    return final_path


def orchestrate_shards(args, shard_config):
    num_shards, gpu_ids = shard_config
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    split_map = {}
    for text_path in args.texts:
        split_map[text_path] = split_text_file_for_shards(text_path, num_shards)

    tasks_per_shard = []
    for shard_idx in range(num_shards):
        shard_tasks = []
        for text_path in args.texts:
            shard_tasks.append(split_map[text_path][shard_idx])
        tasks_per_shard.append(shard_tasks)

    totals = [sum(task['line_count'] for task in shard_tasks) for shard_tasks in tasks_per_shard]

    manager = mp.Manager()
    progress_queue = manager.Queue()
    error_queue = manager.Queue()

    config = build_model_config(args)

    processes = []
    for shard_idx, shard_tasks in enumerate(tasks_per_shard):
        device_id = gpu_ids[shard_idx]
        device_str = device_id if isinstance(device_id, str) and device_id.startswith('cuda') else f"cuda:{device_id}"
        proc = mp.Process(
            target=shard_worker,
            args=(config, device_str, shard_tasks, shard_idx, progress_queue, error_queue),
        )
        proc.start()
        processes.append(proc)

    bars = []
    for shard_idx, total in enumerate(totals):
        total_value = max(total, 1)
        bar = tqdm(
            total=total_value,
            desc=f"Shard{shard_idx + 1}:GPU{gpu_ids[shard_idx]}",
            position=shard_idx,
            leave=True,
        )
        if total == 0:
            bar.update(total_value)
        bars.append(bar)

    finished = 0
    while finished < num_shards:
        shard_idx, payload = progress_queue.get()
        if payload == 'done':
            remaining = bars[shard_idx].total - bars[shard_idx].n
            if remaining > 0:
                bars[shard_idx].update(remaining)
            bars[shard_idx].close()
            finished += 1
        elif payload == 'error':
            remaining = bars[shard_idx].total - bars[shard_idx].n
            if remaining > 0:
                bars[shard_idx].update(remaining)
            bars[shard_idx].close()
            finished += 1
        else:
            bars[shard_idx].update(payload)

    for proc in processes:
        proc.join()

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())

    if errors:
        msgs = '\n'.join(f"Shard {idx + 1} failed:\n{tb}" for idx, tb in errors)
        raise RuntimeError(msgs)

    aggregated_paths = []
    for original_path, shard_infos in split_map.items():
        aggregated_paths.append(aggregate_shard_outputs(original_path, shard_infos))

    for path in aggregated_paths:
        print(f"Aggregated AMR saved to {path}")


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--texts', type=str, required=True, nargs='+',
        help="Required. One or more files containing \n-separated sentences.")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=1,
        help="Beam size.")
    parser.add_argument('--batch-size', type=int, default=1000,
        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--restore-name-ops', action='store_true')
    parser.add_argument('--device', type=str, default='cuda',
        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--only-ok', action='store_true')
    parser.add_argument('--shard-open', type=str,
        help="格式: [N, gpu1, gpu2, ...]，启用并行分片推理。")
    args = parser.parse_args()

    if args.shard_open:
        try:
            shard_config = parse_shard_open(args.shard_open)
        except ValueError as exc:
            parser.error(str(exc))
        orchestrate_shards(args, shard_config)
    else:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        tasks = []
        for text_path in args.texts:
            output_path = RESULT_DIR / f"{Path(text_path).name}.amr"
            tasks.append({
                'input_path': Path(text_path),
                'result_path': output_path,
            })
        config = build_model_config(args)
        run_prediction_tasks(
            tasks=tasks,
            config=config,
            device_str=args.device,
            print_amr=True,
            show_progress=True,
            progress_callback=None,
        )


if __name__ == '__main__':
    main()
