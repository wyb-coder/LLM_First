import argparse
from pathlib import Path
import ast
import csv
import multiprocessing as mp
import sys
import traceback
from typing import Dict, List, Optional, Set, Tuple

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from spring_amr.penman import encode
from spring_amr.utils import instantiate_model_and_tokenizer



RESULT_DIR = Path(__file__).resolve().parent.parent / 'result'
RESULT_TEMP_DIR = RESULT_DIR / 'temp'
DATA_TEMP_DIR_NAME = 'temp'
META_FIELDNAMES = ['review_id', 'sentence_id', 'sentence', 'amr']
SEPARATOR_LINE = "===============================开始推理==============================="
DEFAULT_MAX_SENT_LENGTH = 100


class ContinueValidationError(RuntimeError):
    """Raised when --continue 检查失败"""


def strtobool_flag(value: str) -> bool:
    txt = str(value).strip().lower()
    if txt in {'true', '1', 'yes', 'y'}:
        return True
    if txt in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError("--continue 仅接受 True/False")


def load_sentence_review_map(map_path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with map_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            try:
                sent_id = int(parts[0])
            except ValueError:
                continue
            mapping[sent_id] = parts[1]
    return mapping


def parse_sentence_line(line: str, fallback_idx: int) -> Tuple[int, str]:
    parts = line.split('\t', 1)
    if len(parts) == 2:
        sent_id_txt = parts[0].strip()
        if sent_id_txt.isdigit():
            return int(sent_id_txt), parts[1].strip()
    return fallback_idx, line


def load_meta_rows(meta_path: Path) -> Optional[List[Dict[str, str]]]:
    if not meta_path or not meta_path.exists():
        return None
    rows: List[Dict[str, str]] = []
    with meta_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def write_meta_rows(meta_path: Path, rows: List[Dict[str, str]]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=META_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in META_FIELDNAMES})


def write_amr_from_meta(meta_rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for row in meta_rows:
            amr_txt = (row.get('amr') or '').strip()
            if not amr_txt:
                continue
            handle.write(amr_txt)
            if not amr_txt.endswith('\n'):
                handle.write('\n')
            handle.write('\n')


def aggregate_meta_files(meta_paths: List[Path], aggregated_path: Path) -> None:
    if not meta_paths:
        return
    aggregated_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregated_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=META_FIELDNAMES)
        writer.writeheader()
        for meta_path in meta_paths:
            if not meta_path.exists():
                continue
            with meta_path.open('r', encoding='utf-8', newline='') as reader_handle:
                reader = csv.DictReader(reader_handle)
                for row in reader:
                    writer.writerow({field: row.get(field, '') for field in META_FIELDNAMES})


def resolve_map_paths(text_paths: List[str], map_args: Optional[List[str]]) -> List[Optional[Path]]:
    if map_args and len(map_args) != len(text_paths):
        raise ValueError("--maps 参数数量必须与 --texts 一致")
    resolved: List[Optional[Path]] = []
    for idx, text_path in enumerate(text_paths):
        if map_args:
            candidate = map_args[idx]
            resolved.append(Path(candidate) if candidate else None)
            continue
        default_map = Path(f"{text_path}.map.tsv")
        resolved.append(default_map if default_map.exists() else None)
    return resolved


def load_sentences_with_meta(text_path: Path, map_path: Optional[Path], max_length: int = DEFAULT_MAX_SENT_LENGTH) -> List[Dict[str, str]]:
    review_map = load_sentence_review_map(map_path) if map_path and map_path.exists() else {}
    sentences: List[Dict[str, str]] = []
    fallback_idx = 0
    dropped = 0
    for raw_line in text_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        sentence_id, sentence = parse_sentence_line(line, fallback_idx)
        fallback_idx += 1
        if max_length and len(sentence.split()) > max_length:
            dropped += 1
            continue
        review_id = review_map.get(sentence_id, '')
        sentences.append({
            'sentence_id': str(sentence_id),
            'sentence': sentence,
            'review_id': review_id,
            'amr': '',
        })
    if dropped:
        print(f"[{text_path.name}] skipped {dropped} sentences longer than {max_length} words")
    return sentences


def load_existing_shards(path: Path, num_shards: int) -> List[Dict[str, object]]:
    temp_dir = path.parent / DATA_TEMP_DIR_NAME
    result_temp_dir = RESULT_TEMP_DIR / path.stem
    errors: List[str] = []
    shard_infos: List[Dict[str, object]] = []

    if not temp_dir.exists():
        errors.append(f"缺少分片目录: {temp_dir}")
    if not result_temp_dir.exists():
        errors.append(f"缺少结果临时目录: {result_temp_dir}")

    for shard_idx in range(num_shards):
        shard_name = f"{path.stem}{shard_idx + 1}{path.suffix}"
        shard_input_path = temp_dir / shard_name
        shard_meta_path = result_temp_dir / f"{shard_name}.csv"
        shard_result_path = result_temp_dir / f"{shard_name}.amr"
        if not shard_input_path.exists():
            errors.append(f"找不到输入分片: {shard_input_path}")
            continue
        if not shard_meta_path.exists():
            errors.append(f"找不到中间CSV: {shard_meta_path}")
            continue
        meta_rows = load_meta_rows(shard_meta_path)
        if not meta_rows:
            errors.append(f"CSV 为空: {shard_meta_path}")
            continue
        pending = sum(1 for row in meta_rows if not (row.get('amr') or '').strip())
        shard_infos.append({
            'input_path': shard_input_path,
            'result_path': shard_result_path,
            'meta_path': shard_meta_path,
            'line_count': len(meta_rows),
            'pending_count': pending,
        })

    if errors:
        raise ContinueValidationError('; '.join(errors))

    pending_total = sum(info.get('pending_count', 0) for info in shard_infos)
    print(f"继续模式检测成功：{path.name} 分片 {num_shards} 份，剩余 {pending_total} 句待推理")
    return shard_infos


def read_file_in_batches(path, batch_size=1000, max_length=DEFAULT_MAX_SENT_LENGTH, skip_indices: Optional[Set[int]] = None):

    data = []
    idx = 0
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        n = len(line.split())
        if n > max_length:
            continue
        if skip_indices and idx in skip_indices:
            idx += 1
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


def split_text_file_for_shards(path, map_path, num_shards, continue_mode=False):
    path = Path(path)
    map_path = Path(map_path) if map_path else None
    if continue_mode:
        try:
            return load_existing_shards(path, num_shards)
        except ContinueValidationError as exc:
            print(f"继续模式检测失败：{exc}")
            raise

    temp_dir = path.parent / DATA_TEMP_DIR_NAME
    temp_dir.mkdir(parents=True, exist_ok=True)

    stem = path.stem
    suffix = path.suffix

    sentences = load_sentences_with_meta(path, map_path)
    total = len(sentences)
    base = total // num_shards
    remainder = total % num_shards

    shard_infos = []
    start = 0

    result_temp_dir = RESULT_TEMP_DIR / stem
    result_temp_dir.mkdir(parents=True, exist_ok=True)

    for shard_idx in range(num_shards):
        extra = 1 if shard_idx < remainder else 0
        end = start + base + extra
        shard_rows = sentences[start:end]
        shard_name = f"{stem}{shard_idx + 1}{suffix}"
        shard_path = temp_dir / shard_name
        content = '\n'.join(row['sentence'] for row in shard_rows)
        shard_path.write_text((content + '\n') if content else '', encoding='utf-8')
        shard_result_path = result_temp_dir / f"{shard_name}.amr"
        shard_meta_path = result_temp_dir / f"{shard_name}.csv"
        write_meta_rows(shard_meta_path, shard_rows)
        shard_infos.append({
            'input_path': shard_path,
            'result_path': shard_result_path,
            'meta_path': shard_meta_path,
            'line_count': len(shard_rows),
            'pending_count': len(shard_rows),
        })
        print(f"Prepared {shard_name}: {len(shard_rows)} sentences -> {shard_path} / {shard_meta_path}")
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
        'resume_mode': args.continue_run,
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
        meta_path = Path(task['meta_path']) if task.get('meta_path') else None
        meta_rows = load_meta_rows(meta_path) if meta_path else None
        output_path.parent.mkdir(parents=True, exist_ok=True)

        resume_mode = bool(config.get('resume_mode'))
        skip_indices: Set[int] = set()
        if resume_mode and meta_rows:
            skip_indices = {
                idx for idx, row in enumerate(meta_rows)
                if (row.get('amr') or '').strip()
            }
            if skip_indices:
                print(f"{input_path.name}: 继续模式跳过 {len(skip_indices)} 条已完成句子")

        iterator, nsent = read_file_in_batches(
            input_path,
            batch_size=config['batch_size'],
            max_length=DEFAULT_MAX_SENT_LENGTH,
            skip_indices=skip_indices if skip_indices else None,
        )

        if meta_rows is not None:
            expected_pending = len(meta_rows) - len(skip_indices)
            if nsent != expected_pending:
                print(
                    f"WARNING: {input_path} pending sentences ({nsent}) != meta rows diff ({expected_pending})"
                )

        bar = None
        if show_progress:
            bar = tqdm(desc=str(input_path), total=nsent, leave=False)

        meta_changed = False
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

                batch_meta_updated = False
                for idx_value, graph in bgraphs:
                    encoded_graph = encode(graph)
                    if print_amr:
                        print(encoded_graph)
                        print()
                    output_stream.write(encoded_graph)
                    output_stream.write('\n\n')
                    if meta_rows is not None and 0 <= idx_value < len(meta_rows):
                        meta_rows[idx_value]['amr'] = encoded_graph
                        meta_changed = True
                        batch_meta_updated = True
                        if meta_path:
                            write_meta_rows(meta_path, meta_rows)

                if progress_callback:
                    progress_callback(len(sentences))
                if bar:
                    bar.update(len(sentences))
                if batch_meta_updated and meta_rows is not None and meta_path:
                    write_meta_rows(meta_path, meta_rows)

        if bar:
            bar.close()

        if meta_rows is not None:
            if not meta_changed:
                print(f"NOTE: meta file {meta_path} unchanged (no AMRs assigned)")
            elif meta_path:
                write_meta_rows(meta_path, meta_rows)
            write_amr_from_meta(meta_rows, output_path)

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
    final_meta_path = RESULT_DIR / f"{original_path.name}.csv"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open('w', encoding='utf-8') as fout:
        for info in shard_infos:
            shard_result = Path(info['result_path'])
            meta_path = Path(info['meta_path']) if info.get('meta_path') else None
            if not shard_result.exists() and meta_path and meta_path.exists():
                meta_rows = load_meta_rows(meta_path)
                if meta_rows:
                    write_amr_from_meta(meta_rows, shard_result)
            if shard_result.exists():
                content = shard_result.read_text(encoding='utf-8')
                if content:
                    fout.write(content)
    meta_paths = [Path(info['meta_path']) for info in shard_infos if info.get('meta_path')]
    aggregate_meta_files(meta_paths, final_meta_path)
    return final_path, final_meta_path


def finalize_outputs(split_map):
    for original_path, shard_infos in split_map.items():
        amr_path, meta_path = aggregate_shard_outputs(original_path, shard_infos)
        print(f"Aggregated AMR saved to {amr_path}")
        print(f"Aggregated CSV saved to {meta_path}")


def sync_results_from_meta(shard_infos: List[Dict[str, object]]) -> None:
    for info in shard_infos:
        meta_path = Path(info['meta_path']) if info.get('meta_path') else None
        result_path = Path(info['result_path'])
        if meta_path and meta_path.exists():
            meta_rows = load_meta_rows(meta_path)
            if meta_rows:
                write_amr_from_meta(meta_rows, result_path)


def orchestrate_shards(args, shard_config, map_paths):
    num_shards, gpu_ids = shard_config
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    print(SEPARATOR_LINE)

    split_map = {}
    try:
        for text_path, map_path in zip(args.texts, map_paths):
            split_map[text_path] = split_text_file_for_shards(
                text_path,
                map_path,
                num_shards,
                continue_mode=args.continue_run,
            )
    except ContinueValidationError:
        return

    pending_total = sum(
        info.get('pending_count', info['line_count'])
        for shard_infos in split_map.values()
        for info in shard_infos
    )
    if args.continue_run and pending_total == 0:
        print("继续模式：未发现待处理句子，直接聚合已有结果。")
        for shard_infos in split_map.values():
            sync_results_from_meta(shard_infos)
        finalize_outputs(split_map)
        return

    tasks_per_shard = []
    for shard_idx in range(num_shards):
        shard_tasks = []
        for text_path in args.texts:
            shard_tasks.append(split_map[text_path][shard_idx])
        tasks_per_shard.append(shard_tasks)

    totals = [
        sum(task.get('pending_count', task['line_count']) for task in shard_tasks)
        for shard_tasks in tasks_per_shard
    ]

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

    finalize_outputs(split_map)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--texts', type=str, required=True, nargs='+',
        help="Required. One or more files containing \n-separated sentences.")
    parser.add_argument('--maps', type=str, nargs='*',
        help="Optional. 文件映射 (sentence_id -> review_id)，需与 --texts 一一对应。默认读取 <text>.map.tsv")
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
    parser.add_argument('--continue', dest='continue_run', type=strtobool_flag, default=False,
        help="是否启用断点续推 (True/False)")
    args = parser.parse_args()

    try:
        map_paths = resolve_map_paths(args.texts, args.maps)
    except ValueError as exc:
        parser.error(str(exc))

    if args.shard_open:
        try:
            shard_config = parse_shard_open(args.shard_open)
        except ValueError as exc:
            parser.error(str(exc))
        orchestrate_shards(args, shard_config, map_paths)
    else:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        split_map = {}
        tasks = []
        try:
            for text_path, map_path in zip(args.texts, map_paths):
                shard_infos = split_text_file_for_shards(
                    text_path,
                    map_path,
                    1,
                    continue_mode=args.continue_run,
                )
                split_map[text_path] = shard_infos
                tasks.extend(shard_infos)
        except ContinueValidationError:
            return
        pending_total = sum(task.get('pending_count', task['line_count']) for task in tasks)
        if args.continue_run and pending_total == 0:
            print("继续模式：未发现待处理句子，直接聚合已有结果。")
            for shard_infos in split_map.values():
                sync_results_from_meta(shard_infos)
            finalize_outputs(split_map)
            return
        config = build_model_config(args)
        print(SEPARATOR_LINE)
        run_prediction_tasks(
            tasks=tasks,
            config=config,
            device_str=args.device,
            print_amr=True,
            show_progress=True,
            progress_callback=None,
        )
        finalize_outputs(split_map)


if __name__ == '__main__':
    main()
