from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import json
import lzma
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, TextIO, Tuple, Union

import penman
from tqdm import tqdm

from formatting import extract_triplets

"""1q231"""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AMR_PATH = PROJECT_ROOT / 'result' / 'train.sent.txt.amr'
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'outputs' / 'train_amr_triples.csv'
COMPRESSION_SUFFIX_MAP = {
    '.gz': 'gzip',
    '.bz2': 'bz2',
    '.xz': 'xz',
}


def detect_compression(path: Union[Path, str], option: str) -> str:
    if option != 'infer':
        return option
    suffix = Path(str(path)).suffix.lower()
    return COMPRESSION_SUFFIX_MAP.get(suffix, 'none')


@contextmanager
def smart_open(path: Union[Path, str], compression: str) -> Iterator[TextIO]:
    if str(path) == '-':
        if compression not in {'none', 'infer'}:
            raise ValueError('stdout 仅支持未压缩输出，请改用管道进行压缩。')
        yield sys.stdout
        return
    actual_compression = detect_compression(path, compression)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if actual_compression == 'gzip':
        with gzip.open(file_path, 'wt', encoding='utf-8', newline='') as handle:
            yield handle
    elif actual_compression == 'bz2':
        with bz2.open(file_path, 'wt', encoding='utf-8', newline='') as handle:
            yield handle
    elif actual_compression == 'xz':
        with lzma.open(file_path, 'wt', encoding='utf-8', newline='') as handle:
            yield handle
    else:
        with file_path.open('w', encoding='utf-8', newline='') as handle:
            yield handle


def normalize_concept(value: Optional[str]) -> str:
    if not value:
        return ''
    value = value.strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return value[1:-1]
    if '-' in value:
        head, tail = value.rsplit('-', 1)
        if tail.isdigit():
            return head
    return value


def parse_metadata_line(line: str) -> Tuple[str, str]:
    content = line[4:].strip()
    if not content:
        return '', ''
    parts = content.split(' ', 1)
    key = parts[0].strip()
    value = parts[1].strip() if len(parts) > 1 else ''
    return key, value


def iter_amr_entries(path: Path) -> Iterable[Tuple[Dict[str, str], str]]:
    metadata: Dict[str, str] = {}
    graph_lines: List[str] = []
    with path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.rstrip('\n')
            if line.startswith('# ::'):
                if graph_lines:
                    yield metadata, '\n'.join(graph_lines)
                    metadata = {}
                    graph_lines = []
                key, value = parse_metadata_line(line)
                if key:
                    metadata[key] = value
                continue
            if not line.strip():
                if graph_lines:
                    yield metadata, '\n'.join(graph_lines)
                    metadata = {}
                    graph_lines = []
                continue
            graph_lines.append(line)
    if graph_lines:
        yield metadata, '\n'.join(graph_lines)


def build_relationship_strings(graph: penman.Graph) -> List[str]:
    relationships: List[str] = []
    for triple in graph.edges():
        role = triple.role.lstrip(':')
        if role not in {'ARG0', 'ARG1'}:
            continue
        relationships.append(f"{role}({triple.source}, {triple.target}) ^")
    return relationships


def triples_to_tokens(triples: List[List[str]], id_to_concept: Dict[str, str]) -> List[str]:
    results: List[str] = []
    for triple in triples:
        tokens: List[str] = []
        for symbol in triple:
            concept = id_to_concept.get(symbol, normalize_concept(symbol))
            if concept:
                tokens.append(concept)
        if tokens:
            results.append(' '.join(tokens))
    return results


def amr_to_instance(
    amr_file: Path,
    output_csv: Path,
    show_progress: bool = True,
    include_amr: bool = True,
    compression: str = 'infer',
) -> None:
    if not amr_file.exists():
        raise FileNotFoundError(f"AMR文件不存在: {amr_file}")
    dedup: Dict[Tuple[str, object], Dict[str, object]] = {}
    order_counter = 0
    iterator: Iterable[Tuple[Dict[str, str], str]] = iter_amr_entries(amr_file)
    if show_progress:
        iterator = tqdm(iterator, desc='Parsing AMR', unit='sent', dynamic_ncols=True)

    for metadata, graph_text in iterator:
        if not graph_text.strip():
            continue
        order_counter += 1
        nsent_raw = metadata.get('nsent') or metadata.get('id')
        try:
            nsent_val = int(nsent_raw) if nsent_raw is not None else None
        except ValueError:
            nsent_val = None
        sentence = metadata.get('snt', '').replace('\\n', '\n')
        status = metadata.get('status', '')
        source = metadata.get('source', '')
        try:
            graph = penman.decode(graph_text)
        except Exception as exc:  # pragma: no cover - robust parsing
            print(f"跳过nsent={nsent_val}的AMR，解码失败：{exc}", file=sys.stderr)
            continue
        id_to_concept: Dict[str, str] = {}
        for node, _, concept in graph.instances():
            id_to_concept[node] = normalize_concept(concept)
        relationships = build_relationship_strings(graph)
        triples_id = extract_triplets(relationships)
        triples_text = triples_to_tokens(triples_id, id_to_concept)
        record: Dict[str, object] = {
            'nsent': nsent_val,
            'sentence': sentence,
            'triples': json.dumps(triples_text, ensure_ascii=False),
            'status': status,
            'source': source,
            '_order': order_counter,
        }
        if include_amr:
            record['amr'] = graph_text.strip()

        source_val = record['source'] or ''
        if nsent_val is not None:
            key = (source_val, nsent_val)
        else:
            key = (source_val, f"missing-{record['_order']}")
        dedup[key] = record

    if not dedup:
        raise RuntimeError(f"未在 {amr_file} 中解析到任何AMR块")

    sorted_rows = sorted(
        dedup.values(),
        key=lambda item: (
            item['source'] or '',
            item['nsent'] if item['nsent'] is not None else float('inf'),
            item['_order'],
        ),
    )
    for row in sorted_rows:
        row.pop('_order', None)

    fieldnames = ['nsent', 'sentence', 'triples', 'status', 'source']
    if include_amr:
        fieldnames.append('amr')

    with smart_open(output_csv, compression) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({name: row.get(name, '') for name in fieldnames})

    target_desc = 'stdout' if str(output_csv) == '-' else output_csv
    print(f"已写入 {len(sorted_rows)} 条句子及三元组到 {target_desc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='将AMR文件转换为句子-三元组CSV')
    parser.add_argument(
        '--amr-file',
        type=Path,
        default=DEFAULT_AMR_PATH,
        help='输入AMR文件路径，默认使用 outputs/train_amr.txt',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help='输出CSV路径，默认写入 outputs/train_amr_triples.csv，可传 - 输出到标准输出',
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='禁用进度条显示（默认开启）。',
    )
    parser.add_argument(
        '--omit-amr',
        action='store_true',
        help='不在CSV中包含AMR原文，显著减少文件体积。',
    )
    parser.add_argument(
        '--compression',
        choices=['infer', 'none', 'gzip', 'bz2', 'xz'],
        default='infer',
        help='输出压缩格式，infer 会根据文件扩展名自动判定（.gz/.bz2/.xz）。',
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    amr_to_instance(
        args.amr_file,
        args.output,
        show_progress=not args.no_progress,
        include_amr=not args.omit_amr,
        compression=args.compression,
    )


if __name__ == '__main__':
    main()



