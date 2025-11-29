import argparse, csv, os, re

MIN_SPLIT_WORDS = 8  # skip sentence splitting for very short reviews


def parse_max_sent(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, int):
        return (value, 0)
    txt = str(value).strip()
    if not txt:
        raise argparse.ArgumentTypeError("max-sent-per-review cannot be empty")
    if "," in txt:
        parts = [p.strip() for p in txt.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("Expected N,M for max-sent-per-review")
        try:
            max_sent = int(parts[0])
            min_words = int(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("max-sent-per-review values must be integers") from exc
        if max_sent < 0 or min_words < 0:
            raise argparse.ArgumentTypeError("max-sent-per-review values must be non-negative")
        return (max_sent, min_words)
    try:
        max_sent = int(txt)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-sent-per-review must be an integer or N,M") from exc
    if max_sent < 0:
        raise argparse.ArgumentTypeError("max-sent-per-review must be non-negative")
    return (max_sent, 0)

def simple_sent_split(text: str):
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks = re.findall(r"[^.!?]+(?:[.!?]+|$)", normalized)
    sentences = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and not re.fullmatch(r"[.!?]+", chunk):
            sentences.append(chunk)
    return sentences

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="输入CSV路径（建议使用已添加 review_id 的文件）")
    ap.add_argument("--review-id-col", type=int, default=None, help="review_id 所在列索引（无表头时设置，留空则退回默认行号）。")
    ap.add_argument("--label-col", type=int, default=None, help="label 所在列索引（默认按旧版CSV第0列）。")
    ap.add_argument("--text-col", type=int, default=None, help="正文所在列索引（默认按旧版CSV第1列）。")
    ap.add_argument("--has-header", action="store_true", help="输入CSV首行是否为表头。开启后会按列名 review_id/label/text 自动定位。")
    ap.add_argument("--out", required=True, help="输出txt路径（逐行文本）")
    ap.add_argument("--split-sentences", action="store_true", help="是否做句子级切分")
    ap.add_argument("--max-sent-per-review", type=parse_max_sent, default=parse_max_sent(0),
                   help="每条评论最多保留的句子数；可写N或N,M（最少词数）")
    args = ap.parse_args()

    max_sent_limit, min_sentence_words = args.max_sent_per_review

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    labels_path = args.out + ".labels"
    map_path = args.out + ".map.tsv" if args.split_sentences else None

    n_lines = 0
    with open(args.csv, "r", encoding="utf-8", newline="") as fcsv, \
         open(args.out, "w", encoding="utf-8", newline="\n") as ftxt, \
         open(labels_path, "w", encoding="utf-8", newline="\n") as flab, \
         (open(map_path, "w", encoding="utf-8", newline="\n") if map_path else open(os.devnull, "w")) as fmap:

        reader = csv.reader(fcsv)
        header = None
        review_id_idx = args.review_id_col
        if args.label_col is not None:
            label_idx = args.label_col
        else:
            label_idx = 1 if review_id_idx == 0 else 0
        if args.text_col is not None:
            text_idx = args.text_col
        else:
            text_idx = (label_idx + 1) if label_idx is not None else 1

        if args.has_header:
            header = next(reader, None)
            if header:
                def find_col(name: str, fallback: int | None) -> int | None:
                    try:
                        return header.index(name)
                    except ValueError:
                        return fallback

                review_id_idx = find_col('review_id', review_id_idx)
                label_idx = find_col('label', label_idx)
                text_idx = find_col('text', text_idx)

        def get_value(row, idx, default=''):
            if idx is None or idx >= len(row):
                return default
            return row[idx].strip()

        for row_idx, row in enumerate(reader):
            if len(row) < 2:
                continue
            review_id = get_value(row, review_id_idx, '') if review_id_idx is not None else ''
            if not review_id:
                review_id = str(row_idx)
            label_fallback = row[0].strip() if row else ''
            label = get_value(row, label_idx, label_fallback)
            raw_text = row[text_idx] if text_idx is not None and text_idx < len(row) else (row[1] if len(row) > 1 else '')
            text = raw_text.replace("\r\n", " ").replace("\n", " ").strip()
            if not text:
                continue

            def write_sentence(sentence: str, sent_idx: int) -> None:
                nonlocal n_lines
                ftxt.write(f"{n_lines}\t{sentence}\n")
                flab.write(label + "\n")
                if args.split_sentences:
                    fmap.write(f"{n_lines}\t{review_id}\n")
                n_lines += 1

            if args.split_sentences and len(text.split()) >= MIN_SPLIT_WORDS:
                sents = simple_sent_split(text)
                if not sents:
                    sents = [text]
                if min_sentence_words > 0 or (max_sent_limit and max_sent_limit > 0):
                    sent_infos = []
                    for original_idx, sent in enumerate(sents):
                        word_count = len(sent.split())
                        if min_sentence_words > 0 and word_count < min_sentence_words:
                            continue
                        sent_infos.append((original_idx, sent, word_count))
                    if min_sentence_words > 0:
                        if max_sent_limit and max_sent_limit > 0:
                            sent_infos.sort(key=lambda item: (-item[2], item[0]))
                            sent_infos = sent_infos[:max_sent_limit]
                        else:
                            sent_infos.sort(key=lambda item: item[0])
                        sents = [item[1] for item in sent_infos]
                    elif max_sent_limit and max_sent_limit > 0:
                        sents = [item[1] for item in sent_infos[:max_sent_limit]]
                if max_sent_limit and max_sent_limit > 0 and min_sentence_words == 0:
                    if len(sents) > max_sent_limit:
                        sents = sents[:max_sent_limit]
                if not sents:
                    continue
                for sent_idx, sent in enumerate(sents):
                    write_sentence(sent, sent_idx)
            else:
                write_sentence(text, 0)

    print(f"wrote {n_lines} lines to {args.out}")
    print(f"wrote labels to {labels_path}")
    if map_path:
        with open(map_path, 'r', encoding='utf-8') as fmap_read:
            map_lines = sum(1 for _ in fmap_read)
        print(f"wrote mapping to {map_path} (lines: {map_lines})")
        if map_lines != n_lines:
            print(f"WARNING: map.tsv lines ({map_lines}) != sentences ({n_lines})")

if __name__ == "__main__":
    main()