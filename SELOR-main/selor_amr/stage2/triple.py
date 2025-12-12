"""Triple data structures for AMR-SELOR (stage2 scope).

【溯源说明】
改造自：selor_utils/atom.py（原 SELOR 原子数据结构）
原文件职能：
  - 定义 Atom 类（词汇原子，如 "word >= threshold"）
  - 定义 AtomPool（样本级原子池）、AtomTokenizer（全局原子词表构建）
  - 提供原子匹配、过滤、序列化等工具函数
本文件职能：
  - 定义 Triple 类（语义三元组，如 (staff, :ARG0, help-01)）
  - 定义 TriplePool（样本级三元组池）
  - 定义 GlobalTripleVocab（全局三元组词表，按频率排序）
核心改造：
  - 原子单元：布尔表达式 → 语义三元组 (head, relation, tail)
  - 表示方式：字符串匹配 → 结构化三元组（支持文本编码或查表）
  - 池管理：固定全局 → 动态按样本（每个样本独立三元组集合）

- Triple: basic unit
- TriplePool: per-sample pool
- GlobalTripleVocab: frequency-ordered vocab builder

Kept minimal to avoid touching original selor_utils; used by downstream scripts if needed.
"""
from dataclasses import dataclass
from collections import Counter
from typing import List, Dict


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    triple_idx: int = -1

    @property
    def display_str(self) -> str:
        return f"({self.head} {self.relation} {self.tail})"

    def to_text(self) -> str:
        return f"{self.head} {self.relation} {self.tail}"


class TriplePool:
    """Manage triples for a single sample."""
    def __init__(self, triples: List[Triple]):
        self.triples = triples
        self.triple2idx: Dict[str, int] = {t.to_text(): i for i, t in enumerate(triples)}

    def num_triples(self) -> int:
        return len(self.triples)

    def get_triple(self, idx: int) -> Triple:
        return self.triples[idx]


class GlobalTripleVocab:
    """Frequency-ordered vocab builder."""
    def __init__(self):
        self.triple2idx: Dict[str, int] = {}
        self.idx2triple: Dict[int, Triple] = {}
        self.triple_count = Counter()

    def add_triple(self, triple: Triple):
        text = triple.to_text()
        if text not in self.triple2idx:
            idx = len(self.triple2idx)
            self.triple2idx[text] = idx
            self.idx2triple[idx] = Triple(triple.head, triple.relation, triple.tail, idx)
        self.triple_count[text] += 1

    def add_triples(self, triples: List[Triple]):
        for t in triples:
            self.add_triple(t)

    def get_frequent_triples(self, min_count: int) -> List[Triple]:
        return [self.idx2triple[self.triple2idx[t]] for t, c in self.triple_count.items() if c >= min_count]

    def to_list(self) -> List[str]:
        # return triples ordered by frequency desc
        ordered = sorted(self.triple_count.items(), key=lambda kv: kv[1], reverse=True)
        return [t for t, _ in ordered]
