"""Triple data structures for AMR-SELOR (stage2 scope).

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
