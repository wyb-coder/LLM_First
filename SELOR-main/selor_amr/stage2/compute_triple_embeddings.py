"""Optional: compute triple embeddings via text encoding (BERT).

If you prefer true_matrix@CLS, skip this file. Otherwise, this provides a cacheable
text encoder for triples. Keep in selor_amr to avoid touching original code.
"""
import argparse
import os
import pickle
from typing import List

import torch
from transformers import BertTokenizer, BertModel

from triple import Triple  # assuming run from selor_amr/stage2


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class TripleEncoder:
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.cache = {}

    @torch.no_grad()
    def encode(self, triple: Triple) -> torch.Tensor:
        text = triple.to_text()
        if text in self.cache:
            return self.cache[text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        self.cache[text] = emb
        return emb

    @torch.no_grad()
    def encode_batch(self, triples: List[Triple]) -> torch.Tensor:
        texts = [t.to_text() for t in triples]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu()
        for text, vec in zip(texts, emb):
            self.cache[text] = vec
        return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triple_texts_pkl", type=str, required=True, help="List[str] triple texts")
    parser.add_argument("--out_pkl", type=str, default="./saved_models/triple_embeddings/embedding_cache.pkl")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    triple_texts = pickle.load(open(args.triple_texts_pkl, "rb"))
    encoder = TripleEncoder(args.model_name)
    embeddings = {}
    for t in triple_texts:
        dummy = Triple(*t.split(" "), -1)
        embeddings[t] = encoder.encode(dummy)
    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    torch.save(embeddings, args.out_pkl)
    print("Saved embeddings to", args.out_pkl, "count", len(embeddings))


if __name__ == "__main__":
    main()
