"""
BM25 keyword-based retrieval over the ingested text chunks.
Complements dense (embedding) retrieval for exact-term matches like part
numbers, fuse ratings, and named components that embeddings can under-weight.
"""
import os
import json
import pickle
import re

from rank_bm25 import BM25Okapi

import config

CORPUS_PATH = os.path.join(config.VECTOR_DB_DIR, "bm25_corpus.json")
INDEX_PATH = os.path.join(config.VECTOR_DB_DIR, "bm25_index.pkl")


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def build_index(text_items):
    """Build and persist a BM25 index from the ingested text chunks."""
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(text_items, f)

    tokenized_corpus = [_tokenize(item["content"]) for item in text_items]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"BM25 index built over {len(text_items)} chunks.")


class BM25Retriever:
    def __init__(self):
        if not (os.path.exists(CORPUS_PATH) and os.path.exists(INDEX_PATH)):
            raise FileNotFoundError(
                "BM25 index not found. Run ingestion first (python main.py ingest ...)."
            )
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)
        with open(INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

    def search(self, query, top_k=config.BM25_TOP_K):
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.corpus, scores), key=lambda x: x[1], reverse=True)
        return [
            {"content": item["content"], "page": item["page"], "bm25_score": float(score)}
            for item, score in ranked[:top_k]
            if score > 0
        ]
