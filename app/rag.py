"""Retrieval over the clinic FAQ knowledge base.

Design notes
- ~20 documents x 3 languages: a flat cosine search over a numpy matrix is exact and fast, so no
  external vector database is needed (a DB like Chroma/Qdrant becomes worthwhile at thousands+ passages).
- Each document is indexed per language as separate question and answer passages; results are aggregated to
  the document id (max score), so a Roman Urdu question can retrieve through the Urdu or English passage.
"""
import json
import re
import zlib

import numpy as np

LANGS = ("en", "ur", "rur")


def _clean(text):
    return re.sub(r"[^\w\s]", " ", text.lower())


class HashEmbedder:
    """Char n-gram hashing baseline (no model download). Lexical only: used for tests and as a
    baseline to show what a real multilingual embedding model adds."""
    name = "hash-char-ngram"

    def __init__(self, dim=2048, ngrams=(2, 3, 4)):
        self.dim, self.ngrams = dim, ngrams

    def _one(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        t = f" {' '.join(_clean(text).split())} "
        for n in self.ngrams:
            for i in range(len(t) - n + 1):
                v[zlib.crc32(t[i:i + n].encode("utf-8")) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_queries(self, texts):
        return np.stack([self._one(t) for t in texts])

    embed_passages = embed_queries


class SentenceTransformerEmbedder:
    """Multilingual dense embeddings (default: multilingual-e5-small, ~120M params, supports Urdu)."""

    def __init__(self, model_name="intfloat/multilingual-e5-small", device=None):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.e5 = "e5" in model_name.lower()   # e5 models expect "query: " / "passage: " prefixes

    def _enc(self, texts, prefix):
        texts = [prefix + t for t in texts] if self.e5 else list(texts)
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)

    def embed_queries(self, texts):
        return self._enc(texts, "query: ")

    def embed_passages(self, texts):
        return self._enc(texts, "passage: ")


def build_embedder(kind="hash", model_name="intfloat/multilingual-e5-small"):
    if kind == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(model_name)


class Retriever:
    def __init__(self, kb_path, embedder):
        with open(kb_path, encoding="utf-8") as f:
            self.docs = {d["id"]: d for d in json.load(f)}
        self.embedder = embedder
        self.passages = []          # (doc_id, lang)
        texts = []
        for doc in self.docs.values():
            for lang in LANGS:
                # question and answer are indexed separately: short questions match short user queries,
                # answers catch queries that use the answer's wording; the best score per document wins
                for part in ("q", "a"):
                    self.passages.append((doc["id"], lang))
                    texts.append(doc[lang][part])
        self.matrix = embedder.embed_passages(texts)

    def get(self, doc_id):
        return self.docs[doc_id]

    def search(self, query, k=3):
        q = self.embedder.embed_queries([query])[0]
        sims = self.matrix @ q
        best = {}
        for (doc_id, _lang), s in zip(self.passages, sims):
            best[doc_id] = max(best.get(doc_id, -1.0), float(s))
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [{"doc_id": d, "score": s, "topic": self.docs[d]["topic"]} for d, s in ranked]
