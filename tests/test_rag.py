import json
import os

from app.rag import HashEmbedder, Retriever

KB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", "faq.json")


def test_kb_integrity():
    docs = json.load(open(KB, encoding="utf-8"))
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids)) >= 20
    for d in docs:
        for lang in ("en", "ur", "rur"):
            assert d[lang]["q"].strip() and d[lang]["a"].strip(), (d["id"], lang)


def test_retrieval_exact_question_found_in_every_language():
    r = Retriever(KB, HashEmbedder())
    for d in r.docs.values():
        for lang in ("en", "ur", "rur"):
            assert r.search(d[lang]["q"], k=1)[0]["doc_id"] == d["id"]


def test_search_returns_sorted_topk():
    r = Retriever(KB, HashEmbedder())
    hits = r.search("where is the clinic", k=3)
    assert len(hits) == 3 and hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]
