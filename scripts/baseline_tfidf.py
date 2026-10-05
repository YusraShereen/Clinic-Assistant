#!/usr/bin/env python3
"""Classical baseline for INTENT only: char n-gram TF-IDF + logistic regression (CPU, seconds)."""
import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nlu.core import load_jsonl  # noqa: E402

train = load_jsonl(ROOT / "data/synthetic/train.jsonl")
clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                    LogisticRegression(max_iter=2000, C=5.0))
clf.fit([r["text"] for r in train], [r["intent"] for r in train])

FINAL = "--final" in sys.argv   # final_test only when explicitly requested (evaluate ONCE)
results = {}
for name, p in [("synthetic_test", "data/synthetic/test.jsonl"),
                ("dev_handwritten", "data/dev_handwritten.bio.jsonl"),
                ("final_test", "data/final_test.bio.jsonl")]:
    path = ROOT / p
    if name == "final_test" and not FINAL:
        continue
    if not path.exists():
        continue
    recs = load_jsonl(path)
    pred = clf.predict([r["text"] for r in recs])
    gold = [r["intent"] for r in recs]
    results[name] = {"n": len(recs), "intent_acc": accuracy_score(gold, pred),
                     "intent_macro_f1": f1_score(gold, pred, average="macro", zero_division=0)}
    print(f"{name:<18} n={len(recs):<4} acc={results[name]['intent_acc']:.3f} "
          f"macroF1={results[name]['intent_macro_f1']:.3f}")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/baseline_tfidf.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
