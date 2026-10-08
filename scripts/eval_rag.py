#!/usr/bin/env python3
"""Retrieval evaluation: recall@1/@3, MRR (document level), per language, retrieval latency, and an
ABSTENTION table: can a score cutoff separate answerable from unanswerable questions?

  python scripts/eval_rag.py --embedder hash          # baseline, no download
  python scripts/eval_rag.py --embedder e5            # multilingual-e5-small (needs internet once)
Files: data/rag_dev.jsonl (iterate on this), data/rag_test.jsonl (YOUR independent set; use --final, ONCE).
Line format: {"lang": "rur", "question": "...", "doc_id": "timings"}
Questions the knowledge base CANNOT answer use  "doc_id": null  (used for the abstention table only).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.rag import Retriever, build_embedder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hash", choices=["hash", "e5"])
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--kb", default="kb/faq.json")
    ap.add_argument("--min-margin", type=float, default=None,
                    help="report results at THIS fixed margin cutoff (choose it on dev, then use it unchanged on --final)")
    ap.add_argument("--final", action="store_true", help="also evaluate data/rag_test.jsonl (do this ONCE)")
    args = ap.parse_args()

    emb = build_embedder(args.embedder, args.model)
    ret = Retriever(ROOT / args.kb, emb)
    out = {"embedder": emb.name}
    for name in ("rag_dev", "rag_test"):
        if name == "rag_test" and not args.final:
            continue
        path = ROOT / "data" / f"{name}.jsonl"
        if not path.exists():
            print(f"[skip] {name}: {path.name} not found")
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        bad = [r for r in rows if r["doc_id"] is not None and r["doc_id"] not in ret.docs]
        assert not bad, f"unknown doc_id in {name}: {sorted({r['doc_id'] for r in bad})}"
        ranks, lat, by, pos, neg = [], [], {}, [], []     # pos: (top1_score, correct, margin); neg: (top1_score, margin)
        for r in rows:
            t0 = time.perf_counter()
            hits = ret.search(r["question"], k=len(ret.docs))
            lat.append((time.perf_counter() - t0) * 1000)
            margin = hits[0]["score"] - hits[1]["score"]
            if r["doc_id"] is None:
                neg.append((hits[0]["score"], margin))
                continue
            rank = [h["doc_id"] for h in hits].index(r["doc_id"]) + 1
            ranks.append(rank)
            pos.append((hits[0]["score"], rank == 1, margin))
            by.setdefault(r["lang"], []).append(rank)
        ranks = np.array(ranks)
        res = {"n_answerable": len(ranks), "n_unanswerable": len(neg),
               "recall@1": float((ranks == 1).mean()), "recall@3": float((ranks <= 3).mean()),
               "mrr": float((1 / ranks).mean()),
               "by_lang": {l: {"n": len(v), "recall@1": float(np.mean(np.array(v) == 1)),
                               "recall@3": float(np.mean(np.array(v) <= 3))} for l, v in sorted(by.items())},
               "search_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}}
        print(f"\n=== {name} ({emb.name}) answerable n={len(ranks)}, unanswerable n={len(neg)} ===")
        print(f"recall@1={res['recall@1']:.3f} recall@3={res['recall@3']:.3f} MRR={res['mrr']:.3f}"
              f"  search p50={res['search_ms']['p50']:.1f}ms p95={res['search_ms']['p95']:.1f}ms")
        for l, m in res["by_lang"].items():
            print(f"  {l:<4} n={m['n']:<3} recall@1={m['recall@1']:.2f} recall@3={m['recall@3']:.2f}")
        if pos and neg:
            def sweep_table(title, pos_v, neg_v):
                allv = [v for v, _ in pos_v] + list(neg_v)
                rows_, print_ = [], [f"\n  abstention by {title} (answer only if value >= cutoff):",
                                     "  cutoff   answerable_answered_correctly   unanswerable_correctly_refused   overall"]
                for thr in np.linspace(min(allv), max(allv) + 1e-6, 14):
                    ok = sum(1 for v, c in pos_v if c and v >= thr)
                    refused = sum(1 for v in neg_v if v < thr)
                    overall = (ok + refused) / (len(pos_v) + len(neg_v))
                    rows_.append({"cutoff": float(thr), "answered_correctly": ok / len(pos_v),
                                  "refused": refused / len(neg_v), "overall": overall})
                    print_.append(f"  {thr:.3f}   {ok / len(pos_v):>20.2f}   {refused / len(neg_v):>30.2f}   {overall:>11.2f}")
                best_ = max(rows_, key=lambda x: x["overall"])
                print("\n".join(print_))
                print(f"  best overall cutoff ~ {best_['cutoff']:.3f} (overall {best_['overall']:.2f})")
                return rows_, best_
            always = len([1 for _, c, _m in pos if c]) / (len(pos) + len(neg))
            print(f"\n  reference: ALWAYS answering scores overall = {always:.2f} (answerable-correct only)")
            sc, bs = sweep_table("top-1 score", [(s_, c) for s_, c, _ in pos], [s_ for s_, _ in neg])
            mg, bm = sweep_table("margin (top1 - top2)", [(m_, c) for _, c, m_ in pos], [m_ for _, m_ in neg])
            print(f"\n  mean top-1 score: answerable-correct={np.mean([s_ for s_, c, _ in pos if c]):.3f} "
                  f"answerable-wrong={np.mean([s_ for s_, c, _ in pos if not c]) if any(not c for _, c, _ in pos) else float('nan'):.3f} "
                  f"unanswerable={np.mean([s_ for s_, _ in neg]):.3f}")
            print(f"  mean margin:      answerable-correct={np.mean([m_ for _, c, m_ in pos if c]):.3f} "
                  f"unanswerable={np.mean([m_ for _, m_ in neg]):.3f}   (small n: rough guide only)")
            res["abstention_score_sweep"], res["abstention_margin_sweep"] = sc, mg
            if args.min_margin is not None:
                t = args.min_margin
                ok = sum(1 for _, c, m_ in pos if c and m_ >= t)
                wrong = sum(1 for _, c, m_ in pos if (not c) and m_ >= t)
                ref_pos = sum(1 for _, _, m_ in pos if m_ < t)
                ref_neg = sum(1 for _, m_ in neg if m_ < t)
                print(f"\n  >>> AT FIXED MARGIN CUTOFF {t}:")
                print(f"      answerable: {ok}/{len(pos)} answered correctly, {wrong} answered wrongly, {ref_pos} sent to front desk")
                print(f"      unanswerable: {ref_neg}/{len(neg)} correctly refused, {len(neg) - ref_neg} answered with a wrong FAQ")
                res["fixed_margin"] = {"cutoff": t, "answerable_correct": ok, "answerable_wrong": wrong,
                                       "answerable_refused": ref_pos, "n_answerable": len(pos),
                                       "unanswerable_refused": ref_neg, "n_unanswerable": len(neg)}
            res["always_answer_overall"] = always
        elif not neg:
            print("  (no unanswerable questions in this file: add rows with \"doc_id\": null for the abstention table)")
            if args.min_margin is not None and pos:
                t = args.min_margin
                ok = sum(1 for _, c, m_ in pos if c and m_ >= t)
                wrong = sum(1 for _, c, m_ in pos if (not c) and m_ >= t)
                refused = sum(1 for _, _, m_ in pos if m_ < t)
                print(f"\n  >>> AT FIXED MARGIN CUTOFF {t} (answerable questions only; cost of the cutoff):")
                print(f"      {ok}/{len(pos)} answered correctly, {wrong} answered wrongly, {refused} sent to front desk")
                res["fixed_margin_answerable_only"] = {"cutoff": t, "answerable_correct": ok, "answerable_wrong": wrong,
                                                       "answerable_refused": refused, "n_answerable": len(pos)}
        out[name] = res
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"rag_metrics_{args.embedder}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()