#!/usr/bin/env python3
"""
Evaluate a trained model on the synthetic test set and the hand-written test set,
write results/metrics.json + results/errors_*.jsonl, and measure latency.

  python scripts/eval_nlu.py --model-dir models/nlu
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nlu.core import JointNLU, evaluate, latency, load_jsonl, load_label_map  # noqa: E402


def show(name, res):
    print(f"\n=== {name} ===")
    print(f"{'subset':<10}{'n':>5}{'intent_acc':>12}{'intent_F1':>11}{'entity_F1':>11}{'ent_F1_merged':>15}")
    rows = [("overall", res["overall"])] + [(k, v) for k, v in res["by_lang"].items()]
    for k, m in rows:
        print(f"{k:<10}{m['n']:>5}{m['intent_acc']:>12.3f}{m['intent_macro_f1']:>11.3f}"
              f"{m.get('entity_f1', float('nan')):>11.3f}{m.get('entity_f1_merged', float('nan')):>15.3f}")
    print("per-intent F1 (support):",
          {k: f"{v['f1']:.2f} ({v['support']})" for k, v in res["by_intent_f1"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/nlu")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--final", action="store_true", help="also evaluate data/final_test (do this ONCE, after freezing)")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    mdir = Path(args.model_dir)
    lm = load_label_map(ROOT)
    tok = AutoTokenizer.from_pretrained(mdir)
    enc = AutoModel.from_config(AutoConfig.from_pretrained(mdir))
    model = JointNLU(enc, len(lm["intents"]), len(lm["ner_labels"]))
    model.load_state_dict(torch.load(mdir / "model.pt", map_location="cpu"))
    model.to(device).eval()

    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    metrics = {"model_dir": str(mdir), "device": str(device)}
    sets = {"synthetic_test": ROOT / "data/synthetic/test.jsonl",
            "dev_handwritten": ROOT / "data/dev_handwritten.bio.jsonl",   # used for error analysis / iteration
            "final_test": ROOT / "data/final_test.bio.jsonl"}              # headline numbers; evaluate ONCE
    lat_src = None
    for name, path in sets.items():
        if name == "final_test" and not args.final:
            continue
        if not path.exists():
            print(f"[skip] {name}: {path} not found (run scripts/validate_handwritten.py)")
            continue
        recs = load_jsonl(path)
        res, errors = evaluate(model, tok, recs, lm, device)
        metrics[name] = res
        show(name, res)
        (out / f"errors_{name}.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in errors) + "\n", encoding="utf-8")
        print(f"errors: {len(errors)}/{len(recs)} -> results/errors_{name}.jsonl")
        if name in ("dev_handwritten", "final_test") and lat_src is None:
            lat_src = [r["tokens"] for r in recs]
    if lat_src is None and "synthetic_test" in metrics:
        lat_src = [r["tokens"] for r in load_jsonl(sets["synthetic_test"])[:50]]
    if lat_src:
        metrics["latency"] = latency(model, tok, lat_src, lm, device)
        print("\nlatency:", metrics["latency"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nwrote", out / "metrics.json")


if __name__ == "__main__":
    main()
