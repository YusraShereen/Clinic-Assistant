#!/usr/bin/env python3
"""Choose INTENT_THRESHOLD for the orchestrator from VALIDATION data only (never the final test set).
For each threshold: share of messages the bot would answer, accuracy among those, and the
share it would send to 'please rephrase' / human.

  python scripts/pick_threshold.py --model-dir models/nlu
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nlu.core import JointNLU, load_jsonl, load_label_map, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/nlu")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    mdir, dev = Path(args.model_dir), torch.device(args.device)
    lm = load_label_map(ROOT)
    tok = AutoTokenizer.from_pretrained(mdir)
    model = JointNLU(AutoModel.from_config(AutoConfig.from_pretrained(mdir)), len(lm["intents"]), len(lm["ner_labels"]))
    model.load_state_dict(torch.load(mdir / "model.pt", map_location="cpu"))
    model.to(dev).eval()
    for name, path in [("synthetic_val", ROOT / "data/synthetic/val.jsonl"),
                       ("dev_handwritten", ROOT / "data/dev_handwritten.bio.jsonl")]:
        if not path.exists():
            continue
        recs = load_jsonl(path)
        pi, _, pc = predict(model, tok, [r["tokens"] for r in recs], lm, dev, return_conf=True)
        ok = [p == r["intent"] for p, r in zip(pi, recs)]
        print(f"\n=== {name} (n={len(recs)}) ===\nthreshold  answered  accuracy_when_answered")
        for thr in (0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9):
            kept = [o for o, c in zip(ok, pc) if c >= thr]
            acc = sum(kept) / len(kept) if kept else float("nan")
            print(f"{thr:>9.2f}  {len(kept)/len(recs):>8.2f}  {acc:>10.3f}")


if __name__ == "__main__":
    main()
