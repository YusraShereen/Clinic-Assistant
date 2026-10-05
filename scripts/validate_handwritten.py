#!/usr/bin/env python3
"""
Validate data/dev_handwritten.jsonl and data/final_test.jsonl (if present) and convert them to
token/BIO format (*.bio.jsonl) so they can be scored like the synthetic sets.

Input line format (you write these BY HAND, in your own natural phrasing):
  {"lang": "rur", "text": "kal subah 10 baje ...", "intent": "book_appointment",
   "entities": [{"type": "DATE", "text": "kal"}, ...]}

Rules:
  - entity "text" must match whole whitespace-separated tokens exactly as typed
  - entity "type" must be one of the types in data/label_map.json
  - intent must be one of the intents in data/label_map.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = ["dev_handwritten", "final_test"]   # final_test.jsonl is your rewritten set (use once, at the end)
LABELS = json.loads((ROOT / "data" / "label_map.json").read_text(encoding="utf-8"))
INTENTS = set(LABELS["intents"])
ENTITY_TYPES = {l.split("-", 1)[1] for l in LABELS["ner_labels"] if l != "O"}
MIN_PER_LANG = 20  # target for a defensible real test set


def to_bio(text, entities):
    tokens = text.split()
    tags = ["O"] * len(tokens)
    for e in entities:
        if e["type"] not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type {e['type']!r}")
        et = e["text"].split()
        for i in range(len(tokens) - len(et) + 1):
            if tokens[i:i + len(et)] == et and all(t == "O" for t in tags[i:i + len(et)]):
                tags[i] = "B-" + e["type"]
                for j in range(1, len(et)):
                    tags[i + j] = "I-" + e["type"]
                break
        else:
            raise ValueError(f"entity {e['text']!r} not found as whole tokens in: {text!r}")
    return tokens, tags


def process(name):
    src = ROOT / "data" / f"{name}.jsonl"
    dst = ROOT / "data" / f"{name}.bio.jsonl"
    if not src.exists():
        print(f"[skip] {name}: {src.name} not found")
        return True
    errors, out, lang_c, intent_c = [], [], Counter(), Counter()
    for n, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if r["lang"] not in {"en", "ur", "rur"}:
                raise ValueError(f"bad lang {r['lang']!r}")
            if r["intent"] not in INTENTS:
                raise ValueError(f"bad intent {r['intent']!r}")
            tokens, tags = to_bio(r["text"], r.get("entities", []))
            out.append({"id": f"{name[:4]}-{n:04d}", "lang": r["lang"], "intent": r["intent"],
                        "text": r["text"], "tokens": tokens, "ner_tags": tags})
            lang_c[r["lang"]] += 1
            intent_c[r["intent"]] += 1
        except Exception as ex:  # noqa: BLE001
            errors.append(f"line {n}: {ex}")
    if errors:
        print(f"ERRORS in {name}:")
        print("\n".join(errors))
        return False
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    print(f"OK {name}: {len(out)} examples -> {dst.relative_to(ROOT)}")
    print("  per language:", dict(lang_c))
    print("  per intent:  ", dict(intent_c))
    train_path = ROOT / "data/synthetic/train.jsonl"
    if train_path.exists():
        train_texts = {json.loads(l)["text"] for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        leak = sum(r["text"] in train_texts for r in out)
        print(f"  exact overlap with synthetic train: {leak}" + ("   <-- REMOVE THESE" if leak else ""))
    if name == "final_test":
        for lang in ("en", "ur", "rur"):
            if lang_c[lang] < MIN_PER_LANG:
                print(f"  [todo] {lang}: {lang_c[lang]}/{MIN_PER_LANG}")
    return True


def main():
    ok = all([process(n) for n in SETS])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
