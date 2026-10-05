#!/usr/bin/env python3
"""
Fine-tune a multilingual encoder (default: xlm-roberta-base) jointly for
intent classification + entity extraction.

Colab/Kaggle GPU:
  python scripts/train_nlu.py --out models/nlu
Offline smoke test (CPU, tiny random model, checks the code path only):
  python scripts/train_nlu.py --smoke-test --out /tmp/nlu_smoke
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nlu.core import JointNLU, collate, encode, evaluate, load_jsonl, load_label_map  # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_smoke(records, out_dir):
    """Char-level BERT tokenizer + tiny random BERT, so the pipeline runs without downloads."""
    from transformers import AutoModel, BertConfig, BertTokenizerFast
    chars = sorted({c for r in records for t in r["tokens"] for c in t})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + chars + ["##" + c for c in chars]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    try:  # transformers >= 5 takes the vocab directly
        tok = BertTokenizerFast(vocab={t: i for i, t in enumerate(vocab)}, do_lower_case=False)
    except TypeError:  # transformers 4.x
        tok = BertTokenizerFast(vocab_file=str(out_dir / "vocab.txt"), do_lower_case=False)
    assert len(tok) > 5 and tok.tokenize("abc") != ["[UNK]"], "smoke tokenizer failed to load vocab"
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=128)
    return tok, AutoModel.from_config(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="xlm-roberta-base")
    ap.add_argument("--out", default="models/nlu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = load_label_map(ROOT)
    train = load_jsonl(ROOT / "data/synthetic/train.jsonl")
    val = load_jsonl(ROOT / "data/synthetic/val.jsonl")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        train, val = train[:300], val[:100]
        args.epochs, args.bs = 1, 16
        tok, encoder = build_smoke(train + val, out)
    else:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.encoder)
        encoder = AutoModel.from_pretrained(args.encoder)

    model = JointNLU(encoder, len(lm["intents"]), len(lm["ner_labels"])).to(device)
    enc_train = [encode(r, tok, lm, args.max_len) for r in train]
    dl = DataLoader(enc_train, batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    heads = list(model.intent_head.parameters()) + list(model.ner_head.parameters())
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": heads, "lr": args.head_lr}], weight_decay=0.01)
    total = args.epochs * len(dl)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device={device} train={len(train)} val={len(val)} epochs={args.epochs} steps={total}")
    best, history = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for step, b in enumerate(dl, 1):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, _, _ = model(b["input_ids"], b["attention_mask"], b["intent"], b["ner_labels"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        res, _ = evaluate(model, tok, val, lm, device, args.max_len)
        o = res["overall"]
        score = (o["intent_macro_f1"] + o.get("entity_f1", 0.0)) / 2
        history.append({"epoch": ep, "train_loss": run / len(dl), "val": o})
        print(f"epoch {ep}: loss={run/len(dl):.4f} val intent_f1={o['intent_macro_f1']:.4f} "
              f"entity_f1={o.get('entity_f1', 0):.4f} ({time.time()-t0:.0f}s)")
        if score > best:
            best = score
            torch.save(model.state_dict(), out / "model.pt")
            tok.save_pretrained(out)
            model.encoder.config.save_pretrained(out)
            (out / "meta.json").write_text(json.dumps({
                "encoder": args.encoder, "smoke_test": args.smoke_test, "args": vars(args),
                "best_val_score": best, "history": history}, indent=2), encoding="utf-8")
    print(f"done. best val score={best:.4f}. saved to {out}")


if __name__ == "__main__":
    main()
