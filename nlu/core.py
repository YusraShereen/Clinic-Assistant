"""Shared code: joint intent + NER model, data encoding, prediction, evaluation, latency."""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

IGNORE = -100


def _spans(tags):
    """BIO tags -> set of (type, start, end). An I- tag without a matching open span starts one
    (same lenient behaviour as seqeval's default)."""
    spans, cur = set(), None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[0] != t[2:])):
            if cur is not None:
                spans.add((cur[0], cur[1], i))
                cur = None
            if t != "O":
                cur = (t[2:], i)
    return spans


_MERGE_TYPES = {"DOCTOR", "DEPARTMENT", "DATE", "TIME"}


def merge_adjacent_tags(tags):
    """Diagnostic: turn B-X right after an X span into I-X (X multi-word type). Mirrors serving-time decoding."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("B-") and t[2:] in _MERGE_TYPES and prev != "O" and prev[2:] == t[2:]:
            t = "I-" + t[2:]
        out.append(t)
        prev = t
    return out


def entity_prf(gold_tags, pred_tags):
    """Micro-averaged span-level precision / recall / F1 over a list of tag sequences."""
    tp = fp = fn = 0
    for g, p in zip(gold_tags, pred_tags):
        gs, ps = _spans(g), _spans(p)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_label_map(root):
    return json.loads((Path(root) / "data" / "label_map.json").read_text(encoding="utf-8"))


class JointNLU(nn.Module):
    """Shared encoder; [CLS] -> intent head, every token -> NER (BIO) head."""

    def __init__(self, encoder, n_intents, n_ner, dropout=0.1, ner_weight=1.0):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.intent_head = nn.Linear(h, n_intents)
        self.ner_head = nn.Linear(h, n_ner)
        self.ner_weight = ner_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    def forward(self, input_ids, attention_mask, intent=None, ner_labels=None):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hs = self.drop(hs)
        intent_logits = self.intent_head(hs[:, 0])
        ner_logits = self.ner_head(hs)
        loss = None
        if intent is not None and ner_labels is not None:
            loss = self.ce(intent_logits, intent) + self.ner_weight * self.ce(
                ner_logits.reshape(-1, ner_logits.size(-1)), ner_labels.reshape(-1))
        return loss, intent_logits, ner_logits


def encode(rec, tok, lm, max_len=64):
    enc = tok(rec["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
    word_ids = enc.word_ids()
    labels, prev = [], None
    for w in word_ids:
        if w is None or w == prev:
            labels.append(IGNORE)          # special tokens and non-first sub-tokens
        else:
            labels.append(lm["ner2id"][rec["ner_tags"][w]])
        prev = w
    return {"input_ids": enc["input_ids"], "ner_labels": labels,
            "intent": lm["intent2id"][rec["intent"]], "word_ids": word_ids}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    ner = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        att[i, :L] = 1
        ner[i, :L] = torch.tensor(b["ner_labels"])
    return {"input_ids": ids, "attention_mask": att, "ner_labels": ner,
            "intent": torch.tensor([b["intent"] for b in batch])}


@torch.no_grad()
def predict(model, tok, token_lists, lm, device, max_len=64, bs=64, return_conf=False):
    """-> (intent_names, ner_tag_lists aligned to the input words[, intent_confidences])."""
    model.eval()
    id2intent = {v: k for k, v in lm["intent2id"].items()}
    id2ner = {v: k for k, v in lm["ner2id"].items()}
    intents, tags_out, confs = [], [], []
    for s in range(0, len(token_lists), bs):
        chunk = token_lists[s:s + bs]
        encs = []
        for toks in chunk:
            enc = tok(toks, is_split_into_words=True, truncation=True, max_length=max_len)
            encs.append({"input_ids": enc["input_ids"], "ner_labels": [IGNORE] * len(enc["input_ids"]),
                         "intent": 0, "word_ids": enc.word_ids()})
        batch = collate(encs, tok.pad_token_id)
        _, il, nl = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        pr = il.softmax(-1)
        ip = pr.argmax(-1).cpu().tolist()
        cf = pr.max(-1).values.cpu().tolist()
        npred = nl.argmax(-1).cpu().tolist()
        for toks, e, i, np_, c in zip(chunk, encs, ip, npred, cf):
            tags = ["O"] * len(toks)
            prev = None
            for pos, w in enumerate(e["word_ids"]):
                if w is not None and w != prev:
                    tags[w] = id2ner[np_[pos]]
                prev = w
            intents.append(id2intent[i])
            tags_out.append(tags)
            confs.append(c)
    if return_conf:
        return intents, tags_out, confs
    return intents, tags_out


def _spans(tags):
    """BIO tags -> set of (type, start, end). Lenient: a stray I- starts a new span."""
    spans, typ, start = set(), None, 0
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if typ is not None:
                spans.add((typ, start, i))
            typ = None
            if t != "O":
                typ, start = t[2:], i
    return spans


_MERGE_TYPES = {"DOCTOR", "DEPARTMENT", "DATE", "TIME"}


def merge_adjacent_tags(tags):
    """Diagnostic: turn B-X right after an X span into I-X (X multi-word type). Mirrors serving-time decoding."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("B-") and t[2:] in _MERGE_TYPES and prev != "O" and prev[2:] == t[2:]:
            t = "I-" + t[2:]
        out.append(t)
        prev = t
    return out


def entity_prf(gold_tags, pred_tags):
    """Micro span-level precision / recall / F1 (exact type + boundary match)."""
    tp = fp = fn = 0
    for g, p in zip(gold_tags, pred_tags):
        gs, ps = _spans(g), _spans(p)
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def _metrics(gold_i, pred_i, gold_t, pred_t, intent_names):
    from sklearn.metrics import accuracy_score, f1_score
    out = {"n": len(gold_i),
           "intent_acc": float(accuracy_score(gold_i, pred_i)),
           "intent_macro_f1": float(f1_score(gold_i, pred_i, labels=intent_names,
                                             average="macro", zero_division=0))}
    has_ent = any(t != "O" for tags in gold_t for t in tags)
    if has_ent:
        p, r, f = entity_prf(gold_t, pred_t)
        out["entity_f1"], out["entity_precision"], out["entity_recall"] = float(f), float(p), float(r)
        out["entity_f1_merged"] = float(entity_prf(gold_t, [merge_adjacent_tags(t) for t in pred_t])[2])  # diagnostic
    return out


def evaluate(model, tok, records, lm, device, max_len=64):
    from sklearn.metrics import f1_score
    pi, pt = predict(model, tok, [r["tokens"] for r in records], lm, device, max_len)
    gi = [r["intent"] for r in records]
    gt = [r["ner_tags"] for r in records]
    names = lm["intents"]
    res = {"overall": _metrics(gi, pi, gt, pt, names), "by_lang": {}, "by_intent_f1": {}}
    for lang in sorted({r["lang"] for r in records}):
        idx = [k for k, r in enumerate(records) if r["lang"] == lang]
        res["by_lang"][lang] = _metrics([gi[k] for k in idx], [pi[k] for k in idx],
                                        [gt[k] for k in idx], [pt[k] for k in idx], names)
    per = f1_score(gi, pi, labels=names, average=None, zero_division=0)
    sup = defaultdict(int)
    for g in gi:
        sup[g] += 1
    res["by_intent_f1"] = {n: {"f1": float(f), "support": sup[n]} for n, f in zip(names, per)}
    errors = []
    for k, r in enumerate(records):
        if pi[k] != gi[k] or pt[k] != gt[k]:
            errors.append({"id": r.get("id"), "lang": r["lang"], "text": r["text"],
                           "gold_intent": gi[k], "pred_intent": pi[k],
                           "gold_tags": gt[k], "pred_tags": pt[k]})
    return res, errors


@torch.no_grad()
def latency(model, tok, token_lists, lm, device, n=200, warmup=10):
    """Single-utterance end-to-end latency (tokenise + forward) in ms."""
    times = []
    lists = (token_lists * (n // max(1, len(token_lists)) + 1))[: n + warmup]
    for k, toks in enumerate(lists):
        t0 = time.perf_counter()
        predict(model, tok, [toks], lm, device, bs=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if k >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return {"device": str(device), "n": len(times),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "mean_ms": float(np.mean(times))}
