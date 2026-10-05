#!/usr/bin/env python3
"""Latency benchmark against a RUNNING service (run it on the deployed instance's public URL or locally).

  python scripts/bench_api.py --url http://127.0.0.1:8000 --n 200
Reports client round-trip and server-side latency, p50/p95/mean, and error count.
Uses sentences from the dev set; sessions are unique so no dialogue state is shared.
"""
import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def post(url, payload, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_ready(base, timeout=180):
    """Poll /health until the model has loaded (startup can take a minute on a small CPU)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=3) as r:
                if json.loads(r.read().decode("utf-8")).get("status") == "ok":
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")   # not "localhost": on Windows it adds ~2 s (IPv6 fallback)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    if not wait_ready(args.url):
        raise SystemExit(f"service at {args.url} did not become ready (check the container/pod logs)")
    src = ROOT / "data" / "dev_handwritten.jsonl"
    texts = [json.loads(l)["text"] for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = (texts * (args.n // len(texts) + 2))[: args.n + args.warmup]
    rtt, srv, errors = [], [], 0
    for i, t in enumerate(texts):
        t0 = time.perf_counter()
        try:
            out = post(args.url.rstrip("/") + "/chat", {"text": t, "session_id": uuid.uuid4().hex})
        except Exception:  # noqa: BLE001
            errors += 1
            continue
        dt = (time.perf_counter() - t0) * 1000
        if i >= args.warmup:
            rtt.append(dt)
            srv.append(out["latency_ms"])
    if not rtt:
        raise SystemExit(f"all {errors} requests failed")
    res = {"url": args.url, "n": len(rtt), "errors": errors,
           "client_roundtrip_ms": {"p50": float(np.percentile(rtt, 50)), "p95": float(np.percentile(rtt, 95)),
                                   "mean": float(np.mean(rtt))},
           "server_ms": {"p50": float(np.percentile(srv, 50)), "p95": float(np.percentile(srv, 95)),
                         "mean": float(np.mean(srv))}}
    print(json.dumps(res, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "api_bench.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
