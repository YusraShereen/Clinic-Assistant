# Bilingual (Urdu / Roman Urdu / English) Clinic Assistant

A front-desk assistant for a clinic that understands messages written in **Urdu script, Roman Urdu and English**:

- **NLU:** XLM-R fine-tuned jointly for **intent classification** (8 intents) and **entity extraction** (doctor, department, date, time, symptom).
- **Retrieval:** multilingual FAQ answers (`multilingual-e5-small`, trilingual knowledge base) with margin-based abstention.
- **Dialogue:** slot filling for booking / cancelling / rescheduling, symptom-to-department routing (no diagnosis), human handoff.
- **Serving:** FastAPI, Docker, GitHub Actions CI, Kubernetes (k3s) manifests for a single AWS EC2 node.

> **Scope and honesty.** Training data is synthetic. The knowledge base is a fictional sample clinic. Booking is *request-only* (the assistant records details; the front desk confirms). It never gives medical advice or a diagnosis. Headline numbers come only from an independent test set that was evaluated once (see [Evaluation protocol](#evaluation-protocol)).

## Results

### Headline: independent hand-written test set (evaluated once, model frozen)

`data/final_test.jsonl`: **n = 72** (24 per language, 9 per intent), written by the author in her own phrasing.

| Metric | Value |
|---|---|
| Intent accuracy | **97.2 %** (70 / 72); 95 % Wilson interval 90.4 - 99.2 % |
| Intent macro-F1 | 0.972 |
| Entity span F1 (micro) | **0.925** (precision 0.907, recall 0.944) |
| TF-IDF intent baseline (same set) | accuracy **88.9 %** (64 / 72); 95 % interval 79.6 - 94.3 %; macro-F1 0.892 |

| Language | n | Intent accuracy | Entity F1 |
|---|---|---|---|
| English | 24 | 1.000 | 0.880 |
| Roman Urdu | 24 | 1.000 | 0.957 |
| Urdu | 24 | 0.917 | 0.941 |

XLM-R is ahead of the TF-IDF baseline by 6 of 72 sentences, but the two 95 % intervals overlap, so on this set the gap is suggestive rather than conclusive. On the larger synthetic test the gap is clear (0.981 vs 0.833), but that set holds only ~30 distinct templates per language. The baseline predicts intents only; there is no entity baseline.

Read with care: 72 examples is small (the interval above is wide; per-intent and per-language cells have 8-9 examples each, so they are indicative only). The author had seen the *categories* of earlier error analysis before writing this set.

### Retrieval (independent test set)

`data/rag_test.jsonl`: **60 answerable questions** (20 per language), written by the author; `multilingual-e5-small`, document-level.

| Metric | Value |
|---|---|
| recall@1 | **90.0 %** (54 / 60); 95 % interval 79.9 - 95.3 % |
| recall@3 | **98.3 %** (59 / 60); 95 % interval 91.1 - 99.7 % |
| MRR | 0.940 |
| Retrieval time (laptop CPU) | p50 12.8 ms, p95 45.6 ms |
| Lexical hash baseline, same set | _pending: `python scripts/eval_rag.py --embedder hash --final`_ |

| Language | n | recall@1 | recall@3 |
|---|---|---|---|
| English | 20 | 1.00 | 1.00 |
| Roman Urdu | 20 | 0.80 | 0.95 |
| Urdu | 20 | 0.90 | 1.00 |

**Margin-based abstention at the cutoff fixed on dev beforehand (0.018), on the 60 answerable questions:**
41 answered correctly, **0 answered wrongly**, 19 (31.7 %) sent to the front desk. The cutoff refused all 6 questions whose top document was wrong, and also 13 whose top document was right: it removes wrong answers at the price of handing over about a third of answerable questions (with 41 answered, the 95 % upper bound on the wrong-answer rate is about 9 %). **Refusal of genuinely unanswerable questions was not evaluated independently**: `rag_test` contains no such questions, so that part rests on the dev set only (see Known limitations).

### Latency

| Setting | p50 | p95 | Notes |
|---|---|---|---|
| NLU model only, single utterance | 20.2 ms | 29.2 ms | author's laptop CPU, 200 requests (`scripts/eval_nlu.py`) |
| Full API, server-side | 57.6 ms | 118.1 ms | local laptop CPU, 100 requests (`scripts/bench_api.py`) |
| Full API, client round trip | 77.5 ms | 131.6 ms | same run, via `127.0.0.1` |

### Development diagnostics (optimistic: not headline numbers)

| Set | n | Intent acc. | Entity F1 | Why optimistic |
|---|---|---|---|---|
| Synthetic test | 480 | 0.981 | 0.985 | only ~30 distinct held-out templates per language |
| Dev (hand-written) | 72 | 0.917 | 0.964 | its errors guided data fixes |
| Retrieval dev (e5) | 12 + 9 unanswerable | recall@1 0.75, recall@3 0.92 | - | written by the developer; small |

## Architecture

```
user text -> normalise (strip punctuation, keep 4:30) -> language detect (ur / rur / en)
          -> NLU (XLM-R: intent + BIO entities) -> orchestrator
   book / cancel / reschedule -> slot filling -> "request noted"  (front desk confirms)
   clinic_info / fees_query   -> retrieval (e5, cosine) -> FAQ answer in the user's language
                                  abstain when top-2 margin < cutoff -> offer front desk
   symptom_inquiry            -> symptom -> department routing (+ emergency warning); no diagnosis
   talk_to_human / 2 unclear messages in a row -> handoff        out_of_scope -> polite refusal
```

Design choices
- ~20 documents x 3 languages: **flat cosine search in numpy** is exact and fast; a vector database pays off at thousands of passages.
- Each FAQ is indexed as separate question and answer passages per language; the best score per document wins, so a Roman Urdu question can match via the Urdu or English passage.
- Answers are **extractive** by default (no hallucination, zero cost). An optional OpenAI-compatible generator (`GENERATOR=openai`) can rephrase the retrieved answer and falls back to extractive on any failure.
- Multi-word entities are merged at decode time when the model emits `B-DATE B-DATE` for "next week" (the evaluation reports both raw and merged entity F1; they are identical on the final model).
- One container with in-process NLU + retrieval keeps a single cheap node viable. Prometheus-style `/metrics`; raw user text is not logged unless `LOG_TEXT=1`.

## Repository layout

```
app/            FastAPI service: nlu_service, rag, generator, orchestrator, messages, text, main
nlu/core.py     joint intent+NER model, encoding, prediction, evaluation, latency
scripts/        generate_data, validate_handwritten, validate_rag_test, train_nlu, eval_nlu, baseline_tfidf,
                pick_threshold, eval_rag, bench_api
data/           label_map.json, synthetic/, dev_handwritten.jsonl, final_test.jsonl, rag_dev.jsonl, rag_test.jsonl
kb/             faq.json (20 documents x 3 languages, FICTIONAL), symptom_routing.json (demo table)
deploy/k8s/     Deployment + Service for k3s            tests/    18 tests (no GPU or model needed)
docs/           SPEC.md (labels), FINAL_TEST_GUIDE.md, NEXT_STEPS.md
notebooks/      phase2_colab.ipynb (training on a free GPU)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch && pip install -r requirements-serve.txt -r requirements-dev.txt
python -m pytest -q                                      # 18 tests
```

Run the service (the trained NLU model is not in git; place it at `models/nlu`):

```bash
INTENT_THRESHOLD=0.7 RETRIEVAL_MIN_MARGIN=0.018 EMBEDDER=e5 NLU_MODEL_DIR=models/nlu \
  uvicorn app.main:app --port 8000
# Windows cmd: use `set NAME=value` on separate lines first
curl -s -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"text":"mujhe kal 5 baje dr sara malik se milna hai","session_id":"demo"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /chat` `{text, session_id?}` | reply + intent, confidence, entities, action, source, latency |
| `POST /nlu` | NLU output only (debugging) |
| `GET /health`, `GET /metrics` | liveness; Prometheus-style counters and latency histogram |

Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `NLU_MODEL_DIR` | `models/nlu` | trained NLU model directory |
| `INTENT_THRESHOLD` | `0.7` | below this confidence the bot asks to rephrase (chosen on validation data) |
| `EMBEDDER` / `EMBED_MODEL` | `e5` / `intfloat/multilingual-e5-small` | `hash` = offline baseline |
| `RETRIEVAL_MIN_MARGIN` | `0.0` (compose/k8s: `0.018`) | refuse when top-2 score gap is smaller (chosen on dev) |
| `RETRIEVAL_MIN_SCORE` | `0.0` | raw-score cutoff (does not separate answerable questions; see limitations) |
| `FAQ_FOLLOWUP` | `1` | append the "connect me to the front desk" offer to FAQ answers |
| `CLINIC_PHONE` | sample number | used in handoff messages |
| `GENERATOR`, `GENERATOR_URL`, `GENERATOR_KEY`, `GENERATOR_MODEL` | `extractive` | optional LLM rephrasing |
| `KB_PATH`, `SYMPTOM_ROUTING`, `LABEL_MAP`, `DEVICE`, `LOG_TEXT` | see code | paths / device / logging |

Docker and Kubernetes
```bash
docker compose up --build          # mounts ./models/nlu read-only; first build downloads CPU torch + the embedding model
```
`deploy/k8s/` holds the Deployment and NodePort Service (single node, k3s; model mounted from a hostPath). GitHub Actions runs the tests and builds/pushes the image to GHCR.

## Reproducing the results

```bash
python scripts/generate_data.py                       # synthetic data (seed 42)
python scripts/validate_handwritten.py                # converts dev/final sets, checks overlaps
python scripts/train_nlu.py --out models/nlu --epochs 12      # ~12 s/epoch on a Colab T4
python scripts/eval_nlu.py --model-dir models/nlu             # synthetic test + dev
python scripts/pick_threshold.py --model-dir models/nlu       # confidence threshold, validation only
python scripts/baseline_tfidf.py                              # TF-IDF + logistic regression baseline
python scripts/eval_nlu.py --model-dir models/nlu --final     # ONCE, after freezing the model
```
Model: `xlm-roberta-base` + two linear heads (intent on `[CLS]`, BIO tags per first sub-token), joint cross-entropy, AdamW (lr 3e-5 encoder, 3e-4 heads), batch 32, max length 64, warm-up 10 %. Best epoch (9 of 12) chosen on **validation** score (validation intent F1 0.976, entity F1 0.970). The model's SHA-256 was recorded when it was frozen (`models/nlu/SHA256.txt`, not committed).

## Evaluation protocol

1. **Synthetic data** is generated from templates with slot filling, prefix/suffix variation, casing and spelling noise, and hard negatives (payment words, day-parts, body parts, family words, non-clinic questions). The split is **by template**, with a **coverage guard**: every slot type and error-prone pattern keeps at least 2 templates in training (an earlier split silently removed whole patterns from training; the guard fixed that).
2. `synthetic/test` has 480 examples but only ~30 distinct templates per language, so it is a diagnostic, not a headline.
3. `dev_handwritten.jsonl` (72) was used to find errors and shape data fixes; scores on it are optimistic.
4. `final_test.jsonl` was written independently by the author, committed **before** evaluation, and evaluated **once** with the explicit `--final` flag; the model was frozen first. The same protocol applies to `rag_test.jsonl` (cutoffs chosen on dev, then fixed).

## Known limitations

- **Small final test set** (n = 72): intervals are wide and per-cell numbers are indicative only. Entity F1 has no interval computed.
- **Synthetic training data.** Errors concentrate on phrasing and vocabulary absent from the templates (e.g. the loanword "charges" in Urdu, payment-brand names). The model is over-confident, so the confidence threshold is a weak safeguard; the two-unclear-messages handoff is a backstop.
- **Retrieval cannot say "not in the knowledge base" by score.** On the dev set, e5 scored answerable-correct, answerable-wrong and *unanswerable* questions almost identically (mean top-1 cosine 0.869 / 0.859 / 0.868). The *margin* between the top-2 documents separates them better (mean 0.031 vs 0.009): at a cutoff of 0.018 chosen on dev, 8/12 answerable questions were answered correctly and 9/9 unanswerable ones refused (n = 21, tuned on the same questions, so optimistic; 4/12 answerable questions are handed to the front desk). Every FAQ answer also ends with an offer to connect the front desk. On the independent answerable questions the same cutoff (fixed beforehand) answered 41/60 correctly, answered none wrongly and handed 19/60 (32 %) to the front desk; refusal of genuinely unanswerable questions was not evaluated independently (no such questions in `rag_test`). Margins depend on how similar the documents are: re-tune after editing the KB.
- Roman Urdu retrieval is the weakest language (informal spelling); only 4 dev questions per language.
- Entity labels for role phrases ("skin doctor") are not fully consistent between generator templates.
- The KB, fees, doctors and phone number are fictional; Urdu / Roman Urdu wording should be reviewed by a native speaker. The symptom routing table is a demo and needs clinician review for any real use.
- Sessions are in memory (single process); there is no persistence, authentication or real booking back-end.

## Safety and privacy notes
- No diagnosis or treatment advice; red-flag symptoms (e.g. chest pain) add an emergency warning; the clinic does not handle emergencies.
- Raw user text is not logged by default; logs contain language, intent, confidence, handoff flag and latency.
- Do not expose the NodePort publicly; restrict the security group to your own IP.
