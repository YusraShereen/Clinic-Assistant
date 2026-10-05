# Bilingual (Urdu / Roman Urdu / English) Clinic Assistant

A front-desk assistant for a clinic: **intent classification + entity extraction** (fine-tuned XLM-R),
**multilingual retrieval over a FAQ knowledge base**, a small **dialogue orchestrator** (slot filling,
symptom-to-department routing without medical advice, human handoff), served with **FastAPI**, containerised
with **Docker**, and deployed on **k3s (Kubernetes) on AWS EC2**.

> **Status:** code complete and unit-tested (15 tests). Model training results, retrieval results with real
> embeddings, container build and AWS deployment are run by the author - the sections below are filled in
> only with numbers that were actually measured.

## Architecture
```
user text -> normalise + language detect -> NLU (XLM-R: intent + BIO entities) -> orchestrator
   book / cancel / reschedule -> slot filling -> "request noted" (front desk confirms; no fake bookings)
   clinic_info / fees_query   -> retrieval (multilingual-e5-small, cosine) -> FAQ answer in user's language
   symptom_inquiry            -> symptom -> department routing (+ emergency warning); no diagnosis
   talk_to_human / low confidence x2 -> handoff        out_of_scope -> polite refusal
```
Design choices: ~20 docs x 3 languages -> flat cosine search in numpy (exact; a vector DB pays off at thousands of
passages); answers are extractive by default (no hallucination, zero cost) with an optional OpenAI-compatible
generator; one container with in-process NLU + RAG for a cheap single-node deployment; Prometheus-style `/metrics`;
raw user text is not logged unless `LOG_TEXT=1`.

## Repo layout
```
app/            FastAPI service: nlu_service, rag, generator, orchestrator, messages, main
nlu/core.py     joint intent+NER model, encoding, prediction, evaluation, latency
scripts/        generate_data, validate_handwritten, train_nlu, eval_nlu, baseline_tfidf,
                pick_threshold, eval_rag, bench_api
data/           label_map.json, synthetic/, dev_handwritten.jsonl, rag_dev.jsonl (+ your final_test / rag_test)
kb/             faq.json (20 docs x 3 languages, FICTIONAL sample clinic), symptom_routing.json
deploy/k8s/     deployment + service (k3s)           tests/   15 tests (no GPU/model needed)
docs/           SPEC.md, FINAL_TEST_GUIDE.md, NEXT_STEPS.md
notebooks/      phase2_colab.ipynb (training on a free GPU)
```

## Quickstart
```bash
python scripts/generate_data.py && python scripts/validate_handwritten.py && python -m pytest -q
python scripts/train_nlu.py --out models/nlu         # GPU recommended (Colab notebook provided)
python scripts/eval_nlu.py --model-dir models/nlu
uvicorn app.main:app --port 8000                      # see docs/NEXT_STEPS.md
```

## Data honesty
- Training data is **synthetic** (templates + slot filling + prefix/suffix, casing and spelling noise), 2,160 examples,
  107 templates per language. v3 uses a **coverage-aware template split**: every slot type and error-prone pattern
  keeps >= 2 templates in training, while val/test use unseen templates.
- `synthetic/test` has 480 examples but only ~30 distinct templates per language: it is a diagnostic, not a headline.
- `dev_handwritten.jsonl` (72) was used to find errors and shape fixes, so scores on it are **optimistic**.
- `final_test.jsonl` (author-written, independent, frozen model) is evaluated **once** with `--final`; only those numbers are headline results.
- The KB (clinic details, fees, doctors) is **fictional sample content**. Urdu / Roman Urdu wording should be reviewed by a native speaker.
- Booking is **request-only**: the assistant records details for the front desk and never claims confirmation.
- No medical advice: symptom messages are routed to a department with an emergency warning for red-flag symptoms; the routing table is a demo and needs clinician review for real use.

## Known limitations
- **Retrieval confidence is hard to read.** On the dev set, multilingual-e5-small scored answerable-correct, answerable-wrong and *unanswerable* questions almost identically (mean top-1 cosine 0.869 / 0.859 / 0.868), so a score cutoff does not work. The *margin* between the top-2 documents does separate them (mean 0.031 vs 0.009); with a margin cutoff of 0.018 chosen on the dev set the bot answered 8/12 answerable questions correctly and refused 9/9 unanswerable ones (n=21, cutoff tuned on the same questions, so optimistic; 4/12 answerable questions are handed to the front desk). The honest number is the one on the independent `rag_test.jsonl` at the cutoff fixed beforehand (`eval_rag.py --min-margin 0.018 --final`). Every FAQ answer also ends with an offer to connect the front desk. Margins depend on how similar the FAQ documents are: re-tune after editing the KB.
- Roman Urdu retrieval is the weakest language (informal spelling).
- NLU errors cluster on phrasings absent from the synthetic templates (e.g. unusual loanwords such as "charges" in Urdu, payment-brand names); the model is over-confident, so the confidence threshold is a weak safeguard.
- Entity labels for role phrases ("skin doctor") are not fully consistent between generator templates.
- Synthetic training data; fictional KB; booking is request-only; no diagnosis. See "Data honesty".

## Results (to be filled with measured numbers only)
| Item | Value | n | Notes |
|---|---|---|---|
| NLU intent accuracy / macro-F1 (final test) | _TBD_ | _TBD_ | frozen model, evaluated once |
| NLU entity F1 (final test) | _TBD_ | _TBD_ | span-level micro F1 |
| TF-IDF intent baseline (final test) | _TBD_ | _TBD_ | |
| Retrieval recall@1 / @3 (e5 vs hash baseline) | _TBD_ | _TBD_ | author-written questions |
| API latency p50 / p95 on deployed instance | _TBD_ | _TBD_ | instance type: _TBD_ |
