# Your steps, in order

Legend: **[you]** = you run it, **[send me]** = paste the output so I can check it.
Rule for the whole project: claim on your resume only what you measured and ran.

## A. Write your test sets (can start now)  [you]
1. `data/final_test.jsonl` - follow `docs/FINAL_TEST_GUIDE.md` (>= 24 per language).
2. `data/rag_test.jsonl` - at least 25 retrieval questions (format in the same guide).
3. **Keep both OUT of the zip you upload to Colab** until step C. Do not read model errors while writing them.

## B. Retrain the NLU on v3 data (Colab, ~20 min)  [you]
1. Open Colab, T4 GPU. Delete the old copy first: `!rm -rf /content/clinic-assistant*` then upload the NEW zip and run the notebook cells (unchanged).
2. Check the generate cell prints `train n=2160` and the training cell prints `train=2160`. If not, you are on the old files.
3. After the eval cell, also run: `!python scripts/pick_threshold.py --model-dir models/nlu` and `!python scripts/baseline_tfidf.py`.
4. Save `models/nlu` to Google Drive (about 1.1 GB) and download `results/metrics.json`, `results/errors_synthetic_test.jsonl`, `results/errors_dev_handwritten.jsonl`.
5. **[send me]** metrics.json, the pick_threshold table, the baseline numbers. Expect dev numbers to be optimistic (see README "Data honesty").
   Pick `INTENT_THRESHOLD` from the *synthetic_val* table: the lowest threshold where accuracy-when-answered is high while "answered" stays reasonable (a common choice is around 0.5-0.7).

## C. Freeze, then evaluate the final test set ONCE  [you]
1. Do not change data, templates, model or labels after this point.
2. Record the model fingerprint: `sha256sum models/nlu/model.pt > models/nlu/SHA256.txt` and commit/tag the code: `git tag nlu-frozen`.
3. Copy `final_test.jsonl` into `data/`, then: `python scripts/validate_handwritten.py` (must say overlap = 0).
4. Run once: `python scripts/eval_nlu.py --model-dir models/nlu --final` and `python scripts/baseline_tfidf.py --final`.
5. Copy the `final_test` numbers into README "Results" with n and per-language counts. If lower than dev: report it anyway.
6. **[send me]** results/metrics.json - I will write the honest README results and resume bullets from it.

## D. Run the service locally (laptop)  [you]
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch                                          # CPU is fine
pip install -r requirements-serve.txt -r requirements-dev.txt
# put the trained model at ./models/nlu  (model.pt, config.json, tokenizer files)
INTENT_THRESHOLD=0.6 EMBEDDER=e5 NLU_MODEL_DIR=models/nlu uvicorn app.main:app --port 8000
```
(First start downloads the embedding model, ~470 MB, and loads the NLU model; it can take a minute.)
Try it:
```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"text":"mujhe kal 5 baje dr sara malik se milna hai","session_id":"demo"}'
curl -s -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"text":"کلینک کے اوقات کیا ہیں","session_id":"demo2"}'
curl -s localhost:8000/metrics
```
Retrieval evaluation (real embeddings vs the hash baseline):
```bash
python scripts/eval_rag.py --embedder hash
python scripts/eval_rag.py --embedder e5
```
Use the "mean top-1 score: correct vs wrong" line to choose `RETRIEVAL_MIN_SCORE` (a value between the two means). Iterate on `kb/faq.json` using `rag_dev.jsonl` only; run `--final` (rag_test) once at the end.
Review the Urdu / Roman Urdu wording in `kb/faq.json` and `app/messages.py` (I wrote them; you are the native speaker) and replace sample clinic details with your own fictional ones.
Local latency: `python scripts/bench_api.py --url http://localhost:8000 --n 200`.
**[send me]** eval_rag output (hash and e5), bench output, anything that errors.

## E. Docker (laptop; skip to F if Docker is not installed)  [you]
```bash
docker compose up --build        # first build downloads CPU torch + the embedding model (several GB, ~10 min)
curl -s localhost:8000/health
```
`./models/nlu` is mounted read-only into the container. The Dockerfile was not build-tested by me (no Docker in my environment): send me the error if the build fails.

## F. GitHub + CI  [you]
1. Create a **public** repo, `git init`, commit everything (models/ and results/ are already ignored), push to `main`.
2. Actions tab: the `test` job must go green (it runs the data generator + 15 tests with no GPU/model). The `image` job builds and pushes `ghcr.io/<you>/clinic-assistant:latest` (set the package to *public* in GitHub -> Packages so the server can pull it without login).
3. **[send me]** the workflow log if anything fails.

## G. Deploy on AWS with k3s (about 2-3 h; cost ~ $1-3 if you terminate the same day)  [you]
Before starting: AWS Budgets alert at $5. Never leave the instance running overnight.
1. EC2 -> Launch: Ubuntu Server 24.04, **t3.large** (8 GiB RAM), 30 GiB gp3 disk, create + download a key pair. Security group: SSH (22) from *My IP*, custom TCP **30080** from *My IP*. Do not open it to the world.
2. SSH in: `ssh -i key.pem ubuntu@<public-ip>`; install k3s: `curl -sfL https://get.k3s.io | sh -` then `sudo k3s kubectl get nodes` (Ready).
3. Upload the model from your laptop: `scp -i key.pem -r models/nlu ubuntu@<ip>:/tmp/nlu`, then on the server: `sudo mkdir -p /opt/clinic/models && sudo mv /tmp/nlu /opt/clinic/models/nlu`.
4. Get the manifests onto the server: `git clone https://github.com/<you>/<repo>.git && cd <repo>` (or scp `deploy/k8s`).
5. Edit `deploy/k8s/deployment.yaml`: set `image: ghcr.io/<you>/clinic-assistant:latest`, set `INTENT_THRESHOLD` / `RETRIEVAL_MIN_SCORE` to your chosen values.
   (No registry? Build locally, `docker save clinic-assistant:local | gzip > img.tgz`, scp it, then `gunzip -c img.tgz | sudo k3s ctr images import -` and keep `image: clinic-assistant:local`.)
6. `sudo k3s kubectl apply -f deploy/k8s/` then `sudo k3s kubectl get pods -w` (wait for READY 1/1; if it restarts, `sudo k3s kubectl logs deploy/clinic-assistant` and `describe pod`; memory problems show as OOMKilled -> use a larger instance for the test).
7. From your laptop: `curl -s http://<ip>:30080/health`, then run `python scripts/bench_api.py --url http://<ip>:30080 --n 200` and save `results/api_bench.json`. On the server: `sudo k3s kubectl top pod` for memory.
8. Record a 2-minute screen capture: three chats (English, Urdu, Roman Urdu), a booking request with slot filling, an FAQ answer, a symptom message (shows the no-diagnosis routing), a handoff, then `/metrics`.
9. **Tear down the same session:** EC2 -> terminate the instance; check Volumes, Elastic IPs and Snapshots are empty; look at Billing again the next day.
10. **[send me]** bench json, `kubectl top pod` output, and any errors.

## H. Finish  [you + me]
- README "Results": paste measured numbers only (NLU final test, baseline, RAG recall@k, API latency on the deployed instance) with n.
- Resume bullets: I will write them from your result files.
- Say in the README that data is synthetic, the KB is fictional sample content, and booking is request-only (front desk confirms).
