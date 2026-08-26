# PharmaLens — Deployment

Local Docker first, then Cloud Run. Every gotcha listed here is one that
produces a confusing error rather than a clear one.

---

## Files you need alongside this

- `Dockerfile`
- `app.py`
- `requirements.txt`
- `.dockerignore`
- `.gcloudignore`  ← **the one everybody forgets**

---

## 1. requirements.txt

Do **not** use `pip freeze` — your global environment has easyocr, shapely
and other unrelated packages tangled in. Write it by hand:

```
fastapi>=0.110
uvicorn[standard]>=0.27
pandas>=2.0,<2.3
numpy>=1.26,<2
pyarrow>=14
scikit-learn>=1.3
xgboost>=2.0
sentence-transformers>=2.5
chromadb>=0.4.22
anthropic>=0.25
```

`numpy<2` is pinned deliberately. numpy 2.x broke binary compatibility
with pandas 2.0.x — that is the `numpy.dtype size changed` error you hit
locally, and it will happen in the container too if left unpinned.

Note `torch` is absent. The Dockerfile installs it separately from the
CPU-only index; listing it here would let pip pull the 2.5GB CUDA build.

---

## 2. .dockerignore

Without this the build context includes your raw 40MB CSVs, the `.venv`,
and the git history — slow uploads and a fatter image.

```
data/raw/
data/processed/*.npy
.venv/
venv/
__pycache__/
*.pyc
.git/
.gitignore
reports/
*.md
notebooks/
```

`data/chroma` and the three needed Parquet files are **not** ignored —
they are copied into the image on purpose.

---

## 3. .gcloudignore — read this one

**`gcloud builds submit` ignores `.gitignore` by default.** Your
`data/` directory is gitignored, so without a `.gcloudignore` the build
context silently omits the Chroma index and Parquet files. The build
succeeds, the container starts, and then every request 500s because the
collection does not exist.

Create `.gcloudignore` explicitly, so gcloud stops falling back to
`.gitignore`:

```
.git/
.venv/
venv/
__pycache__/
*.pyc
data/raw/
data/processed/*.npy
reports/
notebooks/
```

---

## 4. Build and test locally

```bash
docker build -t pharmalens:local .
```

First build takes 5–10 minutes (torch and the model download). Check the
size:

```bash
docker images pharmalens:local
```

Expect ~1.5GB. If it is over 3GB, torch pulled the CUDA build — check
that the CPU index line ran before `pip install -r requirements.txt`.

Run it:

```bash
docker run --rm -p 8080:8080 pharmalens:local
```

Startup takes 15–30 seconds: the resolver builds four dictionaries over
253,969 products. Wait for `ready in Ns` in the logs.

Test:

```bash
curl localhost:8080/health
curl "localhost:8080/search?q=crocin"
curl -X POST localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"what are the side effects of Crocin"}'
```

Interactive docs: <http://localhost:8080/docs>

---

## 5. Deploy to Cloud Run

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

gcloud artifacts repositories create pharmalens \
  --repository-format=docker --location=asia-south1

gcloud builds submit \
  --tag asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/pharmalens/api:v1

gcloud run deploy pharmalens \
  --image asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/pharmalens/api:v1 \
  --region asia-south1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 120 \
  --concurrency 10 \
  --min-instances 0 \
  --max-instances 3 \
  --port 8080
```

`asia-south1` is Mumbai — closest region for Indian pharmaceutical data,
and a small detail worth being deliberate about.

### Why each flag

| Flag | Reason |
|---|---|
| `--memory 2Gi` | **Required.** The default 512Mi cannot hold torch + model + Parquet. It fails at startup with an unhelpful OOM. |
| `--cpu 2` | Cold start is CPU-bound on resolver construction. 1 CPU roughly doubles it. |
| `--timeout 120` | Cold start plus an LLM call can exceed the 60s default. |
| `--concurrency 10` | Each request embeds a query; too many in parallel starve the CPU. |
| `--min-instances 0` | Scale to zero. Free when idle, at the cost of cold starts. |
| `--max-instances 3` | A spending cap, not a capacity target. |

### With generation enabled

```bash
echo -n "sk-ant-..." | gcloud secrets create anthropic-key --data-file=-

gcloud run services update pharmalens \
  --region asia-south1 \
  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest
```

Never pass the key with `--set-env-vars` — it becomes visible in the
service description and in deployment logs.

---

## 6. Verify

```bash
URL=$(gcloud run services describe pharmalens \
  --region asia-south1 --format='value(status.url)')

curl $URL/health
curl $URL/stats
curl -X POST $URL/ask -H "Content-Type: application/json" \
  -d '{"question":"can I take Augmentin and Azithral together"}'
```

The last one should return `route: PARTIAL` — the router recognising that
the corpus holds no drug-drug interaction data.

---

## Troubleshooting

**Container fails to start, no useful error**
Almost always memory. Check `gcloud run services logs read pharmalens
--region asia-south1`. Raise to 4Gi to confirm, then tune down.

**`Collection pharmalens_compositions does not exist`**
`data/chroma` did not make it into the image. This is the `.gcloudignore`
problem in section 3. Verify with
`docker run --rm pharmalens:local ls -la /app/data/chroma`.

**Cold start over 60 seconds**
Set `--min-instances 1`. Costs a few dollars a month but keeps one
container warm. Reasonable if you are sending the link to anyone.

**`numpy.dtype size changed`**
An unpinned numpy resolved to 2.x. Confirm `numpy<2` in
`requirements.txt` and rebuild without cache.

**`WinError 1114` locally (not in the container)**
Windows-only DLL load ordering. `import torch` must be the first import
in `app.py`, before pandas or numpy. Already handled — do not reorder it.

---

## Cost

Scale-to-zero with light demo traffic lands within the Cloud Run free
tier — realistically ₹0–200/month. Artifact Registry storage for a 1.5GB
image is a few rupees. Set a budget alert anyway:

```bash
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT \
  --display-name="pharmalens" \
  --budget-amount=500INR
```

---

## What to put in the README

> Containerised FastAPI service on Cloud Run (Mumbai), 2Gi/2CPU,
> scale-to-zero. The Chroma index and embedding model are baked into the
> image for stateless deployment — 1.5GB image, 20–40s cold start. The
> alternative, fetching the index from Cloud Storage at startup, was
> rejected: at 2,857 documents it adds an external dependency and startup
> latency for no benefit.

Naming the tradeoff and the rejected alternative is the part worth
writing down.
