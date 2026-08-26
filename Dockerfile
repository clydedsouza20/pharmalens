# syntax=docker/dockerfile:1
#
# PharmaLens - container image
#
# THE SIZE PROBLEM AND WHY THIS DOCKERFILE LOOKS LIKE IT DOES
# -----------------------------------------------------------
# A naive `pip install torch sentence-transformers` pulls the CUDA build:
# roughly 2.5GB of GPU libraries that will never execute on Cloud Run.
# Installing from the CPU-only index instead cuts that to about 200MB.
#
# The embedding model (~90MB) is baked in at build time. Downloading it at
# startup would add 10-20s to every cold start and make the container
# depend on Hugging Face being reachable.
#
# The Chroma index and Parquet files are copied in too. 2,857 documents is
# small, and shipping them makes the container stateless -- no bucket, no
# IAM, no startup fetch.
#
# Final image: roughly 1.5GB. Cold start 20-40s. Acceptable for a demo,
# and the honest tradeoff to state rather than hide.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- dependencies -------------------------------------------------------
# torch from the CPU index BEFORE anything else, so sentence-transformers
# does not drag in the CUDA build as a transitive dependency.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# --- bake the embedding model in ---------------------------------------
# Done as its own layer so it is cached across code changes.
# HF_HUB_OFFLINE=1 is set globally above so the RUNTIME never phones home.
# It has to be turned off for this one layer, which is the build-time
# download. Per-RUN env vars do not persist to later layers.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2').save('/app/models/all-MiniLM-L6-v2')"
# Load from the saved directory by PATH, not by model name. `.save()`
# writes a plain model dir; SENTENCE_TRANSFORMERS_HOME expects a HF cache
# layout, so the bare name would not resolve and the loader would try to
# fetch — which HF_HUB_OFFLINE=1 correctly blocks.
ENV EMBEDDING_MODEL_PATH=/app/models/all-MiniLM-L6-v2

# --- application + data -------------------------------------------------
COPY stage3c_retrieval.py stage4_answer.py app.py ./
COPY data/processed/documents.parquet data/processed/products.parquet \
     data/processed/price_tier_model.json ./data/processed/
COPY data/chroma ./data/chroma

# Non-root, because containers running as root in production is a habit
# worth not forming.
RUN useradd --create-home --uid 1000 pharmalens \
    && chown -R pharmalens:pharmalens /app
USER pharmalens

EXPOSE 8080
ENV PORT=8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

# One worker: the model and indexes are held in process memory, so a
# second worker would double a ~1.2GB footprint for no throughput gain at
# this scale. Scale with Cloud Run instances, not with workers.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
