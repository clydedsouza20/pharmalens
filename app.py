#!/usr/bin/env python3
"""
PharmaLens - Stage 5: FastAPI service

WHAT THIS EXPOSES
-----------------
    GET  /health                  liveness + what is loaded
    GET  /product/{product_id}    one product, with its composition
    GET  /composition/{key}       one composition document + metadata
    POST /predict                 price tier from product features
    POST /ask                     the full pipeline: resolve -> filter ->
                                  search -> route -> (optionally) generate
    GET  /stats                   corpus and index statistics

DESIGN NOTES THAT MATTER
------------------------
1. EVERYTHING LOADS ONCE, AT STARTUP.
   The embedding model, the Chroma client, the resolver's four
   dictionaries and the XGBoost model are all built during the lifespan
   startup hook and held in module state. Building the resolver takes a
   few seconds over 253,969 products -- doing it per request would make
   every call unusable.

2. THE ENCODERS ARE REBUILT AT STARTUP.
   stage2_model.py saved the trained model but NOT the target-encoding
   mappings, so a new product could not be encoded at prediction time.
   They are recomputed here from products.parquet. That is not a
   workaround -- at serving time you WANT encodings fitted on all
   available data, not on the training split. The out-of-fold discipline
   in stage 2 existed to get an honest test score; it has no role here.

3. /ask RETURNS THE ROUTE AND THE RETRIEVED DOCUMENTS.
   Not just an answer string. A caller can see which drug was resolved,
   how, with what confidence, which route fired, and exactly which
   documents grounded the response. Retrieval that cannot be inspected
   cannot be trusted.

4. GENERATION IS OPTIONAL.
   Without ANTHROPIC_API_KEY the service still runs and returns routing
   decisions plus retrieved documents. Only the free-text summary is
   absent. The safety guarantees live in the router, not the model, so
   they hold either way.

RUNNING LOCALLY
---------------
    pip install fastapi "uvicorn[standard]"
    uvicorn app:app --reload --port 8080

    http://localhost:8080/docs      interactive API docs
"""

import torch  # MUST be first on Windows (WinError 1114 on c10.dll)

import os
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from stage3c_retrieval import (DrugResolver, PharmaLensRetriever,
                               MODEL_NAME, COLLECTION)
from stage4_answer import (route, build_context, generate,
                           MIN_ANSWER_CONFIDENCE, FUZZY_FLOOR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pharmalens")

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
STORE = ROOT / "data" / "chroma"

TIERS = ["budget", "standard", "premium", "specialty"]
HIGH_CARD = ["composition_key", "manufacturer_key"]
LOW_CARD = ["dosage_form", "type"]
NUMERIC = ["pack_qty", "n_ingredients"]

state = {}


# ==========================================================================
# Startup
# ==========================================================================

def build_encoders(df):
    """Recompute target-encoding maps from the full product table.

    See design note 2: stage 2 used out-of-fold encoding to get an honest
    test score. At serving time there is no test set, so the full-data
    mapping is both correct and stronger.
    """
    y = pd.Series(pd.Categorical(df["price_tier"], categories=TIERS).codes,
                  index=df.index)
    valid = y >= 0
    y, df = y[valid], df[valid]
    global_mean = y.mean()
    smoothing = 20

    enc = {}
    for col in HIGH_CARD:
        stats = (pd.DataFrame({"c": df[col].values, "y": y.values})
                 .groupby("c")["y"].agg(["mean", "count"]))
        enc[col] = ((stats["mean"] * stats["count"] + global_mean * smoothing)
                    / (stats["count"] + smoothing)).to_dict()
        enc[f"freq_{col}"] = df[col].value_counts().to_dict()
    enc["_global_mean"] = float(global_mean)
    return enc


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    log.info("loading PharmaLens...")

    import chromadb
    from sentence_transformers import SentenceTransformer
    from xgboost import XGBClassifier

    docs = pd.read_parquet(PROC / "documents.parquet")
    if "qa_any" in docs.columns:
        docs = docs[~docs["qa_any"].astype(bool)].reset_index(drop=True)
    products = pd.read_parquet(PROC / "products.parquet")
    log.info("  data: %d documents, %d products", len(docs), len(products))

        # In the container the model is baked in at a known path; locally it
    # resolves by name from the HF cache. Same code, both environments.
    model_ref = os.environ.get("EMBEDDING_MODEL_PATH", MODEL_NAME)
    model = SentenceTransformer(model_ref)
    log.info("  embedding model: %s", model_ref)
    client = chromadb.PersistentClient(path=str(STORE))
    col = client.get_collection(COLLECTION)
    resolver = DrugResolver(products, docs)
    log.info("  resolver: %d ingredients, %d brand stems",
             len(resolver.by_ingredient), len(resolver.by_stem))

    clf = None
    model_path = PROC / "price_tier_model.json"
    if model_path.exists():
        clf = XGBClassifier()
        clf.load_model(str(model_path))
        log.info("  classifier loaded")

    state.update({
        "docs": docs,
        "products": products.set_index("product_row_id", drop=False),
        "docs_by_key": dict(zip(docs["composition_key"], docs["text"])),
        "doc_rows": docs.set_index("composition_key", drop=False),
        "retriever": PharmaLensRetriever(col, model, resolver, docs),
        "resolver": resolver,
        "clf": clf,
        "encoders": build_encoders(products),
        "feature_names": None,
        "dry_run": not os.environ.get("ANTHROPIC_API_KEY"),
        "loaded_at": time.time(),
    })
    log.info("ready in %.1fs (generation: %s)", time.time() - t0,
             "disabled" if state["dry_run"] else "enabled")
    yield
    state.clear()


app = FastAPI(
    title="PharmaLens",
    description=("Retrieval and price modelling over 253,969 Indian "
                 "pharmaceutical product records. Summarises retail "
                 "catalogue data. Not a medical tool."),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ==========================================================================
# Schemas
# ==========================================================================

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500,
                          examples=["what are the side effects of Crocin"])
    k: int = Field(5, ge=1, le=10)
    generate_answer: bool = Field(True,
                                  description="If false, return routing and "
                                              "documents without calling an LLM")


class ResolvedDrug(BaseModel):
    composition_key: str
    matched_text: str
    method: str
    confidence: float


class RetrievedDoc(BaseModel):
    composition_key: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    route: str
    route_reason: str
    mode: Optional[str]
    confidence: float
    ambiguous: bool
    resolved_drugs: List[ResolvedDrug]
    aspect: str
    documents: List[RetrievedDoc]
    answer: Optional[str]
    generated: bool
    disclaimer: str
    latency_ms: int


class PredictRequest(BaseModel):
    composition_key: str = Field(..., examples=["paracetamol"])
    manufacturer_key: str = Field(..., examples=["cipla"])
    dosage_form: str = Field("tablet", examples=["tablet"])
    pack_qty: float = Field(10, gt=0)
    n_ingredients: int = Field(1, ge=1, le=8)
    type: str = Field("allopathy")


class PredictResponse(BaseModel):
    predicted_tier: str
    probabilities: dict
    tier_ranges_inr: dict
    note: str


DISCLAIMER = ("Derived from a public Indian retail pharmacy dataset "
              "(prices as of November 2022). Not medical advice.")

TIER_RANGES = {"budget": "Rs 1-48", "standard": "Rs 48-79",
               "premium": "Rs 79-140", "specialty": "Rs 140+"}


# ==========================================================================
# Endpoints
# ==========================================================================

@app.get("/health")
def health():
    if not state:
        raise HTTPException(503, "still loading")
    return {
        "status": "ok",
        "documents": len(state["docs"]),
        "products": len(state["products"]),
        "classifier": state["clf"] is not None,
        "generation": not state["dry_run"],
        "uptime_s": round(time.time() - state["loaded_at"], 1),
    }


@app.get("/stats")
def stats():
    r = state["resolver"]
    docs = state["docs"]
    return {
        "corpus": {
            "compositions": len(docs),
            "products": len(state["products"]),
            "median_brands_per_composition":
                float(docs["n_brands"].median()) if "n_brands" in docs else None,
        },
        "resolver": {
            "ingredients": len(r.by_ingredient),
            "spelling_variants": len(r.by_variant),
            "product_names": len(r.by_product),
            "brand_stems": len(r.by_stem),
        },
        "thresholds": {
            "min_answer_confidence": MIN_ANSWER_CONFIDENCE,
            "fuzzy_floor": FUZZY_FLOOR,
        },
    }


@app.get("/product/{product_id}")
def get_product(product_id: int):
    products = state["products"]
    if product_id not in products.index:
        raise HTTPException(404, f"product {product_id} not found")
    row = products.loc[product_id]
    ck = row.get("composition_key")
    return {
        "product_id": int(product_id),
        "name": row["name"],
        "price_inr": float(row["price_inr"]),
        "price_tier": str(row.get("price_tier")),
        "manufacturer": row["manufacturer_name"],
        "pack_size": row["pack_size_label"],
        "dosage_form": row.get("dosage_form"),
        "composition_key": ck,
        "discontinued": bool(row.get("is_discontinued", False)),
        "composition_document": state["docs_by_key"].get(ck),
        "disclaimer": DISCLAIMER,
    }


@app.get("/composition/{key:path}")
def get_composition(key: str):
    rows = state["doc_rows"]
    if key not in rows.index:
        raise HTTPException(404, f"composition '{key}' not found")
    r = rows.loc[key]
    return {
        "composition_key": key,
        "text": r["text"],
        "therapeutic_class": r.get("therapeutic_class"),
        "n_brands": int(r.get("n_brands") or 0),
        "price_min_inr": float(r.get("price_min") or 0),
        "price_median_inr": float(r.get("price_median") or 0),
        "price_max_inr": float(r.get("price_max") or 0),
        "class_agreement": float(r.get("class_agreement") or 0),
        "disclaimer": DISCLAIMER,
    }


@app.get("/search")
def search_products(q: str = Query(..., min_length=2), limit: int = 10):
    """Resolve a brand or generic name to compositions. No LLM involved."""
    resolved = state["resolver"].resolve(q)
    return {
        "query": q,
        "resolved": [
            {"composition_key": ck, "matched_text": span,
             "method": m, "confidence": c}
            for ck, span, m, c in resolved[:limit]
        ],
        "count": len(resolved),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if state["clf"] is None:
        raise HTTPException(503, "classifier not loaded")

    enc = state["encoders"]
    gm = enc["_global_mean"]

    # Same feature order the model was trained on. Column names must match
    # exactly or XGBoost will silently mis-map them.
    feats = {
        "te_composition_key": enc["composition_key"].get(req.composition_key, gm),
        "te_manufacturer_key": enc["manufacturer_key"].get(req.manufacturer_key, gm),
        "freq_composition_key": enc["freq_composition_key"].get(req.composition_key, 0),
        "freq_manufacturer_key": enc["freq_manufacturer_key"].get(req.manufacturer_key, 0),
    }
    forms = ["capsule", "cream", "drops", "gel", "infusion", "injection",
             "lotion", "ointment", "powder", "solution", "spray",
             "suspension", "syrup", "tablet", "unknown"]
    for f in forms:
        feats[f"dosage_form_{f}"] = int(req.dosage_form.lower() == f)
    feats["type_allopathy"] = int(req.type.lower() == "allopathy")
    feats["pack_qty"] = req.pack_qty
    feats["n_ingredients"] = req.n_ingredients

    X = pd.DataFrame([feats])
    if state["feature_names"] is None:
        booster_names = state["clf"].get_booster().feature_names
        state["feature_names"] = booster_names or list(X.columns)
    X = X.reindex(columns=state["feature_names"], fill_value=0)

    proba = state["clf"].predict_proba(X)[0]
    idx = int(np.argmax(proba))

    unknown = (req.composition_key not in enc["composition_key"]
               and req.manufacturer_key not in enc["manufacturer_key"])
    note = ("Both composition and manufacturer are unseen; the prediction "
            "falls back to the global mean and is close to a guess."
            if unknown else
            "Predicts a price quartile, not a price. Adjacent tiers "
            "(e.g. standard vs premium) blur at the boundary.")

    return PredictResponse(
        predicted_tier=TIERS[idx],
        probabilities={t: round(float(p), 4) for t, p in zip(TIERS, proba)},
        tier_ranges_inr=TIER_RANGES,
        note=note,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.time()

    retrieval = state["retriever"].query(req.question, k=req.k)
    decision = route(req.question, retrieval)

    answer, generated = None, False
    if not decision.calls_llm:
        answer = decision.fixed_response
    elif req.generate_answer and not state["dry_run"]:
        ctx = build_context(retrieval, decision, state["docs_by_key"])
        try:
            answer, _ = generate(req.question, ctx, decision.preamble, False)
            generated = True
        except Exception as e:
            log.exception("generation failed")
            answer = None
            decision.reason += f" (generation failed: {type(e).__name__})"

        # On REFUSE and SCOPE we deliberately return NO documents. The router
    # decided these results should not be shown, and a client that ignored
    # the route field could otherwise render unrelated drugs as an answer.
    # That is the exact failure the router exists to prevent.
    if decision.route in ("REFUSE", "SCOPE"):
        hits = []
    elif decision.compositions:
        approved = set(decision.compositions)
        hits = [h for h in retrieval["hits"]
                if h["composition_key"] in approved]
    else:
        hits = retrieval["hits"]

    return AskResponse(
        question=req.question,
        route=decision.route,
        route_reason=decision.reason,
        mode=retrieval["mode"],
        confidence=round(retrieval["confidence"], 3),
        ambiguous=bool(retrieval.get("ambiguous")),
        resolved_drugs=[
            ResolvedDrug(composition_key=ck, matched_text=span,
                         method=m, confidence=c)
            for ck, span, m, c in retrieval["resolved_drugs"][:8]
        ],
        aspect=retrieval["aspect"],
        documents=[
            RetrievedDoc(composition_key=h["composition_key"],
                         score=round(h["score"], 4), text=h["text"])
            for h in hits[:req.k]
        ],
        answer=answer,
        generated=generated,
        disclaimer=DISCLAIMER,
        latency_ms=int((time.time() - t0) * 1000),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
