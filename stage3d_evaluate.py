#!/usr/bin/env python3
"""
PharmaLens - Stage 3d: measure the retrieval system

WHY THIS EXISTS
---------------
Everything so far is anecdote: six brand names and three regression cases.
"I built a RAG pipeline" is what every candidate says. "I measured
recall@k across three retrieval configurations on a 200-case gold set,
and here is where each one fails" is what almost none of them say.

This script produces the table that goes in your README.

HOW THE GOLD SET IS BUILT
-------------------------
Ground truth is GENERATED FROM THE DATA, not hand-written:

    take a real product   'Crocin 500 Tablet'
    its brand stem        'crocin'                     <- becomes the query
    its known composition 'paracetamol'                <- becomes the answer

The mapping comes from the source data, so it is not my opinion about
which brand maps to which drug. That scales to hundreds of cases with
zero manual labelling.

Hand-written cases cover what generation cannot: concept queries with no
drug named, drugs absent from the corpus, and typos.

THE THREE CONFIGURATIONS COMPARED
---------------------------------
    A  semantic_only     pure vector search (the 3b baseline)
    B  resolver_only     exact lookup, no embeddings at all
    C  hybrid            resolve -> filter -> semantic search (shipped)

If C does not beat A, the whole architecture was not worth building. If
C does not beat B, the embeddings are not earning their place. Both are
worth knowing.

METRICS
-------
    recall@k   is the correct composition in the top k?
    MRR        1/rank of the first correct hit, averaged (rewards ranking)
    refusal    on absent drugs, does the system correctly decline?

USAGE
-----
    cd P:\\PharmaLens
    python .\\stage3d_evaluate.py
"""

import torch  # MUST be first on Windows (WinError 1114 on c10.dll)

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the shipped retrieval code rather than reimplementing it -- an
# evaluation that tests a COPY of the system is testing the wrong thing.
from stage3c_retrieval import (DrugResolver, PharmaLensRetriever,
                               brand_stem, MODEL_NAME, COLLECTION)

pd.set_option("display.width", 140)

SEED = 42
N_BRAND_CASES = 120      # generated: brand name -> composition
N_GENERIC_CASES = 60     # generated: ingredient name -> composition


# ==========================================================================
# Hand-written cases: the things generation cannot produce
# ==========================================================================

CONCEPT_CASES = [
    # (query, acceptable therapeutic classes) -- judged by CLASS, not by a
    # single composition, because many drugs are correct answers here.
    ("medicine for high blood pressure", {"CARDIAC"}),
    ("something for acidity and heartburn", {"GASTRO INTESTINAL"}),
    ("treatment for type 2 diabetes", {"ANTI DIABETIC"}),
    ("what helps with allergy and runny nose", {"RESPIRATORY"}),
    ("drug for bacterial infection", {"ANTI INFECTIVES"}),
    ("medicine for joint pain and inflammation", {"PAIN ANALGESICS"}),
    ("something for fungal skin infection", {"DERMA", "ANTI INFECTIVES"}),
    ("treatment for depression", {"NEURO CNS"}),
]

# Real drugs that should NOT be in an Indian retail corpus, or are absent.
# Correct behaviour is REFUSAL, not a confident wrong answer.
ABSENT_CASES = [
    "flibanserin", "esketamine", "lecanemab", "tirzepatide",
    "zuranolone", "omaveloxolone",
]

TYPO_CASES = [
    ("amoxicilin", "amoxycillin"),
    ("paracetmol", "paracetamol"),
    ("metfromin", "metformin"),
    ("azithromicin", "azithromycin"),
    ("pantaprazole", "pantoprazole"),
    ("cetirizin", "cetrizine"),
]


# ==========================================================================
# Gold set construction
# ==========================================================================

def build_gold_set(products, docs, rng):
    """Generate test cases with ground truth taken from the data itself."""
    valid = set(docs["composition_key"])
    cases = []

    # --- brand -> composition -------------------------------------------
    # Sample across the popularity range, not just the head. A system that
    # only works on Crocin is not a system.
    pool = products[products["composition_key"].isin(valid)].copy()
    pool["stem"] = pool["name"].map(brand_stem)
    pool = pool[pool["stem"].str.len() >= 4]

    # Stratify by how many products share the composition, so rare drugs
    # are represented rather than drowned by the top 20 compositions.
    pool["popularity_band"] = pd.qcut(
        pool.groupby("composition_key")["composition_key"]
            .transform("size"), 4, labels=["rare", "uncommon", "common",
                                           "very_common"], duplicates="drop")

    per_band = max(1, N_BRAND_CASES // pool["popularity_band"].nunique())
    for band, grp in pool.groupby("popularity_band", observed=True):
        n = min(per_band, len(grp))
        for _, r in grp.sample(n, random_state=SEED).iterrows():
            cases.append({
                "query": r["stem"],
                "expected": r["composition_key"],
                "category": f"brand_{band}",
                "judge": "composition",
            })

    # --- generic name -> composition -------------------------------------
    singles = [k for k in valid if "+" not in k]
    for name in rng.sample(singles, min(N_GENERIC_CASES, len(singles))):
        cases.append({"query": name, "expected": name,
                      "category": "generic", "judge": "composition"})

    # --- typos -----------------------------------------------------------
    for typo, correct in TYPO_CASES:
        if correct in valid:
            cases.append({"query": f"side effects of {typo}",
                          "expected": correct, "category": "typo",
                          "judge": "composition"})

    # --- concepts (judged by therapeutic class) --------------------------
    for q, classes in CONCEPT_CASES:
        cases.append({"query": q, "expected": sorted(classes),
                      "category": "concept", "judge": "class"})

    # --- absent drugs (judged by refusal) --------------------------------
    for name in ABSENT_CASES:
        if name not in valid:
            cases.append({"query": f"side effects of {name}",
                          "expected": None, "category": "absent",
                          "judge": "refusal"})

    return pd.DataFrame(cases)


# ==========================================================================
# The three configurations
# ==========================================================================

def run_semantic_only(retriever, query, k):
    """Config A: pure vector search. The 3b baseline."""
    qv = retriever.model.encode([query], normalize_embeddings=True)[0]
    res = retriever.col.query(query_embeddings=[qv.tolist()], n_results=k)
    return list(res["ids"][0]), 0.0


def run_resolver_only(retriever, query, k):
    """Config B: exact lookup, no embeddings. Ranked by confidence."""
    resolved = retriever.resolver.resolve(query)
    conf = max([c for *_, c in resolved], default=0.0)
    return [ck for ck, _, _, _ in resolved][:k], conf


def run_hybrid(retriever, query, k):
    """Config C: resolve -> filter -> semantic. What we ship."""
    res = retriever.query(query, k=k)
    return [h["composition_key"] for h in res["hits"]], res["confidence"]


CONFIGS = [("A_semantic_only", run_semantic_only),
           ("B_resolver_only", run_resolver_only),
           ("C_hybrid", run_hybrid)]


# ==========================================================================
# Scoring
# ==========================================================================

def score_case(case, got, conf, docs_by_key, refusal_threshold=0.60):
    """Return (hit@1, hit@3, hit@5, reciprocal_rank)."""
    judge = case["judge"]

    if judge == "refusal":
        # Correct behaviour: resolve nothing, or resolve with confidence
        # below the threshold at which we would present an answer.
        declined = (conf < refusal_threshold)
        return (declined, declined, declined, 1.0 if declined else 0.0)

    if judge == "class":
        acceptable = set(case["expected"])
        ranks = [i for i, ck in enumerate(got)
                 if docs_by_key.get(ck) in acceptable]
    else:
        ranks = [i for i, ck in enumerate(got) if ck == case["expected"]]

    if not ranks:
        return (False, False, False, 0.0)
    r = ranks[0]
    return (r < 1, r < 3, r < 5, 1.0 / (r + 1))


# ==========================================================================

def main():
    root = Path(__file__).resolve().parent
    proc = root / "data" / "processed"
    store = root / "data" / "chroma"
    reports = root / "reports"
    reports.mkdir(exist_ok=True)

    import chromadb
    from sentence_transformers import SentenceTransformer

    rng = random.Random(SEED)

    # ------------------------------------------------------------- load
    print(f"\n{'=' * 78}\n1. LOAD SYSTEM\n{'=' * 78}")
    docs = pd.read_parquet(proc / "documents.parquet")
    if "qa_any" in docs.columns:
        docs = docs[~docs["qa_any"].astype(bool)].reset_index(drop=True)
    products = pd.read_parquet(proc / "products.parquet")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(store))
    col = client.get_collection(COLLECTION)
    resolver = DrugResolver(products, docs)
    retriever = PharmaLensRetriever(col, model, resolver, docs)
    print(f"  documents {len(docs):,}   indexed {col.count():,}")

    docs_by_key = dict(zip(docs["composition_key"], docs["therapeutic_class"]))

    # -------------------------------------------------------- gold set
    print(f"\n{'=' * 78}\n2. BUILD GOLD SET\n{'=' * 78}")
    gold = build_gold_set(products, docs, rng)
    print(f"  total cases: {len(gold):,}\n")
    print(gold["category"].value_counts().to_string())
    gold.to_json(reports / "gold_set.json", orient="records", indent=2)
    print(f"\n  saved -> {reports / 'gold_set.json'}")
    print("\n  Ground truth for brand/generic cases comes FROM THE DATA")
    print("  (a product's own composition_key), not from hand labelling.")

    # ------------------------------------------------------- evaluate
    print(f"\n{'=' * 78}\n3. EVALUATE\n{'=' * 78}")
    rows = []
    for cfg_name, fn in CONFIGS:
        print(f"  running {cfg_name} ...")
        for _, case in gold.iterrows():
            got, conf = fn(retriever, case["query"], 5)
            h1, h3, h5, rr = score_case(case, got, conf, docs_by_key)
            rows.append({"config": cfg_name, "category": case["category"],
                         "query": case["query"], "expected": case["expected"],
                         "got": got[:3], "conf": conf,
                         "h1": h1, "h3": h3, "h5": h5, "rr": rr})
    res = pd.DataFrame(rows)

    # ------------------------------------------------------- headline
    print(f"\n{'=' * 78}\n4. OVERALL  (the README table)\n{'=' * 78}")
    overall = (res.groupby("config")[["h1", "h3", "h5", "rr"]]
                  .mean().mul(100).round(1))
    overall.columns = ["recall@1", "recall@3", "recall@5", "MRR"]
    print(overall.to_string())

    # ------------------------------------------------- by category
    print(f"\n{'=' * 78}\n5. BY CATEGORY  (where each config fails)\n{'=' * 78}")
    bycat = (res.pivot_table(index="category", columns="config",
                             values="h5", aggfunc="mean").mul(100).round(1))
    print(bycat.to_string())
    print("\n  recall@5 by category, percent.")

    # ------------------------------------------------- interpretation
    print(f"\n{'=' * 78}\n6. WHAT THIS MEANS\n{'=' * 78}")
    a = overall.loc["A_semantic_only", "recall@5"]
    b = overall.loc["B_resolver_only", "recall@5"]
    c = overall.loc["C_hybrid", "recall@5"]
    print(f"  A semantic only : {a:5.1f}%")
    print(f"  B resolver only : {b:5.1f}%")
    print(f"  C hybrid        : {c:5.1f}%   ({c - a:+.1f} vs A, {c - b:+.1f} vs B)")
    print()
    if c > a + 5:
        print("  The architecture earns its complexity: filter-first beats")
        print("  pure vector search by a wide margin.")
    if c <= b + 2:
        print("  NOTE: the hybrid barely beats the resolver alone. The")
        print("  embeddings are carrying little weight -- check whether")
        print("  concept queries are the only place they help, and say so")
        print("  honestly rather than implying they do more.")
    else:
        print("  The embedding layer adds real value on top of lookup.")

    # ------------------------------------------------- failure analysis
    print(f"\n{'=' * 78}\n7. WHERE C STILL FAILS  (read these)\n{'=' * 78}")
    fails = res[(res["config"] == "C_hybrid") & (~res["h5"])]
    print(f"  {len(fails)} failures out of {len(gold)} cases\n")
    for cat, grp in fails.groupby("category"):
        print(f"  --- {cat} ({len(grp)}) ---")
        for _, r in grp.head(4).iterrows():
            print(f"    q: '{r['query']}'")
            print(f"       expected {r['expected']}")
            print(f"       got      {r['got']}  (conf {r['conf']:.2f})")

    # ------------------------------------------------------------ save
    res.to_csv(reports / "retrieval_eval.csv", index=False)
    overall.to_csv(reports / "retrieval_summary.csv")
    with open(reports / "retrieval_metrics.json", "w") as f:
        json.dump({"overall": overall.to_dict(),
                   "by_category": bycat.to_dict(),
                   "n_cases": len(gold)}, f, indent=2)
    print(f"\n  saved -> {reports / 'retrieval_eval.csv'}")
    print(f"  saved -> {reports / 'retrieval_summary.csv'}")

    print(f"\n{'=' * 78}")
    print("  Put the section 4 table in your README verbatim. It is the")
    print("  difference between 'I built RAG' and 'I evaluated retrieval'.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
