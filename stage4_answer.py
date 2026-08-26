#!/usr/bin/env python3
"""
PharmaLens - Stage 4: the answer layer

THE CENTRAL DESIGN DECISION
---------------------------
Most guardrails live in CODE, not in the prompt.

A prompt is probabilistic -- it works most of the time. For "never tell
someone an unrecognised drug is safe", most of the time is not good
enough. Stage 3 already emits deterministic signals: confidence, mode,
ambiguous, number of resolved drugs. Route on those BEFORE calling the
model, and three of five routes never reach the LLM at all.

    retrieval result -> ROUTER (plain Python) -> one of:

      REFUSE    nothing resolved / confidence too low     no LLM call
      CLARIFY   one brand maps to several compositions    no LLM call
      SCOPE     clinical question (dosing, "should I")    no LLM call
      ASSUME    fuzzy match -- state the assumption       LLM, caveat first
      PARTIAL   several drugs, no interaction data        LLM, limits stated
      ANSWER    single drug, high confidence              LLM, normal path

EVERY ROUTE COMES FROM A MEASURED FAILURE
-----------------------------------------
  REFUSE   'side effects of flibanserin' -> semantic mode returned
           lomefloxacin at 0.526. Confident-looking garbage.
  CLARIFY  'noxprin' ties ozenoxacin and enoxaparin at 0.98.
  ASSUME   'esketamine' -> ketamine at 0.70. Esketamine is a REAL,
           DIFFERENT drug. No fuzzy cutoff separates that from a genuine
           typo -- 0.86 catches both, 0.92 catches neither. So it is
           surfaced as an assumption instead of hidden.
  PARTIAL  'Augmentin and Azithral together' resolves two compositions,
           but this corpus has NO drug-drug interaction data.

WHAT THE PROMPT DOES (and only this)
------------------------------------
  1. summarise the retrieved documents
  2. cite the composition each claim came from
  3. represent proportions honestly -- an 18% claim must read as a
     minority claim, not sit beside an 82% claim as an equal

PROMPTS ARE VERSIONED
---------------------
Each prompt is a dated constant with a changelog. Prompt changes are code
changes; treating them as untracked strings is how a working system
silently regresses.

USAGE
-----
    cd P:\\PharmaLens
    pip install anthropic
    $env:ANTHROPIC_API_KEY="sk-ant-..."
    python .\\stage4_answer.py

Without an API key it runs in DRY RUN: routing is exercised in full and
prompts are printed, but no model is called. Run it dry first -- the
routing is the part worth reading.
"""

import torch  # MUST be first on Windows (WinError 1114 on c10.dll)

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from stage3c_retrieval import (DrugResolver, PharmaLensRetriever,
                               MODEL_NAME, COLLECTION)

pd.set_option("display.width", 130)


# ==========================================================================
# Thresholds -- deliberately named, not magic numbers
# ==========================================================================

# Below this, we do not answer at all. 0.70 is exactly the fuzzy-match
# confidence, so fuzzy hits sit at the boundary and route to ASSUME.
MIN_ANSWER_CONFIDENCE = 0.90

# At or above MIN_ANSWER_CONFIDENCE -> normal answer.
# In [FUZZY_FLOOR, MIN_ANSWER_CONFIDENCE) -> answer WITH a stated assumption.
FUZZY_FLOOR = 0.60

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 700


# Questions we will not answer regardless of retrieval quality. This is a
# scope boundary, not a capability gap: the corpus contains marketing and
# label metadata scraped from a retail site, not clinical guidance.
CLINICAL_PATTERNS = [
    r"\bshould i\b", r"\bcan i stop\b", r"\bhow (much|many)\s+.*\b(take|dose)",
    r"\bdosage for\b", r"\bhow many (tablets|mg)\b", r"\bmy (doctor|symptoms)\b",
    r"\bis it (right|okay|ok) for me\b", r"\bprescrib", r"\bdiagnos",
    r"\binstead of\b", r"\bswitch (to|from)\b", r"\boverdose\b",
    r"\bmy (child|baby|son|daughter|mother|father)\b",
]
_CLINICAL_RE = re.compile("|".join(CLINICAL_PATTERNS), re.IGNORECASE)


# ==========================================================================
# Prompts -- versioned constants with changelogs
# ==========================================================================

SYSTEM_PROMPT_V3 = """\
You summarise Indian pharmaceutical product records. You are not a \
clinician and you do not give medical advice.

YOUR SOURCE
Your only source is the CONTEXT block supplied with each question. It is \
derived from a public dataset of Indian retail pharmacy listings: product \
names, compositions, prices, therapeutic classes, and side effects \
reported across products sharing a composition.

RULES
1. Use only the CONTEXT. If it does not contain the answer, say so. Never \
supply pharmacological knowledge from memory, even when you are confident \
it is correct.
2. Name the composition you are describing, so the reader knows which drug \
the information belongs to.
3. Percentages in the CONTEXT mean "this proportion of products sharing \
this composition listed this". Reflect them honestly: a claim at 18% is a \
minority listing and must be described as such, never presented alongside \
an 82% claim as an equal.
4. Do not recommend, compare for suitability, or suggest dosing. \
Describing what a dataset records is not advice.
5. Do not state or imply that anything is safe. Absence of a warning in \
retail listing data is not evidence of safety.
6. Be brief: 3-6 sentences. Plain language, no bullet lists.
7. Close by noting the source is retail listing data, and that a \
pharmacist or doctor is the right check for anything that matters.

CHANGELOG
v1  initial: summarise context, cite composition
v2  added rule 3 after 'Treatment of Resistance Tuberculosis (18%)' was
    reported with the same weight as an 82% indication
v3  added rules 5 and 7 after a draft answered a with-alcohol question by
    saying no interaction was listed, which reads as reassurance
"""

USER_TEMPLATE_V2 = """\
QUESTION: {question}

{preamble}CONTEXT:
{context}

Answer the question using only the CONTEXT above."""


# Fixed responses for routes that never reach the model. Deterministic
# text means these can never drift.
REFUSAL_TEXT = (
    "I don't have data on that in this dataset, which covers Indian retail "
    "pharmacy listings. I can't tell you anything about it -- and no data "
    "is not the same as no risk. A pharmacist can look it up properly."
)

SCOPE_TEXT = (
    "That's a clinical question, and this tool can't answer it. It "
    "summarises what a retail pharmacy dataset records about products -- "
    "compositions, prices, listed side effects -- not what anyone should "
    "take. Please ask a doctor or pharmacist."
)


# ==========================================================================
# Router
# ==========================================================================

@dataclass
class Decision:
    route: str                 # REFUSE | CLARIFY | SCOPE | ASSUME | PARTIAL | ANSWER
    reason: str                # why, in one line -- for logs and for the UI
    calls_llm: bool
    fixed_response: str = ""
    preamble: str = ""         # prepended to the user prompt
    compositions: list = field(default_factory=list)


def route(question, retrieval):
    """Decide how to respond. Pure function, no model involved."""

    # --- SCOPE: checked first, because a perfectly resolved drug does
    #     not make a dosing question answerable ----------------------
    if _CLINICAL_RE.search(question):
        return Decision("SCOPE", "clinical//advice-seeking phrasing",
                        False, fixed_response=SCOPE_TEXT)

    resolved = retrieval["resolved_drugs"]
    conf = retrieval["confidence"]

    # --- REFUSE: nothing recognised -------------------------------------
    if not resolved:
        return Decision("REFUSE", "no drug name resolved", False,
                        fixed_response=REFUSAL_TEXT)

    if conf < FUZZY_FLOOR:
        return Decision("REFUSE", f"confidence {conf:.2f} below floor",
                        False, fixed_response=REFUSAL_TEXT)

    top = [ck for ck, _, _, c in resolved if c == conf]

    # --- CLARIFY: genuinely ambiguous brand -----------------------------
        # --- CLARIFY: genuinely ambiguous brand -----------------------------
    # Ambiguity means one BRAND span maps to several compositions. The
    # ingredient hierarchy (exact 1.00 vs contained 0.90) is NOT ambiguity,
    # so ingredient matches are excluded. 'noxprin' hits by_product ->
    # ozenoxacin at 0.98 and by_stem -> enoxaparin at 0.92; ranking by
    # confidence alone picks the wrong drug silently.
    top_span = resolved[0][1]
    brandish = {ck for ck, span, m, _ in resolved
                if span == top_span
                and m in ("product_name", "brand_stem")}
    if len(brandish) > 1:
        opts = "; ".join(sorted(brandish)[:4])
        return Decision(
            "CLARIFY", f"'{top_span}' maps to {len(brandish)} compositions",
            False,
            fixed_response=(
                f"'{top_span}' matches more than one composition in this "
                f"dataset: {opts}. Which did you mean? They are different "
                f"medicines."),
            compositions=sorted(brandish))
    # --- ASSUME: fuzzy match, real drug may simply be absent -------------
    if conf < MIN_ANSWER_CONFIDENCE:
        matched = resolved[0][1]           # looks like 'esketamine~ketamine'
        typed, guess = (matched.split("~") + [""])[:2] if "~" in matched \
            else (matched, top[0])
        return Decision(
            "ASSUME", f"fuzzy match {conf:.2f}: {typed} -> {guess}", True,
            preamble=(
                f"IMPORTANT: '{typed}' was not found in the dataset. The "
                f"closest name is '{guess}'. Begin your answer by stating "
                f"plainly that '{typed}' is not in the data, and ask "
                f"whether '{guess}' was meant. Do not assume they are the "
                f"same drug -- similar names are often different "
                f"medicines. Only then describe '{guess}'.\n\n"),
            compositions=top)

    # --- PARTIAL: several drugs, and we have no interaction data ---------
    distinct = {ck for ck, _, _, c in resolved if c >= MIN_ANSWER_CONFIDENCE}
    if len(distinct) > 1 and _looks_like_combination(question):
        return Decision(
            "PARTIAL", f"{len(distinct)} drugs, no interaction data", True,
            preamble=(
                "IMPORTANT: this question asks about taking drugs "
                "together. This dataset contains NO drug-drug interaction "
                "data. Describe each drug separately from the CONTEXT, "
                "then state clearly that whether they can be combined "
                "cannot be assessed from this source and needs a "
                "pharmacist. Do not infer an interaction, and do not "
                "imply the combination is fine.\n\n"),
            compositions=sorted(distinct))

    return Decision("ANSWER", f"single drug, confidence {conf:.2f}", True,
                    compositions=top[:1])


_COMBO_RE = re.compile(
    r"\b(together|with|combine|combination|and|both|same time|alongside)\b",
    re.IGNORECASE)


def _looks_like_combination(question):
    return bool(_COMBO_RE.search(question))


# ==========================================================================
# Context assembly
# ==========================================================================

def build_context(retrieval, decision, docs_by_key, max_docs=3):
    """Only the documents the router approved. Never the whole top-k.

    If retrieval returned five plausible documents but the router decided
    only one is trustworthy, the model must not see the other four --
    given them, it will use them.
    """
    keys = decision.compositions or [h["composition_key"]
                                     for h in retrieval["hits"][:1]]
    blocks = []
    for ck in keys[:max_docs]:
        text = docs_by_key.get(ck)
        if text:
            blocks.append(f"[composition: {ck}]\n{text}")
    return "\n\n".join(blocks) if blocks else "(no documents retrieved)"


# ==========================================================================
# Generation
# ==========================================================================

def generate(question, context, preamble, dry_run):
    user = USER_TEMPLATE_V2.format(question=question, preamble=preamble,
                                   context=context)
    if dry_run:
        return None, user
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT_V3,
        messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in resp.content if b.type == "text"), user


# ==========================================================================
# The test suite -- every case traces to a real observed failure
# ==========================================================================

TEST_CASES = [
    ("what are the side effects of Crocin",              "ANSWER"),
    ("what is Pan D used for",                           "ANSWER"),
    ("side effects of flibanserin",                      "REFUSE"),
    ("side effects of esketamine",                       "ASSUME"),
    ("what are the side effects of amoxicilin",          "ASSUME"),
    ("can I take Augmentin and Azithral together",       "PARTIAL"),
    ("how many tablets of Dolo should I take",           "SCOPE"),
    ("should I stop taking amlodipine",                  "SCOPE"),
    ("is Dolo safe for my child",                        "SCOPE"),
    ("what is amoxycillin used for",                     "ANSWER"),
    ("medicine for high blood pressure",                 "REFUSE"),
    ("tell me about noxprin",                            "CLARIFY"),
]


def main():
    root = Path(__file__).resolve().parent
    proc = root / "data" / "processed"
    store = root / "data" / "chroma"
    reports = root / "reports"
    reports.mkdir(exist_ok=True)

    import chromadb
    from sentence_transformers import SentenceTransformer

    dry_run = not os.environ.get("ANTHROPIC_API_KEY")

    print(f"\n{'=' * 78}\n1. LOAD\n{'=' * 78}")
    docs = pd.read_parquet(proc / "documents.parquet")
    if "qa_any" in docs.columns:
        docs = docs[~docs["qa_any"].astype(bool)].reset_index(drop=True)
    products = pd.read_parquet(proc / "products.parquet")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(store))
    col = client.get_collection(COLLECTION)
    resolver = DrugResolver(products, docs)
    retriever = PharmaLensRetriever(col, model, resolver, docs)
    docs_by_key = dict(zip(docs["composition_key"], docs["text"]))
    print(f"  documents {len(docs):,}   indexed {col.count():,}")
    print(f"  mode      {'DRY RUN (no API key set)' if dry_run else 'LIVE'}")

    print(f"\n{'=' * 78}\n2. THRESHOLDS\n{'=' * 78}")
    print(f"  MIN_ANSWER_CONFIDENCE {MIN_ANSWER_CONFIDENCE}   "
          f"at or above -> normal answer")
    print(f"  FUZZY_FLOOR           {FUZZY_FLOOR}   below -> refuse outright")
    print(f"  between the two -> answer, but state the assumption first")
    print("\n  Fuzzy matches land at exactly 0.70, so they sit in the")
    print("  middle band by construction rather than by accident.")

    # ------------------------------------------------------- routing test
    print(f"\n{'=' * 78}\n3. ROUTING TEST\n{'=' * 78}")
    print("Every case below traces to a failure observed in stage 3.\n")

    results, passed = [], 0
    for question, expected in TEST_CASES:
        r = retriever.query(question, k=5)
        d = route(question, r)
        ok = d.route == expected
        passed += ok
        results.append({"question": question, "expected": expected,
                        "got": d.route, "pass": ok, "reason": d.reason,
                        "conf": r["confidence"], "llm": d.calls_llm})
        flag = "PASS" if ok else "FAIL"
        print(f"  {flag}  {d.route:<8} (want {expected:<8}) "
              f"conf {r['confidence']:.2f}  '{question}'")
        if not ok:
            print(f"        reason: {d.reason}")

    print(f"\n  {passed}/{len(TEST_CASES)} routed correctly")
    no_llm = sum(1 for r in results if not r["llm"])
    print(f"  {no_llm}/{len(TEST_CASES)} handled WITHOUT calling the model")
    print("\n  Those are hard guarantees. A prompt can be talked around;")
    print("  an if-statement cannot.")

    # ------------------------------------------------------ full examples
    print(f"\n{'=' * 78}\n4. FULL RESPONSES\n{'=' * 78}")
    SHOW = ["side effects of flibanserin",
            "side effects of esketamine",
            "can I take Augmentin and Azithral together",
            "how many tablets of Dolo should I take",
            "what are the side effects of Crocin"]

    for question in SHOW:
        r = retriever.query(question, k=5)
        d = route(question, r)
        print(f"\n{'-' * 78}\n  Q: {question}")
        print(f"  route : {d.route}  ({d.reason})")

        if not d.calls_llm:
            print(f"  no LLM call")
            print(f"\n  RESPONSE:\n    {d.fixed_response}")
            continue

        ctx = build_context(r, d, docs_by_key)
        answer, prompt = generate(question, ctx, d.preamble, dry_run)
        print(f"  context: {len(d.compositions)} document(s) "
              f"-- {', '.join(d.compositions[:3])}")
        if d.preamble:
            first = d.preamble.strip().split("\n")[0]
            print(f"  preamble: {first[:100]}...")
        if dry_run:
            print(f"\n  PROMPT SENT (dry run):\n")
            for line in prompt.split("\n")[:14]:
                print(f"    {line}")
            print("    ...")
        else:
            print(f"\n  RESPONSE:\n")
            for line in (answer or "").split("\n"):
                print(f"    {line}")

    # ------------------------------------------------------------- save
    pd.DataFrame(results).to_csv(reports / "routing_eval.csv", index=False)
    with open(reports / "prompts_v3.json", "w") as f:
        json.dump({"system": SYSTEM_PROMPT_V3, "user": USER_TEMPLATE_V2,
                   "refusal": REFUSAL_TEXT, "scope": SCOPE_TEXT,
                   "thresholds": {"min_answer": MIN_ANSWER_CONFIDENCE,
                                  "fuzzy_floor": FUZZY_FLOOR}}, f, indent=2)
    print(f"\n{'=' * 78}")
    print(f"  saved -> {reports / 'routing_eval.csv'}")
    print(f"  saved -> {reports / 'prompts_v3.json'}")
    if dry_run:
        print("\n  Set ANTHROPIC_API_KEY and rerun to see real answers.")
        print("  Roughly a dozen short calls -- a few cents.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
