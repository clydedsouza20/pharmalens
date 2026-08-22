#!/usr/bin/env python3
"""
PharmaLens - Stage 3a: document construction

TURNS  a row of lists and percentages
INTO   a paragraph an embedding model can actually understand.

WHY THIS STEP EXISTS
--------------------
Embedding models were trained on natural language. Given

    azithromycin, ANTI INFECTIVES, Diarrhea, Nausea, Vomiting

the model sees a word-salad with no grammar and produces a vague vector.
Given

    "Azithromycin is an anti-infective medicine used to treat bacterial
     infections. Commonly reported side effects include diarrhoea,
     nausea and vomiting."

it can build a vector that actually means something. Most RAG tutorials
skip this step because their source is already prose. Ours is not, so
document construction is a design decision we have to make deliberately.

THE THREE DECISIONS ENCODED BELOW
---------------------------------
1. TEXT vs METADATA
   Text  = what gets semantically matched (uses, effects, class, drug names)
   Meta  = what gets filtered on         (price, brands, ingredients, flags)
   Price is metadata: nobody searches "medicines costing Rs 47".

2. BRAND NAMES
   6,459 brands cannot go in the text -- they would drown the content.
   A few representative ones go in the text so brand queries have a
   semantic foothold; the FULL list goes in metadata for exact lookup.
   Two retrieval paths, one document.

3. PERCENTAGES
   Written in. "reported for 94% of products" lets the LLM distinguish a
   dominant effect from a marginal one, and gives it something to cite.

OUTPUT
------
    data/processed/documents.parquet    one row per composition:
        doc_id, text, and all metadata fields

USAGE
-----
    cd P:\\PharmaLens
    python .\\stage3a_documents.py
"""

import re
from pathlib import Path

import pandas as pd
import numpy as np

pd.set_option("display.width", 130)


# ==========================================================================
# Cell coercion
# ==========================================================================

def as_list(v):
    """Coerce a dataframe cell to a plain Python list.

    Parquet hands list columns back as numpy arrays. The natural-looking
    `v or []` raises ValueError, because numpy refuses to collapse a
    multi-element array into a single truth value. `pd.isna(v)` raises for
    the same reason. So: check for None first, then check the type, and
    only fall through to pd.isna() for genuine scalars.

    This is the most common papercut when round-tripping list columns
    through Parquet.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return list(v)
    try:
        if pd.isna(v):
            return []
    except (TypeError, ValueError):
        pass
    return [v]


def as_float(v, default=0.0):
    """Safe scalar float, tolerant of None/NaN."""
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


# ==========================================================================
# Small helpers for readable prose
# ==========================================================================

def oxford(items):
    """['a','b','c'] -> 'a, b and c'   (reads as English, not as a CSV)"""
    items = [str(i).strip() for i in items if str(i).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def humanize_class(cls):
    """'ANTI INFECTIVES' -> 'anti-infective'. Screaming caps hurt embeddings."""
    if not isinstance(cls, str) or not cls.strip():
        return None
    s = cls.strip().lower()
    fixes = {
        "anti infectives": "anti-infective",
        "pain analgesics": "pain relief / analgesic",
        "gastro intestinal": "gastrointestinal",
        "anti diabetic": "antidiabetic",
        "cardiac": "cardiac",
        "respiratory": "respiratory",
        "derma": "dermatological",
        "neuro cns": "neurological / central nervous system",
        "vitamins minerals nutrients": "vitamin, mineral or nutritional",
        "ophthal": "ophthalmic (eye)",
        "ophthal otologicals": "ophthalmic and otological (eye and ear)",
        "gynaec": "gynaecological",
        "urology": "urological",
        "hormones": "hormonal",
        "anti neoplastics": "anticancer",
        "blood related": "blood-related",
        "anti malarials": "antimalarial",
        "sex stimulants rejuvenators": "sexual health",
        "stomatologicals": "oral / dental",
        "otologicals": "otological (ear)",
        "vaccines": "vaccine",
        "anti tb": "anti-tuberculosis",
        "anti viral": "antiviral",
    }
    return fixes.get(s, s)


def ingredient_phrase(comp_key):
    """'amoxycillin + clavulanic acid' -> a readable subject phrase."""
    parts = [p.strip() for p in str(comp_key).split("+") if p.strip()]
    if not parts:
        return "This preparation", []
    if len(parts) == 1:
        return parts[0].title(), parts
    return f"The combination of {oxford([p.title() for p in parts])}", parts


_USE_PREFIX = re.compile(
    r"^\s*(treatment and prevention of|treatment of|management of|"
    r"managing|prevention of)\s+", re.IGNORECASE)


def clean_uses(uses):
    """'Treatment of Bacterial infections' -> 'bacterial infections'.

    The prefix repeats on every use and adds no retrieval signal; stripping
    it makes the sentence read naturally once instead of four times.
    """
    out = []
    for u in uses:
        u = _USE_PREFIX.sub("", str(u)).strip()
        if u:
            out.append(u[0].lower() + u[1:])
    return out


# ==========================================================================
# The document template
# ==========================================================================

def build_text(row, n_brands_in_text=4):
    """Compose one natural-language document for one composition.

    Deliberately written as sentences, not bullet points. The embedding
    model is a language model; give it language.
    """
    subject, parts = ingredient_phrase(row.get("composition_key", ""))
    is_combo = len(parts) > 1
    verb = "are" if is_combo and not subject.startswith("The") else "is"

    sent = []

    # --- 1. Identity + class -------------------------------------------
    cls = humanize_class(row.get("therapeutic_class"))
    if cls:
        article = "an" if cls[0] in "aeiou" else "a"
        sent.append(f"{subject} {verb} {article} {cls} medicine.")
    else:
        sent.append(f"{subject} {verb} a pharmaceutical preparation.")

    # --- 2. What it is used for ----------------------------------------
    uses = as_list(row.get("uses"))
    ushare = as_list(row.get("uses_share"))
    if uses:
        # Pad shares if lengths ever diverge, so zip never silently truncates.
        ushare = (ushare + [1.0] * len(uses))[:len(uses)]
        primary = [u for u, s in zip(uses, ushare) if as_float(s) >= 0.5]
        minor = [u for u, s in zip(uses, ushare) if as_float(s) < 0.5]
        if primary:
            sent.append(f"It is used for {oxford(clean_uses(primary))}.")
        if minor:
            sent.append(f"Some products are also labelled for "
                        f"{oxford(clean_uses(minor))}, though this is "
                        f"reported by a minority of products.")

    # --- 3. Side effects, with frequency -------------------------------
    effects = as_list(row.get("side_effects"))
    eshare = as_list(row.get("side_effects_share"))
    if effects:
        eshare = (eshare + [1.0] * len(effects))[:len(effects)]
        pairs = [(str(e), as_float(s)) for e, s in zip(effects, eshare)]
        common = [e for e, s in pairs if s >= 0.5][:8]
        less = [e for e, s in pairs if s < 0.5][:6]
        if common:
            top_pct = max(s for _, s in pairs if s >= 0.5)
            sent.append(f"Commonly reported side effects include "
                        f"{oxford([e.lower() for e in common])} "
                        f"(reported for up to {top_pct:.0%} of products "
                        f"with this composition).")
        if less:
            sent.append(f"Less frequently reported effects include "
                        f"{oxford([e.lower() for e in less])}.")

    # --- 4. Chemical / action class ------------------------------------
    chem = row.get("chemical_class")
    act = row.get("action_class")
    extra = []
    chem_s = str(chem).strip().lower() if isinstance(chem, str) else ""
    act_s = str(act).strip().lower() if isinstance(act, str) else ""
    if chem_s and chem_s != "nan":
        extra.append(f"chemically classified as {chem_s}")
    if act_s and act_s != "nan" and act_s != chem_s:
        extra.append(f"acting as {act_s}")
    if extra:
        sent.append(f"It is {oxford(extra)}.")

    # --- 5. Market presence: brands + forms + price ---------------------
    brands = [str(b) for b in as_list(row.get("brands"))][:n_brands_in_text]
    n_br = int(as_float(row.get("n_brands"), 0))
    forms = [str(f) for f in as_list(row.get("dosage_forms"))
             if str(f) != "unknown"]
    bits = []
    if brands:
            noun = "brand name" if n_br == 1 else "brand names"
            bits.append(f"sold in India under {n_br:,} {noun} "
            f"including {oxford(brands)}")
            f"including {oxford(brands)}"
    if forms:
        bits.append(f"available as {oxford(sorted(set(forms)))}")
    if bits:
        sent.append(f"It is {oxford(bits)}.")

    pmin = row.get("price_min")
    pmed = row.get("price_median")
    pmax = row.get("price_max")
    if pd.notna(pmed):
        sent.append(f"Retail prices range from Rs {as_float(pmin):,.0f} to "
                    f"Rs {as_float(pmax):,.0f}, with a median of "
                    f"Rs {as_float(pmed):,.0f}.")

    return " ".join(sent)


# ==========================================================================

def main():
    root = Path(__file__).resolve().parent
    proc = root / "data" / "processed"

    src = proc / "compositions.parquet"
    if not src.exists():
        raise SystemExit(f"\nERROR: {src} not found. Run stage1_etl_v2.py first.\n")

    print(f"\n{'=' * 76}\n1. LOAD COMPOSITIONS\n{'=' * 76}")
    comp = pd.read_parquet(src)
    print(f"  compositions : {len(comp):,}")
    print(f"  columns      : {len(comp.columns)}")

    missing = [c for c in ("composition_key", "uses", "side_effects")
               if c not in comp.columns]
    if missing:
        raise SystemExit(f"\nERROR: missing columns {missing}. "
                         f"Re-run stage1_etl_v2.py with the consensus patch.\n")

    # ------------------------------------------------------------ build
    print(f"\n{'=' * 76}\n2. BUILD DOCUMENT TEXT\n{'=' * 76}")
    comp["text"] = comp.apply(build_text, axis=1)
    comp["doc_id"] = ["comp_" + re.sub(r"[^a-z0-9]+", "_", str(k))[:80].strip("_")
                      for k in comp["composition_key"]]

    if comp["doc_id"].duplicated().any():
        dupes = int(comp["doc_id"].duplicated().sum())
        comp.loc[comp["doc_id"].duplicated(keep=False), "doc_id"] += (
            "_" + comp.groupby("doc_id").cumcount().astype(str))
        print(f"  disambiguated {dupes} duplicate doc_ids")

    lens = comp["text"].str.len()
    words = comp["text"].str.split().str.len()
    print(f"  characters : median {lens.median():.0f}, "
          f"p90 {lens.quantile(.9):.0f}, p95 {lens.quantile(.95):.0f}, "
          f"max {lens.max():,}")
    print(f"  words      : median {words.median():.0f}, "
          f"p90 {words.quantile(.9):.0f}, max {words.max():,}")

    print(f"\n  --- CHUNKING DECISION (measured, not assumed) ---")
    p95 = lens.quantile(0.95)
    if p95 < 2000:
        print(f"  95th percentile is {p95:.0f} characters.")
        print("  NO CHUNKING. One composition = one document = one vector.")
        print("  Splitting these would separate a drug's side effects from")
        print("  its own name -- the exact failure chunking is meant to")
        print("  prevent. Most tutorials chunk unconditionally; the right")
        print("  answer depends on the corpus, and this corpus says no.")
    else:
        print(f"  95th percentile is {p95:.0f} characters -- long enough that")
        print("  splitting is worth testing. Revisit at step 3d.")

    # ----------------------------------------------------- metadata prep
    print(f"\n{'=' * 76}\n3. METADATA (what we FILTER on, not match on)\n{'=' * 76}")

    # Ingredient list per composition -- the filter key that makes retrieval
    # reliable. Drug identity is looked up, never left to the embedding.
    comp["ingredients"] = comp["composition_key"].map(
        lambda k: [p.strip() for p in str(k).split("+") if p.strip()])
    comp["n_ingredients"] = comp["ingredients"].str.len()

    # Brand list, lowercased and punctuation-stripped, for exact lookup.
    if "brands" in comp.columns:
        comp["brand_keys"] = comp["brands"].map(
            lambda bs: [re.sub(r"[^a-z0-9 ]", " ", str(b).lower()).strip()
                        for b in as_list(bs)])
    else:
        comp["brand_keys"] = [[] for _ in range(len(comp))]

    meta_cols = ["doc_id", "composition_key", "ingredients", "n_ingredients",
                 "brand_keys", "n_brands", "therapeutic_class",
                 "chemical_class", "action_class", "class_agreement",
                 "n_products", "price_min", "price_median", "price_max",
                 "dosage_forms", "qa_any"]
    for c in meta_cols:
        if c in comp.columns:
            print(f"    {c}")

    # ------------------------------------------------------------ write
    print(f"\n{'=' * 76}\n4. WRITE\n{'=' * 76}")
    keep = ["doc_id", "text"] + meta_cols + ["uses", "side_effects",
                                             "substitutes"]
    keep = [c for c in dict.fromkeys(keep) if c in comp.columns]
    docs = comp[keep].copy()

    # Parquet-safe: every list column becomes a list of plain strings.
    for c in ["ingredients", "brand_keys", "dosage_forms", "uses",
              "side_effects", "substitutes"]:
        if c in docs.columns:
            docs[c] = docs[c].map(lambda v: [str(x) for x in as_list(v)])

    path = proc / "documents.parquet"
    docs.to_parquet(path, index=False)
    print(f"  {path.name}  {len(docs):,} rows  "
          f"({path.stat().st_size / 1e6:.1f} MB)")

    if "qa_any" in docs.columns:
        n_clean = int((~docs["qa_any"].astype(bool)).sum())
        print(f"  passing QA (index these)       : {n_clean:,}")
        print(f"  flagged (exclude at index time): {len(docs) - n_clean:,}")
        show = docs[~docs["qa_any"].astype(bool)]
    else:
        show = docs

    # ------------------------------------------------------- inspection
    print(f"\n{'=' * 76}\n5. READ THESE OUT LOUD\n{'=' * 76}")
    print("If a document does not read like something a person wrote, the")
    print("embedding will be weak. Fix the TEMPLATE, not the model. Reaching")
    print("for a bigger embedding model to compensate for word-salad input")
    print("is the most common wasted day in a RAG project.\n")

    sort_col = "n_products" if "n_products" in show.columns else "n_ingredients"
    for _, r in show.nlargest(3, sort_col).iterrows():
        print(f"\n{'-' * 76}\n[{r['doc_id']}]\n")
        print(f"  {r['text']}\n")

    singles = show[show["n_ingredients"] == 1]
    if len(singles):
        r = singles.nsmallest(1, sort_col).iloc[0]
        print(f"\n{'-' * 76}\n[{r['doc_id']}]   <- a RARE single-ingredient drug")
        print("   (checks the template still reads well with sparse data)\n")
        print(f"  {r['text']}\n")

    combos = show[show["n_ingredients"] >= 3]
    if len(combos):
        r = combos.nlargest(1, sort_col).iloc[0]
        print(f"\n{'-' * 76}\n[{r['doc_id']}]   <- a 3+ ingredient COMBINATION\n")
        print(f"  {r['text']}\n")

    print(f"\n{'=' * 76}")
    print("Next: step 3b -- why pure semantic search fails on drug names.")
    print(f"{'=' * 76}")


if __name__ == "__main__":
    main()
