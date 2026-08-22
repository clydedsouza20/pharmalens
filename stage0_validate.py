#!/usr/bin/env python3
"""
PharmaLens - Stage 0: validation gate (two-Kaggle-dataset setup)

Answers three questions before you build anything:

  1. JOIN KEY   - do the two datasets link by `id`, or must we match on name?
  2. COVERAGE   - what fraction of products get a safety/usage record?
  3. RICHNESS   - is there enough text per drug to build a RAG corpus from?

If (2) is low, the architecture changes. If (3) is low, the document
construction strategy changes. Either way, better to know now.

LAYOUT
------
P:\\PharmaLens\\
  stage0_validate.py
  data\\raw\\
    indian_medicine_data.csv     # az-medicine-dataset-of-india
    medicine_details.csv         # 250k-medicines-usage-side-effects-and-substitutes

USAGE
-----
    cd P:\\PharmaLens
    python .\\stage0_validate.py
"""

import argparse
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

pd.set_option("display.width", 120)


# ==========================================================================
# Normalization
# ==========================================================================

SPELLING_MAP = {
    "sulphate": "sulfate",
    "amoxycillin": "amoxicillin",
    "paracetamol": "acetaminophen",
    "cetrizine": "cetirizine",
    "frusemide": "furosemide",
    "salbutamol": "albuterol",
    "lignocaine": "lidocaine",
    "rifampicin": "rifampin",
}

# Dosage forms and pack words that appear in PRODUCT names but carry no
# identity information. Stripping them is what makes name-matching work.
FORM_WORDS = {
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
    "syrup", "suspension", "injection", "inj", "cream", "ointment", "gel",
    "drop", "drops", "solution", "lotion", "powder", "sachet", "spray",
    "oral", "topical", "eye", "ear", "sr", "xr", "cr", "er", "dt", "md",
    "kit", "strip", "bottle", "tube", "vial", "of", "mg", "ml", "mcg", "gm",
    "g", "iu", "%",
}

DOSAGE_RE = re.compile(r"\([^)]*\)")
NONALPHA_RE = re.compile(r"[^a-z\s]")
WS_RE = re.compile(r"\s+")


def norm_ingredient(raw):
    """Normalize an ingredient string: 'Amoxycillin (500mg)' -> 'amoxicillin'."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = DOSAGE_RE.sub(" ", raw.lower())
    s = NONALPHA_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    toks = [SPELLING_MAP.get(t, t) for t in s.split()]
    s = " ".join(toks).strip()
    return s or None


def norm_product(raw):
    """Normalize a product/brand name by stripping dosage form + pack words.

    'Augmentin 625 Duo Tablet' -> 'augmentin duo'
    Applied to BOTH sides identically -- asymmetry here is a bug.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = NONALPHA_RE.sub(" ", raw.lower())
    s = WS_RE.sub(" ", s).strip()
    toks = [t for t in s.split() if t not in FORM_WORDS and len(t) > 1]
    return " ".join(toks) or None


# ==========================================================================
# Loading + profiling
# ==========================================================================

def profile(df, label, path):
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print(f"path    : {path}")
    print(f"shape   : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"\ncolumns ({len(df.columns)}):")
    for c in df.columns:
        nn = df[c].notna().sum()
        print(f"    {c:<28} non-null {nn:>8,}  ({nn / len(df):6.1%})")
    print(f"\nexact duplicate rows: {df.duplicated().sum():,}")
    print(f"\nfirst 3 rows:\n{df.head(3).to_string(max_colwidth=32)}")
    return df


def detect_groups(df):
    """Find wide-format column families like sideEffect0..41, use0..4."""
    groups = {}
    for c in df.columns:
        m = re.match(r"^([A-Za-z_]+?)(\d+)$", c)
        if m:
            groups.setdefault(m.group(1), []).append(c)
    return {k: sorted(v, key=lambda x: int(re.search(r"\d+$", x).group()))
            for k, v in groups.items() if len(v) > 1}


# ==========================================================================
# Gate 1 - join key
# ==========================================================================

def gate_join(prod, det):
    print(f"\n{'=' * 72}\nGATE 1: JOIN KEY\n{'=' * 72}")

    best = None

    # --- Option A: join on id ---
    if "id" in prod.columns and "id" in det.columns:
        pid, did = set(prod["id"].dropna()), set(det["id"].dropna())
        hit = len(pid & did)
        rate = hit / len(pid) if pid else 0
        print(f"\n[A] id join")
        print(f"    product ids     : {len(pid):,}")
        print(f"    detail ids      : {len(did):,}")
        print(f"    overlap         : {hit:,}  ({rate:.1%} of products)")
        if rate > 0.5:
            print("    -> id spaces align. Use this.")
            best = ("id", rate)
        else:
            print("    -> id spaces do NOT align (separate scrapes). Fall back to name.")

    # --- Option B: join on normalized name ---
    pcol = "name" if "name" in prod.columns else None
    dcol = "name" if "name" in det.columns else None
    if pcol and dcol:
        prod["_pkey"] = prod[pcol].map(norm_product)
        det["_pkey"] = det[dcol].map(norm_product)
        pk = set(prod["_pkey"].dropna())
        dk = set(det["_pkey"].dropna())
        hit = len(pk & dk)
        rate = hit / len(pk) if pk else 0
        print(f"\n[B] normalized-name join")
        print(f"    distinct product keys : {len(pk):,}")
        print(f"    distinct detail keys  : {len(dk):,}")
        print(f"    overlap               : {hit:,}  ({rate:.1%} of products)")
        if best is None or rate > best[1]:
            best = ("name", rate)

    if best is None:
        sys.exit("ERROR: neither `id` nor `name` present in both files.")

    print(f"\n--- VERDICT ---")
    key, rate = best
    print(f"best join key : {key}   coverage: {rate:.1%}")
    if rate >= 0.60:
        print("PROCEED as designed.")
    elif rate >= 0.30:
        print("PROCEED, but scope the demo to the covered subset and say so\n"
              "in the README. Partial coverage is a finding, not a failure.")
    else:
        print("RESCOPE. Too thin to join. Options: (a) join on ingredient\n"
              "instead of product, (b) treat the detail file as a standalone\n"
              "RAG corpus keyed by ingredient rather than by product.")
    return key


# ==========================================================================
# Gate 2 - corpus richness
# ==========================================================================

def gate_richness(det):
    print(f"\n{'=' * 72}\nGATE 2: RAG CORPUS RICHNESS\n{'=' * 72}")
    groups = detect_groups(det)
    if not groups:
        print("No wide-format column families detected. Inspect columns manually.")
        return

    print("Detected wide-format families (these become your document text):\n")
    total_chars = pd.Series(0, index=det.index, dtype=int)

    for fam, cols in groups.items():
        filled = det[cols].notna().sum(axis=1)
        vals = det[cols].astype(str).where(det[cols].notna(), "")
        chars = vals.apply(lambda r: sum(len(x) for x in r), axis=1)
        total_chars += chars
        print(f"  {fam:<16} {len(cols):>3} cols | "
              f"avg filled/drug {filled.mean():5.2f} | "
              f"max {filled.max():>3} | "
              f"drugs with >=1: {(filled > 0).mean():6.1%}")

    print(f"\nestimated document length (chars per drug):")
    print(f"  mean   {total_chars.mean():8.0f}")
    print(f"  median {total_chars.median():8.0f}")
    print(f"  p90    {total_chars.quantile(0.90):8.0f}")
    print(f"  max    {total_chars.max():8.0f}")

    med = total_chars.median()
    print(f"\n--- VERDICT ---")
    if med >= 400:
        print("Rich enough. One document per drug; NO chunking needed.\n"
              "Chunking here would split a drug away from its own name.")
    elif med >= 150:
        print("Moderate. One document per drug, no chunking. Consider enriching\n"
              "each document with therapeutic/chemical class text.")
    else:
        print("Thin. Aggregate UP instead: build documents per ingredient or per\n"
              "therapeutic class, pooling all products that share it. Retrieval\n"
              "over 40k rich docs beats retrieval over 250k one-line docs.")

    # Show a constructed document so you can eyeball what you'd embed.
    print(f"\n{'-' * 72}\nSAMPLE CONSTRUCTED DOCUMENT (what you would embed)\n{'-' * 72}")
    row = det.loc[total_chars.idxmax()]
    name = row.get("name", "<unknown>")
    print(f"\n{name}\n")
    for fam, cols in groups.items():
        vals = [str(row[c]) for c in cols if pd.notna(row[c])]
        if vals:
            print(f"  {fam}: {', '.join(vals[:12])}"
                  f"{' ...' if len(vals) > 12 else ''}")


# ==========================================================================
# Gate 3 - ingredient layer (the seam)
# ==========================================================================

def gate_ingredients(prod):
    print(f"\n{'=' * 72}\nGATE 3: INGREDIENT LAYER\n{'=' * 72}")
    comp_cols = [c for c in prod.columns if c.lower().startswith("short_composition")]
    if not comp_cols:
        print("No short_composition* columns found; skipping.")
        return

    long = (prod[["id"] + comp_cols]
            .melt(id_vars="id", value_vars=comp_cols, value_name="ingredient")
            .dropna(subset=["ingredient"]))
    long["key"] = long["ingredient"].map(norm_ingredient)
    long = long.dropna(subset=["key"])

    per_prod = long.groupby("id").size()
    print(f"ingredient mentions      : {len(long):,}")
    print(f"distinct ingredients     : {long['key'].nunique():,}")
    print(f"single-ingredient products: {(per_prod == 1).sum():,} "
          f"({(per_prod == 1).mean():.1%})")
    print(f"combination products      : {(per_prod > 1).sum():,} "
          f"({(per_prod > 1).mean():.1%})")

    print(f"\ntop 25 ingredients by product count:")
    for k, n in Counter(long["key"]).most_common(25):
        print(f"  {n:>7,}  {k}")


# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--product-file", default="indian_medicine_data.csv")
    ap.add_argument("--detail-file", default="medicine_details.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir \
        else (root / "data" / "raw")

    ppath = data_dir / args.product_file
    dpath = data_dir / args.detail_file

    for p in (ppath, dpath):
        if not p.exists():
            seen = sorted(x.name for x in data_dir.glob("*.csv")) \
                if data_dir.exists() else []
            sys.exit(
                f"\nERROR: missing {p}\n"
                f"  data dir exists : {data_dir.exists()}\n"
                f"  CSVs seen there : {seen or 'none'}\n"
                f"  Fix with --data-dir / --product-file / --detail-file\n"
            )

    prod = profile(pd.read_csv(ppath, low_memory=False),
                   "DATASET 1: INDIAN PRODUCTS (price, manufacturer, composition)",
                   ppath)
    det = profile(pd.read_csv(dpath, low_memory=False),
                  "DATASET 2: MEDICINE DETAILS (uses, side effects, substitutes)",
                  dpath)

    gate_join(prod, det)
    gate_richness(det)
    gate_ingredients(prod)

    print(f"\n{'=' * 72}\nStage 0 complete. Send this whole output back.\n{'=' * 72}")


if __name__ == "__main__":
    main()
