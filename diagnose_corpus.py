#!/usr/bin/env python3
"""
PharmaLens - Diagnostic: why is the composition corpus incoherent?

Stage 1 produced documents like:
    "glimepiride + metformin"  ->  class: ANTI INFECTIVES
                                   uses: Fungal infections, ...

That is wrong. This script distinguishes two possible causes.

  HYPOTHESIS A - COMPOSITION KEY TOO COARSE
      Source has only short_composition1/2, but Indian combination drugs
      often have 3-4 ingredients. Products get truncated to their first
      two and wrongly pooled. Union their side effects -> nonsense.

  HYPOTHESIS B - THE id JOIN IS FAKE
      Both files may just number rows 1..N sequentially. Set overlap of
      integers proves nothing about whether id 5000 is the same medicine
      in both files. Stage 0 measured intersection, not correspondence.

Both can be true at once. Run this before rebuilding anything.

USAGE
-----
    cd P:\\PharmaLens
    python .\\diagnose_corpus.py
"""

import re
from pathlib import Path

import pandas as pd
import numpy as np

pd.set_option("display.width", 130)
pd.set_option("display.max_colwidth", 46)

_NONALPHA = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")
FORM_WORDS = {
    "tablet", "tablets", "tab", "capsule", "capsules", "cap", "syrup",
    "suspension", "injection", "inj", "cream", "ointment", "gel", "drop",
    "drops", "solution", "lotion", "powder", "sachet", "spray", "sr", "xr",
    "cr", "er", "dt", "md", "kit", "of", "mg", "ml", "mcg", "gm", "iu",
}


def norm_name(raw):
    if not isinstance(raw, str):
        return None
    s = _WS.sub(" ", _NONALPHA.sub(" ", raw.lower())).strip()
    toks = [t for t in s.split() if t not in FORM_WORDS and len(t) > 1]
    return " ".join(toks) or None


def main():
    root = Path(__file__).resolve().parent
    raw = root / "data" / "raw"

    prod = pd.read_csv(raw / "indian_medicine_data.csv", low_memory=False)
    det = pd.read_csv(raw / "medicine_details.csv", low_memory=False)

    # ==================================================================
    print(f"\n{'=' * 74}\nTEST 1: IS THE id JOIN REAL?\n{'=' * 74}")
    print("Joining on id, then asking whether the two NAMES agree.\n"
          "If they disagree often, the join is meaningless and every\n"
          "side-effect/use value is attached to the wrong medicine.\n")

    m = prod[["id", "name"]].merge(
        det[["id", "name"]], on="id", how="inner", suffixes=("_prod", "_det"))
    m["k_prod"] = m["name_prod"].map(norm_name)
    m["k_det"] = m["name_det"].map(norm_name)
    m["match"] = m["k_prod"] == m["k_det"]

    rate = m["match"].mean()
    print(f"joined rows           : {len(m):,}")
    print(f"names agree           : {m['match'].sum():,}  ({rate:.1%})")

    # Does agreement decay as id grows? That is the signature of two
    # independently-ordered scrapes drifting apart.
    print(f"\nagreement by id decile (looking for drift):")
    m["decile"] = pd.qcut(m["id"], 10, labels=False, duplicates="drop")
    for d, g in m.groupby("decile"):
        bar = "#" * int(g["match"].mean() * 40)
        print(f"  ids {g['id'].min():>7,}-{g['id'].max():>7,}  "
              f"{g['match'].mean():6.1%}  {bar}")

    print(f"\n--- VERDICT ON HYPOTHESIS B ---")
    if rate >= 0.95:
        print("id join is SOUND. Names agree. Hypothesis B is REJECTED.")
        hyp_b = False
    elif rate >= 0.60:
        print("id join is PARTIALLY sound. Some drift. Join on normalized\n"
              "NAME instead of id, and keep only confident matches.")
        hyp_b = True
    else:
        print("id join is BROKEN. The two files are independently ordered\n"
              "and id is just a row number. This alone explains the garbage.\n"
              "Rebuild Stage 1 joining on normalized name.")
        hyp_b = True

    if not m["match"].all():
        print(f"\nsample mismatches:")
        print(m.loc[~m["match"], ["id", "name_prod", "name_det"]]
              .head(10).to_string(index=False))

    # ==================================================================
    print(f"\n{'=' * 74}\nTEST 2: IS THE COMPOSITION KEY TOO COARSE?\n{'=' * 74}")
    print("If products sharing a 2-ingredient key are really 3- and\n"
          "4-ingredient drugs, their brand names will carry extra\n"
          "ingredient markers (SP, MR, Plus, Forte, D, LS...).\n")

    # Use only rows where the join is trustworthy, so this test is not
    # contaminated by whatever Test 1 found.
    good_ids = set(m.loc[m["match"], "id"]) if not m["match"].all() else set(m["id"])
    sub = prod[prod["id"].isin(good_ids)].copy()

    sub["ckey"] = (sub["short_composition1"].fillna("").str.lower()
                   .str.replace(r"\([^)]*\)", "", regex=True).str.strip()
                   + " + "
                   + sub["short_composition2"].fillna("").str.lower()
                   .str.replace(r"\([^)]*\)", "", regex=True).str.strip())
    sub["ckey"] = sub["ckey"].str.replace(r"\s*\+\s*$", "", regex=True)

    MARKERS = ["sp", "mr", "plus", "forte", "ls", "dsr", "cv", "xt",
               "od", "gm", "px", "tz", "oz", "az", "d3"]

    def has_marker(nm):
        toks = set(re.findall(r"[a-z0-9]+", str(nm).lower()))
        return bool(toks & set(MARKERS))

    sub["marker"] = sub["name"].map(has_marker)

    top = (sub.groupby("ckey")
              .agg(n_products=("id", "size"),
                   marker_share=("marker", "mean"))
              .nlargest(15, "n_products"))
    print("largest composition groups, and how many of their brand names\n"
          "carry an extra-ingredient marker:\n")
    print(top.assign(marker_share=lambda d: (d["marker_share"] * 100).round(1))
             .to_string())

    overall = sub["marker"].mean()
    big = top["marker_share"].mean()
    print(f"\nmarker rate, all products      : {overall:.1%}")
    print(f"marker rate, biggest 15 groups : {big:.1%}")

    print(f"\n--- VERDICT ON HYPOTHESIS A ---")
    if big > overall * 1.4 and big > 0.25:
        print("CONFIRMED. The biggest composition groups are enriched for\n"
              "brand-name markers indicating a 3rd/4th ingredient the source\n"
              "columns cannot hold. These groups pool DIFFERENT drugs.")
    else:
        print("NOT the dominant cause. Groups look compositionally uniform.")

    # ==================================================================
    print(f"\n{'=' * 74}\nTEST 3: HOW HETEROGENEOUS IS EACH GROUP, REALLY?\n{'=' * 74}")
    print("Within one composition key, do the products agree on their\n"
          "therapeutic class? A clean group should be near-unanimous.\n")

    j = sub.merge(det, on="id", how="inner", suffixes=("", "_d"))
    if "Therapeutic Class" in j.columns:
        agree = (j.groupby("ckey")["Therapeutic Class"]
                   .agg(lambda s: s.value_counts(normalize=True).iloc[0]
                        if s.notna().any() else np.nan)
                   .dropna())
        sizes = j.groupby("ckey").size()
        big_groups = agree[sizes.reindex(agree.index) >= 50]

        print(f"groups with >=50 products : {len(big_groups):,}")
        print(f"median class agreement    : {big_groups.median():.1%}")
        print(f"groups below 70% agreement: "
              f"{(big_groups < 0.7).sum():,} "
              f"({(big_groups < 0.7).mean():.1%})")

        print(f"\nworst 10 groups (most internally inconsistent):")
        for k, v in big_groups.nsmallest(10).items():
            classes = (j.loc[j["ckey"] == k, "Therapeutic Class"]
                        .value_counts().head(3))
            print(f"\n  {k}  (top class only {v:.0%})")
            for cls, n in classes.items():
                print(f"      {n:>6,}  {cls}")

    print(f"\n{'=' * 74}\nSend this entire output back.\n{'=' * 74}")


if __name__ == "__main__":
    main()
