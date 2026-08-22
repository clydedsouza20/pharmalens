#!/usr/bin/env python3
"""
PharmaLens - Stage 1 v2: ETL with a REAL join

WHAT CHANGED FROM v1
--------------------
v1 joined the two Kaggle files on `id`. The diagnostic proved that join is
fake: only 86 of 248,218 joined rows had matching names (0.03%). `id` is a
per-file row index, not a shared key, and because the detail file has 5,755
fewer rows the two files drift apart cumulatively down the file.

    details[76] = "aztor 10"        products[77] = "Aztor 10 Tablet"
    details[77] = "atorva 40"       products[78] = "Atorva 40 Tablet"

v2 joins on a normalized product NAME instead, and validates the result
before building anything on top of it.

WHAT THIS SCRIPT PROVES BEFORE IT TRUSTS ITSELF
-----------------------------------------------
  A. name-join coverage, for two competing key designs (with/without digits)
  B. how many keys are ambiguous (one key -> several detail rows)
  C. therapeutic-class agreement WITHIN each composition group
     (this is the Hypothesis A retest the broken diagnostic could not do)

If (C) is high, the composition corpus is sound and Stage 3 can proceed.

USAGE
-----
    cd P:\\PharmaLens
    python .\\stage1_etl_v2.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 130)
pd.set_option("display.max_colwidth", 44)


# ==========================================================================
# Vocabulary
# ==========================================================================

ALIASES = {
    "acetaminophen": "paracetamol", "albuterol": "salbutamol",
    "rifampin": "rifampicin", "furosemide": "frusemide",
    "lidocaine": "lignocaine", "amoxicillin": "amoxycillin",
    "epinephrine": "adrenaline", "norepinephrine": "noradrenaline",
    "sulfate": "sulphate", "cetirizine": "cetrizine",
}

SALT_SUFFIXES = {
    "hydrochloride", "hcl", "sodium", "potassium", "sulphate", "sulfate",
    "phosphate", "maleate", "tartrate", "citrate", "acetate", "succinate",
    "besylate", "mesylate", "fumarate", "nitrate", "carbonate", "oxide",
    "dihydrate", "monohydrate", "anhydrous", "trihydrate", "bromide",
}

LEGAL_SUFFIXES = [
    "pvt ltd", "private limited", "ltd", "limited", "llp", "inc",
    "corporation", "corp", "co", "company", "pharmaceuticals",
    "pharmaceutical", "pharma", "laboratories", "laboratory", "labs", "lab",
    "healthcare", "health care", "industries", "india", "remedies",
    "life sciences", "lifesciences", "biotech", "sciences", "formulations",
]

# Words that describe the PACKAGE, not the medicine. Removing them is what
# makes two spellings of the same product collapse to one key.
FORM_WORDS = {
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
    "syrup", "syp", "suspension", "susp", "injection", "inj", "cream",
    "ointment", "gel", "drop", "drops", "solution", "lotion", "powder",
    "sachet", "spray", "vial", "ampoule", "tube", "bottle", "strip", "kit",
    "oral", "topical", "of", "the", "and", "with",
}

_PAREN = re.compile(r"\(([^)]*)\)")
_NONALNUM = re.compile(r"[^a-z0-9\s]")
_NONALPHA = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")
_STRENGTH = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|iu|%|meq)\b", re.I)

DOSAGE_FORMS = {
    "tablet": ["tablet", "tab", "tablets"],
    "capsule": ["capsule", "cap", "capsules"],
    "syrup": ["syrup", "syp"], "suspension": ["suspension", "susp"],
    "injection": ["injection", "inj", "vial", "ampoule"],
    "cream": ["cream"], "ointment": ["ointment"], "gel": ["gel"],
    "drops": ["drop", "drops"], "solution": ["solution"],
    "lotion": ["lotion"], "powder": ["powder", "granules", "sachet"],
    "spray": ["spray", "inhaler", "rotacap", "respule"],
    "infusion": ["infusion"],
}
FORM_LOOKUP = {v: k for k, vs in DOSAGE_FORMS.items() for v in vs}


# ==========================================================================
# Keys
# ==========================================================================

def name_key(raw, keep_digits=True):
    """Normalize a product name into a join key.

    keep_digits=True  : 'Azithral 500 Tablet' -> 'azithral 500'
        Stricter. Keeps strength, so 500mg and 250mg stay distinct.
    keep_digits=False : 'Azithral 500 Tablet' -> 'azithral'
        Looser. Higher match rate, but silently merges strengths.

    We measure both and choose on evidence, not preference.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    pat = _NONALNUM if keep_digits else _NONALPHA
    s = _WS.sub(" ", pat.sub(" ", raw.lower())).strip()
    toks = [t for t in s.split() if t not in FORM_WORDS and len(t) > 1]
    return " ".join(toks) or None


def norm_ingredient(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = _PAREN.sub(" ", raw.lower())
    s = _WS.sub(" ", _NONALPHA.sub(" ", s)).strip()
    toks = [ALIASES.get(t, t) for t in s.split()]
    while len(toks) > 1 and toks[-1] in SALT_SUFFIXES:
        toks.pop()
    return " ".join(toks).strip() or None


def parse_strength(raw):
    if not isinstance(raw, str):
        return np.nan, None
    m = _PAREN.search(raw)
    m2 = _STRENGTH.search(m.group(1) if m else raw)
    if not m2:
        return np.nan, None
    val, unit = float(m2.group(1)), m2.group(2).lower()
    if unit == "g":
        val, unit = val * 1000, "mg"
    elif unit in ("mcg", "ug"):
        val, unit = val / 1000, "mg"
    return val, unit


def parse_pack(raw):
    if not isinstance(raw, str) or not raw.strip():
        return np.nan, "unknown"
    s = raw.lower()
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    qty = float(nums[0]) if nums else np.nan
    form = next((FORM_LOOKUP[t] for t in re.findall(r"[a-z]+", s)
                 if t in FORM_LOOKUP), "unknown")
    return qty, form


def norm_manufacturer(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = _WS.sub(" ", _NONALPHA.sub(" ", raw.lower())).strip()
    changed = True
    while changed:
        changed = False
        for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
            if s.endswith(" " + suf) or s == suf:
                s, changed = s[: -len(suf)].strip(), True
    return s or None


# ==========================================================================

def main():
    root = Path(__file__).resolve().parent
    raw_dir = root / "data" / "raw"
    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    funnel = []

    def step(label, n):
        funnel.append((label, n))
        print(f"  {label:<54} {n:>10,}")

    # ------------------------------------------------------------- load
    print(f"\n{'=' * 76}\n1. LOAD\n{'=' * 76}")
    prod = pd.read_csv(raw_dir / "indian_medicine_data.csv", low_memory=False)
    det = pd.read_csv(raw_dir / "medicine_details.csv", low_memory=False)
    prod = prod.rename(columns={"price(₹)": "price_inr"})
    step("raw products", len(prod))
    step("raw detail records", len(det))

    # The old id column is NOT a shared key. Rename so it can never be
    # mistaken for one again.
    prod = prod.rename(columns={"id": "product_row_id"})
    det = det.rename(columns={"id": "detail_row_id"})

    # -------------------------------------------------- A. choose the key
    print(f"\n{'=' * 76}\n2. CHOOSE THE JOIN KEY (measure, don't guess)\n{'=' * 76}")
    results = {}
    for keep in (True, False):
        pk = prod["name"].map(lambda x: name_key(x, keep))
        dk = det["name"].map(lambda x: name_key(x, keep))
        matched = pk.isin(set(dk.dropna())).sum()
        ambig = dk.dropna().duplicated().sum()
        results[keep] = (matched / len(prod), ambig)
        label = "keep digits (strict)" if keep else "drop digits (loose)"
        print(f"  {label:<24} coverage {matched/len(prod):6.1%}   "
              f"ambiguous detail keys {ambig:,}")

    strict_cov, _ = results[True]
    loose_cov, _ = results[False]
    # Prefer strict unless it costs more than 10 points of coverage.
    keep_digits = strict_cov >= loose_cov - 0.10
    print(f"\n  -> using {'STRICT (digits kept)' if keep_digits else 'LOOSE (digits dropped)'}")
    print("     strict preserves strength distinctions; we accept it unless")
    print("     it costs more than 10 points of coverage.")

    prod["join_key"] = prod["name"].map(lambda x: name_key(x, keep_digits))
    det["join_key"] = det["name"].map(lambda x: name_key(x, keep_digits))

    # -------------------------------------------------- B. resolve ambiguity
    print(f"\n{'=' * 76}\n3. JOIN\n{'=' * 76}")
    det = det[det["join_key"].notna()].copy()

    # One key may map to several detail rows. Keep the most complete row --
    # the one with the fewest nulls -- and record how often this happened.
    det["_completeness"] = det.notna().sum(axis=1)
    n_before = len(det)
    det = (det.sort_values("_completeness", ascending=False)
              .drop_duplicates(subset="join_key", keep="first")
              .drop(columns="_completeness"))
    step("detail rows collapsed by ambiguous key", n_before - len(det))
    step("usable detail records", len(det))

    df = prod.merge(det.drop(columns=["name"]), on="join_key", how="left")
    matched = df["sideEffect0"].notna()
    step("products", len(df))
    step("  matched to a detail record", int(matched.sum()))
    print(f"  match rate: {matched.mean():.1%}")

    if matched.mean() < 0.50:
        print("\n  WARNING: match rate below 50%. Inspect before continuing.")

    print(f"\n  sanity check -- 8 matched products, names side by side:")
    chk = df[matched].head(8)
    det_names = det.set_index("join_key")["name"]
    for _, r in chk.iterrows():
        print(f"    {r['name'][:40]:<42} | {det_names.get(r['join_key'], '?')[:40]}")

    # ---------------------------------------------------------- cleaning
    print(f"\n{'=' * 76}\n4. CLEAN\n{'=' * 76}")
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    n0 = len(df)
    df = df[df["price_inr"].notna() & (df["price_inr"] > 0)]
    step("dropped bad price", n0 - len(df))

    p999 = df["price_inr"].quantile(0.999)
    df["price_outlier_flag"] = df["price_inr"] > p999
    step(f"flagged price outlier (>Rs {p999:,.0f})", int(df["price_outlier_flag"].sum()))

    df["is_discontinued"] = df["Is_discontinued"].astype(str).str.lower().eq("true")
    step("discontinued", int(df["is_discontinued"].sum()))

    packs = df["pack_size_label"].map(parse_pack)
    df["pack_qty"] = [p[0] for p in packs]
    df["dosage_form"] = [p[1] for p in packs]
    step("pack qty parsed", int(df["pack_qty"].notna().sum()))

    df["manufacturer_key"] = df["manufacturer_name"].map(norm_manufacturer)
    step("distinct manufacturers (normalized)", df["manufacturer_key"].nunique())

    # ------------------------------------------------------- ingredients
    print(f"\n{'=' * 76}\n5. INGREDIENTS\n{'=' * 76}")
    comp_cols = ["short_composition1", "short_composition2"]
    ing = (df[["product_row_id"] + comp_cols]
           .melt(id_vars="product_row_id", value_vars=comp_cols,
                 var_name="slot", value_name="raw")
           .dropna(subset=["raw"]))
    ing["ingredient"] = ing["raw"].map(norm_ingredient)
    ing = ing.dropna(subset=["ingredient"])
    st = ing["raw"].map(parse_strength)
    ing["strength"] = [s[0] for s in st]
    ing["strength_unit"] = [s[1] for s in st]
    step("ingredient mentions", len(ing))
    step("distinct ingredients", ing["ingredient"].nunique())

    ckey = (ing.groupby("product_row_id")["ingredient"]
               .apply(lambda s: " + ".join(sorted(set(s)))))
    df["composition_key"] = df["product_row_id"].map(ckey)
    df = df[df["composition_key"].notna()]
    df["n_ingredients"] = df["composition_key"].str.count(r"\+") + 1

    # ============ HYPOTHESIS A RETEST (impossible before the real join) ==
    print(f"\n{'=' * 76}\n6. IS THE COMPOSITION GROUPING SOUND?\n{'=' * 76}")
    print("Within one composition, do products agree on therapeutic class?\n"
          "A clean group is near-unanimous. This is the test the earlier\n"
          "diagnostic could not run, because it only had 86 valid rows.\n")

    md = df[matched.reindex(df.index, fill_value=False)]
    if "Therapeutic Class" in md.columns and len(md):
        g = md.groupby("composition_key")["Therapeutic Class"]
        agree = g.agg(lambda s: s.value_counts(normalize=True).iloc[0]
                      if s.notna().any() else np.nan).dropna()
        sizes = md.groupby("composition_key").size()
        big = agree[sizes.reindex(agree.index) >= 20]

        print(f"  composition groups with >=20 matched products : {len(big):,}")
        if len(big):
            print(f"  median class agreement                        : {big.median():.1%}")
            print(f"  groups below 70% agreement                    : "
                  f"{(big < 0.7).sum():,} ({(big < 0.7).mean():.1%})")
            print(f"\n  --- VERDICT ---")
            if big.median() >= 0.85:
                print("  SOUND. Composition groups are internally consistent.")
                print("  Hypothesis A rejected. Proceed to Stage 3.")
            elif big.median() >= 0.65:
                print("  MOSTLY SOUND. Some groups pool related drugs. Exclude")
                print("  low-agreement groups at index time and note it.")
            else:
                print("  STILL BROKEN. Two composition columns cannot represent")
                print("  3-4 ingredient products. Fall back to per-INGREDIENT")
                print("  documents instead of per-composition.")
            print(f"\n  worst 5 groups:")
            for k, v in big.nsmallest(5).items():
                top3 = md.loc[md['composition_key'] == k,
                              'Therapeutic Class'].value_counts().head(3)
                print(f"    {k}  (top class {v:.0%})")
                for cls, n in top3.items():
                    print(f"        {n:>5,}  {cls}")

    # --------------------------------------------------- build the corpus
    print(f"\n{'=' * 76}\n7. COMPOSITION CORPUS\n{'=' * 76}")
    se = [c for c in df.columns if c.startswith("sideEffect")]
    us = [c for c in df.columns if re.fullmatch(r"use\d+", c)]
    sb = [c for c in df.columns if c.startswith("substitute")]

    def consensus(frame, min_share=0.10, min_count=2):
        """Keep values appearing in at least `min_share` of the group's rows.

        Union lets one mislabeled product inject garbage into a document
        shared by thousands. A real indication appears across many products;
        a scrape error appears once. Returns [(value, share)].
        """
        n = len(frame)
        counts = frame.stack().value_counts()
        threshold = max(min_count, n * min_share)
        kept = counts[counts >= threshold]
        # Never return nothing: an empty document is unretrievable. If the
        # threshold filters everything out, keep the single most common value
        # so the composition still has some text.
        if kept.empty and not counts.empty:
         kept = counts.head(1)
        return [(v, round(c / n, 3)) for v, c in kept.items()]

    grp = df.groupby("composition_key")
    comp = pd.DataFrame({
        "n_products": grp.size(),
        "n_brands": grp["name"].nunique(),
        "brands": grp["name"].apply(lambda s: sorted(set(s))[:50]),
        "price_min": grp["price_inr"].min(),
        "price_median": grp["price_inr"].median(),
        "price_max": grp["price_inr"].max(),
        "dosage_forms": grp["dosage_form"].apply(lambda s: sorted(set(s))),
        "therapeutic_class": grp["Therapeutic Class"].agg(
            lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
        "chemical_class": grp["Chemical Class"].agg(
            lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
        "action_class": grp["Action Class"].agg(
            lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
        "class_agreement": grp["Therapeutic Class"].agg(
            lambda s: s.value_counts(normalize=True).iloc[0]
            if s.notna().any() else np.nan),
        "side_effects": grp[se].apply(lambda g: consensus(g, min_share=0.10)),
        "uses": grp[us].apply(lambda g: consensus(g, min_share=0.15)),
        "substitutes": grp[sb].apply(lambda g: consensus(g, min_share=0.05)[:20]),
    }).reset_index()
    # Parquet cannot store lists of tuples. Split each [(value, share)]
    # column into parallel value / share columns. str() also guards against
    # stray non-string cells in the source data.
    for col in ["side_effects", "uses", "substitutes"]:
        pairs = comp[col]
        comp[col] = pairs.map(lambda ps: [str(v) for v, _ in ps])
        comp[f"{col}_share"] = pairs.map(lambda ps: [float(s) for _, s in ps])

    comp["n_side_effects"] = comp["side_effects"].str.len()
    comp["n_uses"] = comp["uses"].str.len()

    step("distinct compositions", len(comp))
    print(f"  products per composition: median {comp['n_products'].median():.0f}, "
          f"mean {comp['n_products'].mean():.1f}, max {comp['n_products'].max():,}")
    print(f"  side effects  : median {comp['n_side_effects'].median():.0f}, "
          f"max {comp['n_side_effects'].max()}")
    print(f"  uses          : median {comp['n_uses'].median():.0f}, "
          f"max {comp['n_uses'].max()}")
    print("  (real drugs have ~5-15 side effects and 1-3 uses -- if the")
    print("   medians are far above that, grouping is still too coarse)")

    comp["qa_low_agreement"] = comp["class_agreement"] < 0.7
    comp["qa_no_uses"] = comp["n_uses"] == 0
    comp["qa_too_many_effects"] = comp["n_side_effects"] > 25
    comp["qa_any"] = comp[[c for c in comp if c.startswith("qa_")]].any(axis=1)
    step("compositions with a quality flag", int(comp["qa_any"].sum()))

    # ----------------------------------------------------------- targets
    df["price_tier"] = pd.qcut(df["price_inr"], 4,
                               labels=["budget", "standard", "premium", "specialty"],
                               duplicates="drop")

    # ------------------------------------------------------------- write
    print(f"\n{'=' * 76}\n8. WRITE\n{'=' * 76}")
    keep_cols = ["product_row_id", "name", "join_key", "price_inr", "price_tier",
                 "price_outlier_flag", "is_discontinued", "manufacturer_name",
                 "manufacturer_key", "type", "pack_size_label", "pack_qty",
                 "dosage_form", "n_ingredients", "composition_key",
                 "Therapeutic Class", "Chemical Class", "Action Class",
                 "Habit Forming"]
    products = df[[c for c in keep_cols if c in df.columns]]
    manufacturers = (df.groupby("manufacturer_key")
                       .agg(n_products=("product_row_id", "size"),
                            median_price=("price_inr", "median"))
                       .reset_index())

    for nm, frame in [("products", products),
                      ("ingredients", ing.rename(columns={"product_row_id": "product_row_id"})[
                          ["product_row_id", "ingredient", "strength", "strength_unit"]]),
                      ("compositions", comp),
                      ("manufacturers", manufacturers)]:
        p = out / f"{nm}.parquet"
        frame.to_parquet(p, index=False)
        print(f"  {p.name:<26} {len(frame):>9,} rows  ({p.stat().st_size/1e6:.1f} MB)")

    # -------------------------------------------------------- inspection
    print(f"\n{'=' * 76}\n9. SAMPLE DOCUMENTS -- READ THESE\n{'=' * 76}")
    print("If a diabetes drug is tagged ANTI INFECTIVES again, stop.\n")
    for _, r in comp[~comp["qa_any"]].nlargest(4, "n_products").iterrows():
        print(f"\n--- {r['composition_key']} ---")
        print(f"  class      : {r['therapeutic_class']}  "
              f"(agreement {r['class_agreement']:.0%})")
        print(f"  brands     : {r['n_brands']:,}")
        print(f"  price      : Rs {r['price_min']:,.0f} - {r['price_max']:,.0f}")
        print(f"  uses       : " + ", ".join(
            f"{v} ({s:.0%})" for v, s in
            zip(r['uses'][:4], r['uses_share'][:4])))
        print(f"  effects    : " + ", ".join(
            f"{v} ({s:.0%})" for v, s in
            zip(r['side_effects'][:8], r['side_effects_share'][:8])))

    print(f"\n{'=' * 76}\nFUNNEL\n{'=' * 76}")
    for label, n in funnel:
        print(f"  {label:<54} {n:>10,}")


if __name__ == "__main__":
    main()
