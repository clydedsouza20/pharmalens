#!/usr/bin/env python3
"""
PharmaLens - Stage 1: ETL

Turns two raw Kaggle CSVs into four clean Parquet tables:

  products.parquet      one row per product, cleaned + parsed + price tier
  ingredients.parquet   long: (product_id, ingredient, strength, unit)
  compositions.parquet  one row per UNIQUE composition set  <-- RAG corpus
  manufacturers.parquet normalized manufacturer dimension

The key design decision this script implements: the retrieval unit is the
COMPOSITION, not the product. ~250k products collapse to a few thousand
distinct compositions. Embedding per-product would fill the vector store
with near-duplicates and make top-k retrieval return five brand names for
the same drug. Brands are carried as metadata instead.

USAGE
-----
    cd P:\\PharmaLens
    python .\\stage1_etl.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

pd.set_option("display.width", 130)


# ==========================================================================
# Normalization vocabulary
# ==========================================================================

# CANONICAL = the Indian / BAN term, because that is what this data uses and
# what users will type. US names are aliases for query expansion, NOT the
# canonical form. (This is reversed from the Stage 0 draft, which was built
# to match a US-named interaction database we are no longer using.)
ALIASES = {
    "acetaminophen": "paracetamol",
    "albuterol": "salbutamol",
    "rifampin": "rifampicin",
    "furosemide": "frusemide",
    "lidocaine": "lignocaine",
    "amoxicillin": "amoxycillin",
    "epinephrine": "adrenaline",
    "norepinephrine": "noradrenaline",
    "sulfate": "sulphate",
    "cetirizine": "cetrizine",
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
    "life sciences", "lifesciences", "biotech", "sciences",
]

DOSAGE_FORMS = {
    "tablet": ["tablet", "tab", "tablets"],
    "capsule": ["capsule", "cap", "capsules"],
    "syrup": ["syrup", "syp"],
    "suspension": ["suspension", "susp"],
    "injection": ["injection", "inj", "vial", "ampoule"],
    "cream": ["cream"],
    "ointment": ["ointment"],
    "gel": ["gel"],
    "drops": ["drop", "drops"],
    "solution": ["solution"],
    "lotion": ["lotion"],
    "powder": ["powder", "granules", "sachet"],
    "spray": ["spray", "inhaler", "rotacap", "respule"],
    "infusion": ["infusion"],
}
FORM_LOOKUP = {v: k for k, vs in DOSAGE_FORMS.items() for v in vs}


# ==========================================================================
# Parsers
# ==========================================================================

_PAREN = re.compile(r"\(([^)]*)\)")
_NONALPHA = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")

# "500mg" / "30mg/5ml" / "1.5 g" / "40000IU" / "2 %"
_STRENGTH = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|iu|%|meq)\b", re.IGNORECASE
)


def norm_ingredient(raw):
    """'Amoxycillin  (500mg)' -> 'amoxycillin'  (canonical Indian spelling)."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = _PAREN.sub(" ", raw.lower())
    s = _NONALPHA.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    toks = [ALIASES.get(t, t) for t in s.split()]
    while len(toks) > 1 and toks[-1] in SALT_SUFFIXES:
        toks.pop()
    return " ".join(toks).strip() or None


def parse_strength(raw):
    """Extract (value, unit) from the parenthetical. Returns (nan, None) if absent."""
    if not isinstance(raw, str):
        return np.nan, None
    m = _PAREN.search(raw)
    text = m.group(1) if m else raw
    m2 = _STRENGTH.search(text)
    if not m2:
        return np.nan, None
    val = float(m2.group(1))
    unit = m2.group(2).lower()
    # normalize to mg where meaningful
    if unit == "g":
        val, unit = val * 1000, "mg"
    elif unit in ("mcg", "ug"):
        val, unit = val / 1000, "mg"
    return val, unit


def parse_pack(raw):
    """'strip of 10 tablets' -> (10, 'tablet'); 'bottle of 100 ml Syrup' -> (100, 'syrup')."""
    if not isinstance(raw, str) or not raw.strip():
        return np.nan, "unknown"
    s = raw.lower()
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    qty = float(nums[0]) if nums else np.nan
    form = "unknown"
    for tok in re.findall(r"[a-z]+", s):
        if tok in FORM_LOOKUP:
            form = FORM_LOOKUP[tok]
            break
    return qty, form


def norm_manufacturer(raw):
    """'Glaxo SmithKline Pharmaceuticals Ltd' -> 'glaxo smithkline'."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = _NONALPHA.sub(" ", raw.lower())
    s = _WS.sub(" ", s).strip()
    changed = True
    while changed:
        changed = False
        for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
            if s.endswith(" " + suf) or s == suf:
                s = s[: -len(suf)].strip()
                changed = True
    return s or None


# ==========================================================================

def main():
    root = Path(__file__).resolve().parent
    raw = root / "data" / "raw"
    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    funnel = []

    def step(label, n):
        funnel.append((label, n))
        print(f"  {label:<52} {n:>10,}")

    # ---------------------------------------------------------------- load
    print(f"\n{'=' * 74}\nLOAD + JOIN\n{'=' * 74}")
    prod = pd.read_csv(raw / "indian_medicine_data.csv", low_memory=False)
    det = pd.read_csv(raw / "medicine_details.csv", low_memory=False)
    step("raw products", len(prod))
    step("raw detail records", len(det))

    prod = prod.rename(columns={"price(₹)": "price_inr"})
    df = prod.merge(det.drop(columns=["name"]), on="id", how="left",
                    validate="one_to_one")
    step("joined", len(df))
    step("  with detail record", df["sideEffect0"].notna().sum())

    # ------------------------------------------------------------- cleaning
    print(f"\n{'=' * 74}\nCLEANING\n{'=' * 74}")

    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    before = len(df)
    df = df[df["price_inr"].notna() & (df["price_inr"] > 0)]
    step(f"dropped non-positive / unparseable price", before - len(df))

    # Price outliers: keep, but flag. Indian pharma legitimately spans
    # Rs 2 strips to Rs 200k oncology vials -- do NOT clip blindly.
    p999 = df["price_inr"].quantile(0.999)
    df["price_outlier_flag"] = df["price_inr"] > p999
    step(f"flagged as price outlier (>p99.9 = Rs {p999:,.0f})",
         int(df["price_outlier_flag"].sum()))

    df["is_discontinued"] = df["Is_discontinued"].astype(str).str.lower().eq("true")
    step("discontinued products", int(df["is_discontinued"].sum()))

    # ---------------------------------------------------------- pack parsing
    print(f"\n{'=' * 74}\nPACK SIZE PARSING\n{'=' * 74}")
    packs = df["pack_size_label"].map(parse_pack)
    df["pack_qty"] = [p[0] for p in packs]
    df["dosage_form"] = [p[1] for p in packs]
    step("pack quantity parsed", int(df["pack_qty"].notna().sum()))
    step("dosage form identified", int((df["dosage_form"] != "unknown").sum()))
    print("\n  dosage form distribution:")
    for form, n in df["dosage_form"].value_counts().head(12).items():
        print(f"    {form:<14} {n:>9,}  ({n/len(df):5.1%})")

    # NOTE: price_per_unit is intentionally NOT created as a model feature.
    # It is derived from the target (price) and would leak. It is computed
    # here only for data-quality inspection.
    df["_price_per_unit_inspect"] = df["price_inr"] / df["pack_qty"]

    # ------------------------------------------------------- manufacturers
    print(f"\n{'=' * 74}\nMANUFACTURER NORMALIZATION\n{'=' * 74}")
    df["manufacturer_key"] = df["manufacturer_name"].map(norm_manufacturer)
    step("raw distinct manufacturers", df["manufacturer_name"].nunique())
    step("after normalization", df["manufacturer_key"].nunique())
    reduction = 1 - df["manufacturer_key"].nunique() / df["manufacturer_name"].nunique()
    print(f"  collapsed by {reduction:.1%}")

    # -------------------------------------------------------- ingredients
    print(f"\n{'=' * 74}\nINGREDIENT EXPLOSION\n{'=' * 74}")
    comp_cols = ["short_composition1", "short_composition2"]
    ing = (df[["id"] + comp_cols]
           .melt(id_vars="id", value_vars=comp_cols,
                 var_name="slot", value_name="raw")
           .dropna(subset=["raw"]))
    ing["ingredient"] = ing["raw"].map(norm_ingredient)
    ing = ing.dropna(subset=["ingredient"])
    strengths = ing["raw"].map(parse_strength)
    ing["strength"] = [s[0] for s in strengths]
    ing["strength_unit"] = [s[1] for s in strengths]

    step("ingredient mentions", len(ing))
    step("distinct ingredients", ing["ingredient"].nunique())
    step("mentions with parsed strength", int(ing["strength"].notna().sum()))

    df["n_ingredients"] = df["id"].map(ing.groupby("id").size()).fillna(0).astype(int)

    # ------------------------------------------------- COMPOSITION corpus
    print(f"\n{'=' * 74}\nCOMPOSITION-LEVEL CORPUS  (the RAG unit)\n{'=' * 74}")

    # Composition key = sorted, deduped ingredient set. Sorting matters:
    # A+B and B+A must produce the same key.
    comp_key = (ing.sort_values(["id", "ingredient"])
                   .groupby("id")["ingredient"]
                   .apply(lambda s: " + ".join(sorted(set(s)))))
    df["composition_key"] = df["id"].map(comp_key)
    df = df[df["composition_key"].notna()]

    n_comp = df["composition_key"].nunique()
    step("distinct composition sets", n_comp)
    print(f"\n  >>> DUPLICATION RATIO: {len(df) / n_comp:.1f} products "
          f"per composition")
    print(f"  >>> Embedding per-product would create ~{len(df):,} vectors")
    print(f"  >>> Embedding per-composition creates ~{n_comp:,} vectors")
    print(f"  >>> Reduction: {1 - n_comp/len(df):.1%}")

    # Aggregate detail fields up to composition level.
    se_cols = [c for c in df.columns if c.startswith("sideEffect")]
    use_cols = [c for c in df.columns if re.fullmatch(r"use\d+", c)]
    sub_cols = [c for c in df.columns if c.startswith("substitute")]

    def uniq_join(series_of_lists):
        seen, out_ = set(), []
        for lst in series_of_lists:
            for v in lst:
                if isinstance(v, str) and v.strip() and v not in seen:
                    seen.add(v)
                    out_.append(v.strip())
        return out_

    grp = df.groupby("composition_key")
    comp = pd.DataFrame({
        "n_products": grp.size(),
        "brands": grp["name"].apply(lambda s: sorted(set(s))[:50]),
        "n_brands": grp["name"].nunique(),
        "manufacturers": grp["manufacturer_key"].apply(lambda s: sorted(set(s.dropna()))[:30]),
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
        "habit_forming": grp["Habit Forming"].agg(
            lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
        "side_effects": grp[se_cols].apply(
            lambda g: uniq_join(g.values.tolist())),
        "uses": grp[use_cols].apply(
            lambda g: uniq_join(g.values.tolist())),
        "substitutes": grp[sub_cols].apply(
            lambda g: uniq_join(g.values.tolist())[:20]),
    }).reset_index()

    comp["n_side_effects"] = comp["side_effects"].str.len()
    comp["n_uses"] = comp["uses"].str.len()

    print(f"\n  composition corpus stats:")
    print(f"    side effects per composition : "
          f"mean {comp['n_side_effects'].mean():.1f}, "
          f"median {comp['n_side_effects'].median():.0f}, "
          f"max {comp['n_side_effects'].max()}")
    print(f"    uses per composition         : "
          f"mean {comp['n_uses'].mean():.1f}, "
          f"median {comp['n_uses'].median():.0f}")
    print(f"    brands per composition       : "
          f"mean {comp['n_brands'].mean():.1f}, "
          f"median {comp['n_brands'].median():.0f}, "
          f"max {comp['n_brands'].max():,}")

    # ------------------------------------------------- quality flagging
    print(f"\n{'=' * 74}\nDATA QUALITY FLAGS\n{'=' * 74}")
    comp["qa_no_therapeutic_class"] = comp["therapeutic_class"].isna()
    comp["qa_no_uses"] = comp["n_uses"] == 0
    # Implausible: very long side-effect lists on single-ingredient drugs are
    # usually scrape contamination (the 'Balila Capsule' pattern).
    comp["qa_suspicious_length"] = (
        (comp["n_side_effects"] > 25)
        & (~comp["composition_key"].str.contains(r"\+", regex=True))
    )
    for c in [c for c in comp.columns if c.startswith("qa_")]:
        step(f"{c}", int(comp[c].sum()))
    comp["qa_any"] = comp[[c for c in comp.columns if c.startswith("qa_")]].any(axis=1)
    step("compositions with ANY quality flag", int(comp["qa_any"].sum()))
    print("  (flagged, not dropped -- exclude at index time, document in README)")

    # ------------------------------------------------------ price tiers
    print(f"\n{'=' * 74}\nPRICE TIER TARGET\n{'=' * 74}")
    df["price_tier"] = pd.qcut(
        df["price_inr"], q=4,
        labels=["budget", "standard", "premium", "specialty"],
        duplicates="drop")
    print(df.groupby("price_tier", observed=True)["price_inr"]
            .agg(["count", "min", "median", "max"]).to_string())

    # ----------------------------------------------------------- write
    print(f"\n{'=' * 74}\nWRITE\n{'=' * 74}")
    keep = ["id", "name", "price_inr", "price_tier", "price_outlier_flag",
            "is_discontinued", "manufacturer_name", "manufacturer_key",
            "type", "pack_size_label", "pack_qty", "dosage_form",
            "n_ingredients", "composition_key",
            "Therapeutic Class", "Chemical Class", "Action Class",
            "Habit Forming"]
    products = df[[c for c in keep if c in df.columns]]

    manufacturers = (df.groupby("manufacturer_key")
                       .agg(n_products=("id", "size"),
                            raw_names=("manufacturer_name",
                                       lambda s: sorted(set(s))[:10]),
                            median_price=("price_inr", "median"))
                       .reset_index())

    for name, frame in [("products", products),
                        ("ingredients", ing[["id", "ingredient", "strength",
                                             "strength_unit"]]),
                        ("compositions", comp),
                        ("manufacturers", manufacturers)]:
        path = out / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        print(f"  {path.name:<26} {len(frame):>9,} rows  "
              f"({path.stat().st_size/1e6:.1f} MB)")

    # ------------------------------------------------------- funnel recap
    print(f"\n{'=' * 74}\nFUNNEL (paste this into your README)\n{'=' * 74}")
    for label, n in funnel:
        print(f"  {label:<52} {n:>10,}")

    print(f"\n{'=' * 74}\nSAMPLE COMPOSITION DOCUMENTS\n{'=' * 74}")
    sample = comp[~comp["qa_any"]].nlargest(3, "n_brands")
    for _, r in sample.iterrows():
        print(f"\n--- {r['composition_key']} ---")
        print(f"  therapeutic class : {r['therapeutic_class']}")
        print(f"  sold as           : {r['n_brands']:,} brands "
              f"(e.g. {', '.join(r['brands'][:4])})")
        print(f"  price range       : Rs {r['price_min']:,.0f} - "
              f"{r['price_max']:,.0f} (median {r['price_median']:,.0f})")
        print(f"  used for          : {', '.join(r['uses'][:5])}")
        print(f"  side effects      : {', '.join(r['side_effects'][:10])}")


if __name__ == "__main__":
    main()
