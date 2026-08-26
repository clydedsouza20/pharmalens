#!/usr/bin/env python3
"""
PharmaLens - Stage 2: price tier classifier

QUESTION
--------
From a medicine's ingredients, maker, and pack -- but NOT its price --
can we predict which price bracket it falls into?

WHY IT MATTERS
--------------
Two reasons, neither of which is "because ML is impressive":

  1. It validates stage 1. If a model cannot beat random guessing on
     cleaned data, the cleaning lost the signal.
  2. SHAP turns the model into a STATEMENT about Indian pharmaceutical
     pricing: does the manufacturer or the molecule drive price? That is
     a finding, and findings are what people remember.

THE ONE RULE
------------
No feature may be derived from price. Not price_inr, and not
price-per-unit -- that is price divided by pack size, so the answer is
baked in. The model would score 99% and mean nothing. Leakage is the
most common way a portfolio ML project is quietly wrong.

TIERS (from stage 1, quartiles of price_inr)
--------------------------------------------
    budget      Rs 1 - 48        64,581
    standard    Rs 48 - 79       63,080
    premium     Rs 79 - 140      62,980
    specialty   Rs 140 - 436,000 63,328

Roughly balanced, so macro-F1 is a meaningful metric.

USAGE
-----
    cd P:\\PharmaLens
    pip install xgboost shap matplotlib
    python .\\stage2_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # write files, don't try to open windows
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (f1_score, accuracy_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier


# Everything here is knowable BEFORE the price is known.
FEATURES = ["composition_key",   # what is in it      (high card -> target enc)
            "manufacturer_key",  # who makes it       (high card -> target enc)
            "dosage_form",       # tablet / syrup     (low card  -> one-hot)
            "type",              # allopathy etc.     (low card  -> one-hot)
            "pack_qty",          # numeric
            "n_ingredients"]     # numeric
HIGH_CARD = ["composition_key", "manufacturer_key"]
LOW_CARD = ["dosage_form", "type"]
NUMERIC = ["pack_qty", "n_ingredients"]
TARGET = "price_tier"
TIERS = ["budget", "standard", "premium", "specialty"]


# ==========================================================================
# Target encoding, done without leaking
# ==========================================================================

def fit_target_encoding(X_tr, y_tr, col, n_splits=5, smoothing=20):
    """Learn a category -> mean-tier mapping from TRAINING DATA ONLY.

    Two protections, and both matter:

    1. OUT-OF-FOLD encoding for the training rows. If a category with one
       product were encoded using that product's own row, the encoded
       value would BE the answer and the model would memorise rather than
       learn. Each row is therefore encoded using the other folds only.

    2. SMOOTHING. A category seen twice should not be trusted as much as
       one seen 2,000 times. Small categories are pulled toward the global
       mean; `smoothing` is roughly how many observations a category needs
       before we mostly believe its own average.

    Returns (out-of-fold values for training, mapping for test, global mean).
    """
    global_mean = y_tr.mean()
    oof = np.full(len(X_tr), np.nan)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X_tr, y_tr):
        stats = (pd.DataFrame({"c": X_tr.iloc[tr_idx][col].values,
                               "y": y_tr.iloc[tr_idx].values})
                 .groupby("c")["y"].agg(["mean", "count"]))
        enc = ((stats["mean"] * stats["count"] + global_mean * smoothing)
               / (stats["count"] + smoothing))
        oof[va_idx] = X_tr.iloc[va_idx][col].map(enc).values

    stats = (pd.DataFrame({"c": X_tr[col].values, "y": y_tr.values})
             .groupby("c")["y"].agg(["mean", "count"]))
    mapping = ((stats["mean"] * stats["count"] + global_mean * smoothing)
               / (stats["count"] + smoothing))

    return np.nan_to_num(oof, nan=global_mean), mapping, global_mean


def encode(X_tr, X_te, y_tr):
    """Build the numeric matrices. Fitted on train, applied to test."""
    tr_parts, te_parts = [], []

    for col in HIGH_CARD:
        oof, mapping, gmean = fit_target_encoding(X_tr, y_tr, col)
        tr_parts.append(pd.Series(oof, index=X_tr.index, name=f"te_{col}"))
        te_parts.append(X_te[col].map(mapping).fillna(gmean).rename(f"te_{col}"))

    # Frequency as a second signal: how common is this category at all?
    # A manufacturer with 3 products behaves differently from one with 8,000.
    for col in HIGH_CARD:
        freq = X_tr[col].value_counts()
        tr_parts.append(X_tr[col].map(freq).fillna(0).rename(f"freq_{col}"))
        te_parts.append(X_te[col].map(freq).fillna(0).rename(f"freq_{col}"))

    d_tr = pd.get_dummies(X_tr[LOW_CARD], prefix=LOW_CARD)
    d_te = pd.get_dummies(X_te[LOW_CARD], prefix=LOW_CARD)
    d_te = d_te.reindex(columns=d_tr.columns, fill_value=0)
    tr_parts.append(d_tr)
    te_parts.append(d_te)

    tr_parts.append(X_tr[NUMERIC].fillna(-1))
    te_parts.append(X_te[NUMERIC].fillna(-1))

    return pd.concat(tr_parts, axis=1), pd.concat(te_parts, axis=1)


# ==========================================================================

def main():
    root = Path(__file__).resolve().parent
    proc = root / "data" / "processed"
    figs = root / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------- 1. load
    print(f"\n{'=' * 74}\n1. LOAD\n{'=' * 74}")
    df = pd.read_parquet(proc / "products.parquet")
    df = df[df[TARGET].notna()].copy()

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"\nERROR: missing columns {missing}\n"
                         f"available: {list(df.columns)}\n"
                         f"Re-run stage1_etl.py.\n")

    print(f"  rows: {len(df):,}")
    print(f"\n  tier balance:")
    for tier, n in df[TARGET].value_counts().reindex(TIERS).items():
        print(f"    {tier:<12} {n:>8,}")
    print(f"\n  cardinality:")
    for c in HIGH_CARD + LOW_CARD:
        print(f"    {c:<20} {df[c].nunique():>8,} distinct")
    print("\n  One-hot on the high-cardinality columns would produce")
    print("  ~10,000 features. Target encoding instead.")

    # ------------------------------------------------------- 2. split
    print(f"\n{'=' * 74}\n2. SPLIT  (before encoding -- the order matters)\n{'=' * 74}")
    X = df[FEATURES]
    y = pd.Series(pd.Categorical(df[TARGET], categories=TIERS).codes,
                  index=df.index, name="y")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"  train {len(X_tr):,}   test {len(X_te):,}")
    print("\n  Encoding is fitted AFTER the split, on training data only.")
    print("  Fitting it on everything would leak test information into")
    print("  the encoded values -- a subtle and very common mistake.")

    # ------------------------------------------------------- 3. baseline
    print(f"\n{'=' * 74}\n3. BASELINE  (the floor to beat)\n{'=' * 74}")
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_tr[NUMERIC], y_tr)
    d_pred = dummy.predict(X_te[NUMERIC])
    base_f1 = f1_score(y_te, d_pred, average="macro")
    print(f"  random-guess macro-F1 : {base_f1:.3f}")
    print(f"  random-guess accuracy : {accuracy_score(y_te, d_pred):.3f}")
    print("\n  '0.62 F1' means nothing alone. '0.62 against a 0.25 floor'")
    print("  means something. Always establish the floor.")

    # ------------------------------------------------------- 4. encode
    print(f"\n{'=' * 74}\n4. ENCODE\n{'=' * 74}")
    X_tr_enc, X_te_enc = encode(X_tr, X_te, y_tr)
    print(f"  {X_tr_enc.shape[1]} features:")
    for c in X_tr_enc.columns:
        print(f"    {c}")

    leaked = [c for c in X_tr_enc.columns if "price" in c.lower()]
    print(f"\n  leakage check: {'FAILED -> ' + str(leaked) if leaked else 'no price-derived features'}")

    # ------------------------------------------------------- 5. train
    print(f"\n{'=' * 74}\n5. TRAIN\n{'=' * 74}")
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", num_class=len(TIERS),
        n_jobs=-1, random_state=42, eval_metric="mlogloss")
    model.fit(X_tr_enc, y_tr, eval_set=[(X_te_enc, y_te)], verbose=False)
    print("  done")

    # ------------------------------------------------------- 6. evaluate
    print(f"\n{'=' * 74}\n6. EVALUATE\n{'=' * 74}")
    pred = model.predict(X_te_enc)
    f1 = f1_score(y_te, pred, average="macro")
    acc = accuracy_score(y_te, pred)

    print(f"  macro-F1 : {f1:.3f}   (baseline {base_f1:.3f}, "
          f"lift {f1 - base_f1:+.3f})")
    print(f"  accuracy : {acc:.3f}")
    print(f"\n{classification_report(y_te, pred, target_names=TIERS, digits=3)}")

    cm = confusion_matrix(y_te, pred)
    print("  confusion matrix (rows actual, cols predicted):")
    print(pd.DataFrame(cm, index=TIERS, columns=TIERS).to_string())

    err = cm.copy()
    np.fill_diagonal(err, 0)
    total = err.sum()
    adjacent = sum(err[i, j] for i in range(4) for j in range(4)
                   if abs(i - j) == 1)
    print(f"\n  of {total:,} errors: {adjacent / total:.1%} adjacent-tier, "
          f"{1 - adjacent / total:.1%} distant")
    print("  Confusing budget with standard is forgivable -- those tiers")
    print("  genuinely blur at the boundary. Confusing budget with")
    print("  specialty would mean something is broken.")

    # ------------------------------------------------------- 7. shap
    print(f"\n{'=' * 74}\n7. SHAP -- what did the model learn?\n{'=' * 74}")
    try:
        import shap
        sample = X_te_enc.sample(min(2000, len(X_te_enc)), random_state=42)
        sv = shap.TreeExplainer(model).shap_values(sample)

               # SHAP's multiclass shape varies by version: older releases return
        # a list of (n_samples, n_features) arrays, one per class; 0.4x+
        # returns a single (n_samples, n_features, n_classes) array.
        # Average |SHAP| over every axis EXCEPT the feature axis, which is
        # always axis 1.
        arr = np.array(sv)
        if arr.ndim == 3 and arr.shape[1] != len(sample.columns):
         arr = np.moveaxis(arr, 0, -1)      # old list-of-classes layout
        axes = tuple(i for i in range(arr.ndim) if i != 1)
        imp = np.abs(arr).mean(axis=axes)

        if len(imp) != len(sample.columns):
            raise ValueError(
                f"SHAP returned {len(imp)} values for "
                f"{len(sample.columns)} features; shape was {arr.shape}")

        ranking = pd.Series(imp, index=sample.columns).sort_values(ascending=False)

        print("  mean |SHAP| by feature:")
        for name, val in ranking.items():
            print(f"    {name:<28} {val:.4f}")

        te_manu = ranking.get("te_manufacturer_key", 0)
        te_comp = ranking.get("te_composition_key", 0)
        print(f"\n  manufacturer {te_manu:.4f}  vs  composition {te_comp:.4f}")
        print(f"\n  --- THE FINDING ---")
        if te_manu > te_comp * 1.15:
            print("  WHO MAKES IT outweighs WHAT IS IN IT.")
            print("  That is brand-premium pricing, visible in the data.")
            print("  India is a branded-generics market: the same molecule")
            print("  sells at different prices depending on the label.")
        elif te_comp > te_manu * 1.15:
            print("  WHAT IS IN IT outweighs WHO MAKES IT.")
            print("  The molecule drives price more than the brand -- check")
            print("  whether specialty compositions are carrying this.")
        else:
            print("  Manufacturer and composition contribute comparably.")
        print("\n  Whichever it is, that sentence goes in your README.")

        shap.summary_plot(sv, sample, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(figs / "shap_summary.png", dpi=140)
        plt.close()
        print(f"\n  saved {figs / 'shap_summary.png'}")

    except ImportError:
        print("  shap not installed -- pip install shap")

    # ------------------------------------------------------- 8. save
    model.save_model(str(proc / "price_tier_model.json"))
    pd.DataFrame({"actual": y_te, "predicted": pred}).to_csv(
        root / "reports" / "price_tier_predictions.csv", index=False)
    print(f"\n  saved model -> {proc / 'price_tier_model.json'}")

    print(f"\n{'=' * 74}")
    print("  For the README: macro-F1, the baseline, the lift, the")
    print("  adjacent-error share, and the manufacturer-vs-composition")
    print("  comparison.")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
