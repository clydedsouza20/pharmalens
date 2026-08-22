#!/usr/bin/env python3
"""
PharmaLens - Stage 3b: what is semantic search actually good at?

THIS SCRIPT IS A LESSON, NOT A COMPONENT.
It builds no index you will ship. It exists to produce evidence for one
design decision, and to give you numbers you can quote in an interview.

THE CLAIM UNDER TEST
--------------------
An embedding places text in space so that similar MEANINGS land near each
other. Drug names are not meanings. Two failure modes follow:

  FALSE NEAR : Amlodipine (blood pressure) and Amiodarone (heart rhythm)
               are near-identical strings and entirely different drugs.
               The model was trained on English, not pharmacology.

  FALSE FAR  : Paracetamol and Acetaminophen are the SAME molecule with
               zero string overlap.

So text similarity is not drug identity. If retrieval leans on embeddings
for identity, a question about amlodipine returns amiodarone information --
fluently, confidently, and wrong.

FOUR EXPERIMENTS
----------------
  1. confusable name pairs vs random pairs      expect: FAIL
  2. querying by exact drug name                expect: MOSTLY OK
  3. querying by brand name                     expect: FAIL
  4. querying by symptom / indication           expect: WORK WELL

If 4 works and 1-3 struggle, the architecture writes itself:
    identity -> metadata filter (exact lookup)
    meaning  -> embedding (semantic search)
That is "filter first, then search", and it is the core of Stage 3c.

USAGE
-----
    cd P:\\PharmaLens
    pip install sentence-transformers
    python .\\stage3b_embedding_lesson.py

First run downloads ~90MB for the model. Embedding ~2,900 short documents
takes about a minute on CPU.
"""

import torch
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 130)

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, small, fast, good enough to learn on


# Look-alike / sound-alike pairs that are clinically unrelated.
# These are real confusion risks documented in medication-safety literature.
CONFUSABLE_PAIRS = [
    ("amlodipine", "amiodarone"),      # blood pressure   vs heart rhythm
    ("hydralazine", "hydroxyzine"),    # blood pressure   vs antihistamine
    ("clonidine", "clonazepam"),       # blood pressure   vs anticonvulsant
    ("prednisone", "prednisolone"),    # related but not interchangeable
    ("chlorpromazine", "chlorpropamide"),  # antipsychotic vs antidiabetic
    ("glipizide", "glyburide"),        # both antidiabetic, different drugs
    ("cefixime", "cefuroxime"),        # both cephalosporins, different
    ("tramadol", "trazodone"),         # painkiller       vs antidepressant
    ("nifedipine", "nimodipine"),      # different indications
    ("methotrexate", "metronidazole"), # chemo/DMARD      vs antibiotic
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    root = Path(__file__).resolve().parent
    proc = root / "data" / "processed"

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("\nRun: pip install sentence-transformers\n")

    # ------------------------------------------------------------- load
    print(f"\n{'=' * 76}\n1. LOAD + EMBED\n{'=' * 76}")
    docs = pd.read_parquet(proc / "documents.parquet")
    if "qa_any" in docs.columns:
        docs = docs[~docs["qa_any"].astype(bool)].reset_index(drop=True)
    print(f"  documents : {len(docs):,}")

    print(f"  model     : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  dimensions: {model.get_sentence_embedding_dimension()}")

    print("  embedding (about a minute on CPU)...")
    emb = model.encode(docs["text"].tolist(), batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True)
    print(f"  matrix    : {emb.shape}")

    key_to_idx = {k: i for i, k in enumerate(docs["composition_key"])}

    def search(query, k=5):
        q = model.encode([query], normalize_embeddings=True)[0]
        scores = emb @ q
        top = np.argsort(-scores)[:k]
        return [(docs.iloc[i]["composition_key"], float(scores[i])) for i in top]

    # =================================================================
    print(f"\n{'=' * 76}\n2. EXPERIMENT 1 -- CONFUSABLE DRUG NAMES\n{'=' * 76}")
    print("Are look-alike drug names placed CLOSER together than unrelated")
    print("drugs? If yes, the embedding is encoding spelling, not medicine.\n")

    names = sorted({n for pair in CONFUSABLE_PAIRS for n in pair})
    name_emb = {n: model.encode([n], normalize_embeddings=True)[0]
                for n in names}

    conf_scores = []
    print(f"  {'pair':<40} {'cosine':>8}   both in corpus?")
    for a, b in CONFUSABLE_PAIRS:
        s = cosine(name_emb[a], name_emb[b])
        conf_scores.append(s)
        present = "yes" if (a in key_to_idx and b in key_to_idx) else "no"
        print(f"  {a + ' / ' + b:<40} {s:>8.3f}   {present}")

    # Baseline: random unrelated ingredient names from the corpus.
    rng = np.random.default_rng(42)
    singles = [k for k in docs["composition_key"] if "+" not in k]
    sample = list(rng.choice(singles, size=min(60, len(singles)), replace=False))
    samp_emb = model.encode(sample, normalize_embeddings=True)
    rand_scores = [cosine(samp_emb[i], samp_emb[j])
                   for i in range(len(sample))
                   for j in range(i + 1, len(sample))]

    print(f"\n  confusable pairs : mean cosine {np.mean(conf_scores):.3f}")
    print(f"  random drug pairs: mean cosine {np.mean(rand_scores):.3f}")
    print(f"  ratio            : {np.mean(conf_scores)/np.mean(rand_scores):.2f}x")
    print(f"\n  --- READ THIS ---")
    if np.mean(conf_scores) > np.mean(rand_scores) * 1.3:
        print("  CONFIRMED. Look-alike names sit measurably closer together")
        print("  than unrelated drugs. The model is encoding SPELLING.")
        print("  Any retrieval that relies on embeddings to identify WHICH")
        print("  drug is being asked about will confuse these pairs.")
    else:
        print("  Weaker than expected -- report the actual numbers you got.")

    # =================================================================
    print(f"\n{'=' * 76}\n3. EXPERIMENT 2 -- QUERY BY EXACT DRUG NAME\n{'=' * 76}")
    print("Search the corpus for a drug by name. Is the correct document")
    print("ranked first?\n")

    probes = [n for n in names if n in key_to_idx][:8]
    hits_at_1 = hits_at_5 = 0
    for name in probes:
        res = search(name, k=5)
        ranks = [i for i, (k, _) in enumerate(res) if k == name]
        rank = ranks[0] + 1 if ranks else None
        hits_at_1 += rank == 1
        hits_at_5 += rank is not None
        flag = "OK " if rank == 1 else ("~  " if rank else "MISS")
        print(f"  {flag} query '{name}'  -> rank {rank if rank else '>5'}")
        for k, s in res[:3]:
            mark = " <-- correct" if k == name else ""
            print(f"           {s:.3f}  {k}{mark}")
    if probes:
        print(f"\n  recall@1 {hits_at_1}/{len(probes)}   "
              f"recall@5 {hits_at_5}/{len(probes)}")

    # =================================================================
    print(f"\n{'=' * 76}\n4. EXPERIMENT 3 -- QUERY BY BRAND NAME\n{'=' * 76}")
    print("Indian patients say 'Augmentin', not 'amoxycillin + clavulanic")
    print("acid'. Only 4 sample brands went into each document's text, so")
    print("most brands appear NOWHERE in any embedded string.\n")

    BRAND_TESTS = [
        ("augmentin", "amoxycillin + clavulanic acid"),
        ("azithral", "azithromycin"),
        ("crocin", "paracetamol"),
        ("dolo", "paracetamol"),
        ("glycomet", "metformin"),
        ("pan d", "domperidone + pantoprazole"),
    ]
    brand_hits = 0
    for brand, expected in BRAND_TESTS:
        res = search(brand, k=5)
        got = [k for k, _ in res]
        ok = expected in got
        brand_hits += ok
        print(f"  {'OK  ' if ok else 'MISS'} '{brand}' -> expected "
              f"'{expected}'")
        print(f"        top-3: {', '.join(got[:3])}")

    print(f"\n  brand recall@5: {brand_hits}/{len(BRAND_TESTS)}")
    print(f"\n  --- READ THIS ---")
    print("  Brand names cannot be found semantically unless they happen to")
    print("  be in the embedded text. This is NOT fixable with a better")
    print("  embedding model -- the information is absent. It is fixable")
    print("  with a brand -> composition lookup table, which we already")
    print("  have in metadata (brand_keys). Identity is a LOOKUP problem.")

    # =================================================================
    print(f"\n{'=' * 76}\n5. EXPERIMENT 4 -- QUERY BY SYMPTOM / MEANING\n{'=' * 76}")
    print("Now play to the embedding's actual strength: concepts.\n")

    SYMPTOM_TESTS = [
        "what can I take for acid reflux and heartburn",
        "medicine for type 2 diabetes",
        "treatment for bacterial infection of the chest",
        "something for allergy and runny nose",
        "drug that causes drowsiness and dry mouth",
        "medicine for high blood pressure",
    ]
    for q in SYMPTOM_TESTS:
        res = search(q, k=3)
        print(f"\n  '{q}'")
        for k, s in res:
            cls = docs.loc[docs["composition_key"] == k,
                           "therapeutic_class"].iloc[0]
            print(f"      {s:.3f}  {k:<44} [{cls}]")

    print(f"\n  --- READ THIS ---")
    print("  These should look sensible. Semantic search is GOOD at mapping")
    print("  a described problem onto a therapeutic class. That is real")
    print("  capability, and it is what the embedding should be used for.")

    # =================================================================
    print(f"\n{'=' * 76}\n6. THE DESIGN DECISION THIS PRODUCES\n{'=' * 76}")
    print("""
  Experiments 1-3 show embeddings are unreliable for IDENTITY.
  Experiment 4 shows they are strong for MEANING.

  So do not make one mechanism do both jobs:

      STEP 1  resolve the drug            exact lookup on metadata
              'Augmentin'  -> brand_keys  -> amoxycillin + clavulanic acid
              'amlodipine' -> ingredients -> amlodipine
              (no embedding involved -- no confusion possible)

      STEP 2  filter the vector store to those documents

      STEP 3  semantic search WITHIN the filtered set for the aspect
              asked about: side effects, uses, price, safety

  This is 'filter first, then search'. Chroma supports predicate filters
  natively, which is the actual reason to pick it over FAISS -- not
  popularity.

  If asked in an interview why you did not use pure vector search, the
  answer is a number from Experiment 1, not an opinion.
""")

    np.save(proc / "embeddings_lesson.npy", emb)
    print(f"  cached embeddings -> {proc / 'embeddings_lesson.npy'}")
    print(f"\n{'=' * 76}\nNext: 3c -- build the real index with metadata filtering.\n{'=' * 76}")


if __name__ == "__main__":
    main()
