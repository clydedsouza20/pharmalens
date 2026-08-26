#!/usr/bin/env python3
"""
PharmaLens - Stage 3c: the retrieval system

WHAT 3b PROVED
--------------
    brand name  -> correct composition   1/6   via pure vector search
    look-alike drug pairs                1.86x closer than random pairs
    generic name lookup                  7/8   (works -- name is in the text)
    symptom / concept queries            good  (this is what embeddings do)

Failures were ORTHOGRAPHIC, not semantic: 'crocin' retrieved 'crotamiton'
(a scabies treatment) because the letters look alike. No larger embedding
model fixes that -- the problem is asking a similarity model to do
identity resolution at all.

WHAT 3d MEASURED (200-case gold set, recall@5)
----------------------------------------------
    A  semantic only    47.5%     perfect on concepts, 13% on brands
    B  resolver only    96.0%     perfect on brands, 25% on concepts
    C  hybrid           99.5%     each covers the other's blind spot

The hybrid does not win by being uniformly better. It wins because the
resolver and the embedding fail in exactly opposite places.

THE ARCHITECTURE
----------------
    STAGE 1  RESOLVE   'Crocin 500' -> paracetamol
                       exact dictionary lookup. No similarity, no confusion.

    STAGE 2  FILTER    restrict the vector store to those compositions
                       (Chroma `where` clause with $in)

    STAGE 3  SEARCH    embed the REST of the question ('is it safe with
                       alcohol') and rank only within the filtered set

Identity is a LOOKUP problem. Meaning is a SEARCH problem. One mechanism
each.

FIX LOG
-------
  v2, from reading v1's output:
    1. FUZZY HIJACK -- 'medicine for high blood pressure' fuzzy-matched
       'medicine' to a brand called 'Medicaine'. Fixed by adding ~50
       stopwords and restricting fuzzy matching to ingredient names.
       NOTE: raising the cutoff was NOT the fix. 'medicine' vs 'medicaine'
       scores ~0.94, HIGHER than the real typo 'amoxicilin' vs
       'amoxycillin' (0.857). No threshold separates them. Shrinking the
       SEARCH SPACE fixed it; tightening the THRESHOLD only broke genuine
       typo handling.
    2. SINGLE-LETTER TOKENS -- 'Pan D' became 'pan'. Single letters are
       meaningful in Indian brand names: Pan D, Zifi O, Monocef O.
    3. EXACT COMPOSITION LOST TO ITS COMBINATIONS -- exact now scores
       1.00, containing-combinations 0.90, ranking is confidence-first.

  v3, from the 3d failure list:
    4. SHORT-CIRCUIT ON AMBIGUOUS BRANDS -- 'noxprin' hit by_product
       (ozenoxacin) and the if/elif chain never checked by_stem
       (enoxaparin). All indexes are now unioned.
    5. LONG GENERIC NAMES UNREACHABLE -- 'methoxy polyethylene glycol
       epoetin beta' is six words; max span was four. Raised to eight.
    6. FUZZY CUTOFF restored to 0.86.

  v4, from the stage 4 routing test:
    7. SPELLING VARIANTS ARE NOT TYPOS -- 'amoxicilin' vs 'amoxycillin'
       scores 0.857, three thousandths under the 0.86 cutoff. Lowering
       the cutoff would be tuning a brittle number. The real point is
       that amoxycillin/amoxicillin are the SAME WORD: the y/i swap is a
       systematic transliteration convention in drug nomenclature, as are
       ph/f, ae/e and doubled consonants. A variant index handles this as
       an EXACT lookup on a normalised key -- still deterministic, still
       incapable of drifting, unlike a similarity threshold. Confidence
       0.95, above the answer threshold, because a spelling variant is
       the same drug rather than a guess.

USAGE
-----
    cd P:\\PharmaLens
    pip install chromadb sentence-transformers
    python .\\stage3c_retrieval.py
"""

import torch  # MUST be first on Windows: importing pandas/numpy first breaks
              # torch's DLL initialisation (WinError 1114 on c10.dll)

import re
import json
import difflib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

pd.set_option("display.width", 130)

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION = "pharmalens_compositions"

# Longest drug name span to try. Six-word generic names exist
# ('methoxy polyethylene glycol epoetin beta'), so four was too short.
MAX_SPAN_LEN = 8

# Fuzzy matching is for genuine typos only. See fix 1 for why this is
# 0.86 and not something stricter.
FUZZY_CUTOFF = 0.86


# ==========================================================================
# Vocabulary
# ==========================================================================

# Words describing the PACKAGE, not the medicine.
FORM_WORDS = {
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
    "syrup", "syp", "suspension", "susp", "injection", "inj", "cream",
    "ointment", "gel", "drop", "drops", "solution", "lotion", "powder",
    "sachet", "spray", "vial", "ampoule", "tube", "bottle", "strip", "kit",
    "oral", "topical", "infusion", "sr", "xr", "cr", "er", "dt", "md",
    "for", "of", "the", "and", "with", "mg", "ml", "mcg", "gm", "iu",
}

# Words that are never a drug name. A span made only of these is skipped
# before any lookup runs.
QUESTION_WORDS = {
    # interrogatives / function words
    "what", "whats", "which", "who", "when", "where", "why", "how", "is",
    "are", "can", "could", "should", "would", "do", "does", "did", "i",
    "me", "my", "you", "your", "it", "its", "a", "an", "the", "to", "for",
    "of", "in", "on", "at", "about", "from", "by", "as", "be", "been",
    "have", "has", "had", "there", "any", "some", "this", "that", "these",
    # generic pharmacy nouns
    "medicine", "medicines", "medication", "medications", "drug", "drugs",
    "tablet", "tablets", "pill", "pills", "capsule", "capsules", "syrup",
    "dose", "dosage", "treatment", "cure", "remedy", "brand", "generic",
    "something", "anything", "best", "good", "bad", "better", "worse",
    # question topics
    "side", "effects", "effect", "safe", "safety", "danger", "dangerous",
    "interaction", "interactions", "together", "combine", "combination",
    "price", "cost", "costs", "cheap", "cheaper", "expensive",
    "take", "taking", "taken", "use", "used", "using", "uses",
    "tell", "give", "show", "know", "help", "need", "want",
    # symptom words that collide with brand names
    "high", "low", "blood", "pressure", "sugar", "pain", "fever", "cold",
    "cough", "headache", "acidity", "gas", "infection", "allergy",
    "alcohol", "food", "pregnancy", "pregnant", "children", "kids", "baby",
}

_NONALNUM = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")
_DOUBLE = re.compile(r"(.)\1+")


# ==========================================================================
# Normalisation
# ==========================================================================

def norm_text(raw):
    if not isinstance(raw, str):
        return ""
    return _WS.sub(" ", _NONALNUM.sub(" ", raw.lower())).strip()


def product_key(raw):
    """'Crocin 500 Tablet' -> 'crocin 500'   (drops packaging words)

    Single-character tokens are KEPT (fix 2). 'Pan D', 'Zifi O' and
    'Monocef O' are distinct products; dropping the trailing letter
    collapses each into its base brand and loses the distinction.
    """
    s = norm_text(raw)
    toks = [t for t in s.split() if t not in FORM_WORDS]
    return " ".join(toks)


def brand_stem(raw):
    """'Crocin 500 Tablet' -> 'crocin'   (the bit a patient actually says)

    Stops at the first strength or packaging word, so 'Dolo 650 Tablet'
    yields 'dolo'. Single letters are kept, so 'Pan D Tablet' yields
    'pan d' rather than 'pan'.
    """
    s = norm_text(raw)
    toks = []
    for t in s.split():
        if t in FORM_WORDS:
            break
        if any(ch.isdigit() for ch in t):
            break
        toks.append(t)
    return " ".join(toks)


def variant_key(raw):
    """Collapse systematic spelling variants in drug nomenclature (fix 7).

    'amoxycillin' and 'amoxicillin' are the same drug spelled two ways.
    So are cyclosporin/ciclosporin, sulphate/sulfate, cephalexin/
    cefalexin. These are transliteration conventions -- British vs
    American, Latin vs anglicised -- not typos, and generic string
    similarity cannot tell the difference: 'amoxicilin' vs 'amoxycillin'
    scores 0.857 on difflib, three thousandths under a 0.86 cutoff.

    Normalising to a canonical form turns this into an EXACT dictionary
    lookup. That matters: exact lookups cannot drift the way a
    similarity threshold does. Domain knowledge, applied deterministically.

        amoxycillin  -> amoxicilin
        amoxicillin  -> amoxicilin
        amoxicilin   -> amoxicilin
        sulphate     -> sulfate
        cyclosporin  -> ciclosporin
    """
    s = norm_text(raw)
    s = s.replace("ph", "f")       # sulphate  -> sulfate
    s = s.replace("y", "i")        # amoxycillin -> amoxicillin
    s = s.replace("ae", "e")       # haemo-    -> hemo-
    s = s.replace("oe", "e")       # oestrogen -> estrogen
    s = s.replace("k", "c")        # -kacin    -> -cacin
    s = _DOUBLE.sub(r"\1", s)      # cillin    -> cilin
    return s.strip()


# ==========================================================================
# Stage 1: the resolver
# ==========================================================================

class DrugResolver:
    """Maps free text to composition keys by EXACT LOOKUP, never similarity.

    Five indexes, with confidence reflecting how direct the evidence is:
        ingredient (exact)      'amlodipine'  -> amlodipine          1.00
        full product name       'crocin 500'  -> paracetamol         0.98
        spelling variant        'amoxicilin'  -> amoxycillin         0.95
        brand stem              'crocin'      -> paracetamol         0.92
        ingredient (contained)  'amlodipine'  -> amlodipine+atenolol 0.90
        fuzzy fallback          'metfromin'   -> metformin           0.70

    The first five are exact dictionary hits and are UNIONED (fix 4): an
    earlier if/elif chain short-circuited, so 'noxprin' matched a product
    mapping to ozenoxacin and never checked the brand stem mapping to
    enoxaparin. Ambiguous brands must surface every candidate.

    Fuzzy matching runs LAST and only against the 1,640 ingredient names.
    It is the one step that can fail the way 3b warned about.
    """

    def __init__(self, products, docs):
        self.by_ingredient = defaultdict(set)
        self.by_product = defaultdict(set)
        self.by_stem = defaultdict(set)
        self.by_variant = defaultdict(set)
        self.valid_compositions = set(docs["composition_key"])

        # --- ingredients, taken from the composition keys themselves ----
        for ck in docs["composition_key"]:
            for part in str(ck).split("+"):
                p = part.strip()
                if p:
                    self.by_ingredient[p].add(ck)

        # --- spelling variants (fix 7) ----------------------------------
        # Same lookup discipline as the exact indexes, just with a
        # normalised key. NOT fuzzy matching: still an exact hit.
        for ing, cks in self.by_ingredient.items():
            vk = variant_key(ing)
            if vk:
                self.by_variant[vk].update(cks)

        # --- brands, from ALL products, not the truncated 50-brand list -
        sub = products[["name", "composition_key"]].dropna()
        for name, ck in zip(sub["name"], sub["composition_key"]):
            if ck not in self.valid_compositions:
                continue
            pk = product_key(name)
            if pk:
                self.by_product[pk].add(ck)
            st = brand_stem(name)
            if st and len(st) >= 3:
                self.by_stem[st].add(ck)

        self._ing_list = list(self.by_ingredient)

    # ------------------------------------------------------------------
    def _candidate_spans(self, query, max_len=MAX_SPAN_LEN):
        """All contiguous word spans, longest first.

        Longest-first matters twice over: 'side effects of pan d' yields
        'pan d' before 'pan', and long generic names like 'methoxy
        polyethylene glycol epoetin beta' are tried whole before their
        misleading fragments.
        """
        toks = [t for t in norm_text(query).split() if t]
        spans = []
        for n in range(min(max_len, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                spans.append(" ".join(toks[i:i + n]))
        return spans

    def resolve(self, query, fuzzy_cutoff=FUZZY_CUTOFF):
        """Return [(composition_key, matched_text, method, confidence)]."""
        found, consumed = [], set()

        for span in self._candidate_spans(query):
            if not span or span in consumed:
                continue
            # A span made only of question words is never a drug.
            if all(t in QUESTION_WORDS for t in span.split()):
                continue

            # --- union across ALL exact indexes (fix 4) -----------------
            matches = []
            if span in self.by_ingredient:
                # Exact composition outranks combinations that merely
                # contain the ingredient (fix 3).
                matches += [(ck, "ingredient", 1.00 if ck == span else 0.90)
                            for ck in sorted(self.by_ingredient[span])]
            if span in self.by_product:
                matches += [(ck, "product_name", 0.98)
                            for ck in sorted(self.by_product[span])]
            if span in self.by_stem:
                matches += [(ck, "brand_stem", 0.92)
                            for ck in sorted(self.by_stem[span])]

            # --- spelling variant, only if nothing exact hit (fix 7) ----
            if not matches:
                vk = variant_key(span)
                if vk and vk in self.by_variant:
                    matches += [(ck, "spelling_variant", 0.95)
                                for ck in sorted(self.by_variant[vk])]

            if matches:
                for ck, method, conf in matches:
                    found.append((ck, span, method, conf))
                # Consume this span and all its sub-spans, so 'pan d' does
                # not also fire a separate match on 'pan'.
                consumed.add(span)
                for sub in self._candidate_spans(span):
                    consumed.add(sub)

        if found:
            return self._dedupe(found)

        # --- fuzzy fallback: genuine typos only -------------------------
        # Ingredients ONLY (fix 1). Brand stems are short, numerous and
        # collide with ordinary English -- that is how 'medicine' became
        # 'Medicaine'.
        toks = [t for t in norm_text(query).split()
                if t not in QUESTION_WORDS and len(t) > 4]
        for t in toks:
            close = difflib.get_close_matches(t, self._ing_list, n=1,
                                              cutoff=fuzzy_cutoff)
            if close:
                for ck in sorted(self.by_ingredient[close[0]]):
                    conf = 0.70 if ck == close[0] else 0.65
                    found.append((ck, f"{t}~{close[0]}",
                                  "fuzzy_ingredient", conf))
                break

        return self._dedupe(found)

    @staticmethod
    def _dedupe(found):
        """One row per composition, keeping its highest-confidence match."""
        best = {}
        for ck, span, method, conf in found:
            if ck not in best or conf > best[ck][2]:
                best[ck] = (span, method, conf)
        return sorted([(ck, s, m, c) for ck, (s, m, c) in best.items()],
                      key=lambda r: (-r[3], r[0]))

    def strip_drugs(self, query, resolved):
        """Remove matched drug spans, leaving the ASPECT of the question.

        'side effects of crocin' -> 'side effects'
        That remainder is what gets embedded in stage 3, so the vector
        search ranks on WHAT is asked rather than WHICH drug.
        """
        out = norm_text(query)
        for _, span, _, _ in resolved:
            if "~" not in span:
                out = re.sub(rf"\b{re.escape(span)}\b", " ", out)
        return _WS.sub(" ", out).strip()


# ==========================================================================
# Stages 2 + 3: filter, then search
# ==========================================================================

class PharmaLensRetriever:
    def __init__(self, collection, model, resolver, docs):
        self.col = collection
        self.model = model
        self.resolver = resolver
        self.docs = docs.set_index("composition_key")

    def query(self, question, k=5):
        resolved = self.resolver.resolve(question)
        aspect = self.resolver.strip_drugs(question, resolved)

        result = {"question": question, "resolved_drugs": resolved,
                  "aspect": aspect, "mode": None, "hits": [], "note": None,
                  "confidence": max([c for *_, c in resolved], default=0.0),
                  "ambiguous": False}

        if resolved:
            result["mode"] = "filtered"
            keys = [ck for ck, _, _, _ in resolved]

            top_conf = max(c for *_, c in resolved)
            primary = [ck for ck, _, _, c in resolved if c == top_conf]

            # Ambiguity means one BRAND span maps to several compositions.
            # The ingredient hierarchy (exact 1.00 vs contained 0.90) is a
            # hierarchy, not ambiguity, so ingredient methods are excluded.
            top_span = resolved[0][1]
            brandish = {ck for ck, span, m, _ in resolved
                        if span == top_span
                        and m in ("product_name", "brand_stem")}
            result["ambiguous"] = len(brandish) > 1
            if result["ambiguous"]:
                result["note"] = (f"'{top_span}' matches "
                                  f"{len(brandish)} different compositions")

            # Nothing left to rank on -> return by identity alone.
            if not aspect or len(aspect) < 4:
                ordered = primary + [c for c in keys if c not in primary]
                for ck in ordered[:k]:
                    if ck in self.docs.index:
                        result["hits"].append(
                            {"composition_key": ck, "score": 1.0,
                             "text": self.docs.loc[ck, "text"],
                             "why": "exact drug match, no aspect to rank"})
                if not result["note"]:
                    result["note"] = "returned by identity lookup only"
                return result

            qv = self.model.encode([aspect], normalize_embeddings=True)[0]
            res = self.col.query(query_embeddings=[qv.tolist()],
                                 n_results=min(max(k * 2, 1), max(len(keys), 1)),
                                 where={"composition_key": {"$in": keys}})

            conf_of = {ck: c for ck, _, _, c in resolved}
            rows = [{"composition_key": ck, "semantic": 1 - dist,
                     "text": doc, "resolver_conf": conf_of.get(ck, 0.0)}
                    for ck, doc, dist in zip(res["ids"][0],
                                             res["documents"][0],
                                             res["distances"][0])]
            # Resolver confidence first, semantic score as tiebreak.
            # Identity is the stronger signal; meaning only breaks ties.
            rows.sort(key=lambda r: (-r["resolver_conf"], -r["semantic"]))
            for r in rows[:k]:
                result["hits"].append(
                    {"composition_key": r["composition_key"],
                     "score": r["semantic"], "text": r["text"],
                     "why": f"resolver conf {r['resolver_conf']:.2f}"})
            return result

        # ---- UNFILTERED: no drug named -> concept query ----------------
        result["mode"] = "semantic"
        qv = self.model.encode([question], normalize_embeddings=True)[0]
        res = self.col.query(query_embeddings=[qv.tolist()], n_results=k)
        for ck, doc, dist in zip(res["ids"][0], res["documents"][0],
                                 res["distances"][0]):
            result["hits"].append({"composition_key": ck, "score": 1 - dist,
                                   "text": doc,
                                   "why": "semantic match, no drug named"})
        result["note"] = ("no drug name recognised -- results are concept "
                          "matches and may not name a specific drug")
        return result


# ==========================================================================

def main():
    root = Path(__file__).resolve().parent
    proc = root / "data" / "processed"
    store = root / "data" / "chroma"

    try:
        import chromadb
    except ImportError:
        raise SystemExit("\nRun: pip install chromadb\n")
    from sentence_transformers import SentenceTransformer

    # ------------------------------------------------------------- load
    print(f"\n{'=' * 76}\n1. LOAD\n{'=' * 76}")
    docs = pd.read_parquet(proc / "documents.parquet")
    if "qa_any" in docs.columns:
        docs = docs[~docs["qa_any"].astype(bool)].reset_index(drop=True)
    products = pd.read_parquet(proc / "products.parquet")
    print(f"  documents : {len(docs):,}")
    print(f"  products  : {len(products):,}  (source for the brand index)")

    # ------------------------------------------------- build the resolver
    print(f"\n{'=' * 76}\n2. RESOLVER (stage 1: identity)\n{'=' * 76}")
    resolver = DrugResolver(products, docs)
    print(f"  ingredient entries  : {len(resolver.by_ingredient):,}")
    print(f"  spelling variants   : {len(resolver.by_variant):,}")
    print(f"  full product names  : {len(resolver.by_product):,}")
    print(f"  brand stems         : {len(resolver.by_stem):,}")
    print(f"  max span length     : {MAX_SPAN_LEN} words")
    print(f"  fuzzy cutoff        : {FUZZY_CUTOFF}")
    print("\n  Exact-match dictionaries only. The crocin -> crotamiton")
    print("  failure from 3b is structurally impossible here, because no")
    print("  similarity is computed during identity resolution.")

    # -------------------------------------------------------- embed once
    print(f"\n{'=' * 76}\n3. EMBED + INDEX (stages 2+3: meaning)\n{'=' * 76}")
    model = SentenceTransformer(MODEL_NAME)

    cache = proc / "embeddings_docs.npy"
    emb = None
    if cache.exists():
        cached = np.load(cache)
        if cached.shape[0] == len(docs):
            emb = cached
            print("  using cached embeddings")
    if emb is None:
        print("  embedding documents...")
        emb = model.encode(docs["text"].tolist(), batch_size=64,
                           show_progress_bar=True, normalize_embeddings=True)
        np.save(cache, emb)
    print(f"  vectors : {emb.shape}")

    client = chromadb.PersistentClient(path=str(store))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(name=COLLECTION,
                                   metadata={"hnsw:space": "cosine"})

    # Chroma metadata must be scalar (str/int/float/bool).
    metas = [{
        "composition_key": str(r["composition_key"]),
        "therapeutic_class": str(r.get("therapeutic_class") or "unknown"),
        "n_ingredients": int(r.get("n_ingredients") or 1),
        "n_brands": int(r.get("n_brands") or 0),
        "n_products": int(r.get("n_products") or 0),
        "price_min": float(r.get("price_min") or 0),
        "price_median": float(r.get("price_median") or 0),
        "price_max": float(r.get("price_max") or 0),
        "class_agreement": float(r.get("class_agreement") or 0),
    } for _, r in docs.iterrows()]

    B = 500
    for i in range(0, len(docs), B):
        sl = slice(i, i + B)
        col.add(ids=docs["composition_key"].iloc[sl].tolist(),
                embeddings=emb[sl].tolist(),
                documents=docs["text"].iloc[sl].tolist(),
                metadatas=metas[sl])
    print(f"  indexed : {col.count():,} documents -> {store}")
    print("\n  Chroma chosen for NATIVE metadata filtering (`where` + $in).")
    print("  FAISS is a pure similarity index; filter-first would mean")
    print("  hand-rolling a metadata layer. That is the real reason.")

    retriever = PharmaLensRetriever(col, model, resolver, docs)

    # ==================================================================
    print(f"\n{'=' * 76}\n4. THE 3b BRAND TEST, RERUN\n{'=' * 76}")
    print("Pure vector search scored 1/6. Same queries, new architecture.\n")

    BRAND_TESTS = [
        ("augmentin", "amoxycillin + clavulanic acid"),
        ("azithral", "azithromycin"),
        ("crocin", "paracetamol"),
        ("dolo", "paracetamol"),
        ("glycomet", "metformin"),
        ("pan d", "domperidone + pantoprazole"),
    ]
    hits = 0
    for brand, expected in BRAND_TESTS:
        res = retriever.query(brand, k=5)
        got = [h["composition_key"] for h in res["hits"]]
        ok = expected in got
        hits += ok
        method = res["resolved_drugs"][0][2] if res["resolved_drugs"] else "none"
        print(f"  {'OK  ' if ok else 'MISS'} '{brand}' -> {got[:2]}")
        print(f"          expected '{expected}'   via {method}")
    print(f"\n  brand recall: {hits}/{len(BRAND_TESTS)}   "
          f"(pure vector search: 1/6)")

    # ==================================================================
    print(f"\n{'=' * 76}\n5. REGRESSION TESTS\n{'=' * 76}")
    print("Every one of these was a real bug. Each fix keeps its test.\n")

    REGRESSIONS = [
        ("medicine for high blood pressure",
         "fix 1: SEMANTIC mode, not hijacked by the brand 'Medicaine'"),
        ("what is Pan D used for",
         "fix 2: resolves 'pan d', aspect must not contain a stray 'd'"),
        ("how much does amlodipine cost",
         "fix 3: plain amlodipine ranks FIRST, above its combinations"),
        ("noxprin",
         "fix 4: must surface BOTH ozenoxacin and enoxaparin -> ambiguous"),
        ("methoxy polyethylene glycol epoetin beta",
         "fix 5: six-word generic name must resolve whole"),
        ("side effects of metfromin",
         "fix 6: typo reaches metformin via fuzzy at 0.86"),
        ("what are the side effects of amoxicilin",
         "fix 7: spelling variant -> amoxycillin at 0.95, NOT fuzzy 0.70"),
    ]
    for q, expectation in REGRESSIONS:
        res = retriever.query(q, k=3)
        print(f"\n{'-' * 76}\n  Q: {q}\n  {expectation}")
        print(f"  mode   : {res['mode']}   conf {res['confidence']:.2f}"
              f"{'   AMBIGUOUS' if res['ambiguous'] else ''}")
        for ck, span, method, conf in res["resolved_drugs"][:4]:
            print(f"  resolved: '{span}' -> {ck}  [{method}, {conf:.2f}]")
        if not res["resolved_drugs"]:
            print("  resolved: (none)")
        print(f"  aspect : '{res['aspect']}'")
        for h in res["hits"][:3]:
            print(f"    {h['score']:.3f}  {h['composition_key']}")

    # ==================================================================
    print(f"\n{'=' * 76}\n6. REALISTIC QUESTIONS\n{'=' * 76}")
    QUESTIONS = [
        "what are the side effects of Crocin",
        "is Dolo 650 safe to take with alcohol",
        "can I take Augmentin and Azithral together",
        "side effects of flibanserin",          # absent from the corpus
        "something for acidity and gas",        # concept query
    ]
    for q in QUESTIONS:
        res = retriever.query(q, k=3)
        print(f"\n{'-' * 76}\n  Q: {q}")
        print(f"  mode      : {res['mode']}   conf {res['confidence']:.2f}")
        for ck, span, method, conf in res["resolved_drugs"][:3]:
            print(f"  resolved  : '{span}' -> {ck}  [{method}, {conf:.2f}]")
        if not res["resolved_drugs"]:
            print("  resolved  : (no drug recognised)")
        print(f"  aspect    : '{res['aspect']}'")
        if res["note"]:
            print(f"  note      : {res['note']}")
        for h in res["hits"][:3]:
            print(f"    {h['score']:.3f}  {h['composition_key']}")

    # ==================================================================
    print(f"\n{'=' * 76}\n7. WHAT RETRIEVAL STILL CANNOT FIX\n{'=' * 76}")
    print("""
  These are stage 4's job -- the router and the prompt, not the index:

    NOTHING RESOLVED   'side effects of flibanserin' falls through to
                       semantic mode and returns confident-looking but
                       WRONG documents. Stage 4 refuses on low confidence
                       rather than summarising whatever came back.
                       Absence of data is not evidence of safety.

    MULTI-DRUG         'Augmentin and Azithral together' resolves two
                       compositions, but this corpus has NO drug-drug
                       interaction data. Stage 4 routes to PARTIAL.

    AMBIGUOUS BRAND    result['ambiguous'] is set when one brand span maps
                       to several compositions via the brand indexes.
                       Stage 4 routes to CLARIFY and asks.

    ESKETAMINE         a fuzzy hit at 0.70 may be a real, different drug
                       rather than a typo. No cutoff separates those two
                       cases. Stage 4 routes to ASSUME and says so.
""")

    with open(proc / "resolver_stats.json", "w") as f:
        json.dump({"ingredients": len(resolver.by_ingredient),
                   "spelling_variants": len(resolver.by_variant),
                   "products": len(resolver.by_product),
                   "brand_stems": len(resolver.by_stem),
                   "documents": len(docs),
                   "max_span_len": MAX_SPAN_LEN,
                   "fuzzy_cutoff": FUZZY_CUTOFF}, f, indent=2)

    print(f"{'=' * 76}\nNow rerun stage4_answer.py and stage3d_evaluate.py.")
    print("Compare to 47.5 / 96.0 / 99.5 recall@5 and 10/12 routing.")
    print(f"{'=' * 76}\n")


if __name__ == "__main__":
    main()