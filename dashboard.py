#!/usr/bin/env python3
"""
PharmaLens - Stage 6: dashboard

DESIGN DECISION
---------------
Most RAG demos are a chat box: you type, an answer appears, and you have no
idea where it came from. This one leads with the ROUTING DECISION, because
that is the distinctive thing the system does.

The hero is not the answer text. It is:

    which route fired  ->  why  ->  what evidence backed it

Colour encodes state rather than decorating it:
    red    REFUSE, SCOPE     returns nothing, deliberately
    amber  ASSUME, PARTIAL   answers, with a stated limitation
    blue   CLARIFY           needs the user to disambiguate
    green  ANSWER            single drug, high confidence

The resolution trace below the verdict shows the pipeline as it actually
ran: the span that matched, the method that matched it, the confidence, and
the aspect left over for semantic ranking. Retrieval you cannot inspect is
retrieval you cannot trust.

ARCHITECTURE
------------
This is a CLIENT. It calls the FastAPI service over HTTP and holds no model,
no index and no data of its own. That keeps one copy of the logic and means
the dashboard demonstrates the API as well as itself.

RUNNING
-------
    # window 1
    uvicorn app:app --port 8080
    # or: kubectl port-forward svc/pharmalens 8080:80 -n pharmalens

    # window 2
    pip install streamlit
    streamlit run dashboard.py
"""

import os
import requests
import pandas as pd
import streamlit as st

API = os.environ.get("PHARMALENS_API", "http://localhost:8080")

st.set_page_config(
    page_title="PharmaLens",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Route colours are functional. Red is not "bad" -- REFUSE working correctly
# is the system's best behaviour. Red means "returns no documents".
ROUTE_STYLE = {
    "ANSWER":  ("#1B7F4B", "#E8F4ED", "Single drug, high confidence"),
    "PARTIAL": ("#9A6212", "#FBF1DF", "Multiple drugs — no interaction data exists"),
    "ASSUME":  ("#9A6212", "#FBF1DF", "Fuzzy match — assumption stated up front"),
    "CLARIFY": ("#1F5C8B", "#E7F0F7", "Ambiguous brand — asks rather than guesses"),
    "REFUSE":  ("#A32E2E", "#F9EAEA", "Nothing resolved — returns no documents"),
    "SCOPE":   ("#A32E2E", "#F9EAEA", "Clinical question — outside what this data can answer"),
}

METHOD_LABEL = {
    "ingredient": "exact ingredient",
    "product_name": "full product name",
    "brand_stem": "brand name",
    "spelling_variant": "spelling variant",
    "fuzzy_ingredient": "fuzzy match",
}

st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1180px; }

  .verdict {
      border-left: 5px solid var(--accent);
      background: var(--wash);
      padding: 1.1rem 1.3rem;
      margin: 0.4rem 0 1.1rem 0;
  }
  .verdict-route {
      font-size: 1.45rem; font-weight: 700; letter-spacing: -0.01em;
      color: var(--accent); line-height: 1.1;
  }
  .verdict-why { font-size: 0.93rem; color: #4A4F57; margin-top: 0.3rem; }

  .trace {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.82rem; color: #3A3F47;
      background: #F5F6F7; padding: 0.75rem 0.9rem;
      border: 1px solid #E3E5E8; margin-bottom: 1rem;
  }
  .trace b { color: #1A1D23; }

  .doc {
      border: 1px solid #E3E5E8; padding: 1rem 1.15rem; margin-bottom: 0.7rem;
      background: #FFFFFF;
  }
  .doc-key {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.8rem; color: #6B7076; margin-bottom: 0.45rem;
  }
  .doc-text { font-size: 0.93rem; line-height: 1.6; color: #23272C; }

  .nodocs {
      border: 1px dashed #D6D9DD; padding: 1.4rem; text-align: center;
      color: #6B7076; font-size: 0.92rem; background: #FAFBFC;
  }
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# API client
# ==========================================================================

@st.cache_data(ttl=60)
def get_stats():
    r = requests.get(f"{API}/stats", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_health():
    r = requests.get(f"{API}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def ask(question, k=5):
    r = requests.post(f"{API}/ask", json={"question": question, "k": k},
                      timeout=60)
    r.raise_for_status()
    return r.json()


def resolve(query):
    r = requests.get(f"{API}/search", params={"q": query}, timeout=15)
    r.raise_for_status()
    return r.json()


def predict(payload):
    r = requests.post(f"{API}/predict", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


# ==========================================================================
# Sidebar
# ==========================================================================

with st.sidebar:
    st.markdown("### PharmaLens")
    st.caption("Retrieval over Indian retail pharmacy listings")

    try:
        health = get_health()
        stats = get_stats()
        connected = True
    except Exception as e:
        connected = False
        st.error(
            f"No API at `{API}`.\n\n"
            "Start it with `uvicorn app:app --port 8080`, or set "
            "`PHARMALENS_API` to point elsewhere."
        )
        st.caption(f"{type(e).__name__}")

    if connected:
        st.markdown("**Corpus**")
        c = stats["corpus"]
        st.markdown(
            f"- {c['products']:,} products\n"
            f"- {c['compositions']:,} composition documents\n"
            f"- median {c.get('median_brands_per_composition', 0):.0f} brands each"
        )

        st.markdown("**Resolver**")
        r = stats["resolver"]
        st.markdown(
            f"- {r['ingredients']:,} ingredients\n"
            f"- {r['spelling_variants']:,} spelling variants\n"
            f"- {r['product_names']:,} product names\n"
            f"- {r['brand_stems']:,} brand stems"
        )

        st.markdown("**Thresholds**")
        t = stats["thresholds"]
        st.markdown(
            f"- answer at ≥ {t['min_answer_confidence']}\n"
            f"- refuse below {t['fuzzy_floor']}"
        )

        if not health.get("generation"):
            st.info(
                "Generation is off — no API key set. Routing and retrieval "
                "work regardless; the safety guarantees live in the router, "
                "not the model."
            )

if not connected:
    st.stop()


# ==========================================================================
# Main
# ==========================================================================

tab_ask, tab_price, tab_resolve = st.tabs(
    ["Ask a question", "Predict a price tier", "Resolve a name"])


# --------------------------------------------------------------- ask
with tab_ask:
    st.markdown("#### Ask about a medicine")
    st.caption(
        "Every question is routed before any model sees it. Six routes; "
        "three of them return no documents at all."
    )

    examples = {
        "Side effects of a brand name": "what are the side effects of Crocin",
        "A drug that isn't in the data": "side effects of flibanserin",
        "A dosing question": "how many tablets of Dolo should I take",
        "Two drugs together": "can I take Augmentin and Azithral together",
        "A misspelling": "side effects of metfromin",
        "A symptom, no drug named": "medicine for high blood pressure",
    }

    picked = st.selectbox("Try one of these, or write your own",
                          list(examples), index=0)
    question = st.text_input("Question", value=examples[picked],
                             label_visibility="collapsed")

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Resolving, filtering, ranking…"):
            try:
                res = ask(question)
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        accent, wash, blurb = ROUTE_STYLE.get(
            res["route"], ("#4A4F57", "#F2F3F4", ""))

        st.markdown(
            f"""<div class="verdict" style="--accent:{accent};--wash:{wash}">
                  <div class="verdict-route">{res['route']}</div>
                  <div class="verdict-why">{blurb} · {res['route_reason']}</div>
                </div>""",
            unsafe_allow_html=True)

        # --- the resolution trace ---------------------------------------
        if res["resolved_drugs"]:
            lines = []
            for d in res["resolved_drugs"][:4]:
                method = METHOD_LABEL.get(d["method"], d["method"])
                lines.append(
                    f"&nbsp;&nbsp;'<b>{d['matched_text']}</b>' → "
                    f"<b>{d['composition_key']}</b> "
                    f"({method}, confidence {d['confidence']:.2f})")
            extra = (f"<br>&nbsp;&nbsp;… and "
                     f"{len(res['resolved_drugs']) - 4} more"
                     if len(res["resolved_drugs"]) > 4 else "")
            aspect = res["aspect"] or "(nothing left — matched on identity alone)"
            st.markdown(
                f"""<div class="trace">
                      resolved by lookup:<br>{'<br>'.join(lines)}{extra}
                      <br><br>ranked semantically on: '<b>{aspect}</b>'
                      <br>mode: {res['mode']} · {res['latency_ms']} ms
                    </div>""",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div class="trace">
                      no drug name recognised — nothing to filter on
                      <br>mode: {res['mode']} · {res['latency_ms']} ms
                    </div>""",
                unsafe_allow_html=True)

        # --- answer ------------------------------------------------------
        if res.get("answer"):
            st.markdown("**Response**")
            st.write(res["answer"])
            if not res.get("generated"):
                st.caption(
                    "Fixed text from the router — no model was called.")

        # --- evidence ----------------------------------------------------
        st.markdown("**Evidence**")
        if res["documents"]:
            st.caption(
                f"{len(res['documents'])} document(s) the router approved. "
                "Anything the model says must come from here.")
            for d in res["documents"]:
                st.markdown(
                    f"""<div class="doc">
                          <div class="doc-key">{d['composition_key']}</div>
                          <div class="doc-text">{d['text']}</div>
                        </div>""",
                    unsafe_allow_html=True)
        else:
            st.markdown(
                """<div class="nodocs">
                     No documents returned, on purpose.<br>
                     Retrieval may have found candidates; the router rejected
                     them. Absence of data is not evidence of safety.
                   </div>""",
                unsafe_allow_html=True)

        if res.get("disclaimer"):
            st.caption(res["disclaimer"])


# ------------------------------------------------------------- price
with tab_price:
    st.markdown("#### Predict a price tier")
    st.caption(
        "From composition, manufacturer, form and pack size — never from "
        "price or anything derived from it."
    )

    col1, col2 = st.columns(2)
    with col1:
        comp = st.text_input("Composition", "paracetamol")
        manu = st.text_input("Manufacturer", "cipla")
        n_ing = st.number_input("Ingredients", 1, 8, 1)
    with col2:
        form = st.selectbox("Dosage form", [
            "tablet", "capsule", "syrup", "injection", "suspension",
            "cream", "drops", "gel", "ointment", "powder", "solution",
            "spray", "lotion", "infusion", "unknown"])
        qty = st.number_input("Pack quantity", 1.0, 1000.0, 10.0)

    if st.button("Predict", type="primary"):
        try:
            out = predict({
                "composition_key": comp.strip().lower(),
                "manufacturer_key": manu.strip().lower(),
                "dosage_form": form, "pack_qty": qty,
                "n_ingredients": int(n_ing), "type": "allopathy",
            })
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

        tier = out["predicted_tier"]
        st.markdown(f"### {tier}  ·  {out['tier_ranges_inr'][tier]}")

        probs = pd.DataFrame({
            "tier": list(out["probabilities"]),
            "probability": list(out["probabilities"].values()),
        }).set_index("tier")
        st.bar_chart(probs, height=200)
        st.caption(out["note"])


# ----------------------------------------------------------- resolve
with tab_resolve:
    st.markdown("#### Resolve a name to a composition")
    st.caption(
        "Exact dictionary lookup, no embeddings. This is the step that "
        "fixed brand recall from 1/6 to 6/6."
    )

    q = st.text_input("Brand or generic name", "crocin")

    if st.button("Resolve", type="primary") and q.strip():
        try:
            out = resolve(q.strip())
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

        if not out["resolved"]:
            st.markdown(
                """<div class="nodocs">
                     Not found. No fuzzy guess was made and no similar-looking
                     drug was substituted.
                   </div>""",
                unsafe_allow_html=True)
        else:
            rows = [{
                "composition": r["composition_key"],
                "matched on": r["matched_text"],
                "method": METHOD_LABEL.get(r["method"], r["method"]),
                "confidence": r["confidence"],
            } for r in out["resolved"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)
            if out["count"] > len(rows):
                st.caption(f"{out['count']} matches total, showing "
                           f"{len(rows)}.")


st.divider()
st.caption(
    "Summarises a public dataset of Indian retail pharmacy listings "
    "(prices as of November 2022). Not medical advice."
)
