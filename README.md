\# PharmaLens



\*\*Retrieval and price modelling over 253,969 Indian pharmaceutical product records.\*\*



Pure vector search resolved branded drug names to the correct composition \*\*1 time in 6\*\*. Separating identity resolution from semantic search took that to \*\*6 in 6\*\*, and overall retrieval from \*\*47.5% to 98.5% recall@5\*\* on a 200-case evaluation set.



\---



\## Results



| Metric | Result |

|---|---|

| Retrieval recall@5 | \*\*98.5%\*\* hybrid vs \*\*47.5%\*\* pure vector search |

| Brand-name resolution | \*\*6/6\*\* vs \*\*1/6\*\* for pure vector search |

| Price tier classifier | \*\*0.677\*\* macro-F1 vs \*\*0.254\*\* random baseline |

| Corpus reduction | 253,969 products → \*\*2,959\*\* composition documents |

| Safety routing | \*\*6 of 12\*\* edge cases resolved without an LLM call |

| Deployment | AWS ECS Fargate, 1.37GB image, 53s pull / 2s start |



\---



\## What it does



Given \*"what are the side effects of Crocin"\*, the system:



1\. Resolves \*\*Crocin → paracetamol\*\* by exact dictionary lookup (confidence 0.92, method `brand\_stem`)

2\. Filters a Chroma vector store to that composition

3\. Semantically ranks on the remainder of the question — \*"what are the side effects of"\*

4\. Routes to one of six response types by confidence, then generates a grounded answer



It also predicts which of four price brackets a product falls into from its ingredients, manufacturer, dosage form and pack size.



\---



\## Three findings



\### 1. The two published datasets share an `id` column that is not a shared key



Both Kaggle files carry an `id`. Joining on it gives 97.7% overlap — which looks like success and means nothing, because it is just the intersection of two sequential integer ranges.



Joining on `id` and then checking whether the two files' \*\*product names agreed\*\* gave \*\*86 matches out of 248,218 — 0.03%\*\*. The detail file has 5,755 fewer rows, so the two scrapes drift apart cumulatively:



```

id 76:  products = Amaryl 1mg     details = aztor 10

id 77:  products = Aztor 10       details = atorva 40

id 78:  products = Atorva 40      details = azax 500

```



Rebuilt on normalized product names: \*\*89.3% match rate\*\* (226,696 of 253,973). Therapeutic-class agreement within composition groups went from incoherent — a diabetes drug labelled ANTI INFECTIVES and indicated for fungal infections — to a \*\*median of 100%\*\*, with only 8 of 981 groups below 70%.



The broken version is kept in the repository as `stage1\_etl\_broken\_join.py`.



\### 2. Embedding failures on drug names are orthographic, not semantic



| Query | Retrieved | Actual relationship |

|---|---|---|

| crocin | crotamiton | Paracetamol vs a scabies treatment |

| dolo | doxazosin | Paracetamol vs a blood pressure drug |

| glycomet | glycopyrrolate | Metformin vs an anticholinergic |



Look-alike drug name pairs embedded \*\*1.86× closer\*\* than random pairs. No larger embedding model fixes this, because the model is being asked to do identity resolution — a lookup problem, not a similarity problem.



The redesign: \*\*identity by exact lookup, meaning by embedding.\*\*



\### 3. The molecule sets the price, not the brand



SHAP on the price classifier: composition \*\*1.183\*\* vs manufacturer \*\*0.266\*\* — a \*\*4.4× gap\*\*.



India is a branded-generics market, so brand premium was the expected driver. It is not, because brand premium operates \*within\* a composition (2–3× between labels) while compositions span orders of magnitude — paracetamol's median is ₹24, abatacept's is ₹30,000. Quartile tiers are too coarse to detect brand effects; those surface instead as the 85.3% adjacent-tier error rate.



\---



\## Architecture



```

Kaggle: 253,973 products          Kaggle: 248,218 detail records

(price, manufacturer,             (uses, side effects,

&#x20;composition, pack size)           substitutes, drug classes)

&#x20;        │                                    │

&#x20;        └──── join on normalized name ───────┘

&#x20;                 89.3% match rate

&#x20;                         │

&#x20;           ┌─────────────┴─────────────┐

&#x20;           │                           │

&#x20;   2,959 compositions           253,969 products

&#x20;   (consensus-filtered)         (feature-engineered)

&#x20;           │                           │

&#x20;    ┌──────┴──────┐            XGBoost + SHAP

&#x20;    │             │            price tier model

&#x20;Chroma index   Resolver

&#x20;2,857 docs     1,640 ingredients

&#x20;384-dim        1,639 spelling variants

&#x20;               238,002 product names

&#x20;               176,595 brand stems

&#x20;    └──────┬──────┘

&#x20;           │

&#x20;   filter-first retrieval

&#x20;           │

&#x20;   deterministic router

&#x20;REFUSE · CLARIFY · SCOPE ·

&#x20;ASSUME · PARTIAL · ANSWER

&#x20;           │

&#x20;   grounded, cited answer

&#x20;           │

&#x20;   FastAPI · Docker · ECS Fargate

```



\---



\## Retrieval evaluation



200-case gold set. Ground truth for brand and generic cases is generated \*\*from the source data\*\* — a product's own `composition\_key` — not hand-labelled. Brand cases are stratified by composition popularity so rare drugs are represented, not drowned by the top 20.



| Configuration | recall@1 | recall@3 | recall@5 | MRR |

|---|---|---|---|---|

| A — semantic only | 39.5 | 44.5 | 47.5 | 42.3 |

| B — resolver only | 92.5 | 96.5 | 96.5 | 94.2 |

| \*\*C — hybrid (shipped)\*\* | \*\*95.5\*\* | \*\*98.5\*\* | \*\*98.5\*\* | \*\*96.8\*\* |



The category breakdown is the real result:



| Category | A semantic | B resolver | C hybrid |

|---|---|---|---|

| brand (very common) | 13.3 | 100.0 | 100.0 |

| brand (rare) | 20.0 | 100.0 | 100.0 |

| generic | 100.0 | 100.0 | 100.0 |

| \*\*concept\*\* | \*\*100.0\*\* | \*\*25.0\*\* | \*\*100.0\*\* |

| typo | 83.3 | 100.0 | 66.7 |

| absent (refusal) | 100.0 | 83.3 | 83.3 |



The hybrid does not win by being uniformly better. Semantic search scores \*\*13%\*\* on brand names; lookup scores \*\*25%\*\* on concept queries. Each component covers the other's blind spot exactly.



Stated honestly: \*\*the embedding layer contributes about 2 points overall\*\* and is decisive only on concept queries. On brand and generic lookup, exact matching does the work.



\### Document construction



Composition records are lists of labels, not prose. Each is composed into a paragraph before embedding:



> \*"Paracetamol is a pain relief / analgesic medicine. It is used for pain relief and fever. Commonly reported side effects include nausea, vomiting and stomach pain (reported for up to 87% of products with this composition)... sold in India under 1,788 brand names... Retail prices range from Rs 2 to Rs 970, with a median of Rs 24."\*



Median document length is \*\*582 characters\*\*, 95th percentile \*\*796\*\*. Measured, then \*\*not chunked\*\* — splitting these would separate a drug from its own name, which is the failure chunking exists to prevent.



Consensus filtering replaced union aggregation: a value is kept only if it appears in ≥10% of a composition's products. Union let one mislabelled product inject garbage into a document shared by thousands. Median side effects per document fell from \*\*70.7 to 5\*\*, uses from \*\*33.7 to 1\*\*.



\---



\## Price tier classifier



Predicts one of four price quartiles from composition, manufacturer, dosage form, pack quantity and ingredient count. No price-derived features — verified by an automated leakage check on the final feature matrix.



| | Result |

|---|---|

| macro-F1 | 0.677 (baseline 0.254, lift +0.424) |

| Accuracy | 0.679 |

| Adjacent-tier errors | 85.3% |

| budget→specialty errors | 171 of 50,794 |



Per-tier F1: specialty 0.801, budget 0.747, premium 0.588, standard 0.573. The extremes are easy; the middle boundaries (₹79 vs ₹80) are arbitrary quartile cuts with no market meaning.



High-cardinality categoricals use \*\*out-of-fold target encoding with smoothing\*\*, fitted after the train/test split on training data only.



\---



\## Safety routing



Guardrails run in code before the model is called. A prompt is probabilistic; "never tell someone an unrecognised drug is safe" needs a guarantee.



| Route | Trigger | LLM? |

|---|---|---|

| REFUSE | Nothing resolved, or confidence < 0.60 | No |

| CLARIFY | One brand maps to multiple compositions | No |

| SCOPE | Clinical question (dosing, "should I") | No |

| ASSUME | Fuzzy match — states the assumption first | Yes |

| PARTIAL | Multiple drugs — no interaction data exists | Yes |

| ANSWER | Single drug, confidence ≥ 0.90 | Yes |



\*\*11 of 12\*\* routing tests pass; \*\*6 of 12\*\* never reach a model. Every case traces to an observed failure, not a hypothetical.



Live response from the deployed service for an unknown drug:



```json

{

&#x20; "route": "REFUSE",

&#x20; "route\_reason": "no drug name resolved",

&#x20; "confidence": 0,

&#x20; "resolved\_drugs": \[],

&#x20; "documents": \[],

&#x20; "answer": "I don't have data on that in this dataset... and no data is not the same as no risk.",

&#x20; "latency\_ms": 27

}

```



Retrieval had found five plausible candidates. The router returned none of them. A naive pipeline would have summarised lomefloxacin and flecainide into a confident answer about a drug it has never seen.



Prompts are versioned constants with changelogs. v2 added a proportions rule after a minority indication (18% of products) was presented with equal weight to an 82% one. v3 added a no-implied-safety rule after a draft answered a with-alcohol question in a way that read as reassurance.



\---



\## Deployment



Containerised FastAPI service on \*\*AWS ECS Fargate\*\* (us-east-1), 1 vCPU / 3GB, image in ECR.



The embedding model and Chroma index are baked into the \*\*1.37GB\*\* image, making the container stateless — no S3, no IAM binding, no startup fetch. Measured cost of that choice: \*\*53 seconds pulling the image, 2 seconds to start the container.\*\* Fetching the index from S3 was rejected; at 2,857 documents it adds an external dependency and startup latency for no benefit.



torch is installed from the CPU-only index. The default install pulls a 2.5GB CUDA build that would never execute on Fargate.



All four safety routes were verified on the live endpoint, then the task was torn down — Fargate has no scale-to-zero, and a persistent demo would cost \~$30/month for no additional signal.



Endpoints: `/health`, `/stats`, `/search`, `/product/{id}`, `/composition/{key}`, `/predict`, `/ask`.



`/ask` returns the route, the resolved drugs with methods and confidences, and the retrieved documents — not just an answer string. Retrieval that cannot be inspected cannot be trusted. It also means the service is useful with generation disabled, since the safety guarantees live in the router rather than the model.



\---



\## Limitations



\- \*\*Join coverage is 89.3%.\*\* 27,277 products have no detail record and are excluded from the corpus.

\- \*\*36,847 detail rows (14.8%) were discarded\*\* to ambiguous name keys.

\- \*\*Compositions are capped at two ingredients.\*\* The source has only `short\_composition1` and `short\_composition2`; Indian fixed-dose combinations often have three or four.

\- \*\*Dosage form is not in the join key.\*\* "Ascoril LS Syrup" can match a drops record. Including form would cut the match rate substantially — a conscious tradeoff.

\- \*\*Prices are a November 2022 snapshot\*\* and are not current.

\- \*\*Fuzzy matching resolves `esketamine` → `ketamine` at 0.70.\*\* Esketamine is a real, different drug. No threshold separates that from a genuine typo — 0.86 catches both, 0.92 catches neither — so it is surfaced as a stated assumption rather than silenced.

\- \*\*102 of 2,959 compositions (3.4%) carry quality flags\*\* and are excluded from the index.

\- \*\*All records are allopathic.\*\* The model says nothing about ayurvedic or homeopathic pricing.

\- \*\*No drug-drug interaction data exists in this corpus.\*\* Combination questions describe each drug and state that the combination cannot be assessed.



\---



\## How it was debugged



Retrieval reached 98.5% over four measured iterations, each driven by reading the failure list:



| Version | recall@5 | Change |

|---|---|---|

| v1 | 47.5% | Pure vector search baseline |

| v2 | 93.0% | Filter-first architecture; stopwords; exact-beats-contained hierarchy |

| v3 | 99.5% | Unioned indexes for ambiguous brands; 8-word spans |

| v4 | 98.5% | Spelling-variant index (typo recall for lookup 83→100, hybrid 83→67) |



\*\*v4 is the instructive one.\*\* Adding a variant index fixed the resolver and cost the hybrid two points, because ties at equal confidence broke on a meaningless semantic score. The evaluation harness caught it immediately; without one it would have shipped as an improvement.



A second case: a query for \*"medicine for high blood pressure"\* fuzzy-matched the word "medicine" to a brand called \*Medicaine\*, filtering a clean semantic query down to one antacid. The obvious fix — raise the threshold — was wrong. `medicine`/`medicaine` scores \~0.94, \*\*higher\*\* than the genuine typo `amoxicilin`/`amoxycillin` at 0.857. No threshold separates them. Restricting the \*search space\* to ingredient names fixed it; tightening the \*threshold\* only broke typo handling.



\---



\## Data sources



| Dataset | Records | Licence |

|---|---|---|

| \[A-Z Medicine Dataset of India](https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india) | 253,973 | CC BY-NC-SA 4.0 |

| \[250k Medicines: Usage, Side Effects, Substitutes](https://www.kaggle.com/datasets/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes) | 248,218 | CC BY-NC-SA 4.0 |



Both are scraped Indian retail pharmacy listings. Non-commercial use only. Raw data is not committed; download separately into `data/raw/`.



\*\*This is not a medical tool.\*\* It summarises what a retail catalogue records. It does not provide clinical guidance.



\---



\## Stack



Python · pandas · XGBoost · SHAP · sentence-transformers (`all-MiniLM-L6-v2`) · ChromaDB · FastAPI · Docker · AWS ECR + ECS Fargate



\---



\## Running it



```bash

pip install -r requirements.txt



python stage1\_etl.py           # clean, join, build composition corpus

python stage2\_model.py         # price tier classifier + SHAP

python stage3a\_documents.py    # compose retrieval documents

python stage3c\_retrieval.py    # build index, run regression tests

python stage3d\_evaluate.py     # 200-case retrieval evaluation

python stage4\_answer.py        # routing tests (dry run without an API key)



uvicorn app:app --port 8080    # API at http://localhost:8080/docs

```



Container:



```bash

docker build -t pharmalens:local .

docker run --rm -p 8080:8080 pharmalens:local

```



`stage3b\_embedding\_lesson.py` ships nothing. It exists to produce the evidence in finding 2, and is kept because the numbers justify the architecture.



\---



\## Further work



\- Predicting price \*\*within\*\* a composition would isolate brand premium, which quartile tiers are too coarse to detect

\- The two-ingredient ceiling could be lifted by parsing third ingredients from product names, where they appear as brand suffixes (SP, MR, Plus)

\- Typo handling currently comes from the semantic fallback rather than the fuzzy matcher (hybrid 66.7% vs lookup 100%); reconciling the two would close the last gap



