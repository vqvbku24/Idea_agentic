# PILOT_SPEC.md — Smoke test for idea-level prior-art retrieval (French → English, small sample)

**Purpose:** find out within ~1 week whether the research idea is feasible, using **inference only** (no training) and a **small sample**.
**Audience:** a coding agent. A human supplies the API key and writes the human-authored queries (§9).
**Status:** this spec has **not been run**. Verify every library name, model ID, URL and API behaviour at runtime; do not assume they exist as written.
**Smallest version:** if only a little evidence is needed, use the **mini profile in §16** (20 sources, about 2–3 days). It is an existence check, not a measurement.

---

## 0. Question this pilot answers

Given an *idea description* (English) that has been deliberately disguised, can standard retrievers find the French source paper in a mixed French/English corpus, and **where do they fail**?

It must produce numbers for four decisions (§12):
1. Is the task **saturated** (too easy)?
2. Is there **headroom** between easy and disguised queries?
3. Is the **language axis** real, or does an English abstract / document translation remove it?
4. Do **synthetic** queries agree with **human-written** queries?

Out of scope for this pilot: any training or fine-tuning, RL, agents, Vietnamese sources, large corpora.

---

## 1. Fixed parameters (put in `config/pilot.yaml`, then freeze)

```yaml
seed: 13
n_source: 60            # French source papers (dry run: 10)
n_fr_distractors: 300   # other French TALN papers, not used as sources
n_en_distractors: 3000  # English ACL Anthology papers (title + abstract)
years: [2015, 2024]     # stratify sources evenly across years
body_words_for_index: 1500
top_k: 10
bootstrap_resamples: 1000
variants: [V0, V1, V2, V3]
retrievers: [bm25, dense_bge_m3, hybrid_rrf, doc_translate_bm25, facet_rrf]
index_views: [native, en_abstract, both]   # plus "absent" handled at eval time
```

Dry-run rule: every script accepts `--n-source` so the whole pipeline can run on 10 papers first.

---

## 2. Priority table

| Priority | Task | File / function | Done when |
|---|---|---|---|
| **P0** | Tracer bullet: 10 sources + BM25 only, end to end | all scripts, `--n-source 10 --retrievers bm25` | `results/metrics.csv` exists with numbers |
| **P0** | Download French TALN papers + English distractors | `data_acl.py::list_jeptalnrecital_papers`, `list_english_distractors`, `download_pdf` | PDFs and metadata in `data/raw/` |
| **P0** | Extract French abstract, English abstract, body | `pdf_extract.py::split_abstracts` | `data/processed/papers.jsonl` + extraction stats |
| **P0** | Build corpus and the three index views | `corpus.py::build_index_views` | `corpus.jsonl`, per-view text files |
| **P1** | Facet extraction + disguised queries + validity judge | `facets.py`, `queries.py` | `queries.jsonl` with accepted/rejected counts |
| **P1** | Retrievers + evaluation | `retrievers.py`, `evaluate.py` | metrics with bootstrap CIs, per variant × view × retriever |
| **P1** | Report generator | `report.py` | `results/report.md` with decision table |
| **P2** | Facet-coverage audit (English abstract vs full text) on 30 papers | `facets.py::coverage_audit` | `results/facet_coverage.csv` |
| **P2** | Import human-written queries and compare | `queries.py::load_human_queries` | synthetic-vs-human table |
| **P3** | Optional: reranker, plots, natural-pair candidate search | `retrievers.py::rerank` (optional) | not required for the decision |

---

## 3. Repository layout

```
prior_art_pilot/
├── README.md
├── requirements.txt
├── .gitignore                 # data/, cache/, .env, results/raw
├── config/
│   ├── pilot.yaml             # §1  (frozen before first full run)
│   └── decision_rules.yaml    # §12 (frozen before first full run)
├── src/pilot/
│   ├── events.py              # typed JSONL event log
│   ├── llm.py                 # cached LLM wrapper
│   ├── data_acl.py
│   ├── pdf_extract.py
│   ├── corpus.py
│   ├── facets.py
│   ├── queries.py
│   ├── retrievers.py
│   ├── evaluate.py
│   └── report.py
├── scripts/
│   ├── run_01_download.py
│   ├── run_02_extract_build.py
│   ├── run_03_queries.py
│   ├── run_04_retrieve_eval.py
│   ├── run_05_facet_audit.py
│   └── run_06_report.py
├── data/{raw,processed,human}/
├── cache/llm/
├── results/
├── events.jsonl
└── tests/
```

Suggested libraries (verify availability): `pymupdf`, `acl-anthology` (or the official Anthology data repository), `lingua-language-detector` or `langdetect`, `bm25s` or `rank_bm25`, `sentence-transformers` or `FlagEmbedding` (for BGE-M3), `faiss-cpu`, `transformers` (MT), `pandas`, `numpy`, `scikit-learn` (AUC), `pyyaml`, `tqdm`.

---

## 4. Data acquisition — `data_acl.py`

**Source A (French):** the JEP-TALN-RECITAL venue on the ACL Anthology. Discover volumes programmatically from the Anthology metadata; do not hard-code paper IDs.
**Source B (English distractors):** English papers from the same Anthology, any venue except JEP-TALN-RECITAL.

```python
@dataclass
class PaperMeta:
    paper_id: str
    title: str
    year: int
    venue: str
    pdf_url: str | None
    abstract_meta: str | None   # metadata abstract, language unknown
    authors: list[str]

def list_jeptalnrecital_papers(years: tuple[int, int]) -> list[PaperMeta]: ...
def list_english_distractors(n: int, seed: int, exclude_venue: str = "jeptalnrecital") -> list[PaperMeta]: ...
def download_pdf(meta: PaperMeta, dest_dir: Path) -> Path | None:
    """Rate limit <= 1 request/sec, retries with backoff, skip existing files,
    identifying User-Agent. Never download more than needed."""
```

Rules:
- English distractors need **title + abstract only** (no PDFs). Detect the abstract language and keep only English.
- Download PDFs only for French sources and French distractors.
- Log `PdfDownloaded` / `DownloadFailed` events. A failed download must not stop the run.

---

## 5. Extraction — `pdf_extract.py`

```python
def extract_text(pdf_path: Path) -> str: ...            # pymupdf, page order preserved

def split_abstracts(text: str) -> dict:
    """Return {fr_abstract, en_abstract, body, provenance}.
    Heuristic: on the first 1-2 pages find a 'Résumé' header and an 'Abstract' header,
    take text until the next section/keywords line.
    provenance in {'pdf', 'metadata', 'missing'} for each abstract."""

def detect_lang(text: str) -> tuple[str, float]: ...    # (iso code, confidence)
```

**Source-paper eligibility** (all must hold, otherwise the paper is a distractor or dropped):
1. body language is French (confidence ≥ 0.9)
2. French abstract and English abstract both extracted, `provenance == 'pdf'`
3. body ≥ 1500 words
4. sample is stratified by year

**Do not silently repair missing English abstracts with machine translation.** If an English abstract is machine-generated, mark `provenance: 'mt'` and **exclude that paper from the `en_abstract` view**. The point of that view is the *author-written* English abstract.

Report in `results/g1_stats.csv`: number of papers scanned; share with French body; share with extractable English abstract; share eligible; failures by reason.

---

## 6. Corpus and index views — `corpus.py`

`corpus.jsonl` fields: `doc_id, kind (source|fr_distractor|en_distractor), lang, title, abstract_fr, abstract_en, body_fr, year`.

```python
def build_index_views(docs) -> dict[str, dict[str, str]]:
    """Return {view_name: {doc_id: text}} for views:
       native      : French docs -> title + abstract_fr + first 1500 words of body_fr
                     English docs -> title + abstract_en
       en_abstract : French docs -> title + abstract_en   (author-written only)
                     English docs -> title + abstract_en
       both        : native text + "\n" + en_abstract text for French docs
    """
```

**Absent control** is not a stored view: at evaluation time, drop the source's own `doc_id` from the candidate list for that query (`evaluate.py::rank_without_source`). Used for the score-separation test in §10.

Note the asymmetry: English distractors only have abstracts. Log it in the report; do not hide it.

---

## 7. Queries — `facets.py` and `queries.py`

### 7.1 Facet extraction (one LLM call per source paper)

Input: French abstract + English abstract + first 1500 words of body. Output JSON, English, ≤ 25 words per facet:

```json
{"problem": "...", "mechanism": "...", "setting": "...", "outcome": "...", "key_terms": ["...", "..."]}
```

### 7.2 Variants (English queries, ≤ 80 words each)

| Variant | Construction |
|---|---|
| **V0** | Neutral idea statement from the four facets (control, no disguise) |
| **V1** | **D1** terminology drift + **D5** scrub: rewrite so none of `key_terms` appear; use descriptions, hypernyms or another community's vocabulary; keep all four facets |
| **V2** | **D2** facet subset + **D5**: keep exactly two facets chosen with the seeded RNG; record which are kept |
| **V3** | **D3** round trip EN → DE → EN, then **D5** |

**D5 scrub:** remove proper names, acronyms, dataset/system names and numbers; replace with generic descriptions.

### 7.3 Validity judge (separate LLM call, ideally a different model from the generator)

```json
{"preserved_facets": ["problem", "outcome"], "leaks_identifiers": false, "fluent": true}
```
Accept a variant only if `preserved_facets` ⊇ the facets it should keep, `leaks_identifiers == false`, and `fluent == true`. Store rejected variants with the reason.

Plus a deterministic leak check (always computed, reported):

```python
def lexical_leak(query: str, title: str) -> float:
    """Share of distinct non-stopword title tokens (len>=5) that appear in the query."""
    toks = {t for t in re.findall(r"\w+", title.lower()) if len(t) >= 5}
    q = set(re.findall(r"\w+", query.lower()))
    return len(toks & q) / max(1, len(toks))
```
Flag variants V1–V3 with `lexical_leak > 0.3` and report them; do not silently delete them.

### 7.4 Human-written queries — `load_human_queries(path)`

Read `data/human/human_queries.csv` (§9). Mark `variant = "H"`. Never use these queries for any tuning.

---

## 8. Retrievers — `retrievers.py`

All retrievers expose `rank(query: str, view: str) -> list[tuple[doc_id, score]]`, top-k = 10.

| Name | Description |
|---|---|
| `bm25` | Lowercase, unicode-normalized, accent-stripped tokens; no stemming; one index per view |
| `dense_bge_m3` | A multilingual dense model (default `BAAI/bge-m3`; verify it loads); embeddings cached to disk; cosine similarity; document max length ≥ 1024 tokens |
| `hybrid_rrf` | Reciprocal rank fusion of `bm25` and `dense_bge_m3` |
| `doc_translate_bm25` | Translate French docs' `native` text to English once at index time with an open MT model (default `Helsinki-NLP/opus-mt-fr-en`; verify), then BM25 on English-only text |
| `facet_rrf` | LLM splits the query into ≤ 4 facet sub-queries (JSON list), run `hybrid_rrf` per sub-query, fuse with RRF |

```python
def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

Optional (P3): cross-encoder reranker on the hybrid top-50, only if time allows.

---

## 9. Human tasks (not for the agent)

1. **Human-written queries.** Pick 30 source papers **not shown** to you as LLM variants. Write one English description of each paper's idea **without LLM help**, in your own words, without copying title words. Save as `data/human/human_queries.csv`:
   `source_id,query_text,written_by,date`
2. **Natural-pair candidates.** For 20 source papers, search by title and authors for an English counterpart (journal extension or translation by the same group). Save `data/human/natural_pairs.csv`:
   `source_id,candidate_url,same_contribution (yes/no/unsure),notes`
3. **Spot-check.** Read 30 randomly chosen accepted LLM variants and mark whether they are faithful and leak-free (`data/human/spotcheck.csv`).

---

## 10. Evaluation — `evaluate.py`

For every (retriever, view, variant) with the source present in the index:

- `Recall@1/5/10`, `MRR`
- `false_gap@10 = 1 − Recall@10` (the user would conclude "nothing covers this idea")
- bootstrap 95% CI over **source papers**, not over queries

```python
def bootstrap_ci(per_paper_values: np.ndarray, n=1000, seed=13):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_paper_values), size=(n, len(per_paper_values)))
    means = per_paper_values[idx].mean(axis=1)
    return per_paper_values.mean(), np.percentile(means, [2.5, 97.5])
```

**Absent control:** for each query rank the corpus twice, with and without the source. Compute the AUC of the top-1 score separating "present" from "absent". A high AUC means the system can tell when the source exists; a low AUC means it is likely to report a false match or a false gap.

Report tables for: variant × retriever (view `both`), view × retriever (variant V2 and V3), and the same for the human set `H`.

---

## 11. Facet-coverage audit (RQ6) — `facets.py::coverage_audit`

For 30 eligible sources, an LLM judge labels for each of the four facets whether it is clearly present in (a) the author-written English abstract and (b) the French body (first 1500 words). Output `results/facet_coverage.csv` with columns `paper_id, facet, in_en_abstract, in_fr_body`. Summarize the share of facets present in the body but missing from the English abstract. Hand-check 10 papers and record agreement.

---

## 12. Decision rules — `config/decision_rules.yaml` (**freeze before the first full run**)

All thresholds are **starting heuristics**, not established values. Commit the file before running; never edit it afterwards for this pilot.

```yaml
saturation:    {metric: "recall@10", threshold: 0.95, variants: [V1, V2, V3], retrievers: "all strong"}
headroom:      {min_drop_points: 15, from: V0, to: "hardest of V1-V3"}
language_gap:  {max_points: 5, comparisons: ["native vs en_abstract", "best retriever vs doc_translate_bm25"]}
human_vs_syn:  {max_relative_drop: 0.30, metric: "recall@10"}
facet_gain:    {min_points: 5, over: "hybrid_rrf", ci_excludes_zero: true}
g1:            {max_en_abstract_share: 0.90, min_natural_pairs: 100}
```

| Outcome | Interpretation | Next step |
|---|---|---|
| All strong retrievers ≥ 0.95 Recall@10 on V1–V3 | Task saturated | Make it harder, move to real labels, or stop |
| Drop from V0 to hardest variant ≥ 15 points, with non-trivial false_gap | Room for a benchmark | Proceed with Stage A |
| `native` vs `en_abstract` and doc-translation within 5 points | Language axis weak | Reposition on terminology/facet shift |
| Human-written recall far below synthetic (> 30% relative) | Synthetic labels unreliable | Fix generation before proceeding |
| `facet_rrf` beats `hybrid_rrf` by ≥ 5 points with CI excluding 0 | Signal for facet-aware method | Consider Stage B |
| > 90% of French sources have an English abstract, and < 100 natural pairs | G1 triggered | Natural labels supplementary only; narrow cross-lingual claim |

`report.py` must print this table filled in with measured values and **state "insufficient data"** rather than a verdict when a number is missing.

---

## 13. Logging and reproducibility

- `events.jsonl`, one typed event per line: `SourceSelected, PdfDownloaded, AbstractExtracted, FacetsExtracted, QueryGenerated, QueryRejected, RetrieverRun, MetricComputed`.
- Every LLM call goes through `llm.py::complete(prompt, system=None, temperature=0.0)` and is cached on disk by a hash of (model, system, prompt). Provider and model come from environment variables (`LLM_PROVIDER`, `LLM_MODEL`), never hard-coded.
- Fix all random seeds from `config/pilot.yaml`. Record library versions, model IDs and the git commit in `results/run_meta.json`.

---

## 14. No-touch zones

| Do not | Reason |
|---|---|
| Train or fine-tune anything | Pilot is inference only |
| Tune prompts, thresholds or retrievers on the human queries `H` | `H` is the frozen test of whether synthetic results transfer |
| Edit `config/decision_rules.yaml` after the first full run | Prevents moving the goalposts |
| Replace missing author-written English abstracts with MT and still use the `en_abstract` view | Invalidates the language-gap test |
| Delete flagged or rejected queries without reporting them | Counts of rejects and leaks are results |
| Evaluate on the same papers used to choose prompts | Use a separate 5-paper development subset, excluded from the 60 |
| Hit the Anthology servers faster than ~1 request/second, or scrape sites that disallow it | Rate and terms-of-use |
| Commit API keys, PDFs or `data/` to git; redistribute PDFs | Credentials and licences (check licence per paper before sharing anything) |
| Add Vietnamese sources, new retrievers or an agent loop | Out of scope for the smoke test |

---

## 15. Definition of done

- [ ] Tracer bullet (10 papers, BM25) ran end to end
- [ ] `g1_stats.csv`, `papers.jsonl`, `corpus.jsonl` produced; extraction failure reasons listed
- [ ] ≥ 40 eligible sources (target 60); accepted/rejected variant counts reported
- [ ] `metrics.csv` with bootstrap CIs for all (retriever × view × variant)
- [ ] Absent-control AUC reported
- [ ] Facet-coverage audit done on 30 papers, 10 hand-checked
- [ ] Human-written set evaluated (after you write it)
- [ ] `report.md` fills the decision table, lists leaks/rejects, and states limitations (small N, asymmetric English distractors, LLM familiarity with ACL papers)

**Estimated effort (planning guess):** about 5–7 working days, dominated by extraction debugging and writing the human queries.

---

## 16. Mini profile (existence check, smallest useful version)

**What it can show:** the pipeline works end to end; French papers with an English abstract can be obtained; a plain query is found but disguised queries are sometimes **missed** (a failure exists); concrete failure cases to show the advisor.
**What it cannot show:** failure *rates*, differences between retrievers, the *size* of the language gap, or any gain from facet methods. With 20 papers one paper is 5 percentage points, and a 5-of-20 miss rate has a 95% exact interval of roughly 9–49%. Do not write numbers from this profile as if they were estimates.

### 16.1 Parameters (`config/pilot_mini.yaml`, freeze before the run)

```yaml
seed: 13
n_source: 20             # floor 15; below 10 the result is an anecdote
n_fr_distractors: 100
n_en_distractors: 500
years: [2015, 2024]      # about 2 sources per year
variants: [V0, V1, V2, V3]
retrievers: [bm25, dense_bge_m3, hybrid_rrf, doc_translate_bm25]   # no facet_rrf, no reranker
index_views: [native, en_abstract]
human_queries: 10        # written by you, never tuned on
facet_audit_papers: 10   # hand-check all 10
top_k: 10
```

Everything else in §4–§11 and the no-touch zones in §14 apply unchanged. Use a separate development subset of **3** papers (outside the 20) for prompt debugging.

### 16.2 Evidence criteria (`config/decision_rules_mini.yaml`, freeze before the run)

These are descriptive existence criteria, not statistical tests. Report counts as "k of n" with exact (Clopper-Pearson) intervals.

| ID | Check | Counts as supporting evidence if | If not |
|---|---|---|---|
| E1 | Pipeline and data | ≥ 15 eligible sources and ≥ 60 accepted queries | Fix extraction or enlarge the candidate pool; report the failure reasons (this is itself a G1 result) |
| E2 | Sanity | `hybrid_rrf` finds the source in the top 10 for ≥ 90% of V0 queries | Suspect a bug, leaky distractors or a corpus problem; do not interpret anything else |
| E3 | A failure exists | On the hardest of V1–V3, the best retriever misses the source (top 10) for at least 5 of 20 sources | Task looks saturated at this size; harder variants or real labels are needed before continuing |
| E4 | Language (descriptive only) | Report how many sources are found under `native` but not `en_abstract`, and the reverse, and whether `doc_translate_bm25` closes the gap | No threshold; if the counts are near zero, note that the language axis looks weak |
| E5 | Human queries (descriptive only) | Report hits and misses among the 10 human queries; note whether the pattern matches the synthetic one | If human queries are all found while synthetic ones fail, the synthetic disguise may be unrealistic |
| E6 | Facet coverage (descriptive only) | Report how many of the 10 audited papers have at least one facet in the French body that is absent from the English abstract | No threshold |

The heuristics (90%, 5 of 20) are starting values chosen for a small sample and are not established.

### 16.3 Required output: failure cases

Write `results/failure_cases.md` with one entry per miss (top 10) from V1–V3 and H:

```
source: <title>, <year>
variant: V2 (kept facets: problem, outcome)
query: <text>
rank of source: 37 (hybrid_rrf, native)
top-3 retrieved: <titles>
why it likely failed: <one human-written line>
```

This file is the main artefact to show the advisor. A handful of clear, honest failure cases is more convincing than a table of numbers at n = 20.

### 16.4 Time and ordering

1. Day 1: tracer bullet on 10 papers, BM25 only (§2, P0).
2. Day 1–2: extraction, corpus and queries for 20 papers; you write the 10 human queries in parallel.
3. Day 2–3: retrievers, evaluation, failure cases, facet audit on 10 papers.

**Stop rule:** if E1 fails (fewer than 15 eligible sources), report that and stop. Do not lower the eligibility rules to reach 20.
