# Research Proposal: Idea-Level Prior-Art Recovery under Terminology and Language Shift

**Working title:** *Finding the Same Idea in Another Language: Idea-Level Prior-Art Retrieval under Terminology and Language Shift*
**Project lineage:** Disguise-and-Seek → Robust Prior-Art Recovery (v1.0) → this consolidated proposal
**Realistic target venues:** ACL Findings / workshop (ACL Main only if the method results are strong, see §15)
**Status:** Research proposal — **no experimental results are claimed**
**Version:** 2.7 — 27 October 2026 (v2.6 + staged plan A → B → C for advisor discussion, ResearchOS stage 12–13 positioning, false-gap-rate metric)

---

## 1. Executive summary

Researchers constantly ask "has this been done before?", yet automatic novelty and literature tools mostly search English-language literature. Prior work written in Vietnamese or French (for example in a national journal or a TALN proceeding), or described with different terminology, can be missed, and "not found" is then mistaken for "does not exist".

This project studies **idea-level prior-art recovery**: given a description of a research idea whose surface form differs from the source paper (different terminology, different subset of the contribution, different language), can a retrieval system recover papers that already contain the contribution, and where does it fail?

The consolidated design differs from the earlier drafts in three deliberate ways:

1. **Two-part contribution: benchmark + method.** The task and its labels (three-tier, graded, per-shift evaluation) are the foundation. Generic agentic search, RL search policies, adaptive stopping and proposer–solver self-play are already crowded, so the method contribution is narrowed to something the benchmark makes testable: a **facet-conditioned agentic search policy** (§9).
2. **Labels come in three tiers**: real cross-lingual same-contribution pairs, citation-derived acceptance sets, and synthetic planted labels. Planted labels alone would reduce the task to paraphrase retrieval.
3. **The work is staged (§1.1).** Stage A (benchmark and inference-only analysis) is the committed core. Stage B (facet-aware retrieval and a zero-shot agent) and Stage C (the FacetHunter policy with SFT/RL) are conditional on pilot gates. If the gates fail, the Stage A paper stands on its own.

The project is built as **pilot first, training later**, with explicit kill criteria (§13).

---

### 1.1 Staged research plan (for advisor discussion)

**Positioning.** The project can be read as a research module for two stages of the advisor's ResearchOS workflow: **stage 12** (competing-review analysis: find existing surveys or papers that already cover a scope) and **stage 13** (research-gap candidates: before a gap is called plausible, search by keyword, synonym, semantics and citations). Both ask whether something already exists when it may be worded differently. The failure this project measures is a **false gap**: the system reports that nothing covers an idea although a valid paper is in the corpus. This mapping is an interpretation of ResearchOS, not something it states.

| Stage | What is done | Experiments | Compute (to size with advisor) | Output | Gate to continue |
|---|---|---|---|---|---|
| **A. Benchmark + analysis** (committed) | Corpus and three-tier labels (§6), obfuscation families D1–D3 and D5 (§7), existing retrievers (§10.1), failure analysis by shift type and index condition, false-gap rate (§11), small frozen human-written set | **Inference only**, no training; on the order of 100 configurations (e.g. 6 retrievers × 4 transformation families × 4 index conditions) | Embedding/API calls and annotation effort; little or no GPU | Benchmark + failure analysis; realistic venues Findings / EACL / LREC / workshop | G1 data audit passes; strong retrievers do **not** saturate on moderate/hard cases; false-gap rate is non-trivial |
| **B. Facet-aware retrieval + zero-shot agent** (conditional) | Facet representation and graded matching (§8); a prompted agent with search/rewrite/translate/compare actions at matched budget (§9.1, 9.3 levels 3–4) | Light fine-tuning of a small encoder; matched-budget agent runs; oracle-routing analysis | Small GPU plus API | Method result on top of Stage A (or a stronger analysis paper) | G2 oracle headroom and G3 facet-aware gain (§13) |
| **C. FacetHunter with SFT/RL** (conditional) | Facet-conditioned policy trained on filtered trajectories, RL only if SFT shows headroom (§9.5–9.6) | Multiple seeds, ablations, held-out transformations, T-human transfer | GPU rollouts; largest cost and instability | Method paper candidate (ACL/EMNLP Main only if G4 and G5 hold) | G4 Pareto and G5 transfer |

**Planning estimates** (to revise with the advisor): Stage A about 4–6 weeks including the §13 pilot; Stage B about 4 further weeks; Stage C about 4–8 further weeks. Stages are decided at gates, not committed in advance.

**Decisions requested from the advisor:** (1) agreement that Stage A is the committed scope and B/C are conditional; (2) whether the ResearchOS stage 12–13 framing is acceptable; (3) available compute for Stages B and C; (4) language scope and permitted Vietnamese sources.

---

## 2. Motivation and rationale for choosing this topic

### 2.1 Why not automatic survey writing, and why not the ResearchOS workflow as the core contribution

**What was on the table.** Two earlier directions came from the advisor: (1) an agentic system that writes a survey automatically, and (2) *Research Survey OS* ("ResearchOS"), a proposed AI-assisted literature-review workflow (search strategy, deduplication, screening, verification, citation snowballing, structured extraction, taxonomy, research-gap candidates) built around human-approval gates and a traceable event history. This section records why neither is the core of this proposal and what was kept from both.

**Automatic survey writing is a crowded space.** A first literature check found an active cluster of systems: AutoSurvey, SurveyForge, SurveyX, LLM×MapReduce-V2, SurveyG (multi-agent, hierarchical citation graph; Nguyen et al., arXiv 2510.07733, reported to outperform existing survey-generation frameworks), Deep Academic Survey (DAS), and evaluation benchmarks such as SurveyBench. Only SurveyG was re-checked in the latest editing pass. Competing here as a sole contribution is judged infeasible for one student within one or two semesters, and it would not isolate what is new relative to existing systems.

**ResearchOS itself does not ask for automatic writing.** Its §24 ("Non-Goals for First MVP") lists "automatic paper writing" among the things not to prioritize, and its §25 ("Success Criteria") states: "The main objective is not to automatically write the survey. The objective is to create a reproducible, evidence-grounded, AI-assisted research survey workflow." Moving away from automatic writing is therefore consistent with the advisor's own document.

**Why the ResearchOS workflow was not adopted as the core contribution:**

1. **A different type of contribution.** ResearchOS is scoped as a systems MVP. It does define an evaluation (§22: manual search vs. a single LLM-generated query vs. the workflow, measured by gold-seed recall, relevant-paper recall, CORE papers found, papers screened, citation-expansion yield, human time and reproducibility) and screening baselines (§8). What it does not state is a falsifiable research hypothesis with pre-set decision thresholds; its §25 checklist asks reproducibility and traceability questions ("What did we search? Why? Can another researcher reproduce the process?"). Such a hypothesis could be added, so this is a difference of emphasis, not a flaw.
2. **The adjacent tool space is populated.** AI-assisted systematic-review tools already overlap with ResearchOS's screening and extraction stages: ASReview (already named in ResearchOS §8 as a baseline) and AutoLit/Nested Knowledge, whose validation paper (Kallmes et al., *Cochrane Evidence Synthesis and Methods*, October 2025) was retracted in June 2026 for a conflict-of-interest policy breach, not because its results were found invalid (Retraction Watch, 13 July 2026). LLAssist and a multi-agent human-in-the-loop clinical-trial SLR system were mentioned in earlier drafts and not re-checked. A ResearchOS thesis would have to state what is new beyond engineering quality relative to this cluster.
3. **Domain mismatch.** ResearchOS's worked example, *Foundation Models for Visual Structural Defect Inspection* (crack detection), is civil-engineering/computer-vision, unconnected to the multilingual NLP prior-art question already under development through Disguise-and-Seek.

**What was kept, not discarded.** Several ResearchOS principles are adopted directly and credited by name: saturation-based stopping instead of a fixed search budget alone (§9.2, Direction C; ResearchOS §10 snowballing saturation); a typed, replayable event log as the search certificate (§9.2, Direction E; ResearchOS §18 research memory); and the stance that a seed-recall check is "a search-quality test, not proof that every relevant paper has been found" (ResearchOS §11), which parallels this proposal's rule that negative judgments are always conditional (§3). The "AI assists, human decides" philosophy is reflected in the gates of §13 and in the rule that the agentic claim is dropped if it does not earn its cost (§9.3). The redirection is: keep the design principles the advisor pointed to, place them inside a narrower and falsifiable research question, and drop automatic writing, which ResearchOS itself already recommends.

### 2.2 The practical problem
- Every new research topic starts with a prior-art check; doing it by hand across venues and languages is slow and error-prone.
- Automated novelty checkers (idea-novelty and literature-comparison agents) help, but search mostly English literature.
- The gap between **"not found"** and **"does not exist"** is the seed of the project. A verifiable approach to measuring that gap has practical value for literature review and for the evidence-gathering stage of novelty assessment.

### 2.3 Why this idea was selected

| Selection criterion | How this idea scores | Evidence / caveat |
|---|---|---|
| **Verifiable ground truth** | Positive targets can be checked automatically (source ID, citation sets, real bilingual pairs) | Planted source ≠ only valid prior art; handled with acceptance sets (§6) |
| **Cheap to falsify** | A no-training pilot of ~4 weeks can show whether the task is hard and whether anything beats strong baselines | Gates in §13 |
| **Fit with local resources** | Vietnamese and French scholarly text is under-served by English-centric tools | Corpus availability and licensing must be verified first (largest feasibility risk) |
| **Honest agentic angle** | Search language, corpus, query form and stopping are real decisions, and the agentic claim can be dropped if it does not help | Control experiment in §9 |
| **Not dependent on a heavy RL system** | Phase 1 needs only retrieval models and LLM-assisted data construction | RL is deferred |

### 2.4 Alternatives considered and why they were not chosen

| Alternative | Why not the main plan |
|---|---|
| "A general scientific search agent trained with RL" | Crowded; weak novelty; hard to isolate what helped |
| "Planted-source benchmark only (LLM obfuscates paper P, system finds P)" | Reduces to paraphrase retrieval; very close to cross-lingual plagiarism detection and PAN source retrieval; may saturate on strong multilingual retrievers |
| "Proposer–Hunter self-play as the core claim" | Self-play for search agents already exists; high instability and compute cost |
| "Full novelty assessor" | Cannot be verified: failing to retrieve a source never proves novelty |

### 2.5 Scope and non-goals

**In scope:** recovery of relevant prior-art papers from a defined multilingual corpus; controlled and human-written obfuscations; full vs partial contribution match; auditable search traces; English, Vietnamese and French as the initial languages, subject to corpus feasibility.

**Out of scope:** claiming that an idea is globally novel because nothing was retrieved; replacing expert novelty assessment or peer review; treating an unsupported LLM "no prior art" verdict as ground truth; building a paper-writing AI scientist.

---

## 3. Problem formulation

**Input:** an idea description *I* (any supported language), a corpus *C* of papers with language and metadata, and a search budget *B*.
**Output:** a ranked list of candidate papers, a match grade per candidate (**full / partial / none**) with facet-level justification, and a **search certificate**.

**Contribution representation.** A paper's contribution is described by a small facet set (initial design, to be validated, not assumed):

```
z(P) = { problem, mechanism, setting, outcome }
```

**Match grades.**
- **Full:** the candidate instantiates all facets stated in *I*.
- **Partial:** the candidate instantiates some but not all stated facets.
- **None:** topical neighbour only (a hard negative).

**Claim boundary.** Negative judgments are always conditional ("this candidate does not cover these facets in this corpus"), never proofs of non-existence.

---

## 4. Related work and novelty positioning

This is a **first-pass audit** and must be redone with full papers before submission. Bibliographic details below must be verified.

### 4.1 Closest lines of work

| Line of work | What it provides | What remains different here |
|---|---|---|
| **Cross-language plagiarism detection (CLPD)**, e.g. CL-ASA, CL-CTS and later embedding/LLM-based methods | Cross-lingual detection of reused content; mature evaluation practice | Here the query is an **idea**, not a reused text; the target is **contribution-level prior art** with full/partial grades |
| **PAN plagiarism shared tasks** (source retrieval, text alignment; artificial obfuscation including random, translation and summary obfuscation) | Obfuscation taxonomy and source-retrieval protocol, closest ancestor of D1–D5 | LLM-era obfuscation, scientific-contribution facets, held-out transformation families, human-written evaluation |
| **Faceted scientific retrieval** (CSFCube, aspect-based document similarity, multi-vector aspect models) | Facet-level similarity of papers (background / method / result) | Facets under **terminology and language shift**; graded matching with hard topical negatives |
| **CLIRudit** (Valentini et al., MRL 2025): English author keywords → French abstracts from Érudit | First English–French academic CLIR dataset; benchmark of sparse/dense retrievers with and without MT | Query = same paper's own keywords, so it measures cross-lingual *document* retrieval, not idea-level prior art by *other* papers; **closest cross-lingual academic retrieval baseline and a candidate French corpus** |
| **UniFAR** (Dou et al., arXiv Feb 2026): unified facet-aware retrieval for doc–doc and question–doc scientific retrieval | Learnable facet anchors, multi-granularity aggregation, facet-aware joint training | Facet-aware retrieval itself is **no longer novel**; appears monolingual and not built for obfuscation or graded prior-art labels (verify before claiming); strongest facet-aware baseline candidate |
| **MAPLE**, "Can Retrievers Find the Same Paper from Different Aspects?" (Wei, arXiv 2608.15624, 16 Aug 2026): expert-validated multi-aspect full-paper retrieval benchmark, plus MAPLE-Synth query generation from OpenReview discussions | Tests whether retrievers recover the *same paper* from queries targeting its motivation, method and experimental findings; strongest model reported at 98.1% AnyAspect@20 but only 15.7% AllAspect@20 | **Closest benchmark to D2 (facet subsetting).** Appears English-only and has no obfuscation, cross-lingual or graded prior-art axis (verify from the full paper); its any-aspect vs. all-aspect gap suggests difficulty sits in multi-facet consistency, not single-facet recall |
| **Chain of Retrieval / PRISM** (Park et al., arXiv 2507.10057; v1 titled PRISM) with **SciFullBench** | Paper-to-paper retrieval that decomposes a query paper into aspect-specific views and iteratively expands search | Multi-aspect decomposition plus iterative search expansion is close to FacetHunter's mechanism; not cross-lingual or obfuscation-based (verify); **baseline candidate and novelty risk for the method claim** |
| **DORIS-MAE** (multi-level aspect-based scientific queries, NeurIPS 2023) and **FaBle** (multi-facet blending, LLM-synthesized facet-specific training pairs, Do et al., ACL 2025) | Complex aspect-based query benchmark; synthetic facet-level training data | Same facet idea in English only; FaBle informs how to build facet-level training pairs (§8) |
| **Patent prior-art search** (CLEF-IP and cross-lingual patent retrieval) | Long-standing prior-art task framing | Scientific papers, low-resource language pair, LLM-generated obfuscation |
| **Multilingual / cross-lingual retrieval** (e.g. BGE-M3, CLIR benchmarks) | Strong baselines | Used as baselines; the claim is about failure modes at the contribution level |
| **Agentic search** (e.g. Agentic-R, AutoSearch, search-agent RL) | RL-trained agents, adaptive depth, agent–retriever co-training | Not a competitor on novelty; used as baselines and as the basis for Phase 2 |
| **Scientific claim-to-source retrieval** (e.g. CheckThat! 2026 Task 1 systems) | Source retrieval for implicit scientific claims | Compare task setup, languages and label provenance |

Verified during preparation of this document: Agentic-R (Liu et al., Findings of ACL 2026, arXiv 2601.11888) trains a retriever for agentic search using local relevance and global answer correctness with iterative agent–retriever optimization. Also verified: **CLIRudit** (Valentini, Kozlowski, Larivière, MRL 2025, arXiv 2504.16264), **UniFAR** (Dou et al., arXiv 2602.23766; abstract checked, experimental details not), **DORIS-MAE** (Wang et al., NeurIPS 2023, arXiv 2310.04678), and **FaBle** (Do, Ryu, Kim, Lee, originally arXiv 2412.01443, published as Do et al., ACL 2025, aclanthology.org/2025.acl-long.1388). Also checked at abstract level in v2.5: **MAPLE** (Wei, arXiv 2608.15624), **Chain of Retrieval/PRISM** (Park et al., arXiv 2507.10057), **SurveyG** (arXiv 2510.07733) and the AutoLit retraction (Retraction Watch, 13 July 2026); methods and results of these were not read. Other 2026 works listed in earlier drafts (PaperSearchQA, AutoSearch, XBCP, Claim2Source, mRewriter-R1) were **not independently verified**; mRewriter-R1 could not be located in a single search and must be checked before being cited.

### 4.2 What is *not* defensible as the main novelty
- "An LLM agent searches papers."
- "The agent decides when to stop."
- "The agent translates or rewrites queries."
- "Trained with RL and verifiable rewards."
- "Proposer–solver / self-play."
- "English-only search misses non-English papers."

### 4.3 The candidate gap (to be tested, not asserted)
**Idea-level prior-art retrieval with naturally occurring cross-lingual labels, evaluated under held-out obfuscation families and human-written queries, with facet-aware graded matching.** The delta versus CLPD/PAN should be statable in three sentences; if it cannot be, the project should be redirected (§13). After adding UniFAR and CLIRudit, the defensible delta is the **combination**: facet-aware matching under obfuscation and language shift with graded prior-art labels. Neither facet-aware retrieval nor cross-lingual academic retrieval is novel on its own. MAPLE and Chain of Retrieval (§4.1) must be read in Week 0–1: MAPLE already tests recovering the same paper from different aspect queries, and Chain of Retrieval already decomposes by aspect and expands search iteratively. The defensible delta therefore narrows to what they do not appear to have: obfuscation under terminology and language shift, graded prior-art labels with citation-derived and natural provenance, and a facet-state policy evaluated against a strong fixed pipeline. If either already covers cross-lingual or obfuscated queries, the gap narrows again.

---

## 5. Research questions and hypotheses

| RQ | Question | Hypothesis | Evidence needed |
|---|---|---|---|
| RQ1 | Which shifts (terminology, facet omission, language, abstraction) damage strong retrievers most? | Failure patterns differ by shift type and are diagnosable | Per-family, per-language breakdown at matched index conditions |
| RQ2 | Does facet-aware graded retrieval beat strong dense/hybrid + rerank baselines? | Gains concentrate on partial descriptions and held-out transformations | Recall@k/MRR, ablations |
| RQ3 | Do results on synthetic obfuscations transfer to human-written and naturally occurring pairs? | The gap is limited and predictable | T-syn vs T-human vs T-natural |
| RQ4 | Can the system separate full prior art from partial overlap? | Facet supervision reduces false "full" claims on topical hard negatives | Graded labels, confusion matrices |
| RQ5 *(method, gated)* | Does a search policy that conditions its actions on unresolved-facet state beat a strong fixed pipeline **and** an adaptive agent without facet state at equal budget? | Yes, if unresolved-facet state drives action choice; contingent on oracle-routing headroom | Recall–cost Pareto curves; agency ladder; facet-state ablation |
| RQ6 | Are failures caused by facets that are missing from the indexed text (e.g. short English abstracts) rather than by language itself? | Facet coverage of the indexed text predicts retrieval success better than language does | Facet-coverage analysis across the index conditions in §6.4 |

Hypotheses must not be written as results in the final paper.

---

## 6. Data and benchmark design

### 6.1 Corpus
Start with English, French and Vietnamese only if source coverage, licensing and metadata are adequate. Candidate sources: ACL Anthology (including TALN/RECITAL proceedings) and the publicly released CLIRudit/Érudit data for French (note: Érudit is a Quebec platform that skews toward humanities and social sciences, so topical fit with NLP/CS must be checked, and its records already carry English abstracts, which interacts with the index conditions in §6.4), and Vietnamese scholarly journals or repositories to be identified with the advisor. Each source is checked for terms of use, redistribution, text availability and metadata quality.

Each record keeps: stable ID and metadata; original-language abstract and author-written English abstract (if any); searchable text and language; index-inclusion condition; publication date (for leakage controls).

### 6.2 Three-tier label design (core novelty of the data)

| Tier | Label source | Role | Strength / limit |
|---|---|---|---|
| **T-natural** | Real cross-lingual same-contribution pairs: the same group publishing a Vietnamese/French paper and an English counterpart, or an extended version | Naturally occurring cross-lingual prior art; connects to self-plagiarism / redundant-publication detection | Highest validity; likely small, so size must be counted in Week 0–1 |
| **T-cite** | Acceptance sets from citations: papers cited as prior work in P's related-work section are accepted answers for an obfuscated statement of P's contribution | Real, multi-answer labels beyond a single planted source | Citation ≠ equivalence; needs judging and partial-grade labels |
| **T-syn** | LLM-generated obfuscations of P validated by a separate frozen judge | Scale and controlled difficulty | Circularity risk; never the only evidence |

Plus **T-human** (human-written obfuscations, frozen test set never used for tuning) and **T-hardneg** (topically close papers judged not to cover the stated contribution).

**Expectation management for T-natural.** Real same-contribution pairs published by *different* documents in different languages are likely scarce. CLIRudit does **not** supply them: it pairs one paper's English keywords with the same paper's French abstract (one document, two languages). Plan on T-syn + T-cite as the main evidence and treat T-natural as an opportunistic bonus whose size is measured in Week 0–1.

**Leakage controls for T-natural.** Bilingual versions by the same group share style, notation and argument structure, which a dense model can exploit instead of matching ideas. Remove author, affiliation and self-citation signals; normalize notation; report self-citation rates; and run a style-only control (a model given only stylistic features should not be able to solve the task).

**Citation function for T-cite.** A cited paper is not automatically a valid answer. Label the citation function (builds on / compared against / contrasted with / background) and accept only functions that count as prior art for the stated contribution; report annotator agreement.

### 6.3 Splits
Split by **source paper**, never by query variant. Hold out at least one transformation family (or a transformation × language combination) at test time. Add a **temporal split** (develop on older papers, test on newer ones) to reduce contamination and leakage between related papers.

### 6.4 Index conditions (separating coverage from semantics)
1. Original-language document indexed
2. Author-written English abstract indexed
3. Both indexed
4. Neither indexed (negative control: the source paper is absent from the index, used to measure false "match" claims and abstention behaviour)

Report each separately. If the cross-lingual gap disappears whenever an English abstract is indexed, state this explicitly and rest the claim on terminology/abstraction shift instead. CLIRudit reports that dense retrievers without translation perform close to MT-based systems on English-keyword → French-abstract retrieval, so a small cross-lingual gap for strong multilingual retrievers should be treated as a plausible outcome rather than an anomaly.

**Abstract-completeness hypothesis (to be measured, not assumed).** Author-written English abstracts may omit mechanism and setting details that appear in the full native-language text, so facet coverage can differ between indexed representations. If measured, this turns the English-abstract confound into a finding: failures come from facets missing in the indexed text, not from language alone (RQ6). Do not assume that regional journals or proceedings always supply English abstracts; measure the rate in the actual corpus.

### 6.5 Quality assurance
- Frozen validity judge whose prompt/model differs from the obfuscation generator
- Stratified human check across languages and transformation families; report inter-annotator agreement for full/partial/none
- Shortcut detection with BM25 and metadata-only controls; report rather than silently remove easy cases
- Closed-book and publication-date controls to estimate LLM memorization of well-known papers
- Strict isolation of test queries, source IDs and human-written examples from training and tuning

---

## 7. Obfuscation taxonomy (inherited from Disguise-and-Seek)

| ID | Transformation | Tests | Phase |
|---|---|---|---|
| D1 | **Terminology drift**: replace domain terms with descriptions or another community's vocabulary | Vocabulary robustness | MVP |
| D2 | **Facet subsetting**: keep only some contribution facets | Partial matching; basis of graded labels | MVP |
| D3 | **Cross-lingual round-trip**: translate through another language and back, with quality control | Cross-lingual semantic drift | MVP |
| D4 | **Abstraction shift**: generalize or specialize while preserving the relation to the source | Matching across abstraction levels | After pilot |
| D5 | **Anti-shortcut scrubbing**: remove unique names, acronyms, dataset identifiers, numbers | Prevents trivial identification | MVP |

---

## 8. Phase 1 method: facet-aware graded retrieval

Keep **one** methodological component so its effect can be isolated.

1. **Facet extraction** for the query *I* and for candidates: which facets are preserved, generalized, or absent, with uncertainty.
2. **Facet-aware retriever** trained with positive pairs, graded partial matches and hard topical negatives.
3. **Graded decision**: full / partial / none per candidate, with facet-level evidence.

Candidate training signals (introduce one at a time, after the pilot):
- contrastive loss over source and accepted alternatives
- facet alignment loss
- graded-match loss
- hard-negative loss

Priority order: (a) fixed retrieval with facet representation and graded matching; (b) compare against generic contrastive training; (c) only then Phase 2.

FaBle-style decompose-and-recompose augmentation is a candidate way to synthesize facet-level positive/negative pairs without hand-written facet labels; it must be tested against the frozen human-checked set because LLM-made pairs can be circular.

**Key failure mode to guard against:** broadening "Vietnamese word segmentation" to "language processing" must not make every language-processing paper look relevant.

---

## 9. Method contribution: facet-conditioned agentic search (Phase 2, gated by the pilot)

The agentic angle comes from the original Disguise-and-Seek design and the advisor's ResearchOS feedback. It is the **method contribution** of the paper, narrowed to what is defensible: merely calling search repeatedly, translating every query or stopping after a fixed number of turns does not count as agency and is not novel.

**Core algorithm (working name: FacetHunter).** The policy keeps an explicit *facet state* and chooses each action to resolve what is still missing.

```
state s_t = (idea I, facet status F_t over {problem, mechanism, setting, outcome}
             each in {resolved, partial, unresolved}, candidates C_t, spent budget b_t)

repeat:
    a_t = policy(s_t)         # search | rewrite | translate | inspect | compare | stop
    # actions target an UNRESOLVED facet: choose language/corpus/backend and query form for it
    execute a_t, update C_t and F_t from retrieved evidence
    stop if (saturation(C_t) AND no facet unresolved-with-untried-actions) OR budget exhausted
verdict per candidate via compare(candidate, facet schema)   # full / partial / none
return ranked list + graded verdicts + typed event-log certificate
```

**Claim to test (H-agent):** a policy with facet state beats (i) a strong fixed pipeline and (ii) an adaptive agent *without* facet state, at matched search budget, on held-out transformations and on T-human. If (ii) is not beaten, the facet-conditioning claim is dropped.

**Training:** zero-shot prompting first; then SFT on filtered successful, cost-efficient trajectories; RL only if SFT shows headroom, with the graded, facet-aware reward in §9.5.

### 9.1 Agent definition

**Observation:** current query, candidate list, facet coverage and uncertainty, past actions, remaining budget.
**Action set (bounded):**

```
search(query, language, backend, corpus)
rewrite(query, operator)
translate(query, target_language)
inspect(candidate, section)
compare(candidate, facet_schema)      # self-check before committing
note(search_state)
stop_and_report()
```

### 9.2 Research directions

| Direction | Idea | Testable hypothesis | Success criterion | Main risk |
|---|---|---|---|---|
| **A. Facet-conditioned action selection** | Choose the next action from the *unresolved facet* (e.g. search the native-language index when terminology mismatch is suspected, search for the missing facet instead of repeating the whole query) | Facet-conditioned policies beat generic rewrite/stop policies | Higher recall at equal cost than adaptive agent without facet state | Gains explained by generic agentic training |
| **B. Language/corpus routing** | Learn which language index or backend to query for a given shift type | Routing exploits shift-specific structure | Oracle-routing headroom first, then learned routing closes part of the gap | Fixed "search all languages" pipeline already near-optimal |
| **C. Saturation-aware stopping** | Stop when few new candidates appear across rounds *and* facet coverage is resolved, not just on a fixed budget (adopted from ResearchOS feedback) | Better recall–cost Pareto than fixed budget | Same recall at lower cost | Prior adaptive-depth work already covers generic stopping |
| **D. Self-verification via `compare`** | Candidate comparison against the facet schema before verdict | Fewer false "full" verdicts on hard negatives | Lower false-full rate at equal recall | Judge/verifier errors compound |
| **E. Verifiable search certificate** | Typed event log: `IdeaProposed`, `SearchIssued`, `CandidateFound`, `VerdictReached` (adopted from ResearchOS feedback), with costs, languages, stop reason | Auditable, replayable traces | Certificates replay deterministically from cached results; usable for error analysis and as training trajectories | Over-engineering without measurable value |
| **F. Optional co-evolving Proposer–Hunter** | Proposer generates valid but challenging obfuscations while the Hunter improves | Harder curriculum improves transfer to T-human | Gain on frozen T-human not seen during co-evolution | Collapse, invalid generations, high compute; defer |

### 9.3 Agency ladder (matched corpus access and budget at every level)
1. One-shot retriever
2. Fixed multi-stage pipeline (strong baseline)
3. Zero-shot agent with tools
4. Adaptive agent with facet state
5. Trained adaptive agent (SFT, then RL if warranted)
6. Optional co-evolving Proposer–Hunter

**Rule:** if levels 3–5 do not improve quality or cost over level 2, **drop the agentic claim** and report the useful remaining contribution.

### 9.4 Headroom analysis (before any training)
Run an **oracle routing** upper bound (diagnostic only, not deployable). If the oracle barely improves over the best fixed routing, adaptive routing cannot be the core contribution.

### 9.5 Provisional reward for later RL stage

```
R = α·R_rank + β·R_facet + γ·R_grade − λ·R_cost − μ·R_redundancy
```

- `R_rank`: placing a known source or accepted equivalent high
- `R_facet`: accurate facet-level matching
- `R_grade`: correct full/partial/none classification
- `R_cost`: search calls, tokens, latency, expensive inspections
- `R_redundancy`: repeated or low-information actions

This is a design sketch. The source-ID reward must not dominate, otherwise the policy may exploit source-specific artifacts; held-out transformations and T-human evaluation are the safeguards. Use cached search results and judgments to control compute and annotation cost.

### 9.6 Training plan if gates are passed
- **Stage A:** train the facet-aware retriever; validate on held-out papers and transformations.
- **Stage B:** SFT on successful, cost-efficient trajectories, then RL with the verifiable reward; compare generic source-ID reward vs facet-aware reward.
- **Stage C (optional):** self-play with a fixed proposer/data mixture, evaluated on a frozen T-human set.
- Prefer parameter-efficient fine-tuning and short pilot runs before scaling rollouts.

---

## 10. Baselines

### 10.1 Retrieval baselines
1. BM25
2. Strong multilingual dense retriever (e.g. BGE-M3 or a stronger model after a current benchmark check)
3. Hybrid sparse + dense
4. Hybrid + cross-encoder reranking
5. Translate-then-search and multilingual query expansion
6. Facet extraction followed by fixed retrieval
7. Fixed multilingual pipeline searching all configured languages in a predefined order
8. A CLPD-style baseline (cross-lingual similarity/alignment method) as a direct line to prior work
9. CLIRudit-style protocols: dense retrieval without translation vs. query translation vs. document translation
10. UniFAR as the strongest facet-aware baseline (if code or weights are available; otherwise a reimplemented facet-anchor baseline, clearly marked)
11. Agentic-R as the retriever inside the agent loop, if released and usable here (it was trained for QA-style agentic search, so report it as an adapted baseline)
12. Translate-then-search with a commercial or strong open MT system plus the best English retriever, as the practical ceiling that multilingual methods must beat
13. Chain of Retrieval (aspect decomposition + iterative expansion) as the closest iterative multi-aspect method, if code is available (reference implementation linked from its paper)

### 10.2 Agent baselines
Zero-shot ReAct-style agent; fixed-policy multi-step agent; adaptive agent without facet state; adaptive agent with facet state, no RL; SFT-only; SFT + RL with generic source reward; full facet-aware agent.

### 10.3 Comparisons to closest agentic work
Adapt, where feasible, the stopping principle of adaptive-depth methods and the agent–retriever training recipe of Agentic-R as baselines, and clearly mark any adaptation. Do not claim a fair reproduction if data, access or budget differ.

---

## 11. Evaluation

**Primary metrics:** Recall@1/5/10; **false-gap rate** (share of queries whose valid prior art is in the index but the system reports no sufficient match, using a decision threshold fixed on development data; this is the failure that matters for ResearchOS stage 13); MRR and median rank of the first accepted candidate; macro-F1 and confusion matrix for full/partial/none; facet-level precision/recall/F1 if labels are reliable; search calls, tokens, latency and cost; recall–cost Pareto curves at small, medium and large budgets. Report wall-clock latency and token cost per query as measured, never estimated. A few points of recall at an order-of-magnitude higher cost is a weak result, so the agent's case must also rest on the value of auditable certificates (which facets are covered, by which evidence), which has to be demonstrated, for example with a small human study of whether certificates help verify matches.

**Robustness analysis:** by transformation family and severity; by query language and source language; held-out transformations; T-syn → T-human gap; the four index conditions; performance after scrubbing identifiers; sensitivity to topical hard negatives.

**Statistics:** bootstrap confidence intervals by **source paper**; paired tests on matched queries and budgets; at least three seeds for final RL configurations; absolute scores and effect sizes; qualitative error taxonomy.

---

## 12. Ablations that matter

| Ablation | Question |
|---|---|
| No facet representation | Does structure help beyond text embeddings? |
| No partial-match labels | Does graded matching reduce false "full" claims? |
| No hard negatives | Does the model confuse topical similarity with overlap? |
| Fixed vs adaptive policy | Is agentic control necessary? |
| No multilingual actions | Value of language-aware search |
| No cost penalty | Does the agent over-search? |
| T-syn only vs T-syn + T-human | Does training generalize beyond synthetic style? |
| Abstract-only vs full-text inspection | Is deeper reading needed? |
| Generic reward vs facet-aware reward | Is the signal distinct from generic verifiable search reward? |

Prioritize ablations that isolate the claimed novelty over many small ones.

---

## 13. Pilot plan, timeline and go/no-go gates

| Week | Work |
|---|---|
| **0–1** | Read PAN source retrieval, cross-language plagiarism detection literature, CSFCube, one patent prior-art paper, and the "same paper from different aspects" benchmark (§4.1); verify every citation in this document; **count** natural bilingual pairs and English-abstract availability; check licensing; write the 3-sentence delta versus CLPD/PAN; finalize the facet guide and test agreement |
| **1–2** | Build the corpus and source-level splits; create a pilot set (≈200–500 source papers, ≈500–1,500 valid obfuscations as a planning target); D1–D3 and D5 first; human-check a stratified sample |
| **2–3** | Run fixed baselines (including CLPD-style); record metrics per transformation |
| **3–4** | Test facet-aware retrieval; run the agency test with oracle routing at matched budget |
| **4–8 (only if gates pass)** | Build FacetHunter: zero-shot version, then SFT on filtered trajectories; ablate against adaptive-without-facet-state and the fixed pipeline; RL only if SFT shows headroom |

### Kill / redirect criteria (initial heuristics, not universal thresholds)

| Condition | Action |
|---|---|
| Delta versus CLPD/PAN cannot be stated in 3 sentences | Redirect or stop |
| Corpus too small, poorly licensed, or almost all papers have English abstracts | Narrow the cross-lingual claim; focus on terminology/abstraction shift |
| Strongest fixed pipeline near-saturates Recall@10 on moderate/hard cases | Move to natural labels (T-natural, T-cite) and held-out transformations, or stop |
| Facet-aware retrieval does not beat strong dense/hybrid | Do not start RL; revise representation or reposition as analysis paper |
| Oracle routing barely beats fixed routing | Drop adaptive routing as the core contribution |
| T-human collapses relative to T-syn | Improve task design or narrow claims |

Large-scale SFT/RL proceeds **only** if the pilot shows a reproducible gap and a plausible mechanism.

**Initial numeric thresholds.** These are starting heuristics: fix them before looking at results and revise them with variance estimates from the pilot.

| Gate | Check | Starting threshold for "fail" | If failed |
|---|---|---|---|
| G1 data audit (Wk 0–1) | Natural pairs; share of papers with usable English abstracts | < ~100 natural pairs, or > ~90% with usable English abstracts | T-natural becomes a small supplementary set; narrow the cross-lingual claim |
| G2 oracle routing (Wk 2–3) | Oracle routing vs. best fixed pipeline | gap < ~3 MRR points or ~5 Recall@10 points, and inside the bootstrap CI | Drop the adaptive-routing claim; benchmark + analysis paper |
| G3 facet retrieval (Wk 3–4) | Facet-aware vs. best hybrid + rerank | no significant gain | No RL; revise facets or move to error analysis |
| G4 Pareto (Wk 4–8) | FacetHunter vs. fixed pipeline and no-facet-state agent at equal budget | not on the Pareto frontier | Report as negative result; target Findings/workshop |
| G5 transfer (Wk 6–8) | T-syn → T-human | relative drop > ~30% | Audit generation rules for artifacts; narrow claims |

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Synthetic task is only paraphrase retrieval | Three-tier labels; held-out transformations; T-human; T-natural |
| Cross-lingual gap vanishes with English abstracts | Index-condition reporting; shift the claim if needed |
| Small Vietnamese corpus makes retrieval trivially easy | Add distractor papers at scale; report corpus statistics |
| LLM memorization of known papers | Closed-book and date controls; prefer less-famous venues |
| Facet labels subjective or noisy | Guidelines, agreement reporting, adjudication; limit claims to reliable facets |
| Circularity (LLM generates query, answer and judge) | Separate frozen judge; human validation; natural labels |
| Agent shows no gain | Pre-registered agency ladder; drop the claim and report it |
| Scope too large for one student | Phase 1 only until gates pass; defer RL and self-play |
| Citation errors from AI-generated drafts | Verify every reference against the full paper before use |

---

## 15. Expected contributions and venue strategy

**Contributions if validated:**
1. **Task and benchmark:** idea-level prior-art recovery with three-tier labels and per-shift evaluation.
2. **Failure analysis:** which shifts and language pairs break strong retrievers, and whether the effect survives English-abstract indexing.
3. **Method result:** whether facet-aware graded retrieval helps and whether it transfers to human-written and natural pairs.
4. **Method:** a facet-conditioned agentic search policy (FacetHunter) with matched-budget evidence against a strong fixed pipeline and an adaptive agent without facet state, plus transfer to T-human. If it does not hold, the paper reports that honestly and stands on items 1–3.

**Venue decision follows results, not prestige:**

| Outcome | Venue direction |
|---|---|
| Benchmark **and** FacetHunter beats strong fixed pipeline and no-facet-state agent at matched cost, with transfer to T-human | ACL / EMNLP Main candidate |
| Solid benchmark and analysis, moderate method gain | Findings / EACL / LREC / EMNLP |
| Meaningful failure phenomenon, no reliable agent benefit | Analysis/resource paper or workshop |
| Fixed pipelines saturate or labels unreliable | Stop or redirect |

**Why the combination, and what it costs.** A benchmark plus a method that is *motivated by the benchmark's failure analysis* has a higher ceiling than either alone. But each part must be able to stand by itself: the benchmark paper is the fallback if the agent does not beat the fixed pipeline, and the scope (three label tiers, four index conditions, agent training) is the main execution risk, so Phase 2 stays behind the §13 gates.

---

## 15.1 Anticipated reviewer questions

1. **"How is this different from cross-lingual plagiarism detection?"** CLPD detects reuse of text (translated or paraphrased). Here the query is an idea-level description that has been deliberately obfuscated, the answers are graded (full/partial/none) at the contribution level, and evaluation uses held-out transformation families, human-written queries and citation-derived labels. If the experiments show the task collapses to paraphrase retrieval, the project says so (G3/G5).
2. **"Aren't the labels circular if LLMs generate, validate and judge?"** Hence the frozen T-human set, a judge distinct from the generator, human validation, citation-derived and natural labels, and the T-syn → T-human gap as a reported number.
3. **"Why accept the cost and latency of an agent?"** Only if it is on the recall–cost Pareto frontier against a strong fixed pipeline and an agent without facet state (G4). Otherwise it is reported as a negative result.
4. **"Is the language barrier real if abstracts are in English?"** Measured, not assumed: four index conditions, the abstract-completeness analysis (RQ6), and a narrowed claim if the gap vanishes.
5. **"Why not build an automatic survey writer or a literature-review workflow tool?"** See §2.1: automatic survey writing is a crowded, well-resourced space, and ResearchOS's own success criteria explicitly reject automatic writing as the goal. The ResearchOS design principles that matter — reproducibility, human gates, saturation-based stopping, a typed audit trail — are kept and placed inside a falsifiable research question instead.

*Not adopted:* redefining the task as cross-domain analogy (same mechanism applied to different problems). It would be a different and harder task with no available labels, and would blur the contribution-level prior-art framing.

---

## 16. Open questions for the advisor
1. **Language scope:** keep Vietnamese/French, or broaden?
2. **Vietnamese sources:** which journals/repositories are suitable and permitted?
3. **Natural pairs:** are there bilingual publications by the group that could seed T-natural?
4. **Compute:** what is available for a possible SFT/RL stage after the pilot?
5. **Annotation:** can human-written obfuscations and full/partial judgments be annotated by lab members?
6. **Direction change:** does §2.1's reasoning for moving away from automatic survey writing and from the ResearchOS workflow tool match the advisor's own thinking, or is there a reason to keep either of those as the main deliverable instead?

---

## References (first-pass; verify all details before citing)

1. Liu et al. (2026). *Agentic-R: Learning to Retrieve for Agentic Search.* Findings of ACL 2026. https://aclanthology.org/2026.findings-acl.785/ (arXiv 2601.11888). *Checked.*
2. Potthast, M., et al. Cross-language plagiarism detection (Language Resources and Evaluation, 2011) and the PAN plagiarism detection shared tasks (source retrieval, text alignment, artificial obfuscation corpora). *Not re-checked.*
3. Barrón-Cedeño, A., et al. Cross-language plagiarism detection (CL-ASA and follow-up work); Franco-Salvador, M., et al. (CL-CTS and knowledge-graph methods). *Not re-checked.*
4. Mysore, S., et al. (2021). *CSFCube: A test collection of computer science research articles for faceted query by example.* *Not re-checked.*
5. Ostendorff, M., et al. (2020). *Aspect-based document similarity for research papers.* COLING 2020. *Not re-checked.*
6. CLEF-IP and cross-lingual patent prior-art retrieval literature. *To be identified.*
7. Chen, J., et al. (2024). *BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings.* *Not re-checked.*
8. Valentini, F., Kozlowski, D., Larivière, V. (2025). *CLIRudit: Cross-Lingual Information Retrieval of Scientific Documents.* MRL 2025, ACL Anthology 2025.mrl-main.16; arXiv 2504.16264. *Abstract checked.*
9. Dou, Z., et al. (2026). *UniFAR: A Unified Facet-Aware Retrieval Framework for Scientific Documents.* arXiv 2602.23766. *Abstract checked; benchmarks and language coverage to be confirmed from the full paper.*
10. Wang, J., Wang, K., Wang, X., Naidu, P., Bergen, L., Paturi, R. (2023). *DORIS-MAE: Scientific Document Retrieval using Multi-level Aspect-based Queries.* NeurIPS 2023; arXiv 2310.04678. *Checked.*
11. Do, H., Ryu, S., Kim, J., Lee, G. G. (2025). *Multi-Facet Blending for Faceted Query-by-Example Retrieval* (FaBle). ACL 2025 (long paper), aclanthology.org/2025.acl-long.1388; originally arXiv 2412.01443 (Dec 2024). *Checked.*
12. Wei, Y. (2026). *Can Retrievers Find the Same Paper from Different Aspects? A Multi-Aspect Full-Paper Scientific Retrieval Benchmark* (MAPLE). arXiv 2608.15624, 16 Aug 2026. *Abstract checked; read in full in Week 0–1.*
12b. Park, S., Baek, J., Jeong, S., Hwang, S. J. *Chain of Retrieval: Multi-Aspect Iterative Search Expansion and Post-Order Search Aggregation for Full Paper Retrieval* (SciFullBench; v1 titled PRISM). arXiv 2507.10057. *Abstract checked.*
13. Earlier-draft items **not independently verified:** PaperSearchQA (EACL 2026), AutoSearch (Findings of ACL 2026), mRewriter-R1 (arXiv, Oct 2026; could not be located), Claim2Source / CheckThat! 2026 Task 1, XBCP, Search Self-Play (ICLR 2026), and the slide-deck items LcRL / CroSearch-R1 and CROWN-QA.
14. ResearchOS (*Research Survey OS — Single-Path MVP Proposal*), advisor document, undated. Cited in §2.1 for its own §24 (Non-Goals) and §25 (Success Criteria). Not a peer-reviewed publication; cited as a primary source for what it itself recommends, not as external validation.
15. Nguyen, M.-A., et al. (2025). *SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation.* arXiv 2510.07733. *Abstract checked.* AutoSurvey, SurveyForge, SurveyX, LLM×MapReduce-V2, Deep Academic Survey (DAS) and SurveyBench are cited in §2.1 as evidence that automatic survey writing is a populated space; *not re-checked in this pass, full bibliographic details to be restored before submission.*
16. Kallmes, K. M., et al. (2025). *Human-in-the-Loop Artificial Intelligence System for Systematic Literature Review: Methods and Validations for the AutoLit Review Software.* Cochrane Evidence Synthesis and Methods, DOI 10.1002/cesm.70059; retracted June 2026 for a conflict-of-interest policy breach (Retraction Watch, 13 July 2026). *Checked.* ASReview is cited from ResearchOS §8; LLAssist and the multi-agent clinical-trial SLR system were *not re-checked*.

**Citation caveat:** These references establish neighbouring research directions only. They do not establish that the proposed gap is novel. A submission-ready related-work section requires reading the full papers and following their citations, especially for cross-language plagiarism detection, PAN source retrieval, faceted scientific retrieval, cross-lingual and patent prior-art retrieval, and agentic search policies. In references 15 and 16, only SurveyG and the AutoLit retraction were re-checked in the latest pass; the remaining survey-writing and systematic-review tools should be re-confirmed with full citation details before the proposal is finalized.
