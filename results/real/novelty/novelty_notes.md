# Novelty Notes: Related Work & Positioning Analysis

> [!IMPORTANT]
> **EXPLORATORY - notes, verification pending**

This document provides a systematic review of existing benchmarks and literature relevant to our study on **idea-level prior-art recovery under controlled terminology obfuscation**, analyzing where existing works leave open the questions we address.

---

## 1. Primary Benchmarks & Retrieval Frameworks

### 1.1 LitSearch (Ajith et al., EMNLP 2024 / arXiv:2407.18940)
- **Full Title**: *LitSearch: A Benchmark for Scientific Literature Search*
- **Authors**: Anirudh Ajith, Mengzhou Xia, Alexis Chevalier, Tanya Goyal, Danqi Chen, Tianyu Gao (Princeton University).
- **Task**: Scientific literature search addressing complex, realistic research information needs in machine learning and natural language processing.
- **How Queries are Built**: 597 queries created via GPT-4 from paragraphs containing inline citations, combined with human-written questions from paper authors, followed by expert review.
- **Corpus & Domain**: Recent ML and NLP papers from arXiv and conference venues (full text and abstracts).
- **Languages**: English only (monolingual).
- **Terminology Shift / Keyword Removal Controlled**: **Partly/No**. Queries exhibit natural syntactic and lexical variation, but lexical overlap with targets is not systematically masked, ablated, or controlled.
- **Label Source**: Real citation linkages and expert-verified multi-paper relevance judgments.
- **Metrics**: Recall@k (k=5, 10, 20), nDCG@10, Mean Reciprocal Rank (MRR).
- **Methods**: Sparse retrieval (BM25), dense retrievers (Contriever, BGE, SPECTER), LLM re-ranking pipelines, and commercial search engines.
- **Headline Results**: Dense retrievers significantly outperform sparse baselines: "dense models outperforming BM25 by a large margin (e.g., a 24.8% absolute difference in Recall@5)" (Ajith et al., p. 2). Commercial search engines lag behind top dense models by up to 32 recall points.
- **Code / Data Availability**: Publicly available on GitHub (`princeton-nlp/LitSearch`).
- **Overlap with Our Study**: Both investigate retrieving relevant scientific papers from natural-language queries rather than simple keyword lookups.
- **Difference**: LitSearch is monolingual English and evaluates queries against drafted papers with naturally co-occurring vocabulary; it does not test cross-lingual retrieval, early-stage idea obfuscation, or controlled facet removal.

---

### 1.2 BRIGHT (Su et al., ICLR 2025 / arXiv:2407.12883)
- **Full Title**: *BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval*
- **Authors**: Hongjin Su, Howard Yen, Mengzhou Xia, Danqi Chen, Tao Yu, et al.
- **Task**: Reasoning-intensive retrieval where queries require multi-step inference, problem-solving, or domain understanding to identify target documents.
- **How Queries are Built**: 1,398 real-world challenging queries harvested from diverse problem-solving communities (e.g., StackExchange, LeetCode, math competitions).
- **Corpus & Domain**: Multi-domain technical corpora covering biology, earth science, economics, psychology, robotics, software development, and mathematics.
- **Languages**: English only (monolingual).
- **Terminology Shift / Keyword Removal Controlled**: **Partly**. The benchmark intrinsically selects queries with low lexical overlap requiring reasoning, but keyword retention or facet masking is not experimentally controlled.
- **Label Source**: Real human accepted solutions, forum answers, and community ground truth. Multi-answer in most domains.
- **Metrics**: nDCG@10, Recall@10, Recall@100.
- **Methods**: Standard dense retrievers, BM25, instruction-tuned retrievers, ReasonIR, and Chain-of-Thought (CoT) query expansion re-rankers.
- **Headline Results**: State-of-the-art retrievers experience severe degradation: "top-performing models on standard leaderboards experience drastic drops in nDCG@10 scores" (Su et al., p. 1).
- **Code / Data Availability**: Publicly released at `https://brightbenchmark.github.io/`.
- **Overlap with Our Study**: Examines retrieval failure when surface lexical matching is insufficient and conceptual connections are required.
- **Difference**: BRIGHT focuses on general reasoning/problem-solving tasks across broad domains rather than scientific prior-art recovery; it does not feature cross-lingual search or controlled ablation of problem/mechanism facets.

---

### 1.3 DORIS-MAE (Wang et al., NeurIPS 2023 / arXiv:2310.04678)
- **Full Title**: *DORIS-MAE: Scientific Document Retrieval using Multi-level Aspect-based Queries*
- **Authors**: Jianyou Wang, Kaicheng Wang, Xiaoyue Wang, Prudhviraj Naidu, Leon Bergen, Ramamohan Paturi (UC San Diego).
- **Task**: Scientific document retrieval using complex, multi-faceted, hierarchical research queries.
- **How Queries are Built**: Human-authored complex primary queries with hierarchical aspect structure, decomposed into sub-queries; validated via an LLM annotation framework (Anno-GPT).
- **Corpus & Domain**: Computer science academic publications.
- **Languages**: English only (monolingual).
- **Terminology Shift / Keyword Removal Controlled**: **Partly**. Queries decompose into explicit rhetorical aspects, but lexical leak elimination and vocabulary obfuscation are not systematically controlled.
- **Label Source**: Graded multi-aspect relevance judgments annotated by human experts and LLMs. Multi-answer.
- **Metrics**: nDCG@k, MAP, Precision@k.
- **Methods**: BM25, SciBERT, SPECTER, Contriever, cross-encoder re-rankers.
- **Headline Results**: Current retrievers suffer substantial performance drops on multifaceted scientific queries compared to standard single-intent benchmarks.
- **Code / Data Availability**: Publicly released on GitHub (`Jack-J-Wang/DORIS-MAE`).
- **Overlap with Our Study**: Decomposes scientific search queries into explicit functional facets.
- **Difference**: Monolingual English; queries are long, multi-paragraph user specifications; does not test idea-stage retrieval before drafting, cross-lingual transfer, or controlled disguise strength.

---

### 1.4 MAPLE (Wei et al., arXiv:2608.15624)
- **Full Title**: *Can Retrievers Find the Same Paper from Different Aspects? A Multi-Aspect Full-Paper Scientific Retrieval Benchmark*
- **Authors**: Yiyang Wei, Fang Guo, Qiji Zhou, Zhizhang Fu, Mengru Ding, Kai Yang, Yue Zhang.
- **Task**: Multi-aspect full-paper scientific retrieval: evaluates whether search systems can "consistently recover the same paper from queries targeting its motivation, method, and experimental findings" (Abstract, p. 1).
- **How Queries are Built**: 2,095 queries synthesized via MAPLE-Synth ("retrieval-based in-context learning pipeline that leverages OpenReview discussions and human-written query exemplars", Abstract, p. 1).
- **Corpus & Domain**: Recent Machine Learning (ML) and Natural Language Processing (NLP) papers from OpenReview (covering text and tables).
- **Languages**: English only (monolingual).
- **Terminology Shift / Keyword Removal Controlled**: **No**. Queries target specific paper facets without deliberate vocabulary obfuscation or lexical leak stripping.
- **Label Source**: Expert-validated facet relevance mappings linking queries to target full papers.
- **Metrics**: `AnyAspect@k`, `AllAspect@k` (e.g., AnyAspect@20, AllAspect@20), Recall@k, nDCG@k.
- **Methods**: Sparse retrieval (BM25), dense retrievers, multimodal models, and multi-chunk score aggregation.
- **Headline Results**: Significant gap across aspects: "the strongest model achieves 98.1% AnyAspect@20 but only 15.7% AllAspect@20" (Abstract, p. 1).
- **Code / Data Availability**: Released by authors (exact licensing terms: NOT VERIFIED).
- **Overlap with Our Study**: Decomposes scientific retrieval by conceptual facets (motivation/problem, method/mechanism, findings/outcome).
- **Difference**: Monolingual English full-paper retrieval; does not investigate cross-lingual retrieval (English idea searching French literature) or controlled idea-stage terminology disguise (V0–V3).

---

### 1.5 Chain of Retrieval (CoR) / PRISM with SciFullBench (Park et al., arXiv:2507.10057)
- **Full Title**: *Chain of Retrieval: Multi-Aspect Iterative Search Expansion and Post-Order Search Aggregation for Full Paper Retrieval*
- **Authors**: Sangwoo Park, Jinheon Baek, Soyeong Jeong, Sung Ju Hwang (KAIST).
- **Task**: Full scientific paper-to-paper retrieval, identifying relevant prior literature given an entire draft manuscript.
- **How Queries are Built**: The framework "decomposes each query paper into multiple aspect-specific views" (Abstract, p. 1) from draft paper sections.
- **Corpus & Domain**: SCIFULLBENCH — complete and section-segmented contexts of full AI/CS research papers.
- **Languages**: English only (monolingual).
- **Terminology Shift / Keyword Removal Controlled**: **No**. The query is an existing, fully drafted manuscript with standard technical terminology.
- **Label Source**: Citation networks and document relevance linkages. Multi-answer.
- **Metrics**: Recall@k, nDCG@k across full candidate corpus.
- **Methods**: Iterative tree-structured search expansion ("iteratively expands the search by promoting top-ranked results as new queries", Abstract, p. 1) aggregated via post-order traversal.
- **Headline Results**: CoR significantly outperforms flat single-step abstract-based and full-text retrieval baselines across full-paper corpora.
- **Code / Data Availability**: Available on GitHub (`psw0021/Chain-of-Retrieval-Official`).
- **Overlap with Our Study**: Utilizes multi-aspect query decomposition and iterative / multi-step retrieval loops to recover related work.
- **Difference**: CoR requires a complete 10k–50k token manuscript as input within English citation graphs; our work investigates short, cross-lingual idea statements (1–2 sentences) before drafting begins.

---

### 1.6 UniFAR (Dou et al., arXiv:2602.23766)
- **Full Title**: *UniFAR: A Unified Facet-Aware Retrieval Framework for Scientific Documents*
- **Authors**: Zheng Dou, Zhao Zhang, Deqing Wang, Fuzhen Zhuang, Yikun Ban.
- **Task**: Joint document-to-document (doc-doc) and question-to-document (q-doc) scientific document retrieval within a shared semantic space.
- **How Queries are Built**: Standard benchmark questions and seed documents from established SDR benchmarks.
- **Corpus & Domain**: Standard scientific document retrieval benchmarks across computer science and biomedicine.
- **Languages**: English only (monolingual).
- **Terminology Shift / Keyword Removal Controlled**: **No**. Uses standard scientific queries and papers without intentional vocabulary masking.
- **Label Source**: Standard benchmark labels (citations for doc-doc; relevance judgments for q-doc).
- **Metrics**: nDCG@k, Recall@k, MAP.
- **Methods**: Multi-granularity representation, learnable facet anchors, and facet-aware joint contrastive training.
- **Headline Results**: UniFAR consistently outperforms strong baseline methods across both doc-doc and q-doc benchmarks.
- **Code / Data Availability**: Code availability: NOT VERIFIED.
- **Overlap with Our Study**: Employs facet-level semantic modeling of scientific discourse structure.
- **Difference**: Focuses on representation learning architecture on existing English benchmarks; does not diagnose retrieval failure modes or examine cross-lingual vocabulary gaps.

---

### 1.7 CLIRudit (Valentini et al., arXiv:2504.16264)
- **Full Title**: *CLIRudit: Cross-Lingual Information Retrieval of Scientific Documents*
- **Authors**: Francisco Valentini, Diego Kozlowski, Vincent Larivière.
- **Task**: Cross-lingual scientific information retrieval (English queries searching French scientific literature).
- **How Queries are Built**: English author-written keywords from bilingual metadata of academic articles on the Érudit platform.
- **Corpus & Domain**: French academic articles on Érudit (humanities, social sciences, and multidisciplinary journals).
- **Languages**: Cross-lingual (English queries retrieving French document abstracts).
- **Terminology Shift / Keyword Removal Controlled**: **No**. Queries consist of explicit, literal author keywords without vocabulary masking or facet ablation.
- **Label Source**: Real document-level ground truth (article's own keywords matching its abstract). Single-target.
- **Metrics**: nDCG@10, Recall@10, MRR.
- **Methods**: Zero-shot sparse (BM25, SPLADE) and dense multilingual retrievers, comparing query translation against document translation.
- **Headline Results**: "translating documents is generally more effective than translating queries" (Valentini et al., p. 1); large multilingual dense embeddings without translation perform competitively.
- **Code / Data Availability**: Hugging Face dataset and GitHub code available.
- **Overlap with Our Study**: Directly shares the cross-lingual language pair (English queries retrieving French scientific papers) and evaluates dense vs. doc-translate BM25.
- **Difference**: CLIRudit uses explicit, verbatim author keywords (e.g., standard topical keywords like "sociologie urbaine"). It does not evaluate idea-stage search, does not ablate rhetorical facets, and does not test controlled lexical leak removal or agentic iterative recovery.

---

## 2. Robustness to Query Variation & Vocabulary Mismatch

### 2.1 Query Variation Generators (Penha et al., ECIR 2022 / arXiv:2111.13057)
- **Full Title**: *Evaluating the Robustness of Retrieval Pipelines with Query Variation Generators*
- **Authors**: Gustavo Penha, Arthur Câmara, Claudia Hauff.
- **Task**: Evaluating the robustness of neural retrieval pipelines to semantic-preserving query variations.
- **How Queries are Built**: Built a taxonomy of query transformations from user query variations (UQV100) and generated synthetic variants via automated rule-based and neural methods.
- **Corpus & Domain**: General web and passage retrieval (MS MARCO, TREC).
- **Languages**: English.
- **Terminology Shift / Keyword Removal Controlled**: **Yes** (controlled syntax, paraphrasing, and term substitution).
- **Label Source**: Standard benchmark relevance judgments. Multi-answer.
- **Metrics**: Robustness metrics, nDCG@10, MRR.
- **Methods**: BM25, dense retrievers, neural BERT re-rankers.
- **Headline Results**: Neural retrieval pipelines are brittle to query variations, experiencing an average effectiveness drop of ~20%.
- **Difference from Our Study**: Tests general search queries without scientific discourse structure; does not address cross-lingual literature retrieval or facet ablation.

---

### 2.2 LLM Dense Retriever Stability (Li et al., arXiv:2604.16576, 2026)
- **Full Title**: *On the Robustness of LLM-Based Dense Retrievers: A Systematic Analysis of Generalizability and Stability*
- **Authors**: Yongkang Li, Panagiotis Eustratiadis, Yixing Fan, Evangelos Kanoulas.
- **Task**: Systematic analysis of generalizability and stability of LLM-based dense retrievers under query perturbations.
- **How Queries are Built**: Systematic perturbations including typos, paraphrasing, synonym substitution, and corpus poisoning.
- **Corpus & Domain**: 30 retrieval datasets across 4 standard IR benchmarks.
- **Languages**: English.
- **Terminology Shift / Keyword Removal Controlled**: **Yes** (categorized perturbation types).
- **Label Source**: Standard IR ground truth.
- **Headline Results**: LLM retrievers resist surface typos well but remain sensitive to semantic perturbations like synonym substitution.
- **Difference from Our Study**: Evaluates passage retrieval robustness to noise/perturbations; does not address cross-lingual idea recovery or rhetorical facet ablation.

---

### 2.3 SPAR (Chen et al., EMNLP Findings 2022 / arXiv:2110.06918)
- **Full Title**: *Salient Phrase Aware Dense Retrieval: Can a Dense Retriever Imitate a Sparse One?*
- **Authors**: Xilun Chen, Kushal Lakhotia, Barlas Oğuz, Patrick Lewis, Wen-tau Yih, et al.
- **Task**: Enhancing dense retrieval with explicit lexical and salient phrase awareness.
- **How Queries are Built**: Standard open-domain QA and passage queries (MS MARCO, BEIR, EntityQuestions).
- **Terminology Shift / Keyword Removal Controlled**: **Partly** (studies performance drops when exact entities/keywords are absent vs. present).
- **Headline Results**: Dense retrievers underperform sparse models on rare entities and exact keyword constraints; distilling sparse matching into dense models (SPAR) recovers lexical robustness.
- **Difference from Our Study**: An architectural solution for monolingual QA/passage search, not a diagnostic study of idea-stage prior-art recovery under cross-lingual terminology disguise.

---

## 3. Scientific Search with Low Lexical Overlap & Conceptual Analogy

### 3.1 CSFCube (Mysore et al., NeurIPS 2021 / arXiv:2103.12906)
- **Full Title**: *CSFCube: A Test Collection of Computer Science Research Articles for Faceted Query by Example*
- **Authors**: Sarthak Mysore, Arman Cohan, Edward Loper, Eric Wong, Matthew E. Peters, Iz Beltagy.
- **Task**: Faceted Query-by-Example (QBE) scientific document retrieval, finding papers similar along specific functional facets (Background, Method, Result).
- **How Queries are Built**: 50 query documents paired with expert-annotated rhetorical structure facets.
- **Corpus & Domain**: Computer Science research papers from the ACL Anthology and ML conferences.
- **Languages**: English only (monolingual).
- **Low Lexical Overlap**: Papers frequently share high-level problems or methods despite using distinct technical vocabularies across subfields.
- **Label Source**: Graded relevance judgments via TREC depth-k pooling with high inter-annotator agreement. Multi-answer.
- **Metrics**: nDCG@k, MAP.
- **Headline Results**: State-of-the-art document embedding models struggle to isolate fine-grained functional aspects compared to whole-document similarity.
- **Difference from Our Study**: Query is an entire published document (Query-by-Example), monolingual English; does not address short idea queries before drafting or cross-lingual literature recovery.

---

### 3.2 Analogy Search Engine (ASE) (Zhou et al., arXiv:1812.06974 / 2212.00766)
- **Full Title**: *Analogy Search Engine: Finding Analogies in Cross-Domain Research Papers*
- **Authors**: Jieli Zhou, Yuntao Xu, Yi Zhou, et al. (Carnegie Mellon University).
- **Task**: Cross-domain scientific analogy search, identifying papers in distant domains that share structural problem/solution mechanisms despite near-zero lexical overlap.
- **How Queries are Built**: User queries formulated around purpose and mechanism descriptions across AI subfields (ML, Robotics, HCI, Computational Biology).
- **Corpus & Domain**: Cross-disciplinary AI research papers.
- **Languages**: English only (monolingual).
- **Low Lexical Overlap**: Inherently near-zero lexical overlap due to cross-disciplinary domain jargon divergence.
- **Label Source**: Expert-annotated structural analogy matches.
- **Headline Results**: Analogy-driven semantic search surfaces creative, cross-domain prior art that standard lexical search engines (e.g., Elasticsearch) completely miss.
- **Difference from Our Study**: Focuses on cross-disciplinary creative analogy mining within English papers; does not systematically investigate cross-lingual idea recovery, diagnostic ranking headroom, or controlled disguise levels (V0–V3).

---

## 4. Systematic Comparison Table

| Work | Controls keyword retention? | Graded or multi-answer labels? | Real labels? | Iterative / agentic method? | What remains open for us |
|---|:---:|:---:|:---:|:---:|---|
| **LitSearch** (EMNLP 2024) | No | Multi-answer | Yes (Citations + Human) | No (Single-pass + Re-ranker) | Cross-lingual retrieval; controlled disguise/leak removal; idea-stage formulation. |
| **BRIGHT** (ICLR 2025) | Partly | Multi-answer | Yes (Real solutions) | Partly (CoT expansion) | Scientific prior-art domain; cross-lingual recovery; rhetorical facet ablation. |
| **DORIS-MAE** (NeurIPS 2023) | Partly | Graded & Multi-answer | Yes (Expert + LLM) | No (Single-pass) | Cross-lingual gap; short idea queries; systematic disguise strength calibration. |
| **MAPLE** (arXiv:2608.15624) | No | Multi-aspect (Single-target per aspect) | Yes (Expert-validated) | No (Chunk aggregation) | Cross-lingual barrier (EN $\rightarrow$ FR); intentional lexical leak removal; agentic recovery. |
| **Chain of Retrieval** (arXiv:2507.10057) | No | Multi-answer | Yes (Citations) | Yes (Tree expansion) | Pre-draft ideation queries (1–2 sentences); cross-lingual terminology mismatch. |
| **UniFAR** (arXiv:2602.23766) | No | Multi-answer | Yes (Citations + Q&A) | No (Representation learning) | Diagnosing ranking failure vs. under-specification; cross-lingual vocabulary shifts. |
| **CLIRudit** (arXiv:2504.16264) | No | Single-target | Yes (Author metadata) | No (First-stage CLIR) | Idea-level queries; facet ablation; controlled terminology disguise (V0–V3). |
| **Penha et al.** (ECIR 2022) | Yes | Multi-answer | Yes (TREC/MS MARCO) | No (Evaluation suite) | Scientific literature domain; cross-lingual retrieval; rhetorical facet structure. |
| **CSFCube** (NeurIPS 2021) | Partly | Graded & Multi-answer | Yes (Expert TREC pool) | No (Faceted QBE) | Cross-lingual retrieval; short idea statements vs. full query documents. |
| **Analogy Search Engine** (CMU) | Partly | Multi-answer | Yes (Expert analogies) | No (Semantic mapping) | Cross-lingual prior-art recovery; systematic ranking headroom diagnosis. |
| **Our Study** | **Yes (Strict V0–V3 ablation)** | Single-target (Proxy + Human audit) | In Progress (Human audit pending) | Evaluated (Oracle routing / Probes) | Directly establishes whether V1 misses are ranking failures or under-specified queries under cross-lingual disguise. |

---

## 5. Three Candidate Sentences for What Could Be New

Below are three distinct candidate formulations of our core contribution, each paired with the existing literature that could potentially challenge or qualify it. *(Listed without preference for your selection.)*

### Candidate 1 (Diagnostic & Theoretical Angle)
> *"We provide the first empirical diagnosis separating retriever ranking capacity limits from query under-specification when recovering prior art from pre-draft, idea-level descriptions."*
- **Potential Contradictions / Prior Work to Consider**:
  - **MAPLE (2026)** also separates retrieval failure across aspects (e.g., AnyAspect vs. AllAspect), showing that models fail on specific facets.
  - **BRIGHT (2025)** investigates whether reasoning limits or missing information cause retrieval failure in complex queries.
  - **DORIS-MAE (2023)** examines how multi-level aspect queries constrain retrieval performance across hierarchical sub-queries.

### Candidate 2 (Cross-Lingual & Controlled Disguise Angle)
> *"We establish the first controlled benchmark evaluating cross-lingual scientific prior-art recovery under systematic lexical leak removal and rhetorical facet ablation (V0–V3)."*
- **Potential Contradictions / Prior Work to Consider**:
  - **CLIRudit (2025)** already benchmarks cross-lingual scientific document retrieval for English queries searching French literature (Érudit corpus), though it uses author keywords rather than ablated idea queries.
  - **Penha et al. (2022)** systematically controls query transformations and lexical variations, though on English general-domain search.
  - **CSFCube (2021)** and **MAPLE (2026)** ablate rhetorical facets (background, method, results), though in monolingual English full-paper settings.

### Candidate 3 (Agentic / Routing Headroom Angle)
> *"We quantify the headroom between fixed cross-lingual retrievers and oracle routing, demonstrating when iterative agentic reformulation is mathematically necessary to overcome terminology shift."*
- **Potential Contradictions / Prior Work to Consider**:
  - **Chain of Retrieval (2025)** demonstrates that iterative multi-aspect expansion outperforms fixed single-step retrievers on full-paper discovery.
  - **UniFAR (2026)** explores unified facet modeling across query-document and document-document paradigms.
  - Standard hybrid fusion studies (e.g., BM25 + Dense RRF benchmarks) already show headroom over single fixed retrievers, though typically without aspect-level oracle routing.

