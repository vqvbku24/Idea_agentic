# MANUAL_TASKS.md — Human-in-the-loop tasks for the prior-art pilot

**Purpose:** list every task that is done by a person (or a delegated agent **verified by a person**) rather than by the pipeline, with inputs, steps, output formats, integrity rules and a verification protocol.
**Why these tasks exist:** they are the independent checks on the pipeline's LLM steps. If the same model produces and judges its own work, the check proves nothing.
**Status:** nothing here has been run. Paths follow `PILOT_SPEC.md` and the repository as it stands after tag `sources-frozen`.

---

## 0. Two modes

| Mode | Who produces the labels | Who may claim what |
|---|---|---|
| **A. Human mode** | The researcher | Outputs may be called "human-written" / "human-coded" |
| **B. Delegated-agent mode** | A *separate* agent, then the researcher **verifies** | Outputs must be called **"agent-drafted, human-verified (k of n checked)"**. They must **not** be called human-written |

Mode B is acceptable for getting early results quickly, but every report that uses its outputs must say so. The hypothesis tests that depend on independence (E5 human-vs-synthetic transfer, E6 facet coverage) become weaker under Mode B and must be labelled **"H-proxy"** in the report.

### Rules for Mode B (all mandatory)

1. **Different agent.** The delegated agent must be a different model or session from the one that generates variants or judges them in the pipeline (`LLM_MODEL` / `JUDGE_MODEL`).
2. **Blindness.** The delegated agent receives **only** the input files named in each task. It must not have access to `results/`, `queries.jsonl`, any LLM facet output, retrieval results, or `PILOT_SPEC.md` decision rules.
3. **Fresh session per pass** where a task says so (see T4).
4. **Provenance labels.** Every output row carries `written_by` (`agent:<model>` or `human`) and `verified_by_human` (`pending`, `yes`, `edited`).
5. **No silent fixes.** If the researcher edits an agent's output, the original is kept in a separate column.
6. **Tags tell the truth.** Use `human-queries-frozen` and `facet-audit-human` only when the researcher wrote or fully re-coded the content. Otherwise use `proxy-queries-frozen` and `facet-audit-proxy`.

---

## 1. Overview and order

| ID | Task | Delegable (Mode B)? | Needed before | Verification by researcher | Est. human time (Mode A) |
|---|---|---|---|---|---|
| **T1** | Extraction review of 20 final sources | Yes | Pipeline Part 2 | All flagged + 5 random unflagged | 45 min |
| **T2** | Near-duplicate review | Yes | Pipeline Part 2 | Every "yes" + 5 random "no" | 30 min |
| **T3** | 10 human-written queries | Yes (drafts) | LLM variant generation | **All 10** | 2–3 h |
| **T4** | Facet-coverage audit (10 papers × 4 facets × 2 sources) | Yes (two separate sessions) | Looking at any LLM facet output | **All abstract cells + body cells for 4 papers** | 2–3 h |
| **T5** | Configuration: API keys, model choice, cost cap, exclusion decisions | **No** | Pipeline Part 2 | n/a | 30 min |
| **T6** | Natural Pairs: English counterparts of French papers | Yes | Optional | Every "yes" candidate | 1–2 h |
| **T7** | Spot-check of 30 LLM-generated variants | Yes (different agent from generator and judge) | After variants exist | 10 of 30 | 45 min |
| **T8** | One-line failure explanations | Partly (draft only) | After retrieval results | All entries | 1 h |
| **T9** | Leak-check review (blind-input FAIL cells) | Draft only; **human signs** | Tag `retrieval-frozen-v2`, blind inputs for T3/T4 | **All** (about 10 items) | 15 min |
| **T10** | Review of candidate extractor rule v2.1 (changed abstracts) | Draft only; **human signs** | Same | **All** (about 26 items) | 30 min |

**Order that preserves integrity:** T5 → (T1, T2, T3, T4 in any order, T3 and T4 before the pipeline sees the matching papers) → pipeline Part 2 → T7 → T8. T6 any time.

---

## 2. Shared files

```
data/human/
  human_assignments.csv       # already created: the 10 papers for T3 (seed 13)
  human_queries.csv           # T3 output
  human_notes.md              # T3 notes
  review_notes.csv            # T1 output
  near_dup_review.csv         # T2 output
  facet_audit_abstract.csv    # T4 pass A output
  facet_audit_body.csv        # T4 pass B output
  natural_pairs.csv           # T6 output
  spotcheck.csv               # T7 output
  verification_log.csv        # all tasks: agreement between agent and researcher
results/real/
  extraction_review.md        # T1 input
  near_duplicates.md          # T2 input
  facet_audit_sheet.csv       # T4 template (paper_id, facet, in_en_abstract, in_fr_body)
  facet_audit_materials.jsonl # T4 input
```

`verification_log.csv` columns: `task, item_id, field, agent_value, human_value, agree (1/0), note`. Agreement is computed from this file.

---

## 3. T1 — Extraction review (20 final sources)

**Goal:** confirm the frozen extractor produced usable abstracts and body text.
**Input:** `results/real/extraction_review.md` (paper ID, track, first 300 characters of French abstract, English abstract, body, provenance, flags).

**Steps**
1. For each paper, check: (a) the English abstract is a complete author-written abstract: not French, not truncated at either end, and **without the paper's title or a challenge label prepended**; (b) the French abstract is French, complete (it starts at the first word of the abstract, **not mid-sentence**) and from the same paper; (c) the body excerpt: record whether it contains **repeated page headers/footers**, a **reference list**, or a **copy of the abstract text** (the last matters for leakage between index views). A title/author block at the very start of the body is expected: note it, do not penalize it.
2. Mark `broken` only if (a) or (b) fails or an abstract is missing or from the wrong paper; `suspicious` for title-prefixed abstracts, repeated headers/footers, reference text or abstract duplication in the body; `ok` otherwise (with a note if there is a title/author block).
3. **Do not edit data. Do not drop papers.** If you want to exclude a paper, write the reason in `review_notes.csv`, and apply the same rule to every paper, **before** any retrieval result is seen.

**Output:** `data/human/review_notes.csv` — `paper_id, status (ok|suspicious|broken), note, written_by, verified_by_human`.

**Verification (Mode B):** check every `suspicious` or `broken`, and 5 random `ok`. Log disagreements in `verification_log.csv`.

**Delegation prompt**
```
You receive results/real/extraction_review.md only. For each of the 20 papers decide whether
(a) the English abstract is a complete author-written abstract, not truncated at either end, with no title or challenge label prepended,
(b) the French abstract is French, complete (starts at its first word, not mid-sentence) and from the same paper,
(c) whether the body excerpt contains repeated page headers/footers, reference text or a copy of the abstract (a title/author block at the very start is expected: note it only).
Use status broken only if (a) or (b) fails; suspicious for the problems in (c) or a title prefix in (a).
Output a CSV with paper_id,status(ok|suspicious|broken),note. Do not modify any file, do not exclude papers.
Quote the exact fragment that makes you doubt an item.
```

---

## 4. T2 — Near-duplicate review

**Goal:** find English papers in the corpus that already contain a source's contribution (extended or translated version), because they (a) can make a correct retrieval look wrong and (b) are real natural labels.
**Input:** `results/real/near_duplicates.md` (top 3 English candidates per source with similarity).

**Steps:** for each candidate decide `yes` (same contribution by the same group, extension or translation), `partial`, `no`, or `unsure`. Open the candidate's page if title and abstract are not enough.

**Output:** `data/human/near_dup_review.csv` — `source_id, candidate_id, same_contribution, evidence, written_by, verified_by_human`.

**Rule:** a `yes` candidate is **not removed** from the corpus. The evaluation will treat it as an additional accepted answer and report it separately.

**Verification (Mode B):** open every `yes` and 5 random `no`.

**Delegation prompt**
```
You receive results/real/near_duplicates.md only. For each source and candidate decide whether the candidate is
the same contribution (an extended or translated version by the same authors): yes / partial / no / unsure.
Cite evidence from the two abstracts (author overlap, same method and result). If you can browse, check authors and venue;
say whether you did. Output CSV: source_id,candidate_id,same_contribution,evidence. Do not modify files.
```

---

## 5. T3 — Ten human-written queries

**Goal:** a query set that is independent of the LLM pipeline, to test whether results on synthetic disguises transfer.
**Input:** `data/human/human_assignments.csv` (10 IDs, chosen at random with seed 13 so that easy papers are not picked on purpose). For each, only the **abstracts** (not the title).

**Mode A steps**
1. Read the abstract (translate the French one if needed), then **close it**.
2. Write one English description of the idea, 40–80 words, as if telling a colleague what the paper does and why.
3. Rules: no copying of phrases, no title words, no proper names, acronyms, dataset or system names, no numbers; describe terms instead of naming them.
4. Note in `human_notes.md`: which language you read, which papers were hard (demo, DEFT, speech papers may have no clear idea), and how long you spent.
5. Freeze: commit, tag `human-queries-frozen`. **Do not edit the file afterwards.**

**Output:** `data/human/human_queries.csv` — `source_id, query_text, written_by, date, verified_by_human, query_text_agent`.

**Mode B:** the delegated agent writes the drafts (`written_by = agent:<model>`); you then **read all 10**, compare each with the abstract, and edit as needed (keep the original in `query_text_agent`; set `verified_by_human` to `yes` or `edited`). Tag `proxy-queries-frozen` unless you rewrote them yourself.

**Verification checklist (every query)**
- [ ] Faithful: it describes this paper, not just its topic
- [ ] No leak: no title words, proper names, acronyms, dataset or system names, numbers
- [ ] Not a near-copy of abstract sentences (compare side by side)
- [ ] 40–80 words, English, fluent
- [ ] Flag papers where no clear idea exists (record, do not drop)

**Limitation to report:** writing from the abstract biases queries toward abstract wording. Say so in the paper.

**Delegation prompt**
```
You receive, for 10 papers, only the English and French abstract (no title). For each paper write one English description
of the research idea in 40-80 words, in your own words, as if explaining to a colleague. Do not copy phrases, do not
use proper names, acronyms, dataset or system names, or numbers; describe them instead. If a paper has no clear
research idea (demo, challenge system), still write the best faithful description and add "NO_CLEAR_IDEA" in a note column.
Output CSV: source_id,query_text,note. You must not have seen any other generated query set for these papers.
```

---

## 6. T4 — Facet-coverage audit (RQ6)

**Goal:** measure whether the author-written English abstract keeps the facets that appear in the French body. This decides whether the language axis is real.
**Input:** `results/real/facet_audit_materials.jsonl` for 10 randomly chosen papers (seed 13): English abstract and first 1500 body words.

**Facet definitions** (write them down once and keep them fixed)
- **problem:** the task or problem the paper addresses
- **mechanism:** the method, algorithm or technical approach
- **setting:** data, corpus, language, domain, conditions or experimental setup
- **outcome:** results, findings or the claimed contribution/evaluation

**Coding rule:** `1` only if the facet is **explicitly stated**. `0` if absent or only implicit. `NA` if the paper has no such facet (for example a demo with no evaluation). Note one line of evidence (≤ 15 words) for every `1`.

**Two separate passes (never together)**
- **Pass A:** English abstract only → `facet_audit_abstract.csv`
- **Pass B:** French body only → `facet_audit_body.csv`
Code all of Pass A before starting Pass B. In Mode B use **two fresh sessions**, each receiving only its own input.

**Output columns:** `paper_id, facet, value (1|0|NA), evidence, written_by, verified_by_human`.

**Verification (Mode B):** re-code **all 10 abstract cells (40 judgements)** yourself, and the **body** cells for 4 papers chosen with seed 13 (16 judgements). Compute agreement per pass in `verification_log.csv`. If agreement on either pass is below about 80%, re-code that whole pass yourself (heuristic threshold). If you do not read French well, have a colleague code 3 body papers and log the agreement.

**Do not look at any LLM facet extraction until both passes are committed and tagged `facet-audit-human` (or `facet-audit-proxy`).**

**Delegation prompts**
```
PASS A (fresh session). You receive only the English abstracts of 10 papers. For each paper and each facet
(problem, mechanism, setting, outcome) answer 1 if the facet is explicitly stated, 0 if absent or only implicit, NA if the paper has
no such facet. Give evidence of at most 15 words for each 1. Definitions: problem = task addressed; mechanism = method or
technical approach; setting = data/corpus/language/domain/conditions; outcome = results/findings/contribution.
Output CSV: paper_id,facet,value,evidence.
```
```
PASS B (a different fresh session). You receive only the first 1500 words of the French body of the same 10 papers.
Apply the same definitions and rule. Output CSV: paper_id,facet,value,evidence. You have not seen the abstracts.
```

---

## 7. T5 — Configuration (not delegable)

1. Choose the generator model (`LLM_MODEL`) and a **different** judge model (`JUDGE_MODEL`); set `LLM_PROVIDER` and keys in an environment file kept out of git.
2. Set a cost cap (`LLM_MAX_CALLS`, optionally a spend limit at the provider).
3. Check the machine can run `BAAI/bge-m3` and `Helsinki-NLP/opus-mt-fr-en`; if not, choose an API embedding option and record it.
4. Decide, **before seeing any results**, whether any paper will be excluded (from T1) and write the rule in `data/human/exclusions.md`.
5. Record what you can read (French, English) in `human_notes.md`; it affects what you may verify.

---

## 8. T6 — Natural pairs (optional)

**Goal:** find English counterparts (journal extension, translation) of the French sources, which are genuine cross-lingual same-contribution pairs.
**Input:** the 20 final sources (title, authors, year).

**Steps:** search the authors and keywords on HAL, ACL Anthology, Google Scholar or the group's page. For each candidate record URL and decide `yes / partial / no / unsure`.

**Output:** `data/human/natural_pairs.csv` — `source_id, candidate_url, same_contribution, notes, written_by, verified_by_human`.

**Verification (Mode B):** open every `yes`/`partial` link and confirm authors and content; log any wrong ones.

**Delegation prompt**
```
For each of these 20 French papers (title, authors, year) look for an English extension or translation by the same group.
Report candidate_url, same_contribution (yes/partial/no/unsure) and a note. State whether you actually opened the page.
Never invent URLs; write "not found" if you cannot find one.
```

---

## 9. T7 — Spot-check of 30 LLM variants (after Part 2 produces them)

**Input:** a random sample of 30 accepted variants with the facet JSON they came from.
**Check each:** faithful to the kept facets; no leak of title words, names or numbers; fluent English. Mark `ok / unfaithful / leaks / disfluent`.
**Output:** `data/human/spotcheck.csv` — `variant_id, status, note, written_by, verified_by_human`.
**Mode B:** a third agent (not the generator, not the judge); you re-check 10 of the 30. Report the share the judge accepted that you reject.

---

## 10. T8 — Failure explanations

After retrieval, `failure_cases.md` has one entry per miss. Add one line per entry: why it likely failed (terminology gap, facet dropped, language, abstract lacks mechanism, domain shift, topical neighbour outranked). An agent may draft; you read **all** entries and edit. Mark the final text as human-verified.

---

## 10b. T9 — Leak-check review (gate for the blind inputs)

**Input:** `results/real/leak_fail_cases_x2.md` (file, paper_id, first 200 characters of `abstract_en`, rule that fired).
**For each FAIL:** `true_leak` if the first words of the abstract are the paper's English title or a challenge label (it would reveal the paper); `false_positive` if the abstract genuinely starts with an ordinary sentence.
**Output:** `data/human/leak_review.csv` — `file, paper_id, status, evidence, reviewer, signed (initials + date)`.
**Rule:** a FAIL may be waived only by a **human-signed** `false_positive`. An agent draft alone does not open the gate for the T3/T4 handoff or for tag `retrieval-frozen-v2`.

## 10c. T10 — Review of candidate rule v2.1

**Input:** `results/real/candidate_rule_v2_1_review.md` (each abstract the rule would change: text before/after, the removed line, the branch that fired).
**For each:** `correct` if the removed line is a title or label; `false_positive` if it is the first sentence of the abstract; `unsure`.
**Output:** `data/human/v21_review.csv` — `set, paper_id, status, removed_text, reviewer, signed`.
**Decision (heuristic, fix it before looking):** apply v2.1 to **all** documents only if false positives are at most about 2 of 26 and none are among the 20 sources; otherwise reject v2.1 and use the declared sensitivity analysis instead. Either way, the same decision applies to sources, dev and distractors.

---

## 11. Verification protocol and sign-off

| Task | Minimum human check | Agreement threshold (heuristic) | If below |
|---|---|---|---|
| T1 | all flagged + 5 random ok | — | Re-review all |
| T2 | all "yes" + 5 random "no" | any wrong "yes" is logged | Re-review all |
| T3 | all 10 queries | — | Edit; keep originals |
| T4 | all abstract cells + 4 body papers | ≥ ~80% | Re-code that pass fully |
| T6 | all "yes"/"partial" | — | Remove unverified |
| T7 | 10 of 30 | — | Report reject rate |

At the end write `data/human/SIGNOFF.md`:

```
Task | mode (A/B) | agent model | items | checked by me | edited | agreement | date
```
Every claim in the paper about "human" labels must match this table.

---

## 12. What the report must say

- Which tasks were Mode A and which Mode B, with the sign-off table.
- For H queries and the facet audit: whether they are human or **proxy** (agent-drafted, human-verified on k of n items), and the agreement numbers.
- That human queries were written from abstracts and may resemble them.
- That conclusions about transfer (E5) and facet coverage (E6) are weaker under Mode B.
