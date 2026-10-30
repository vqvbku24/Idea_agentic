# Final pre-freeze run checklist

This is a checklist only. No tag is created and rule v2.1 is not applied by this document.

## Human decision gate

After human review, create these tags only if the decision gate passes:

- `extractor-v2` (verify the existing tag; do not move it)
- `amendment-2` (verify the existing tag; do not move it)
- `retrieval-frozen-v2` (create only after the final run and leak decision pass)

## Colab preparation

Clone at an immutable commit or tag, then mount durable storage:

```bash
git clone <REPOSITORY_URL> /content/Idea_agentic
cd /content/Idea_agentic
git checkout <COMMIT_OR_TAG>
git status --porcelain
python scripts/colab_run_manifest.py --ref <COMMIT_OR_TAG>
```

The manifest command must be the first project command. It aborts if the working tree is dirty or any
`src/pilot/*.py`, `config/*.yaml`, or `data/processed/sources_final.json` hash differs from `HEAD`.
Copy the generated `results/real/run_meta_*.json` to Drive after it is created.

## Exact rerun commands

```bash
cd /content/Idea_agentic
python scripts/run_extraction_v2.py
python scripts/audit_x2.py
python scripts/run_retrieval_v2.py
python scripts/finalize_x2_audit.py
```

Run them with the repository on a mounted Drive, or copy the generated outputs and cache directories
back to Drive after each successful command. Do not run any LLM/provider command.

## Durable artefacts

The retrieval run must preserve:

- `results/real/translations_x2.jsonl`
- `data/cache/translations-v2/` (JSON caches include translated chunks)
- `data/cache/embeddings-v2/` (dense `.npz` caches)
- `results/real/dense_reference_x2.csv`
- `results/real/retrieval_v2_diagnostics.md`
- `results/real/blind_leak_checks_x2.csv`
- `results/real/leak_fail_cases_x2.md`
- `results/real/candidate_rule_v2_1_review.md`
- `results/real/run_meta_*.json`
## Hash manifest

Hash exactly these inputs before the human decision:

- every `src/pilot/*.py`
- every `config/*.yaml`
- `data/processed/sources_final.json`

Hash these final outputs for the audit archive:

- `results/real/papers_x2.jsonl`
- `results/real/development_papers_x2.jsonl`
- `results/real/french_distractors_native_x2.jsonl`
- `results/real/translations_x2.jsonl`
- `results/real/dense_reference_x2.csv`
- `results/real/retrieval_v2_diagnostics.md`
- `results/real/blind_leak_checks_x2.csv`
- `results/real/leak_fail_cases_x2.md`
- `results/real/candidate_rule_v2_1_review.md`
- `results/real/run_meta_*.json`
