> [!IMPORTANT]
> **EXPLORATORY - agent-proxy labels, human verification pending**

# English Terminology Dose-Response Pilot: DEV Run Acceptance Report

Evaluated on 30 DEV papers across 4 retention levels (total queries: 120).

## 1. Acceptance and Rejection Breakdown by Level

| Retention Level | Total | Accepted | Rejection Rate | Acceptance Rate | Mean Realized Retention |
|---|---|---|---|---|---|
| `R100` | 30 | **20** | 10 (33.3%) | **66.7%** | 10.33 terms |
| `R50` | 30 | **6** | 24 (80.0%) | **20.0%** | 4.83 terms |
| `R25` | 30 | **5** | 25 (83.3%) | **16.7%** | 2.40 terms |
| `R0` | 30 | **18** | 12 (40.0%) | **60.0%** | 0.00 terms |

## 2. Granular Failure Reasons by Level

| Retention Level | Failure Reason Category | Count | Example Detail |
|---|---|---|---|
| `R100` | `lexical_leak_high` | 6 | lexical_leak_high (0.333 > 0.30) |
| `R100` | `judge_rejected` | 4 | judge_rejected (unfaithful): The query accurately captures the main pr |
| `R50` | `removed_terms_present` | 12 | removed_terms_present: linguistic problems, working languages |
| `R50` | `lexical_leak_high` | 11 | lexical_leak_high (0.429 > 0.30) |
| `R50` | `retention_mismatch` | 1 | retention_mismatch (realized 4 vs target 6 +- 1) |
| `R25` | `lexical_leak_high` | 10 | lexical_leak_high (0.364 > 0.30) |
| `R25` | `removed_terms_present` | 14 | removed_terms_present: word embeddings, word embedding models |
| `R25` | `word_count_out_of_range` | 1 | word_count_out_of_range (81) |
| `R0` | `judge_rejected` | 6 | judge_rejected (unfaithful): The query captures the idea of building e |
| `R0` | `lexical_leak_high` | 3 | lexical_leak_high (0.333 > 0.30) |
| `R0` | `removed_terms_present` | 2 | removed_terms_present: textual narrative |
| `R0` | `word_count_out_of_range` | 1 | word_count_out_of_range (84) |

## 3. Human Sample Sheet Status

- Sampled **20 blind queries** into `results/real/en/human_sample_en.csv`.
- Human evaluation columns (`human_faithful`, `human_hides_terms`, `human_fluent`) left strictly **EMPTY** per Rule G2.
- Sealed key preserved in `results/real/en/sealed/human_sample_key.csv`.

## 4. Next Step

A human reviewer must inspect the 20 blind samples, record ratings, and execute:
```bash
python scripts/sign_en.py --reviewer <initials>
```