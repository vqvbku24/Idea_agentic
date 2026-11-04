> [!IMPORTANT]
> **EXPLORATORY - agent-proxy labels, human verification pending**

# Identifiability Probe: Disentangling Ranking Failure vs Under-Specified Queries

## 1. Executive Summary & Decision Rule Verdicts

This probe tests whether an LLM judge (`openai/gpt-oss-120b`, temp=0.0) can identify the true source paper when forced into the top-20 candidate pool.

| Decision Rule / Heuristic | Condition & Threshold | Observed Value | Programmatic Verdict |
|---|---|---|---|
| **identifiable** | In oracle-inserted condition, source ranked top-3 in `>= 60.0%` of V1 misses | `100.0%` | **`True`** |
| **under_specified** | In oracle-inserted condition, source ranked top-3 in `< 30.0%` | `100.0%` | **`False`** |
| **format_leak_control** | No-query control accuracy `<= 10.0%` (random baseline = 5.0%) | `8.0%` | **`True`** (PASS) |

## 2. Probe Summary Table Across Conditions

| Query Subgroup | N | As-Retrieved Top-1 | As-Retrieved Top-3 | Oracle-Inserted Top-1 | Oracle-Inserted Top-3 [95% CI] | No-Query Control Acc [95% CI] |
|---|---|---|---|---|---|---|
| `v1_miss` | 5 | 0.600 | 0.600 | **0.800** | **1.000** [0.478, 1.000] | 0.200 [0.005, 0.716] |
| `v1_hit` | 10 | 1.000 | 1.000 | **1.000** | **1.000** [0.692, 1.000] | 0.100 [0.003, 0.445] |
| `v0_calibration` | 10 | 1.000 | 1.000 | **1.000** | **1.000** [0.692, 1.000] | 0.000 [0.000, 0.308] |
| `ALL` | 25 | 0.920 | 0.920 | **0.960** | **1.000** [0.863, 1.000] | 0.080 [0.010, 0.260] |

## 3. Individual Test Item Details

| Query ID | Group | Dense Rank | In Top-20? | As-Retrieved Top-1 | As-Retrieved Top-3 | Oracle-Ins Top-1 | Oracle-Ins Top-3 | No-Query Leak? |
|---|---|---|---|---|---|---|---|---|
| `2019.jeptalnrecital-court.7_V1_9` | `v1_miss` | 15 | True | True | True | **False** | **True** | False |
| `2019.jeptalnrecital-recital.2_V1_69` | `v1_miss` | 1 | True | True | True | **True** | **True** | False |
| `2020.jeptalnrecital-deft.5_V1_73` | `v1_miss` | 38 | False | False | False | **True** | **True** | True |
| `2020.jeptalnrecital-taln.1_V1_41` | `v1_miss` | 45 | False | False | False | **True** | **True** | False |
| `2021.jeptalnrecital-deft.7_V1_77` | `v1_miss` | 17 | True | True | True | **True** | **True** | False |
| `2015.jeptalnrecital-court.13_V1_21` | `v1_hit` | 1 | True | True | True | **True** | **True** | False |
| `2015.jeptalnrecital-recital.3_V1_53` | `v1_hit` | 1 | True | True | True | **True** | **True** | False |
| `2016.jeptalnrecital-jep.41_V1_57` | `v1_hit` | 1 | True | True | True | **True** | **True** | False |
| `2016.jeptalnrecital-jep.65_V1_25` | `v1_hit` | 1 | True | True | True | **True** | **True** | False |
| `2017.jeptalnrecital-court.5_V1_29` | `v1_hit` | 3 | True | True | True | **True** | **True** | False |
| `2018.jeptalnrecital-court.4_V1_5` | `v1_hit` | 1 | True | True | True | **True** | **True** | False |
| `2018.jeptalnrecital-long.14_V1_33` | `v1_hit` | 2 | True | True | True | **True** | **True** | True |
| `2018.jeptalnrecital-long.1_V1_65` | `v1_hit` | 1 | True | True | True | **True** | **True** | False |
| `2021.jeptalnrecital-taln.4_V1_13` | `v1_hit` | 1 | True | True | True | **True** | **True** | False |
| `2024.jeptalnrecital-jep.32_V1_49` | `v1_hit` | 1 | True | True | True | **True** | **True** | False |
| `2016.jeptalnrecital-jep.65_V0_24` | `v0_calibration` | 1 | True | True | True | **True** | **True** | False |
| `2017.jeptalnrecital-court.2_V0_0` | `v0_calibration` | 1 | True | True | True | **True** | **True** | False |
| `2017.jeptalnrecital-recital.5_V0_60` | `v0_calibration` | 1 | True | True | True | **True** | **True** | False |
| `2018.jeptalnrecital-long.1_V0_64` | `v0_calibration` | 1 | True | True | True | **True** | **True** | False |
| `2019.jeptalnrecital-court.18_V0_36` | `v0_calibration` | 1 | True | True | True | **True** | **True** | False |
| `2019.jeptalnrecital-court.7_V0_8` | `v0_calibration` | 1 | True | True | True | **True** | **True** | False |
| `2019.jeptalnrecital-recital.2_V0_68` | `v0_calibration` | 1 | True | True | True | **True** | **True** | False |
| `2020.jeptalnrecital-deft.5_V0_72` | `v0_calibration` | 1 | True | True | True | **True** | **True** | False |
| `2020.jeptalnrecital-taln.1_V0_40` | `v0_calibration` | 1 | True | True | True | **True** | **True** | False |
| `2023.jeptalnrecital-long.11_V0_16` | `v0_calibration` | 1 | True | True | True | **True** | **True** | False |