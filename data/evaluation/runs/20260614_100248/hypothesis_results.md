# Hypothesis Results

| Hypothesis | Verdict | Test | p-value |
| --- | --- | --- | --- |
| H1 | **not supported** | binomial (one-sided, p>0.90), Wilson CI | 0.9880 |
| H2 | **supported** | McNemar (exact), paired within paraphrase groups | 1.0000 |
| H3 | **supported** | McNemar (exact) on correctness; Wilcoxon signed-rank on latency | 0.0063 |
| H4 | **not supported** | logistic regression (correctness ~ level) | 0.0083 |

## H1 — **not supported**

- **Metric:** fully_correct_call rate
- **Test:** binomial (one-sided, p>0.90), Wilson CI
- **p-value:** 0.9880
- **Result:** {'overall_rate': 0.843, 'overall_ci': [0.774, 0.894], 'per_model': {'llama-3.1-8b-instant': {'successes': 54, 'n': 70, 'rate': 0.771, 'wilson_ci': [0.66, 0.854], 'p_value_greater': 0.999566480188762, 'meets_threshold': False}, 'qwen/qwen3-32b': {'successes': 64, 'n': 70, 'rate': 0.914, 'wilson_ci': [0.825, 0.96], 'p_value_greater': 0.441809757229648, 'meets_threshold': True}}}
- **Interpretation:** Pooled correct-call rate is 84.3% (below the 90% target).
- **Limitations:** Single dataset/facility; corpus authored by the developer.

## H2 — **supported**

- **Metric:** fully_correct_call (standard vs paraphrase)
- **Test:** McNemar (exact), paired within paraphrase groups
- **Statistic:** 4.0000
- **p-value:** 1.0000
- **Result:** {'pooled_discordant': {'std_correct_para_wrong': 4, 'std_wrong_para_correct': 5}, 'per_model': {'llama-3.1-8b-instant': {'standard_rate': 0.767, 'paraphrase_rate': 0.775, 'discordant_std_correct_para_wrong': 4, 'discordant_std_wrong_para_correct': 3, 'p_value': 1.0}, 'qwen/qwen3-32b': {'standard_rate': 0.867, 'paraphrase_rate': 0.95, 'discordant_std_correct_para_wrong': 0, 'discordant_std_wrong_para_correct': 2, 'p_value': 0.5}}}
- **Interpretation:** No significant difference between standard and reworded queries (p=1.000); paraphrasing did not lower correctness.
- **Limitations:** Few discordant pairs make McNemar low-powered on a small corpus.

## H3 — **supported**

- **Metric:** fully_correct_call + latency_total
- **Test:** McNemar (exact) on correctness; Wilcoxon signed-rank on latency
- **Statistic:** 1.0000
- **p-value:** 0.0063
- **Result:** {'model_a': 'qwen/qwen3-32b', 'model_b': 'llama-3.1-8b-instant', 'correct_rate_a': 0.914, 'correct_rate_b': 0.771, 'discordant_a_correct_b_wrong': 11, 'discordant_b_correct_a_wrong': 1, 'latency_wilcoxon_stat': 658.0, 'latency_wilcoxon_p': 0.0006248298875215616, 'median_latency_a_ms': 20043.8, 'median_latency_b_ms': 16159.5}
- **Interpretation:** Reliability differs between models (McNemar p=0.006).
- **Limitations:** McNemar needs discordant pairs; latency depends on network/load.

## H4 — **not supported**

- **Metric:** answer_correct by complexity level
- **Test:** logistic regression (correctness ~ level)
- **Statistic:** 0.7408
- **p-value:** 0.0083
- **Result:** {'rate_by_level': {'L1': 0.875, 'L2': 0.667, 'L3': 0.969, 'L4': 0.962}, 'monotonic_decrease': False, 'slope_negative': False}
- **Interpretation:** No clear decreasing trend in correctness across complexity levels.
- **Limitations:** Few queries per level limit statistical power.
