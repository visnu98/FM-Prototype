# Evaluation Summary

- Total query runs: **140**
- Models: **llama-3.1-8b-instant, qwen/qwen3-32b**
- Unique queries: **70**

## Metrics by model

| model                |   fully_correct_call |   function_accuracy |   parameter_accuracy |   answer_accuracy |   execution_success |   mean_latency_ms |
|:---------------------|---------------------:|--------------------:|---------------------:|------------------:|--------------------:|------------------:|
| llama-3.1-8b-instant |                0.771 |               0.929 |                0.771 |             0.714 |               0.843 |         15368.638 |
| qwen/qwen3-32b       |                0.914 |               0.957 |                0.914 |             0.914 |               0.957 |         19824.105 |

## Metrics by complexity level (answer accuracy)

| complexity_level   |   llama-3.1-8b-instant |   qwen/qwen3-32b |
|:-------------------|-----------------------:|-----------------:|
| L1                 |                  0.750 |            1.000 |
| L2                 |                  0.515 |            0.818 |
| L3                 |                  0.938 |            1.000 |
| L4                 |                  0.923 |            1.000 |

## Metrics by category (fully-correct call)

| category    |   llama-3.1-8b-instant |   qwen/qwen3-32b |
|:------------|-----------------------:|-----------------:|
| aggregation |                  0.875 |            1.000 |
| attribute   |                  1.000 |            1.000 |
| counting    |                  0.571 |            0.786 |
| metadata    |                  1.000 |            1.000 |
| multistep   |                  0.846 |            1.000 |