# Model Comparison

| model                |      n |   fully_correct_call |   function_accuracy |   parameter_accuracy |   answer_accuracy |   execution_success |   mean_latency_ms |   median_latency_ms |
|:---------------------|-------:|---------------------:|--------------------:|---------------------:|------------------:|--------------------:|------------------:|--------------------:|
| llama-3.1-8b-instant | 70.000 |                0.771 |               0.929 |                0.771 |             0.714 |               0.843 |         15368.638 |           16159.461 |
| qwen/qwen3-32b       | 70.000 |                0.914 |               0.957 |                0.914 |             0.914 |               0.957 |         19824.105 |           20043.794 |

## Per-model latency breakdown (mean ms)

| model                |   latency_tool_call |   latency_sql |   latency_final_answer |   latency_total |
|:---------------------|--------------------:|--------------:|-----------------------:|----------------:|
| llama-3.1-8b-instant |             11197.6 |         120.3 |                 4050.7 |         15368.6 |
| qwen/qwen3-32b       |             13083.0 |         141.9 |                 6599.2 |         19824.1 |