# Error Analysis

| error_category                        |   llama-3.1-8b-instant |   qwen/qwen3-32b |
|:--------------------------------------|-----------------------:|-----------------:|
| correct_call_incorrect_final_response |                      6 |                2 |
| incorrect_parameter_value             |                     11 |                3 |
| none                                  |                     48 |               62 |
| wrong_function_selected               |                      5 |                3 |
| missing_required_parameter            |                      0 |                0 |
| hallucinated_or_unsupported_function  |                      0 |                0 |
| execution_or_runtime_failure          |                      0 |                0 |
| ambiguity_or_underspecified_query     |                      0 |                0 |
| no_tool_call_when_tool_required       |                      0 |                0 |
| invalid_json_or_schema_error          |                      0 |                0 |

## Most common failure cases

- `q17` [qwen/qwen3-32b] **incorrect_parameter_value** — expected `count_components`, got `count_components`.
- `q18` [qwen/qwen3-32b] **incorrect_parameter_value** — expected `count_components`, got `count_components`.
- `q28` [qwen/qwen3-32b] **correct_call_incorrect_final_response** — expected `get_component_attributes`, got `get_component_attributes`.
- `q29` [qwen/qwen3-32b] **correct_call_incorrect_final_response** — expected `get_component_attributes`, got `get_component_attributes`.
- `q51` [qwen/qwen3-32b] **wrong_function_selected** — expected `count_components`, got `list_queryable_floors`.
- `q57` [qwen/qwen3-32b] **wrong_function_selected** — expected `count_components`, got `list_queryable_component_types`.
- `q58` [qwen/qwen3-32b] **incorrect_parameter_value** — expected `count_components`, got `count_components`.
- `q59` [qwen/qwen3-32b] **wrong_function_selected** — expected `count_components`, got `list_queryable_component_types`.
- `q02` [llama-3.1-8b-instant] **correct_call_incorrect_final_response** — expected `list_queryable_floors`, got `list_queryable_floors`.
- `q03` [llama-3.1-8b-instant] **correct_call_incorrect_final_response** — expected `list_queryable_floors`, got `list_queryable_floors`.
- `q12` [llama-3.1-8b-instant] **incorrect_parameter_value** — expected `count_components`, got `count_components`.
- `q17` [llama-3.1-8b-instant] **incorrect_parameter_value** — expected `count_components`, got `count_components`.
- `q18` [llama-3.1-8b-instant] **incorrect_parameter_value** — expected `count_components`, got `count_components`.
- `q19` [llama-3.1-8b-instant] **incorrect_parameter_value** — expected `count_components`, got `count_components`.
- `q20` [llama-3.1-8b-instant] **incorrect_parameter_value** — expected `count_components`, got `count_components`.
- `q21` [llama-3.1-8b-instant] **wrong_function_selected** — expected `count_components`, got `get_database_capabilities`.
- `q22` [llama-3.1-8b-instant] **wrong_function_selected** — expected `count_components`, got `get_database_capabilities`.
- `q25` [llama-3.1-8b-instant] **correct_call_incorrect_final_response** — expected `get_component_attributes`, got `get_component_attributes`.
- `q26` [llama-3.1-8b-instant] **correct_call_incorrect_final_response** — expected `get_component_attributes`, got `get_component_attributes`.
- `q28` [llama-3.1-8b-instant] **correct_call_incorrect_final_response** — expected `get_component_attributes`, got `get_component_attributes`.