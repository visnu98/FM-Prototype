# Architecture (for the thesis)

> A self-contained description of the prototype's architecture, design
> principles and evaluation method, written to be reused in the thesis text.

## 1. Problem and approach

Facility managers need to retrieve structured Facility Management / BIM
information from existing building data using natural language. The risk with a
naive "text-to-SQL" chatbot is that a Large Language Model generates and
executes arbitrary SQL — uncontrolled, hard to audit, and unsafe on a
production database.

This prototype takes the opposite approach: **controlled function calling**. The
LLM never writes SQL. It may only *select* one of a fixed set of predefined
Python functions and provide arguments. A registry validates every proposed call
before any database access, and only approved calls run. This directly answers
**Sub-RQ 1** (architectural principles for secure, controlled retrieval).

## 2. Design principles

1. **Schema first.** The database is discovered and documented before any
   retrieval function is designed. No table, column, floor label or component
   type is assumed; all are derived from the live schema and data
   (`schema_discovery`, `data_dictionary`). This is **Sub-RQ 2** — the concrete
   steps to build the prototype.
2. **Least privilege.** A read-only access layer enforces read-only transactions,
   rejects any non-SELECT statement, applies connection and statement timeouts,
   and uses only parameterised SQL. User input is never concatenated into SQL.
3. **Whitelisted capability.** The LLM's action space is exactly the registered
   tools. Unknown ("hallucinated") functions, missing/extra/mis-typed parameters
   and invalid enum values are rejected and classified before execution.
4. **Normalization at the boundary.** Natural-language terms (English/German
   synonyms, positional floor phrases) are mapped to real database values, with
   both the original and normalized value recorded.
5. **Grounded answers.** The final natural-language answer is generated only from
   the structured tool result, with source IDs where available.
6. **Total traceability.** Every call is logged (timestamp, model, query,
   function, raw + normalized arguments, validation/execution status, latency,
   error category), enabling the evaluation and audit.

## 3. Components

```
Streamlit UI ──► ChatbotService
                    │
                    ├─► LLMClient (Groq; MODEL_A / MODEL_B)   selects a tool
                    │
                    ├─► ToolRegistry   validates + executes + logs + classifies
                    │        │
                    │        ├─► normalization   NL terms ─► real DB values
                    │        └─► fm_functions     parameterised read-only SQL
                    │                 │
                    │                 └─► db (read-only engine) ─► PostgreSQL
                    │
                    └─► LLMClient   phrases a grounded final answer
```

The same `ChatbotService` pipeline is reused by the evaluation runner, so the
experiment and the live chatbot cannot diverge.

## 4. Data model used (discovered, facility 124851)

- **Components**: `asset_component`, with the IFC type in `ext_object`.
- **Floor of a component**: resolved through space placement
  `asset_component → space_components → space.floor_id → floor`. (Components not
  placed in a space are excluded from floor-scoped queries — a documented data
  limitation.)
- **Geometry**: German key/value attributes (`Breite`/`Höhe`) in `attribute`,
  joined via `asset_component_attributes`; area is the stored area attribute when
  present, otherwise width × height. Values are parsed from German number format.
- **Scoping**: the database is multi-facility, so every function filters by
  `DEFAULT_FACILITY_ID`.

## 5. Evaluation method

- **Corpus**: ~70 natural-language queries across five categories (metadata,
  attribute, counting, aggregation, multi-step) and four complexity levels
  (L1–L4), organised into paraphrase groups for H2.
- **Ground truth**: computed deterministically by executing the expected function
  with canonical arguments (parameterised SQL), independent of the LLM.
- **Metrics**: answer correctness (exact for counts, ±tolerance for areas, set
  recall for lists), function-selection correctness, parameter accuracy,
  execution success, latency, and a nine-category error taxonomy.
- **Models**: two Groq models (`llama-3.3-70b-versatile`,
  `llama-3.1-8b-instant`) run with identical prompts, tools and temperature
  (0.0) — answering **Sub-RQ 3** (accuracy/reliability of NL→function-call
  translation) and **Sub-RQ 4** (model comparison).

## 6. Hypotheses and tests

| Hypothesis | Statement | Test |
| --- | --- | --- |
| H1 | ≥ 90% correct function calls | one-sided binomial + Wilson CI |
| H2 | paraphrasing does not lower the correct-call rate | McNemar (paired in groups) |
| H3 | the two models differ in reliability | McNemar (correctness) + Wilcoxon (latency) |
| H4 | correctness decreases L1→L4 | logistic trend / Spearman |

## 7. Why direct SQL rather than the LIBAL API

Direct read-only SQL access lets the prototype (a) discover the true schema,
(b) compute deterministic ground truth for the evaluation, and (c) implement
aggregations (area per floor, comparisons) not reliably exposed by the API,
while keeping access read-only and parameterised throughout.

## 8. Limitations

- Single facility / dataset; the corpus is developer-authored.
- Floor-scoped queries see only space-placed components.
- Area for some component types (e.g. slabs) is unavailable in the dataset and is
  reported as such rather than fabricated.
- Equipment and maintenance questions are out of scope: no such tables were
  confidently identified, so those functions are intentionally not implemented.
- Results depend on a hosted LLM provider (Groq); provider-side model changes can
  affect exact reproducibility.
