# Function Calling for Supporting LLMs in Question Answering — FM/BIM Prototype

Master-thesis prototype: **"Function Calling for Supporting Large Language
Models in Question Answering."** It enables facility managers to retrieve
structured Facility Management / BIM information from a LIBAL PostgreSQL ETL
database using natural language — **safely**, via function calling rather than
free-form text-to-SQL.

> **Status:** complete — database discovery, data dictionary, controlled
> function-calling core, two-model LLM layer, chatbot + evaluation-dashboard UIs,
> and the full evaluation pipeline (corpus, ground truth, metrics, H1–H4
> statistics, reports). 84 tests; ruff/black/mypy clean.

## For the examiner (start here)

You can evaluate this project **without any credentials**: all results are
pre-generated and committed.

- **See the results now** (no setup): open
  `data/evaluation/runs/<timestamp>/` — `summary.md`, `hypothesis_results.md`
  (H1–H4 verdicts), `model_comparison.md`, `error_analysis.md`, and `plots/*.png`.
  The discovered schema is in `data/schema_reports/` (`schema_summary.md`,
  `data_dictionary.md`, `confirmed_findings.md`).
- **Read the architecture**: `docs/architecture.md` (reusable thesis text).
- **Run the offline parts** (no DB/LLM needed): `pip install -r requirements.txt`
  then `pytest` (84 tests; DB/LLM-backed tests skip automatically), and
  `python -m app.evaluation.report` to regenerate the reports from the committed
  run data.
- **Run the live demo** (DB + LLM): requires a `.env` with PostgreSQL
  credentials and a Groq API key. These are **not** in this repository; they are
  shared separately by the author. Copy `.env.example` to `.env`, fill them in,
  then use the chatbot / discovery / runner commands below.

Two models were compared: **qwen/qwen3-32b** and **llama-3.1-8b-instant** (Groq).
The assistant answers via a **multi-step tool-calling loop** over atomic data
functions (no bespoke function per question). See the latest run under
`data/evaluation/runs/` for the current H1–H4 results and model comparison.

## Why function calling instead of text-to-SQL

The LLM never generates or executes arbitrary SQL. Instead it may only *select*
from a fixed set of predefined Python functions, each containing safe,
parameterised SQL. A registry validates the function name, parameters, types and
allowed values before anything runs, and every call is logged. This gives the
controlled, auditable retrieval the thesis argues for (Sub-RQ 1).

## Why direct PostgreSQL access instead of the LIBAL API

Direct read-only SQL access lets the prototype (a) discover the true schema,
(b) compute deterministic ground truth for evaluation, and (c) implement
aggregations (areas per floor, comparisons) that the API does not expose
reliably. Access is read-only and parameterised throughout.

## Project layout (current)

```
prototype/
  app/
    config.py            # env-based settings (Pydantic), secrets kept secret
    db.py                # read-only engine + execute_read_query guard
    schema_discovery.py  # Phase 3: inspect DB -> CSV + schema_summary.md
    data_dictionary.py   # Phase 4: map discovered schema -> data_dictionary.md
    tools/
      models.py          # Phase 5: Pydantic tool/call/log models
      registry.py        # Phase 5: validate + execute + log + classify errors
      normalization.py   # Phase 6: floors/types/attributes from live DB
      fm_functions.py    # Phase 7: 10 SQL-backed FM functions (the only callables)
    llm/
      base.py            # Phase 8: common LLM interface
      groq_client.py     # Phase 8: Groq native tool calling (2 models)
      json_tool_parser.py# Phase 8: strict JSON fallback parser
      prompt_templates.py# Phase 8: shared system + final-answer prompts
    chatbot/service.py   # Phase 9: question -> tool -> grounded answer pipeline
    ui/streamlit_app.py  # Phase 9: chat UI + function-call trace panel
    evaluation/
      corpus.py          # Phase 10: ~70 queries, paraphrase groups, L1-L4
      ground_truth.py    # Phase 11: deterministic expected results (SQL)
      metrics.py         # Phase 12: correctness / parameters / latency
      error_taxonomy.py  # Phase 13: 9 error categories
      runner.py          # Phase 14: run both models -> raw + aggregated outputs
      statistics.py      # Phase 15: H1-H4 hypothesis tests
      report.py          # Phase 16: thesis-ready reports + plots
  data/schema_reports/   # discovery reports (git-ignored by default)
  data/evaluation/       # corpus, ground truth, runs/<timestamp>/...
  data/logs/             # function_calls.jsonl audit log
  tests/                 # 84 tests (read-only, registry, normalization, FM, parser, eval)
  .env.example
  pyproject.toml
```

## Controlled function-calling architecture

```
user question
   │
   ▼
LLM (Groq, MODEL_A / MODEL_B)  ──selects──▶  ONE tool + arguments  (native tool calling)
   │                                              │
   │                                              ▼
   │                                   ToolRegistry.execute()
   │                          validate name / required / extra / type / enum
   │                                              │  (only approved calls run)
   │                                              ▼
   │                          fm_functions  ──parameterised read-only SQL──▶  PostgreSQL
   │                              (normalization resolves floors/types/attrs)
   │                                              │
   ▼                                              ▼
LLM phrases a grounded final answer  ◀──  structured ToolResult (+ source IDs)
   │
   ▼  every step logged to data/logs/function_calls.jsonl
final answer
```

The LLM never sees SQL or the database. It can only request a named, validated
function; the registry is the single execution gate.

## Setup

```bash
cd prototype
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS
pip install -e ".[dev]"
cp .env.example .env             # then edit .env with real credentials
```

### `.env` configuration

Fill in the database block in `.env` (see `.env.example`). A **read-only**
database role is strongly recommended. Credentials are loaded from the
environment only; nothing is hard-coded and the password is never logged.

| Variable | Meaning |
| --- | --- |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | PostgreSQL connection |
| `DB_SSL_MODE` | libpq sslmode (`prefer`, `require`, …) |
| `DB_CONNECT_TIMEOUT`, `DB_STATEMENT_TIMEOUT` | timeouts in seconds |
| `DEFAULT_FACILITY_ID`, `DEFAULT_PROJECT_ID` | facility/project scoping |
| `LLM_PROVIDER`, `GROQ_API_KEY` | LLM provider + key (Groq) |
| `MODEL_A`, `MODEL_B` | the two compared models (temperature is hard-coded to 0.0) |

## Run database discovery (Phase 3)

```bash
python -m app.db                 # quick connectivity check (prints OK/FAILED)
python -m app.schema_discovery   # generates all CSV reports + schema_summary.md
```

Outputs in `data/schema_reports/`:
`schemas.csv`, `tables.csv`, `columns.csv`, `primary_keys.csv`,
`foreign_keys.csv`, `indexes.csv`, `row_counts.csv`, `sample_values.csv`,
`relevant_candidates.csv`, `schema_summary.md`.

## Inspect the schema report (Phase 4)

```bash
python -m app.data_dictionary    # reads Phase 3 CSVs -> data_dictionary.md
```

Open `data/schema_reports/schema_summary.md` and `data_dictionary.md`. The data
dictionary lists candidate tables/columns per FM concept (facilities, floors,
components, attributes, geometry, …) with a confidence level and a "manual
confirmation needed" flag. **Confirm the uncertain items before Phase 7.**

## Run the chatbot (Phases 8–9)

Set `GROQ_API_KEY` in `.env` (free key from https://console.groq.com), then:

```bash
streamlit run app/ui/streamlit_app.py
```

Pick a model (MODEL_A = `llama-3.3-70b-versatile`, MODEL_B = `llama-3.1-8b-instant`)
in the sidebar and toggle **Show function-call trace** to see the selected
function, raw + normalized arguments, tool result and per-step latency. Try:
"What floors can I query?", "How many windows are on the second floor?",
"Total window area on the first floor?", "Which floor has the largest window area?".

## Run the evaluation (Phases 10–16)

The evaluation answers the research questions and tests the hypotheses.

```bash
# 1) (optional) inspect the corpus and ground truth
python -m app.evaluation.corpus          # ~70 queries -> data/evaluation/corpus.csv
python -m app.evaluation.ground_truth    # deterministic expected results (SQL, not LLM)

# 2) run both models over the full corpus (makes ~280 Groq calls; a few minutes)
python -m app.evaluation.runner          # add --limit N for a quick subset

# 3) statistics + thesis-ready reports for the latest run
python -m app.evaluation.statistics      # H1–H4 -> hypothesis_results.json
python -m app.evaluation.report          # summary/hypothesis/model_comparison/error + plots
```

Outputs land in `data/evaluation/runs/<timestamp>/`:
`raw_results.jsonl`, `metrics.csv`, `model_comparison.csv`, `error_taxonomy.csv`,
`summary.md`, `hypothesis_results.md`, `model_comparison.md`, `error_analysis.md`,
`plots/*.png`.

### Metrics measured

answer correctness (exact for counts, ±tolerance for areas, set recall for lists),
function-selection correctness, parameter accuracy (per-parameter + fully-correct
call), execution success, latency (tool-call / SQL / final-answer / total), and a
9-category error taxonomy.

### Visualise a run (interactive dashboard)

```bash
streamlit run app/ui/eval_dashboard.py
```

Pick a run in the sidebar to browse: headline metrics, the H1–H4 verdict cards,
charts (correct-call rate, accuracy by complexity, error categories), and a
**per-query explorer** that shows the ground truth next to each model's actual
function call, arguments and final answer — filterable by model / category /
complexity / failures, with a side-by-side drill-down per query. Read-only;
loads the artifacts under `data/evaluation/runs/<timestamp>/`.

### Interpreting the results

- **H1** — fully-correct-call rate ≥ 90% (binomial test + Wilson CI).
- **H2** — paraphrases do not lower the correct-call rate (McNemar, paired in groups).
- **H3** — the two models differ in reliability (McNemar) and latency (Wilcoxon).
- **H4** — answer correctness decreases L1→L4 (logistic trend / Spearman).

Each hypothesis is reported as **supported / not supported** with the test, p-value,
plain-language interpretation and limitations in `hypothesis_results.md`.

### Reproducing the experiment

Deterministic decoding (`LLM_TEMPERATURE=0.0`), identical prompt/tools for both
models, ground truth computed from SQL (independent of the LLM), and a fixed,
versioned corpus make runs reproducible up to provider-side model changes.

## Tests & code quality

```bash
pytest            # 84 tests; DB/LLM-backed ones skip automatically if offline
ruff check .
black --check .
mypy app
```

DB-backed tests skip when the database is unreachable (see `tests/conftest.py`).

## Security model

- No hard-coded credentials, IDs or secrets — everything via `.env`.
- Read-only DB role preferred; sessions set `TRANSACTION READ ONLY`.
- `execute_read_query` rejects INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE
  and stacked statements.
- Parameterised SQL only; user input is never concatenated into SQL.
- Connect + statement timeouts on every connection.
- Row sampling is limited; sensitive-looking columns are redacted in reports.

## Limitations (current phase)

The prototype currently only **discovers and documents** the database. It does
not yet answer questions. The FM functions, LLM integration, chatbot and
evaluation are intentionally deferred until the schema is confirmed.
