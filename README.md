# Function Calling for Supporting LLMs in Question Answering — FM/BIM Prototype

Master-thesis prototype: **"Function Calling for Supporting Large Language Models in Question Answering."**
Facility managers can query structured FM/BIM data from a LIBAL PostgreSQL database using plain natural language — safely, via controlled function calling instead of text-to-SQL.

Two models are compared: **qwen/qwen3-32b** and **llama-3.1-8b-instant** (both via Groq).
Both were evaluated against 78 queries across four complexity levels (L1–L4).

---

## What you need before starting

| Requirement | Where to get it |
|---|---|
| Python 3.11 or 3.12 | https://www.python.org |
| PostgreSQL credentials (read-only) | Shared separately |
| Groq API key (free) | https://console.groq.com |

---

## Step 1 — Check out and enter the project folder

```bash
cd prototype
```

All commands below must be run from inside the `prototype/` folder.

---

## Step 2 — Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt.

---

## Step 3 — Install dependencies

```bash
pip install -e ".[dev]"
```

This installs the project in editable mode along with all dev tools (pytest, ruff, black, mypy).

---

## Step 4 — Configure your `.env` file

```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in your values:

```dotenv
# --- Database ---
DB_HOST=<host>
DB_PORT=5432
DB_NAME=libalv2
DB_USER=<user>
DB_PASSWORD=<password>
DB_SSL_MODE=disable
DB_CONNECT_TIMEOUT=10
DB_STATEMENT_TIMEOUT=30

# --- Facility / project scoping ---
DEFAULT_FACILITY_ID=124851
DEFAULT_PROJECT_ID=114051

# --- LLM ---
GROQ_API_KEY=<your-groq-key>
MODEL_A=qwen/qwen3-32b
MODEL_B=llama-3.1-8b-instant
```

The password is read only from the environment — it is never logged or hard-coded anywhere.

---

## Step 5 — Verify the database connection

```bash
python -m app.core.db
```

Expected output: `DB connection OK` followed by a row count. If it prints `FAILED`, check your `.env` host/port/credentials and whether VPN access is required.

---

## Step 6 — Start the chatbot

```bash
streamlit run app/chatbot/ui.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

**What you can do in the UI:**

- Use the **sidebar** to pick a model (`MODEL_A` = qwen/qwen3-32b, `MODEL_B` = llama-3.1-8b-instant).
- Toggle **"Show function-call trace"** to see which FM function was selected, what arguments were passed, how they were normalized, and per-step latency.
- Type a question in the chat box and press Enter.

**The 8 callable FM functions:**

| Function | What it does |
|---|---|
| `get_database_capabilities` | Lists queryable floors, component types, and supported question kinds |
| `list_queryable_floors` | Returns all floors/storeys for the facility |
| `list_queryable_component_types` | Returns all component/IFC types with counts |
| `count_components` | Counts components of a type, optionally scoped to one floor |
| `get_component_attributes` | Returns height, width, area etc. per component |
| `calculate_total_component_area` | Total area for a component type (optionally per floor) |
| `calculate_area_by_floor` | Component area grouped by every floor |
| `calculate_floor_area` | Sum of room/space areas per floor (the floor's own area) |

**Example questions to try:**

```
What floors can I query?
How many windows are on the second floor?
What is the total window area on the first floor?
Which floor has the largest window area?
Are there more doors on floor 1 or floor 2?
```

Press `Ctrl+C` in the terminal to stop the server when done.

---

## Step 7 — Run the evaluation pipeline

The evaluation measures how accurately each model selects functions, provides parameters, and produces correct answers — across 78 queries at four complexity levels (L1–L4).

The ground truth is a committed reference file (`data/evaluation/ground_truth.json`) — no regeneration needed.

### 7a — Run both models against the ground truth

```bash
python -m app.evaluation.runner
```

This makes ~310 Groq API calls (78 queries × 2 models × up to ~2 tool-call steps per query). Expect a few minutes. Add `--limit N` to test on a smaller subset first:

```bash
python -m app.evaluation.runner --limit 10
```

Output lands in `data/evaluation/runs/<timestamp>/`: `raw_results.jsonl`, `metrics.csv`, `model_comparison.csv`, `error_taxonomy.csv`.

### 7b — Generate reports and plots

```bash
python -m app.evaluation.report
```

Writes into `data/evaluation/runs/<timestamp>/`:

| File | Contents |
|---|---|
| `summary.md` | Headline metrics table by model and complexity |
| `model_comparison.md` | Side-by-side model breakdown |
| `error_analysis.md` | 9-category error taxonomy |
| `plots/*.png` | Bar charts and comparison figures |

---

## Step 8 — Browse results in the interactive dashboard

```bash
streamlit run app/evaluation/dashboard.py
```

Open the printed URL (`http://localhost:8501`) in your browser.

**What the dashboard shows:**

- **Sidebar** — pick any saved run by timestamp.
- **Headline cards** — fully-correct-call rate, answer accuracy, execution success, mean latency.
- **Charts** — correct-call rate by model, accuracy by complexity level, error category breakdown.
- **Per-query explorer** — filter by model / category / complexity / failures; click any row for a side-by-side drill-down showing ground truth vs. each model's actual function call, arguments and final answer.

This is read-only — it loads the artifacts already generated in Step 7.

---

## Project layout

```
prototype/
  app/
    core/
      config.py             env-based settings (Pydantic); no secrets in code
      db.py                 read-only engine; rejects non-SELECT statements
    tools/
      models.py             Pydantic models for tool calls, results, logs
      registry.py           validates + executes + logs + error-classifies every call
      normalization.py      maps NL floor/type/attribute terms to real DB values
      fm_functions.py       8 SQL-backed FM functions — the only callable actions
    llm/
      base.py               common LLM interface
      groq_client.py        Groq native tool calling (MODEL_A / MODEL_B)
      json_tool_parser.py   strict JSON fallback parser
      prompt_templates.py   system + final-answer prompts
    chatbot/
      service.py            question → tool loop → grounded answer pipeline
      ui.py                 Streamlit chat UI + function-call trace panel
    evaluation/
      ground_truth.py       load the committed ground truth
      metrics.py            correctness / parameter / latency scoring
      error_taxonomy.py     9 error categories
      runner.py             runs both models → raw + aggregated outputs
      report.py             reports + plots
      dashboard.py          interactive results browser (read-only)
  data/
    schema_reports/         discovered schema + data dictionary
    evaluation/
      ground_truth.json     committed reference answers (78 queries, SQL-computed)
      runs/<timestamp>/     per-run artifacts (metrics, reports, plots)
    logs/
      function_calls.jsonl  per-call audit log
  tests/                    84 tests
  .env.example
  requirements.txt
  pyproject.toml
```

---


