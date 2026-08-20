# DataAgent

A small, deployable, read-only data analysis Agent built with LangGraph. It connects a natural-language question to approved business knowledge, guarded SQL execution, and an evidence-backed answer.

The stable Agent core is **Pure B0 + Safety**:

- one Main Agent LLM;
- complete ReAct history;
- `GLOBAL / SUBGLOBAL / REGLOBAL` dynamic knowledge navigation;
- deterministic Tool Safety before every Tool execution;
- no reviewer LLM, rolling summary, or pre-emptive fallback step.

The product shell adds a single Web entry for analysis, database profiles, Knowledge import, model selection, and compact run status. It does not change the Main Agent prompt or navigation policy.

## Quick start with Docker

Prerequisite: Docker Desktop.

```powershell
docker compose up --build -d
```

Open [http://localhost:8080](http://localhost:8080). On Windows, `start.cmd` runs the same command.

The first start creates ignored mutable files under `runtime/`. Open **Model settings** in the Web page, select DeepSeek V4 Pro or DeepSeek V4 Flash, enter the API key, and save it. The full key is never returned to the browser or committed to Git. Restart the service only when replacing a key after the Agent has already handled requests.

Useful commands:

```powershell
docker compose logs -f
docker compose down
```

The service is bound to `127.0.0.1:8080` by default rather than exposed to the local network.

### Local Python development

Use a Python environment with `requirements.txt` installed, then run:

```powershell
python -m uvicorn config_ui_server:app --app-dir src --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). The prebuilt static frontend is included, so Node.js is not required for normal local use.

## Product workflow

### Analysis

Ask a question, keep a multi-turn conversation through `thread_id`, and switch each request between:

- `deepseek-v4-pro`
- `deepseek-v4-flash`

The answer view can also show the actual SQL, latest SQL result preview, elapsed time, SQL count, and Knowledge view used. Reaching the graph step budget is reported by the Web boundary without consuming another LLM call.

### Data source

The configuration page supports:

- PostgreSQL
- MySQL
- DuckDB

It tests the connection with a read-only transaction and stores passwords only in ignored local runtime files. When DataAgent runs in Docker and the database runs on the host machine, use `host.docker.internal` instead of `127.0.0.1`.

Database and Knowledge changes take effect after a service restart. This keeps the proven Agent Runtime import path unchanged rather than introducing a second hot-reload system.

### Model settings

Select DeepSeek V4 Pro or DeepSeek V4 Flash and save the API key directly in the Web page. The backend stores the key only in the ignored local secrets file; the API exposes only whether a key is configured.

### Knowledge

Knowledge can be selected from a local directory or uploaded as a ZIP. ZIP imports are bounded by file count and size, reject path traversal and symbolic links, and must pass the existing KnowledgeCard loader before they can be applied.

The bundled example package is derived from the public LiveSQLBench Base-Full v1 `cold_chain_pharma_compliance` materials. It is benchmark data, not a private enterprise configuration.

### Runs

The Web service keeps only the latest 50 compact in-process run records: status, model, latency, Tool counts, SQL count, and thread ID. It does not persist questions or answers in this list.

## Agent architecture

```text
User question
    -> Main Agent LLM
    -> Tool Safety
    -> Tool Execution
    -> Main Agent LLM
    -> ...
    -> Final answer when the model stops calling tools
```

The Main Agent receives a domain-neutral system prompt, the conversation, a lightweight Knowledge directory, and one navigation view:

- `GLOBAL`: the complete lightweight relationship graph used to locate an entry point;
- `SUBGLOBAL`: read cards plus their explicit unread neighbors;
- `REGLOBAL`: a global re-entry when the local path no longer provides a useful lead.

The navigation graph is an index. The model must still open a KnowledgeCard before treating its payload as evidence.

## LLM-facing tools

| Tool | Purpose |
| --- | --- |
| `browse_knowledge` | Browse the virtual Knowledge directory. |
| `search_knowledge` | Search compact card metadata. |
| `read_knowledge` | Open one or more exact KnowledgeCards. |
| `execute_readonly_sql` | Execute one approved read-only SQL statement. |

All calls cross a deterministic safety boundary. It validates the Tool name, JSON arguments, schema, Knowledge path and batch limits, SQL read-only policy, statement count, external-access patterns, timeout, and result limits. Safety adds no LLM call and does not alter the Main Agent prompt.

## Knowledge package contract

A replaceable package must provide valid YAML KnowledgeCards with unique `knowledge_id` values and the required fields:

```text
knowledge_id
knowledge_type
title
summary
payload
```

Explicit relationship references are used to build the dynamic navigation graph. The loader does not depend on fixed domain folder names. A reusable builder skill is included under `skills/enterprise-knowledge-builder/`.

## Evaluation

Natural Benchmark v2 was built and audited from the public LiveSQLBench Base-Full v1 cold-chain database, Schema, Column Meaning, and business-knowledge materials. Every required business semantic can be independently recovered from the public source material; the benchmark does not depend on Gold SQL or answer leakage.

The system comparison fixed the model, temperature, database, questions, concurrency, scorer, and recursion limit. Each system ran 30 questions three times, for 90 executions:

| Metric | Generic LangChain SQL Agent baseline | Pure B0 |
| --- | ---: | ---: |
| Presentation-normalized Core accuracy | 74.4% | **85.6%** |
| Hard-case Core accuracy | 40.0% | **63.3%** |
| Non-convergence/runtime rate | 22.2% | **8.9%** |
| P50 latency | 74.4 s | **39.6 s** |
| P99 latency | 422.2 s | **199.8 s** |
| Tokens per correct case | 124,040 | **100,150** |

These are repeated hosted-model measurements, not deterministic guarantees. Core Result, Output Contract, and Strict scoring are separated so presentation differences are not confused with business-answer correctness.

## Local development

Create machine-local configuration from the templates:

```powershell
Copy-Item config/settings.env.example config/settings.env
Copy-Item config/secrets.env.example config/secrets.env
```

Product users need only Docker and the Web page. For direct Python development, start the same product API with:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m uvicorn config_ui_server:app --app-dir src --host 127.0.0.1 --port 8080
```

LangGraph Studio remains optional for graph debugging through `langgraph.json`. Use `langgraph dev --no-reload`; runtime checkpoint files change continuously and the installed CLI cannot exclude them from its watcher.

Python tests:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m pytest -q src/tests
```

Frontend tests:

```powershell
cd frontend
npm test
```

## Repository layout

```text
src/          Agent Runtime, product API, prompts, Tools, Safety, and tests
config/       public templates and example profiles
knowledge/    replaceable public KnowledgeCard package
databases/    PostgreSQL, MySQL, and DuckDB read-only adapters
frontend/     compact React product interface
docker/       container entrypoint
runtime/      ignored mutable configuration, imports, and checkpoints
skills/       reusable Knowledge package builder skill
```

## Boundaries

- Read-only historical analysis only; no write SQL, DDL, or SQL-based external file/network access.
- Database and Knowledge configuration are deployment inputs, not hard-coded business logic.
- Hosted LLM behavior remains non-deterministic even at temperature `0`; use repeated evaluation and trajectory-level failure analysis.
- Credentials, checkpoints, imported Knowledge, local databases, logs, and caches are excluded from Git.
