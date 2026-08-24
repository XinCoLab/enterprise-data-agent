# DataAgent

DataAgent is a deployable, read-only data analysis agent built with LangGraph. It turns natural-language business questions into evidence-backed analysis by combining replaceable enterprise knowledge, live database queries, and deterministic tool safety.

It is designed for situations where knowing the database schema is not enough: the agent must also find the correct business definition, join path, grain, filter, and calculation rule before executing SQL.

## Product preview

> **Demo assets placeholder**
>
> - Product screenshot or short GIF: to be added
> - Example business question: to be added
> - Answer and SQL preview: to be added

## Highlights

- **Dynamic knowledge navigation** — locates a useful entry point in a large KnowledgeCard graph, follows explicit relationships around the evidence already opened, and returns to the global view only when the local path is insufficient.
- **Replaceable deployment inputs** — database, model, and Knowledge package can be configured from the Web interface without hard-coding a business domain into the Agent Runtime.
- **Guarded read-only execution** — every tool request crosses a deterministic policy layer that validates the tool name, arguments, Knowledge access, SQL shape, timeout, and result limits.
- **Evidence-aware answers** — the interface can display the executed SQL, result preview, elapsed time, tool activity, and Knowledge view used for the analysis.
- **Multi-turn analysis** — a conversation can refine requirements, change an earlier calculation rule, or return to a previous task through a stable `thread_id`.

## Quick start

### Docker

Prerequisite: Docker Desktop.

```powershell
docker compose up --build -d
```

Open [http://localhost:8080](http://localhost:8080). On Windows, `start.cmd` runs the same command.

The first start creates ignored, machine-local files under `runtime/`. Complete the initial setup in the Web interface:

1. Open **Model settings**, select DeepSeek V4 Pro or DeepSeek V4 Flash, and enter the API key.
2. Open **Data source**, configure and test a PostgreSQL, MySQL, or DuckDB connection.
3. Open **Knowledge**, select a valid local KnowledgeCard directory or upload a ZIP package.
4. Return to **Analysis** and ask a business question.

Model, database, and Knowledge changes take effect after they pass validation and are applied. Secrets are stored only in ignored local runtime files and are never returned in full to the browser.

Useful commands:

```powershell
docker compose logs -f
docker compose down
```

The service binds to `127.0.0.1:8080` by default instead of exposing itself to the local network.

### Local Python development

Install `requirements.txt` in a Python environment, then run:

```powershell
python -m uvicorn api.app:app --app-dir src --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). The prebuilt frontend is included, so Node.js is not required for normal local use.

### Code reading order

To follow one Agent request from its real entrypoint to the database, read these files in order:

```text
src/api/app.py
  -> src/api/routers/chat.py
  -> src/runtime/agent_runtime.py
  -> src/graph/round_graph.py
  -> src/graph/nodes/
  -> src/tools/
```

`src/api/configuration_app.py` is the separate configuration and Knowledge-import subsystem. Benchmark runners live in `benchmarks/`, developer CLIs in `scripts/`, and tests in `tests/`; none of them are part of the main request path above.

## How it works

```text
User question
    |
    v
Web / API boundary
    |
    v
Main Agent LLM <-------------------------------+
    |                                          |
    +--> Dynamic Knowledge view                |
    |      GLOBAL -> SUBGLOBAL -> REGLOBAL     |
    |                                          |
    +--> Knowledge tools ----------------------+
    |                                          |
    +--> Tool Safety -> Read-only SQL -> DB ----+
    |
    v
Answer + SQL + evidence summary
```

The Main Agent receives a domain-neutral system prompt, the conversation, a lightweight Knowledge directory, and one navigation view:

- `GLOBAL` presents the complete lightweight relationship graph for locating an entry point.
- `SUBGLOBAL` presents opened cards and their explicit unread neighbors for focused exploration.
- `REGLOBAL` restores the global view when the current local path no longer provides a useful lead.

The navigation graph is an index, not evidence. The model must open a KnowledgeCard before using its payload as an approved business or database fact.

## Tools and safety

| Tool | Purpose |
| --- | --- |
| `browse_knowledge` | Browse the virtual Knowledge directory. |
| `search_knowledge` | Search compact card metadata. |
| `read_knowledge` | Open one or more exact KnowledgeCards. |
| `execute_readonly_sql` | Execute one approved read-only SQL statement. |

Before execution, the safety layer validates:

- tool identity and JSON arguments;
- argument schema and batch limits;
- Knowledge paths and identifiers;
- SQL read-only policy and statement count;
- external file or network access patterns;
- execution timeout and result-size limits.

## Configurable product workflow

### Analysis

The analysis page supports multi-turn questions and model selection per request. It can show the answer together with the actual SQL, latest result preview, elapsed time, SQL count, and Knowledge view used.

### Data source

The configuration page supports PostgreSQL, MySQL, and DuckDB. It tests the connection under read-only constraints and stores passwords only in ignored local runtime files.

When DataAgent runs in Docker and the database runs on the host machine, use `host.docker.internal` instead of `127.0.0.1`.

### Model settings

The current product interface supports DeepSeek V4 Pro and DeepSeek V4 Flash. The backend exposes only whether a key is configured; it never returns the stored key.

### Knowledge

Knowledge can be selected from a local directory or uploaded as a ZIP. Imports are bounded by file count and size, reject path traversal and symbolic links, and must pass the KnowledgeCard loader before activation.

The bundled example package is derived from the public LiveSQLBench Base-Full v1 `cold_chain_pharma_compliance` materials. It is public benchmark data, not a private enterprise configuration. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for source attribution and licensing.

### Runs

The service keeps the latest 50 compact in-process run records, including status, model, latency, tool counts, SQL count, and thread ID. Questions and answers are not persisted in this list.

## Knowledge package contract

A replaceable package contains valid YAML KnowledgeCards with unique `knowledge_id` values and these required fields:

```text
knowledge_id
knowledge_type
title
summary
payload
```

Explicit relationship references build the navigation graph. The loader does not depend on fixed business-domain folder names. A reusable builder skill is included under `skills/enterprise-knowledge-builder/` for producing and validating compatible packages.

## Evaluation

Natural Benchmark v2 was constructed and manually audited from the public LiveSQLBench Base-Full v1 cold-chain database, schema, column meanings, and business-knowledge materials. The business semantics needed by every question can be recovered independently from those public materials without relying on Gold SQL or leaked answers.

The comparison fixed the model, temperature, database, questions, concurrency, scorer, and graph step limit. Both systems ran the same 30 questions three times, for 90 executions each.

| Metric | Generic LangChain SQL Agent baseline | DataAgent |
| --- | ---: | ---: |
| Business-answer accuracy after normalizing presentation differences | 74.4% | **85.6%** |
| Business-answer accuracy on the hard subset | 40.0% | **63.3%** |
| Runs that did not reach a final answer | 22.2% | **8.9%** |
| Median latency (P50) | 74.4 s | **39.6 s** |
| Tail latency (P99) | 422.2 s | **199.8 s** |
| Total tokens per correct answer | 124,040 | **100,150** |

These are repeated measurements of a hosted model, not deterministic guarantees. Evaluation separates:

- **Core Result** — whether the business result is correct;
- **Output Contract** — whether the requested columns, order, labels, and shape are respected;
- **Strict Result** — whether both the business result and output contract pass.

<!-- TODO: Add links to the public evaluation protocol, aggregate report, and reproducible result artifacts. -->

## Development

Create machine-local configuration from the templates when running without the Web setup flow:

```powershell
Copy-Item config/settings.env.example config/settings.env
Copy-Item config/secrets.env.example config/secrets.env
```

Run Python tests:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m pytest -q tests
```

Run frontend tests:

```powershell
cd frontend
npm test
```

LangGraph Studio is optional for inspecting the graph. Start it with:

```powershell
langgraph dev --no-reload
```

## Repository layout

```text
src/          Agent Runtime, product API, prompts, tools, safety, and tests
config/       public templates and example profiles
knowledge/    replaceable public KnowledgeCard package
databases/    PostgreSQL, MySQL, and DuckDB read-only adapters
frontend/     React product interface
docker/       container entrypoint
runtime/      ignored mutable configuration, imports, and checkpoints
skills/       reusable Knowledge package builder skill
```

## Scope and boundaries

- Read-only historical analysis; write SQL, DDL, and SQL-based external file or network access are rejected.
- Database and Knowledge configuration are deployment inputs rather than hard-coded business logic.
- Credentials, checkpoints, imported Knowledge, local databases, logs, and caches are excluded from Git.
- Hosted LLM behavior can vary even at temperature `0`; important evaluations should use repeated runs and case-level failure analysis.
