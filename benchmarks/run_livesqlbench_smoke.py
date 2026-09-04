"""Run one public LiveSQLBench question, then score its captured SQL privately."""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import psycopg2
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_public_question(instance_id: str) -> dict:
    path = (
        _project_root()
        / "eval/livesqlbench_base_full_v1/curation/questions/selected_questions.json"
    )
    records = json.loads(path.read_text(encoding="utf-8"))
    return next(record for record in records if record["instance_id"] == instance_id)


def _load_public_conditions(instance_id: str) -> dict:
    path = (
        _project_root()
        / "eval/livesqlbench_base_full_v1/official_source/public/livesqlbench_data.jsonl"
    )
    with path.open(encoding="utf-8") as handle:
        return next(
            record.get("conditions") or {}
            for line in handle
            if (record := json.loads(line)).get("instance_id") == instance_id
        )


def _configure_runtime() -> None:
    root = _project_root()
    os.environ.setdefault("DATA_AGENT_DATABASE_BACKEND", "postgresql")
    os.environ.setdefault("DATA_AGENT_POSTGRES_HOST", "127.0.0.1")
    os.environ.setdefault("DATA_AGENT_POSTGRES_PORT", "5432")
    os.environ.setdefault("DATA_AGENT_POSTGRES_USER", "postgres")
    os.environ.setdefault("DATA_AGENT_POSTGRES_DATABASE", "cold_chain_pharma_compliance")
    os.environ.setdefault(
        "DATA_AGENT_KNOWLEDGE_ROOT",
        str(
            root
            / "eval/livesqlbench_base_full_v1/curation/knowledge/cold_chain_pharma_compliance"
        ),
    )


def _tool_result_succeeded(message: ToolMessage) -> bool:
    try:
        payload = json.loads(message.content)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and "columns" in payload and "rows" in payload


class UsageRecorder(BaseCallbackHandler):
    """Capture every real Main Agent LLM call in the experiment graph."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.total_tokens = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.llm_calls += 1

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            message = response.generations[0][0].message
            usage = message.usage_metadata or {}
            self.total_tokens += int(usage.get("total_tokens") or 0)
        except (AttributeError, IndexError, KeyError, TypeError):
            token_usage = (response.llm_output or {}).get("token_usage") or {}
            self.total_tokens += int(token_usage.get("total_tokens") or 0)


def _execute_full(sql: str):
    with psycopg2.connect(
        host=os.environ["DATA_AGENT_POSTGRES_HOST"],
        port=int(os.environ["DATA_AGENT_POSTGRES_PORT"]),
        user=os.environ["DATA_AGENT_POSTGRES_USER"],
        password=os.environ["DATA_AGENT_POSTGRES_PASSWORD"],
        dbname=os.environ["DATA_AGENT_POSTGRES_DATABASE"],
        connect_timeout=5,
    ) as connection:
        connection.set_session(readonly=True)
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = 10000")
            cursor.execute(sql)
            return cursor.fetchall()


def _normal(value):
    if isinstance(value, Decimal):
        return round(value, 2)
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _official_evaluation_helpers():
    """Load the official result normalization helpers without exposing GT to Agent."""

    source_dir = (
        _project_root()
        / "eval/livesqlbench_base_full_v1/official_source/"
        "evaluation_repository/evaluation/src"
    )
    module_path = source_dir / "test_utils.py"
    sys.path.insert(0, str(source_dir))
    spec = importlib.util.spec_from_file_location(
        "livesqlbench_official_test_utils",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official evaluator helpers: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows_match(actual: list, gold: list, ordered: bool) -> bool:
    """Compare rows using the benchmark's existing order semantics."""

    return actual == gold if ordered else set(actual) == set(gold)


def _results_match(actual: list, gold: list, ordered: bool) -> bool:
    """Allow omission only of constant textual label columns from Gold."""

    if _rows_match(actual, gold, ordered):
        return True
    if not actual or not gold or len(actual) != len(gold):
        return False

    actual_widths = {len(row) for row in actual}
    gold_widths = {len(row) for row in gold}
    if len(actual_widths) != 1 or len(gold_widths) != 1:
        return False
    actual_width = actual_widths.pop()
    gold_width = gold_widths.pop()
    if actual_width >= gold_width:
        return False

    droppable_label_columns = [
        column_index
        for column_index in range(gold_width)
        if all(isinstance(row[column_index], str) for row in gold)
        and len({row[column_index] for row in gold}) == 1
    ]
    missing_column_count = gold_width - actual_width
    if len(droppable_label_columns) < missing_column_count:
        return False

    for dropped_columns in combinations(
        droppable_label_columns,
        missing_column_count,
    ):
        dropped = set(dropped_columns)
        projected_gold = [
            tuple(
                value
                for column_index, value in enumerate(row)
                if column_index not in dropped
            )
            for row in gold
        ]
        if _rows_match(actual, projected_gold, ordered):
            return True
    return False


def _score(instance_id: str, generated_sql: str) -> tuple[bool, list, list]:
    private_path = (
        _project_root()
        / "eval/livesqlbench_base_full_v1/official_source/private/"
        "livesqlbench_base_full_v1_gt_kg_testcases_20260613.jsonl"
    )
    with private_path.open(encoding="utf-8") as handle:
        gold_record = next(
            record
            for line in handle
            if (record := json.loads(line)).get("instance_id") == instance_id
        )
    gold_sql = gold_record["sol_sql"][0]
    conditions = _load_public_conditions(instance_id)
    official = _official_evaluation_helpers()
    generated_clean = official.remove_round(
        official.remove_distinct(official.remove_comments([generated_sql]))
    )[0]
    gold_clean = official.remove_round(
        official.remove_distinct(official.remove_comments([gold_sql]))
    )[0]
    actual = official.preprocess_results(_execute_full(generated_clean))
    gold = official.preprocess_results(_execute_full(gold_clean))
    correct = _results_match(
        actual,
        gold,
        ordered=bool(conditions.get("order", False)),
    )
    return correct, actual, gold


async def main() -> None:
    # The agent may emit Unicode characters that the Windows GBK console cannot
    # encode. The parent batch runner always reads this machine payload as UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", default="cold_chain_pharma_compliance_2")
    args = parser.parse_args()

    _configure_runtime()
    if not os.environ.get("DATA_AGENT_POSTGRES_PASSWORD"):
        raise SystemExit("Set DATA_AGENT_POSTGRES_PASSWORD before running.")

    # Import only after runtime variables are set. Gold is not opened until
    # the Agent has completed and its final successful SQL has been captured.
    from graph.data_agent_graph import create_conversation_graph
    from memory.conversation_checkpointer import (
        open_conversation_checkpoint_database,
    )

    question = _load_public_question(args.instance_id)
    usage_recorder = UsageRecorder()
    config = {
        "configurable": {"thread_id": f"lsb-smoke-{uuid4()}"},
        "callbacks": [usage_recorder],
    }
    pending_sql: dict[str, str] = {}
    successful_sql: list[str] = []
    tool_sequence: list[str] = []
    tool_errors: list[dict] = []
    read_knowledge_calls: list[list[str]] = []
    knowledge_view_trace: list[dict] = []
    seen_tool_call_ids: set[str] = set()
    final_answer = ""

    database_connection, checkpointer = (
        await open_conversation_checkpoint_database()
    )
    conversation_graph = create_conversation_graph(checkpointer)
    started = perf_counter()
    try:
        async for update in conversation_graph.astream(
            {"messages": [HumanMessage(content=question["query"])]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_update in update.items():
                print(f"NODE {node_name}")
                for message in node_update.get("messages", []):
                    if isinstance(message, AIMessage):
                        if node_name == "Main Agent LLM":
                            view_trace = (message.response_metadata or {}).get(
                                "knowledge_view"
                            )
                            if isinstance(view_trace, dict):
                                knowledge_view_trace.append(view_trace)
                        if message.content and not message.tool_calls:
                            final_answer = str(message.content)
                        for call in message.tool_calls:
                            if call["id"] in seen_tool_call_ids:
                                continue
                            seen_tool_call_ids.add(call["id"])
                            tool_sequence.append(call["name"])
                            if call["name"] == "read_knowledge":
                                knowledge_ids = (call.get("args") or {}).get(
                                    "knowledge_ids",
                                    [],
                                )
                                read_knowledge_calls.append(
                                    [str(item) for item in knowledge_ids]
                                    if isinstance(knowledge_ids, list)
                                    else []
                                )
                            if call["name"] == "execute_readonly_sql":
                                pending_sql[call["id"]] = call["args"]["sql"]
                    elif isinstance(message, ToolMessage):
                        try:
                            tool_payload = json.loads(message.content)
                        except (TypeError, json.JSONDecodeError):
                            tool_payload = None
                        if (
                            isinstance(tool_payload, dict)
                            and tool_payload.get("status")
                            in {"error", "rejected", "denied"}
                        ):
                            tool_errors.append(
                                {
                                    "tool_name": message.name,
                                    **tool_payload,
                                }
                            )
                        if message.tool_call_id in pending_sql:
                            if _tool_result_succeeded(message):
                                successful_sql.append(
                                    pending_sql[message.tool_call_id]
                                )
    finally:
        await database_connection.close()

    latency_ms = round((perf_counter() - started) * 1000, 2)
    final_sql = successful_sql[-1] if successful_sql else ""
    correct = False
    actual = []
    gold = []
    if final_sql:
        correct, actual, gold = _score(args.instance_id, final_sql)

    result = {
        "instance_id": args.instance_id,
        "correct": correct,
        "generated_sql": final_sql,
        "actual_result": actual,
        "gold_result": gold,
        "latency_ms": latency_ms,
        "total_tokens": usage_recorder.total_tokens,
        "tool_calls": len(tool_sequence),
        "tool_counts": dict(Counter(tool_sequence)),
        "tool_sequence": tool_sequence,
        "read_knowledge_calls": read_knowledge_calls,
        "llm_calls": usage_recorder.llm_calls,
        "tool_errors": tool_errors,
        "knowledge_view_trace": knowledge_view_trace,
        "final_answer": final_answer,
    }
    print("SMOKE_RESULT")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    raise SystemExit(0 if correct else 1)


if __name__ == "__main__":
    asyncio.run(main())
