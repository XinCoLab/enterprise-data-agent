"""Run the frozen ten-case LiveSQLBench slice with resumable JSON records."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median


RESULT_MARKER = "SMOKE_RESULT\n"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {record["instance_id"]: record for record in records}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["knowledge", "no_knowledge"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    selected_path = (
        root
        / "eval/livesqlbench_base_full_v1/curation/questions/selected_questions.json"
    )
    questions = json.loads(selected_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    records_path = output_dir / "per_case_results.json"
    by_id = _load_existing(records_path) if args.resume else {}

    environment = os.environ.copy()
    selected_case_ids = set(args.case_id)
    for question in questions:
        instance_id = question["instance_id"]
        if selected_case_ids and instance_id not in selected_case_ids:
            continue
        if instance_id in by_id and not selected_case_ids:
            print(f"SKIP {instance_id}", flush=True)
            continue

        print(f"RUN {instance_id}", flush=True)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("run_livesqlbench_smoke.py")),
                    "--instance-id",
                    instance_id,
                ],
                cwd=Path(__file__).resolve().parent,
                env=environment,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
        except subprocess.TimeoutExpired as error:
            record = {
                "instance_id": instance_id,
                "correct": False,
                "runner_exit_code": None,
                "runner_error": "CASE_TIMEOUT",
                "stdout": str(error.stdout or "")[-4000:],
                "stderr": str(error.stderr or "")[-4000:],
            }
            by_id[instance_id] = record
            ordered_records = [
                by_id[item["instance_id"]]
                for item in questions
                if item["instance_id"] in by_id
            ]
            _write_json(records_path, ordered_records)
            print(f"DONE {instance_id} correct=False timeout=True", flush=True)
            continue
        stdout = completed.stdout
        record = {
            "instance_id": instance_id,
            "correct": False,
            "runner_exit_code": completed.returncode,
        }
        if RESULT_MARKER in stdout:
            payload = stdout.split(RESULT_MARKER, 1)[1]
            try:
                record = json.loads(payload)
                record["runner_exit_code"] = completed.returncode
            except json.JSONDecodeError as error:
                record["runner_error"] = f"Result JSON parse error: {error}"
        else:
            record["runner_error"] = "SMOKE_RESULT marker missing"
        record["stderr"] = completed.stderr[-4000:]
        by_id[instance_id] = record
        ordered_records = [by_id[item["instance_id"]] for item in questions if item["instance_id"] in by_id]
        _write_json(records_path, ordered_records)
        print(
            f"DONE {instance_id} correct={record.get('correct')} "
            f"latency_ms={record.get('latency_ms')} tool_calls={record.get('tool_calls')}",
            flush=True,
        )

    records = [by_id[item["instance_id"]] for item in questions if item["instance_id"] in by_id]
    latencies = [float(record.get("latency_ms") or 0) for record in records]
    tokens = [int(record.get("total_tokens") or 0) for record in records]
    summary = {
        "variant": args.variant,
        "created_at": datetime.now().astimezone().isoformat(),
        "correct": sum(bool(record.get("correct")) for record in records),
        "total": len(records),
        "accuracy": (
            sum(bool(record.get("correct")) for record in records) / len(records)
            if records
            else 0.0
        ),
        "sql_runtime_success": sum(bool(record.get("generated_sql")) for record in records),
        "p50_latency_ms": median(latencies) if latencies else 0.0,
        "p99_latency_ms": _percentile(latencies, 0.99),
        "average_latency_ms": mean(latencies) if latencies else 0.0,
        "total_tokens": sum(tokens),
        "average_tokens": mean(tokens) if tokens else 0.0,
        "average_llm_calls": mean([record.get("llm_calls", 0) for record in records]) if records else 0.0,
        "average_tool_calls": mean([record.get("tool_calls", 0) for record in records]) if records else 0.0,
        "tool_errors": sum(len(record.get("tool_errors") or []) for record in records),
        "sql_errors": sum(
            sum(
                error.get("tool_name") == "execute_readonly_sql"
                for error in (record.get("tool_errors") or [])
            )
            for record in records
        ),
        "runtime_errors": sum(bool(record.get("runner_error")) for record in records),
        "failed_cases": [record["instance_id"] for record in records if not record.get("correct")],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
