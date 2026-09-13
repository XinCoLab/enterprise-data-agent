"""手动运行的真实 OSS 冒烟测试；pytest 导入此文件不会发起模型请求。"""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-only", action="store_true", help="新进程读取上次测试，验证持久化；不再次写入或调用提取模型")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    store = root / "runtime/mem0-smoke"
    os.environ["DATA_AGENT_MEMORY_DIR"] = str(store)
    os.environ["DATA_AGENT_MEMORY_ENABLED"] = "1"
    os.environ["MEM0_TELEMETRY"] = "false"
    sys.path[:0] = [str(root / "src"), str(root)]
    logging.basicConfig(level=logging.INFO)
    from memory.mem0_client import close_memory_client, get_mem0_client, memory_filters, save_memory, search_memories

    state_path = store / "last_run.json"
    if args.read_only:
        scope = json.loads(state_path.read_text(encoding="utf-8"))["scope"]
    else:
        scope = {"user_id": "oss-smoke-user", "workspace_id": "oss-smoke-workspace", "thread_id": "oss-smoke-" + uuid4().hex}
    started = perf_counter()
    report = {"mode": "read_only" if args.read_only else "add_and_search", "scope": scope}
    try:
        if not args.read_only:
            assert search_memories("粉丝回访约定", **scope) == [], "测试会话必须从空记忆开始"
            report["add_result"] = save_memory(
                "请记住本会话的虚拟偶像粉丝回访约定：只选等级 tier_step 不低于 12 的粉丝，候选最多 7 人，以后默认使用中文表格。",
                **scope,
            )
            assert report["add_result"]["status"] == "SUCCEEDED", report["add_result"]
        report["stored"] = get_mem0_client().get_all(filters=memory_filters(**scope))["results"]
        report["recalled"] = search_memories("粉丝回访的最低等级和候选人数是怎么约定的？", **scope)
        text = "\n".join(item["memory"] for item in report["recalled"])
        assert report["stored"] and "12" in text and "7" in text, report["recalled"]
        for field in ("user_id", "workspace_id", "thread_id"):
            other = {**scope, field: scope[field] + "-other"}
            assert search_memories("粉丝回访约定", **other) == [], f"记忆跨 {field} 泄漏"
        report["isolation"] = "passed: user, workspace, thread"
        report["status"] = "passed"
    finally:
        report["seconds"] = round(perf_counter() - started, 2)
        store.mkdir(parents=True, exist_ok=True)
        output = store / ("restart_check.json" if args.read_only else "last_run.json")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        close_memory_client()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
