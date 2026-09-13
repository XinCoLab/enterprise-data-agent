import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from memory import mem0_client


def test_search_is_scoped_to_same_user_workspace_and_thread(monkeypatch):
    requests = []
    def search(**kwargs):
        requests.append(kwargs)
        return {"results": []}
    monkeypatch.setattr(mem0_client, "get_mem0_client", lambda: SimpleNamespace(search=search))
    mem0_client.search_memories("回访约定", user_id="u", workspace_id="w", thread_id="t")
    assert requests[0]["filters"] == {"user_id": "u", "run_id": "t", "workspace_id": "w"}


def test_disabled_process_has_no_memory_tools_prompt_or_injection():
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "DATA_AGENT_MEMORY_ENABLED": "0"}
    code = '''
import json
from tools.tool_registry import TOOLS
from prompts.prompt_loader import get_agent_prompt
from prompts.runtime_memory_context import build_memory_context_message
from memory.mem0_client import get_mem0_client
from agent_runtime.agent_runtime import MEMORY_ENABLED
assert not MEMORY_ENABLED
for name in ("add_memory", "update_memory", "delete_memory"):
    assert name not in [tool.name for tool in TOOLS]
    assert name not in get_agent_prompt().invoke({"messages": []}).to_messages()[0].content
assert build_memory_context_message([{"memory": "stale cached memory"}]) is None
try:
    get_mem0_client()
except RuntimeError:
    pass
else:
    raise AssertionError("Disabled process initialized Mem0")
print(json.dumps({"disabled": True}))
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["disabled"]


def test_add_uses_same_thread_scope_as_search(monkeypatch):
    from tools import add_memory as module
    requests = []
    client = SimpleNamespace(add=lambda **kwargs: requests.append(kwargs) or {
        "results": [{"id": "memory-1", "event": "ADD", "memory": "约定"}],
    })
    monkeypatch.setattr(mem0_client, "get_mem0_client", lambda: client)
    monkeypatch.setattr(module, "read_agent_run_context", lambda _config: SimpleNamespace(
        user_id="u", workspace_id="w", thread_id="t",
    ))
    module.add_memory.func("约定", {})
    assert requests[0]["run_id"] == "t"
    assert requests[0]["user_id"] == "u"
    assert requests[0]["metadata"]["workspace_id"] == "w"
