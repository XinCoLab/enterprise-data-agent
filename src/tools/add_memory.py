import json
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agent_runtime.agent_run_context import read_agent_run_context
from memory.mem0_client import save_memory

@tool("add_memory")
def add_memory(
    content: Annotated[
        str,
        "用户明确表达、以后仍有用的信息。写成完整表述，补全指代，保留原意。",
    ],
    config: RunnableConfig,
) -> str:
    """保存当前会话后续可复用的用户偏好、已确认约定或长期事实。

    用户明确要求记住，或信息对后续交流有持续价值时调用。
    不保存随口玩笑，也不把模型推测或未经用户确认的建议当作用户事实。
    返回 SUCCEEDED 表示已保存；NO_CHANGE 表示本次没有新增条目；FAILED 表示保存失败。
    """
    run_context = read_agent_run_context(config)
    if run_context is None:
        raise RuntimeError("缺少当前请求的用户信息。")

    result = save_memory(
        content,
        user_id=run_context.user_id,
        workspace_id=run_context.workspace_id,
        thread_id=run_context.thread_id,
    )
    return json.dumps(result, ensure_ascii=False)
