import json
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agent_runtime.agent_run_context import read_agent_run_context
from memory.mem0_client import update_memory_record


@tool("update_memory")
def update_memory(
    memory_id: Annotated[str, "待修改记忆的真实 id，取自当前检索结果或本会话的记忆工具结果，不得编造。"],
    content: Annotated[str, "替换后的完整记忆。保留该条中仍有效的约定，仅修正用户明确变更的部分。"],
    config: RunnableConfig,
) -> str:
    """用户纠正已保存事实或明确变更长期约定时，替换当前会话的一条记忆。

    先找到对应记忆 ID，更新整条内容而非追加一条冲突记录。
    临时要求不修改长期约定。SUCCEEDED 才表示修改成功。
    NO_CHANGE 表示内容相同；NOT_FOUND 表示当前会话没有该条；FAILED 表示修改失败。
    """
    context = read_agent_run_context(config)
    if context is None:
        raise RuntimeError("缺少当前请求的用户信息。")
    return json.dumps(update_memory_record(
        memory_id, content, user_id=context.user_id,
        workspace_id=context.workspace_id, thread_id=context.thread_id,
    ), ensure_ascii=False)
