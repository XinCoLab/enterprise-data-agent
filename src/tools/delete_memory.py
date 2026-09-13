import json
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agent_runtime.agent_run_context import read_agent_run_context
from memory.mem0_client import delete_memory_record


@tool("delete_memory")
def delete_memory(
    memory_id: Annotated[str, "用户要求删除的记忆真实 id，取自当前检索结果或本会话的记忆工具结果，不得编造。"],
    config: RunnableConfig,
) -> str:
    """用户明确要求忘记、删除或撤销已保存的约定时，删除当前会话的一条记忆。

    仅删除用户指定的条目，不因单次例外或模型自行判断而删除其他长期约定。
    该操作删除记忆库条目，不删除聊天记录。SUCCEEDED 才表示删除成功；
    NOT_FOUND 表示当前会话没有该条；FAILED 表示删除失败。
    """
    context = read_agent_run_context(config)
    if context is None:
        raise RuntimeError("缺少当前请求的用户信息。")
    return json.dumps(delete_memory_record(
        memory_id, user_id=context.user_id,
        workspace_id=context.workspace_id, thread_id=context.thread_id,
    ), ensure_ascii=False)
