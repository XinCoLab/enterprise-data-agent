"""Main Agent LLM node for the current directory-based Agent."""

from functools import lru_cache

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from graph.graph_state import GraphState
from knowledge_runtime.catalog import browse_catalog
from knowledge_runtime import current_knowledge
from knowledge_runtime.knowledge_view import (
    build_subglobal_knowledge_graph,
    render_subglobal_knowledge_graph,
    select_knowledge_view,
)
from model_clients.llm_api_clients import (
    ALLOWED_MAIN_MODELS,
    clear_main_llm_cache,
    configured_main_model,
    get_main_llm,
)
from prompts.prompt_loader import build_model_input
from prompts.runtime_database_context import inject_runtime_database_context
from agent_runtime.agent_run_context import AgentRunContext, read_agent_run_context
from tools.tool_registry import TOOLS


DEFAULT_MAIN_MODEL = configured_main_model()
MAIN_LLM_WITH_TOOLS = get_main_llm(DEFAULT_MAIN_MODEL).bind_tools(TOOLS)


@lru_cache(maxsize=max(1, len(ALLOWED_MAIN_MODELS) - 1))
def _alternate_model_with_tools(model_name: str):
    return get_main_llm(model_name).bind_tools(TOOLS)


def _model_with_tools(model_name: str):
    # Keep the original default binding directly patchable for existing
    # runners/tests; only alternate model bindings are cached.
    if model_name == DEFAULT_MAIN_MODEL:
        return MAIN_LLM_WITH_TOOLS
    return _alternate_model_with_tools(model_name)


def refresh_model_runtime() -> None:
    """Replace cached clients after an explicit settings save."""

    global DEFAULT_MAIN_MODEL
    global MAIN_LLM_WITH_TOOLS

    clear_main_llm_cache()
    _alternate_model_with_tools.cache_clear()
    DEFAULT_MAIN_MODEL = configured_main_model()
    MAIN_LLM_WITH_TOOLS = get_main_llm(DEFAULT_MAIN_MODEL).bind_tools(TOOLS)


def selected_model_name(
    run_context: AgentRunContext | None,
    config: RunnableConfig | None,
) -> str:
    configurable = (config or {}).get("configurable", {})
    model_name = str(
        run_context.model
        if run_context is not None
        else configurable.get("model", DEFAULT_MAIN_MODEL)
    ).strip()
    if model_name not in ALLOWED_MAIN_MODELS:
        raise ValueError(f"Unsupported model: {model_name!r}")
    return model_name


def _invalid_tool_json_fallback(model_output: AIMessage) -> AIMessage | None:
    """Convert provider-parsed malformed Tool JSON into a terminal response."""

    invalid_calls = list(getattr(model_output, "invalid_tool_calls", []) or [])
    if not invalid_calls:
        return None
    return AIMessage(
        content=(
            "The model produced a malformed Tool call that could not be safely "
            "validated or executed. No Tool action was performed. Please retry "
            "the request."
        ),
        response_metadata={
            "termination_reason": "invalid_tool_json",
            "error_code": "INVALID_JSON",
            "graceful": True,
            "invalid_tool_call_count": len(invalid_calls),
        },
    )


async def main_agent_llm_node(
    state: GraphState,
    config: RunnableConfig | None = None,
):
    """为本轮模型组装上下文，然后让模型决定调用工具还是直接回答。"""

    run_context = read_agent_run_context(config)
    model_name = selected_model_name(run_context, config)

    # 根目录只告诉模型当前有哪些 Knowledge 类型及可浏览路径，相当于知识库首页。
    runtime_directory = browse_catalog(current_knowledge.KNOWLEDGE_CATALOG, "/")

    # 根据本轮已发生的 read/search/browse 行为选择知识视野：
    # GLOBAL=完整图，SUBGLOBAL=已读知识的一跳局部图，REGLOBAL=两者同时提供。
    knowledge_view_mode = select_knowledge_view(state)

    # 以本轮成功读取的 KnowledgeCard 为中心，从完整导航图中提取一跳邻居。
    # 即使当前选择 GLOBAL，这一步也很轻量；没有已读卡片时会得到空局部图。
    subglobal_graph = build_subglobal_knowledge_graph(
        state,
        current_knowledge.KNOWLEDGE_NAVIGATION_GRAPH,
    )

    # Python 图结构不能直接作为聊天消息发送，因此渲染成紧凑的文本索引。
    subglobal_graph_text = render_subglobal_knowledge_graph(subglobal_graph)

    # 组装：主 System Prompt + Knowledge 根目录 + 选中的导航图 + 对话历史。
    model_input = build_model_input(
        state["messages"],
        runtime_directory=runtime_directory,
        runtime_navigation_graph=current_knowledge.KNOWLEDGE_NAVIGATION_GRAPH_TEXT,
        knowledge_view_mode=knowledge_view_mode,
        runtime_subglobal_graph=subglobal_graph_text,
    )

    # 再补充当前数据库引擎、SQL 方言、数据库名和默认 schema，防止模型猜方言。
    model_input = inject_runtime_database_context(model_input)

    # 到这里才真正调用 LLM。模型返回一个 AIMessage：可能包含 Tool Call，
    # 也可能不调用工具、直接给出最终回答。
    model_output = await _model_with_tools(model_name).ainvoke(model_input)

    # Provider 有时会返回无法解析的 Tool JSON。此时不执行工具，改成安全的终止回答。
    invalid_json_fallback = _invalid_tool_json_fallback(model_output)
    if invalid_json_fallback is not None:
        return {"messages": [invalid_json_fallback]}

    # 下面只计算本轮 Knowledge 上下文的可观测信息，方便测试和排查；
    # 它不会改变模型刚才作出的回答或 Tool Call。
    global_graph_included = knowledge_view_mode in {"GLOBAL", "REGLOBAL"}
    subglobal_graph_included = knowledge_view_mode in {"SUBGLOBAL", "REGLOBAL"}
    navigation_context_chars = (
        len(current_knowledge.KNOWLEDGE_NAVIGATION_GRAPH_TEXT)
        if global_graph_included
        else 0
    ) + (len(subglobal_graph_text) if subglobal_graph_included else 0)
    knowledge_view_trace = {
        "knowledge_view_mode": knowledge_view_mode,
        "global_graph_included": global_graph_included,
        "subglobal_node_count": len(subglobal_graph["read_nodes"]),
        "frontier_node_count": len(subglobal_graph["frontier_nodes"]),
        "subglobal_knowledge_ids": [
            node["knowledge_id"] for node in subglobal_graph["read_nodes"]
        ],
        "navigation_context_chars": navigation_context_chars,
        "navigation_context_token_estimate": round(navigation_context_chars / 4),
    }
    model_output = model_output.model_copy(
        update={
            "response_metadata": {
                **(model_output.response_metadata or {}),
                "knowledge_view": knowledge_view_trace,
            }
        }
    )

    # 返回的是图状态增量。GraphState 的 messages reducer 会把这个
    # AIMessage 追加到已有对话中，Checkpointer 随后保存更新后的状态。
    return {"messages": [model_output]}
