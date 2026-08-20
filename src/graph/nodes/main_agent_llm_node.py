"""Main Agent LLM node for the current directory-based Agent."""

from functools import lru_cache

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from graph.text2sql_state import Text2SQLState
from knowledge_runtime.catalog import browse_catalog
from knowledge_runtime.current_knowledge import (
    KNOWLEDGE_CATALOG,
    KNOWLEDGE_NAVIGATION_GRAPH,
    KNOWLEDGE_NAVIGATION_GRAPH_TEXT,
)
from knowledge_runtime.knowledge_view import (
    build_subglobal_knowledge_graph,
    render_subglobal_knowledge_graph,
    select_knowledge_view,
)
from model_clients.llm_api_clients import (
    ALLOWED_MAIN_MODELS,
    DEFAULT_MAIN_MODEL,
    MAIN_LLM,
    get_main_llm,
)
from prompts.prompt_loader import build_model_input
from tools.tool_registry import TOOLS


MAIN_LLM_WITH_TOOLS = MAIN_LLM.bind_tools(TOOLS)


@lru_cache(maxsize=max(1, len(ALLOWED_MAIN_MODELS) - 1))
def _alternate_model_with_tools(model_name: str):
    return get_main_llm(model_name).bind_tools(TOOLS)


def _model_with_tools(model_name: str):
    # Keep the original default binding directly patchable for existing
    # runners/tests; only alternate model bindings are cached.
    if model_name == DEFAULT_MAIN_MODEL:
        return MAIN_LLM_WITH_TOOLS
    return _alternate_model_with_tools(model_name)


def _requested_model(config: RunnableConfig | None) -> str:
    configurable = (config or {}).get("configurable", {})
    model_name = str(configurable.get("model", DEFAULT_MAIN_MODEL)).strip()
    if model_name not in ALLOWED_MAIN_MODELS:
        raise ValueError(f"Unsupported model: {model_name!r}")
    return model_name


def _invalid_tool_json_fallback(reply: AIMessage) -> AIMessage | None:
    """Convert provider-parsed malformed Tool JSON into a terminal response."""

    invalid_calls = list(getattr(reply, "invalid_tool_calls", []) or [])
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


def main_agent_llm_node(state: Text2SQLState, config: RunnableConfig | None = None):
    """Inject the selected Knowledge view, then request tools or answer."""

    runtime_directory = browse_catalog(KNOWLEDGE_CATALOG, "/")
    knowledge_view_mode = select_knowledge_view(state)
    subglobal_graph = build_subglobal_knowledge_graph(
        state,
        KNOWLEDGE_NAVIGATION_GRAPH,
    )
    subglobal_graph_text = render_subglobal_knowledge_graph(subglobal_graph)
    model_input = build_model_input(
        state["messages"],
        runtime_directory=runtime_directory,
        runtime_navigation_graph=KNOWLEDGE_NAVIGATION_GRAPH_TEXT,
        knowledge_view_mode=knowledge_view_mode,
        runtime_subglobal_graph=subglobal_graph_text,
    )
    reply = _model_with_tools(_requested_model(config)).invoke(model_input)
    invalid_json_fallback = _invalid_tool_json_fallback(reply)
    if invalid_json_fallback is not None:
        return {"messages": [invalid_json_fallback]}
    global_graph_included = knowledge_view_mode in {"GLOBAL", "REGLOBAL"}
    subglobal_graph_included = knowledge_view_mode in {"SUBGLOBAL", "REGLOBAL"}
    navigation_context_chars = (
        len(KNOWLEDGE_NAVIGATION_GRAPH_TEXT) if global_graph_included else 0
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
    reply = reply.model_copy(
        update={
            "response_metadata": {
                **(reply.response_metadata or {}),
                "knowledge_view": knowledge_view_trace,
            }
        }
    )
    return {"messages": [reply]}
