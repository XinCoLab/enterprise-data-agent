"""Prompt assembly for the current generic directory-based Agent."""

import json
from functools import lru_cache

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.project_paths import MAIN_PROMPT_PATH


@lru_cache(maxsize=1)
def get_agent_prompt() -> ChatPromptTemplate:
    """Load the current generic prompt once per application process."""

    system_text = MAIN_PROMPT_PATH.read_text(encoding="utf-8")
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_text),
            MessagesPlaceholder("messages"),
        ]
    )


def build_model_input(
    messages: list[AnyMessage],
    runtime_directory: dict,
    runtime_navigation_graph: str,
    knowledge_view_mode: str = "GLOBAL",
    runtime_subglobal_graph: str = "",
) -> list[AnyMessage]:
    """Combine the prompt, selected Knowledge view, and dialogue."""

    prompt_value = get_agent_prompt().invoke({"messages": messages})
    model_input = prompt_value.to_messages()

    directory_text = json.dumps(
        runtime_directory,
        ensure_ascii=False,
        indent=2,
    )
    model_input.insert(
        1,
        SystemMessage(
            content=(
                "[Runtime knowledge directory]\n"
                "This view was generated from the currently configured "
                "Knowledge Root. Use only paths returned here or by "
                "browse_knowledge.\n\n"
                f"{directory_text}"
            )
        ),
    )

    insertion_index = 2
    if knowledge_view_mode in {"GLOBAL", "REGLOBAL"}:
        model_input.insert(
            insertion_index,
            SystemMessage(
                content=(
                    "[Runtime knowledge navigation graph]\n"
                    "This complete graph was generated from explicit links in "
                    "the currently configured Knowledge Root. Every node lists "
                    "its exact knowledge_id once; edges use the short node refs. "
                    "The graph is an index, not full card content.\n\n"
                    f"{runtime_navigation_graph}"
                )
            ),
        )
        insertion_index += 1

    if knowledge_view_mode in {"SUBGLOBAL", "REGLOBAL"} and runtime_subglobal_graph:
        model_input.insert(
            insertion_index,
            SystemMessage(
                content=(
                    "[Runtime subglobal knowledge graph]\n"
                    "READ nodes were successfully opened in the current user "
                    "turn. FRONTIER nodes are unread one-hop neighbors connected "
                    "by explicit edges in the full graph. This is a navigation "
                    "view, not a replacement for KnowledgeCard content.\n\n"
                    f"{runtime_subglobal_graph}"
                )
            ),
        )
        insertion_index += 1

    return model_input
