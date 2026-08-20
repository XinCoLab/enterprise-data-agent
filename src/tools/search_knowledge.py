"""LLM tool for searching compact knowledge metadata."""

import json
from typing import Annotated

from langchain_core.tools import tool

from knowledge_runtime.catalog import search_knowledge_cards
from knowledge_runtime import current_knowledge


@tool("search_knowledge")
def search_knowledge(
    query: Annotated[
        str,
        (
            "A known business term, database object name, alias, or keyword. "
            "Use browse_knowledge instead when you do not yet know a useful term."
        ),
    ],
    knowledge_type: Annotated[
        str,
        (
            "Optional exact knowledge type used to narrow the results, such "
            "as a type returned in the root directory. Leave empty to search "
            "all types."
        ),
    ] = "",
) -> str:
    """Search knowledge metadata when a useful term is already known.

    This optional shortcut returns summaries and exact knowledge IDs. Use
    browse_knowledge from "/" when the relevant terminology is not known,
    and use read_knowledge to open selected results.
    """

    result = {
        "query": query,
        "results": search_knowledge_cards(
            cards=current_knowledge.KNOWLEDGE_CARDS,
            query=query,
            knowledge_type=knowledge_type,
        ),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)
