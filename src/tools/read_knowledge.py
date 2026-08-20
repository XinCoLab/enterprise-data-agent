"""LLM tool for opening exact knowledge cards."""

import json
from typing import Annotated

from langchain_core.tools import tool

from knowledge_runtime.catalog import read_knowledge_cards
from knowledge_runtime import current_knowledge


@tool("read_knowledge")
def read_knowledge(
    knowledge_ids: Annotated[
        list[str],
        (
            "One or more exact knowledge_id values copied from a previous "
            "browse_knowledge or search_knowledge result. Do not invent IDs."
        ),
    ],
) -> str:
    """Open full knowledge cards using exact IDs returned by browse or search.

    Copy knowledge_id values from earlier tool output. This tool does not
    discover IDs and must not be called with guessed identifiers.
    """

    try:
        result = {
            "cards": read_knowledge_cards(
                cards=current_knowledge.KNOWLEDGE_CARDS,
                knowledge_ids=knowledge_ids,
            )
        }
    except KeyError:
        result = {
            "error": "KNOWLEDGE_ID_NOT_FOUND",
            "knowledge_ids": knowledge_ids,
            "message": (
                "One or more knowledge IDs do not exist. Copy exact IDs from "
                "browse_knowledge or search_knowledge output."
            ),
        }

    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
