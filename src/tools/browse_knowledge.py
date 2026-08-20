"""LLM tool for browsing one virtual knowledge directory."""

import json
from typing import Annotated

from langchain_core.tools import tool

from knowledge_runtime.catalog import browse_catalog
from knowledge_runtime import current_knowledge


@tool("browse_knowledge")
def browse_knowledge(
    directory_path: Annotated[
        str,
        (
            "The exact virtual knowledge directory path to open. Use '/' "
            "for the first call. For later calls, copy an exact directory "
            "path returned by a previous browse_knowledge result, such as "
            "'/table'. This is not a filesystem path or a knowledge_id."
        ),
    ] = "/",
) -> str:
    """List the contents of one virtual knowledge directory.

    On the first call, open "/". To go deeper, copy an exact directory path
    returned by this tool. Knowledge entries contain a title, summary, and
    knowledge_id; pass a selected ID to read_knowledge for full content.
    """

    try:
        result = browse_catalog(
            catalog=current_knowledge.KNOWLEDGE_CATALOG,
            path=directory_path,
        )
    except KeyError:
        result = {
            "error": "CATALOG_PATH_NOT_FOUND",
            "directory_path": directory_path,
            "message": (
                "The requested catalog path does not exist. "
                "Call browse_knowledge with directory_path='/' to see the "
                "available directories."
            ),
        }

    return json.dumps(result, ensure_ascii=False, indent=2)
