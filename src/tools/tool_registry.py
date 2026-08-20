"""Tool registry for the current generic directory-based Data Agent."""

from tools.browse_knowledge import browse_knowledge
from tools.execute_readonly_sql import execute_readonly_sql
from tools.read_knowledge import read_knowledge
from tools.search_knowledge import search_knowledge


TOOLS = [
    browse_knowledge,
    search_knowledge,
    read_knowledge,
    execute_readonly_sql,
]

TOOLS_BY_NAME = {registered_tool.name: registered_tool for registered_tool in TOOLS}
