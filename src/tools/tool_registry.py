"""Tool registry for the current generic directory-based Data Agent."""

from tools.browse_knowledge import browse_knowledge
from tools.compose_dashboard import compose_dashboard
from tools.create_chart import create_chart
from tools.create_metric_cards import create_metric_cards
from tools.execute_readonly_sql import execute_readonly_sql
from tools.export_report import export_report
from tools.read_knowledge import read_knowledge
from tools.search_knowledge import search_knowledge


TOOLS = [
    browse_knowledge,
    search_knowledge,
    read_knowledge,
    execute_readonly_sql,
    create_metric_cards,
    create_chart,
    compose_dashboard,
    export_report,
]

TOOLS_BY_NAME = {registered_tool.name: registered_tool for registered_tool in TOOLS}
