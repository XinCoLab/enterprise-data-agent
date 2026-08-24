"""Product server entrypoint.

Reading order:
    api/app.py -> api/routers/chat.py -> runtime/agent_runtime.py
    -> graph/round_graph.py -> graph/nodes -> tools
"""

from fastapi import FastAPI

from api.configuration_app import app as configuration_app
from api.routers.chat import router as chat_router


app = FastAPI(title="DataAgent", docs_url=None, redoc_url=None)
app.include_router(chat_router)

# Configuration endpoints and the compiled frontend are a separate subsystem.
# Mount it last so the explicit Agent routes above remain the visible main road.
app.mount("/", configuration_app)
