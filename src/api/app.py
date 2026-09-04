"""DataAgent Web 服务入口。

核心阅读顺序：
    frontend/app/page.tsx: sendQuestion()
    -> api/routers/chat.py: chat_stream()
    -> agent_runtime/agent_runtime.py: stream_agent()
    -> graph/data_agent_graph.py -> graph/nodes -> tools
    -> agent_runtime/agent_runtime.py: final event
    -> frontend/app/page.tsx: render
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.configuration_app import app as configuration_app
from api.routers.accounts import router as accounts_router
from api.routers.artifacts import router as artifacts_router
from api.routers.chat import router as chat_router
from api.routers.conversations import router as conversations_router
from graph.data_agent_graph import create_single_round_graph
from memory.conversation_checkpointer import (
    open_conversation_checkpoint_database,
)
from memory.conversation_history_database import create_chat_history_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    database_connection, checkpointer = (
        await open_conversation_checkpoint_database()
    )
    try:
        app.state.checkpointer = checkpointer
        app.state.single_round_graph = create_single_round_graph(checkpointer)
        yield
    finally:
        await database_connection.close()


# 创建 FastAPI 应用。这里只是在组装服务，还没有处理任何用户请求。
app = FastAPI(
    title="DataAgent",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


# 创建历史聊天表
create_chat_history_tables()
# 把 chat.py 中定义的 /api/chat、/api/chat/stream 等接口注册到应用。
# include_router() 只负责登记路由；收到对应 HTTP 请求后，路由函数才会执行。
app.include_router(chat_router)
app.include_router(artifacts_router)
app.include_router(conversations_router)
app.include_router(accounts_router)

# 配置接口和编译后的前端属于另一个子系统，最后挂载，避免遮住上面的 Agent 路由。
app.mount("/", configuration_app)
