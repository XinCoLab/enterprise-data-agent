"""DataAgent Web 服务入口。

核心阅读顺序：
    api/app.py -> api/routers/chat.py -> runtime/agent_runtime.py
    -> graph/round_graph.py -> graph/nodes -> tools
"""

from fastapi import FastAPI

from api.configuration_app import app as configuration_app
from api.routers.accounts import router as accounts_router
from api.routers.artifacts import router as artifacts_router
from api.routers.chat import router as chat_router
from api.routers.conversations import router as conversations_router
from memory.conversation_history_database import create_chat_history_tables


# 创建 FastAPI 应用。这里只是在组装服务，还没有处理任何用户请求。
app = FastAPI(title="DataAgent", docs_url=None, redoc_url=None)
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
