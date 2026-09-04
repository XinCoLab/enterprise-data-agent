"""LangGraph 会话记忆：使用 SQLite 持久化每个 thread_id 的图状态。"""

import aiosqlite

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config.project_paths import CHECKPOINT_DATABASE_PATH


async def open_conversation_checkpoint_database():
    """打开会话存档数据库，并创建 LangGraph checkpointer。"""

    CHECKPOINT_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    database_connection = await aiosqlite.connect(
        CHECKPOINT_DATABASE_PATH,
    )
    checkpointer = AsyncSqliteSaver(database_connection)
    await checkpointer.setup()
    return database_connection, checkpointer
