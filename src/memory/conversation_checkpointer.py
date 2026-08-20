"""LangGraph 会话记忆：使用 SQLite 持久化每个 thread_id 的图状态。"""

import atexit
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from config.project_paths import CHECKPOINT_DATABASE_PATH


CHECKPOINT_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

MEMORY_CONNECTION = sqlite3.connect(
    CHECKPOINT_DATABASE_PATH,
    check_same_thread=False,
)
CHECKPOINTER = SqliteSaver(MEMORY_CONNECTION)

atexit.register(MEMORY_CONNECTION.close)
