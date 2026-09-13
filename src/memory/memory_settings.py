"""进程启动时固定记忆开关；A/B 使用独立进程，避免运行中互相切换。"""

import os

from config import project_paths as _project_paths  # 先加载本地环境配置。


_setting = os.getenv("DATA_AGENT_MEMORY_ENABLED", "1").strip()
if _setting not in {"0", "1"}:
    raise ValueError("DATA_AGENT_MEMORY_ENABLED must be 0 or 1.")
MEMORY_ENABLED = _setting == "1"
