"""一次 Agent 请求在进入图之前固定下来的身份、资源和权限。

组织关系：

Workspace (workspace_id)
├── Members
│   └── User (user_id)
├── Resources
│   └── DataSource
│       ├── allowed_data_source_ids：该用户可以访问的数据源
│       └── selected_data_source_ids：本次请求实际选择的数据源
└── Conversations
    ├── Thread (thread_id)
    │   ├── Request / Agent Run (request_id)
    │   └── Request / Agent Run (request_id)
    └── Thread (thread_id)
        └── Request / Agent Run (request_id)

同一账号在同一 Workspace 中新建两个会话时，workspace_id 和 user_id 不变，
thread_id 随会话变化，request_id 随每次发送消息变化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentRunContext:
    """保存上面一条 Request 从开始到结束都不能变化的信息。

    创建位置：agent_runtime.build_agent_config()。
    传递方式：存入 RunnableConfig["configurable"]["agent_run_context"]，
    LangGraph 再把同一份 config 传给每个节点和工具。

    它不是 GraphState：GraphState 保存会被节点持续更新的消息；这里保存的是
    本次请求固定使用的用户、工作空间、模型、权限和数据源。
    """

    request_id: str  # 本次“点击发送 → 收到最终结果”的唯一编号
    thread_id: str  # 当前会话编号；同一会话的多次请求共用它
    workspace_id: str  # 当前请求所属的工作空间，即资源隔离范围
    user_id: str  # 发起本次请求的用户编号
    model: str  # 本次请求固定使用的模型
    permissions: frozenset[str]  # 用户在本次请求中拥有的权限
    allowed_data_source_ids: tuple[str, ...]  # 用户有权访问的数据源
    selected_data_source_ids: tuple[str, ...]  # 本次请求实际选择的数据源


def read_agent_run_context(
    config: Mapping[str, Any] | None,
) -> AgentRunContext | None:
    """从 LangGraph config 中取回本次请求的 AgentRunContext。

    正式的聊天请求一定会创建它；返回 None 只用于兼容仍在直接调用图的旧代码
    和测试代码。
    """

    # configurable 是 LangGraph 专门存放自定义运行参数的位置。
    configurable = (config or {}).get("configurable", {})

    # 这个对象由 build_agent_config() 在请求入口创建并放入 config。
    context = configurable.get("agent_run_context")

    # 只有类型正确才返回，避免把同名字符串或普通字典当成运行上下文。
    return context if isinstance(context, AgentRunContext) else None
