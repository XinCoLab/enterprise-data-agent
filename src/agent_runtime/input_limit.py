"""调用前估算输入 token 并提供本地输入限额，用于小窗口实验和压缩判断。"""

import math

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import count_tokens_approximately


# 本地输入预算：超过 200K 开始尝试压缩，额外留出 16,384 tokens 的缓冲。
# 不修改服务端窗口，也不设置主模型的输出长度上限。
RESERVE_TOKENS = 16_384
MAX_INPUT_TOKENS = 200_000 + RESERVE_TOKENS
INPUT_TOKEN_ESTIMATE_METHOD = "langchain_approx_plus_reasoning"


def estimate_input_tokens(*, messages: list[BaseMessage], tool_definitions: list[dict]) -> int:
    """直接粗估消息和工具定义；不构造请求体，不调用模型。"""

    tokens = count_tokens_approximately(
        messages, tools=tool_definitions, chars_per_token=4,
    )
    # LangChain 的粗估不包含 additional_kwargs 中的回传推理内容，单独补计。
    reasoning_chars = sum(
        len(message.additional_kwargs.get("reasoning_content") or "")
        for message in messages if isinstance(message, AIMessage)
    )
    return tokens + math.ceil(reasoning_chars / 4)


class InputBudgetExceeded(RuntimeError):
    """本地模拟超限，不代表模型服务端返回了上下文错误。"""

    code = "LOCAL_INPUT_BUDGET_EXCEEDED"

    def __init__(self, *, estimated_input_tokens: int, max_input_tokens: int):
        self.estimated_input_tokens = estimated_input_tokens
        self.max_input_tokens = max_input_tokens
        self.estimate_method = INPUT_TOKEN_ESTIMATE_METHOD
        super().__init__(
            f"{self.code}: estimated_input_tokens={estimated_input_tokens}, "
            f"max_input_tokens={max_input_tokens}, estimate_method={self.estimate_method}"
        )
