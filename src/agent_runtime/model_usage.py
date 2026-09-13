"""调用后读取模型服务返回的实际 token 用量，供展示使用，不执行输入限额检查。"""

from typing import Any


# https://api-docs.deepseek.com/quick_start/agent_integrations/crush/
MODEL_CONTEXT_WINDOWS = {
    "deepseek-v4-pro": 1_048_576,
    "deepseek-v4-flash": 1_048_576,
}


def context_usage_for_message(
    message: Any,
    *,
    model_name: str = "",
) -> dict:
    """Return the latest call's input size, never a sum or a text estimate."""

    metadata = getattr(message, "response_metadata", None) or {}
    tagged_usage = metadata.get("context_usage") or {}
    model = str(
        model_name
        or tagged_usage.get("model")
        or metadata.get("model_name")
        or metadata.get("model")
        or ""
    )
    usage = getattr(message, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens")
    if type(input_tokens) is not int or input_tokens < 0:
        token_usage = metadata.get("token_usage") or {}
        input_tokens = token_usage.get("prompt_tokens")
    if type(input_tokens) is not int or input_tokens < 0:
        input_tokens = None

    return {
        "input_tokens": input_tokens,
        "context_window": MODEL_CONTEXT_WINDOWS.get(model),
        "model": model,
        "source": "provider" if input_tokens is not None else "unavailable",
    }
