"""大模型 API 连接：只读取配置并创建客户端，不执行节点逻辑。"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from config.project_paths import BACKEND_ENV_PATH


load_dotenv(BACKEND_ENV_PATH)


ALLOWED_MAIN_MODELS = ("deepseek-v4-pro", "deepseek-v4-flash")
DEFAULT_MAIN_MODEL = os.getenv("DATA_AGENT_MODEL", "deepseek-v4-pro").strip()
if DEFAULT_MAIN_MODEL not in ALLOWED_MAIN_MODELS:
    raise RuntimeError(
        "DATA_AGENT_MODEL must be deepseek-v4-pro or deepseek-v4-flash."
    )


class DeepSeekThinkingChatModel(ChatDeepSeek):
    """Preserve DeepSeek thinking content across Tool-call sub-turns.

    DeepSeek requires an assistant Tool-call message's ``reasoning_content``
    to be sent back on later requests. ``ChatDeepSeek`` preserves that field on
    the LangChain ``AIMessage`` response, while this small payload adapter also
    copies it back into the next DeepSeek request.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(
            input_,
            stop=stop,
            **kwargs,
        )
        source_messages = self._convert_input(input_).to_messages()
        request_messages = payload.get("messages", [])
        for source, request_message in zip(source_messages, request_messages):
            if not isinstance(source, AIMessage):
                continue
            reasoning_content = source.additional_kwargs.get("reasoning_content")
            if reasoning_content is not None:
                request_message["reasoning_content"] = reasoning_content
            if source.tool_calls and request_message.get("content") is None:
                request_message["content"] = ""
        return payload


@lru_cache(maxsize=len(ALLOWED_MAIN_MODELS))
def get_main_llm(model_name: str = DEFAULT_MAIN_MODEL) -> DeepSeekThinkingChatModel:
    """Create one cached client per explicitly supported DeepSeek model."""

    if model_name not in ALLOWED_MAIN_MODELS:
        raise ValueError(f"Unsupported model: {model_name!r}")
    return DeepSeekThinkingChatModel(
        model=model_name,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        temperature=0,
        stream_usage=True,
    )


# Compatibility alias for existing tests and non-product runners. The default
# remains exactly the previous Pure B0 model unless configuration overrides it.
MAIN_LLM = get_main_llm()
