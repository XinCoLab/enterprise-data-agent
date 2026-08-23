"""Temporary Runtime prompt used only after an Agent run exhausts its loop."""

import json

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage, SystemMessage

from model_clients.llm_api_clients import get_main_llm


RUNTIME_PAUSE_SUMMARY_PROMPT = """[Runtime pause event]

The Agent has exhausted the recursion budget for this run. This is a Runtime
pause, not evidence that the database or a Tool failed.

Using only the existing conversation, Tool Calls, and Tool Results:
- summarize what has already been completed;
- identify the last successfully completed action;
- state which parts of the original request are still unfinished;
- state the exact next executable action when it is known;
- state uncertainty only when the transcript truly does not determine the next action;
- preserve the user's unfinished requirements, filters, and statistical scope exactly;
- do not call Tools;
- do not invent results that are absent from Tool Results;
- do not claim that the original task is complete;
- keep the report concise and actionable.
"""


def generate_runtime_pause_summary(
    messages: list[AnyMessage],
    *,
    model_name: str,
) -> AnyMessage:
    """Generate one Tool-free pause report without persisting its prompt."""

    transcript: list[str] = []
    for message in messages:
        content = message.content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)

        if isinstance(message, HumanMessage):
            transcript.append(f"User message:\n{content}")
        elif isinstance(message, AIMessage):
            if content.strip():
                transcript.append(f"Assistant text:\n{content}")
            for call in message.tool_calls:
                transcript.append(
                    "Tool requested:\n"
                    f"name={call.get('name', '')}\n"
                    f"arguments={json.dumps(call.get('args', {}), ensure_ascii=False, default=str)}"
                )
        elif isinstance(message, ToolMessage):
            transcript.append(
                "Tool result:\n"
                f"name={getattr(message, 'name', '') or ''}\n"
                f"result={content}"
            )
        elif content.strip():
            transcript.append(f"Conversation message:\n{content}")

    model = get_main_llm(model_name)
    return model.invoke(
        [
            SystemMessage(content=RUNTIME_PAUSE_SUMMARY_PROMPT),
            HumanMessage(
                content="[Existing run transcript]\n\n" + "\n\n".join(transcript)
            ),
        ]
    )
