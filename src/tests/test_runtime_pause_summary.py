from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from prompts import runtime_pause_summary as pause_summary


class _FakeModel:
    def __init__(self):
        self.input_messages = []

    def invoke(self, messages):
        self.input_messages = list(messages)
        return AIMessage(content="当前进度总结")


def test_runtime_pause_prompt_is_temporary_and_uses_an_unbound_model(monkeypatch):
    model = _FakeModel()
    conversation = [
        HumanMessage(content="复杂分析任务"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "read_knowledge",
                    "args": {"knowledge_ids": ["table.example"]},
                }
            ],
        ),
        ToolMessage(
            content='{"table": "example"}',
            tool_call_id="call-1",
            name="read_knowledge",
        ),
    ]
    monkeypatch.setattr(pause_summary, "get_main_llm", lambda _name: model)

    reply = pause_summary.generate_runtime_pause_summary(
        conversation,
        model_name="deepseek-v4-pro",
    )

    assert reply.content == "当前进度总结"
    assert isinstance(model.input_messages[0], SystemMessage)
    assert "Runtime pause event" in model.input_messages[0].content
    assert len(model.input_messages) == 2
    assert isinstance(model.input_messages[1], HumanMessage)
    transcript = model.input_messages[1].content
    assert "复杂分析任务" in transcript
    assert "read_knowledge" in transcript
    assert '"table": "example"' in transcript
    assert not any(
        isinstance(message, ToolMessage) for message in model.input_messages
    )
    assert len(conversation) == 3


def test_runtime_pause_prompt_does_not_force_an_output_language():
    prompt = pause_summary.RUNTIME_PAUSE_SUMMARY_PROMPT.lower()

    assert "language" not in prompt
    assert "chinese" not in prompt
    assert "english" not in prompt
