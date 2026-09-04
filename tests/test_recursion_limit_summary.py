import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from prompts import recursion_limit_summary as summary_module


class _FakeModel:
    def __init__(self):
        self.input_messages = []

    async def ainvoke(self, messages):
        self.input_messages = list(messages)
        return AIMessage(content="当前进度总结")


def test_recursion_limit_prompt_is_temporary_and_uses_an_unbound_model(monkeypatch):
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
    monkeypatch.setattr(summary_module, "get_main_llm", lambda _name: model)

    model_output = asyncio.run(
        summary_module.generate_recursion_limit_summary(
            conversation,
            model_name="deepseek-v4-pro",
        )
    )

    assert model_output.content == "当前进度总结"
    assert isinstance(model.input_messages[0], SystemMessage)
    assert "Recursion limit reached" in model.input_messages[0].content
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
    prompt = summary_module.RECURSION_LIMIT_SUMMARY_PROMPT.lower()

    assert "language" not in prompt
    assert "chinese" not in prompt
    assert "english" not in prompt
