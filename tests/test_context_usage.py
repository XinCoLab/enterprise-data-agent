import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

from agent_runtime import agent_runtime
from agent_runtime.context_usage import MODEL_CONTEXT_WINDOWS, context_usage_for_message
from agent_runtime.translate_graph_events import translate_llm_round_event
from memory import conversation_history_database


MODEL = "deepseek-v4-flash"


def model_message(input_tokens, *, content="Answer", model=MODEL):
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": 17,
            "total_tokens": input_tokens + 17,
        },
        response_metadata={"model_name": model},
    )


def test_provider_input_usage_takes_precedence_and_does_not_count_output():
    message = model_message(124)
    message.response_metadata["token_usage"] = {"prompt_tokens": 999, "total_tokens": 2000}

    assert context_usage_for_message(message) == {
        "input_tokens": 124,
        "context_window": 1_048_576,
        "model": MODEL,
        "source": "provider",
    }
    assert context_usage_for_message(model_message(0))["input_tokens"] == 0


def test_response_metadata_prompt_tokens_are_supported_without_guessing_capacity():
    message = AIMessage(
        content="Answer",
        response_metadata={
            "model_name": "unknown-model",
            "token_usage": {"prompt_tokens": 456, "total_tokens": 700},
        },
    )

    assert context_usage_for_message(message) == {
        "input_tokens": 456,
        "context_window": None,
        "model": "unknown-model",
        "source": "provider",
    }


@pytest.mark.parametrize("tokens", [None, -1, True, "123"])
def test_missing_or_invalid_usage_is_unavailable_not_zero(tokens):
    message = AIMessage(
        content="A long answer must not become a character-based token estimate.",
        response_metadata={"token_usage": {"prompt_tokens": tokens}},
    )

    assert context_usage_for_message(message, model_name=MODEL) == {
        "input_tokens": None,
        "context_window": 1_048_576,
        "model": MODEL,
        "source": "unavailable",
    }


def test_round_and_final_report_the_latest_call_not_a_turn_total():
    latest = model_message(240)
    event = translate_llm_round_event(
        {"data": {"Main Agent LLM": {"messages": [latest]}}}, 2,
    )
    result = agent_runtime.build_turn_result(
        [HumanMessage(content="Question"), model_message(100), latest],
        model_name=MODEL,
    )

    assert event["context_usage"] == result["context_usage"]
    assert result["context_usage"]["input_tokens"] == 240


def test_latest_missing_usage_clears_earlier_known_usage():
    latest = AIMessage(content="Final answer without provider usage")
    result = agent_runtime.build_turn_result(
        [HumanMessage(content="Question"), model_message(100), latest],
        model_name=MODEL,
    )

    assert result["context_usage"]["input_tokens"] is None
    assert result["context_usage"]["source"] == "unavailable"


def test_new_user_turn_does_not_reuse_an_older_turn_snapshot():
    result = agent_runtime.build_turn_result(
        [model_message(500), HumanMessage(content="New question")],
        model_name=MODEL,
    )

    assert result["context_usage"]["input_tokens"] is None


def test_tool_result_does_not_replace_the_last_model_call_usage():
    result = agent_runtime.build_turn_result(
        [
            HumanMessage(content="Question"),
            model_message(240),
            ToolMessage(content='{"returned_rows":1}', tool_call_id="call-1"),
        ],
        model_name=MODEL,
    )

    assert result["context_usage"]["input_tokens"] == 240


def test_context_snapshot_survives_existing_conversation_details_storage():
    conversation_history_database.save_user_message("context-thread", "Question")
    response = agent_runtime.build_chat_response(
        request_id="context-request",
        thread_id="context-thread",
        model=MODEL,
        status="success",
        latency_ms=1,
        details=agent_runtime.build_turn_result(
            [HumanMessage(content="Question"), model_message(240)],
            model_name=MODEL,
        ),
    )
    conversation_history_database.save_assistant_message(
        "context-thread", response["answer"], response,
    )

    stored = conversation_history_database.read_conversation_info("context-thread")
    assert stored["messages"][-1]["details"]["context_usage"] == response["context_usage"]


def test_model_node_tags_the_actual_selected_model_and_invalid_json_keeps_usage(monkeypatch):
    from graph.nodes import main_agent_llm_node as node_module

    output = AIMessage(
        content="",
        invalid_tool_calls=[{"id": "bad-call", "name": "read_knowledge", "args": "{"}],
        usage_metadata={"input_tokens": 640, "output_tokens": 17, "total_tokens": 657},
    )

    class FakeModel:
        async def ainvoke(self, _messages):
            return output

    monkeypatch.setattr(node_module, "_model_with_tools", lambda _model: FakeModel())
    result = asyncio.run(node_module.main_agent_llm_node(
        {"messages": [HumanMessage(content="Question")]},
        {"configurable": {"model": MODEL}},
    ))
    fallback = result["messages"][-1]

    assert fallback.tool_calls == []
    assert fallback.response_metadata["termination_reason"] == "invalid_tool_json"
    assert fallback.usage_metadata == output.usage_metadata
    assert context_usage_for_message(fallback) == {
        "input_tokens": 640,
        "context_window": 1_048_576,
        "model": MODEL,
        "source": "provider",
    }


def test_pause_summary_keeps_its_own_usage_while_removing_tool_calls(monkeypatch):
    from prompts import recursion_limit_summary

    async def fake_summary(_messages, **_kwargs):
        output = model_message(720, content='<|DSML|tool_calls>')
        output.tool_calls = [{"id": "unsafe", "name": "read_knowledge", "args": {}}]
        return output

    monkeypatch.setattr(recursion_limit_summary, "generate_recursion_limit_summary", fake_summary)
    summary = asyncio.run(agent_runtime.generate_round_limit_summary(
        [HumanMessage(content="Question"), model_message(640)], model_name=MODEL,
    ))

    assert summary.tool_calls == []
    assert "DSML" not in summary.content
    assert context_usage_for_message(summary)["input_tokens"] == 720


def test_failed_pause_summary_reports_unavailable_instead_of_previous_usage(monkeypatch):
    from prompts import recursion_limit_summary

    async def failing_summary(_messages, **_kwargs):
        raise RuntimeError("Provider failed")

    monkeypatch.setattr(recursion_limit_summary, "generate_recursion_limit_summary", failing_summary)
    messages = [HumanMessage(content="Question"), model_message(640)]
    summary = asyncio.run(agent_runtime.generate_round_limit_summary(messages, model_name=MODEL))
    result = agent_runtime.build_turn_result([*messages, summary], model_name=MODEL)

    assert result["context_usage"]["input_tokens"] is None
    assert result["context_usage"]["source"] == "unavailable"


def test_page_configuration_exposes_context_windows_in_both_workspace_states(client):
    configured = client.get("/api/page_configuration")
    unconfigured = client.get("/api/page_configuration", headers={"X-Dev-User": "analyst-b"})

    assert configured.status_code == 200
    assert unconfigured.status_code == 200
    assert configured.json()["model_context_windows"] == MODEL_CONTEXT_WINDOWS
    assert unconfigured.json()["model_context_windows"] == MODEL_CONTEXT_WINDOWS
