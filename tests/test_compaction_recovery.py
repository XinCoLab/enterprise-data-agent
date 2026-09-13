import asyncio
from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent_runtime.input_limit import MAX_INPUT_TOKENS
from graph.nodes import context_compaction_node as compaction
from graph.nodes import main_agent_llm_node as main_node


MODEL = "deepseek-v4-flash"


def reply(text, reason="stop"):
    return AIMessage(content=text, response_metadata={"finish_reason": reason})


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))
        return next(self.responses)


def use_model(monkeypatch, responses):
    model = ScriptedModel(responses)
    monkeypatch.setattr(compaction, "get_main_llm", lambda _name: model)
    return model


def bounded(request, **kwargs):
    return compaction.generate_bounded_summary(
        request, model_name=MODEL, output_tokens=256,
        summary_budget_tokens=128, max_input_tokens=10_000, **kwargs,
    )


def test_truncated_drafts_are_rewritten_from_original_and_third_call_can_succeed(monkeypatch):
    model = use_model(monkeypatch, [
        reply("TRUNCATED-ONE", "length"),
        reply("TRUNCATED-TWO", "length"),
        reply("Keep 6 seats; returning accounts DESC, participants DESC, ID ASC."),
    ])
    original = [HumanMessage(content="ORIGINAL rules and complete history")]
    before = deepcopy(original)

    result = asyncio.run(bounded(original))

    assert result.startswith("Keep 6 seats")
    assert len(model.calls) == 3
    assert original == before
    for messages, kwargs in model.calls:
        source = "\n".join(str(m.content) for m in messages)
        assert "ORIGINAL rules and complete history" in source
        assert "TRUNCATED-ONE" not in source and "TRUNCATED-TWO" not in source
        assert kwargs["max_tokens"] == 256


def test_complete_long_summary_is_shortened_instead_of_discarded(monkeypatch):
    draft = "A complete but unnecessarily detailed account. " * 100
    model = use_model(monkeypatch, [reply(draft), reply("Keep 6 seats and the agreed ordering.")])

    result = asyncio.run(bounded([HumanMessage(content="Original long conversation")]))

    assert result == "Keep 6 seats and the agreed ordering."
    assert len(model.calls) == 2
    assert draft.strip() in str(model.calls[1][0][1].content)


def test_context_fit_is_checked_even_if_summary_itself_is_short(monkeypatch):
    model = use_model(monkeypatch, [reply("Complete but too big for the available space."), reply("Fits.")])

    result = asyncio.run(bounded(
        [HumanMessage(content="Source")], accept_summary=lambda text: text == "Fits.",
    ))

    assert result == "Fits."
    assert len(model.calls) == 2


@pytest.mark.parametrize("reason", ["content_filter", "insufficient_system_resource", "aborted"])
def test_non_length_failures_are_not_treated_as_complete_summaries(monkeypatch, reason):
    model = use_model(monkeypatch, [reply("Partial result", reason), reply("Must not be called")])

    with pytest.raises(RuntimeError, match=reason):
        asyncio.run(bounded([HumanMessage(content="Source")]))

    assert len(model.calls) == 1


def node_setup(monkeypatch, *, memory_enabled=False, system_tokens=0):
    state = {
        "messages": [
            HumanMessage(content="Old history. " * 400, id="old"),
            HumanMessage(content="Continue.", id="recent"),
        ],
        "retrieved_memories": [{"memory": "The agreed seat limit is 6."}] if memory_enabled else [],
    }

    def build_input(candidate, *, context_messages=None):
        messages = context_messages if context_messages is not None else compaction.build_context_messages(candidate)
        memory = "\n".join(m["memory"] for m in candidate.get("retrieved_memories", []))
        return [SystemMessage(content="s" * (4 * system_tokens) + memory), *messages], {}

    monkeypatch.setattr(main_node, "build_main_model_input", build_input)
    monkeypatch.setattr(main_node, "_model_with_tools", lambda _name: Mock(kwargs={"tools": []}))
    monkeypatch.setattr(main_node, "selected_model_name", lambda *_args: MODEL)
    monkeypatch.setattr(compaction, "read_agent_run_context", lambda _config: None)
    monkeypatch.setattr(compaction, "get_stream_writer", lambda: Mock())
    monkeypatch.setattr(compaction, "split_history", lambda *_args: compaction.HistorySplit(
        cut_index=1, history=state["messages"][:1], turn_prefix=[], previous_summary="",
    ))
    return state


@pytest.mark.parametrize("memory_enabled", [False, True], ids=["A", "B"])
def test_three_call_failure_preserves_context_in_both_arms(monkeypatch, memory_enabled):
    state = node_setup(monkeypatch, memory_enabled=memory_enabled)
    state["compaction"] = {
        "summary": "Previous valid summary.", "first_kept_message_id": "old",
        "tokens_before": 200_100, "tokens_after": 70_000,
    }
    before = deepcopy(state)
    model = use_model(monkeypatch, [
        reply("Cut off", "length"), reply("Cut off", "length"), reply("Cut off", "length"),
        reply("A fourth call would succeed but must not happen"),
    ])

    with pytest.raises(RuntimeError, match="exhausted its 3-call budget"):
        asyncio.run(compaction.context_compaction_node(state, threshold_already_crossed=True))

    assert len(model.calls) == 3
    assert state == before


@pytest.mark.parametrize("memory_enabled", [False, True], ids=["A", "B"])
def test_history_turn_prefix_and_final_refinement_share_three_calls(monkeypatch, memory_enabled):
    state = node_setup(monkeypatch, memory_enabled=memory_enabled)
    before = deepcopy(state)
    monkeypatch.setattr(compaction, "RESERVE_TOKENS", 200)
    monkeypatch.setattr(compaction, "split_history", lambda *_args: compaction.HistorySplit(
        cut_index=1, history=state["messages"][:1],
        turn_prefix=[HumanMessage(content="Part of a large turn")], previous_summary="",
    ))
    # Each component fits separately, but their combined size exceeds 160 tokens.
    model = use_model(monkeypatch, [reply("h" * 500), reply("t" * 320), reply("Complete compact summary.")])

    result = asyncio.run(compaction.context_compaction_node(state, threshold_already_crossed=True))

    assert len(model.calls) == 3
    assert [kwargs["max_tokens"] for _, kwargs in model.calls] == [160, 100, 160]
    assert result["compaction"]["summary"] == "Complete compact summary."
    assert result["compaction"]["first_kept_message_id"] == "recent"
    assert result["compaction"]["tokens_after"] < result["compaction"]["tokens_before"]
    assert state == before


def test_failed_final_refinement_cannot_start_a_separate_retry_budget(monkeypatch):
    state = node_setup(monkeypatch)
    before = deepcopy(state)
    monkeypatch.setattr(compaction, "RESERVE_TOKENS", 200)
    monkeypatch.setattr(compaction, "split_history", lambda *_args: compaction.HistorySplit(
        cut_index=1, history=state["messages"][:1],
        turn_prefix=[HumanMessage(content="Part of a large turn")], previous_summary="",
    ))
    model = use_model(monkeypatch, [
        reply("h" * 500), reply("t" * 320), reply("Incomplete replacement", "length"),
        reply("Fourth call must not happen"),
    ])

    with pytest.raises(RuntimeError, match="exhausted its 3-call budget"):
        asyncio.run(compaction.context_compaction_node(state, threshold_already_crossed=True))

    assert len(model.calls) == 3
    assert state == before


def test_complete_summary_is_refined_when_assembled_main_input_is_too_large(monkeypatch):
    state = node_setup(monkeypatch, system_tokens=MAX_INPUT_TOKENS - 50)
    model = use_model(monkeypatch, [reply("Detailed summary. " * 60), reply("Compact.")])

    result = asyncio.run(compaction.context_compaction_node(state, threshold_already_crossed=True))

    assert len(model.calls) == 2
    assert result["compaction"]["summary"] == "Compact."
    assert result["compaction"]["tokens_after"] <= MAX_INPUT_TOKENS
