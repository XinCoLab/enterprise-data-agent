"""上下文压缩节点：主流程在前，辅助实现在后。

阅读顺序：
1. context_compaction_node：检查容量 → 切分历史 → 生成摘要 → 校验并返回状态。
2. split_history / generate_compaction_summary：切分与摘要的具体实现。
3. build_context_messages 及其余辅助函数：组装历史视图、定位边界、序列化消息。

图每轮都会进入节点，但只有超过阈值才尝试压缩。原始 messages 不删除，
主 LLM 与循环上限总结通过 build_context_messages 读取“摘要 + 近期原文”。
"""

from dataclasses import dataclass
import json
import logging
from typing import Callable

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from agent_runtime.agent_run_context import read_agent_run_context
from agent_runtime.input_limit import (
    MAX_INPUT_TOKENS,
    RESERVE_TOKENS,
    InputBudgetExceeded,
    estimate_input_tokens,
)
from agent_runtime.translate_graph_events import extract_visible_ai_content
from graph.graph_state import CompactionState, GraphState
from model_clients.llm_api_clients import get_main_llm
from prompts.context_compaction import (
    DATA_AGENT_SUMMARY_FOCUS,
    SUMMARIZATION_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    SUMMARY_LENGTH_INSTRUCTION,
    SUMMARY_REFINEMENT_PROMPT,
    TURN_PREFIX_SUMMARIZATION_PROMPT,
    UPDATE_SUMMARIZATION_PROMPT,
)


logger = logging.getLogger("Agent")

# 压缩开关与历史保留参数；输入上限和预留空间统一来自 input_limit。
COMPACTION_ENABLED = True
KEEP_RECENT_TOKENS = 20_000
TOOL_RESULT_MAX_CHARS = 2_000
# 包含首次生成，历史摘要、半轮摘要及精简共享整次压缩的三次调用额度。
SUMMARY_MAX_ATTEMPTS = 3


@dataclass
class SummaryCallBudget:
    calls: int = 0


async def context_compaction_node(
    state: GraphState, config: RunnableConfig | None = None, *, threshold_already_crossed: bool = False,
):
    """图每轮先执行此入口：检查容量，必要时压缩，再返回状态更新。"""

    # 主 LLM 节点也使用本文件的 build_context_messages；在调用时导入，避免循环导入。
    from graph.nodes import main_agent_llm_node as main_node

    # 1. 开关关闭或实际组装的输入未超过阈值，就直接进入下一个图节点。
    if not COMPACTION_ENABLED:
        return {}
    model_name = main_node.selected_model_name(read_agent_run_context(config), config)
    model_with_tools = main_node._model_with_tools(model_name)
    tool_definitions = getattr(model_with_tools, "kwargs", {}).get("tools") or []
    model_input, _ = main_node.build_main_model_input(state)
    tokens_before = estimate_input_tokens(messages=model_input, tool_definitions=tool_definitions)
    # 硬上限减去预留空间，压缩触发线正好是 200,000 tokens。
    # 延后压缩实验可传入“本题已经超线”：动态导航缩小输入后，仍兑现先前安排的压缩。
    # 正常图不传此参数，保持原来的调用前容量判断。
    if tokens_before <= MAX_INPUT_TOKENS - RESERVE_TOKENS and not threshold_already_crossed:
        return {}

    # 2. 切出待摘要的旧历史，近期完整消息组继续保留原文。
    history_split = split_history(state, KEEP_RECENT_TOKENS)
    if history_split is None:
        # None 表示按当前保留规则，已经没有能移出去做摘要的旧消息。
        # 例如：剩余原文全部放得进近期保留预算；或只剩必须整组保留的最新工具调用及结果。
        # 总输入超了 200K，仍可能是系统提示、导航图或已有摘要占用过多。
        # 这里不更新 state，继续主 LLM 节点；主节点仍检查最终输入硬上限。
        return {}

    # 最新完整组和动态 System/工具定义本身已超限时，摘要也无法解决。
    retained_input, _ = main_node.build_main_model_input(
        state, context_messages=state["messages"][history_split.cut_index:],
    )
    retained_tokens = estimate_input_tokens(messages=retained_input, tool_definitions=tool_definitions)
    if retained_tokens > MAX_INPUT_TOKENS:
        raise InputBudgetExceeded(estimated_input_tokens=retained_tokens, max_input_tokens=MAX_INPUT_TOKENS)

    # 3. 调用摘要模型；具体的切分和摘要实现都在本文件下方。
    logger.info("上下文压缩开始：估算输入 %s tokens", tokens_before)
    get_stream_writer()({"type": "context_compaction_start"})
    call_budget = SummaryCallBudget()
    summary = await generate_compaction_summary(
        history_split, model_name=model_name, max_input_tokens=MAX_INPUT_TOKENS,
        reserve_tokens=RESERVE_TOKENS, call_budget=call_budget,
    )
    # 4. 用候选摘要重新组装输入；只有缩短且未超限才接受。
    def measure_summary(candidate: str) -> CompactionState:
        compaction: CompactionState = {
            "summary": candidate,
            "first_kept_message_id": state["messages"][history_split.cut_index].id,
            "tokens_before": tokens_before,
            "tokens_after": 0,
        }
        compacted_input, _ = main_node.build_main_model_input({**state, "compaction": compaction})
        compaction["tokens_after"] = estimate_input_tokens(
            messages=compacted_input, tool_definitions=tool_definitions,
        )
        return compaction

    # 两段摘要合并后也要检查；能生成完整摘要，不代表最终主模型输入一定放得下。
    summary_budget = int(0.8 * RESERVE_TOKENS)
    input_target = min(MAX_INPUT_TOKENS, tokens_before - 1)

    def fits(candidate: str) -> bool:
        return (
            summary_token_count(candidate) <= summary_budget
            and measure_summary(candidate)["tokens_after"] <= input_target
        )

    compaction = measure_summary(summary)
    if not fits(summary):
        empty_tokens = measure_summary("")["tokens_after"]
        available_tokens = min(summary_budget, input_target - empty_tokens)
        if available_tokens <= 0:
            raise InputBudgetExceeded(estimated_input_tokens=empty_tokens, max_input_tokens=input_target)
        logger.info("合并摘要或最终输入超预算，继续精简摘要：目标 %s tokens", available_tokens)
        summary = await generate_bounded_summary(
            refinement_request(summary), model_name=model_name,
            output_tokens=summary_budget, summary_budget_tokens=available_tokens,
            max_input_tokens=MAX_INPUT_TOKENS - RESERVE_TOKENS,
            accept_summary=fits, call_budget=call_budget,
        )
        compaction = measure_summary(summary)
    tokens_after = compaction["tokens_after"]
    if tokens_after > MAX_INPUT_TOKENS:
        raise InputBudgetExceeded(estimated_input_tokens=tokens_after, max_input_tokens=MAX_INPUT_TOKENS)
    if tokens_after >= tokens_before:
        raise RuntimeError("Context compaction did not reduce the model input")
    compaction["tokens_after"] = tokens_after
    logger.info(
        "上下文压缩完成：%s → %s tokens，保留 %s 条近期原文；完整历史 %s 条仍在 checkpoint",
        tokens_before, tokens_after, len(state["messages"]) - history_split.cut_index, len(state["messages"]),
    )
    # 5. 只返回摘要和边界的状态更新，原始 messages 保留；由现有 checkpointer 保存。
    return {"compaction": compaction}


# 以下是本节点的辅助实现，先列主流程调用的函数，再列底层消息处理函数。

@dataclass(frozen=True)
class HistorySplit:
    """历史切分结果：近期原文的起点、待摘要的历史及已有摘要。"""

    cut_index: int
    history: list[AnyMessage]
    turn_prefix: list[AnyMessage]
    previous_summary: str


def split_history(state: GraphState, keep_recent_tokens: int) -> HistorySplit | None:
    """按近期原文预算切分历史，返回待摘要部分及近期原文的起点。

    从上次压缩留下的原文起点开始切；首次压缩则从第一条消息开始。
    返回 None 的情况：原文区为空、全部原文都能放进保留预算，
    或只剩一个必须完整保留的最新消息组（即使它超过保留预算也不拆开）。
    这些情况下切点没有向后推进，没有旧消息可以移出去生成摘要。
    """

    start = retained_start_index(state)
    messages = state["messages"]
    cut = find_cut_point(messages, start, keep_recent_tokens)
    if cut <= start:
        # 切点仍在原文区开头：全部继续保留，待摘要部分为空。
        return None
    if not messages[cut].id:
        raise ValueError("Compaction requires checkpoint message IDs")

    # Pi 会额外概括被切开用户轮次的前半段，给保留下来的后半段补任务背景。
    turn_start = cut
    if not isinstance(messages[cut], HumanMessage):
        for index in range(cut - 1, start - 1, -1):
            if isinstance(messages[index], HumanMessage):
                turn_start = index
                break
    return HistorySplit(
        cut_index=cut,
        history=messages[start:turn_start],
        turn_prefix=messages[turn_start:cut],
        previous_summary=(state.get("compaction") or {}).get("summary", ""),
    )


async def generate_compaction_summary(
    history_split: HistorySplit, *, model_name: str, max_input_tokens: int,
    reserve_tokens: int = RESERVE_TOKENS,
    call_budget: SummaryCallBudget | None = None,
) -> str:
    """同一个模型、无工具绑定；旧摘要 + 新退出上下文的原文，增量更新。"""
    call_budget = call_budget if call_budget is not None else SummaryCallBudget()

    async def summarize(messages, instruction, output_tokens, previous_summary=""):
        text = f"<conversation>\n{serialize_conversation(messages)}\n</conversation>\n\n"
        if previous_summary:
            text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        request = [
            SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
            HumanMessage(content=text + instruction + "\n\n" + DATA_AGENT_SUMMARY_FOCUS),
        ]
        return await generate_bounded_summary(
            request, model_name=model_name, output_tokens=output_tokens,
            summary_budget_tokens=output_tokens,
            max_input_tokens=max_input_tokens - reserve_tokens,
            call_budget=call_budget,
        )

    history_summary = history_split.previous_summary
    if history_split.history:
        history_summary = await summarize(
            history_split.history,
            UPDATE_SUMMARIZATION_PROMPT if history_split.previous_summary else SUMMARIZATION_PROMPT,
            int(0.8 * reserve_tokens),
            history_split.previous_summary,
        )
    if history_split.turn_prefix:
        turn_summary = await summarize(
            history_split.turn_prefix, TURN_PREFIX_SUMMARIZATION_PROMPT, int(0.5 * reserve_tokens),
        )
        history_summary = (history_summary or "No prior history.") + (
            "\n\n---\n\n**Turn Context (split turn):**\n\n" + turn_summary
        )
    return history_summary


def summary_token_count(summary: str) -> int:
    """沿用项目的本地 token 估算口径；只计算最终可见摘要。"""
    return estimate_input_tokens(messages=[HumanMessage(content=summary)], tool_definitions=[])


def refinement_request(summary: str) -> list[AnyMessage]:
    return [
        SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
        HumanMessage(content=f"<summary-to-shorten>\n{summary}\n</summary-to-shorten>\n\n{SUMMARY_REFINEMENT_PROMPT}"),
    ]


async def generate_bounded_summary(
    request: list[AnyMessage], *, model_name: str, output_tokens: int,
    summary_budget_tokens: int, max_input_tokens: int,
    accept_summary: Callable[[str], bool] | None = None,
    call_budget: SummaryCallBudget | None = None,
) -> str:
    """完整且预算合格才返回；截断重写原材料，过长完整稿继续精简。"""
    target_tokens = summary_budget_tokens
    feedback = ""
    source_request = request
    last_problem = "no model calls remain in this compaction"
    max_attempts = SUMMARY_MAX_ATTEMPTS
    call_budget = call_budget if call_budget is not None else SummaryCallBudget()
    while call_budget.calls < max_attempts:
        length_instruction = SUMMARY_LENGTH_INSTRUCTION.format(
            target_tokens=target_tokens, feedback=feedback,
        )
        current_request = [*source_request, HumanMessage(content=length_instruction)]
        input_tokens = estimate_input_tokens(messages=current_request, tool_definitions=[])
        if input_tokens > max_input_tokens:
            raise InputBudgetExceeded(estimated_input_tokens=input_tokens, max_input_tokens=max_input_tokens)
        call_budget.calls += 1
        logger.info(
            "本次压缩模型调用 %s/%s：输出上限 %s tokens，摘要目标 %s tokens",
            call_budget.calls, max_attempts, output_tokens, target_tokens,
        )
        response = await get_main_llm(model_name).ainvoke(current_request, max_tokens=output_tokens)
        if response.tool_calls or response.invalid_tool_calls:
            raise RuntimeError("Context compaction attempted to call a tool")
        finish_reason = (response.response_metadata or {}).get("finish_reason")
        if finish_reason not in {"stop", "length"}:
            raise RuntimeError(f"Context compaction did not finish: {finish_reason}")
        if finish_reason == "length":
            # 不能把残缺草稿当成完整信息源，也不能将它持久化为新的摘要。
            last_problem = "generation reached the output limit"
            feedback = "The previous generation was cut off. Rewrite from the supplied source with fewer details; do not continue the cut-off draft."
        else:
            summary = extract_visible_ai_content(response.content)
            if not summary:
                raise RuntimeError("Context compaction returned an empty summary")
            summary_tokens = summary_token_count(summary)
            if summary_tokens <= summary_budget_tokens and (accept_summary is None or accept_summary(summary)):
                return summary
            last_problem = f"complete summary or assembled context exceeded budget (summary estimate: {summary_tokens})"
            feedback = "The previous complete draft is still too long for the next model input. Condense it further without changing the current rules."
            source_request = refinement_request(summary)
        if call_budget.calls < max_attempts:
            target_tokens = max(1, target_tokens // 2)
            logger.warning(
                "摘要需要继续精简：%s；第 %s/%s 次尝试，下一次目标 %s tokens",
                last_problem, call_budget.calls, max_attempts, target_tokens,
            )
    raise RuntimeError(
        f"Context compaction exhausted its {max_attempts}-call budget: {last_problem}; original context was not replaced"
    )


def build_context_messages(state: GraphState) -> list[AnyMessage]:
    """只构造发送视图；不覆盖原始 state['messages']。"""

    start = retained_start_index(state)
    compaction = state.get("compaction")
    if compaction is None:
        return list(state["messages"])
    return [
        HumanMessage(content=(
            "The conversation before this point was compacted into the following summary:\n\n"
            f"<summary>\n{compaction['summary']}\n</summary>"
        )),
        *state["messages"][start:],
    ]


def retained_start_index(state: GraphState) -> int:
    compaction = state.get("compaction")
    if compaction is None:
        return 0
    for index, message in enumerate(state["messages"]):
        if message.id == compaction["first_kept_message_id"]:
            return index
    raise ValueError("Compaction boundary is missing from the conversation history")


def find_cut_point(messages: list[AnyMessage], start: int, keep_recent_tokens: int) -> int:
    """从尾部保留约定预算，AI 的一批 tool_calls 和全部结果作为完整组。

    最近一组即使超过 keepRecentTokens 也完整保留，由最终输入预算检查决定
    能否发送。没有业务分类，不按“看起来重要”选择零散消息。
    """

    groups: list[tuple[int, int]] = []
    index = start
    while index < len(messages):
        group_start = index
        message = messages[index]
        index += 1
        if isinstance(message, ToolMessage):
            raise ValueError("Cannot compact an orphan ToolMessage")
        if isinstance(message, AIMessage) and message.tool_calls:
            call_ids = [str(call.get("id") or "") for call in message.tool_calls]
            if not all(call_ids) or len(set(call_ids)) != len(call_ids):
                raise ValueError("Cannot compact invalid tool call IDs")
            pending = set(call_ids)
            while pending:
                if index >= len(messages):
                    raise ValueError("Cannot compact incomplete tool calls")
                result = messages[index]
                if not isinstance(result, ToolMessage) or result.tool_call_id not in pending:
                    raise ValueError("Cannot compact mismatched tool results")
                pending.remove(result.tool_call_id)
                index += 1
        groups.append((group_start, index))

    accumulated = 0
    cut = start
    for group_start, group_end in reversed(groups):
        tokens = estimate_input_tokens(messages=messages[group_start:group_end], tool_definitions=[])
        if accumulated and accumulated + tokens > keep_recent_tokens:
            break
        accumulated += tokens
        cut = group_start
    return cut


def serialize_conversation(messages: list[AnyMessage]) -> str:
    """只缩短摘要模型的工具结果输入；checkpoint 和保留区原文不截断。"""

    parts = []
    for message in messages:
        content = extract_visible_ai_content(message.content)
        if isinstance(message, AIMessage):
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning:
                parts.append(f"[Assistant thinking]: {reasoning}")
            if content:
                parts.append(f"[Assistant]: {content}")
            if message.tool_calls:
                parts.append("[Assistant tool calls]: " + json.dumps(
                    message.tool_calls, ensure_ascii=False, default=str,
                ))
        elif isinstance(message, ToolMessage):
            if len(content) > TOOL_RESULT_MAX_CHARS:
                omitted = len(content) - TOOL_RESULT_MAX_CHARS
                content = content[:TOOL_RESULT_MAX_CHARS] + f"\n[... {omitted} more characters truncated]"
            parts.append(f"[Tool result {message.name or ''} ({message.tool_call_id})]: {content}")
        else:
            parts.append(f"[{message.type}]: {content}")
    return "\n\n".join(parts)
