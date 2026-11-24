"""
The LangGraph research agent.
Conversation state is saved per thread_id by the checkpointer, so follow-up
questions on the same thread see the earlier questions and reports.
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .prompts import CRITIC_SYSTEM, RESEARCHER_SYSTEM
from .tools import TOOLS

MODEL = os.getenv("RESEARCH_MODEL", "claude-opus-5-5")
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))
# Max researcher steps that call tools, per question (revisions included).
# One step can hold several parallel tool calls.
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "12"))

BUDGET_EXHAUSTED = (
    "Research budget reached: you cannot call any more tools. "
    "Write the report now using only the sources you already found. "
    "Drop or clearly qualify any claim you could not source."
)
EVIDENCE_CHARS_PER_SOURCE = 3_000

URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
CITATION_RE = re.compile(r"\[(\d+)\]")
SOURCE_LINE_RE = re.compile(r"^\s*[-*]?\s*\[(\d+)\]\s*(.*)$", re.MULTILINE)


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turn_start: int  # index of the current question in `messages`
    revisions: int  # how many times the critic sent the draft back this turn
    tool_rounds: int  # researcher steps that called tools this turn
    report: str  # last draft / final report for this turn


class Critique(BaseModel):
    approved: bool = Field(description="True only if every factual claim is cited and supported.")
    problems: list[str] = Field(default_factory=list, description="Concrete problems to fix; empty if approved.")


def _llm(effort: str) -> ChatAnthropic:
    return ChatAnthropic(
        model=MODEL,
        max_tokens=16_000,
        reasoning_effort=effort,
        # If a safety classifier declines the request, the API retries on a suitable model instead of failing.
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )


_researcher_base = _llm("high")
researcher_llm = _researcher_base.bind_tools(TOOLS)
# Same tools stay declared (the history contains tool calls), but the model may not call them.
researcher_no_tools_llm = _researcher_base.bind_tools(TOOLS, tool_choice={"type": "none"})
critic_llm = _llm("medium").with_structured_output(Critique, method="json_schema")


# ---------------------------------------------------------------- nodes


def start(state: ResearchState) -> dict:
    """Mark where the new question begins and reset per-question counters."""
    return {"turn_start": len(state["messages"]) - 1, "revisions": 0, "tool_rounds": 0, "report": ""}


def researcher(state: ResearchState) -> dict:
    """Decide the next step: call tools, or write (or revise) the report."""
    update: dict = {"messages": []}
    if state["tool_rounds"] >= MAX_TOOL_ROUNDS:
        # Budget spent
        notice = HumanMessage(BUDGET_EXHAUSTED, name="budget")
        update["messages"].append(notice)
        llm, history = researcher_no_tools_llm, [*state["messages"], notice]
    else:
        llm, history = researcher_llm, state["messages"]

    response = llm.invoke([SystemMessage(RESEARCHER_SYSTEM), *history])
    update["messages"].append(response)
    if response.tool_calls:
        update["tool_rounds"] = state["tool_rounds"] + 1
    else:
        update["report"] = response.text
    return update


tool_node = ToolNode(TOOLS)


def _normalize(url: str) -> str:
    return url.rstrip(".,;/").lower()


def _citation_problems(report: str, seen_urls: set[str]) -> list[str]:
    """Deterministic checks the critic does without any LLM usage"""
    problems = []
    body, _, sources = report.partition("## Sources")
    if not sources:
        return ["The report has no '## Sources' section."]

    listed = {int(n): line for n, line in SOURCE_LINE_RE.findall(sources)}
    cited = {int(n) for n in CITATION_RE.findall(body)}
    if not cited:
        problems.append("The report body contains no inline citations.")
    for n in sorted(cited - listed.keys()):
        problems.append(f"Citation [{n}] is used in the text but missing from Sources.")
    for n, line in sorted(listed.items()):
        urls = URL_RE.findall(line)
        if not urls:
            problems.append(f"Source [{n}] has no URL.")
        elif not any(_normalize(u) in seen_urls for u in urls):
            problems.append(f"Source [{n}] ({urls[0]}) never appeared in your search or fetch results; use only sources you actually found.")
    return problems


def critic(state: ResearchState) -> dict:
    """Reflection step: check that every claim in the draft has a supporting source."""
    turn = state["messages"][state["turn_start"]:]
    tool_outputs = [m for m in turn if isinstance(m, ToolMessage)]
    seen_urls = {_normalize(u) for m in tool_outputs for u in URL_RE.findall(str(m.content))}

    problems = _citation_problems(state["report"], seen_urls)

    fetched = [str(m.content) for m in tool_outputs if m.name == "fetch_page"]
    evidence = "\n\n---\n\n".join(c[:EVIDENCE_CHARS_PER_SOURCE] for c in fetched) or "(no pages were fetched)"
    verdict: Critique = critic_llm.invoke(
        [
            SystemMessage(CRITIC_SYSTEM),
            HumanMessage(
                f"<question>\n{turn[0].content}\n</question>\n\n"
                f"<report>\n{state['report']}\n</report>\n\n"
                f"<evidence>\n{evidence}\n</evidence>"
            ),
        ]
    )
    problems += verdict.problems if not verdict.approved else []

    if not problems:
        return {}  # approved

    feedback = "Reviewer feedback - revise the report and fix these problems:\n" + "\n".join(
        f"- {p}" for p in problems
    )
    return {"messages": [HumanMessage(feedback, name="critic")], "revisions": state["revisions"] + 1}


# ---------------------------------------------------------------- edges


def route_after_researcher(state: ResearchState) -> Literal["tools", "critic"]:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "critic"


def route_after_critic(state: ResearchState) -> Literal["researcher", "__end__"]:
    last = state["messages"][-1]
    sent_back = isinstance(last, HumanMessage) and last.name == "critic"
    if sent_back and state["revisions"] <= MAX_REVISIONS:
        return "researcher"
    return END


def build_graph(checkpointer=None):
    g = StateGraph(ResearchState)
    g.add_node("start", start)
    g.add_node("researcher", researcher)
    g.add_node("tools", tool_node)
    g.add_node("critic", critic)

    g.add_edge(START, "start")
    g.add_edge("start", "researcher")
    g.add_conditional_edges("researcher", route_after_researcher)
    g.add_edge("tools", "researcher")
    g.add_conditional_edges("critic", route_after_critic)
    return g.compile(checkpointer=checkpointer)
