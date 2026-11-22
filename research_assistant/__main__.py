"""Command-line entry point.

    python -m research_assistant --thread demo "What is retrieval-augmented generation?"
    python -m research_assistant --thread demo "Who first proposed it?"     # follow-up, same memory
    python -m research_assistant --thread demo                              # interactive chat
    python -m research_assistant --graph                                    # print the Mermaid diagram
"""

from __future__ import annotations

import argparse
import sqlite3
import uuid

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from .graph import MAX_REVISIONS, build_graph  # noqa: E402


def print_update(node: str, update: dict | None) -> None:
    if not update:
        if node == "critic":
            print("  [critic] approved: every claim has a source")
        return
    for msg in update.get("messages", []):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                arg = call["args"].get("query") or call["args"].get("url")
                print(f"  [researcher] {call['name']}: {arg}")
        elif isinstance(msg, AIMessage):
            print("  [researcher] draft written")
        elif isinstance(msg, ToolMessage):
            first = str(msg.content).splitlines()[0][:100] if msg.content else ""
            print(f"  [tools] {msg.name} -> {first}")
        elif isinstance(msg, HumanMessage) and msg.name == "critic":
            print(f"  [critic] sent back (revision {update.get('revisions')}):")
            for line in str(msg.content).splitlines()[1:]:
                print(f"           {line}")


def ask(graph, thread_id: str, question: str) -> None:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    for chunk in graph.stream({"messages": [HumanMessage(question)]}, config, stream_mode="updates"):
        for node, update in chunk.items():
            print_update(node, update)

    state = graph.get_state(config).values
    last = state["messages"][-1]
    if isinstance(last, HumanMessage) and last.name == "critic":
        print(f"\n(warning: the critic still had objections after {MAX_REVISIONS} revisions)")
    print("\n" + state["report"] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Web research agent with a citation critic.")
    parser.add_argument("question", nargs="?", help="Question to research. Omit for interactive mode.")
    parser.add_argument("--thread", default=None, help="Conversation id; reuse it to continue a conversation.")
    parser.add_argument("--db", default="checkpoints.db", help="SQLite file that stores thread memory.")
    parser.add_argument("--graph", action="store_true", help="Print the graph as a Mermaid diagram and exit.")
    args = parser.parse_args()

    if args.graph:
        print(build_graph().get_graph().draw_mermaid())
        return

    thread_id = args.thread or uuid.uuid4().hex[:8]
    with sqlite3.connect(args.db, check_same_thread=False) as conn:
        graph = build_graph(SqliteSaver(conn))
        print(f"thread: {thread_id}")
        if args.question:
            ask(graph, thread_id, args.question)
            return
        while True:
            try:
                question = input("question> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question in {"", "exit", "quit"}:
                break
            ask(graph, thread_id, question)


if __name__ == "__main__":
    main()
