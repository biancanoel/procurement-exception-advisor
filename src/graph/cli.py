"""Command-line entry point for the LangGraph procurement workflow."""

from __future__ import annotations

import argparse

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv

from graph.workflow import build_graph


def run_question(question: str) -> dict:
    """Run one question through the existing compiled procurement graph."""

    if not question.strip():
        raise ValueError("question must not be blank")
    load_dotenv()
    return build_graph().invoke(
        {"messages": [HumanMessage(content=question)]}
    )


def print_result(result: dict) -> None:
    """Print selected tools and the graph's final model response."""

    tool_calls: list[dict] = []
    tool_observations = 0
    final_response: AIMessage | None = None

    for message in result["messages"]:
        if isinstance(message, AIMessage):
            tool_calls.extend(message.tool_calls)
            if not message.tool_calls:
                final_response = message
        elif isinstance(message, ToolMessage):
            tool_observations += 1

    if tool_calls:
        print("Tools requested:")
        for call in tool_calls:
            print(f"- {call['name']}")
            print(f"  Arguments: {call['args']}")
            print(f"  Tool-call ID: {call.get('id') or 'not provided'}")
        print(f"Tool observations returned: {tool_observations}")
    else:
        print("Tools requested: none")

    if final_response is None:
        raise RuntimeError("graph ended without a final model response")

    print(f"Final response:\n{final_response.content}")


def main() -> None:
    """Ask a question using the LangGraph procurement workflow."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("question", help="Procurement question to ask")
    args = parser.parse_args()

    try:
        print_result(run_question(args.question))
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
