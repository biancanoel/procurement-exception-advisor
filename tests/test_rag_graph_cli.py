"""Tests for the LangGraph command-line entry point."""

import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag.graph_cli import main, print_result, run_question


def test_run_question_invokes_existing_graph(monkeypatch) -> None:
    captured: dict = {}
    dotenv_loaded = False
    expected = {"messages": [AIMessage(content="Done")]}

    def fake_load_dotenv() -> None:
        nonlocal dotenv_loaded
        dotenv_loaded = True

    class FakeGraph:
        def invoke(self, state: dict) -> dict:
            captured.update(state)
            return expected

    monkeypatch.setattr("rag.graph_cli.build_graph", lambda: FakeGraph())
    monkeypatch.setattr("rag.graph_cli.load_dotenv", fake_load_dotenv)

    result = run_question("Find similar emergency awards")

    assert result is expected
    assert dotenv_loaded is True
    assert isinstance(captured["messages"][0], HumanMessage)
    assert captured["messages"][0].content == "Find similar emergency awards"


def test_print_result_shows_tool_and_final_response(capsys) -> None:
    requested = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_government_awards",
                "args": {"keywords": ["emergency pumps"]},
                "id": "call-001",
                "type": "tool_call",
            }
        ],
    )
    observation = ToolMessage(
        content='{"results": []}',
        tool_call_id="call-001",
    )

    print_result(
        {
            "messages": [
                HumanMessage(content="Find awards"),
                requested,
                observation,
                AIMessage(content="I found no matching awards."),
            ]
        }
    )

    output = capsys.readouterr().out
    assert "search_government_awards" in output
    assert "emergency pumps" in output
    assert "Tool observations returned: 1" in output
    assert "I found no matching awards." in output


def test_main_prints_normal_response_without_tool(monkeypatch, capsys) -> None:
    result = {"messages": [AIMessage(content="No tool was needed.")]}
    monkeypatch.setattr(
        "rag.graph_cli.run_question",
        lambda _question: result,
    )
    monkeypatch.setattr(sys, "argv", ["ask-procurement-graph", "Hello"])

    main()

    output = capsys.readouterr().out
    assert "Tools requested: none" in output
    assert "No tool was needed." in output


def test_run_question_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        run_question("   ")
