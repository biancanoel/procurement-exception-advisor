"""Offline tests for the minimal LangGraph tool loop."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from data.case_loader import load_case
from decision.emergency_criteria import EMERGENCY_CRITERIA
from models.criteria import CriterionStatus
from rag.graph import (
    build_graph,
    evaluate_emergency_case,
    route_model_response,
    run_graph,
)


@tool
def example_lookup(query: str) -> dict[str, str]:
    """Look up an example value."""

    return {"result": f"Found {query}"}


class FakeBoundModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = iter(responses)
        self.inputs: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.inputs.append(messages)
        return next(self.responses)


class FakeChatModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.bound_tools: list[Any] = []
        self.bound_model = FakeBoundModel(responses)

    def bind_tools(self, tools: list[Any]) -> FakeBoundModel:
        self.bound_tools = tools
        return self.bound_model


def tool_request() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "example_lookup",
                "args": {"query": "emergency pumps"},
                "id": "call-001",
                "type": "tool_call",
            }
        ],
    )


def test_graph_executes_tool_and_loops_back_to_model() -> None:
    final = AIMessage(content="Example Vendor received a similar award.")
    model = FakeChatModel([tool_request(), final])

    result = build_graph(
        chat_model=model,
        tools=[example_lookup],
    ).invoke({"messages": [HumanMessage(content="Find similar awards")]})

    assert model.bound_tools == [example_lookup]
    assert result["messages"][-1] is final
    assert len(model.bound_model.inputs) == 2
    second_model_input = model.bound_model.inputs[1]
    assert isinstance(second_model_input[-1], ToolMessage)
    assert second_model_input[-1].tool_call_id == "call-001"
    assert "Found emergency pumps" in str(second_model_input[-1].content)


def test_graph_ends_when_model_returns_normal_response() -> None:
    final = AIMessage(content="No tool is needed.")
    model = FakeChatModel([final])

    response = run_graph(
        "Hello",
        chat_model=model,
        tools=[example_lookup],
    )

    assert response is final
    assert len(model.bound_model.inputs) == 1


def test_router_selects_tools_or_end() -> None:
    assert route_model_response({"messages": [tool_request()]}) == "tools"
    assert route_model_response(
        {"messages": [AIMessage(content="Done")]}
    ) == "evaluate_emergency_case"


def test_run_graph_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        run_graph("   ", chat_model=FakeChatModel([]), tools=[example_lookup])


def test_evaluate_emergency_case_builds_all_structured_results() -> None:
    case = load_case("EM-001")
    state = {
        "messages": [
            HumanMessage(content="Evaluate EM-001"),
            ToolMessage(
                content=case.model_dump_json(),
                name="get_case_facts",
                tool_call_id="case-call-001",
            ),
            ToolMessage(
                content='{"results": [{"section": "2.24.090"}]}',
                name="search_procurement_rules",
                tool_call_id="rules-call-001",
            ),
            AIMessage(content="I have the case facts."),
        ],
        "assessment": None,
    }

    update = evaluate_emergency_case(state)
    assessment = update["assessment"]

    assert assessment is not None
    assert assessment.case_id == "EM-001"
    assert len(assessment.criterion_results) == len(EMERGENCY_CRITERIA) == 13
    assert all(
        result.status == CriterionStatus.NOT_EVALUATED
        for result in assessment.criterion_results
    )
    assert all(result.missing_evidence for result in assessment.criterion_results)
    assert all(result.follow_up_questions for result in assessment.criterion_results)
    assert "EM001-D01" in assessment.source_ids_used
    assert "rules-call-001" in assessment.source_ids_used


def test_evaluate_emergency_case_without_case_facts_is_safe() -> None:
    update = evaluate_emergency_case(
        {
            "messages": [AIMessage(content="No case was requested.")],
            "assessment": None,
        }
    )

    assert update == {"assessment": None}
