"""Offline tests for a single procurement tool-use cycle."""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from models.cases import EmergencyCaseInput
from rag.retriever import RetrievalResult
from rag.tool_call_demo import (
    ToolExecutionResult,
    complete_tool_cycle,
    print_execution_result,
    print_tool_calls,
    request_and_execute_tool,
    request_tool_call,
    tool_message_from_result,
)
from rag.tools import get_case_facts, search_procurement_rules


class FakeBoundModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.input: Any = None

    def invoke(self, model_input: Any) -> AIMessage:
        self.input = model_input
        return self.response


class FakeChatModel:
    def __init__(self, response: AIMessage) -> None:
        self.bound_tools: list[Any] = []
        self.bound_model = FakeBoundModel(response)

    def bind_tools(self, tools: list[Any]) -> FakeBoundModel:
        self.bound_tools = tools
        return self.bound_model


def test_request_tool_call_binds_tool_and_preserves_model_request(
    monkeypatch,
) -> None:
    retrieval_was_called = False

    def fail_if_retrieved(*_args: Any, **_kwargs: Any) -> None:
        nonlocal retrieval_was_called
        retrieval_was_called = True
        raise AssertionError("retrieval must not execute")

    monkeypatch.setattr(
        "rag.tools.retrieve_diversified_chunks",
        fail_if_retrieved,
    )
    expected = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_procurement_rules",
                "args": {
                    "query": "Who may make emergency purchases?",
                    "top_k": 5,
                },
                "id": "call-123",
                "type": "tool_call",
            }
        ],
    )
    chat_model = FakeChatModel(expected)

    response = request_tool_call(
        "Who has authority to make emergency purchases in Santa Monica?",
        chat_model=chat_model,
    )

    assert chat_model.bound_tools == [search_procurement_rules, get_case_facts]
    assert len(chat_model.bound_model.input) == 1
    assert isinstance(chat_model.bound_model.input[0], HumanMessage)
    assert chat_model.bound_model.input[0].content == (
        "Who has authority to make emergency purchases in Santa Monica?"
    )
    assert response is expected
    assert response.tool_calls[0]["name"] == "search_procurement_rules"
    assert response.tool_calls[0]["args"] == {
        "query": "Who may make emergency purchases?",
        "top_k": 5,
    }
    assert response.tool_calls[0]["id"] == "call-123"
    assert retrieval_was_called is False


def test_print_tool_calls_shows_request_details(capsys) -> None:
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_procurement_rules",
                "args": {"query": "emergency purchasing"},
                "id": "call-456",
                "type": "tool_call",
            }
        ],
    )

    print_tool_calls(response)

    output = capsys.readouterr().out
    assert "Tool requested: yes" in output
    assert "Tool name: search_procurement_rules" in output
    assert "'query': 'emergency purchasing'" in output
    assert "Tool-call ID: call-456" in output


def make_tool_call(name: str = "search_procurement_rules") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": {"query": "emergency authority", "top_k": 3},
                "id": "call-789",
                "type": "tool_call",
            }
        ],
    )


def make_case_tool_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_case_facts",
                "args": {"case_id": "EM-001"},
                "id": "call-case-001",
                "type": "tool_call",
            }
        ],
    )


def make_case() -> EmergencyCaseInput:
    return EmergencyCaseInput.model_validate(
        {
            "schema_version": "1.0",
            "case_id": "EM-001",
            "title": "Emergency Sewer Main Repair",
            "workflow_type": "emergency_procurement",
            "jurisdiction": {
                "state": "California",
                "agency": "Pilot City",
            },
            "department": "Public Works",
            "estimated_amount_usd": 184000,
            "proposed_vendor": "Inland Utility Contractors",
            "request_text": "A sewer main ruptured.",
            "available_documents": [],
        }
    )


def make_retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        distance=0.1,
        semantic_similarity=0.9,
        rerank_score=0.88,
        diversity_score=0.48,
        max_selected_overlap=0.1,
        chunk_id="SM-MUNICIPAL-CODE-2.24-p0002-p0003-s007-c001",
        text="The Emergency Services Manager may make emergency purchases.",
        document_id="SM-MUNICIPAL-CODE-2.24",
        authority_level="local_law",
        jurisdiction="California",
        section="2.24.090",
        page=2,
        page_end=3,
        title="Santa Monica Municipal Code Chapter 2.24",
        agency="City of Santa Monica",
    )


def test_request_and_execute_tool_passes_generated_arguments() -> None:
    expected_output = [make_retrieval_result()]

    class FakeTool:
        def __init__(self) -> None:
            self.arguments: dict[str, Any] | None = None

        def invoke(self, arguments: dict[str, Any]) -> list[RetrievalResult]:
            self.arguments = arguments
            return expected_output

    fake_tool = FakeTool()
    result = request_and_execute_tool(
        "Who has authority?",
        chat_model=FakeChatModel(make_tool_call()),
        tool_registry={"search_procurement_rules": fake_tool},
    )

    assert result.executed is True
    assert result.tool_call is not None
    assert result.tool_call["id"] == "call-789"
    assert fake_tool.arguments == {"query": "emergency authority", "top_k": 3}
    assert result.output is expected_output


def test_request_and_execute_tool_handles_model_selected_case_tool() -> None:
    expected_output = make_case()

    class FakeCaseTool:
        def __init__(self) -> None:
            self.arguments: dict[str, Any] | None = None

        def invoke(self, arguments: dict[str, Any]) -> EmergencyCaseInput:
            self.arguments = arguments
            return expected_output

    fake_tool = FakeCaseTool()
    result = request_and_execute_tool(
        "What is the estimated value for case EM-001?",
        chat_model=FakeChatModel(make_case_tool_call()),
        tool_registry={"get_case_facts": fake_tool},
    )

    assert result.executed is True
    assert result.tool_call is not None
    assert result.tool_call["name"] == "get_case_facts"
    assert result.tool_call["id"] == "call-case-001"
    assert fake_tool.arguments == {"case_id": "EM-001"}
    assert result.output is expected_output


def test_case_tool_result_becomes_observation() -> None:
    execution = ToolExecutionResult(
        tool_call=make_case_tool_call().tool_calls[0],
        output=make_case(),
        executed=True,
    )

    observation = tool_message_from_result(execution)
    content = json.loads(str(observation.content))

    assert observation.tool_call_id == "call-case-001"
    assert content["output"]["case_id"] == "EM-001"
    assert content["output"]["proposed_vendor"] == (
        "Inland Utility Contractors"
    )


def test_request_and_execute_tool_does_nothing_without_call() -> None:
    result = request_and_execute_tool(
        "Hello",
        chat_model=FakeChatModel(AIMessage(content="No tool needed.")),
    )

    assert result.executed is False
    assert result.tool_call is None
    assert result.output is None
    assert result.error is None


def test_request_and_execute_tool_handles_unknown_tool_safely() -> None:
    result = request_and_execute_tool(
        "Use an unknown tool",
        chat_model=FakeChatModel(make_tool_call("unknown_tool")),
    )

    assert result.executed is False
    assert result.tool_call is not None
    assert result.tool_call["name"] == "unknown_tool"
    assert result.output is None
    assert result.error == "Unsupported tool: unknown_tool"


def test_print_execution_result_shows_retrieval_observation(capsys) -> None:
    class FakeTool:
        def invoke(self, _arguments: dict[str, Any]) -> list[RetrievalResult]:
            return [make_retrieval_result()]

    result = request_and_execute_tool(
        "Who has authority?",
        chat_model=FakeChatModel(make_tool_call()),
        tool_registry={"search_procurement_rules": FakeTool()},
    )

    print_execution_result(result)

    output = capsys.readouterr().out
    assert "Tool executed: yes" in output
    assert "Santa Monica Municipal Code Chapter 2.24" in output
    assert "section 2.24.090" in output
    assert "authority=local_law" not in output
    assert "Emergency Services Manager may make emergency purchases" not in output


def test_complete_tool_cycle_sends_observation_in_expected_sequence() -> None:
    requested = make_tool_call()
    final = AIMessage(content="The Emergency Services Manager has authority.")

    class CycleChatModel(FakeChatModel):
        def __init__(self) -> None:
            super().__init__(requested)
            self.final_messages: list[Any] = []
            self.final_invocations = 0

        def invoke(self, messages: list[Any]) -> AIMessage:
            self.final_invocations += 1
            self.final_messages = messages
            return final

    class FakeTool:
        def __init__(self) -> None:
            self.invocations = 0

        def invoke(self, _arguments: dict[str, Any]) -> list[RetrievalResult]:
            self.invocations += 1
            return [make_retrieval_result()]

    chat_model = CycleChatModel()
    tool = FakeTool()

    first, execution, observation, final_response = complete_tool_cycle(
        "Who has authority?",
        chat_model=chat_model,
        tool_registry={"search_procurement_rules": tool},
    )

    assert first is requested
    assert isinstance(execution, ToolExecutionResult)
    assert execution.executed is True
    assert isinstance(observation, ToolMessage)
    assert observation.tool_call_id == "call-789"
    assert chat_model.final_messages[0] == HumanMessage(
        content="Who has authority?"
    )
    assert chat_model.final_messages[1] is requested
    assert chat_model.final_messages[2] is observation
    assert final_response is final
    assert tool.invocations == 1
    assert chat_model.final_invocations == 1


def test_failed_tool_execution_becomes_tool_observation() -> None:
    class FailingTool:
        def invoke(self, _arguments: dict[str, Any]) -> None:
            raise RuntimeError("database unavailable")

    execution = request_and_execute_tool(
        "Who has authority?",
        chat_model=FakeChatModel(make_tool_call()),
        tool_registry={"search_procurement_rules": FailingTool()},
    )
    observation = tool_message_from_result(execution)
    content = json.loads(str(observation.content))

    assert isinstance(execution, ToolExecutionResult)
    assert execution.executed is False
    assert execution.error == "Tool execution failed: database unavailable"
    assert observation.tool_call_id == "call-789"
    assert content["executed"] is False
    assert content["error"] == "Tool execution failed: database unavailable"
    assert content["output"] == []


def test_complete_tool_cycle_without_tool_call_has_no_follow_up() -> None:
    response = AIMessage(content="No retrieval is needed.")

    class NoCallChatModel(FakeChatModel):
        def invoke(self, _messages: list[Any]) -> AIMessage:
            raise AssertionError("no follow-up model call should occur")

    first, execution, observation, final = complete_tool_cycle(
        "Hello",
        chat_model=NoCallChatModel(response),
    )

    assert first is response
    assert execution.executed is False
    assert observation is None
    assert final is response
