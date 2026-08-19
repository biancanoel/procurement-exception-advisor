"""Offline tests for the minimal LangGraph tool loop."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from data.case_loader import load_case
from decision.emergency_criteria import EMERGENCY_CRITERIA
from models.assessment import (
    CriterionResult,
    EmergencyAssessment,
    EvidenceReference,
    FinalRecommendation,
)
from models.cases import EmergencyCaseInput
from models.criteria import CriterionStatus
import rag.graph as graph_module
from rag.graph import (
    MAX_RESEARCH_ROUNDS,
    build_graph,
    check_evidence_gaps,
    evaluate_emergency_case,
    prepare_gap_research,
    route_evidence_gaps,
    route_model_response,
    run_graph,
)
from rag.tools import get_case_facts


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


class FakeStructuredModel:
    def __init__(
        self,
        responses: list[EmergencyAssessment | Exception],
    ) -> None:
        self.responses = iter(responses)
        self.inputs: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> EmergencyAssessment:
        self.inputs.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChatModel:
    def __init__(
        self,
        responses: list[AIMessage],
        assessments: list[EmergencyAssessment | Exception] | None = None,
    ) -> None:
        self.bound_tools: list[Any] = []
        self.bound_model = FakeBoundModel(responses)
        self.structured_schema: type[EmergencyAssessment] | None = None
        self.structured_method: str | None = None
        self.structured_model = FakeStructuredModel(assessments or [])

    def bind_tools(self, tools: list[Any]) -> FakeBoundModel:
        self.bound_tools = tools
        return self.bound_model

    def with_structured_output(
        self,
        schema: type[EmergencyAssessment],
        *,
        method: str,
    ) -> FakeStructuredModel:
        self.structured_schema = schema
        self.structured_method = method
        return self.structured_model


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


def criterion_result(
    criterion_id: str,
    *,
    status: CriterionStatus = CriterionStatus.NOT_EVALUATED,
    missing_evidence: list[str] | None = None,
    follow_up_questions: list[str] | None = None,
) -> CriterionResult:
    return CriterionResult(
        criterion_id=criterion_id,
        status=status,
        rationale=f"Assessment rationale for {criterion_id}.",
        missing_evidence=missing_evidence or [],
        follow_up_questions=follow_up_questions or [],
        confidence=0.0 if status == CriterionStatus.NOT_EVALUATED else 0.9,
    )


def assessment_with(*results: CriterionResult) -> EmergencyAssessment:
    return EmergencyAssessment(
        case_id="EM-001",
        recommendation=FinalRecommendation.ADDITIONAL_EVIDENCE_REQUIRED,
        executive_summary="Evidence review is in progress.",
        classification="Emergency procurement",
        criterion_results=list(results),
        overall_confidence=0.5,
    )


def substantive_assessment(case_id: str) -> EmergencyAssessment:
    return EmergencyAssessment(
        case_id=case_id,
        recommendation=FinalRecommendation.SUFFICIENTLY_SUPPORTED,
        executive_summary="The available evidence substantively supports the request.",
        classification="Emergency procurement",
        criterion_results=[
            CriterionResult(
                criterion_id=criterion.criterion_id,
                status=CriterionStatus.SUPPORTED,
                rationale=(
                    f"Available evidence supports {criterion.name.lower()}."
                ),
                confidence=0.8,
            )
            for criterion in EMERGENCY_CRITERIA
        ],
        overall_confidence=0.8,
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
    expected = substantive_assessment("EM-001")
    model = FakeChatModel([], [expected])
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

    update = evaluate_emergency_case(state, chat_model=model)
    assessment = update["assessment"]

    assert assessment is expected
    assert assessment.case_id == "EM-001"
    assert len(assessment.criterion_results) == len(EMERGENCY_CRITERIA) == 13
    assert all(
        result.status == CriterionStatus.SUPPORTED
        for result in assessment.criterion_results
    )
    assert model.structured_schema is EmergencyAssessment
    assert model.structured_method == "json_schema"
    system_prompt = model.structured_model.inputs[0][0][1]
    assessment_prompt = model.structured_model.inputs[0][1][1]
    assert case.request_text in assessment_prompt
    assert "2.24.090" in assessment_prompt
    assert "approval_authority" in assessment_prompt
    assert "Absence of evidence is not evidence of failure" in system_prompt
    assert "funding source, threshold" in system_prompt
    assert "account for resolved\nadverse findings" in system_prompt
    assert "EM001-D01" in assessment.source_ids_used
    assert "rules-call-001" in assessment.source_ids_used


def test_assessment_validation_error_is_returned_to_model_for_retry() -> None:
    case = load_case("EM-001")
    try:
        CriterionResult(
            criterion_id="immediate_harm",
            status=CriterionStatus.SUPPORTED,
            rationale="The harm is supported despite a material conflict.",
            conflicting_evidence=[
                EvidenceReference(
                    source_id="EM001-D01",
                    source_type="case_document",
                    description="The incident timing is materially disputed.",
                )
            ],
            confidence=0.8,
        )
    except ValidationError as error:
        validation_error = error
    else:
        pytest.fail("expected inconsistent result validation to fail")

    corrected = substantive_assessment("EM-001")
    model = FakeChatModel([], [validation_error, corrected])
    update = evaluate_emergency_case(
        {
            "messages": [
                ToolMessage(
                    content=case.model_dump_json(),
                    name="get_case_facts",
                    tool_call_id="case-call-001",
                )
            ],
            "assessment": None,
        },
        chat_model=model,
    )

    assert update["assessment"] is corrected
    assert len(model.structured_model.inputs) == 2
    retry_instruction = model.structured_model.inputs[1][-1][1]
    assert "failed Pydantic validation" in retry_instruction
    assert "immediate_harm" in retry_instruction
    assert "conflicting_evidence" in retry_instruction


def test_evaluate_emergency_case_without_case_facts_is_safe() -> None:
    update = evaluate_emergency_case(
        {
            "messages": [AIMessage(content="No case was requested.")],
            "assessment": None,
        }
    )

    assert update == {"assessment": None}


def test_real_case_tool_artifact_reaches_assessment() -> None:
    case_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_case_facts",
                "args": {"case_id": "EM-005"},
                "id": "case-call-005",
                "type": "tool_call",
            }
        ],
    )
    final = AIMessage(content="Case facts retrieved.")
    model = FakeChatModel(
        [case_request, final],
        [substantive_assessment("EM-005")],
    )

    result = build_graph(
        chat_model=model,
        tools=[get_case_facts],
    ).invoke(
        {
            "messages": [HumanMessage(content="Evaluate EM-005")],
            "research_rounds": 0,
            "max_research_rounds": 0,
            "gap_research_active": False,
            "gap_research_tools_used": False,
        }
    )

    tool_message = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    )
    assert isinstance(tool_message.artifact, EmergencyCaseInput)
    assert tool_message.artifact.case_id == "EM-005"
    assert result["assessment"] is not None
    assert result["assessment"].case_id == "EM-005"


def test_check_evidence_gaps_preserves_all_unresolved_results() -> None:
    unresolved_one = criterion_result(
        "approval_authority",
        missing_evidence=["Applicable approval requirements"],
    )
    resolved = criterion_result(
        "immediate_harm",
        status=CriterionStatus.SUPPORTED,
    )
    unresolved_two = criterion_result(
        "documentation_complete",
        status=CriterionStatus.PARTIALLY_SUPPORTED,
        follow_up_questions=["Is the justification signed?"],
    )
    assessment = assessment_with(
        unresolved_one,
        resolved,
        unresolved_two,
    )

    update = check_evidence_gaps(
        {"messages": [], "assessment": assessment}
    )

    assert update["unresolved_criteria"] == [
        unresolved_one,
        unresolved_two,
    ]
    assert update["unresolved_criteria"][0] is unresolved_one
    assert update["unresolved_criteria"][1] is unresolved_two


def test_zero_unresolved_criteria_routes_to_finalization() -> None:
    assessment = assessment_with(
        criterion_result(
            "immediate_harm",
            status=CriterionStatus.SUPPORTED,
        )
    )
    update = check_evidence_gaps(
        {"messages": [], "assessment": assessment}
    )

    assert update["unresolved_criteria"] == []
    assert route_evidence_gaps(update) == "finalize"


def test_gap_routing_distinguishes_resolved_and_unresolved_statuses() -> None:
    supported = criterion_result(
        "immediate_harm",
        status=CriterionStatus.SUPPORTED,
    )
    adverse = CriterionResult(
        criterion_id="competition_impracticable",
        status=CriterionStatus.NOT_SUPPORTED,
        rationale=(
            "A qualified alternate vendor can deliver within the required "
            "timeframe, so competition is not impracticable."
        ),
        supporting_evidence=[
            EvidenceReference(
                source_id="EM005-D07",
                source_type="case_document",
                description="An alternate vendor can deliver in time.",
            )
        ],
        missing_evidence=["Written file determination"],
        follow_up_questions=["Was the alternative vendor contacted again?"],
        confidence=0.85,
    )
    partial = criterion_result(
        "price_reasonableness",
        status=CriterionStatus.PARTIALLY_SUPPORTED,
        missing_evidence=["Comparable pricing"],
    )
    not_evaluated = criterion_result(
        "threshold_and_funding",
        missing_evidence=["Funding source"],
    )
    assessment = EmergencyAssessment(
        case_id="EM-001",
        recommendation=FinalRecommendation.NOT_SUFFICIENTLY_SUPPORTED,
        executive_summary=(
            "An adverse competition finding prevents sufficient support."
        ),
        classification="Emergency procurement",
        criterion_results=[
            supported,
            adverse,
            partial,
            not_evaluated,
        ],
        overall_confidence=0.7,
    )

    update = check_evidence_gaps(
        {"messages": [], "assessment": assessment}
    )

    assert update["unresolved_criteria"] == [partial, not_evaluated]
    assert adverse in assessment.criterion_results
    assert (
        assessment.recommendation
        == FinalRecommendation.NOT_SUFFICIENTLY_SUPPORTED
    )
    assert route_evidence_gaps(update) == "research"


def test_complete_unresolved_batch_is_passed_to_model_context() -> None:
    unresolved = [
        criterion_result(
            "approval_authority",
            missing_evidence=["Approval evidence"],
        ),
        criterion_result(
            "price_reasonableness",
            missing_evidence=["Vendor quote"],
        ),
        criterion_result(
            "documentation_complete",
            follow_up_questions=["Is the justification signed?"],
        ),
    ]
    state = {
        "messages": [],
        "assessment": assessment_with(*unresolved),
        "unresolved_criteria": unresolved,
        "research_rounds": 0,
        "max_research_rounds": MAX_RESEARCH_ROUNDS,
        "gap_research_active": False,
        "gap_research_tools_used": False,
    }

    assert route_evidence_gaps(state) == "research"
    update = prepare_gap_research(state)
    context = str(update["messages"][0].content)

    assert update["research_rounds"] == 1
    assert context.count('"criterion_id"') == 3
    assert "approval_authority" in context
    assert "price_reasonableness" in context
    assert "documentation_complete" in context


def test_model_can_decline_gap_research_and_preserve_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unresolved = criterion_result(
        "documentation_complete",
        missing_evidence=["Signed justification"],
    )
    assessment = assessment_with(unresolved)
    assessment_calls = 0

    def fake_assessment(
        state: dict[str, Any],
        *,
        chat_model: Any | None = None,
    ) -> dict[str, EmergencyAssessment]:
        nonlocal assessment_calls
        assessment_calls += 1
        return {"assessment": assessment}

    monkeypatch.setattr(
        graph_module,
        "evaluate_emergency_case",
        fake_assessment,
    )
    initial_response = AIMessage(content="Initial research complete.")
    no_tool_response = AIMessage(
        content="The remaining evidence must be provided by the agency."
    )
    model = FakeChatModel([initial_response, no_tool_response])

    result = build_graph(
        chat_model=model,
        tools=[example_lookup],
    ).invoke({"messages": [HumanMessage(content="Evaluate the case")]})

    assert assessment_calls == 1
    assert result["messages"][-1] is no_tool_response
    assert result["unresolved_criteria"] == [unresolved]
    assert result["research_rounds"] == 1
    assert not any(
        isinstance(message, ToolMessage) for message in result["messages"]
    )


def test_gap_tool_calls_share_one_round_then_trigger_reassessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unresolved = assessment_with(
        criterion_result(
            "approval_authority",
            missing_evidence=["Authority evidence"],
        )
    )
    resolved = assessment_with(
        criterion_result(
            "approval_authority",
            status=CriterionStatus.SUPPORTED,
        )
    )
    assessments = iter([unresolved, resolved])
    assessment_calls = 0

    def fake_assessment(
        state: dict[str, Any],
        *,
        chat_model: Any | None = None,
    ) -> dict[str, EmergencyAssessment]:
        nonlocal assessment_calls
        assessment_calls += 1
        return {"assessment": next(assessments)}

    monkeypatch.setattr(
        graph_module,
        "evaluate_emergency_case",
        fake_assessment,
    )
    two_tool_calls = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "example_lookup",
                "args": {"query": "local authority"},
                "id": "gap-call-1",
                "type": "tool_call",
            },
            {
                "name": "example_lookup",
                "args": {"query": "state authority"},
                "id": "gap-call-2",
                "type": "tool_call",
            },
        ],
    )
    model = FakeChatModel(
        [
            AIMessage(content="Initial research complete."),
            two_tool_calls,
            AIMessage(content="Gap research complete."),
        ]
    )

    result = build_graph(
        chat_model=model,
        tools=[example_lookup],
    ).invoke({"messages": [HumanMessage(content="Evaluate the case")]})

    assert assessment_calls == 2
    assert result["assessment"] is resolved
    assert result["unresolved_criteria"] == []
    assert result["research_rounds"] == 1
    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert [message.tool_call_id for message in tool_messages] == [
        "gap-call-1",
        "gap-call-2",
    ]


def test_gap_research_stops_after_three_rounds_with_evidence_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unresolved_result = criterion_result(
        "approval_authority",
        missing_evidence=["Authority evidence"],
    )
    unresolved_assessment = assessment_with(unresolved_result)
    assessment_calls = 0

    def fake_assessment(
        state: dict[str, Any],
        *,
        chat_model: Any | None = None,
    ) -> dict[str, EmergencyAssessment]:
        nonlocal assessment_calls
        assessment_calls += 1
        return {"assessment": unresolved_assessment}

    monkeypatch.setattr(
        graph_module,
        "evaluate_emergency_case",
        fake_assessment,
    )
    responses: list[AIMessage] = [
        AIMessage(content="Initial research complete.")
    ]
    for round_number in range(1, MAX_RESEARCH_ROUNDS + 1):
        responses.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "example_lookup",
                            "args": {"query": f"round {round_number}"},
                            "id": f"round-{round_number}-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content=f"Research round {round_number} complete."),
            ]
        )
    model = FakeChatModel(responses)

    result = build_graph(
        chat_model=model,
        tools=[example_lookup],
    ).invoke({"messages": [HumanMessage(content="Evaluate the case")]})

    assert result["research_rounds"] == MAX_RESEARCH_ROUNDS == 3
    assert assessment_calls == 4
    assert result["unresolved_criteria"] == [unresolved_result]
    assert len(model.bound_model.inputs) == 7
    assert sum(
        isinstance(message, ToolMessage) for message in result["messages"]
    ) == 3
