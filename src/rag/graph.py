"""Minimal LangGraph loop for model-directed procurement tools."""

from __future__ import annotations

import os
import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from decision.assessment_builder import create_initial_assessment
from decision.emergency_criteria import EMERGENCY_CRITERIA
from models.assessment import EmergencyAssessment
from models.cases import EmergencyCaseInput
from rag.answerer import DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE
from rag.tool_call_demo import AVAILABLE_TOOLS


class ProcurementGraphState(MessagesState):
    """Shared messages plus the structured emergency assessment."""

    assessment: EmergencyAssessment | None


def create_chat_model() -> ChatOpenAI:
    """Create the same configured ChatOpenAI model used elsewhere."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to run the graph")
    return ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        temperature=DEFAULT_TEMPERATURE,
        api_key=api_key,
    )


def route_model_response(state: MessagesState) -> str:
    """Route tool requests to execution; otherwise finish the graph."""

    latest = state["messages"][-1]
    if isinstance(latest, AIMessage) and latest.tool_calls:
        return "tools"
    return "evaluate_emergency_case"


def _case_from_messages(messages: list[BaseMessage]) -> EmergencyCaseInput | None:
    """Read case facts returned by get_case_facts from tool observations."""

    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if message.name != "get_case_facts":
            continue

        content = message.content
        try:
            data = json.loads(content) if isinstance(content, str) else content
            return EmergencyCaseInput.model_validate(data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def evaluate_emergency_case(
    state: ProcurementGraphState,
) -> dict[str, EmergencyAssessment | None]:
    """Create a conservative 13-criterion assessment from current evidence."""

    case = _case_from_messages(state["messages"])
    if case is None:
        return {"assessment": None}

    assessment = create_initial_assessment(case)
    assessment.source_ids_used = [
        document.document_id
        for document in case.available_documents
    ]
    assessment.source_ids_used.extend(
        message.tool_call_id
        for message in state["messages"]
        if isinstance(message, ToolMessage)
        and message.name != "get_case_facts"
        and message.tool_call_id not in assessment.source_ids_used
    )
    criteria_by_id = {
        criterion.criterion_id: criterion
        for criterion in EMERGENCY_CRITERIA
    }
    for result in assessment.criterion_results:
        criterion = criteria_by_id[result.criterion_id]
        result.rationale = (
            "Current case facts and document summaries do not yet provide "
            "enough criterion-specific evidence for a supported conclusion."
        )
        result.missing_evidence = list(criterion.expected_evidence)
        result.follow_up_questions = list(criterion.questions_to_answer)

    return {"assessment": assessment}


def build_graph(
    *,
    chat_model: Any | None = None,
    tools: Sequence[BaseTool] = AVAILABLE_TOOLS,
) -> Any:
    """Build a model → tools → model StateGraph over shared messages."""

    model = chat_model or create_chat_model()
    model_with_tools = model.bind_tools(list(tools))

    def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = model_with_tools.invoke(state["messages"])
        if not isinstance(response, AIMessage):
            raise RuntimeError("ChatOpenAI returned an unexpected response type")
        return {"messages": [response]}

    builder = StateGraph(ProcurementGraphState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(list(tools)))
    builder.add_node("evaluate_emergency_case", evaluate_emergency_case)
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        route_model_response,
        {
            "tools": "tools",
            "evaluate_emergency_case": "evaluate_emergency_case",
        },
    )
    builder.add_edge("tools", "model")
    builder.add_edge("evaluate_emergency_case", END)
    return builder.compile()


def run_graph(
    question: str,
    *,
    chat_model: Any | None = None,
    tools: Sequence[BaseTool] = AVAILABLE_TOOLS,
) -> AIMessage:
    """Run the graph until the model returns a normal response."""

    if not question.strip():
        raise ValueError("question must not be blank")

    result = build_graph(chat_model=chat_model, tools=tools).invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    final_message = result["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("graph ended without a model response")
    return final_message
