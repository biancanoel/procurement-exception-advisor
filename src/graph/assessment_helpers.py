"""Runtime helpers shared by procurement assessment graph stages."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from graph.shared import EMERGENCY_VERIFICATION_STAGE
from models.assessment import CriterionResult
from models.cases import EmergencyCaseInput
from models.criteria import EmergencyCriterion
from rag.answerer import DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE


MAX_ASSESSMENT_GENERATION_ATTEMPTS = 3

STATUS_SEMANTICS_PROMPT = """Apply these criterion status meanings exactly:

- SUPPORTED is resolved and favorable: affirmative evidence shows the criterion
  is satisfied.
- PARTIALLY_SUPPORTED is unresolved: meaningful evidence supports the criterion,
  but material missing or conflicting evidence prevents a final determination.
- NOT_SUPPORTED is resolved and adverse: affirmative evidence shows the
  criterion is not satisfied. Absence of evidence is not evidence of failure.
- NOT_EVALUATED is unresolved: the record is too incomplete to make a
  substantive favorable or adverse determination.

Use CONTRADICTED only for a resolved adverse determination based on direct
contradictory evidence, NOT_APPLICABLE only when the criterion genuinely does
not apply, and HUMAN_REVIEW_REQUIRED when a human legal, approval, or factual
determination is needed.

For each result, explain the evidence and remaining uncertainty. Put only truly
outstanding items in missing_evidence and follow_up_questions. Preserve material
conflicts in conflicting_evidence. Do not invent facts, documents, approvals,
rules, or source IDs. Source IDs must be document IDs or tool-call IDs present
in the supplied evidence. A SUPPORTED result must have empty
conflicting_evidence and human review set to false. Non-material documentation
improvements may remain in missing_evidence or follow_up_questions, but the
rationale must make clear that they do not prevent the favorable finding. If an
item could change the conclusion, use PARTIALLY_SUPPORTED or NOT_EVALUATED
instead. A NOT_SUPPORTED result must cite affirmative adverse evidence in
supporting_evidence or conflicting_evidence; never select it merely because a
funding source, threshold, document, approval, or other fact is unknown."""


def create_chat_model() -> ChatOpenAI:
    """Create the configured ChatOpenAI model used by assessment stages."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to run the graph")
    return ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        temperature=DEFAULT_TEMPERATURE,
        api_key=api_key,
    )


def route_model_response(state: Mapping[str, Any]) -> str:
    """Route tool calls or continue the state's current assessment stage."""

    latest = state["messages"][-1]
    if isinstance(latest, AIMessage) and latest.tool_calls:
        return "tools"
    if state.get("gap_research_active"):
        if not state.get("gap_research_tools_used"):
            return "finalize"
        return state.get(
            "assessment_stage",
            EMERGENCY_VERIFICATION_STAGE,
        )
    return EMERGENCY_VERIFICATION_STAGE


def case_from_messages(
    messages: Sequence[BaseMessage],
) -> EmergencyCaseInput | None:
    """Read case facts returned by get_case_facts from tool observations."""

    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if message.name != "get_case_facts":
            continue

        if message.artifact is not None:
            try:
                return EmergencyCaseInput.model_validate(message.artifact)
            except (TypeError, ValueError):
                pass

        content = message.content
        try:
            data = json.loads(content) if isinstance(content, str) else content
            return EmergencyCaseInput.model_validate(data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def tool_evidence(messages: Sequence[BaseMessage]) -> str:
    """Format completed non-case tool observations as assessment evidence."""

    observations: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name == "get_case_facts":
            continue
        observations.append(
            "\n".join(
                [
                    f"Tool: {message.name or 'unknown'}",
                    f"Tool-call ID: {message.tool_call_id}",
                    f"Status: {message.status}",
                    f"Observation:\n{message.content}",
                ]
            )
        )
    return "\n\n---\n\n".join(observations) or "No additional tool evidence."


def criteria_context(criteria: Sequence[EmergencyCriterion]) -> str:
    """Serialize one stage's criterion definitions."""

    return json.dumps(
        [criterion.model_dump(mode="json") for criterion in criteria],
        indent=2,
    )


def observed_source_ids(
    case: EmergencyCaseInput,
    messages: Sequence[BaseMessage],
) -> list[str]:
    """Return case-document and completed research observation IDs."""

    source_ids = [
        document.document_id
        for document in case.available_documents
    ]
    source_ids.extend(
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage)
        and message.name != "get_case_facts"
        and message.tool_call_id not in source_ids
    )
    return source_ids


def _assessment_validation_feedback(error: ValidationError) -> str:
    """Format concise Pydantic feedback for a structured-output retry."""

    issues: list[str] = []
    for issue in error.errors(include_url=False):
        location = ".".join(str(part) for part in issue["loc"])
        input_value = issue.get("input")
        criterion_id = (
            input_value.get("criterion_id")
            if isinstance(input_value, dict)
            else None
        )
        criterion = f" ({criterion_id})" if criterion_id else ""
        issues.append(f"- {location}{criterion}: {issue['msg']}")
    return "\n".join(issues)


def invoke_structured_output(
    *,
    model: Any,
    schema: type[Any],
    messages: list[tuple[str, str]],
    output_name: str,
) -> Any:
    """Generate validated structured output with bounded correction retries."""

    structured_model = model.with_structured_output(
        schema,
        method="json_schema",
    )
    request_messages = list(messages)
    for attempt in range(1, MAX_ASSESSMENT_GENERATION_ATTEMPTS + 1):
        try:
            candidate = structured_model.invoke(request_messages)
        except ValidationError as error:
            if attempt == MAX_ASSESSMENT_GENERATION_ATTEMPTS:
                raise RuntimeError(
                    f"chat model could not produce a consistent {output_name} "
                    f"after {attempt} attempts"
                ) from error
            request_messages = [
                *messages,
                (
                    "human",
                    "Your previous structured response failed Pydantic "
                    "validation. Regenerate the complete response and correct "
                    "every issue below. If a gap prevents a final finding, use "
                    "PARTIALLY_SUPPORTED, NOT_EVALUATED, or "
                    "HUMAN_REVIEW_REQUIRED. Use NOT_SUPPORTED only with "
                    "affirmative adverse evidence.\n\n"
                    f"{_assessment_validation_feedback(error)}",
                ),
            ]
            continue
        if not isinstance(candidate, schema):
            raise RuntimeError(
                f"chat model returned no parseable {output_name}"
            )
        return candidate
    raise RuntimeError(f"chat model returned no {output_name}")


def order_stage_results(
    results: list[CriterionResult],
    criteria: Sequence[EmergencyCriterion],
    stage_name: str,
) -> list[CriterionResult]:
    """Validate and order one stage's criterion results."""

    expected_ids = [criterion.criterion_id for criterion in criteria]
    results_by_id = {
        result.criterion_id: result
        for result in results
    }
    if set(results_by_id) != set(expected_ids):
        raise RuntimeError(
            f"{stage_name} did not return every configured criterion"
        )
    return [results_by_id[criterion_id] for criterion_id in expected_ids]


def create_model_node(
    model_with_tools: Any,
) -> Callable[[Mapping[str, Any]], dict[str, list[BaseMessage] | bool]]:
    """Create the model node shared by assessment graphs."""

    def call_model(
        state: Mapping[str, Any],
    ) -> dict[str, list[BaseMessage] | bool]:
        response = model_with_tools.invoke(state["messages"])
        if not isinstance(response, AIMessage):
            raise RuntimeError("ChatOpenAI returned an unexpected response type")
        update: dict[str, list[BaseMessage] | bool] = {
            "messages": [response]
        }
        if state.get("gap_research_active") and response.tool_calls:
            update["gap_research_tools_used"] = True
        return update

    return call_model
