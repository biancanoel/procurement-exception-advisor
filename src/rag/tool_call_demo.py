"""Demonstrate one model-requested procurement tool execution. This will be replaced by LangGraph or an agent loop in the future."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag.answerer import DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE
from rag.tools import (
    get_case_facts,
    search_government_awards,
    search_procurement_rules,
)


AVAILABLE_TOOLS = [
    search_procurement_rules,
    get_case_facts,
    search_government_awards,
]


@dataclass(frozen=True)
class ToolExecutionResult:
    """A requested tool call and its one-step execution outcome."""

    tool_call: dict[str, Any] | None
    output: Any | None
    executed: bool
    error: str | None = None


def request_tool_call(
    question: str,
    *,
    chat_model: Any | None = None,
) -> AIMessage:
    """Let the model request an available tool without executing it."""

    if not question.strip():
        raise ValueError("question must not be blank")

    if chat_model is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set to request a tool call"
            )
        chat_model = ChatOpenAI(
            model=os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
            temperature=DEFAULT_TEMPERATURE,
            api_key=api_key,
        )

    model_with_tools = chat_model.bind_tools(AVAILABLE_TOOLS)
    response = model_with_tools.invoke([HumanMessage(content=question)])
    if not isinstance(response, AIMessage):
        raise RuntimeError("ChatOpenAI returned an unexpected response type")
    return response


def print_tool_calls(response: AIMessage) -> None:
    """Print the model's requested tool calls without executing them."""

    if not response.tool_calls:
        print("Tool requested: no")
        return

    print("Tool requested: yes")
    for call in response.tool_calls:
        print(f"Tool name: {call['name']}")
        print(f"Arguments: {call['args']}")
        print(f"Tool-call ID: {call.get('id') or 'not provided'}")


def request_and_execute_tool(
    question: str,
    *,
    chat_model: Any | None = None,
    tool_registry: dict[str, Any] | None = None,
    tool_call_response: AIMessage | None = None,
) -> ToolExecutionResult:
    """Request one tool call and execute it once without model follow-up."""

    response = tool_call_response or request_tool_call(
        question,
        chat_model=chat_model,
    )
    if not response.tool_calls:
        return ToolExecutionResult(
            tool_call=None,
            output=None,
            executed=False,
        )

    tool_call = response.tool_calls[0]
    available_tools = tool_registry or {
        search_procurement_rules.name: search_procurement_rules,
        get_case_facts.name: get_case_facts,
        search_government_awards.name: search_government_awards,
    }
    requested_tool = available_tools.get(tool_call["name"])
    if requested_tool is None:
        return ToolExecutionResult(
            tool_call=tool_call,
            output=None,
            executed=False,
            error=f"Unsupported tool: {tool_call['name']}",
        )

    try:
        output = requested_tool.invoke(tool_call["args"])
    except Exception as error:
        return ToolExecutionResult(
            tool_call=tool_call,
            output=None,
            executed=False,
            error=f"Tool execution failed: {error}",
        )
    return ToolExecutionResult(
        tool_call=tool_call,
        output=output,
        executed=True,
    )


def tool_message_from_result(result: ToolExecutionResult) -> ToolMessage:
    """Serialize a tool execution outcome with its original call ID."""

    if result.tool_call is None:
        raise ValueError("a tool call is required to create a ToolMessage")

    output = result.output
    if output is None:
        serialized_output: Any = []
    elif isinstance(output, BaseModel):
        serialized_output = output.model_dump(mode="json")
    elif isinstance(output, list):
        serialized_output = [
            item.model_dump(mode="json")
            if isinstance(item, BaseModel)
            else item
            for item in (output or [])
        ]
    else:
        serialized_output = output

    observation = {
        "executed": result.executed,
        "error": result.error,
        "output": serialized_output,
    }
    return ToolMessage(
        content=json.dumps(observation, ensure_ascii=False),
        tool_call_id=result.tool_call.get("id") or "missing-tool-call-id",
        name=result.tool_call["name"],
    )


def complete_tool_cycle(
    question: str,
    *,
    chat_model: Any | None = None,
    tool_registry: dict[str, Any] | None = None,
) -> tuple[
    AIMessage,
    ToolExecutionResult,
    ToolMessage | None,
    AIMessage,
]:
    """Run exactly one tool request, execution, and model follow-up."""

    if chat_model is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set to complete a tool cycle"
            )
        chat_model = ChatOpenAI(
            model=os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
            temperature=DEFAULT_TEMPERATURE,
            api_key=api_key,
        )

    first_response = request_tool_call(question, chat_model=chat_model)
    execution = request_and_execute_tool(
        question,
        tool_registry=tool_registry,
        tool_call_response=first_response,
    )
    if execution.tool_call is None:
        return first_response, execution, None, first_response

    observation = tool_message_from_result(execution)
    final_response = chat_model.invoke(
        [
            HumanMessage(content=question),
            first_response,
            observation,
        ]
    )
    if not isinstance(final_response, AIMessage):
        raise RuntimeError("ChatOpenAI returned an unexpected response type")
    return first_response, execution, observation, final_response


def print_execution_result(result: ToolExecutionResult) -> None:
    """Print a requested call and the retrieval observation."""

    if result.tool_call is None:
        print("Tool requested: no")
        print("Tool executed: no")
        return

    print("Tool requested: yes")
    print(f"Tool name: {result.tool_call['name']}")
    print(f"Arguments: {result.tool_call['args']}")
    print(f"Tool-call ID: {result.tool_call.get('id') or 'not provided'}")
    print(f"Tool executed: {'yes' if result.executed else 'no'}")
    if result.error:
        print(f"Error: {result.error}")
        return

    if isinstance(result.output, BaseModel):
        print(f"Case: {result.output.case_id} — {result.output.title}")
        print("Available documents:")
        for document in result.output.available_documents:
            print(f"- {document.title}")
    elif isinstance(result.output, dict):
        structured = result.output.get("structuredContent") or {}
        awards = structured.get("results") or []
        print("Government award results:")
        for rank, award in enumerate(awards, start=1):
            print(
                f"{rank}. {award.get('Award ID') or 'N/A'} — "
                f"{award.get('Recipient Name') or 'N/A'}"
            )
    else:
        print("Retrieval results:")
        for rank, retrieval in enumerate(result.output or [], start=1):
            print(
                f"{rank}. {retrieval.title} — "
                f"section {retrieval.section or 'not specified'}"
            )


def main() -> None:
    """Run one complete procurement tool-use cycle."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("question")
    args = parser.parse_args()
    try:
        first_response, result, observation, final_response = (
            complete_tool_cycle(args.question)
        )
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"User question: {args.question}")
    print_execution_result(result)
    if observation is not None:
        print("Tool observation returned to model: yes")
    print(f"Final model response: {final_response.content}")


if __name__ == "__main__":
    main()
