"""Tests for the minimal Gradio procurement interface."""

import gradio as gr
import pytest
from langchain_core.messages import AIMessage

from models.cases import EmergencyCaseInput
from ui.app import (
    build_app,
    evaluate_description,
    respond_to_message,
    stream_response_to_message,
)


def test_evaluate_description_passes_dynamic_case_to_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_graph(
        question: str,
        *,
        case_input: EmergencyCaseInput,
    ) -> AIMessage:
        captured["question"] = question
        captured["case_input"] = case_input
        return AIMessage(content="## Emergency verification\nAssessment complete.")

    monkeypatch.setattr("ui.app.run_graph", fake_run_graph)

    result = evaluate_description("A critical water pump failed unexpectedly.")

    assert result == "## Emergency verification\nAssessment complete."
    assert captured["question"] == (
        "Evaluate the supplied emergency procurement situation."
    )
    case_input = captured["case_input"]
    assert isinstance(case_input, EmergencyCaseInput)
    assert case_input.description == (
        "A critical water pump failed unexpectedly."
    )


def test_chat_handler_reads_multiple_txt_files_into_dynamic_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_graph(
        question: str,
        *,
        case_input: EmergencyCaseInput,
    ) -> AIMessage:
        captured["question"] = question
        captured["case_input"] = case_input
        return AIMessage(content="The case was evaluated.")

    monkeypatch.setattr("ui.app.run_graph", fake_run_graph)

    incident = tmp_path / "incident_timeline.txt"
    incident.write_text("The pump failed at 08:15.", encoding="utf-8")
    supplier = tmp_path / "supplier_delay.txt"
    supplier.write_text(
        "The supplier cannot deliver for fourteen days.",
        encoding="utf-8",
    )

    result = respond_to_message(
        {
            "text": "A pump failed after the prior inspection.",
            "files": [{"path": str(supplier), "orig_name": supplier.name}],
        },
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "path": str(incident),
                            "orig_name": incident.name,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Essential water service is interrupted.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "This prior agent response is not a case fact.",
            },
        ],
    )

    assert result == "The case was evaluated."
    case_input = captured["case_input"]
    assert isinstance(case_input, EmergencyCaseInput)
    assert case_input.description == (
        "Essential water service is interrupted.\n\n"
        "A pump failed after the prior inspection."
    )
    assert case_input.available_documents == []
    assert [document.evidence_id for document in case_input.case_evidence] == [
        "CASE-D01",
        "CASE-D02",
    ]
    assert [document.filename for document in case_input.case_evidence] == [
        "incident_timeline.txt",
        "supplier_delay.txt",
    ]
    assert [document.extracted_text for document in case_input.case_evidence] == [
        "The pump failed at 08:15.",
        "The supplier cannot deliver for fourteen days.",
    ]
    assert all(
        document.file_type == "txt" for document in case_input.case_evidence
    )


def test_chat_handler_rejects_unsupported_file_type(tmp_path) -> None:
    unsupported = tmp_path / "incident.pdf"
    unsupported.write_text("Not actually a PDF.", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Only \.txt files are accepted"):
        respond_to_message(
            {"text": "A pump failed.", "files": [str(unsupported)]},
            [],
        )


def test_chat_handler_reports_empty_and_invalid_utf8_files(tmp_path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n", encoding="utf-8")
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="empty.txt.*is empty"):
        respond_to_message(
            {"text": "A pump failed.", "files": [str(empty)]},
            [],
        )
    with pytest.raises(ValueError, match="invalid.txt.*not valid UTF-8"):
        respond_to_message(
            {"text": "A pump failed.", "files": [str(invalid)]},
            [],
        )


def test_uploaded_evidence_is_not_added_to_procurement_rule_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    evidence = tmp_path / "timeline.txt"
    evidence.write_text("The outage began at noon.", encoding="utf-8")
    index_calls = 0

    def fail_if_indexed(*args, **kwargs):
        nonlocal index_calls
        index_calls += 1
        raise AssertionError("case evidence must not enter policy indexing")

    monkeypatch.setattr("rag.retriever.index_chunks", fail_if_indexed)
    monkeypatch.setattr(
        "ui.app.run_graph",
        lambda question, *, case_input: AIMessage(content="Evaluated."),
    )

    assert respond_to_message(
        {"text": "A pump failed.", "files": [str(evidence)]},
        [],
    ) == "Evaluated."
    assert index_calls == 0


def test_evaluate_description_rejects_blank_input() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        evaluate_description("   ")


def test_chat_handler_streams_stage_progress_and_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Verification:
        emergency_is_verified = True

    def fake_stream_graph(
        question: str,
        *,
        case_input: EmergencyCaseInput,
    ):
        assert question == "Evaluate the supplied emergency procurement situation."
        assert case_input.description == "A critical pump failed."
        yield (
            "emergency_verification_subagent",
            {"emergency_verification": Verification()},
        )
        yield "procurement_context_subagent", {}
        yield "audit_readiness_subagent", {}
        yield (
            "finalize_assessment",
            {"messages": [AIMessage(content="## Final assessment")]},
        )

    monkeypatch.setattr("ui.app.stream_graph", fake_stream_graph)

    updates = list(
        stream_response_to_message(
            {"text": "A critical pump failed.", "files": []},
            [],
        )
    )

    assert updates[0].startswith("Review started")
    assert updates[1].startswith("Emergency verified")
    assert updates[2].startswith("Procurement context established")
    assert updates[3].startswith("Audit-readiness review complete")
    assert updates[-1] == "## Final assessment"


def test_build_app_has_multimodal_chat_components() -> None:
    app = build_app()

    assert isinstance(app, gr.Blocks)
    components = app.get_config_file()["components"]
    component_types = {component["type"] for component in components}
    assert "chatbot" in component_types
    assert "multimodaltextbox" in component_types
    textbox = next(
        component
        for component in components
        if component["type"] == "multimodaltextbox"
    )
    assert textbox["props"]["file_types"] == [".txt"]
    assert textbox["props"]["file_count"] == "multiple"
