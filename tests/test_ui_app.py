"""Tests for the minimal Gradio procurement interface."""

import gradio as gr
import pytest
from langchain_core.messages import AIMessage

from models.cases import EmergencyCaseInput
from ui.app import build_app, evaluate_description, respond_to_message


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


def test_chat_handler_preserves_text_and_attachment_names(
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
        return AIMessage(content="The case was evaluated.")

    monkeypatch.setattr("ui.app.run_graph", fake_run_graph)

    result = respond_to_message(
        {
            "text": "A pump failed after the prior inspection.",
            "files": [
                "/tmp/inspection.pdf",
                {"path": "/tmp/vendor_quote.csv"},
            ],
        },
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {"path": "/tmp/incident_photo.png"},
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
    assert [
        document.title for document in case_input.available_documents
    ] == ["incident_photo.png", "inspection.pdf", "vendor_quote.csv"]
    assert all(
        "not yet been extracted" in document.summary
        for document in case_input.available_documents
    )


def test_evaluate_description_rejects_blank_input() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        evaluate_description("   ")


def test_build_app_has_multimodal_chat_components() -> None:
    app = build_app()

    assert isinstance(app, gr.Blocks)
    component_types = {
        component["type"]
        for component in app.get_config_file()["components"]
    }
    assert "chatbot" in component_types
    assert "multimodaltextbox" in component_types
