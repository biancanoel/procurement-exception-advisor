"""Tests for the case-facts CLI demo."""

import sys
from types import SimpleNamespace

import pytest
from langchain_core.tools import ToolException

from models.cases import EmergencyCaseInput
from rag.case_tool_demo import main


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


def test_case_tool_demo_passes_case_id_and_prints_json(
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, str] = {}

    def fake_invoke(arguments: dict[str, str]) -> EmergencyCaseInput:
        captured.update(arguments)
        return make_case()

    monkeypatch.setattr(
        "rag.case_tool_demo.get_case_facts",
        SimpleNamespace(invoke=fake_invoke),
    )
    monkeypatch.setattr(sys, "argv", ["get-case-facts", "em-001"])

    main()

    output = capsys.readouterr().out
    assert captured == {"case_id": "em-001"}
    assert '"case_id": "EM-001"' in output
    assert '"title": "Emergency Sewer Main Repair"' in output


def test_case_tool_demo_prints_safe_unknown_case_error(
    monkeypatch,
    capsys,
) -> None:
    def missing_case(_arguments: dict[str, str]) -> None:
        raise ToolException("Case EM-999 was not found")

    monkeypatch.setattr(
        "rag.case_tool_demo.get_case_facts",
        SimpleNamespace(invoke=missing_case),
    )
    monkeypatch.setattr(sys, "argv", ["get-case-facts", "EM-999"])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 1
    assert "Case EM-999 was not found" in capsys.readouterr().err
