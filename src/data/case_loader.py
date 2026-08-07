"""Read and validate mock emergency-procurement cases from JSON files."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from models.cases import (
    CaseManifest,
    EmergencyCaseExpected,
    EmergencyCaseInput,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_ROOT = PROJECT_ROOT / "data" / "cases"

_CASE_ID_PATTERN = re.compile(r"^EM-[0-9]{3}$")


class CaseDataError(ValueError):
    """Raised when a case file contains invalid or inconsistent data."""


class CaseFileNotFoundError(FileNotFoundError):
    """Raised when a requested case or manifest file does not exist."""


def normalize_case_id(case_id: str) -> str:
    """
    Normalize IDs such as ``em-001`` or ``001`` to ``EM-001``.
    """

    normalized = case_id.strip().upper()

    if re.fullmatch(r"[0-9]{3}", normalized):
        normalized = f"EM-{normalized}"

    if not _CASE_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Invalid case ID {case_id!r}. "
            "Expected 'EM-' followed by three digits."
        )

    return normalized


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk with clear error messages."""

    if not path.is_file():
        raise CaseFileNotFoundError(
            f"Case data file not found: {path}"
        )

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise CaseDataError(
            f"Invalid JSON in {path}: "
            f"line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(data, dict):
        raise CaseDataError(
            f"Expected a JSON object at the root of {path}"
        )

    return data


def load_manifest(
    cases_root: Path = DEFAULT_CASES_ROOT,
) -> CaseManifest:
    """Load and validate the case dataset manifest."""

    path = cases_root / "manifest.json"

    try:
        return CaseManifest.model_validate(
            _read_json(path)
        )
    except ValidationError as exc:
        raise CaseDataError(
            f"Manifest validation failed for {path}: {exc}"
        ) from exc


def list_case_ids(
    cases_root: Path = DEFAULT_CASES_ROOT,
) -> list[str]:
    """Return case IDs in manifest order."""

    manifest = load_manifest(cases_root)

    return [
        entry.case_id
        for entry in manifest.cases
    ]


def load_case(
    case_id: str,
    cases_root: Path = DEFAULT_CASES_ROOT,
) -> EmergencyCaseInput:
    """Load the agent-facing input for one case."""

    normalized = normalize_case_id(case_id)

    path = (
        cases_root
        / "inputs"
        / f"{normalized.lower()}.json"
    )

    try:
        case = EmergencyCaseInput.model_validate(
            _read_json(path)
        )
    except ValidationError as exc:
        raise CaseDataError(
            f"Input validation failed for {path}: {exc}"
        ) from exc

    if case.case_id != normalized:
        raise CaseDataError(
            f"Input filename requested {normalized}, "
            f"but file contains {case.case_id}"
        )

    return case


def load_expected(
    case_id: str,
    cases_root: Path = DEFAULT_CASES_ROOT,
) -> EmergencyCaseExpected:
    """Load the hidden expected result for an evaluation case."""

    normalized = normalize_case_id(case_id)

    path = (
        cases_root
        / "expected"
        / f"{normalized.lower()}.expected.json"
    )

    try:
        expected = EmergencyCaseExpected.model_validate(
            _read_json(path)
        )
    except ValidationError as exc:
        raise CaseDataError(
            f"Expected-result validation failed for {path}: {exc}"
        ) from exc

    if expected.case_id != normalized:
        raise CaseDataError(
            f"Expected filename requested {normalized}, "
            f"but file contains {expected.case_id}"
        )

    return expected


def load_case_pair(
    case_id: str,
    cases_root: Path = DEFAULT_CASES_ROOT,
) -> tuple[EmergencyCaseInput, EmergencyCaseExpected]:
    """
    Load a case and its answer key, then validate cross-file
    references.
    """

    case = load_case(case_id, cases_root)
    expected = load_expected(case_id, cases_root)

    if case.case_id != expected.case_id:
        raise CaseDataError(
            f"Input case ID {case.case_id} does not match "
            f"expected case ID {expected.case_id}"
        )

    referenced_document_ids = {
        document_id
        for assessment in expected.criteria_assessment
        for document_id in assessment.evidence_document_ids
    }

    unknown_document_ids = (
        referenced_document_ids
        - case.available_document_ids
    )

    if unknown_document_ids:
        unknown = ", ".join(
            sorted(unknown_document_ids)
        )

        raise CaseDataError(
            f"Expected result for {case.case_id} "
            f"references unavailable documents: {unknown}"
        )

    return case, expected


def iter_case_pairs(
    cases_root: Path = DEFAULT_CASES_ROOT,
) -> Iterator[
    tuple[EmergencyCaseInput, EmergencyCaseExpected]
]:
    """Yield all validated case pairs in manifest order."""

    for case_id in list_case_ids(cases_root):
        yield load_case_pair(
            case_id,
            cases_root,
        )