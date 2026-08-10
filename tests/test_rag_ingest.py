"""Tests for minimal page-aware PDF ingestion."""

from pathlib import Path
import zipfile

import pytest

from rag.ingest import (
    chunk_page_text,
    clean_page_text,
    ingest_docx,
    ingest_page_marked_text,
    ingest_pdf,
    santa_monica_metadata,
    santa_monica_municipal_code_metadata,
    santa_monica_ordinance_metadata,
)
from rag.metadata import ProcurementDocumentMetadata


class FakePage:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class FakeReader:
    def __init__(self, _path: Path) -> None:
        self.pages = [
            FakePage("First paragraph.\n\nSecond paragraph."),
            FakePage("Page two policy text."),
            FakePage("   "),
        ]


def make_metadata() -> ProcurementDocumentMetadata:
    return ProcurementDocumentMetadata(
        document_id="TEST-POLICY",
        title="Test Policy",
        jurisdiction="California",
        agency="Test Agency",
        document_type="policy",
        authority_level="local_executive_order",
        exception_type="emergency",
        source_path="original.pdf",
    )


def test_santa_monica_source_metadata_uses_canonical_authority_levels() -> None:
    assert (
        santa_monica_metadata("emergency.pdf").authority_level
        == "local_executive_order"
    )
    assert (
        santa_monica_ordinance_metadata("ordinance.txt").authority_level
        == "local_law"
    )


def test_chunk_page_text_prefers_paragraph_boundaries() -> None:
    chunks = chunk_page_text(
        "Alpha paragraph.\n\nBeta paragraph that is longer.",
        max_chars=25,
    )

    assert chunks == ["Alpha paragraph.", "Beta paragraph that is", "longer."]


def test_chunk_page_text_ignores_empty_text() -> None:
    assert chunk_page_text(None) == []
    assert chunk_page_text(" \n\t ") == []


def test_clean_page_text_removes_page_number_and_docusign_id() -> None:
    cleaned = clean_page_text(
        " 8 \nUseful policy language remains.\n"
        "Docusign Envelope ID: 165751DF-ADF6-4F19-B24D-EBE4FB727869"
    )

    assert cleaned == "Useful policy language remains."


def test_clean_page_text_removes_generic_export_metadata() -> None:
    cleaned = clean_page_text(
        "Emergency purchasing remains authorized. Downloaded from "
        "https://example.gov/code on 2026-08-10\n"
        "§ 2.24.060 § 2.24.090\n"
        "The provision continues."
    )

    assert cleaned == (
        "Emergency purchasing remains authorized.\nThe provision continues."
    )


def test_chunk_page_text_ignores_noise_without_meaningful_prose() -> None:
    text = (
        "10\n"
        "Docusign Envelope ID: 165751DF-ADF6-4F19-B24D-EBE4FB727869\n"
        "1/10/2025"
    )

    assert chunk_page_text(text) == []


def test_top_level_numbered_section_starts_new_chunk() -> None:
    chunks = chunk_page_text(
        "Prior section language continues here.\n"
        "8. Procurement officials may waive notice periods.\n"
        "Additional language for section eight."
    )

    assert chunks == [
        "Prior section language continues here.",
        (
            "8. Procurement officials may waive notice periods. "
            "Additional language for section eight."
        ),
    ]


def test_municipal_code_citation_is_not_treated_as_section_heading() -> None:
    chunks = chunk_page_text(
        "Price-gouging restrictions apply.\n\n"
        "Section 4.36.161 establishes related requirements."
    )

    assert chunks == [
        (
            "Price-gouging restrictions apply.\n\n"
            "Section 4.36.161 establishes related requirements."
        )
    ]


def test_ingest_pdf_creates_chunks_with_metadata_and_pages(monkeypatch) -> None:
    monkeypatch.setattr("rag.ingest.PdfReader", FakeReader)

    page_count, chunks = ingest_pdf("test.pdf", make_metadata())

    assert page_count == 3
    assert len(chunks) == 1
    assert chunks[0].metadata.page == 1
    assert chunks[0].metadata.page_end == 2
    assert chunks[0].metadata.document_id == "TEST-POLICY"
    assert chunks[0].metadata.source_path == "test.pdf"
    assert chunks[0].chunk_id == "TEST-POLICY-p0001-p0002-s001-c001"
    assert "First paragraph." in chunks[0].text
    assert "Page two policy text." in chunks[0].text


def test_ingest_pdf_skips_empty_pages(monkeypatch) -> None:
    monkeypatch.setattr("rag.ingest.PdfReader", FakeReader)

    page_count, chunks = ingest_pdf("test.pdf", make_metadata())

    assert page_count == 3
    assert all(chunk.metadata.page_end != 3 for chunk in chunks)


def test_section_continues_across_page_boundary(monkeypatch) -> None:
    class SectionReader:
        def __init__(self, _path: Path) -> None:
            self.pages = [
                FakePage("8. Procurement authority begins here."),
                FakePage("The same provision continues on the next page."),
                FakePage("9. A new provision begins here."),
            ]

    monkeypatch.setattr("rag.ingest.PdfReader", SectionReader)

    _, chunks = ingest_pdf("test.pdf", make_metadata())

    assert len(chunks) == 2
    assert chunks[0].metadata.page == 1
    assert chunks[0].metadata.page_end == 2
    assert chunks[0].metadata.section == "8."
    assert "continues on the next page" in chunks[0].text
    assert chunks[1].text.startswith("9.")


def test_long_section_splits_only_after_limit(monkeypatch) -> None:
    class LongSectionReader:
        def __init__(self, _path: Path) -> None:
            self.pages = [FakePage("8. " + "policy language. " * 30)]

    monkeypatch.setattr("rag.ingest.PdfReader", LongSectionReader)

    _, chunks = ingest_pdf(
        "test.pdf",
        make_metadata(),
        max_chunk_chars=100,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert all(chunk.metadata.page == 1 for chunk in chunks)


def test_page_marked_text_preserves_pages_and_code_sections(tmp_path) -> None:
    text_path = tmp_path / "ordinance.txt"
    text_path.write_text(
        "Transcription note.\n\n"
        "[PAGE 1]\n2.24.060 City Manager authority.\n"
        "Emergency purchase authority begins here.\n\n"
        "[PAGE 2]\nThe same provision continues here.\n\n"
        "2.24.070 City Attorney authority.\nNew provision.",
        encoding="utf-8",
    )

    page_count, chunks = ingest_page_marked_text(
        text_path,
        santa_monica_ordinance_metadata(text_path),
    )

    assert page_count == 2
    assert len(chunks) == 2
    assert chunks[0].metadata.document_id == "SM-ORD-2849-2026"
    assert chunks[0].metadata.authority_level == "local_law"
    assert chunks[0].metadata.section == "2.24.060"
    assert chunks[0].metadata.page == 1
    assert chunks[0].metadata.page_end == 2
    assert "continues here" in chunks[0].text
    assert chunks[1].metadata.section == "2.24.070"


def test_page_marked_text_rejects_missing_page_markers(tmp_path) -> None:
    text_path = tmp_path / "ordinance.txt"
    text_path.write_text("No page markers here.", encoding="utf-8")

    with pytest.raises(ValueError, match=r"no \[PAGE n\] markers"):
        ingest_page_marked_text(
            text_path,
            santa_monica_ordinance_metadata(text_path),
        )


def test_ingest_docx_preserves_pages_and_removes_boundary_artifacts(
    tmp_path,
) -> None:
    docx_path = tmp_path / "municipal-code.docx"
    paragraphs = [
        "City of Example, CA",
        "§ 2.24.060. City Manager authority.",
        (
            "Emergency purchases are authorized. Downloaded from "
            "https://example.gov/code on 2026-08-10"
        ),
        "City of Example, CA",
        "§ 2.24.060 § 2.24.070",
        "The same provision continues on page two.",
        "§ 2.24.070. Purchasing Agent authority.",
        "The Purchasing Agent may make emergency purchases.",
    ]
    xml_paragraphs = "".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main"><w:body>'
        f"{xml_paragraphs}</w:body></w:document>"
    )
    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    page_count, chunks = ingest_docx(
        docx_path,
        santa_monica_municipal_code_metadata(docx_path),
    )

    assert page_count == 2
    assert len(chunks) == 2
    assert chunks[0].metadata.section == "2.24.060"
    assert chunks[0].metadata.page == 1
    assert chunks[0].metadata.page_end == 2
    assert chunks[1].metadata.section == "2.24.070"
    assert chunks[1].metadata.page == 2
    assert all("Downloaded from" not in chunk.text for chunk in chunks)
    assert all("City of Example" not in chunk.text for chunk in chunks)
    assert all("§ 2.24.060 § 2.24.070" not in chunk.text for chunk in chunks)


def test_municipal_code_metadata_uses_canonical_local_law() -> None:
    metadata = santa_monica_municipal_code_metadata("municipal-code.docx")

    assert metadata.authority_level == "local_law"
    assert metadata.document_type == "municipal_code"
