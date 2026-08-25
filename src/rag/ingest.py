"""Minimal page-aware ingestion for procurement policy documents."""

from __future__ import annotations

import argparse
import re
import zipfile
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader

from rag.metadata import DocumentChunk, ProcurementDocumentMetadata


DEFAULT_PDF_PATH = Path(
    "data/policies/santa_monica/"
    "Amended_Local_Emergency_2025_-_Palisades_Fire_1.10.25.pdf"
)
DEFAULT_ORDINANCE_TEXT_PATH = Path(
    "data/policies/santa_monica/santa-monica-ordinance-2849-2026.txt"
)
DEFAULT_MUNICIPAL_CODE_DOCX_PATH = Path(
    "data/policies/santa_monica/SM Municipal Code.docx"
)
DEFAULT_BIDDING_THRESHOLDS_DOCX_PATH = Path(
    "data/policies/santa_monica/Bidding Thresholds.docx"
)
DEFAULT_CALIFORNIA_PCC_1102_PATH = Path(
    "data/policies/california/PCC_1102.pdf"
)
DEFAULT_MAX_CHUNK_CHARS = 4_000
_DOCUSIGN_ID_RE = re.compile(
    r"^docusign envelope id:\s*[0-9a-f-]+$",
    re.IGNORECASE,
)
_STANDALONE_PAGE_NUMBER_RE = re.compile(r"^\d+$")
_NAMED_SECTION_RE = re.compile(
    r"^(?:§\s*)?(?:(?P<code>\d+\.\d+\.\d+)\.?\s+|"
    r"(?P<statute>\d{3,6}(?:\.\d+)?)\.\s+|"
    r"(?P<ordinance>SECTION\s+\d+\.)(?=\s))",
    re.IGNORECASE,
)
_SIMPLE_NUMBERED_SECTION_RE = re.compile(r"^(?P<number>\d{1,2}\.)\s+(?=[A-Z])")
_PAGE_MARKER_RE = re.compile(r"^\[PAGE\s+(\d+)\]$", re.IGNORECASE)
_PROSE_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_DOWNLOAD_METADATA_RE = re.compile(
    r"\s*Downloaded from https?://\S+(?:\s+on\s+\d{4}-\d{2}-\d{2})?\s*",
    re.IGNORECASE,
)
_SECTION_RANGE_HEADER_RE = re.compile(
    r"^§\s*\d+(?:\.\d+)+\s+§\s*\d+(?:\.\d+)+$"
)
_LEGAL_EXPORT_HEADER_RE = re.compile(
    r"^(?:State of [A-Za-z ]+|[A-Z][A-Z ]+ CODE|Section\s+\d+(?:\.\d+)?)$"
)
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _section_label(
    text: str,
    *,
    include_simple_numbered_sections: bool = True,
) -> str | None:
    """Return a simple legal-section label when text begins with one."""

    match = _NAMED_SECTION_RE.match(text)
    if not match and include_simple_numbered_sections:
        match = _SIMPLE_NUMBERED_SECTION_RE.match(text)
    if not match:
        return None
    return next(value for value in match.groupdict().values() if value)


def clean_page_text(
    text: str | None,
    *,
    include_simple_numbered_sections: bool = True,
) -> str:
    """Remove common export artifacts while preserving paragraphs."""

    if not text:
        return ""

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = _DOWNLOAD_METADATA_RE.sub("", raw_line).strip()
        if not line:
            if not raw_line.strip() and cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        if _STANDALONE_PAGE_NUMBER_RE.fullmatch(line):
            continue
        if _DOCUSIGN_ID_RE.fullmatch(line):
            continue
        if _SECTION_RANGE_HEADER_RE.fullmatch(line):
            continue
        if _LEGAL_EXPORT_HEADER_RE.fullmatch(line):
            continue
        if _section_label(
            line,
            include_simple_numbered_sections=include_simple_numbered_sections,
        ) and cleaned_lines:
            if cleaned_lines[-1]:
                cleaned_lines.append("")
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _contains_meaningful_prose(text: str) -> bool:
    """Return whether text contains at least one prose-like word."""

    return bool(_PROSE_WORD_RE.search(text))


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split oversized text at sentence or word boundaries."""

    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidates: list[str] = []
        candidate = ""
        for word in sentence.split():
            if len(word) > max_chars:
                if candidate:
                    candidates.append(candidate)
                    candidate = ""
                candidates.extend(
                    word[index : index + max_chars]
                    for index in range(0, len(word), max_chars)
                )
                continue
            combined_word = f"{candidate} {word}".strip()
            if candidate and len(combined_word) > max_chars:
                candidates.append(candidate)
                candidate = word
            else:
                candidate = combined_word
        if candidate:
            candidates.append(candidate)

        for candidate in candidates:
            combined = f"{current} {candidate}".strip()
            if current and len(combined) > max_chars:
                pieces.append(current)
                current = candidate
            else:
                current = combined

    if current:
        pieces.append(current)
    return pieces


def chunk_page_text(
    text: str | None,
    *,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    include_simple_numbered_sections: bool = True,
) -> list[str]:
    """Group page text by paragraph, merging small adjacent paragraphs."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    cleaned_text = clean_page_text(
        text,
        include_simple_numbered_sections=include_simple_numbered_sections,
    )
    if not cleaned_text:
        return []

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", cleaned_text)
        if paragraph.strip()
    ]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if _section_label(
            paragraph,
            include_simple_numbered_sections=include_simple_numbered_sections,
        ) and current:
            if _contains_meaningful_prose(current):
                chunks.append(current)
            current = ""

        parts = (
            [paragraph]
            if len(paragraph) <= max_chars
            else _split_long_text(paragraph, max_chars)
        )
        for part in parts:
            combined = f"{current}\n\n{part}".strip()
            if current and len(combined) > max_chars:
                if _contains_meaningful_prose(current):
                    chunks.append(current)
                current = part
            else:
                current = combined

    if current and _contains_meaningful_prose(current):
        chunks.append(current)
    return chunks


def _logical_sections(
    page_texts: list[str | None],
    *,
    include_simple_numbered_sections: bool = True,
) -> list[tuple[str, int, int, str | None]]:
    """Assemble top-level sections across physical PDF page boundaries."""

    sections: list[tuple[str, int, int, str | None]] = []
    current_paragraphs: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    section_label: str | None = None

    def finish_section() -> None:
        nonlocal current_paragraphs, page_start, page_end, section_label
        text = "\n\n".join(current_paragraphs).strip()
        if text and page_start is not None and page_end is not None:
            sections.append((text, page_start, page_end, section_label))
        current_paragraphs = []
        page_start = None
        page_end = None
        section_label = None

    for page_number, raw_text in enumerate(page_texts, start=1):
        cleaned_text = clean_page_text(
            raw_text,
            include_simple_numbered_sections=include_simple_numbered_sections,
        )
        paragraphs = [
            re.sub(r"\s+", " ", paragraph).strip()
            for paragraph in re.split(r"\n\s*\n", cleaned_text)
            if paragraph.strip()
        ]
        for paragraph in paragraphs:
            section = _section_label(
                paragraph,
                include_simple_numbered_sections=include_simple_numbered_sections,
            )
            if section:
                if current_paragraphs:
                    finish_section()
                section_label = section
            if not current_paragraphs:
                page_start = page_number
            current_paragraphs.append(paragraph)
            page_end = page_number

    finish_section()
    return sections


def _chunks_from_pages(
    page_texts: list[str | None],
    metadata: ProcurementDocumentMetadata,
    *,
    source_path: str | Path,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    include_simple_numbered_sections: bool = True,
) -> list[DocumentChunk]:
    """Build section-aware chunks from ordered source pages."""

    chunks: list[DocumentChunk] = []

    for section_number, (
        section_text,
        page_start,
        page_end,
        section_label,
    ) in enumerate(
        _logical_sections(
            page_texts,
            include_simple_numbered_sections=include_simple_numbered_sections,
        ),
        start=1,
    ):
        chunk_metadata = metadata.model_copy(
            update={
                "page": page_start,
                "page_end": page_end,
                "section": section_label,
                "source_path": str(source_path),
            }
        )
        for section_part_number, text in enumerate(
            chunk_page_text(
                section_text,
                max_chars=max_chunk_chars,
                include_simple_numbered_sections=(
                    include_simple_numbered_sections
                ),
            ),
            start=1,
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"{metadata.document_id}-"
                        f"p{page_start:04d}-p{page_end:04d}-"
                        f"s{section_number:03d}-c{section_part_number:03d}"
                    ),
                    text=text,
                    metadata=chunk_metadata,
                )
            )

    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    metadata: ProcurementDocumentMetadata,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> tuple[int, list[DocumentChunk]]:
    """Extract a PDF into section-aware chunks with page-span metadata."""

    path = Path(pdf_path)
    reader = PdfReader(path)
    page_texts = [page.extract_text() for page in reader.pages]
    chunks = _chunks_from_pages(
        page_texts,
        metadata,
        source_path=path,
        max_chunk_chars=max_chunk_chars,
    )
    return len(reader.pages), chunks


def _docx_page_texts(docx_path: str | Path) -> list[str]:
    """Extract DOCX paragraphs, honoring page breaks and export footers."""

    with zipfile.ZipFile(docx_path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))

    pages: list[list[str]] = [[]]
    for paragraph in root.iter(f"{_WORD_NS}p"):
        parts: list[str] = []
        for element in paragraph.iter():
            if element.tag == f"{_WORD_NS}t" and element.text:
                parts.append(element.text)
            elif element.tag == f"{_WORD_NS}tab":
                parts.append("\t")
            elif element.tag == f"{_WORD_NS}lastRenderedPageBreak" or (
                element.tag == f"{_WORD_NS}br"
                and element.get(f"{_WORD_NS}type") == "page"
            ):
                text = "".join(parts).strip()
                if text:
                    pages[-1].append(text)
                pages.append([])
                parts = []

        paragraph_text = "".join(parts).strip()
        footer = _DOWNLOAD_METADATA_RE.search(paragraph_text)
        if footer:
            before_footer = paragraph_text[: footer.start()].strip()
            if before_footer:
                pages[-1].append(before_footer)
            if pages[-1]:
                pages.append([])
        elif paragraph_text:
            pages[-1].append(paragraph_text)

    return ["\n\n".join(page) for page in pages if page]


def _remove_repeated_docx_headers(page_texts: list[str]) -> list[str]:
    """Remove short paragraphs repeated near the start of exported pages."""

    boundary_counts: dict[str, int] = {}
    for page_text in page_texts:
        paragraphs = [part.strip() for part in page_text.split("\n\n") if part.strip()]
        for paragraph in paragraphs[:3]:
            normalized = re.sub(r"\s+", " ", paragraph).casefold()
            if len(normalized) <= 100:
                boundary_counts[normalized] = boundary_counts.get(normalized, 0) + 1

    repeated = {text for text, count in boundary_counts.items() if count >= 2}
    cleaned_pages: list[str] = []
    for page_text in page_texts:
        paragraphs = [part.strip() for part in page_text.split("\n\n") if part.strip()]
        cleaned_pages.append(
            "\n\n".join(
                paragraph
                for index, paragraph in enumerate(paragraphs)
                if not (
                    index < 3
                    and re.sub(r"\s+", " ", paragraph).casefold()
                    in repeated
                )
            )
        )
    return cleaned_pages


def ingest_docx(
    docx_path: str | Path,
    metadata: ProcurementDocumentMetadata,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> tuple[int, list[DocumentChunk]]:
    """Extract a DOCX into section-aware chunks with page-span metadata."""

    path = Path(docx_path)
    page_texts = _remove_repeated_docx_headers(_docx_page_texts(path))
    chunks = _chunks_from_pages(
        page_texts,
        metadata,
        source_path=path,
        max_chunk_chars=max_chunk_chars,
        include_simple_numbered_sections=False,
    )
    return len(page_texts), chunks


def _parse_page_marked_text(text: str) -> list[str]:
    """Parse a transcription containing ordered ``[PAGE n]`` markers."""

    pages: list[str] = []
    current_lines: list[str] | None = None
    expected_page = 1

    for line in text.splitlines():
        marker = _PAGE_MARKER_RE.fullmatch(line.strip())
        if marker:
            page_number = int(marker.group(1))
            if page_number != expected_page:
                raise ValueError(
                    f"expected [PAGE {expected_page}], found [PAGE {page_number}]"
                )
            if current_lines is not None:
                pages.append("\n".join(current_lines).strip())
            current_lines = []
            expected_page += 1
        elif current_lines is not None:
            current_lines.append(line)

    if current_lines is None:
        raise ValueError("text transcription contains no [PAGE n] markers")
    pages.append("\n".join(current_lines).strip())
    return pages


def ingest_page_marked_text(
    text_path: str | Path,
    metadata: ProcurementDocumentMetadata,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> tuple[int, list[DocumentChunk]]:
    """Ingest a UTF-8 transcription while preserving its source-page markers."""

    path = Path(text_path)
    page_texts = _parse_page_marked_text(path.read_text(encoding="utf-8"))
    chunks = _chunks_from_pages(
        page_texts,
        metadata,
        source_path=path,
        max_chunk_chars=max_chunk_chars,
        include_simple_numbered_sections=False,
    )
    return len(page_texts), chunks


def santa_monica_metadata(pdf_path: str | Path) -> ProcurementDocumentMetadata:
    """Return source metadata for the Santa Monica emergency declaration."""

    return ProcurementDocumentMetadata(
        document_id="SM-PALISADES-FIRE-EMERGENCY-2025",
        title="Amended Local Emergency - 2025 Palisades Fire",
        jurisdiction="California",
        agency="City of Santa Monica",
        document_type="emergency_declaration",
        effective_date=date(2025, 1, 10),
        authority_level="local_executive_order",
        exception_type="emergency",
        source_path=str(pdf_path),
    )


def santa_monica_ordinance_metadata(
    text_path: str | Path,
) -> ProcurementDocumentMetadata:
    """Return source metadata for Santa Monica Ordinance 2849 (CCS)."""

    return ProcurementDocumentMetadata(
        document_id="SM-ORD-2849-2026",
        title=(
            "Ordinance 2849 (CCS) - Modernized Procurement Policies, "
            "Practices and Procedures"
        ),
        jurisdiction="California",
        agency="City of Santa Monica",
        document_type="ordinance",
        effective_date=date(2026, 4, 9),
        authority_level="local_law",
        source_path=str(text_path),
    )


def santa_monica_municipal_code_metadata(
    docx_path: str | Path,
) -> ProcurementDocumentMetadata:
    """Return source metadata for Santa Monica Municipal Code Chapter 2.24."""

    return ProcurementDocumentMetadata(
        document_id="SM-MUNICIPAL-CODE-2.24",
        title="Santa Monica Municipal Code Chapter 2.24 - Purchasing System",
        jurisdiction="California",
        agency="City of Santa Monica",
        document_type="municipal_code",
        authority_level="local_law",
        source_path=str(docx_path),
    )


def santa_monica_bidding_thresholds_metadata(
    docx_path: str | Path,
) -> ProcurementDocumentMetadata:
    """Return metadata for Santa Monica's procurement threshold guidance."""

    return ProcurementDocumentMetadata(
        document_id="SM-BIDDING-THRESHOLDS",
        title="City of Santa Monica Bidding Thresholds",
        jurisdiction="Santa Monica, California",
        agency="City of Santa Monica",
        document_type="procurement_policy",
        subject="procurement classification and solicitation thresholds",
        authority_level="procurement_policy",
        source_path=str(docx_path),
    )


def california_pcc_1102_metadata(
    pdf_path: str | Path,
) -> ProcurementDocumentMetadata:
    """Return statewide metadata for California Public Contract Code § 1102."""

    return ProcurementDocumentMetadata(
        document_id="CA-PCC-1102",
        title="California Public Contract Code § 1102 - Emergency Definition",
        jurisdiction="California",
        agency="State of California",
        document_type="statute",
        effective_date=date(1995, 1, 1),
        authority_level="statute",
        exception_type="emergency",
        section="1102",
        source_path=str(pdf_path),
    )


def main() -> None:
    """Ingest the document and print a compact chunk preview."""

    parser = argparse.ArgumentParser(
        description="Preview page-aware chunks from the Santa Monica policy PDF."
    )
    parser.add_argument("pdf_path", nargs="?", type=Path, default=DEFAULT_PDF_PATH)
    args = parser.parse_args()

    page_count, chunks = ingest_pdf(
        args.pdf_path,
        santa_monica_metadata(args.pdf_path),
    )
    print(f"Pages: {page_count}")
    print(f"Chunks: {len(chunks)}")
    for chunk in chunks:
        preview = re.sub(r"\s+", " ", chunk.text)[:200]
        page_label = str(chunk.metadata.page)
        if chunk.metadata.page_end != chunk.metadata.page:
            page_label = f"{chunk.metadata.page}-{chunk.metadata.page_end}"
        print(
            f"{chunk.chunk_id} | pages {page_label} | {preview}"
        )


if __name__ == "__main__":
    main()
