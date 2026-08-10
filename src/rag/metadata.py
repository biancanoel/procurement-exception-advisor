"""Metadata models for procurement-policy retrieval."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProcurementDocumentMetadata(BaseModel):
    """Filterable source metadata attached to a policy document or chunk.

    Section and page identify a location within the source and are optional so
    the same model can represent both a complete document and a retrieved
    document chunk.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    jurisdiction: str = Field(min_length=1)
    agency: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    effective_date: date | None = None
    authority_level: str = Field(min_length=1)
    exception_type: str | None = Field(default=None, min_length=1)
    section: str | None = Field(default=None, min_length=1)
    page: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    source_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def page_span_must_be_ordered(self) -> ProcurementDocumentMetadata:
        """Reject an ending page that precedes the starting page."""

        if self.page_end is not None and self.page is None:
            raise ValueError("page is required when page_end is provided")
        if (
            self.page is not None
            and self.page_end is not None
            and self.page_end < self.page
        ):
            raise ValueError("page_end cannot precede page")
        return self


class DocumentChunk(BaseModel):
    """A retrievable unit of text with its source metadata."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: ProcurementDocumentMetadata
