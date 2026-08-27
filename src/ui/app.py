"""Minimal Gradio interface for submitting emergency procurement cases."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from graph.workflow import run_graph, stream_graph
from models.cases import EmergencyCaseInput, EvidenceDocument


def _text_from_user_history(history: list[dict[str, Any]]) -> list[str]:
    """Collect prior user-authored text without treating agent output as facts."""

    text_values: list[str] = []
    for item in history:
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            text_values.append(content.strip())
        elif isinstance(content, list):
            text_values.extend(
                str(block["text"]).strip()
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and str(block.get("text", "")).strip()
            )
    return text_values


def _file_path(upload: Any) -> str | None:
    """Read the temporary path from Gradio's supported upload shapes."""

    if isinstance(upload, str):
        return upload
    if isinstance(upload, dict):
        path = upload.get("path")
        return str(path) if path else None
    path = getattr(upload, "path", None)
    return str(path) if path else None


def _original_filename(upload: Any, path: str) -> str:
    """Preserve Gradio's original upload name with a path-name fallback."""

    candidate: Any = None
    if isinstance(upload, dict):
        candidate = upload.get("orig_name") or upload.get("name")
    else:
        candidate = getattr(upload, "orig_name", None)
    return Path(str(candidate or path)).name


def _files_from_user_history(history: list[dict[str, Any]]) -> list[Any]:
    """Collect earlier user attachments so follow-up turns retain them."""

    files: list[Any] = []
    for item in history:
        if item.get("role") != "user":
            continue
        content = item.get("content")
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "file":
                continue
            files.append(block.get("file", block))
    return files


def _case_evidence(files: list[Any]) -> list[EvidenceDocument]:
    """Read uploaded UTF-8 TXT files into structured, case-local evidence."""

    documents: list[EvidenceDocument] = []
    seen_paths: set[str] = set()
    for upload in files:
        path = _file_path(upload)
        if path is None:
            raise ValueError("An uploaded evidence file has no readable path")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        filename = _original_filename(upload, path)
        if Path(filename).suffix.lower() != ".txt":
            raise ValueError(
                f"Unsupported evidence file '{filename}'. "
                "Only .txt files are accepted."
            )
        try:
            extracted_text = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Evidence file '{filename}' is not valid UTF-8 text."
            ) from error
        except OSError as error:
            raise ValueError(
                f"Could not read evidence file '{filename}': {error}"
            ) from error
        if not extracted_text.strip():
            raise ValueError(f"Evidence file '{filename}' is empty.")
        documents.append(
            EvidenceDocument(
                evidence_id=f"CASE-D{len(documents) + 1:02d}",
                filename=filename,
                file_type="txt",
                extracted_text=extracted_text,
            )
        )
    return documents


def respond_to_message(
    message: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    """Evaluate accumulated user text and uploaded TXT case evidence."""

    current_text = str(message.get("text", "")).strip()
    prior_text = _text_from_user_history(history)
    description_parts = [*prior_text, current_text] if current_text else prior_text
    if not description_parts:
        raise ValueError("Emergency description must not be blank")

    files = [
        *_files_from_user_history(history),
        *list(message.get("files") or []),
    ]
    case_input = EmergencyCaseInput(
        description="\n\n".join(description_parts),
        case_evidence=_case_evidence(files),
    )
    response = run_graph(
        "Evaluate the supplied emergency procurement situation.",
        case_input=case_input,
    )
    return str(response.content)


def stream_response_to_message(
    message: dict[str, Any],
    history: list[dict[str, Any]],
) -> Iterator[str]:
    """Stream concise workflow progress before yielding the final assessment."""

    current_text = str(message.get("text", "")).strip()
    prior_text = _text_from_user_history(history)
    description_parts = [*prior_text, current_text] if current_text else prior_text
    if not description_parts:
        raise ValueError("Emergency description must not be blank")

    files = [
        *_files_from_user_history(history),
        *list(message.get("files") or []),
    ]
    case_input = EmergencyCaseInput(
        description="\n\n".join(description_parts),
        case_evidence=_case_evidence(files),
    )

    yield (
        "Review started — verifying whether the situation qualifies as an "
        "emergency procurement…"
    )
    for node_name, update in stream_graph(
        "Evaluate the supplied emergency procurement situation.",
        case_input=case_input,
    ):
        if node_name == "emergency_verification_subagent":
            verification = update.get("emergency_verification")
            if (
                verification is not None
                and verification.emergency_is_verified is True
            ):
                yield (
                    "Emergency verified — determining the applicable "
                    "procurement context…"
                )
            else:
                yield (
                    "Emergency verification complete — preparing the "
                    "assessment…"
                )
        elif node_name == "procurement_context_subagent":
            yield (
                "Procurement context established — reviewing whether the "
                "procurement file is audit-ready…"
            )
        elif node_name == "audit_readiness_subagent":
            yield "Audit-readiness review complete — preparing the assessment…"
        elif node_name == "finalize_assessment":
            messages = update.get("messages", [])
            if not messages:
                raise RuntimeError("graph finalization returned no response")
            yield str(messages[-1].content)


def evaluate_description(description: str) -> str:
    """Preserve the original text-only handler for programmatic callers."""

    return respond_to_message(
        {"text": description, "files": []},
        [],
    )


def build_app() -> gr.Blocks:
    """Build the multimodal Procurement Exception Advisor chat UI."""

    with gr.Blocks(title="Procurement Exception Advisor") as app:
        gr.Markdown("# Procurement Exception Advisor")
        gr.Markdown(
            "Describe the emergency situation and optionally attach relevant "
            "TXT evidence files. Attached filenames appear in your message so "
            "you can confirm what will be evaluated. Unknown details can be "
            "left out and will be identified during the assessment."
        )
        gr.ChatInterface(
            fn=stream_response_to_message,
            multimodal=True,
            chatbot=gr.Chatbot(
                label="Procurement advisor",
                height=600,
            ),
            textbox=gr.MultimodalTextbox(
                file_count="multiple",
                file_types=[".txt"],
                sources=["upload"],
                placeholder=(
                    "Describe the emergency and attach any available TXT files."
                ),
            ),
        )

    return app


def main() -> None:
    """Load local environment settings and launch the Gradio application."""

    load_dotenv()
    build_app().launch(show_error=True)


if __name__ == "__main__":
    main()
