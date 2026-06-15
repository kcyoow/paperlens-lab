from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_INDEX_DIR = Path("outputs") / "source_index"
DEFAULT_TRANSLATION_CACHE_DIR = Path("outputs") / "translation_cache"


def source_index_dir() -> Path:
    return Path(os.getenv("PAPERLENS_SOURCE_INDEX_DIR", str(DEFAULT_SOURCE_INDEX_DIR)))


def translation_cache_dir() -> Path:
    return Path(os.getenv("PAPERLENS_TRANSLATION_CACHE_DIR", str(DEFAULT_TRANSLATION_CACHE_DIR)))


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def save_source_index(
    paper_id: str,
    *,
    title: str,
    source_label: str,
    pdf_url: str,
    source_text: str,
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    spans = []
    for section in sections:
        for paragraph in section.get("paragraphs", []):
            for span in paragraph.get("spans", []):
                original = str(span.get("original", ""))
                spans.append(
                    {
                        "span_id": span.get("id", ""),
                        "section_id": section.get("id", ""),
                        "section_title": section.get("title", ""),
                        "paragraph_id": paragraph.get("id", ""),
                        "position": len(spans),
                        "text": original,
                        "text_hash": text_hash(original),
                    }
                )
    record = {
        "paper_id": paper_id,
        "title": title,
        "source_label": source_label,
        "pdf_url": pdf_url,
        "source_text_hash": text_hash(source_text),
        "source_text_chars": len(source_text),
        "spans": spans,
        "created_at": time.time(),
    }
    path = _source_index_path(paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return record


def load_source_index(paper_id: str) -> dict[str, Any] | None:
    path = _source_index_path(paper_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def evidence_window(paper_id: str, span_id: str, *, radius: int = 3) -> dict[str, Any] | None:
    record = load_source_index(paper_id)
    if not record:
        return None
    spans = record.get("spans", [])
    index = next((idx for idx, span in enumerate(spans) if span.get("span_id") == span_id), None)
    if index is None:
        return None
    start = max(0, index - radius)
    end = min(len(spans), index + radius + 1)
    window_spans = spans[start:end]
    return {
        "paper_id": paper_id,
        "span_id": span_id,
        "span_range": f"{window_spans[0]['span_id']}-{window_spans[-1]['span_id']}" if window_spans else span_id,
        "source_hash": record.get("source_text_hash", ""),
        "text": " ".join(span.get("text", "") for span in window_spans),
        "spans": window_spans,
    }


def get_span_text(paper_id: str, span_id: str) -> str:
    record = load_source_index(paper_id)
    if not record:
        return ""
    for span in record.get("spans", []):
        if span.get("span_id") == span_id:
            return str(span.get("text", ""))
    return ""


def get_cached_translation(
    paper_id: str,
    span_id: str,
    source_text: str,
    *,
    locale: str,
    model: str,
) -> str:
    path = _translation_cache_path(paper_id, span_id, source_text, locale=locale, model=model)
    if not path.exists():
        return ""
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(body.get("translation", ""))


def save_cached_translation(
    paper_id: str,
    span_id: str,
    source_text: str,
    translation: str,
    *,
    locale: str,
    model: str,
) -> None:
    path = _translation_cache_path(paper_id, span_id, source_text, locale=locale, model=model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "span_id": span_id,
                "source_hash": text_hash(source_text),
                "locale": locale,
                "model": model,
                "translation": translation,
                "created_at": time.time(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _source_index_path(paper_id: str) -> Path:
    return source_index_dir() / f"{_safe_key(paper_id)}.json"


def _translation_cache_path(
    paper_id: str,
    span_id: str,
    source_text: str,
    *,
    locale: str,
    model: str,
) -> Path:
    key = "__".join(_safe_key(item) for item in (paper_id, span_id, locale, model, text_hash(source_text)))
    return translation_cache_dir() / f"{key}.json"


def _safe_key(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)[:120]
