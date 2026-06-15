from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_PATH = Path("outputs") / "paper_memory.jsonl"


def memory_path() -> Path:
    return Path(os.getenv("PAPERLENS_MEMORY_PATH", str(DEFAULT_MEMORY_PATH)))


def paper_key(title_or_source: str) -> str:
    normalized = " ".join((title_or_source or "untitled-paper").lower().split())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"paper:{digest}"


def append_memory(
    paper_id: str,
    *,
    kind: str,
    payload: dict[str, Any],
    evidence_id: str | None = None,
) -> dict[str, Any]:
    record = {
        "id": evidence_id or f"{kind}:{uuid.uuid4().hex[:10]}",
        "paper_id": paper_id,
        "kind": kind,
        "payload": payload,
        "created_at": time.time(),
    }
    path = memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def load_memories(paper_id: str, *, limit: int = 80) -> list[dict[str, Any]]:
    path = memory_path()
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("paper_id") != paper_id:
                continue
            records.append(_as_prompt_memory(record))
    return records[-limit:]


def _as_prompt_memory(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    summary = payload.get("summary")
    if not summary and "idea" in payload:
        idea = payload.get("idea")
        summary = json.dumps(idea, ensure_ascii=False) if isinstance(idea, dict) else str(idea)
    return {
        "id": record.get("id", ""),
        "kind": record.get("kind", ""),
        "summary": str(summary or "")[:1200],
    }
