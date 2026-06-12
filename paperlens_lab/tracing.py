from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRACE_PATH = Path("outputs") / "agent_traces.jsonl"


@dataclass
class TraceRecord:
    schema_version: str
    trace_id: str
    task: str
    provider: str
    model: str
    status: str
    latency_ms: int
    input: dict[str, Any]
    output: dict[str, Any]
    error: str | None = None
    created_at: float = field(default_factory=time.time)


def new_trace_id(prefix: str = "tr") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def trace_path() -> Path:
    return Path(os.getenv("PAPERLENS_TRACE_PATH", str(DEFAULT_TRACE_PATH)))


def tracing_enabled() -> bool:
    return os.getenv("PAPERLENS_TRACE_ENABLED", "1").lower() not in {"0", "false", "no"}


def trace_content_enabled() -> bool:
    return os.getenv("PAPERLENS_TRACE_CONTENT", "0").lower() in {"1", "true", "yes"}


def write_trace(record: TraceRecord) -> None:
    if not tracing_enabled():
        return
    try:
        path = trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        return


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not _looks_secret(str(k))}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _looks_secret(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in ("token", "secret", "password", "authorization", "api_key"))
