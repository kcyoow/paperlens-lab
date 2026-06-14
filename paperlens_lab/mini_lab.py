from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .ingest import clean_text
from .scenario_eval import run_starter_code, source_contains_quote
from .source_index import evidence_window, get_span_text, text_hash


DEFAULT_PROVIDER = "local"
MODAL_SCRIPT = Path(__file__).with_name("modal_minilab_app.py")


class MiniLabError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def mini_lab_provider() -> str:
    provider = os.getenv("PAPERLENS_MINILAB_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    return provider if provider in {"local", "modal"} else DEFAULT_PROVIDER


def run_mini_lab_job(
    *,
    code: str,
    paper_id: str,
    paper_title: str,
    span_id: str,
    selected_span: str,
    provider: str | None = None,
) -> dict[str, Any]:
    job = _build_job_payload(
        code=code,
        paper_id=paper_id,
        paper_title=paper_title,
        span_id=span_id,
        selected_span=selected_span,
    )
    requested_provider = (provider or mini_lab_provider()).strip().lower()
    if requested_provider == "modal":
        raw_result = _run_modal_mini_lab_with_cli(job)
    else:
        raw_result = _run_local_mini_lab(job)
    return _validated_result(raw_result, job, requested_provider=requested_provider)


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _build_job_payload(
    *,
    code: str,
    paper_id: str,
    paper_title: str,
    span_id: str,
    selected_span: str,
) -> dict[str, Any]:
    cleaned_code = code or ""
    if not cleaned_code.strip():
        raise MiniLabError("Starter code is required.")
    if not span_id:
        raise MiniLabError("Selected span id is required.")

    indexed_span = get_span_text(paper_id, span_id) if paper_id else ""
    if paper_id and not indexed_span:
        raise MiniLabError("Selected span was not found in the paper index.", status_code=404)

    source_span = clean_text(indexed_span or selected_span)
    selected_clean = clean_text(selected_span)
    if not source_span:
        raise MiniLabError("Selected source span is required.")
    if selected_clean and selected_clean != source_span:
        if source_contains_quote(source_span, selected_clean):
            source_span = selected_clean
        else:
            raise MiniLabError("Selected span does not match the paper index.")

    selected_hash = text_hash(source_span)
    evidence_rows = build_evidence_rows(
        paper_id=paper_id,
        span_id=span_id,
        selected_span=source_span,
    )
    if not evidence_rows:
        raise MiniLabError("Mini-lab requires indexed paper evidence rows.", status_code=404)
    evidence_hash = text_hash(json.dumps(evidence_rows, ensure_ascii=False, sort_keys=True))
    return {
        "code": cleaned_code,
        "paperId": paper_id,
        "paperTitle": paper_title or "Untitled paper",
        "spanId": span_id,
        "selectedSpan": source_span,
        "sourceHash": selected_hash,
        "selectedSpanHash": selected_hash,
        "codeHash": code_hash(cleaned_code),
        "sourceIndexBound": bool(indexed_span),
        "evidenceRows": evidence_rows,
        "evidenceRowCount": len(evidence_rows),
        "evidenceHash": evidence_hash,
        "createdAt": time.time(),
    }


def _run_local_mini_lab(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    execution = run_starter_code(
        str(job.get("code", "")),
        evidence_rows=list(job.get("evidenceRows") or []),
        require_evidence_rows=True,
    )
    return {
        "provider": "local",
        "executionMode": "local-source-bound-subprocess",
        "runner": "paperlens-local-minilab",
        "paperId": job.get("paperId", ""),
        "paperTitle": job.get("paperTitle", ""),
        "spanId": job.get("spanId", ""),
        "sourceHash": job.get("sourceHash", ""),
        "selectedSpanHash": job.get("selectedSpanHash", ""),
        "codeHash": job.get("codeHash", ""),
        "sourceIndexBound": bool(job.get("sourceIndexBound")),
        "evidenceRowCount": int(job.get("evidenceRowCount") or 0),
        "evidenceHash": job.get("evidenceHash", ""),
        "passed": bool(execution.get("passed")),
        "reasons": list(execution.get("reasons") or []),
        "rows": execution.get("rows", []),
        "logs": [
            "local mini-lab runner executed generated code with indexed paper evidence rows",
            f"rows={len(execution.get('rows') or [])}",
            f"evidence_rows={len(job.get('evidenceRows') or [])}",
        ],
        "durationMs": int((time.time() - started) * 1000),
    }


def _run_modal_mini_lab_with_cli(job: dict[str, Any]) -> dict[str, Any]:
    modal_bin = _modal_bin()
    if not modal_bin:
        return _failed_modal_result(job, "Modal CLI was not found. Set PAPERLENS_MODAL_BIN or install modal.")

    timeout = _modal_timeout_seconds()
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="paperlens-modal-minilab-") as tmp:
        tmp_path = Path(tmp)
        payload_path = tmp_path / "payload.json"
        result_path = tmp_path / "result.json"
        payload_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")

        command = [
            modal_bin,
            "run",
            "-q",
            "--write-result",
            str(result_path),
            str(MODAL_SCRIPT),
            "--payload-path",
            str(payload_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return _failed_modal_result(job, f"Modal mini-lab timed out after {timeout}s.")

        logs = _trim_logs(completed.stdout, completed.stderr)
        if completed.returncode != 0:
            detail = logs[-1] if logs else "no Modal output"
            return _failed_modal_result(job, f"Modal CLI exited with {completed.returncode}: {detail}", logs=logs)
        if not result_path.exists():
            return _failed_modal_result(job, "Modal CLI did not write a result payload.", logs=logs)
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _failed_modal_result(job, f"Modal result payload was not JSON: {exc}", logs=logs)
        if not isinstance(result, dict):
            return _failed_modal_result(job, "Modal result payload was not an object.", logs=logs)
        result.setdefault("logs", [])
        result["logs"] = [*logs, *list(result.get("logs") or [])]
        result["durationMs"] = int((time.time() - started) * 1000)
        return result


def _validated_result(
    result: dict[str, Any],
    job: dict[str, Any],
    *,
    requested_provider: str,
) -> dict[str, Any]:
    reasons = list(result.get("reasons") or [])
    validation = {
        "paperIdMatches": result.get("paperId") == job.get("paperId"),
        "spanIdMatches": result.get("spanId") == job.get("spanId"),
        "sourceHashMatches": result.get("sourceHash") == job.get("sourceHash"),
        "selectedSpanHashMatches": result.get("selectedSpanHash") == job.get("selectedSpanHash"),
        "codeHashMatches": result.get("codeHash") == job.get("codeHash"),
        "evidenceHashMatches": result.get("evidenceHash") == job.get("evidenceHash"),
        "sourceIndexBound": bool(job.get("sourceIndexBound")),
        "providerMatches": result.get("provider") == requested_provider,
    }
    for key, passed in validation.items():
        if key == "sourceIndexBound":
            continue
        if not passed:
            reasons.append(f"mini-lab validation failed: {key}")

    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    if not rows:
        reasons.append("mini-lab returned no rows")
    evidence_rows = list(job.get("evidenceRows") or [])
    evidence_by_id = {
        str(row.get("source_id") or ""): str(row.get("text_hash") or "")
        for row in evidence_rows
        if isinstance(row, dict) and str(row.get("source_id") or "")
    }
    selected_ids = {
        str(row.get("source_id") or "")
        for row in evidence_rows
        if isinstance(row, dict) and (str(row.get("label") or "") == "selected" or row.get("gold") is True)
    }
    selected_seen = False
    if evidence_rows and not result.get("evidenceHash"):
        reasons.append("mini-lab result did not echo the evidence hash")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            reasons.append(f"mini-lab row {index} is not an object")
            continue
        for key in ("baseline_score", "prototype_score", "metric", "failure_condition"):
            if key not in row:
                reasons.append(f"mini-lab row {index} missing {key}")
        if evidence_by_id:
            source_id = str(row.get("source_id") or "")
            if not source_id:
                reasons.append(f"mini-lab row {index} missing source_id")
            elif source_id not in evidence_by_id:
                reasons.append(f"mini-lab row {index} source_id is outside indexed paper evidence")
            else:
                if source_id in selected_ids:
                    selected_seen = True
                row_hash = str(row.get("text_hash") or "")
                if not row_hash:
                    reasons.append(f"mini-lab row {index} missing text_hash")
                elif evidence_by_id[source_id] and row_hash != evidence_by_id[source_id]:
                    reasons.append(f"mini-lab row {index} text_hash does not match indexed paper evidence")
    if selected_ids and not selected_seen:
        reasons.append("mini-lab rows must include the selected paper evidence row")

    claim_comparison = _claim_comparison(rows)
    if reasons:
        claim_comparison = {
            **claim_comparison,
            "verdict": "inconclusive",
            "limitations": [
                *claim_comparison.get("limitations", []),
                "The mini-lab result did not satisfy evidence-row binding.",
            ],
        }
    return {
        "passed": bool(result.get("passed")) and not reasons,
        "reasons": reasons,
        "rows": rows,
        "logs": list(result.get("logs") or []),
        "provider": result.get("provider") or requested_provider,
        "executionMode": result.get("executionMode", ""),
        "runner": result.get("runner", ""),
        "paperId": job.get("paperId", ""),
        "paperTitle": job.get("paperTitle", ""),
        "spanId": job.get("spanId", ""),
        "sourceHash": job.get("sourceHash", ""),
        "selectedSpanHash": job.get("selectedSpanHash", ""),
        "codeHash": job.get("codeHash", ""),
        "sourceIndexBound": bool(job.get("sourceIndexBound")),
        "evidenceRowCount": int(job.get("evidenceRowCount") or 0),
        "evidenceHash": job.get("evidenceHash", ""),
        "validation": validation,
        "claimComparison": claim_comparison,
        "durationMs": int(result.get("durationMs") or 0),
    }


def build_evidence_rows(*, paper_id: str, span_id: str, selected_span: str) -> list[dict[str, Any]]:
    window = evidence_window(paper_id, span_id, radius=4) if paper_id and span_id else None
    if not window:
        return []
    rows: list[dict[str, Any]] = []
    for item in window.get("spans", []):
        source_id = str(item.get("span_id") or "")
        text = clean_text(str(item.get("text") or ""))
        if not source_id or not text:
            continue
        label = "selected" if source_id == span_id else "context_control"
        rows.append(
            {
                "source_id": source_id,
                "text": text,
                "text_hash": str(item.get("text_hash") or text_hash(text)),
                "label": label,
                "gold": label == "selected",
                "query": selected_span,
            }
        )
    return rows


def _claim_comparison(rows: list[Any]) -> dict[str, Any]:
    valid_rows = [row for row in rows if isinstance(row, dict)]
    improved = 0
    failed = 0
    comparable = 0
    metrics: list[str] = []
    for row in valid_rows:
        baseline = row.get("baseline_score")
        prototype = row.get("prototype_score")
        if isinstance(row.get("metric"), str) and row.get("metric"):
            metrics.append(str(row["metric"]))
        if isinstance(baseline, (int, float)) and isinstance(prototype, (int, float)):
            comparable += 1
            if prototype > baseline:
                improved += 1
        if row.get("failure_condition") is True:
            failed += 1
    if comparable == 0:
        verdict = "inconclusive"
    elif improved > 0 and failed == 0:
        verdict = "directionally_supports"
    elif failed > 0:
        verdict = "mixed_or_not_supported"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "improvedRows": improved,
        "failedRows": failed,
        "comparableRows": comparable,
        "metrics": list(dict.fromkeys(metrics)),
        "limitations": [
            "This mini-lab compares a source-bound run against the generated baseline.",
            "A directional result is not a reproduction of the full paper claim.",
        ],
    }


def _failed_modal_result(job: dict[str, Any], reason: str, *, logs: list[str] | None = None) -> dict[str, Any]:
    return {
        "provider": "modal",
        "executionMode": "modal-cli",
        "runner": "paperlens-modal-minilab",
        "paperId": job.get("paperId", ""),
        "paperTitle": job.get("paperTitle", ""),
        "spanId": job.get("spanId", ""),
        "sourceHash": job.get("sourceHash", ""),
        "selectedSpanHash": job.get("selectedSpanHash", ""),
        "codeHash": job.get("codeHash", ""),
        "sourceIndexBound": bool(job.get("sourceIndexBound")),
        "evidenceRowCount": int(job.get("evidenceRowCount") or 0),
        "evidenceHash": job.get("evidenceHash", ""),
        "passed": False,
        "reasons": [reason],
        "rows": [],
        "logs": logs or [],
        "durationMs": 0,
    }


def _modal_bin() -> str:
    configured = os.getenv("PAPERLENS_MODAL_BIN", "").strip()
    if configured:
        return configured if Path(configured).exists() or shutil.which(configured) else ""
    found = shutil.which("modal")
    if found:
        return found
    anaconda_modal = Path("/opt/anaconda3/bin/modal")
    return str(anaconda_modal) if anaconda_modal.exists() else ""


def _modal_timeout_seconds() -> float:
    raw = os.getenv("PAPERLENS_MODAL_MINILAB_TIMEOUT", "180")
    try:
        value = float(raw)
    except ValueError:
        value = 180.0
    return max(30.0, min(value, 600.0))


def _trim_logs(stdout: str, stderr: str) -> list[str]:
    lines = []
    for stream in (stdout, stderr):
        for line in stream.splitlines():
            cleaned = line.strip()
            if cleaned:
                lines.append(cleaned)
    return lines[-20:]
