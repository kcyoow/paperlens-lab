from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_VALIDATION_ROOT = Path("outputs") / "service_demo_validation"


def validation_root() -> Path:
    return Path(os.getenv("PAPERLENS_VALIDATION_ROOT", str(DEFAULT_VALIDATION_ROOT)))


def build_validation_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or validation_root()
    warnings: list[str] = []
    if not root.exists():
        return {
            "ok": False,
            "validationRoot": str(root),
            "warnings": [f"validation root not found: {root}"],
            "realPaperRun": None,
            "modelTraces": None,
            "localDemo": None,
            "memory": None,
        }

    real_paper_run = _best_real_paper_run(root, warnings)
    trace_summary = _trace_summary_for_run(root, real_paper_run, warnings)
    local_demo = _local_demo_summary(root, warnings)
    memory_summary = _memory_summary_for_run(root, real_paper_run, warnings)

    ok = bool(
        real_paper_run
        and real_paper_run.get("passed")
        and real_paper_run.get("paperCount", 0) >= 3
        and real_paper_run.get("evaluationTotal", 0) > 0
        and real_paper_run.get("evaluationPassed") == real_paper_run.get("evaluationTotal")
        and real_paper_run.get("evidenceConsistencyPassed")
        and _adversarial_litm_passed(real_paper_run)
        and trace_summary
        and trace_summary.get("total", 0) > 0
        and trace_summary.get("modelCount") == trace_summary.get("total")
        and trace_summary.get("fallbackCount") == 0
        and trace_summary.get("errorCount") == 0
        and local_demo
        and local_demo.get("sourceIndexConsistent", True)
        and not local_demo.get("usedFallback")
        and not local_demo.get("translationUsedFallback")
    )
    return {
        "ok": ok,
        "validationRoot": str(root),
        "warnings": warnings,
        "realPaperRun": real_paper_run,
        "modelTraces": trace_summary,
        "localDemo": local_demo,
        "memory": memory_summary,
    }


def _best_real_paper_run(root: Path, warnings: list[str]) -> dict[str, Any] | None:
    candidates = []
    for path in root.rglob("summary.json"):
        body = _read_json(path)
        if not isinstance(body, dict):
            continue
        paper_count = int(body.get("paper_count") or len(body.get("runs", [])) or 0)
        evidence_issues = _real_paper_evidence_issues(body)
        candidates.append((path.stat().st_mtime, paper_count, path, body, evidence_issues))
    if not candidates:
        warnings.append("no real-paper summary.json found")
        return None

    _, _, path, body, evidence_issues = max(candidates, key=lambda item: item[0])
    if evidence_issues:
        warnings.append(
            f"real-paper summary evidence consistency needs rerun: {len(evidence_issues)} issue(s)"
        )
    runs = body.get("runs", []) if isinstance(body.get("runs"), list) else []
    papers = []
    evaluation_total = 0
    evaluation_passed = 0
    for run in runs:
        evaluations = run.get("evaluations", []) if isinstance(run, dict) else []
        source = run.get("source", {}) if isinstance(run, dict) else {}
        reader = run.get("reader", {}) if isinstance(run, dict) else {}
        memory = run.get("memory", {}) if isinstance(run, dict) else {}
        passed_here = sum(1 for item in evaluations if item.get("passed"))
        evaluation_total += len(evaluations)
        evaluation_passed += passed_here
        papers.append(
            {
                "name": (run.get("case") or {}).get("name", ""),
                "arxiv": (run.get("case") or {}).get("arxiv", ""),
                "title": source.get("title", ""),
                "pdfUrl": source.get("pdf_url", ""),
                "pageMarkers": source.get("page_marker_count", 0),
                "sourceTextChars": source.get("text_chars", 0),
                "wordCount": source.get("word_count", 0),
                "totalSentenceCount": (reader.get("metadata") or {}).get("totalSentenceCount", 0),
                "readerSpanLimit": (reader.get("metadata") or {}).get("readerSpanLimit", 0),
                "translatedSpanCount": (reader.get("metadata") or {}).get("translatedSpanCount", 0),
                "readerSpans": reader.get("visible_span_count", 0),
                "adversarialLitm": reader.get("adversarial_litm", {}),
                "selectedSpanPositions": reader.get("selected_span_positions", []),
                "evaluationsPassed": passed_here,
                "evaluationsTotal": len(evaluations),
                "evaluations": [
                    {
                        "name": item.get("name", ""),
                        "passed": bool(item.get("passed")),
                        "reasons": item.get("reasons", []),
                    }
                    for item in evaluations
                ],
                "memoryRecordsAfterGrowth": memory.get("records_after_growth", 0),
            }
        )

    fine_tuning = body.get("fine_tuning") if isinstance(body.get("fine_tuning"), dict) else {}
    return {
        "summaryPath": str(path),
        "runName": path.parent.name,
        "artifactDate": path.parent.parent.name,
        "passed": bool(body.get("passed")),
        "paperCount": int(body.get("paper_count") or len(papers)),
        "evaluationPassed": evaluation_passed,
        "evaluationTotal": evaluation_total,
        "evidenceConsistencyPassed": not evidence_issues,
        "evidenceConsistencyIssues": evidence_issues[:12],
        "fineTuningRecommendation": fine_tuning.get("recommendation", "unknown"),
        "fineTuningReason": fine_tuning.get("reason", ""),
        "repeatedFailures": fine_tuning.get("repeated_failures", []),
        "papers": papers,
    }


def _adversarial_litm_passed(real_paper_run: dict[str, Any]) -> bool:
    papers = real_paper_run.get("papers", []) if isinstance(real_paper_run.get("papers"), list) else []
    if not papers:
        return False
    for paper in papers:
        stats = paper.get("adversarialLitm") if isinstance(paper, dict) else {}
        evaluations = paper.get("evaluations", []) if isinstance(paper, dict) else []
        if not isinstance(stats, dict) or not stats:
            return False
        if int(stats.get("context_chars") or 0) < 8000:
            return False
        ratio = float(stats.get("target_char_offset_ratio") or 0)
        if ratio < 0.35 or ratio > 0.65:
            return False
        if not any(
            item.get("name") == "adversarial_lost_in_the_middle" and item.get("passed")
            for item in evaluations
            if isinstance(item, dict)
        ):
            return False
    return True


def _trace_summary_for_run(
    root: Path,
    real_paper_run: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any] | None:
    trace_paths = []
    if real_paper_run:
        summary_path = Path(real_paper_run.get("summaryPath", ""))
        if summary_path.name == "summary.json":
            candidate = summary_path.parent.parent / f"{summary_path.parent.name}_traces.jsonl"
            if candidate.exists():
                trace_paths.append(candidate)
    if not trace_paths:
        warnings.append("no matching trace JSONL found for the selected real-paper summary")
        return None

    task_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    error_count = 0
    total = 0
    for record in _read_jsonl(trace_paths[0]):
        if not isinstance(record, dict) or not record.get("task"):
            continue
        total += 1
        task_counts[str(record.get("task"))] += 1
        status_counts[str(record.get("status") or "unknown")] += 1
        provider_counts[str(record.get("provider") or "unknown")] += 1
        model_counts[str(record.get("model") or "unknown")] += 1
        if record.get("error"):
            error_count += 1

    return {
        "tracePath": str(trace_paths[0]),
        "total": total,
        "modelCount": status_counts.get("model", 0),
        "fallbackCount": status_counts.get("fallback", 0),
        "errorCount": error_count,
        "byTask": dict(sorted(task_counts.items())),
        "byProvider": dict(sorted(provider_counts.items())),
        "byModel": dict(sorted(model_counts.items())),
    }


def _local_demo_summary(root: Path, warnings: list[str]) -> dict[str, Any] | None:
    ask_paths = sorted(root.rglob("local_after_source_index_ask_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    translate_paths = sorted(
        root.rglob("local_after_source_index_translate_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    paper_paths = sorted(root.rglob("local_after_source_index_paper.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not ask_paths:
        warnings.append("no local selected-span Q&A proof found")
        return None

    ask = _read_json(ask_paths[0]) or {}
    translate = _read_json(translate_paths[0]) if translate_paths else {}
    paper = _read_json(paper_paths[0]) if paper_paths else {}
    evidence_window = ask.get("evidenceWindow") if isinstance(ask.get("evidenceWindow"), dict) else {}
    evidence = ask.get("evidence", []) if isinstance(ask.get("evidence"), list) else []
    metadata = paper.get("metadata", {}) if isinstance(paper, dict) else {}
    source_index = _source_index_for_local_demo(root, evidence_window)
    source_index_hash = source_index.get("source_text_hash", "") if source_index else ""
    source_index_chars = source_index.get("source_text_chars", 0) if source_index else 0
    source_index_consistent = True
    allowed_evidence_ids = {
        str(item.get("spanId", ""))
        for item in evidence_window.get("spans", [])
        if isinstance(item, dict) and item.get("spanId")
    }
    if evidence_window.get("spanId"):
        allowed_evidence_ids.add(str(evidence_window.get("spanId")))
    evidence_ids = [
        str(item.get("source_id", ""))
        for item in evidence
        if isinstance(item, dict) and item.get("source_id")
    ]
    unknown_evidence_ids = sorted(
        source_id for source_id in set(evidence_ids) if allowed_evidence_ids and source_id not in allowed_evidence_ids
    )
    if source_index_hash and evidence_window.get("sourceHash") and source_index_hash != evidence_window.get("sourceHash"):
        source_index_consistent = False
        warnings.append(
            "local selected-span source hash differs from current source index; rerun local browser/API proof for hash-bound evidence"
        )
    if source_index_chars and metadata.get("sourceTextChars") and source_index_chars != metadata.get("sourceTextChars"):
        source_index_consistent = False
        warnings.append(
            "local paper metadata source length differs from current source index; rerun local browser/API proof"
        )
    if unknown_evidence_ids:
        source_index_consistent = False
        warnings.append(
            "local selected-span answer cites evidence outside the source-index window; rerun local browser/API proof"
        )
    return {
        "askPath": str(ask_paths[0]),
        "translatePath": str(translate_paths[0]) if translate_paths else "",
        "paperPath": str(paper_paths[0]) if paper_paths else "",
        "paperTitle": paper.get("title", ""),
        "readerSpanCount": metadata.get("readerSpanCount", 0),
        "sourceTextChars": metadata.get("sourceTextChars", 0),
        "selectedSpanId": evidence_window.get("spanId", ""),
        "evidenceWindow": evidence_window.get("spanRange", ""),
        "sourceHash": evidence_window.get("sourceHash", ""),
        "sourceIndexHash": source_index_hash,
        "sourceIndexConsistent": source_index_consistent,
        "neighborSpans": evidence_window.get("spans", []),
        "evidenceIds": evidence_ids,
        "unknownEvidenceIds": unknown_evidence_ids,
        "quoteIdsWithinWindow": not unknown_evidence_ids,
        "quoteCount": len(evidence),
        "confidence": ask.get("confidence", ""),
        "needsMoreContext": bool(ask.get("needsMoreContext")),
        "provider": ask.get("provider", ""),
        "model": ask.get("model", ""),
        "traceId": ask.get("traceId", ""),
        "usedFallback": bool(ask.get("usedFallback")),
        "translationStatus": translate.get("status", "") if isinstance(translate, dict) else "",
        "translationTraceId": translate.get("traceId", "") if isinstance(translate, dict) else "",
        "translationUsedFallback": bool(translate.get("usedFallback")) if isinstance(translate, dict) else False,
    }


def _real_paper_evidence_issues(body: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    runs = body.get("runs", []) if isinstance(body.get("runs"), list) else []
    for run in runs:
        case_name = (run.get("case") or {}).get("name", "unknown") if isinstance(run, dict) else "unknown"
        qa_runs = ((run.get("model_outputs") or {}).get("qa") or []) if isinstance(run, dict) else []
        for qa in qa_runs:
            if not isinstance(qa, dict):
                continue
            span = qa.get("span") if isinstance(qa.get("span"), dict) else {}
            span_id = span.get("id", "unknown")
            source_evidence = qa.get("source_evidence") if isinstance(qa.get("source_evidence"), dict) else {}
            result = qa.get("result") if isinstance(qa.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
            if not source_evidence:
                issues.append(f"{case_name}:{span_id} missing source_evidence map")
                continue
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                source_id = str(item.get("source_id", ""))
                quote = str(item.get("quote", "")).strip()
                if source_id and source_id not in source_evidence:
                    issues.append(f"{case_name}:{span_id} cites unknown evidence {source_id}")
                    continue
                if quote and source_id and quote not in str(source_evidence.get(source_id, "")):
                    issues.append(f"{case_name}:{span_id} quote missing from {source_id}")
        adversarial = ((run.get("model_outputs") or {}).get("adversarial_litm") or {}) if isinstance(run, dict) else {}
        if isinstance(adversarial, dict) and adversarial:
            stats = adversarial.get("stats") if isinstance(adversarial.get("stats"), dict) else {}
            span_id = str(stats.get("target_span_id") or "unknown")
            source_evidence = (
                adversarial.get("source_evidence")
                if isinstance(adversarial.get("source_evidence"), dict)
                else {}
            )
            result = adversarial.get("result") if isinstance(adversarial.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
            if not source_evidence:
                issues.append(f"{case_name}:{span_id} missing adversarial source_evidence map")
                continue
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                source_id = str(item.get("source_id", ""))
                quote = str(item.get("quote", "")).strip()
                if source_id and source_id not in source_evidence:
                    issues.append(f"{case_name}:{span_id} adversarial cites unknown evidence {source_id}")
                    continue
                if quote and source_id and quote not in str(source_evidence.get(source_id, "")):
                    issues.append(f"{case_name}:{span_id} adversarial quote missing from {source_id}")
    return issues


def _memory_summary_for_run(
    root: Path,
    real_paper_run: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any] | None:
    memory_paths = []
    if real_paper_run:
        summary_path = Path(real_paper_run.get("summaryPath", ""))
        if summary_path.name == "summary.json":
            candidate = summary_path.parent.parent / f"{summary_path.parent.name}_memory.jsonl"
            if candidate.exists():
                memory_paths.append(candidate)
    if not memory_paths:
        memory_paths = sorted(root.rglob("*memory.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)[:1]
    if not memory_paths:
        warnings.append("no memory JSONL found")
        return None
    records = [record for record in _read_jsonl(memory_paths[0]) if isinstance(record, dict)]
    kinds = Counter(str(record.get("kind") or "unknown") for record in records)
    papers = Counter(str(record.get("paper_id") or "unknown") for record in records)
    return {
        "memoryPath": str(memory_paths[0]),
        "recordCount": len(records),
        "byKind": dict(sorted(kinds.items())),
        "paperCount": len(papers),
    }


def _source_index_for_local_demo(root: Path, evidence_window: dict[str, Any]) -> dict[str, Any] | None:
    paper_id = str(evidence_window.get("paperId") or "")
    if not paper_id:
        return None
    safe_name = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in paper_id)[:120]
    candidates = sorted(root.rglob(f"{safe_name}.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        body = _read_json(path)
        if body and body.get("paper_id") == paper_id and isinstance(body.get("spans"), list):
            return body
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[Any]:
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
