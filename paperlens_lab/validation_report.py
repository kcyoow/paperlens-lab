from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .model_adapter import DEFAULT_SMALL_MULTILINGUAL_MODEL
from .scenario_eval import evaluate_experiment_spec, evaluate_starter_code, source_contains_quote
from .source_index import load_source_index, source_index_dir, text_hash
from .tracing import trace_path


DEFAULT_VALIDATION_ROOT = Path("outputs") / "real_paper_validation"
REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_OUT_DIR = REPO_ROOT / "frontend" / "out"
REQUIRED_FRONTEND_EXPORT_FILES = ("index.html", "reader/index.html")
REQUIRED_REAL_PAPER_EVALS = {
    "pdf_parse_and_reader_spans",
    "translation_fidelity",
    "grounded_qa",
    "adversarial_lost_in_the_middle",
    "experiment_spec",
    "starter_code_source_run",
    "growth_ideas",
    "research_growth_iteration",
    "model_backing",
}
MIN_REAL_PAPER_SOURCE_CHARS = 6000
MIN_REAL_PAPER_READER_SPANS = 30


def validation_root() -> Path:
    return Path(os.getenv("PAPERLENS_VALIDATION_ROOT", str(DEFAULT_VALIDATION_ROOT)))


def build_validation_summary(root: Path | None = None) -> dict[str, Any]:
    root = root or validation_root()
    warnings: list[str] = []
    frontend_static_export = _frontend_static_export_summary(warnings=warnings)
    if not root.exists():
        return {
            "ok": False,
            "validationRoot": str(root),
            "warnings": [f"validation root not found: {root}", *warnings],
            "currentModelContract": _current_model_contract(),
            "frontendStaticExport": frontend_static_export,
            "realPaperRun": None,
            "modelTraces": None,
            "localDemo": None,
            "memory": None,
        }

    real_paper_run = _best_real_paper_run(root, warnings)
    trace_summary = _trace_summary_for_run(root, real_paper_run, warnings)
    local_demo = _local_demo_summary(root, warnings)
    memory_summary = _memory_summary_for_run(root, real_paper_run, warnings)
    current_contract = _current_model_contract()

    ok = bool(
        real_paper_run
        and real_paper_run.get("passed")
        and real_paper_run.get("paperCount", 0) >= 3
        and real_paper_run.get("evaluationTotal", 0) > 0
        and real_paper_run.get("evaluationPassed") == real_paper_run.get("evaluationTotal")
        and real_paper_run.get("evidenceConsistencyPassed")
        and real_paper_run.get("artifactContractPassed")
        and _adversarial_litm_passed(real_paper_run)
        and real_paper_run.get("growthIterationPassed")
        and real_paper_run.get("starterCodePassed")
        and trace_summary
        and trace_summary.get("traceIdsPassed")
        and trace_summary.get("total", 0) > 0
        and trace_summary.get("modelCount") == trace_summary.get("total")
        and trace_summary.get("fallbackCount") == 0
        and trace_summary.get("errorCount") == 0
        and trace_summary.get("currentContractMatched")
        and local_demo
        and local_demo.get("artifactBundleCoherent", False)
        and local_demo.get("traceIdsPassed")
        and local_demo.get("currentContractMatched")
        and local_demo.get("sourceIndexConsistent", True)
        and local_demo.get("quotesInSourceIndex")
        and local_demo.get("translationSourceConsistent")
        and not local_demo.get("usedFallback")
        and not local_demo.get("translationUsedFallback")
        and frontend_static_export.get("ready")
    )
    return {
        "ok": ok,
        "validationRoot": str(root),
        "warnings": warnings,
        "currentModelContract": current_contract,
        "frontendStaticExport": frontend_static_export,
        "realPaperRun": real_paper_run,
        "modelTraces": trace_summary,
        "localDemo": local_demo,
        "memory": memory_summary,
    }


def _frontend_static_export_summary(
    out_dir: Path | None = None,
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or FRONTEND_OUT_DIR
    issues: list[str] = []
    files = [path for path in out_dir.rglob("*") if path.is_file()] if out_dir.exists() else []
    total_bytes = sum(path.stat().st_size for path in files)
    required_file_status = {
        relative_path: (out_dir / relative_path).is_file()
        for relative_path in REQUIRED_FRONTEND_EXPORT_FILES
    }
    next_static_dir = out_dir / "_next" / "static"
    next_static_file_count = (
        sum(1 for path in next_static_dir.rglob("*") if path.is_file())
        if next_static_dir.is_dir()
        else 0
    )
    reader_chunk_paths = sorted((out_dir / "_next" / "static" / "chunks" / "app" / "reader").glob("page-*.js"))
    has_reader_chunk = any(path.is_file() for path in reader_chunk_paths)

    if not out_dir.exists():
        issues.append(f"frontend static export directory is missing: {out_dir}")
    for relative_path, exists in required_file_status.items():
        if not exists:
            issues.append(f"frontend static export missing required file: {relative_path}")
    if not next_static_dir.is_dir():
        issues.append("frontend static export missing Next static asset directory: _next/static")
    elif next_static_file_count < 1:
        issues.append("frontend static export has no Next static asset files")
    if not has_reader_chunk:
        issues.append("frontend static export missing reader page chunk: _next/static/chunks/app/reader/page-*.js")
    if files and total_bytes <= 0:
        issues.append("frontend static export files are empty")

    if issues and warnings is not None:
        warnings.append(f"frontend static export is not deploy-ready: {issues[0]}")

    return {
        "ready": not issues,
        "outDir": str(out_dir),
        "fileCount": len(files),
        "totalBytes": total_bytes,
        "requiredFiles": required_file_status,
        "hasIndex": required_file_status.get("index.html", False),
        "hasReader": required_file_status.get("reader/index.html", False),
        "hasNextStatic": next_static_dir.is_dir(),
        "hasReaderChunk": has_reader_chunk,
        "readerChunkCount": len(reader_chunk_paths),
        "nextStaticFileCount": next_static_file_count,
        "issues": issues,
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
    artifact_issues = _real_paper_artifact_issues(body, path)
    if evidence_issues:
        warnings.append(
            f"real-paper summary evidence consistency needs rerun: {len(evidence_issues)} issue(s)"
        )
    if artifact_issues:
        warnings.append(f"real-paper artifact contract needs rerun: {len(artifact_issues)} issue(s)")
    runs = body.get("runs", []) if isinstance(body.get("runs"), list) else []
    papers = []
    evaluation_total = 0
    evaluation_passed = 0
    for run in runs:
        evaluations = run.get("evaluations", []) if isinstance(run, dict) else []
        source = run.get("source", {}) if isinstance(run, dict) else {}
        reader = run.get("reader", {}) if isinstance(run, dict) else {}
        memory = run.get("memory", {}) if isinstance(run, dict) else {}
        eval_names = {str(item.get("name", "")) for item in evaluations if isinstance(item, dict)}
        missing_required = sorted(REQUIRED_REAL_PAPER_EVALS - eval_names)
        growth_iteration = (((run.get("model_outputs") or {}).get("growth_iteration") or {}).get("data") or {})
        growth_iteration_evidence = _growth_iteration_evidence(growth_iteration)
        growth_iteration_idea_evidence = _growth_iteration_idea_evidence(growth_iteration)
        growth_iteration_eval_passed = any(
            item.get("name") == "research_growth_iteration" and item.get("passed")
            for item in evaluations
            if isinstance(item, dict)
        )
        starter_code_eval_passed = any(
            item.get("name") == "starter_code_source_run" and item.get("passed")
            for item in evaluations
            if isinstance(item, dict)
        )
        starter_output = ((run.get("model_outputs") or {}).get("starter_code") or {})
        starter_code = _starter_code_text(starter_output)
        starter_evidence_rows = _starter_evidence_rows_for_run(run, path)
        starter_code_eval = evaluate_starter_code(
            starter_code,
            evidence_rows=starter_evidence_rows,
            require_evidence_rows=True,
        )
        starter_code_passed = bool(starter_code_eval_passed and starter_code_eval.passed)
        experiment_data = (((run.get("model_outputs") or {}).get("experiment") or {}).get("data") or {})
        experiment_spec_eval = evaluate_experiment_spec(experiment_data if isinstance(experiment_data, dict) else {})
        source_contract_issues = _real_paper_source_contract_issues(source, reader)
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
                "requiredEvalMissing": missing_required,
                "requiredEvalPassed": not missing_required,
                "sourceContractIssues": source_contract_issues,
                "sourceContractPassed": not source_contract_issues,
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
                "memoryRecordsBeforeGrowthIteration": memory.get("records_before_growth_iteration", 0),
                "starterCodePassed": starter_code_passed,
                "starterCodeRechecked": starter_code_eval.passed,
                "starterCodeReasons": starter_code_eval.reasons,
                "starterCodeUsedFallback": bool(starter_output.get("used_fallback")) if isinstance(starter_output, dict) else False,
                "starterCodeTraceId": str(starter_output.get("trace_id") or "") if isinstance(starter_output, dict) else "",
                "starterCodeModel": str(starter_output.get("model") or "") if isinstance(starter_output, dict) else "",
                "experimentSpecRechecked": experiment_spec_eval.passed,
                "experimentSpecReasons": experiment_spec_eval.reasons,
                "growthIterationPassed": growth_iteration_eval_passed,
                "growthIterationEvidence": growth_iteration_evidence,
                "growthIterationIdeaEvidence": growth_iteration_idea_evidence,
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
        "artifactContractPassed": not artifact_issues,
        "artifactContractIssues": artifact_issues[:12],
        "growthIterationPassed": _growth_iteration_passed_for_papers(papers),
        "starterCodePassed": all(paper.get("starterCodePassed") for paper in papers) if papers else False,
        "requiredTraceIds": _required_trace_ids_from_summary(body),
        "fineTuningRecommendation": fine_tuning.get("recommendation", "unknown"),
        "fineTuningReason": fine_tuning.get("reason", ""),
        "repeatedFailures": fine_tuning.get("repeated_failures", []),
        "papers": papers,
    }


def _growth_iteration_evidence(data: dict[str, Any]) -> list[str]:
    evidence_ids: list[str] = []
    for idea_evidence in _growth_iteration_idea_evidence(data):
        for value in idea_evidence:
            if value and value not in evidence_ids:
                evidence_ids.append(value)
    return evidence_ids


def _growth_iteration_idea_evidence(data: dict[str, Any]) -> list[list[str]]:
    ideas = data.get("ideas", []) if isinstance(data.get("ideas"), list) else []
    idea_evidence: list[list[str]] = []
    for idea in ideas:
        if not isinstance(idea, dict):
            continue
        evidence_ids: list[str] = []
        for source_id in idea.get("source_evidence") or []:
            value = str(source_id)
            if value and value not in evidence_ids:
                evidence_ids.append(value)
        if evidence_ids:
            idea_evidence.append(evidence_ids)
    return idea_evidence


def _growth_iteration_passed_for_papers(papers: list[dict[str, Any]]) -> bool:
    if not papers:
        return False
    for paper in papers:
        complete_idea = any(
            _growth_iteration_evidence_set_is_complete(set(idea_evidence))
            for idea_evidence in paper.get("growthIterationIdeaEvidence") or []
        )
        if not (paper.get("growthIterationPassed") and complete_idea):
            return False
    return True


def _growth_iteration_evidence_set_is_complete(evidence_ids: set[str]) -> bool:
    cites_paper = "paper:selected-middle" in evidence_ids or any(
        source_id.startswith("paper:") for source_id in evidence_ids
    )
    cites_run = "run:r1" in evidence_ids
    cites_prior_growth = any(source_id.startswith("growth_idea:") for source_id in evidence_ids)
    return cites_paper and cites_run and cites_prior_growth


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
    contract = _current_model_contract()
    trace_paths = []
    if real_paper_run:
        summary_path = Path(real_paper_run.get("summaryPath", ""))
        if summary_path.name == "summary.json":
            trace_paths.extend(_candidate_real_paper_trace_paths(root, summary_path))
    if not trace_paths:
        warnings.append("no matching trace JSONL found for the selected real-paper summary")
        return None

    task_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    error_count = 0
    trace_records: list[dict[str, Any]] = []
    used_trace_paths: list[Path] = []
    for trace_candidate in trace_paths:
        if not trace_candidate.exists():
            continue
        records = [record for record in _read_jsonl(trace_candidate) if isinstance(record, dict)]
        if not records:
            continue
        used_trace_paths.append(trace_candidate)
        trace_records.extend(records)
    if not trace_records:
        warnings.append("matching trace JSONL candidates were empty")
        return None
    records_by_id = {
        str(record.get("trace_id")): record
        for record in trace_records
        if record.get("trace_id")
    }
    required_trace_ids = real_paper_run.get("requiredTraceIds", []) if real_paper_run else []
    trace_id_issues: list[str] = []
    contract_issues: list[str] = []
    selected_trace_records: list[dict[str, Any]] = []
    seen_trace_ids: set[str] = set()
    for expected in required_trace_ids:
        if not isinstance(expected, dict):
            continue
        trace_id = str(expected.get("trace_id") or "")
        task = str(expected.get("task") or "")
        if not trace_id:
            trace_id_issues.append(f"{expected.get('paper', 'unknown')}:{task} missing trace_id")
            continue
        record = records_by_id.get(trace_id)
        if not record:
            trace_id_issues.append(f"{expected.get('paper', 'unknown')}:{task} trace {trace_id} missing from JSONL")
            continue
        if trace_id not in seen_trace_ids:
            selected_trace_records.append(record)
            seen_trace_ids.add(trace_id)
        if record.get("task") != task:
            trace_id_issues.append(f"{trace_id} task mismatch: expected {task}, saw {record.get('task')}")
        if record.get("status") != "model":
            trace_id_issues.append(f"{trace_id} is not model-backed")
        if record.get("error"):
            trace_id_issues.append(f"{trace_id} has error")
        contract_issues.extend(_trace_contract_issues(record, task, contract, label=trace_id))
    if trace_id_issues:
        warnings.append(f"real-paper trace ids need rerun: {len(trace_id_issues)} issue(s)")
    if contract_issues:
        warnings.append(f"real-paper traces are stale for current model contract: {len(contract_issues)} issue(s)")

    for record in selected_trace_records:
        if not isinstance(record, dict) or not record.get("task"):
            continue
        task_counts[str(record.get("task"))] += 1
        status_counts[str(record.get("status") or "unknown")] += 1
        provider_counts[str(record.get("provider") or "unknown")] += 1
        model_counts[str(record.get("model") or "unknown")] += 1
        if record.get("error"):
            error_count += 1

    return {
        "tracePath": str(used_trace_paths[0]) if used_trace_paths else "",
        "tracePaths": [str(path) for path in used_trace_paths],
        "scannedTraceRecordCount": len(trace_records),
        "total": len(selected_trace_records),
        "modelCount": status_counts.get("model", 0),
        "fallbackCount": status_counts.get("fallback", 0),
        "errorCount": error_count,
        "byTask": dict(sorted(task_counts.items())),
        "byProvider": dict(sorted(provider_counts.items())),
        "byModel": dict(sorted(model_counts.items())),
        "traceIdsPassed": not trace_id_issues,
        "traceIdIssues": trace_id_issues[:12],
        "requiredTraceIdCount": len(required_trace_ids),
        "currentModelContract": contract,
        "currentContractMatched": not contract_issues,
        "currentContractIssues": contract_issues[:12],
    }


def _candidate_real_paper_trace_paths(root: Path, summary_path: Path) -> list[Path]:
    run_name = summary_path.parent.name
    candidates = [
        summary_path.parent / "agent_traces.jsonl",
        summary_path.parent / f"{run_name}_traces.jsonl",
        summary_path.parent / "traces.jsonl",
        summary_path.parent.parent / f"{run_name}_traces.jsonl",
        root / f"{run_name}_traces.jsonl",
        root / "agent_traces.jsonl",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _local_demo_summary(root: Path, warnings: list[str]) -> dict[str, Any] | None:
    contract = _current_model_contract()
    bundle = _latest_local_demo_bundle(root)
    if not bundle:
        warnings.append("no local selected-span Q&A proof found")
        return None

    ask_path = bundle["ask_path"]
    translate_path = bundle["translate_path"]
    paper_path = bundle["paper_path"]
    ask = _read_json(ask_path) or {}
    translate = _read_json(translate_path) if translate_path else {}
    paper = _read_json(paper_path) if paper_path else {}
    evidence_window = ask.get("evidenceWindow") if isinstance(ask.get("evidenceWindow"), dict) else {}
    evidence = ask.get("evidence", []) if isinstance(ask.get("evidence"), list) else []
    metadata = paper.get("metadata", {}) if isinstance(paper, dict) else {}
    source_index, source_index_path, source_index_runtime_bound = _source_index_for_local_demo(root, evidence_window)
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
    source_text_by_span_id = _source_index_text_by_span_id(source_index, allowed_evidence_ids)
    selected_span_id = str(evidence_window.get("spanId", ""))
    selected_source_text = source_text_by_span_id.get(selected_span_id, "")
    expected_translation_source_hash = text_hash(selected_source_text) if selected_source_text else ""
    translation_source_hash = str(translate.get("sourceHash", "")) if isinstance(translate, dict) else ""
    translation_source_index_bound = bool(translate.get("sourceIndexBound")) if isinstance(translate, dict) else False
    translation_source_consistent = bool(
        translation_source_index_bound
        and translation_source_hash
        and expected_translation_source_hash
        and translation_source_hash == expected_translation_source_hash
    )
    local_trace_check = _local_demo_trace_check(root, ask_path, ask, translate, contract)
    local_contract_issues = _local_demo_artifact_contract_issues(ask, translate, contract)
    bad_quote_ids = sorted(
        {
            str(item.get("source_id", ""))
            for item in evidence
            if isinstance(item, dict)
            and item.get("source_id")
            and str(item.get("source_id", "")) in source_text_by_span_id
            and str(item.get("quote", "")).strip()
            and not source_contains_quote(
                source_text_by_span_id[str(item.get("source_id", ""))],
                str(item.get("quote", "")).strip(),
            )
        }
    )
    missing_quote_text_ids = sorted(
        {
            str(item.get("source_id", ""))
            for item in evidence
            if isinstance(item, dict)
            and item.get("source_id")
            and str(item.get("source_id", "")) in allowed_evidence_ids
            and str(item.get("source_id", "")) not in source_text_by_span_id
        }
    )
    if evidence_window.get("paperId") and not source_index:
        source_index_consistent = False
        warnings.append("local selected-span source index is missing; rerun local browser/API proof")
    elif evidence_window.get("paperId") and not source_index_runtime_bound:
        source_index_consistent = False
        warnings.append(
            "local selected-span source index is not bound to the current runtime; rerun local browser/API proof"
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
    if bad_quote_ids or missing_quote_text_ids:
        source_index_consistent = False
        warnings.append(
            "local selected-span answer quote is not verifiable in the source-index window; rerun local browser/API proof"
        )
    if translate and not translation_source_consistent:
        source_index_consistent = False
        warnings.append(
            "local selected-span translation source hash is not bound to the source index; rerun local translation proof"
        )
    if local_contract_issues:
        warnings.append("local selected-span proof is stale for the current model contract; rerun local browser/API proof")
    warnings.extend(local_trace_check["warnings"])
    quotes_in_source_index = bool(evidence) and not unknown_evidence_ids and not bad_quote_ids and not missing_quote_text_ids
    return {
        "askPath": str(ask_path),
        "translatePath": str(translate_path) if translate_path else "",
        "paperPath": str(paper_path) if paper_path else "",
        "bundleDir": str(ask_path.parent),
        "artifactBundleCoherent": bool(bundle.get("coherent")),
        "paperTitle": paper.get("title", ""),
        "readerSpanCount": metadata.get("readerSpanCount", 0),
        "sourceTextChars": metadata.get("sourceTextChars", 0),
        "selectedSpanId": evidence_window.get("spanId", ""),
        "evidenceWindow": evidence_window.get("spanRange", ""),
        "sourceHash": evidence_window.get("sourceHash", ""),
        "sourceIndexPath": source_index_path,
        "sourceIndexHash": source_index_hash,
        "sourceIndexRuntimeBound": source_index_runtime_bound,
        "sourceIndexConsistent": source_index_consistent,
        "neighborSpans": evidence_window.get("spans", []),
        "evidenceIds": evidence_ids,
        "unknownEvidenceIds": unknown_evidence_ids,
        "badQuoteIds": bad_quote_ids,
        "missingQuoteTextIds": missing_quote_text_ids,
        "quoteIdsWithinWindow": not unknown_evidence_ids,
        "quotesInSourceIndex": quotes_in_source_index,
        "quoteCount": len(evidence),
        "confidence": ask.get("confidence", ""),
        "needsMoreContext": bool(ask.get("needsMoreContext")),
        "provider": ask.get("provider", ""),
        "model": ask.get("model", ""),
        "traceId": ask.get("traceId", ""),
        "tracePath": local_trace_check["trace_path"],
        "traceIdsPassed": local_trace_check["passed"],
        "traceIdIssues": local_trace_check["issues"],
        "currentModelContract": contract,
        "currentContractMatched": not local_contract_issues and local_trace_check["contract_matched"],
        "currentContractIssues": (local_contract_issues + local_trace_check["contract_issues"])[:12],
        "usedFallback": bool(ask.get("usedFallback")),
        "translationStatus": translate.get("status", "") if isinstance(translate, dict) else "",
        "translationTraceId": translate.get("traceId", "") if isinstance(translate, dict) else "",
        "translationUsedFallback": bool(translate.get("usedFallback")) if isinstance(translate, dict) else False,
        "translationSourceHash": translation_source_hash,
        "translationExpectedSourceHash": expected_translation_source_hash,
        "translationSourceIndexBound": translation_source_index_bound,
        "translationSourceConsistent": translation_source_consistent,
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
                if quote and source_id and not source_contains_quote(str(source_evidence.get(source_id, "")), quote):
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
                if quote and source_id and not source_contains_quote(str(source_evidence.get(source_id, "")), quote):
                    issues.append(f"{case_name}:{span_id} adversarial quote missing from {source_id}")
    return issues


def _real_paper_artifact_issues(body: dict[str, Any], summary_path: Path) -> list[str]:
    issues: list[str] = []
    runs = body.get("runs", []) if isinstance(body.get("runs"), list) else []
    for run in runs:
        if not isinstance(run, dict):
            issues.append("summary contains a non-object paper run")
            continue
        case_name = (run.get("case") or {}).get("name", "unknown")
        evaluations = run.get("evaluations", []) if isinstance(run.get("evaluations"), list) else []
        eval_names = {str(item.get("name", "")) for item in evaluations if isinstance(item, dict)}
        for name in sorted(REQUIRED_REAL_PAPER_EVALS - eval_names):
            issues.append(f"{case_name} missing required evaluation {name}")
        for issue in _real_paper_source_contract_issues(
            run.get("source", {}) if isinstance(run.get("source"), dict) else {},
            run.get("reader", {}) if isinstance(run.get("reader"), dict) else {},
        ):
            issues.append(f"{case_name} {issue}")
        starter_code = _starter_code_text(((run.get("model_outputs") or {}).get("starter_code") or {}))
        starter_eval = evaluate_starter_code(
            starter_code,
            evidence_rows=_starter_evidence_rows_for_run(run, summary_path),
            require_evidence_rows=True,
        )
        if not starter_eval.passed:
            issues.append(f"{case_name} starter code does not rerun: {', '.join(starter_eval.reasons)}")
        experiment_data = (((run.get("model_outputs") or {}).get("experiment") or {}).get("data") or {})
        experiment_eval = evaluate_experiment_spec(experiment_data if isinstance(experiment_data, dict) else {})
        if not experiment_eval.passed:
            issues.append(f"{case_name} experiment spec does not revalidate: {', '.join(experiment_eval.reasons)}")
    return issues


def _real_paper_source_contract_issues(source: dict[str, Any], reader: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    pdf_url = str(source.get("pdf_url") or "")
    page_markers = int(source.get("page_marker_count") or 0)
    source_chars = int(source.get("text_chars") or 0)
    reader_spans = int(reader.get("visible_span_count") or 0)
    if not pdf_url.startswith(("http://", "https://")) or "pdf" not in pdf_url.lower():
        issues.append("is not bound to a PDF URL")
    if page_markers < 1:
        issues.append("has no parsed PDF page markers")
    if source_chars < MIN_REAL_PAPER_SOURCE_CHARS:
        issues.append(f"has too little source text ({source_chars} chars)")
    if reader_spans < MIN_REAL_PAPER_READER_SPANS:
        issues.append(f"has too few reader spans ({reader_spans})")
    return issues


def _starter_evidence_rows_for_run(run: dict[str, Any], summary_path: Path) -> list[dict[str, Any]]:
    reader = run.get("reader", {}) if isinstance(run.get("reader"), dict) else {}
    paper_id = str(reader.get("document_id") or "")
    if not paper_id:
        return []
    source_index = _source_index_for_real_paper_run(paper_id, summary_path)
    spans = source_index.get("spans", []) if isinstance(source_index, dict) else []
    if not isinstance(spans, list) or not spans:
        return []
    selected_positions = reader.get("selected_span_positions", [])
    selected_span_id = ""
    if isinstance(selected_positions, list):
        selected_span_id = str(
            next(
                (
                    item.get("span_id")
                    for item in selected_positions
                    if isinstance(item, dict) and item.get("position_label") == "middle"
                ),
                "",
            )
        )
        if not selected_span_id and selected_positions:
            middle_item = selected_positions[len(selected_positions) // 2]
            if isinstance(middle_item, dict):
                selected_span_id = str(middle_item.get("span_id") or "")
    if not selected_span_id:
        return []
    selected_index = next((idx for idx, span in enumerate(spans) if span.get("span_id") == selected_span_id), None)
    if selected_index is None:
        return []
    start = max(0, selected_index - 4)
    end = min(len(spans), selected_index + 5)
    selected_text = str(spans[selected_index].get("text") or "").strip()
    rows: list[dict[str, Any]] = []
    for span in spans[start:end]:
        if not isinstance(span, dict):
            continue
        source_id = str(span.get("span_id") or "")
        text = str(span.get("text") or "").strip()
        if not source_id or not text:
            continue
        label = "selected" if source_id == selected_span_id else "context_control"
        rows.append(
            {
                "source_id": source_id,
                "text": text,
                "text_hash": str(span.get("text_hash") or text_hash(text)),
                "label": label,
                "gold": label == "selected",
                "query": selected_text,
            }
        )
    return rows


def _source_index_for_real_paper_run(paper_id: str, summary_path: Path) -> dict[str, Any] | None:
    for base_dir in (
        summary_path.parent / "source_index",
        summary_path.parent.parent / "source_index",
        source_index_dir(),
    ):
        path = _source_index_path_for_paper_id(paper_id, base_dir)
        body = _read_json(path)
        if isinstance(body, dict):
            return body
    return load_source_index(paper_id)


def _required_trace_ids_from_summary(body: dict[str, Any]) -> list[dict[str, str]]:
    required: list[dict[str, str]] = []
    runs = body.get("runs", []) if isinstance(body.get("runs"), list) else []
    for run in runs:
        if not isinstance(run, dict):
            continue
        paper = str((run.get("case") or {}).get("name", "unknown"))
        outputs = run.get("model_outputs") if isinstance(run.get("model_outputs"), dict) else {}
        _append_trace_id(required, outputs.get("translation"), "translation", paper)
        for qa in outputs.get("qa") or []:
            if isinstance(qa, dict):
                _append_trace_id(required, qa.get("result"), "grounded_qa", paper)
        adversarial = outputs.get("adversarial_litm") if isinstance(outputs.get("adversarial_litm"), dict) else {}
        _append_trace_id(required, adversarial.get("result"), "adversarial_grounded_qa", paper)
        _append_trace_id(required, outputs.get("experiment"), "experiment_spec", paper)
        _append_trace_id(required, outputs.get("starter_code"), "starter_code", paper)
        _append_trace_id(required, outputs.get("growth"), "research_growth", paper)
        _append_trace_id(required, outputs.get("growth_iteration"), "research_growth", paper)
    return required


def _starter_code_text(starter_output: Any) -> str:
    if not isinstance(starter_output, dict):
        return ""
    data = starter_output.get("data")
    if isinstance(data, dict) and data.get("code"):
        return str(data.get("code") or "")
    return str(starter_output.get("code") or "")


def _append_trace_id(required: list[dict[str, str]], result: Any, task: str, paper: str) -> None:
    if not isinstance(result, dict):
        required.append({"paper": paper, "task": task, "trace_id": ""})
        return
    required.append({"paper": paper, "task": task, "trace_id": str(result.get("trace_id") or "")})


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


def _source_index_for_local_demo(
    root: Path,
    evidence_window: dict[str, Any],
) -> tuple[dict[str, Any] | None, str, bool]:
    paper_id = str(evidence_window.get("paperId") or "")
    if not paper_id:
        return None, "", False
    safe_name = _safe_path_key(paper_id)
    bundle_candidates = [
        root / "source_index" / f"{safe_name}.json",
        root / f"{safe_name}.json",
    ]
    bundle_candidates.extend(
        sorted(root.rglob(f"{safe_name}.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    )
    seen_bundle_paths: set[str] = set()
    for path in bundle_candidates:
        key = str(path)
        if key in seen_bundle_paths:
            continue
        seen_bundle_paths.add(key)
        body = _read_json(path)
        if body and body.get("paper_id") == paper_id and isinstance(body.get("spans"), list):
            return body, str(path), True
    runtime_record = load_source_index(paper_id)
    if runtime_record and isinstance(runtime_record.get("spans"), list):
        return runtime_record, str(_source_index_path_for_paper_id(paper_id, source_index_dir())), True
    return None, "", False


def _latest_local_demo_bundle(root: Path) -> dict[str, Any] | None:
    ask_paths = sorted(root.rglob("local_after_source_index_ask_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not ask_paths:
        return None

    for ask_path in ask_paths:
        suffix = _local_demo_span_suffix(ask_path, prefix="local_after_source_index_ask_")
        translate_path = ask_path.parent / f"local_after_source_index_translate_{suffix}.json"
        paper_path = ask_path.parent / "local_after_source_index_paper.json"
        if translate_path.exists() and paper_path.exists():
            return {
                "ask_path": ask_path,
                "translate_path": translate_path,
                "paper_path": paper_path,
                "coherent": True,
            }

    ask_path = ask_paths[0]
    suffix = _local_demo_span_suffix(ask_path, prefix="local_after_source_index_ask_")
    translate_path = ask_path.parent / f"local_after_source_index_translate_{suffix}.json"
    paper_path = ask_path.parent / "local_after_source_index_paper.json"
    return {
        "ask_path": ask_path,
        "translate_path": translate_path if translate_path.exists() else None,
        "paper_path": paper_path if paper_path.exists() else None,
        "coherent": False,
    }


def _local_demo_span_suffix(path: Path, *, prefix: str) -> str:
    stem = path.stem
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def _safe_path_key(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)[:120]


def _source_index_path_for_paper_id(paper_id: str, base_dir: Path) -> Path:
    return base_dir / f"{_safe_path_key(paper_id)}.json"


def _local_demo_trace_check(
    root: Path,
    ask_path: Path,
    ask: dict[str, Any],
    translate: dict[str, Any] | None,
    contract: dict[str, str],
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    contract_issues: list[str] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    used_trace_paths: list[str] = []
    for candidate in _candidate_local_trace_paths(root, ask_path):
        if not candidate.exists():
            continue
        used_trace_paths.append(str(candidate))
        for record in _read_jsonl(candidate):
            if isinstance(record, dict) and record.get("trace_id"):
                records_by_id[str(record.get("trace_id"))] = record

    expected = [
        ("grounded_qa", str(ask.get("traceId") or "")),
        ("translation", str((translate or {}).get("traceId") or "")),
    ]
    for task, trace_id in expected:
        if not trace_id:
            issues.append(f"local {task} trace_id is missing")
            continue
        record = records_by_id.get(trace_id)
        if not record:
            issues.append(f"local {task} trace {trace_id} missing from JSONL")
            continue
        if record.get("task") != task:
            issues.append(f"local trace {trace_id} task mismatch: expected {task}, saw {record.get('task')}")
        if record.get("status") != "model":
            issues.append(f"local trace {trace_id} is not model-backed")
        if record.get("error"):
            issues.append(f"local trace {trace_id} has error")
        contract_issues.extend(_trace_contract_issues(record, task, contract, label=f"local {trace_id}"))
    if issues:
        warnings.append("local selected-span trace binding is incomplete; rerun local browser/API proof")
    if contract_issues:
        warnings.append("local selected-span traces are stale for the current model contract; rerun local browser/API proof")
    return {
        "passed": not issues,
        "issues": issues[:12],
        "contract_matched": not contract_issues,
        "contract_issues": contract_issues[:12],
        "warnings": warnings,
        "trace_path": used_trace_paths[0] if used_trace_paths else "",
    }


def _current_model_contract() -> dict[str, str]:
    general_model = os.getenv("PAPERLENS_MODEL", DEFAULT_SMALL_MULTILINGUAL_MODEL)
    return {
        "provider": os.getenv("PAPERLENS_PROVIDER", "fallback"),
        "model": general_model,
        "translationModel": os.getenv("PAPERLENS_TRANSLATION_MODEL", general_model),
        "qualityModel": os.getenv("PAPERLENS_QUALITY_MODEL", general_model),
    }


def _expected_model_for_task(task: str, contract: dict[str, str]) -> str:
    if task == "translation":
        return contract["translationModel"]
    if task in {"starter_code", "research_growth"}:
        return contract["qualityModel"]
    return contract["model"]


def _trace_contract_issues(
    record: dict[str, Any],
    task: str,
    contract: dict[str, str],
    *,
    label: str,
) -> list[str]:
    issues: list[str] = []
    provider = str(record.get("provider") or "")
    model = str(record.get("model") or "")
    expected_model = _expected_model_for_task(task, contract)
    if provider != contract["provider"]:
        issues.append(f"{label} provider mismatch: expected {contract['provider']}, saw {provider or 'missing'}")
    if model != expected_model:
        issues.append(f"{label} model mismatch: expected {expected_model}, saw {model or 'missing'}")
    return issues


def _local_demo_artifact_contract_issues(
    ask: dict[str, Any],
    translate: dict[str, Any] | None,
    contract: dict[str, str],
) -> list[str]:
    issues: list[str] = []
    ask_provider = str(ask.get("provider") or "")
    ask_model = str(ask.get("model") or "")
    if ask_provider != contract["provider"]:
        issues.append(f"local ask provider mismatch: expected {contract['provider']}, saw {ask_provider or 'missing'}")
    if ask_model != contract["model"]:
        issues.append(f"local ask model mismatch: expected {contract['model']}, saw {ask_model or 'missing'}")
    if translate:
        translate_provider = str(translate.get("provider") or ask_provider)
        translate_model = str(translate.get("model") or "")
        if translate_provider != contract["provider"]:
            issues.append(
                f"local translation provider mismatch: expected {contract['provider']}, saw {translate_provider or 'missing'}"
            )
        if translate_model and translate_model != contract["translationModel"]:
            issues.append(
                f"local translation model mismatch: expected {contract['translationModel']}, saw {translate_model}"
            )
    return issues


def _candidate_local_trace_paths(root: Path, ask_path: Path) -> list[Path]:
    candidates = [
        ask_path.parent / "local_after_source_index_traces.jsonl",
        root / "local_after_source_index_traces.jsonl",
        root / "agent_traces.jsonl",
        trace_path(),
    ]
    candidates.extend(sorted(ask_path.parent.glob("*traces.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _source_index_text_by_span_id(
    source_index: dict[str, Any] | None,
    allowed_evidence_ids: set[str],
) -> dict[str, str]:
    if not source_index or not isinstance(source_index.get("spans"), list):
        return {}
    text_by_id: dict[str, str] = {}
    for span in source_index.get("spans", []):
        if not isinstance(span, dict):
            continue
        span_id = str(span.get("span_id") or span.get("spanId") or "")
        text = str(span.get("text") or "")
        if span_id and span_id in allowed_evidence_ids and text:
            text_by_id[span_id] = text
    return text_by_id


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
