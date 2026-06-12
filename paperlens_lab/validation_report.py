from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .scenario_eval import evaluate_experiment_spec, evaluate_starter_code, source_contains_quote
from .source_index import text_hash


DEFAULT_VALIDATION_ROOT = Path("outputs") / "service_demo_validation"
REQUIRED_REAL_PAPER_EVALS = {
    "pdf_parse_and_reader_spans",
    "translation_fidelity",
    "grounded_qa",
    "adversarial_lost_in_the_middle",
    "experiment_spec",
    "starter_code_smoke",
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
        and local_demo
        and local_demo.get("sourceIndexConsistent", True)
        and local_demo.get("quotesInSourceIndex")
        and local_demo.get("translationSourceConsistent")
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
    artifact_issues = _real_paper_artifact_issues(body)
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
            item.get("name") == "starter_code_smoke" and item.get("passed")
            for item in evaluations
            if isinstance(item, dict)
        )
        starter_code = str((((run.get("model_outputs") or {}).get("starter_code") or {}).get("code") or ""))
        starter_code_eval = evaluate_starter_code(starter_code)
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
    trace_records = [record for record in _read_jsonl(trace_paths[0]) if isinstance(record, dict)]
    records_by_id = {
        str(record.get("trace_id")): record
        for record in trace_records
        if record.get("trace_id")
    }
    required_trace_ids = real_paper_run.get("requiredTraceIds", []) if real_paper_run else []
    trace_id_issues: list[str] = []
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
        if record.get("task") != task:
            trace_id_issues.append(f"{trace_id} task mismatch: expected {task}, saw {record.get('task')}")
        if record.get("status") != "model":
            trace_id_issues.append(f"{trace_id} is not model-backed")
        if record.get("error"):
            trace_id_issues.append(f"{trace_id} has error")
    if trace_id_issues:
        warnings.append(f"real-paper trace ids need rerun: {len(trace_id_issues)} issue(s)")

    for record in trace_records:
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
        "traceIdsPassed": not trace_id_issues,
        "traceIdIssues": trace_id_issues[:12],
        "requiredTraceIdCount": len(required_trace_ids),
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
    quotes_in_source_index = bool(evidence) and not unknown_evidence_ids and not bad_quote_ids and not missing_quote_text_ids
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


def _real_paper_artifact_issues(body: dict[str, Any]) -> list[str]:
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
        starter_code = str((((run.get("model_outputs") or {}).get("starter_code") or {}).get("code") or ""))
        starter_eval = evaluate_starter_code(starter_code)
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
        _append_trace_id(required, outputs.get("growth"), "research_growth", paper)
        _append_trace_id(required, outputs.get("growth_iteration"), "research_growth", paper)
    return required


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
