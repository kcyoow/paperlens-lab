from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .ingest import PaperSource, build_source
from .memory_store import append_memory, load_memories, paper_key
from .model_adapter import DEFAULT_MODEL, DEFAULT_PROVIDER, QUALITY_MODEL, ModelGateway, evidence_map
from .scenario_eval import (
    EvalResult,
    FailureRecord,
    evaluate_experiment_spec,
    evaluate_grounded_qa,
    evaluate_growth_ideas,
    evaluate_translation,
    fine_tuning_gate,
)
from .server import paper_document_from_source


DEFAULT_OUTPUT_DIR = Path("outputs") / "real_paper_validation"


@dataclass(frozen=True)
class RealPaperCase:
    name: str
    arxiv: str
    question: str
    idea: str


DEFAULT_REAL_PAPERS = [
    RealPaperCase(
        name="attention_is_all_you_need",
        arxiv="1706.03762",
        question="What exactly does this selected span claim, and what should not be inferred beyond it?",
        idea="Turn one highlighted Transformer mechanism into a 30-minute comparison against a simpler sequence baseline.",
    ),
    RealPaperCase(
        name="retrieval_augmented_generation",
        arxiv="2005.11401",
        question="What is the selected span's concrete method or result, and what evidence would be missing for a broader claim?",
        idea="Build a tiny retrieval-plus-generation ablation on a hand-built question set.",
    ),
    RealPaperCase(
        name="lora",
        arxiv="2106.09685",
        question="Does this selected span justify fine-tuning, prompting, or only a narrower adapter-style test?",
        idea="Compare a prompting baseline with a tiny adapter-like memory or parameter-efficient update proxy.",
    ),
]


def run_real_paper_case(
    case: RealPaperCase,
    *,
    source: PaperSource | None = None,
    gateway: ModelGateway | None = None,
    use_model: bool = False,
    max_pdf_pages: int = 8,
    max_translate_spans: int = 12,
    max_reader_spans: int = 180,
    locale: str = "ko",
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    gateway = gateway or ModelGateway()
    source = source or build_source(
        uploaded_pdf=None,
        arxiv_or_url=case.arxiv,
        pasted_text="",
        max_pdf_pages=max_pdf_pages,
    )
    document = paper_document_from_source(
        source,
        use_model=use_model,
        max_translate_spans=max_translate_spans,
        max_reader_spans=max_reader_spans,
    )
    spans = _flatten_reader_spans(document)
    selected = _selected_spans(spans)
    paper_id = paper_key(document["id"] or document["title"])

    translations = gateway.translate_spans(
        document["title"],
        [{"span_id": item["id"], "text": item["original"]} for item in selected],
        locale=locale,
        use_model=use_model,
    )
    translation_by_id = {
        item.get("span_id", ""): item.get("translation", "")
        for item in translations.data.get("translations", [])
        if isinstance(item, dict)
    }

    qa_runs = []
    for item in selected:
        source_evidence = evidence_map(source.text, item["original"], span_id=item["id"])
        qa = gateway.answer_span(
            paper_title=document["title"],
            span_id=item["id"],
            selected_span=item["original"],
            translated_span=translation_by_id.get(item["id"], item.get("translated", "")),
            question=f"{case.question} Use the selected span id {item['id']} as the primary evidence.",
            source_text=source.text,
            locale=locale,
            use_model=use_model,
        )
        qa_runs.append(
            {
                "position": item["position"],
                "span": item,
                "source_evidence": source_evidence,
                "result": _public_result(qa),
            }
        )

    experiment = gateway.experiment_spec(
        paper_title=document["title"],
        selected_span=selected[len(selected) // 2]["original"] if selected else "",
        translated_span=translation_by_id.get(selected[len(selected) // 2]["id"], "") if selected else "",
        source_text=source.text,
        idea=case.idea,
        locale=locale,
        use_model=use_model,
    )

    append_memory(
        paper_id,
        kind="paper_span",
        payload={
            "paper_title": document["title"],
            "summary": selected[len(selected) // 2]["original"] if selected else "",
        },
        evidence_id="paper:selected-middle",
    )
    append_memory(
        paper_id,
        kind="mini_lab_result",
        payload={"paper_title": document["title"], "summary": experiment.text[:1200]},
        evidence_id="run:r1",
    )
    memories = load_memories(paper_id)
    growth = gateway.growth_ideas(
        paper_title=document["title"],
        paper_memory=memories,
        mini_lab_result=experiment.text,
        selected_span=selected[len(selected) // 2]["original"] if selected else "",
        locale=locale,
        use_model=use_model,
    )
    for idea in growth.data.get("ideas", []):
        append_memory(
            paper_id,
            kind="growth_idea",
            payload={"paper_title": document["title"], "idea": idea},
        )

    evals = [
        evaluate_pdf_parse(source, document, spans),
        *_evaluate_run(selected, translations.data, qa_runs, experiment.data, growth.data, memories),
    ]
    if use_model:
        evals.append(evaluate_model_backing(translations, qa_runs, experiment, growth))
    failures = _failure_records(case.name, evals, qa_runs, experiment, growth)
    result = {
        "case": asdict(case),
        "passed": all(item.passed for item in evals),
        "source": {
            "title": source.title,
            "authors": source.authors,
            "source_label": source.source_label,
            "pdf_url": source.pdf_url,
            "text_chars": len(source.text),
            "word_count": len(source.text.split()),
            "page_marker_count": source.text.count("[page "),
        },
        "reader": {
            "document_id": document["id"],
            "metadata": document.get("metadata", {}),
            "visible_span_count": len(spans),
            "selected_span_positions": [
                {
                    "position": item["position"],
                    "position_label": item.get("position_label", ""),
                    "span_id": item["id"],
                    "chars": len(item["original"]),
                }
                for item in selected
            ],
        },
        "evaluations": [asdict(item) for item in evals],
        "fine_tuning": fine_tuning_gate(failures) if use_model else _no_model_fine_tuning_decision(),
        "model_outputs": {
            "translation": _public_result(translations),
            "qa": qa_runs,
            "experiment": _public_result(experiment),
            "growth": _public_result(growth),
        },
        "memory": {
            "paper_id": paper_id,
            "records_before_growth": len(memories),
            "records_after_growth": len(load_memories(paper_id)),
        },
    }
    if output_dir is not None:
        _write_result(Path(output_dir), case.name, result)
    return result


def run_real_papers(
    cases: list[RealPaperCase] | None = None,
    *,
    gateway: ModelGateway | None = None,
    use_model: bool = False,
    max_pdf_pages: int = 8,
    max_translate_spans: int = 12,
    max_reader_spans: int = 180,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    selected_cases = cases or DEFAULT_REAL_PAPERS
    runs = [
        run_real_paper_case(
            case,
            gateway=gateway,
            use_model=use_model,
            max_pdf_pages=max_pdf_pages,
            max_translate_spans=max_translate_spans,
            max_reader_spans=max_reader_spans,
            output_dir=output_dir,
        )
        for case in selected_cases
    ]
    failures = [
        FailureRecord(
            task=evaluation["name"],
            label=reason,
            scenario_id=run["case"]["name"],
            model=run["model_outputs"]["translation"]["model"],
            severity="high",
            root_cause=_root_cause(reason),
            fix_attempted=True,
        )
        for run in runs
        for evaluation in run["evaluations"]
        for reason in evaluation["reasons"]
    ]
    summary = {
        "passed": all(run["passed"] for run in runs),
        "paper_count": len(runs),
        "fine_tuning": fine_tuning_gate(failures) if use_model else _no_model_fine_tuning_decision(),
        "runs": runs,
    }
    if output_dir is not None:
        _write_result(Path(output_dir), "summary", summary)
    return summary


def _flatten_reader_spans(document: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for section in document.get("sections", []):
        for paragraph in section.get("paragraphs", []):
            for span in paragraph.get("spans", []):
                spans.append(
                    {
                        "id": span.get("id", ""),
                        "original": span.get("original", ""),
                        "translated": span.get("translated", ""),
                        "position": len(spans),
                    }
                )
    return spans


def _selected_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(spans) <= 3:
        return [{**span, "position_label": f"span-{index + 1}"} for index, span in enumerate(spans)]
    anchors = [
        ("front", max(0, len(spans) // 10)),
        ("middle", len(spans) // 2),
        ("end", max(0, len(spans) - 2)),
    ]
    selected = []
    seen = set()
    for label, anchor in anchors:
        position = _nearest_informative_position(spans, anchor, seen)
        if position in seen:
            continue
        seen.add(position)
        selected.append({**spans[position], "position_label": label})
    return selected


def _nearest_informative_position(spans: list[dict[str, Any]], anchor: int, seen: set[int]) -> int:
    window = max(12, min(36, len(spans) // 5))
    start = max(0, anchor - window)
    end = min(len(spans), anchor + window + 1)
    candidates = [position for position in range(start, end) if position not in seen]
    if not candidates:
        return min(anchor, len(spans) - 1)
    return max(
        candidates,
        key=lambda position: (_span_information_score(spans[position]), -abs(position - anchor), -position),
    )


def _span_information_score(span: dict[str, Any]) -> float:
    text = str(span.get("original", "")).strip()
    lower = text.lower()
    length = len(text)
    score = min(length, 240) / 60
    if length < 45:
        score -= 3
    if length > 420:
        score -= 1
    for marker in (
        "we ",
        "propose",
        "show",
        "result",
        "experiment",
        "method",
        "model",
        "baseline",
        "metric",
        "limitation",
        "improve",
        "reduce",
        "increase",
    ):
        if marker in lower:
            score += 1
    if any(char.isdigit() for char in text):
        score += 1
    if any(char.isupper() for char in text):
        score += 0.5
    if lower.startswith(("table ", "figure ", "fig. ", "http", "www.")):
        score -= 2
    if "@" in text or "copyright" in lower or "permission" in lower:
        score -= 3
    return score


def _evaluate_run(
    selected: list[dict[str, Any]],
    translation_data: dict[str, Any],
    qa_runs: list[dict[str, Any]],
    experiment_data: dict[str, Any],
    growth_data: dict[str, Any],
    memories: list[dict[str, Any]],
) -> list[EvalResult]:
    evals: list[EvalResult] = []
    expected_ids = [item["id"] for item in selected]
    evals.append(
        evaluate_translation(
            " ".join(item["original"] for item in selected),
            translation_data,
            expected_span_ids=expected_ids,
        )
    )
    for run in qa_runs:
        span = run["span"]
        evals.append(
            evaluate_grounded_qa(
                run["result"]["data"],
                span["id"],
                source_evidence=run.get("source_evidence") or {span["id"]: span["original"]},
                require_needs_more_context=False,
            )
        )
    evals.append(evaluate_lost_in_the_middle(qa_runs))
    evals.append(evaluate_experiment_spec(experiment_data))
    known_ids = {item.get("id", "") for item in memories}
    known_ids.update({"paper:selected-middle", "run:r1"})
    evals.append(evaluate_growth_ideas(growth_data, known_evidence_ids=known_ids, require_multiple_sources=True))
    return evals


def evaluate_pdf_parse(source: PaperSource, document: dict[str, Any], spans: list[dict[str, Any]]) -> EvalResult:
    reasons: list[str] = []
    if source.pdf_url and source.text.count("[page ") < 1:
        reasons.append("missing PDF page markers")
    if source.pdf_url and len(source.text) < 6000:
        reasons.append("PDF text is too short for real-paper validation")
    if source.warnings:
        reasons.extend(source.warnings)
    if len(spans) < 30:
        reasons.append("reader produced fewer than 30 visible spans")
    span_ids = [span["id"] for span in spans]
    if len(span_ids) != len(set(span_ids)):
        reasons.append("duplicate reader span ids")
    if document.get("metadata", {}).get("readerSpanCount") != len(spans):
        reasons.append("reader metadata does not match visible spans")
    return EvalResult("pdf_parse_and_reader_spans", not reasons, reasons)


def evaluate_lost_in_the_middle(qa_runs: list[dict[str, Any]]) -> EvalResult:
    reasons: list[str] = []
    middle_runs = [run for run in qa_runs if run["span"].get("position_label") == "middle"]
    if not middle_runs and len(qa_runs) >= 3:
        middle_runs = [qa_runs[1]]
    if not middle_runs:
        reasons.append("missing middle-position QA run")
    for run in middle_runs:
        span_id = run["span"]["id"]
        data = run["result"]["data"]
        evidence = data.get("evidence", []) if isinstance(data, dict) else []
        cited = {item.get("source_id") for item in evidence if isinstance(item, dict)}
        if span_id not in cited:
            reasons.append(f"middle span {span_id} was not cited")
        if run["result"].get("used_fallback"):
            reasons.append("middle QA used fallback instead of model output")
    return EvalResult("lost_in_the_middle", not reasons, reasons)


def evaluate_model_backing(
    translation: Any,
    qa_runs: list[dict[str, Any]],
    experiment: Any,
    growth: Any,
) -> EvalResult:
    reasons: list[str] = []
    if translation.used_fallback:
        reasons.append("translation used fallback")
    for run in qa_runs:
        if run["result"].get("used_fallback"):
            reasons.append(f"qa {run['span']['id']} used fallback")
    if experiment.used_fallback:
        reasons.append("experiment used fallback")
    if growth.used_fallback:
        reasons.append("growth used fallback")
    return EvalResult("model_backing", not reasons, reasons)


def _failure_records(
    scenario_id: str,
    evals: list[EvalResult],
    qa_runs: list[dict[str, Any]],
    experiment: Any,
    growth: Any,
) -> list[FailureRecord]:
    model = experiment.model or growth.model or "unknown"
    records: list[FailureRecord] = []
    for item in evals:
        for reason in item.reasons:
            records.append(
                FailureRecord(
                    task=item.name,
                    label=reason,
                    scenario_id=scenario_id,
                    model=model,
                    severity="high" if item.name in {"grounded_qa", "lost_in_the_middle"} else "medium",
                    root_cause=_root_cause(reason),
                    fix_attempted=True,
                )
            )
    for run in qa_runs:
        if run["result"].get("used_fallback"):
            records.append(
                FailureRecord(
                    task="grounded_qa",
                    label=f"{run['span']['id']} used fallback",
                    scenario_id=scenario_id,
                    model=run["result"].get("model", "unknown"),
                    severity="medium",
                    root_cause="model_capability",
                    fix_attempted=True,
                )
            )
    return records


def _root_cause(reason: str) -> str:
    lowered = reason.lower()
    if "schema" in lowered or "json" in lowered or "missing" in lowered:
        return "schema"
    if "translation" in lowered or "term" in lowered:
        return "terminology"
    if "fallback" in lowered or "middle" in lowered or "unsupported" in lowered:
        return "model_capability"
    return "retrieval"


def _no_model_fine_tuning_decision() -> dict[str, Any]:
    return {
        "recommendation": "no",
        "reason": "Model-backed validation was not enabled, so fallback behavior cannot justify fine-tuning.",
        "repeated_failures": [],
    }


def _public_result(result: Any) -> dict[str, Any]:
    return {
        "task": result.task,
        "provider": result.provider,
        "model": result.model,
        "trace_id": result.trace_id,
        "error": result.error,
        "used_fallback": result.used_fallback,
        "data": result.data,
    }


def _write_result(output_dir: Path, name: str, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _case_by_name_or_arxiv(value: str) -> RealPaperCase:
    for case in DEFAULT_REAL_PAPERS:
        if value in {case.name, case.arxiv}:
            return case
    return RealPaperCase(
        name=value.replace("/", "_").replace(".", "_"),
        arxiv=value,
        question=DEFAULT_REAL_PAPERS[0].question,
        idea=DEFAULT_REAL_PAPERS[0].idea,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaperLens Lab validation on actual arXiv PDFs.")
    parser.add_argument("--paper", action="append", default=[], help="arXiv id or bundled case name. Repeatable.")
    parser.add_argument("--use-model", action="store_true", help="Call the configured model provider.")
    parser.add_argument("--provider", default=None, help="Override PAPERLENS_PROVIDER.")
    parser.add_argument("--model", default=None, help="Override PAPERLENS_MODEL.")
    parser.add_argument("--quality-model", default=None, help="Override PAPERLENS_QUALITY_MODEL.")
    parser.add_argument("--max-pdf-pages", type=int, default=8)
    parser.add_argument("--max-translate-spans", type=int, default=12)
    parser.add_argument("--max-reader-spans", type=int, default=180)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    gateway = ModelGateway(
        provider=args.provider or DEFAULT_PROVIDER,
        model_id=args.model or DEFAULT_MODEL,
        quality_model_id=args.quality_model or QUALITY_MODEL,
    )
    cases = [_case_by_name_or_arxiv(item) for item in args.paper] if args.paper else DEFAULT_REAL_PAPERS
    result = run_real_papers(
        cases,
        gateway=gateway,
        use_model=args.use_model,
        max_pdf_pages=args.max_pdf_pages,
        max_translate_spans=args.max_translate_spans,
        max_reader_spans=args.max_reader_spans,
        output_dir=args.output_dir,
    )
    if args.compact:
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "paper_count": result["paper_count"],
                    "fine_tuning": result["fine_tuning"],
                    "papers": [
                        {
                            "name": run["case"]["name"],
                            "title": run["source"]["title"],
                            "passed": run["passed"],
                            "text_chars": run["source"]["text_chars"],
                            "page_marker_count": run["source"]["page_marker_count"],
                            "visible_span_count": run["reader"]["visible_span_count"],
                            "selected_span_positions": run["reader"]["selected_span_positions"],
                        }
                        for run in result["runs"]
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
