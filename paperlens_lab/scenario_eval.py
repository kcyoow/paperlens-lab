from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import re
from typing import Any


@dataclass
class EvalResult:
    name: str
    passed: bool
    reasons: list[str]


@dataclass(frozen=True)
class FailureRecord:
    task: str
    label: str
    scenario_id: str
    model: str = "unknown"
    severity: str = "medium"
    root_cause: str = "unknown"
    fix_attempted: bool = False


def evaluate_translation(
    source: str,
    translation_data: dict[str, Any],
    expected_span_ids: list[str] | None = None,
) -> EvalResult:
    reasons: list[str] = []
    translations = translation_data.get("translations", [])
    if not translations:
        reasons.append("missing translations")
    if expected_span_ids is not None:
        ids = [item.get("span_id") for item in translations if isinstance(item, dict)]
        for span_id in expected_span_ids:
            if ids.count(span_id) != 1:
                reasons.append(f"span {span_id} does not map to exactly one translation")
    joined = " ".join(str(item.get("translation", "")) for item in translations if isinstance(item, dict))
    for number in _numbers(source):
        if number not in joined:
            reasons.append(f"changed or dropped number {number}")
    for marker in _citation_markers(source):
        if not _marker_preserved(marker, joined):
            reasons.append(f"changed or dropped citation/table marker {marker}")
    for term in _technical_terms(source):
        if not _term_preserved(term, joined):
            reasons.append(f"changed or dropped technical term {term}")
    if _has_negation_or_limit(source) and not _has_negation_or_limit(joined):
        reasons.append("lost negation, limitation, or result qualifier")
    if any("translation" not in item or not item.get("translation") for item in translations):
        reasons.append("translation item missing translation field")
    if _adds_unsupported_strength(source, joined):
        reasons.append("translation adds unsupported strong claim")
    return EvalResult("translation_fidelity", not reasons, reasons)


def evaluate_grounded_qa(
    answer_data: dict[str, Any],
    expected_span_id: str,
    source_evidence: dict[str, str] | None = None,
    require_needs_more_context: bool = False,
) -> EvalResult:
    reasons: list[str] = []
    answer = str(answer_data.get("answer", ""))
    evidence = answer_data.get("evidence", [])
    ids = [item.get("source_id") for item in evidence if isinstance(item, dict)]
    if not answer:
        reasons.append("missing answer")
    if expected_span_id not in ids:
        reasons.append("selected span is not cited")
    if answer_data.get("confidence") not in {"high", "medium", "low"}:
        reasons.append("missing confidence label")
    if require_needs_more_context:
        if answer_data.get("needs_more_context") is not True:
            reasons.append("should ask for more context")
        if answer_data.get("confidence") == "high":
            reasons.append("unsupported question should not have high confidence")
    if source_evidence:
        known_ids = {str(source_id) for source_id in source_evidence}
        for source_id in ids:
            if str(source_id) not in known_ids:
                reasons.append(f"answer cites unknown evidence {source_id}")
        for item in evidence:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id")
            quote = str(item.get("quote", "")).strip()
            source_text = source_evidence.get(str(source_id), "")
            if quote and source_text and not source_contains_quote(source_text, quote):
                reasons.append(f"quote for {source_id} is not in source evidence")
    joined_evidence = " ".join(source_evidence.values()) if source_evidence else ""
    if _adds_unsupported_strength(joined_evidence, answer):
        unsupported = _flatten_text(answer_data.get("unsupported_assumptions", ""))
        if not (answer_data.get("needs_more_context") and _mentions_strong_marker(unsupported)):
            reasons.append("answer adds unsupported strong claim")
    return EvalResult("grounded_qa", not reasons, reasons)


def evaluate_experiment_spec(spec: dict[str, Any]) -> EvalResult:
    reasons: list[str] = []
    required = ["research_question", "mini_lab_goal", "dataset", "baseline", "metric", "steps"]
    for key in required:
        if not spec.get(key):
            reasons.append(f"missing {key}")
    if len(spec.get("steps", [])) < 3:
        reasons.append("mini-lab needs at least three steps")
    if not spec.get("failure_condition"):
        reasons.append("missing failure condition")
    dataset_text = _flatten_text(spec.get("dataset", ""))
    spec_text = _flatten_text(spec)
    if any(term in spec_text.lower() for term in ("8xa100", "a100", "gpu cluster", "proprietary dataset")):
        if not any(term in dataset_text.lower() for term in ("toy", "hand-built", "fallback", "small", "sample")):
            reasons.append("large or proprietary setup needs a small dataset fallback")
    if "metric" in spec and spec.get("metric") and spec.get("failure_condition"):
        metric_head = _normalize_metric_token(str(spec["metric"]).split(",")[0].split()[0])
        failure_text = _normalize_metric_token(str(spec["failure_condition"]))
        if metric_head and metric_head not in failure_text and "metric" not in failure_text:
            reasons.append("failure condition should reference the metric")
    if spec.get("ablation"):
        ablation = str(spec["ablation"]).lower()
        if not any(term in ablation for term in ("remove", "disable", "only", "one", "without", "isolate")):
            reasons.append("ablation should isolate one variable")
    return EvalResult("experiment_spec", not reasons, reasons)


def evaluate_growth_ideas(
    data: dict[str, Any],
    known_evidence_ids: set[str] | None = None,
    require_multiple_sources: bool = False,
) -> EvalResult:
    reasons: list[str] = []
    ideas = data.get("ideas", [])
    if not ideas:
        reasons.append("missing ideas")
    has_multi_source_idea = False
    for idx, idea in enumerate(ideas, start=1):
        evidence_ids = idea.get("source_evidence") or []
        if not evidence_ids:
            reasons.append(f"idea {idx} missing evidence")
        if known_evidence_ids:
            missing = [source_id for source_id in evidence_ids if source_id not in known_evidence_ids]
            if missing:
                reasons.append(f"idea {idx} cites unknown evidence {', '.join(missing)}")
        if len(set(evidence_ids)) >= 2:
            has_multi_source_idea = True
        if not idea.get("testable_next_step"):
            reasons.append(f"idea {idx} missing testable next step")
        if not idea.get("risk"):
            reasons.append(f"idea {idx} missing risk")
        if _looks_like_restatement(str(idea.get("idea", "")), str(idea.get("testable_next_step", ""))):
            reasons.append(f"idea {idx} restates the test instead of adding a direction")
    if require_multiple_sources and not has_multi_source_idea:
        reasons.append("growth mode should combine at least two evidence sources")
    return EvalResult("growth_ideas", not reasons, reasons)


def fine_tuning_gate(failures: list[str | FailureRecord]) -> dict[str, Any]:
    if failures and all(isinstance(failure, FailureRecord) for failure in failures):
        return _fine_tuning_gate_records([failure for failure in failures if isinstance(failure, FailureRecord)])
    labels = [failure if isinstance(failure, str) else failure.label for failure in failures]
    counts = {failure: labels.count(failure) for failure in set(labels)}
    repeated = sorted(name for name, count in counts.items() if count >= 3)
    if not repeated:
        return {
            "recommendation": "no",
            "reason": "Fix prompt, retrieval, parsing, or scenario coverage before fine-tuning.",
            "repeated_failures": [],
        }
    return {
        "recommendation": "maybe",
        "reason": "Repeated model-output failures survived the same task boundary and may justify a tiny task-specific fine-tune.",
        "repeated_failures": repeated,
    }


def _fine_tuning_gate_records(failures: list[FailureRecord]) -> dict[str, Any]:
    trainable_causes = {"schema", "style", "terminology", "model_capability"}
    eligible = [
        failure
        for failure in failures
        if failure.fix_attempted and failure.root_cause in trainable_causes and failure.severity in {"medium", "high"}
    ]
    grouped = Counter((failure.task, failure.label, failure.model) for failure in eligible)
    repeated = [
        {"task": task, "label": label, "model": model, "count": count}
        for (task, label, model), count in sorted(grouped.items())
        if count >= 3
    ]
    if repeated:
        return {
            "recommendation": "maybe",
            "reason": "Repeated task-specific failures remain after prompt/RAG/parser fixes; prepare a tiny SFT/LoRA probe.",
            "repeated_failures": repeated,
        }
    untried = [
        failure
        for failure in failures
        if failure.root_cause in trainable_causes and not failure.fix_attempted
    ]
    return {
        "recommendation": "no",
        "reason": (
            "Try prompt, retrieval, parser, or rubric fixes before fine-tuning."
            if untried
            else "Failures do not yet point to a trainable model-style or terminology gap."
        ),
        "repeated_failures": [],
    }


def _numbers(text: str) -> list[str]:
    import re

    return re.findall(r"\b\d+(?:\.\d+)?%?\b", text)


def _citation_markers(text: str) -> list[str]:
    import re

    markers = re.findall(r"\[[0-9,\s-]+\]", text)
    markers.extend(re.findall(r"\b(?:Table|Figure|Fig\.)\s+\d+\b", text))
    return markers


def _marker_preserved(marker: str, output: str) -> bool:
    if marker in output:
        return True
    match = re.fullmatch(r"(Table|Figure|Fig\.)\s+(\d+)", marker)
    if not match:
        return False
    label, number = match.groups()
    if label == "Table":
        return bool(re.search(rf"(?:Table|표)\s*{re.escape(number)}(?!\d)", output, re.IGNORECASE))
    return bool(re.search(rf"(?:Figure|Fig\.|그림)\s*{re.escape(number)}(?!\d)", output, re.IGNORECASE))


def source_contains_quote(source_text: str, quote: str) -> bool:
    if quote in source_text:
        return True
    return _quote_match_text(quote) in _quote_match_text(source_text)


def _quote_match_text(text: str) -> str:
    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r",(?=[A-Za-z0-9])", ", ", text)
    return text.strip().casefold()


def _technical_terms(text: str) -> list[str]:
    import re

    patterns = [
        r"\b[A-Za-z]+-\d+(?:\.\d+)?-[A-Za-z0-9]+\b",
        r"\b[A-Z]{2,}[A-Za-z0-9-]*\b",
        r"\bp\s*<\s*0\.\d+\b",
    ]
    terms: list[str] = []
    for pattern in patterns:
        terms.extend(re.findall(pattern, text))
    return list(dict.fromkeys(terms))


def _has_negation_or_limit(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "does not",
            "do not",
            "no ",
            "not ",
            "fails",
            "failed",
            "weak",
            "limitation",
            "however",
            "only",
            "disappears",
            "않",
            "없",
            "아니",
            "만",
            "실패",
            "제한",
            "그러나",
            "하지만",
            "약함",
            "사라",
        )
    )


def _adds_unsupported_strength(source: str, output: str) -> bool:
    source_lower = source.lower()
    output_lower = output.lower()
    return any(
        marker in output_lower and not _strong_marker_supported(source_lower, marker)
        for marker in _strong_markers()
    )


def _mentions_strong_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _strong_markers())


def _strong_markers() -> tuple[str, ...]:
    return ("state-of-the-art", "sota", "proves", "guarantees", "입증", "증명", "최고", "완벽")


def _strong_marker_supported(source_lower: str, marker: str) -> bool:
    if marker in source_lower:
        return True
    equivalents = {
        "sota": ("state-of-the-art",),
        "state-of-the-art": ("sota",),
        "입증": ("prove", "proves", "proven", "demonstrate", "demonstrates", "demonstrated"),
        "증명": ("prove", "proves", "proven", "demonstrate", "demonstrates", "demonstrated"),
        "최고": ("best", "state-of-the-art", "sota", "top-performing", "top performing"),
        "완벽": ("perfect", "perfectly", "guarantee", "guarantees"),
    }
    return any(_contains_phrase(source_lower, term) for term in equivalents.get(marker, ()))


def _contains_phrase(text: str, phrase: str) -> bool:
    if phrase.isascii() and phrase.replace("-", "").replace(" ", "").isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None
    return phrase in text


def _term_preserved(term: str, output: str) -> bool:
    if term in output:
        return True
    term_lower = term.lower()
    output_lower = output.lower()
    if term_lower in output_lower:
        return True
    equivalents = {
        "qa": ("question answering", "질문 응답", "질의응답", "질문응답", "문답"),
    }
    if any(alias in output_lower for alias in equivalents.get(term_lower, ())):
        return True
    if term_lower.endswith("s") and len(term_lower) > 3 and term_lower[:-1] in output_lower:
        return True
    return False


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def _looks_like_restatement(idea: str, next_step: str) -> bool:
    idea_words = {word for word in idea.lower().split() if len(word) > 4}
    next_words = {word for word in next_step.lower().split() if len(word) > 4}
    if not idea_words or not next_words:
        return False
    overlap = len(idea_words & next_words) / max(1, len(idea_words))
    return overlap > 0.85


def _normalize_metric_token(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")
