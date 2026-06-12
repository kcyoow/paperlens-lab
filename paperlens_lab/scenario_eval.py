from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalResult:
    name: str
    passed: bool
    reasons: list[str]


def evaluate_translation(source: str, translation_data: dict[str, Any]) -> EvalResult:
    reasons: list[str] = []
    translations = translation_data.get("translations", [])
    if not translations:
        reasons.append("missing translations")
    for number in _numbers(source):
        joined = " ".join(str(item.get("translation", "")) for item in translations)
        if number not in joined:
            reasons.append(f"changed or dropped number {number}")
    if any("translation" not in item for item in translations):
        reasons.append("translation item missing translation field")
    return EvalResult("translation_fidelity", not reasons, reasons)


def evaluate_grounded_qa(answer_data: dict[str, Any], expected_span_id: str) -> EvalResult:
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
    return EvalResult("experiment_spec", not reasons, reasons)


def evaluate_growth_ideas(data: dict[str, Any]) -> EvalResult:
    reasons: list[str] = []
    ideas = data.get("ideas", [])
    if not ideas:
        reasons.append("missing ideas")
    for idx, idea in enumerate(ideas, start=1):
        if not idea.get("source_evidence"):
            reasons.append(f"idea {idx} missing evidence")
        if not idea.get("testable_next_step"):
            reasons.append(f"idea {idx} missing testable next step")
        if not idea.get("risk"):
            reasons.append(f"idea {idx} missing risk")
    return EvalResult("growth_ideas", not reasons, reasons)


def fine_tuning_gate(failures: list[str]) -> dict[str, Any]:
    counts = {failure: failures.count(failure) for failure in set(failures)}
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


def _numbers(text: str) -> list[str]:
    import re

    return re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
