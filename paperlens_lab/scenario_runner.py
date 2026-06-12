from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from .model_adapter import DEFAULT_MODEL, DEFAULT_PROVIDER, QUALITY_MODEL, ModelGateway
from .scenario_eval import (
    EvalResult,
    evaluate_experiment_spec,
    evaluate_grounded_qa,
    evaluate_growth_ideas,
    evaluate_translation,
    fine_tuning_gate,
)


@dataclass(frozen=True)
class PaperScenario:
    name: str
    title: str
    source_text: str
    span_id: str
    selected_span: str
    question: str
    idea: str
    locale: str = "ko"
    translated_span: str = ""


def default_scenarios() -> list[PaperScenario]:
    return [
        PaperScenario(
            name="evidence_reranking",
            title="Evidence-Linked Reranking for Small RAG Systems",
            source_text=(
                "We propose evidence-linked reranking for retrieval augmented generation. "
                "The method improves top-5 precision by 3.2 points over a relevance-only baseline on 128 queries. "
                "Limitations include weak performance on ambiguous questions and sensitivity to noisy citations."
            ),
            span_id="P0.S1",
            selected_span=(
                "The method improves top-5 precision by 3.2 points over a relevance-only baseline on 128 queries."
            ),
            question="이 결과가 정확히 무엇을 비교한 거야?",
            idea="Try a tiny evidence-linked reranking ablation.",
        ),
        PaperScenario(
            name="adapter_ablation",
            title="Tiny Adapter Ablations for Code Repair",
            source_text=(
                "We train a 4B parameter repair model with rank-8 LoRA adapters on 640 curated bug-fix pairs. "
                "The adapter improves pass@1 by 8.5% compared with prompting the base model. "
                "The gain disappears when the validation set contains unseen library APIs."
            ),
            span_id="P0.S1",
            selected_span="The adapter improves pass@1 by 8.5% compared with prompting the base model.",
            question="이 문장이 파인튜닝이 꼭 필요하다는 뜻이야?",
            idea="Compare prompting-only repair with a small adapter-style pattern memory.",
        ),
        PaperScenario(
            name="multilingual_reading",
            title="Cross-Lingual Paper Reading With Grounded Glossaries",
            source_text=(
                "The system keeps English method terms visible while translating explanations into Korean. "
                "A glossary-constrained decoder reduces terminology drift from 17% to 6% in a 50-paper study. "
                "However, it still fails on equations and undefined acronyms."
            ),
            span_id="P0.S1",
            selected_span=(
                "A glossary-constrained decoder reduces terminology drift from 17% to 6% in a 50-paper study."
            ),
            question="용어 보존이 왜 중요한 거야?",
            idea="Build a glossary-preservation smoke test for translated paper spans.",
        ),
    ]


def run_scenario(
    scenario: PaperScenario,
    gateway: ModelGateway | None = None,
    *,
    use_model: bool = False,
) -> dict[str, Any]:
    gateway = gateway or ModelGateway()
    translation = gateway.translate_spans(
        scenario.title,
        [{"span_id": scenario.span_id, "text": scenario.selected_span}],
        locale=scenario.locale,
        use_model=use_model,
    )
    translated_span = _first_translation(translation.data) or scenario.translated_span
    qa = gateway.answer_span(
        scenario.title,
        scenario.span_id,
        scenario.selected_span,
        translated_span,
        scenario.question,
        scenario.source_text,
        scenario.locale,
        use_model=use_model,
    )
    experiment = gateway.experiment_spec(
        scenario.title,
        scenario.selected_span,
        translated_span,
        scenario.source_text,
        scenario.idea,
        scenario.locale,
        use_model=use_model,
    )
    growth = gateway.growth_ideas(
        scenario.title,
        paper_memory=[
            {
                "id": "paper:s1",
                "span_id": scenario.span_id,
                "summary": scenario.selected_span,
                "translation": translated_span,
            }
        ],
        mini_lab_result=experiment.text,
        selected_span=scenario.selected_span,
        locale=scenario.locale,
        use_model=use_model,
    )

    evals = [
        evaluate_translation(scenario.selected_span, translation.data, expected_span_ids=[scenario.span_id]),
        evaluate_grounded_qa(
            qa.data,
            scenario.span_id,
            source_evidence={scenario.span_id: scenario.selected_span},
        ),
        evaluate_experiment_spec(experiment.data),
        evaluate_growth_ideas(
            growth.data,
            known_evidence_ids={scenario.span_id, "paper:s1", "run:r1"},
            require_multiple_sources=True,
        ),
    ]
    failures = _failure_labels(evals)
    return {
        "scenario": asdict(scenario),
        "passed": all(result.passed for result in evals),
        "evaluations": [asdict(result) for result in evals],
        "fine_tuning": fine_tuning_gate(failures),
        "model_outputs": {
            "translation": _public_result(translation),
            "qa": _public_result(qa),
            "experiment": _public_result(experiment),
            "growth": _public_result(growth),
        },
    }


def run_scenarios(
    scenarios: list[PaperScenario] | None = None,
    gateway: ModelGateway | None = None,
    *,
    use_model: bool = False,
) -> dict[str, Any]:
    scenario_list = scenarios or default_scenarios()
    runs = [run_scenario(scenario, gateway=gateway, use_model=use_model) for scenario in scenario_list]
    failures = [
        reason
        for run in runs
        for result in run["evaluations"]
        for reason in result["reasons"]
        if not result["passed"]
    ]
    return {
        "passed": all(run["passed"] for run in runs),
        "scenario_count": len(runs),
        "fine_tuning": fine_tuning_gate(failures),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaperLens Lab backend model scenarios.")
    parser.add_argument("--use-model", action="store_true", help="Call the configured model provider.")
    parser.add_argument("--provider", default=None, help="Override PAPERLENS_PROVIDER for this run.")
    parser.add_argument("--model", default=None, help="Override PAPERLENS_MODEL for this run.")
    parser.add_argument("--quality-model", default=None, help="Override PAPERLENS_QUALITY_MODEL for this run.")
    parser.add_argument("--compact", action="store_true", help="Print only scenario pass/fail summary.")
    args = parser.parse_args()

    gateway = ModelGateway(
        provider=args.provider or DEFAULT_PROVIDER,
        model_id=args.model or DEFAULT_MODEL,
        quality_model_id=args.quality_model or QUALITY_MODEL,
    )
    result = run_scenarios(gateway=gateway, use_model=args.use_model)
    if args.compact:
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "scenario_count": result["scenario_count"],
                    "fine_tuning": result["fine_tuning"],
                    "scenarios": [
                        {"name": run["scenario"]["name"], "passed": run["passed"]}
                        for run in result["runs"]
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _first_translation(data: dict[str, Any]) -> str:
    translations = data.get("translations", [])
    if translations and isinstance(translations[0], dict):
        return str(translations[0].get("translation", ""))
    return ""


def _failure_labels(evals: list[EvalResult]) -> list[str]:
    labels: list[str] = []
    for result in evals:
        if not result.passed:
            labels.extend(result.reasons or [result.name])
    return labels


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


if __name__ == "__main__":
    main()
