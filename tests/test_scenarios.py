import unittest

from paperlens_lab.scenario_eval import (
    FailureRecord,
    evaluate_experiment_spec,
    evaluate_grounded_qa,
    evaluate_growth_ideas,
    evaluate_starter_code,
    evaluate_starter_grounding,
    evaluate_translation,
    fine_tuning_gate,
)


class ScenarioEvalTests(unittest.TestCase):
    def test_translation_preserves_numbers(self):
        result = evaluate_translation(
            "The method improves F1 by 3.2 points on 128 examples.",
            {
                "translations": [
                    {
                        "span_id": "P0.S1",
                        "translation": "이 방법은 128개 예제에서 F1을 3.2점 향상시킨다.",
                    }
                ]
            },
        )
        self.assertTrue(result.passed, result.reasons)

    def test_translation_flags_lost_direction_and_markers(self):
        result = evaluate_translation(
            "Llama-3.1-8B does not improve F1 in Table 2 [14], with p < 0.05.",
            {
                "translations": [
                    {
                        "span_id": "P0.S1",
                        "translation": "Llama 모델은 F1을 크게 향상시킨다.",
                    }
                ]
            },
            expected_span_ids=["P0.S1"],
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("dropped number" in reason for reason in result.reasons))
        self.assertTrue(any("negation" in reason for reason in result.reasons))

    def test_translation_allows_supported_cross_lingual_superlatives(self):
        result = evaluate_translation(
            "The model reaches a single-model state-of-the-art BLEU score after eight GPUs.",
            {
                "translations": [
                    {
                        "span_id": "P0.S1",
                        "translation": "이 모델은 8개의 GPU 이후 단일 모델 기준 최고 BLEU 점수에 도달한다.",
                    }
                ]
            },
        )

        self.assertTrue(result.passed, result.reasons)

    def test_translation_allows_localized_figure_marker(self):
        result = evaluate_translation(
            "A very low rank (i.e., r in Figure 1 can be one or two) suffices.",
            {
                "translations": [
                    {
                        "span_id": "P0.S1",
                        "translation": "매우 낮은 랭크, 즉 그림 1에서 r이 1 또는 2이면 충분하다.",
                    }
                ]
            },
        )

        self.assertTrue(result.passed, result.reasons)

    def test_translation_allows_qa_as_korean_question_answering(self):
        result = evaluate_translation(
            "The model obtains strong results on open-domain QA.",
            {
                "translations": [
                    {
                        "span_id": "P0.S1",
                        "translation": "이 모델은 오픈 도메인 질문 응답에서 강한 결과를 얻었다.",
                    }
                ]
            },
        )

        self.assertTrue(result.passed, result.reasons)

    def test_translation_still_rejects_unsupported_proof_claims(self):
        result = evaluate_translation(
            "The method improves F1 in a controlled evidence setting.",
            {
                "translations": [
                    {
                        "span_id": "P0.S1",
                        "translation": "이 방법은 모든 환경에서 F1 향상을 증명한다.",
                    }
                ]
            },
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("unsupported strong claim" in reason for reason in result.reasons))

    def test_grounded_qa_requires_selected_span_evidence(self):
        result = evaluate_grounded_qa(
            {
                "answer": "선택 문장은 F1 개선 주장을 말한다.",
                "evidence": [{"source_id": "P0.S1", "quote": "improves F1"}],
                "confidence": "high",
            },
            "P0.S1",
        )
        self.assertTrue(result.passed, result.reasons)

    def test_grounded_qa_flags_overclaiming_when_context_is_missing(self):
        result = evaluate_grounded_qa(
            {
                "answer": "이 문장은 SOTA를 증명한다.",
                "evidence": [{"source_id": "P0.S1", "quote": "controlled evidence setting improves F1"}],
                "confidence": "high",
                "needs_more_context": False,
            },
            "P0.S1",
            source_evidence={"P0.S1": "In a controlled evidence setting, the method improves F1."},
            require_needs_more_context=True,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("more context" in reason for reason in result.reasons))
        self.assertTrue(any("strong claim" in reason for reason in result.reasons))

    def test_grounded_qa_allows_rejecting_strong_unsupported_claims(self):
        result = evaluate_grounded_qa(
            {
                "answer": "이 문장은 실험 범위만 말한다. RAG가 모든 작업에서 최고라는 주장은 추가 근거가 필요하다.",
                "evidence": [{"source_id": "P0.S1", "quote": "We experiment with RAG in a wide range of tasks."}],
                "confidence": "medium",
                "needs_more_context": True,
                "unsupported_assumptions": ["RAG가 모든 작업에서 최고라는 주장"],
            },
            "P0.S1",
            source_evidence={"P0.S1": "We experiment with RAG in a wide range of tasks."},
        )
        self.assertTrue(result.passed, result.reasons)

    def test_grounded_qa_allows_pdf_ligature_and_spacing_normalization(self):
        result = evaluate_grounded_qa(
            {
                "answer": "The source says a low rank can suffice.",
                "evidence": [
                    {
                        "source_id": "P0.S1",
                        "quote": "a very low rank (i.e., r in Figure 1 can be one or two) suffices",
                    }
                ],
                "confidence": "medium",
                "needs_more_context": False,
            },
            "P0.S1",
            source_evidence={
                "P0.S1": "a very low rank (i.e.,r in Figure 1 can be one or two) sufﬁces even"
            },
        )

        self.assertTrue(result.passed, result.reasons)

    def test_grounded_qa_rejects_unknown_evidence_ids(self):
        result = evaluate_grounded_qa(
            {
                "answer": "선택 문장과 별도 근거를 함께 설명한다.",
                "evidence": [
                    {"source_id": "P0.S1", "quote": "The selected claim is narrow."},
                    {"source_id": "S9", "quote": "A separate global sentence."},
                ],
                "confidence": "medium",
                "needs_more_context": False,
            },
            "P0.S1",
            source_evidence={"P0.S1": "The selected claim is narrow."},
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("unknown evidence S9" in reason for reason in result.reasons))

    def test_experiment_spec_requires_runnable_fields(self):
        result = evaluate_experiment_spec(
            {
                "research_question": "Does reranking help?",
                "mini_lab_goal": "Compare baseline and reranker.",
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index spans"},
                "baseline": "BM25",
                "metric": "top-5 precision",
                "steps": ["Prepare data", "Run baseline", "Run reranker"],
                "failure_condition": "No top-5 precision gain",
            }
        )
        self.assertTrue(result.passed, result.reasons)

    def test_starter_code_source_run_requires_runnable_run_rows(self):
        code = """
def baseline(example):
    return {"prediction": "base"}

def paper_inspired(example):
    return {"prediction": "proto"}

def score(output, expected):
    return 1.0 if output["prediction"] == expected else 0.0

def run(evidence_rows=None):
    examples = evidence_rows or []
    rows = []
    for example in examples:
        base = baseline(example)
        proto = paper_inspired(example)
        expected = "proto" if example.get("gold") else "base"
        baseline_score = score(base, expected)
        prototype_score = score(proto, expected)
        rows.append({
            "source_id": example["source_id"],
            "text_hash": example["text_hash"],
            "baseline_score": baseline_score,
            "prototype_score": prototype_score,
            "metric": "accuracy",
            "failure_condition": prototype_score <= baseline_score,
            "failure_rule": "prototype_score <= baseline_score",
        })
    return rows
"""
        result = evaluate_starter_code(code, evidence_rows=_source_rows(), require_evidence_rows=True)

        self.assertTrue(result.passed, result.reasons)

    def test_starter_code_source_run_rejects_missing_run_contract(self):
        result = evaluate_starter_code("def baseline(example):\n    return {}\n")

        self.assertFalse(result.passed)
        self.assertTrue(any("run()" in reason for reason in result.reasons))

    def test_starter_code_source_run_rejects_failure_flag_mismatch(self):
        code = """
def baseline(example):
    return {"prediction": "base"}

def paper_inspired(example):
    return {"prediction": "proto"}

def score(output, expected):
    return 1.0 if output["prediction"] == expected else 0.0

def run(evidence_rows=None):
    return [
        {
            "source_id": "P0.S1",
            "baseline_score": 1.0,
            "prototype_score": 1.0,
            "metric": "accuracy",
            "failure_condition": False,
            "failure_rule": "prototype_score <= baseline_score",
        },
        {
            "source_id": "P0.S2",
            "baseline_score": 0.0,
            "prototype_score": 1.0,
            "metric": "accuracy",
            "failure_condition": False,
            "failure_rule": "prototype_score <= baseline_score",
        },
    ]
"""
        result = evaluate_starter_code(code, evidence_rows=_source_rows(), require_evidence_rows=True)

        self.assertFalse(result.passed)
        self.assertTrue(any("failure_condition must match" in reason for reason in result.reasons))

    def test_starter_grounding_rejects_unrelated_generic_examples(self):
        code = """
SELECTED_SPAN = "compact evidence reranker improves top five precision"

def baseline(example):
    return {"prediction": example["candidates"][0]}

def paper_inspired(example):
    return {"prediction": "Paris"}

def score(output, expected):
    return 1.0 if output["prediction"] == expected else 0.0

def run():
    return [
        {
            "query": "What is the capital of France?",
            "context": SELECTED_SPAN,
            "candidates": ["Paris", "London"],
            "baseline_score": 0.0,
            "prototype_score": 1.0,
            "metric": "Top-5 Precision",
            "failure_condition": False,
            "failure_rule": "prototype_score <= baseline_score",
        },
        {
            "query": "Who wrote Hamlet?",
            "context": "contrast control",
            "candidates": ["Shakespeare", "Marlowe"],
            "baseline_score": 1.0,
            "prototype_score": 1.0,
            "metric": "Top-5 Precision",
            "failure_condition": True,
            "failure_rule": "prototype_score <= baseline_score",
        },
    ]
"""
        result = evaluate_starter_grounding(
            code,
            "compact evidence reranker improves top five precision",
        )

        self.assertFalse(result.passed)
        self.assertIn("starter code uses unrelated generic examples", result.reasons)

    def test_starter_code_source_run_allows_math_and_isinstance(self):
        code = """
import math

def baseline(example):
    values = example.get("values", [])
    if not isinstance(values, list) or not values:
        return {"prediction": 0.0}
    return {"prediction": values[0]}

def paper_inspired(example):
    values = example.get("values", [])
    norm = math.sqrt(sum(value * value for value in values)) if values else 0.0
    return {"prediction": round(norm, 3)}

def score(output, expected):
    return 1.0 if output["prediction"] == expected else 0.0

def run(evidence_rows=None):
    examples = [
        {**row, "values": [3.0, 4.0], "expected": 5.0}
        if row.get("gold")
        else {**row, "values": [2.0, 2.0], "expected": 2.0}
        for row in (evidence_rows or [])
    ]
    rows = []
    for example in examples:
        base = baseline(example)
        proto = paper_inspired(example)
        baseline_score = score(base, example["expected"])
        prototype_score = score(proto, example["expected"])
        rows.append({
            "source_id": example["source_id"],
            "text_hash": example["text_hash"],
            "baseline_score": baseline_score,
            "prototype_score": prototype_score,
            "metric": "source evidence accuracy",
            "failure_condition": prototype_score <= baseline_score,
            "failure_rule": "prototype_score <= baseline_score",
        })
    return rows
"""
        result = evaluate_starter_code(code, evidence_rows=_source_rows(), require_evidence_rows=True)

        self.assertTrue(result.passed, result.reasons)

    def test_experiment_spec_rejects_large_setup_without_indexed_evidence(self):
        result = evaluate_experiment_spec(
            {
                "research_question": "Does the method scale?",
                "mini_lab_goal": "Run the original benchmark.",
                "dataset": {"name": "Proprietary dataset"},
                "baseline": "Original paper baseline",
                "metric": "accuracy",
                "steps": ["Rent 8xA100", "Train full model", "Compare"],
                "ablation": "Change everything at once",
                "failure_condition": "No gain",
            }
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("indexed paper evidence" in reason for reason in result.reasons))

    def test_experiment_spec_rejects_synthetic_or_simulated_inputs(self):
        result = evaluate_experiment_spec(
            {
                "research_question": "Does attention handle long-range dependencies?",
                "mini_lab_goal": "Create a synthetic sequence and compare it with a simulated baseline.",
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index rows"},
                "baseline": "Moving average proxy",
                "metric": "source evidence score",
                "steps": ["Create synthetic examples", "Run baseline", "Run variant"],
                "ablation": "Disable only the attention scorer.",
                "failure_condition": "source evidence score does not improve.",
            }
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("synthetic" in reason for reason in result.reasons))

    def test_experiment_spec_rejects_dataset_fallback_source(self):
        result = evaluate_experiment_spec(
            {
                "research_question": "Does attention handle long-range dependencies?",
                "mini_lab_goal": "Compare attention on indexed rows.",
                "dataset": {
                    "name": "Indexed PaperLens evidence window",
                    "source": "source-index rows",
                    "fallback": "random vectors",
                },
                "baseline": "Direct baseline",
                "metric": "source evidence score",
                "steps": ["Load indexed rows", "Run baseline", "Run variant"],
                "ablation": "Disable only the attention scorer.",
                "failure_condition": "source evidence score does not improve.",
            }
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("fallback input source" in reason for reason in result.reasons))

    def test_experiment_spec_rejects_generated_vector_inputs(self):
        result = evaluate_experiment_spec(
            {
                "research_question": "Does attention handle long-range dependencies?",
                "mini_lab_goal": "Compare attention over a randomly initialized sequence of vectors.",
                "dataset": {
                    "name": "Indexed PaperLens evidence window",
                    "source": "Randomly initialized sequence of vectors",
                },
                "baseline": "Direct baseline",
                "metric": "source evidence score",
                "steps": ["Build random vectors", "Run baseline", "Run variant"],
                "ablation": "Disable only the attention scorer.",
                "failure_condition": "source evidence score does not improve.",
            }
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("generated vector" in reason for reason in result.reasons))

    def test_growth_ideas_need_evidence_and_next_step(self):
        result = evaluate_growth_ideas(
            {
                "ideas": [
                    {
                        "idea": "Test ambiguity-conditioned reranking.",
                        "source_evidence": ["paper:s1", "run:r1"],
                        "testable_next_step": "Split examples by ambiguity.",
                        "risk": "Small buckets may be noisy.",
                    }
                ]
            }
        )
        self.assertTrue(result.passed, result.reasons)

    def test_growth_ideas_reject_synthetic_followup_inputs(self):
        result = evaluate_growth_ideas(
            {
                "ideas": [
                    {
                        "idea": "Try a synthetic follow-up sequence before the next paper-bound run.",
                        "source_evidence": ["paper:selected-span", "run:r1"],
                        "testable_next_step": "Create simulated examples and compare them.",
                        "risk": "This may not stay tied to the paper evidence.",
                    }
                ]
            },
            known_evidence_ids={"paper:selected-span", "run:r1"},
            require_multiple_sources=True,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("synthetic" in reason for reason in result.reasons))

    def test_growth_ideas_require_known_multi_source_evidence(self):
        result = evaluate_growth_ideas(
            {
                "ideas": [
                    {
                        "idea": "Test ambiguity-conditioned reranking.",
                        "source_evidence": ["paper:s1", "unknown:run"],
                        "testable_next_step": "Split examples by ambiguity.",
                        "risk": "Small buckets may be noisy.",
                    }
                ]
            },
            known_evidence_ids={"paper:s1", "run:r1"},
            require_multiple_sources=True,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("unknown evidence" in reason for reason in result.reasons))

    def test_fine_tuning_gate_waits_for_repeated_failures(self):
        self.assertEqual(fine_tuning_gate(["missing_metric"])["recommendation"], "no")
        decision = fine_tuning_gate(["malformed_json", "malformed_json", "malformed_json"])
        self.assertEqual(decision["recommendation"], "maybe")
        self.assertIn("malformed_json", decision["repeated_failures"])

    def test_fine_tuning_gate_requires_fix_attempted_records(self):
        failures = [
            FailureRecord(
                task="translation",
                label="terminology_drift",
                scenario_id=f"s{idx}",
                model="small-model",
                root_cause="terminology",
                fix_attempted=idx > 0,
            )
            for idx in range(3)
        ]
        self.assertEqual(fine_tuning_gate(failures)["recommendation"], "no")
        ready = [
            FailureRecord(
                task="translation",
                label="terminology_drift",
                scenario_id=f"s{idx}",
                model="small-model",
                root_cause="terminology",
                fix_attempted=True,
            )
            for idx in range(3)
        ]
        self.assertEqual(fine_tuning_gate(ready)["recommendation"], "maybe")


def _source_rows() -> list[dict]:
    return [
        {
            "source_id": "P0.S1",
            "text": "The selected paper evidence supports the prototype behavior.",
            "text_hash": "row1",
            "label": "selected",
            "gold": True,
            "query": "selected paper evidence",
        },
        {
            "source_id": "P0.S2",
            "text": "A contrast source span supports the baseline behavior.",
            "text_hash": "row2",
            "label": "context_control",
            "gold": False,
            "query": "selected paper evidence",
        },
    ]


if __name__ == "__main__":
    unittest.main()
