import unittest

from paperlens_lab.scenario_eval import (
    evaluate_experiment_spec,
    evaluate_grounded_qa,
    evaluate_growth_ideas,
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

    def test_experiment_spec_requires_runnable_fields(self):
        result = evaluate_experiment_spec(
            {
                "research_question": "Does reranking help?",
                "mini_lab_goal": "Compare baseline and reranker.",
                "dataset": {"name": "Toy set", "fallback": "Hand-built examples"},
                "baseline": "BM25",
                "metric": "top-5 precision",
                "steps": ["Prepare data", "Run baseline", "Run reranker"],
                "failure_condition": "No top-5 precision gain",
            }
        )
        self.assertTrue(result.passed, result.reasons)

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

    def test_fine_tuning_gate_waits_for_repeated_failures(self):
        self.assertEqual(fine_tuning_gate(["missing_metric"])["recommendation"], "no")
        decision = fine_tuning_gate(["malformed_json", "malformed_json", "malformed_json"])
        self.assertEqual(decision["recommendation"], "maybe")
        self.assertIn("malformed_json", decision["repeated_failures"])


if __name__ == "__main__":
    unittest.main()
