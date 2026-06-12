import json
import os
import tempfile
import unittest
from pathlib import Path

from paperlens_lab.model_adapter import ModelGateway
from paperlens_lab.scenario_runner import default_scenarios, run_scenarios


class ScenarioRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["PAPERLENS_TRACE_PATH"] = str(Path(self.tempdir.name) / "runner_traces.jsonl")

    def tearDown(self):
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
        self.tempdir.cleanup()

    def test_default_scenarios_pass_in_fallback_mode(self):
        result = run_scenarios(use_model=False)

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["scenario_count"], len(default_scenarios()))
        self.assertEqual(result["fine_tuning"]["recommendation"], "no")
        for run in result["runs"]:
            self.assertTrue(run["passed"], run)
            self.assertEqual(len(run["evaluations"]), 4)

    def test_model_scenarios_parse_structured_outputs(self):
        gateway = ModelGateway(provider="hf", call_model=self.fake_call)
        result = run_scenarios(scenarios=default_scenarios()[:1], gateway=gateway, use_model=True)

        self.assertTrue(result["passed"], result)
        run = result["runs"][0]
        self.assertEqual(run["model_outputs"]["translation"]["provider"], "hf")
        self.assertEqual(run["model_outputs"]["qa"]["data"]["confidence"], "high")
        self.assertEqual(result["fine_tuning"]["recommendation"], "no")

    def fake_call(self, prompt: str, model_id: str, max_new_tokens: int):
        if '"translations"' in prompt:
            return json.dumps(
                {
                    "translations": [
                        {
                            "span_id": "P0.S1",
                            "translation": "이 방법은 128개 질의에서 관련성만 사용하는 baseline보다 top-5 precision을 3.2점 향상시킨다.",
                            "preserved_terms": ["top-5 precision"],
                            "uncertain_phrases": [],
                        }
                    ],
                    "notes": [],
                }
            )
        if '"confidence"' in prompt:
            return json.dumps(
                {
                    "answer": "관련성만 쓰는 baseline과 evidence-linked reranking을 비교한 결과다.",
                    "evidence": [
                        {
                            "source_id": "P0.S1",
                            "quote": "top-5 precision by 3.2 points over a relevance-only baseline",
                        }
                    ],
                    "confidence": "high",
                    "needs_more_context": False,
                    "unsupported_assumptions": [],
                }
            )
        if '"research_question"' in prompt:
            return json.dumps(
                {
                    "research_question": "Does evidence-linked reranking improve top-5 precision?",
                    "mini_lab_goal": "Compare relevance-only retrieval with evidence-linked reranking.",
                    "dataset": {"name": "Toy retrieval set", "fallback": "10 hand-built query/passage pairs"},
                    "baseline": "Relevance-only ranking",
                    "metric": "top-5 precision",
                    "steps": ["Create toy examples", "Run baseline", "Run reranker", "Compare errors"],
                    "ablation": "Remove evidence links",
                    "failure_condition": "No top-5 precision gain",
                    "expected_result": "Reranking may help citation-heavy questions.",
                    "faithfulness_notes": ["This does not reproduce the full paper."],
                    "starter_code_plan": ["rank", "rerank", "score"],
                    "support_span_ids": ["P0.S1"],
                }
            )
        return json.dumps(
            {
                "ideas": [
                    {
                        "idea": "Test whether the reranker helps only on ambiguous questions.",
                        "source_evidence": ["P0.S1", "run:r1"],
                        "novelty_angle": "Condition the next test on ambiguity.",
                        "testable_next_step": "Split ten examples by ambiguity and compare precision deltas.",
                        "risk": "Manual ambiguity labels may be noisy.",
                    }
                ],
                "fine_tuning_signal": "none",
                "reason": "The structured path is stable enough for prompting.",
            }
        )


if __name__ == "__main__":
    unittest.main()
