import json
import os
import tempfile
import unittest
from pathlib import Path

from paperlens_lab.model_adapter import ModelGateway


class ModelGatewayTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.trace_path = Path(self.tempdir.name) / "traces.jsonl"
        os.environ["PAPERLENS_TRACE_PATH"] = str(self.trace_path)

    def tearDown(self):
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
        self.tempdir.cleanup()

    def fake_call(self, prompt: str, model_id: str, max_new_tokens: int):
        if '"translations"' in prompt:
            return json.dumps(
                {
                    "translations": [
                        {
                            "span_id": "P0.S1",
                            "translation": "우리는 F1을 3.2점 향상시키는 방법을 제안한다.",
                            "preserved_terms": ["F1"],
                            "uncertain_phrases": [],
                        }
                    ],
                    "notes": [],
                }
            )
        if "Long evidence packet:" in prompt:
            return json.dumps(
                {
                    "answer": "P4.S9 contains the middle-only evidence and does not prove a full-paper claim.",
                    "evidence": [{"source_id": "P4.S9", "quote": "MIDSTREAM-LITM-427 improves F1 by 3.2 points."}],
                    "confidence": "medium",
                    "needs_more_context": True,
                    "unsupported_assumptions": ["full-paper superiority needs broader evidence"],
                }
            )
        if '"confidence"' in prompt:
            return json.dumps(
                {
                    "answer": "선택 문장은 F1 개선 주장을 말한다.",
                    "evidence": [{"source_id": "P0.S1", "quote": "We improve F1 by 3.2 points."}],
                    "confidence": "high",
                    "needs_more_context": False,
                    "unsupported_assumptions": [],
                }
            )
        if '"research_question"' in prompt:
            return json.dumps(
                {
                    "research_question": "Does evidence reranking improve top-k precision?",
                    "mini_lab_goal": "Compare baseline retrieval with evidence reranking.",
                    "dataset": {"name": "Toy QA set", "fallback": "10 hand-built examples"},
                    "baseline": "BM25 only",
                    "metric": "top-5 precision",
                    "steps": ["Build toy set", "Run baseline", "Run variant", "Compare failures"],
                    "ablation": "Remove evidence score",
                    "failure_condition": "No precision gain",
                    "expected_result": "Variant may improve precision on evidence-heavy examples.",
                    "faithfulness_notes": ["Toy run is not paper reproduction."],
                    "starter_code_plan": ["baseline", "variant", "score"],
                    "support_span_ids": ["P0.S1"],
                }
            )
        return json.dumps(
            {
                "ideas": [
                    {
                        "idea": "Test evidence score only on ambiguous queries.",
                        "source_evidence": ["paper:s1", "run:r1"],
                        "novelty_angle": "Condition the method on query ambiguity.",
                        "testable_next_step": "Bucket 10 examples by ambiguity and compare deltas.",
                        "risk": "Manual buckets may bias results.",
                    }
                ],
                "fine_tuning_signal": "none",
                "reason": "Prompted output is structured enough.",
            }
        )

    def test_model_paths_parse_json_and_write_trace(self):
        gateway = ModelGateway(provider="hf", call_model=self.fake_call)

        translation = gateway.translate_spans(
            "Demo Paper",
            [{"span_id": "P0.S1", "text": "We improve F1 by 3.2 points."}],
            use_model=True,
        )
        self.assertEqual(translation.data["translations"][0]["span_id"], "P0.S1")
        self.assertIn("3.2", translation.data["translations"][0]["translation"])

        answer = gateway.answer_span(
            "Demo Paper",
            "P0.S1",
            "We improve F1 by 3.2 points.",
            "",
            "무슨 뜻이야?",
            "We improve F1 by 3.2 points.",
            "ko",
            use_model=True,
        )
        self.assertEqual(answer.data["confidence"], "high")

        probe = gateway.answer_evidence_probe(
            "Demo Paper",
            "Find MIDSTREAM-LITM-427.",
            "P4.S9",
            "MIDSTREAM-LITM-427",
            [
                {"source_id": "P0.S1", "text": "Front distractor."},
                {"source_id": "P4.S9", "text": "MIDSTREAM-LITM-427 improves F1 by 3.2 points."},
                {"source_id": "P9.S1", "text": "End distractor."},
            ],
            "ko",
            use_model=True,
        )
        self.assertEqual(probe.data["evidence"][0]["source_id"], "P4.S9")

        spec = gateway.experiment_spec(
            "Demo Paper",
            "We improve retrieval with evidence reranking.",
            "",
            "We improve retrieval with evidence reranking.",
            "Try reranking",
            "ko",
            use_model=True,
        )
        self.assertIn("top-5 precision", spec.text)

        growth = gateway.growth_ideas(
            "Demo Paper",
            [{"id": "paper:s1", "summary": "Evidence reranking improves precision."}],
            "run:r1 improved precision but failed on ambiguous queries",
            "Evidence reranking improves precision.",
            "ko",
            use_model=True,
        )
        self.assertEqual(growth.data["fine_tuning_signal"], "none")

        lines = self.trace_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 5)
        trace_text = self.trace_path.read_text(encoding="utf-8")
        self.assertNotIn("HF_TOKEN", trace_text)
        self.assertNotIn("We improve F1 by 3.2 points", trace_text)

    def test_invalid_model_output_falls_back_to_structured_result(self):
        gateway = ModelGateway(provider="hf", call_model=lambda *_: '{"unexpected": "shape"}')
        result = gateway.answer_span(
            "Demo Paper",
            "P0.S1",
            "We improve F1 by 3.2 points.",
            "",
            "무슨 뜻이야?",
            "We improve F1 by 3.2 points.",
            "ko",
            use_model=True,
        )

        self.assertTrue(result.used_fallback)
        self.assertIn("invalid model output", result.error or "")
        self.assertIn("answer", result.data)
        self.assertIn("P0.S1", result.data["evidence"][0]["source_id"])

    def test_fallback_paths_are_structured(self):
        gateway = ModelGateway()
        result = gateway.experiment_spec(
            "Fallback Paper",
            "The method improves retrieval precision.",
            "",
            "The method improves retrieval precision.",
            "",
            "ko",
            use_model=False,
        )
        self.assertEqual(result.provider, "fallback")
        self.assertIn("baseline", result.data)
        self.assertIn("Failure condition", result.text)


if __name__ == "__main__":
    unittest.main()
