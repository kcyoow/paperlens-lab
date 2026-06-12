import json
import os
import tempfile
import unittest
from pathlib import Path

from paperlens_lab.model_adapter import ModelGateway
from paperlens_lab.scenario_eval import evaluate_experiment_spec


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
                    "failure_condition": "top-5 precision does not improve",
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

    def test_experiment_spec_heavy_model_plan_is_reduced_to_smoke_test(self):
        def heavy_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "research_question": "Does the Transformer improve BLEU on WMT14?",
                    "mini_lab_goal": "Train an LSTM and Transformer on WMT14.",
                    "dataset": {"name": "WMT14", "fallback": "wmt14_small_tiny"},
                    "baseline": "PyTorch LSTM seq2seq",
                    "metric": "BLEU with sacrebleu",
                    "steps": ["Download and load WMT14", "Train for 100 epochs", "Evaluate BLEU"],
                    "ablation": "Remove self-attention.",
                    "failure_condition": "Transformer BLEU is lower.",
                    "expected_result": "Transformer improves BLEU.",
                    "faithfulness_notes": [],
                    "starter_code_plan": ["Use PyTorch", "Use sacrebleu"],
                    "support_span_ids": ["selected"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=heavy_call)
        result = gateway.experiment_spec(
            "Attention Is All You Need",
            "We trained our models on one machine with 8 NVIDIA P100 GPUs.",
            "",
            "We trained our models on one machine with 8 NVIDIA P100 GPUs.",
            "Try the attention mechanism.",
            "ko",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIn("repair_notes", result.data)
        self.assertTrue(evaluate_experiment_spec(result.data).passed)
        self.assertIn("dependency-free", " ".join(result.data["steps"]).lower())
        self.assertNotIn("wmt14", json.dumps(result.data).lower())
        self.assertNotIn("100 epochs", json.dumps(result.data).lower())

    def test_experiment_spec_cuda_multiday_plan_is_reduced_to_smoke_test(self):
        def heavy_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "research_question": "Can a full training run on CUDA P100 reproduce the paper?",
                    "mini_lab_goal": "Run multi-day distributed training.",
                    "dataset": {"name": "large dataset", "fallback": "toy examples"},
                    "baseline": "TensorFlow full model",
                    "metric": "full benchmark score",
                    "steps": ["Provision CUDA", "Run multi-day full training run", "Compare results"],
                    "ablation": "Disable only the adapter.",
                    "failure_condition": "benchmark score is lower.",
                    "expected_result": "A full training run should reproduce the paper.",
                    "faithfulness_notes": ["Requires P100 GPUs and multi-day training."],
                    "starter_code_plan": ["CUDA training script"],
                    "support_span_ids": ["selected"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=heavy_call)
        result = gateway.experiment_spec(
            "LoRA",
            "LoRA adapts models with low-rank updates.",
            "",
            "LoRA adapts models with low-rank updates.",
            "Try the adapter idea.",
            "ko",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertTrue(evaluate_experiment_spec(result.data).passed)
        repaired_text = json.dumps(result.data).lower()
        self.assertIn("dependency-free", repaired_text)
        self.assertNotIn("cuda", repaired_text)
        self.assertNotIn("multi-day", repaired_text)

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
