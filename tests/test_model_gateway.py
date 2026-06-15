import json
import os
import tempfile
import unittest
from pathlib import Path

from paperlens_lab.model_adapter import ModelGateway
from paperlens_lab.prompts import gpu_script_prompt
from paperlens_lab.scenario_eval import evaluate_experiment_spec, evaluate_starter_code, run_starter_code


class ModelGatewayTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.trace_path = Path(self.tempdir.name) / "traces.jsonl"
        os.environ["PAPERLENS_TRACE_PATH"] = str(self.trace_path)

    def tearDown(self):
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
        self.tempdir.cleanup()

    def source_bound_starter_code(self, *, import_json: bool = False, bad_failure_flag: bool = False) -> str:
        prefix = "import json\n\n" if import_json else ""
        maybe_json_wrap = "json.loads(json.dumps(payload))" if import_json else "payload"
        failure_expr = "False" if bad_failure_flag else "prototype_score <= baseline_score"
        return (
            prefix
            + f'''
# Source-bound mechanisms: attention, recurrence, compact evidence reranker, top five precision.
def baseline(example):
    if not example.get("gold"):
        return {{"prediction": "recurrence", "mode": "local"}}
    return {{"prediction": "recurrence", "mode": "local"}}

def paper_inspired(example):
    if example.get("gold"):
        return {{"prediction": "attention", "mode": "global"}}
    return {{"prediction": "recurrence", "mode": "control"}}

def score(output, gold):
    return 1.0 if output["prediction"] == gold else 0.0

def run(evidence_rows=None):
    rows = []
    for example in evidence_rows or []:
        gold = "attention" if example.get("gold") else "recurrence"
        baseline_score = score(baseline(example), gold)
        prototype_score = score(paper_inspired(example), gold)
        payload = {{
            "source_id": example["source_id"],
            "text_hash": example["text_hash"],
            "baseline_score": baseline_score,
            "prototype_score": prototype_score,
            "metric": "source evidence accuracy",
            "mode": "global" if example.get("gold") else "control",
            "failure_condition": {failure_expr},
            "failure_rule": "prototype_score <= baseline_score",
        }}
        rows.append({maybe_json_wrap})
    return rows
'''.strip()
        )

    def lora_source_bound_starter_code(self) -> str:
        return '''
# Source-bound mechanisms: LoRA, low-rank adapters, implementation metadata, PyTorch integration.
def baseline(example):
    context = (example.get("text") or example.get("context") or "").lower()
    if "baseline" in context:
        return {"prediction": "direct", "mode": "baseline"}
    return {"prediction": "direct", "mode": "local"}

def paper_inspired(example):
    context = (example.get("text") or example.get("context") or "").lower()
    if "lora" in context or "low-rank" in context or "adapter" in context:
        return {"prediction": "lora_adapter", "mode": "source_bound_adapter"}
    return {"prediction": "direct", "mode": "control"}

def score(output, gold):
    return 1.0 if output["prediction"] == gold else 0.0

def run(evidence_rows=None):
    rows = []
    for example in evidence_rows or []:
        text = (example.get("text") or "").lower()
        gold = "lora_adapter" if ("lora" in text or "low-rank" in text or "adapter" in text) else "direct"
        baseline_score = score(baseline(example), gold)
        prototype_score = score(paper_inspired(example), gold)
        rows.append({
            "source_id": example["source_id"],
            "text_hash": example["text_hash"],
            "baseline_score": baseline_score,
            "prototype_score": prototype_score,
            "metric": "source evidence accuracy",
            "mode": paper_inspired(example)["mode"],
            "failure_condition": prototype_score <= baseline_score,
            "failure_rule": "prototype_score <= baseline_score",
        })
    return rows
'''.strip()

    def test_gpu_script_prompt_requires_model_authored_visual_report(self):
        prompt = gpu_script_prompt(
            "Deep Residual Learning for Image Recognition",
            "Residual networks are easier to optimize.",
            [{"source_id": "S1", "text": "Residual networks are easier to optimize."}],
            {
                "id": "gpu-probe",
                "title": "Residual vs plain network probe",
                "reproduction_level": "probe",
                "run_plan": {"dataset": "CIFAR-10"},
            },
            "probe",
            "en",
            [],
        )

        self.assertIn("`reportHtml` must be authored by the generated script/model", prompt)
        self.assertIn("PaperLens will sanitize and display it", prompt)
        self.assertIn("will not append, synthesize, or prettify", prompt)
        self.assertIn("at least one meaningful self-contained visual artifact", prompt)
        self.assertIn("using inline `<svg>` or `<figure>`", prompt)
        self.assertIn("must not be a generic metric dashboard", prompt)
        self.assertIn("paper claim", prompt)
        self.assertIn("paper evidence/source span", prompt)
        self.assertIn("experiment setup/code path", prompt)
        self.assertIn("measured metrics/result", prompt)
        self.assertIn("comparison to the paper claim", prompt)
        self.assertIn("limitations/next step", prompt)

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
                    "research_question": "Does evidence reranking improve top-5 precision on indexed paper evidence?",
                    "mini_lab_goal": "Compare baseline retrieval with evidence reranking on source-index rows.",
                    "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index rows"},
                    "baseline": "BM25 only",
                    "metric": "top-5 precision on indexed paper evidence",
                    "steps": ["Load indexed evidence rows", "Run baseline", "Run variant", "Compare failures"],
                    "ablation": "Remove evidence score",
                    "failure_condition": "top-5 precision on indexed paper evidence does not improve",
                    "expected_result": "Variant may improve precision on evidence-heavy examples.",
                    "faithfulness_notes": ["Source-bound run is not full paper reproduction."],
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

    def test_translation_uses_repair_before_fallback(self):
        calls: list[str] = []

        def repairing_translation_call(prompt, model_id, max_new_tokens):
            calls.append(prompt)
            if "repairing a PaperLens Lab translation response" in prompt:
                return json.dumps(
                    {
                        "translations": [
                            {
                                "span_id": "P0.S1",
                                "translation": "우리는 attention mechanisms만 사용하는 Transformer를 제안한다.",
                                "preserved_terms": ["attention mechanisms", "Transformer"],
                                "uncertain_phrases": [],
                            }
                        ],
                        "notes": [],
                    }
                )
            return "P0.S1: 우리는 attention mechanisms만 사용하는 Transformer를 제안한다."

        gateway = ModelGateway(provider="hf", call_model=repairing_translation_call)
        result = gateway.translate_spans(
            "Attention Is All You Need",
            [
                {
                    "span_id": "P0.S1",
                    "text": "We propose the Transformer based solely on attention mechanisms.",
                }
            ],
            locale="ko",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIsNone(result.error)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.data["translations"][0]["span_id"], "P0.S1")
        self.assertIn("Transformer", result.data["translations"][0]["translation"])

    def test_starter_code_uses_repair_before_fallback(self):
        calls: list[str] = []

        def repairing_call(prompt, model_id, max_new_tokens):
            calls.append(prompt)
            if "repairing a PaperLens Lab starter-code JSON response" in prompt:
                return json.dumps(
                    {
                        "code": self.source_bound_starter_code(),
                        "why_this_matches_span": "The repaired code contrasts attention against removed recurrence and convolutions.",
                        "limitations": ["Source-bound run is not a full paper reproduction."],
                    }
                )
            return json.dumps(
                {
                    "code": "def baseline(example):\n    return {}\n",
                    "why_this_matches_span": "First pass is still too generic.",
                    "limitations": ["Missing explicit contrast modes."],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=repairing_call)
        result = gateway.starter_code(
            "Attention Is All You Need",
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. A recurrence baseline remains as a contrast source span.",
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. A recurrence baseline remains as a contrast source span.",
            {
                "research_question": "Can an attention-style global scorer recover the selected claim better than a local baseline?",
                "mini_lab_goal": "Compare a local baseline against an attention-style scorer on indexed paper evidence.",
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "selected span plus contrast spans"},
                "baseline": "Local or first-match heuristic without the attention-style bonus.",
                "metric": "label accuracy on indexed paper evidence",
                "ablation": "Remove only the attention-style global scoring bonus and keep everything else fixed.",
                "failure_condition": "The mini-lab fails if label accuracy on indexed paper evidence does not improve.",
                "expected_result": "A small directional gain on long-range or distractor-heavy examples.",
            },
            "ko",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn('"mode": "global"', result.data["code"])

    def test_starter_code_salvages_fenced_python_without_json_wrapper(self):
        raw_code = f"""
The code block below stays grounded to the selected mechanism.

```python
{self.source_bound_starter_code()}
```
""".strip()

        gateway = ModelGateway(provider="hf", call_model=lambda *_: raw_code)
        result = gateway.starter_code(
            "Attention Is All You Need",
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. A recurrence baseline remains as a contrast source span.",
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. A recurrence baseline remains as a contrast source span.",
            {
                "research_question": "Can an attention-style global scorer recover the selected claim better than a local baseline?",
                "mini_lab_goal": "Compare a local baseline against an attention-style scorer on indexed paper evidence.",
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "selected span plus contrast spans"},
                "baseline": "Local or first-match heuristic without the attention-style bonus.",
                "metric": "label accuracy on indexed paper evidence",
                "ablation": "Remove only the attention-style global scoring bonus and keep everything else fixed.",
                "failure_condition": "The mini-lab fails if label accuracy on indexed paper evidence does not improve.",
                "expected_result": "A small directional gain on long-range or distractor-heavy examples.",
            },
            "ko",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIsNone(result.error)
        self.assertIn('def paper_inspired(example):', result.data["code"])
        self.assertTrue(result.data["recovered_from_non_json"])

    def test_starter_code_adds_missing_safe_import_before_fallback(self):
        code = self.source_bound_starter_code(import_json=True).replace("import json\n\n", "", 1)
        gateway = ModelGateway(
            provider="hf",
            call_model=lambda *_: json.dumps(
                {
                    "code": code,
                    "why_this_matches_span": "Uses the compact evidence reranker span as the mechanism.",
                    "limitations": ["Source-bound run is not full paper reproduction."],
                }
            ),
        )

        result = gateway.starter_code(
            "Demo Paper",
            "compact evidence reranker improves top five precision",
            "The compact evidence reranker improves top five precision in indexed source evidence. A direct retrieval baseline remains as a contrast source span.",
            {
                "research_question": "Does compact evidence reranking improve top five precision?",
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index rows"},
                "metric": "Top-5 Precision",
            },
            "en",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIsNone(result.error)
        self.assertTrue(result.data["code"].startswith("import json"))

    def test_starter_code_prompt_includes_read_only_repo_manifest_context(self):
        prompts: list[str] = []

        def repo_manifest_call(prompt, model_id, max_new_tokens):
            prompts.append(prompt)
            return json.dumps(
                {
                    "code": self.lora_source_bound_starter_code(),
                    "why_this_matches_span": (
                        "Uses the source-listed LoRA repository metadata only as read-only context while "
                        "running on indexed paper evidence rows."
                    ),
                    "limitations": ["Does not execute repository code or reproduce full LoRA training."],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=repo_manifest_call)
        result = gateway.starter_code(
            "LoRA: Low-Rank Adaptation of Large Language Models",
            "We release a package that facilitates the integration of LoRA with PyTorch models.",
            (
                "We release a package that facilitates the integration of LoRA with PyTorch models. "
                "See https://github.com/microsoft/LoRA for the official implementation."
            ),
            {
                "research_question": "Can source-listed LoRA implementation metadata guide a source-bound evidence run?",
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index rows"},
                "metric": "source evidence accuracy",
                "implementation_repositories": [
                    {
                        "url": "https://github.com/microsoft/LoRA",
                        "source_url": "https://github.com/microsoft/LoRA",
                    }
                ],
            },
            "en",
            implementation_repo_manifests=[
                {
                    "source_id": "implementation:github:1",
                    "url": "https://github.com/microsoft/LoRA",
                    "source_url": "https://github.com/microsoft/LoRA",
                    "status": "inspected",
                    "execution": "none",
                    "commit": "c4593f060e6a368d7bb5af5273b8e42810cdef90",
                    "default_branch": "main",
                    "file_count": 12,
                    "truncated": True,
                    "files": [{"path": "loralib/layers.py", "kind": "source"}],
                    "readme": {"path": "README.md", "excerpt": "LoRA implementation."},
                    "license": {"path": "LICENSE.md", "excerpt": "MIT"},
                    "error": "",
                }
            ],
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIsNone(result.error)
        self.assertIn("https://github.com/microsoft/LoRA", prompts[0])
        self.assertIn("c4593f060e6a368d7bb5af5273b8e42810cdef90", prompts[0])
        self.assertIn("read-only context", prompts[0])
        self.assertIn("Do not clone, install, import from, execute", prompts[0])
        self.assertIn("run only on supplied `evidence_rows`", prompts[0])
        eval_result = evaluate_starter_code(
            result.data["code"],
            evidence_rows=[
                {
                    "source_id": "P0.S1",
                    "text_hash": "hash1",
                    "text": "LoRA adds low-rank adapters to model weights.",
                    "gold": "attention",
                },
                {
                    "source_id": "P0.S2",
                    "text_hash": "hash2",
                    "text": "A direct baseline is included for contrast.",
                    "gold": "",
                },
            ],
            require_evidence_rows=True,
        )
        self.assertTrue(eval_result.passed, eval_result.reasons)

    def test_starter_code_rejects_failure_condition_mismatch_before_fallback(self):
        code = self.source_bound_starter_code(bad_failure_flag=True)
        gateway = ModelGateway(
            provider="hf",
            call_model=lambda *_: json.dumps(
                {
                    "code": code,
                    "why_this_matches_span": "Uses the compact evidence reranker selected span.",
                    "limitations": ["Source-bound run is not full paper reproduction."],
                }
            ),
        )

        result = gateway.starter_code(
            "Demo Paper",
            "compact evidence reranker improves top five precision",
            "The compact evidence reranker improves top five precision in indexed source evidence. A direct retrieval baseline remains as a contrast source span.",
            {
                "research_question": "Does compact evidence reranking improve top five precision?",
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index rows"},
                "metric": "Top-5 Precision",
            },
            "en",
            use_model=True,
        )

        self.assertTrue(result.used_fallback)
        self.assertIn("failure_condition must match", result.error or "")

    def test_starter_code_uses_second_repair_attempt_before_fallback(self):
        calls: list[str] = []

        def repairing_call(prompt, model_id, max_new_tokens):
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(
                    {
                        "code": "def baseline(example):\n    return {}\n",
                        "why_this_matches_span": "Initial output is too generic.",
                        "limitations": ["Needs stricter repair."],
                    }
                )
            if len(calls) == 2:
                return json.dumps(
                    {
                        "code": "def baseline(example):\n    return {}\n\ndef run(evidence_rows=None):\n    return []\n",
                        "why_this_matches_span": "First repair still leaves the trivial baseline.",
                        "limitations": ["Needs one more pass."],
                    }
                )
            return json.dumps(
                {
                    "code": self.source_bound_starter_code(),
                    "why_this_matches_span": "Second repair adds explicit contrast modes and a non-trivial local baseline.",
                    "limitations": ["Source-bound run is not a full paper reproduction."],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=repairing_call)
        result = gateway.starter_code(
            "Attention Is All You Need",
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. A recurrence baseline remains as a contrast source span.",
            "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            {
                "research_question": "Can an attention-style global scorer recover the selected claim better than a local baseline?",
                "mini_lab_goal": "Compare a local baseline against an attention-style scorer on indexed paper evidence.",
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "selected span plus contrast spans"},
                "baseline": "Local or first-match heuristic without the attention-style bonus.",
                "metric": "label accuracy on indexed paper evidence",
                "ablation": "Remove only the attention-style global scoring bonus and keep everything else fixed.",
                "failure_condition": "The mini-lab fails if label accuracy on indexed paper evidence does not improve.",
                "expected_result": "A small directional gain on long-range or distractor-heavy examples.",
            },
            "ko",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIsNone(result.error)
        self.assertEqual(len(calls), 3)
        self.assertIn('"mode": "global"', result.data["code"])

    def test_experiment_spec_heavy_model_plan_is_reduced_to_source_run(self):
        def heavy_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "research_question": "Does the Transformer improve BLEU on WMT14?",
                    "mini_lab_goal": "Train an LSTM and Transformer on WMT14.",
                    "dataset": {"name": "WMT14", "fallback": "full benchmark subset"},
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
        self.assertIn("source", json.dumps(result.data).lower())
        self.assertNotIn("wmt14", json.dumps(result.data).lower())
        self.assertNotIn("100 epochs", json.dumps(result.data).lower())

    def test_experiment_spec_cuda_multiday_plan_is_reduced_to_source_run(self):
        def heavy_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "research_question": "Can a full training run on CUDA P100 reproduce the paper?",
                    "mini_lab_goal": "Run multi-day distributed training.",
                    "dataset": {"name": "large dataset", "fallback": "full benchmark subset"},
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

    def test_experiment_spec_dataset_is_bound_to_indexed_paper_evidence(self):
        def loose_dataset_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "research_question": "Can attention-only scoring separate selected evidence from context?",
                    "mini_lab_goal": "Compare a local baseline with an attention-style scorer.",
                    "dataset": {"name": "small sentence set"},
                    "baseline": "Local first-match scorer",
                    "metric": "label accuracy on selected evidence",
                    "steps": ["Load rows", "Run baseline", "Run variant", "Compare metric"],
                    "ablation": "Disable only the attention-style global scoring bonus.",
                    "failure_condition": "label accuracy on selected evidence does not improve.",
                    "expected_result": "The attention-style scorer may improve the selected row.",
                    "faithfulness_notes": [],
                    "starter_code_plan": ["baseline", "variant", "score"],
                    "support_span_ids": ["selected"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=loose_dataset_call)
        result = gateway.experiment_spec(
            "Attention Is All You Need",
            "We propose the Transformer based solely on attention mechanisms.",
            "",
            "We propose the Transformer based solely on attention mechanisms.",
            "Try this span.",
            "en",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertTrue(evaluate_experiment_spec(result.data).passed)
        dataset_text = json.dumps(result.data["dataset"], ensure_ascii=False).lower()
        self.assertIn("indexed", dataset_text)
        self.assertIn("paperlens", dataset_text)
        self.assertIn("evidence", dataset_text)

    def test_experiment_spec_rejects_toy_wording_before_fallback(self):
        def toy_wording_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "research_question": "Can the selected mechanism improve a source-bound signal?",
                    "mini_lab_goal": "Run a toy setup for the selected mechanism.",
                    "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index rows"},
                    "baseline": "Direct baseline",
                    "metric": "source-bound label accuracy",
                    "steps": ["Load indexed rows", "Run baseline", "Run variant", "Compare metric"],
                    "ablation": "Disable only the selected mechanism.",
                    "failure_condition": "source-bound label accuracy does not improve.",
                    "expected_result": "The source-bound variant may improve.",
                    "faithfulness_notes": ["The scale is reduced to a toy problem for speed."],
                    "starter_code_plan": ["baseline", "variant", "score"],
                    "support_span_ids": ["selected"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=toy_wording_call)
        result = gateway.experiment_spec(
            "Demo Paper",
            "The selected mechanism improves a measurable source-bound behavior.",
            "",
            "The selected mechanism improves a measurable source-bound behavior.",
            "Try this span.",
            "en",
            use_model=True,
        )

        self.assertTrue(result.used_fallback)
        self.assertIn("toy", result.error or "")
        self.assertNotIn("toy", json.dumps(result.data).lower())

    def test_experiment_spec_uses_model_repair_for_toy_wording(self):
        calls = []

        def repairing_toy_wording_call(prompt, model_id, max_new_tokens):
            calls.append(prompt)
            if "repairing a PaperLens Lab experiment spec" in prompt:
                return json.dumps(
                    {
                        "research_question": "Can the selected mechanism improve a source-bound signal?",
                        "mini_lab_goal": "Run a source-indexed comparison for the selected mechanism.",
                        "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index rows"},
                        "baseline": "Direct baseline",
                        "metric": "source-bound label accuracy",
                        "steps": ["Load indexed rows", "Run baseline", "Run variant", "Compare metric"],
                        "ablation": "Disable only the selected mechanism.",
                        "failure_condition": "source-bound label accuracy does not improve.",
                        "expected_result": "The source-bound variant may improve.",
                        "faithfulness_notes": ["Use only the indexed evidence rows visible in the paper reader."],
                        "starter_code_plan": ["baseline", "variant", "score"],
                        "support_span_ids": ["selected"],
                    }
                )
            return json.dumps(
                {
                    "research_question": "Can the selected mechanism improve a source-bound signal?",
                    "mini_lab_goal": "Run a toy setup for the selected mechanism.",
                    "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index rows"},
                    "baseline": "Direct baseline",
                    "metric": "source-bound label accuracy",
                    "steps": ["Load indexed rows", "Run baseline", "Run variant", "Compare metric"],
                    "ablation": "Disable only the selected mechanism.",
                    "failure_condition": "source-bound label accuracy does not improve.",
                    "expected_result": "The source-bound variant may improve.",
                    "faithfulness_notes": ["The scale is reduced to a toy problem for speed."],
                    "starter_code_plan": ["baseline", "variant", "score"],
                    "support_span_ids": ["selected"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=repairing_toy_wording_call)
        result = gateway.experiment_spec(
            "Demo Paper",
            "The selected mechanism improves a measurable source-bound behavior.",
            "",
            "The selected mechanism improves a measurable source-bound behavior.",
            "Try this span.",
            "en",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIsNone(result.error)
        self.assertEqual(len(calls), 2)
        self.assertTrue(evaluate_experiment_spec(result.data).passed)
        self.assertNotIn("toy", json.dumps(result.data).lower())

    def test_experiment_spec_rejects_synthetic_sequence_wording_before_fallback(self):
        def synthetic_wording_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "research_question": "Can the selected mechanism improve a source-bound signal?",
                    "mini_lab_goal": "Create a synthetic sequence for the selected mechanism.",
                    "dataset": {"name": "Synthetic dataset", "source": "simulated examples"},
                    "baseline": "Direct baseline",
                    "metric": "source-bound label accuracy",
                    "steps": ["Create synthetic examples", "Run baseline", "Run variant"],
                    "ablation": "Disable only the selected mechanism.",
                    "failure_condition": "source-bound label accuracy does not improve.",
                    "expected_result": "The source-bound variant may improve.",
                    "faithfulness_notes": ["Use synthetic patterns for speed."],
                    "starter_code_plan": ["baseline", "variant", "score"],
                    "support_span_ids": ["selected"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=synthetic_wording_call)
        result = gateway.experiment_spec(
            "Demo Paper",
            "The selected mechanism improves a measurable source-bound behavior.",
            "",
            "The selected mechanism improves a measurable source-bound behavior.",
            "Try this span.",
            "en",
            use_model=True,
        )

        serialized = json.dumps(result.data).lower()
        self.assertTrue(result.used_fallback)
        self.assertIn("synthetic", result.error or "")
        self.assertNotIn("synthetic", serialized)
        self.assertNotIn("simulated", serialized)

    def test_experiment_spec_rejects_legacy_fallback_random_vector_dataset_before_fallback(self):
        def random_dataset_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "research_question": "Can attention connect source evidence rows?",
                    "mini_lab_goal": "Compare attention over a randomly initialized sequence of vectors.",
                    "dataset": {
                        "name": "Indexed PaperLens evidence window",
                        "fallback": "Randomly initialized sequence of vectors",
                    },
                    "baseline": "Direct baseline",
                    "metric": "source-bound label accuracy",
                    "steps": ["Build random-vector dataset", "Run baseline", "Run variant"],
                    "ablation": "Disable only the selected mechanism.",
                    "failure_condition": "source-bound label accuracy does not improve.",
                    "expected_result": "The source-bound variant may improve.",
                    "faithfulness_notes": ["Use generated-inputs only for speed."],
                    "starter_code_plan": ["baseline", "variant", "score"],
                    "support_span_ids": ["selected"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=random_dataset_call)
        result = gateway.experiment_spec(
            "Demo Paper",
            "The selected mechanism improves a measurable source-bound behavior.",
            "",
            "The selected mechanism improves a measurable source-bound behavior.",
            "Try this span.",
            "en",
            use_model=True,
        )

        serialized = json.dumps(result.data).lower()
        self.assertTrue(result.used_fallback)
        self.assertIn("fallback input source", result.error or "")
        self.assertNotIn("fallback", result.data["dataset"])
        self.assertNotIn("randomly initialized", serialized)
        self.assertNotIn("random vectors", serialized)
        self.assertNotIn("random-vector", serialized)
        self.assertNotIn("generated inputs", serialized)
        self.assertNotIn("generated-inputs", serialized)

    def test_experiment_spec_uses_only_source_github_implementation_links(self):
        def repo_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "research_question": "Can the selected mechanism improve a source-bound signal?",
                    "mini_lab_goal": "Run a source-bound probe for the selected mechanism.",
                    "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index rows"},
                    "baseline": "Direct baseline",
                    "metric": "source-bound label accuracy",
                    "steps": [
                        "Load indexed rows",
                        "Inspect https://github.com/made/up before running the baseline.",
                        "Run variant",
                    ],
                    "ablation": "Disable only the selected mechanism.",
                    "failure_condition": "source-bound label accuracy does not improve.",
                    "expected_result": "The source-bound variant may improve.",
                    "faithfulness_notes": ["Do not rely on https://github.com/made/up unless it appears in the paper."],
                    "implementation_repositories": [
                        {"url": "https://github.com/made/up", "usage": "invented repo"}
                    ],
                    "starter_code_plan": ["baseline", "variant", "score"],
                    "support_span_ids": ["selected"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=repo_call)
        result = gateway.experiment_spec(
            "LoRA: Low-Rank Adaptation of Large Language Models",
            "We provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2.",
            "",
            (
                "We release a package that facilitates the integration of LoRA with PyTorch models "
                "and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 "
                "at https://github.com/microsoft/LoRA."
            ),
            "Try this span.",
            "en",
            use_model=True,
        )

        repos = result.data["implementation_repositories"]
        serialized = json.dumps(result.data).lower()
        self.assertFalse(result.used_fallback)
        self.assertTrue(evaluate_experiment_spec(result.data).passed)
        self.assertEqual(repos[0]["url"], "https://github.com/microsoft/LoRA")
        self.assertEqual(repos[0]["source_url"], "https://github.com/microsoft/LoRA")
        self.assertNotIn("made/up", serialized)
        self.assertIn("Implementation repositories", result.text)

    def test_experiment_candidates_accept_list_fields_without_fallback(self):
        def candidate_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "candidates": [
                        {
                            "id": "gpu-replication-probe",
                            "title": "GPU replication probe for the selected training claim",
                            "kind": "gpu_replication_probe",
                            "is_recommended": True,
                            "recommendation_reason": "It directly checks the selected GPU-backed training claim.",
                            "hypothesis": "A short CUDA-backed run can validate whether the selected method produces a measurable training signal.",
                            "paper_evidence_ids": ["selected"],
                            "paper_evidence_quotes": [
                                "The selected span reports a GPU-backed training result for the proposed method."
                            ],
                            "dataset": {
                                "name": "Paper-specified public benchmark subset",
                                "source": "dataset named in the selected paper evidence",
                                "requires_download": True,
                            },
                            "implementation": {
                                "type": "source_bound_probe",
                                "repo_url": "",
                                "reason": "No source-listed repository is present in the selected evidence.",
                            },
                            "gpu_required": True,
                            "estimated_runtime_minutes": 12,
                            "expected_metric": "validation_accuracy",
                            "limitations": ["Short replication probe, not the full paper-scale run."],
                            "approval_question": "Run this GPU replication probe?",
                        },
                        {
                            "id": "source-window-audit",
                            "title": "Source evidence audit for the selected claim",
                            "kind": "source_bound_probe",
                            "is_recommended": False,
                            "recommendation_reason": "Useful as a pre-run evidence check.",
                            "hypothesis": "The selected claim can be checked against surrounding source evidence before launching the GPU run.",
                            "paper_evidence_ids": ["selected"],
                            "paper_evidence_quotes": [
                                "The selected span reports a GPU-backed training result for the proposed method."
                            ],
                            "dataset": {
                                "name": "PaperLens indexed evidence window",
                                "source": "selected span and adjacent source-index rows",
                                "requires_download": False,
                            },
                            "implementation": {
                                "type": "source_bound_probe",
                                "repo_url": "",
                                "reason": "Evidence-only audit before execution.",
                            },
                            "gpu_required": False,
                            "estimated_runtime_minutes": 1,
                            "expected_metric": "evidence_support",
                            "limitations": ["Does not execute the training claim."],
                            "approval_question": "Run the evidence audit first?",
                        },
                    ],
                    "recommended_candidate_id": "gpu-replication-probe",
                }
            )

        gateway = ModelGateway(provider="hf", call_model=candidate_call)
        result = gateway.experiment_candidates(
            paper_title="GPU Training Paper",
            selected_span="The selected span reports a GPU-backed training result for the proposed method.",
            translated_span="",
            source_text=(
                "The selected span reports a GPU-backed training result for the proposed method. "
                "The surrounding paper text names the benchmark and metric."
            ),
            question="What experiment should we run from this span?",
            locale="en",
            use_model=True,
        )

        self.assertFalse(result.used_fallback, result.error)
        self.assertIsNone(result.error)
        self.assertEqual(result.data["recommended_candidate_id"], "gpu-replication-probe")
        self.assertEqual(len(result.data["candidates"]), 2)
        self.assertTrue(result.data["candidates"][0]["gpu_required"])

    def test_experiment_candidates_repair_unapproved_repo_url_before_failing(self):
        calls: list[str] = []

        def candidate_call(prompt, model_id, max_new_tokens):
            calls.append(prompt)
            if "Repair PaperLens Lab research-direction JSON" in prompt:
                return json.dumps(
                    {
                        "candidates": [
                            {
                                "id": "gpu-replication-probe",
                                "title": "GPU replication probe",
                                "kind": "gpu_replication_probe",
                                "reproduction_level": "probe",
                                "faithfulness": {
                                    "level": "probe",
                                    "summary": "Bounded public-dataset probe.",
                                    "why_not_exact": "No source-listed implementation repo is present.",
                                    "paper_targets": ["accuracy"],
                                    "resource_note": "Short GPU run.",
                                },
                                "is_recommended": True,
                                "recommendation_reason": "It gives the clearest visible signal without overclaiming exact reproduction.",
                                "hypothesis": "A bounded CIFAR-10 run can directionally test the selected image-classification claim.",
                                "paper_evidence_ids": ["selected"],
                                "paper_evidence_quotes": ["The selected span reports a GPU-backed image-classification result."],
                                "dataset": {
                                    "name": "CIFAR-10",
                                    "source": "torchvision.datasets.CIFAR10 public dataset",
                                    "requires_download": True,
                                },
                                "implementation": {
                                    "type": "public_dataset",
                                    "repo_url": "",
                                    "reason": "Use torchvision as a library, not as a paper implementation repo.",
                                },
                                "run_plan": {
                                    "repo_url": "",
                                    "config_path": "",
                                    "command": "PaperLens Modal GPU run with generated experiment.py",
                                    "dataset": "CIFAR-10",
                                    "expected_artifact": "accuracy and loss table",
                                },
                                "gpu_required": True,
                                "estimated_runtime_minutes": 8,
                                "expected_metric": "validation accuracy and loss",
                                "limitations": ["Probe only; not exact paper reproduction."],
                                "approval_question": "Run this bounded GPU probe?",
                            },
                            {
                                "id": "depth-ablation-probe",
                                "title": "Depth ablation probe",
                                "kind": "gpu_replication_probe",
                                "reproduction_level": "probe",
                                "faithfulness": {
                                    "level": "probe",
                                    "summary": "Bounded depth comparison.",
                                    "why_not_exact": "No source-listed implementation repo is present.",
                                    "paper_targets": ["depth effect"],
                                    "resource_note": "Short GPU run.",
                                },
                                "is_recommended": False,
                                "recommendation_reason": "It is useful after the first probe.",
                                "hypothesis": "A shallow/deeper comparison can expose the claim direction on a bounded dataset.",
                                "paper_evidence_ids": ["selected"],
                                "paper_evidence_quotes": ["The selected span reports a GPU-backed image-classification result."],
                                "dataset": {
                                    "name": "CIFAR-10",
                                    "source": "torchvision.datasets.CIFAR10 public dataset",
                                    "requires_download": True,
                                },
                                "implementation": {
                                    "type": "public_dataset",
                                    "repo_url": "",
                                    "reason": "No source-listed repo is available.",
                                },
                                "run_plan": {
                                    "repo_url": "",
                                    "config_path": "",
                                    "command": "PaperLens Modal GPU run with generated experiment.py",
                                    "dataset": "CIFAR-10",
                                    "expected_artifact": "depth comparison metrics",
                                },
                                "gpu_required": True,
                                "estimated_runtime_minutes": 8,
                                "expected_metric": "validation accuracy and loss",
                                "limitations": ["Probe only; not exact paper reproduction."],
                                "approval_question": "Run the depth comparison probe?",
                            },
                        ],
                        "recommended_candidate_id": "gpu-replication-probe",
                    }
                )
            return json.dumps(
                {
                    "candidates": [
                        {
                            "id": "gpu-replication-probe",
                            "title": "GPU replication probe",
                            "kind": "gpu_replication_probe",
                            "reproduction_level": "probe",
                            "faithfulness": {
                                "level": "probe",
                                "summary": "Bounded public-dataset probe.",
                                "why_not_exact": "No source-listed implementation repo is present.",
                                "paper_targets": ["accuracy"],
                                "resource_note": "Short GPU run.",
                            },
                            "is_recommended": True,
                            "recommendation_reason": "It uses a public image dataset.",
                            "hypothesis": "A bounded run can directionally test the selected claim.",
                            "paper_evidence_ids": ["selected"],
                            "paper_evidence_quotes": ["The selected span reports a GPU-backed image-classification result."],
                            "dataset": {
                                "name": "CIFAR-10",
                                "source": "torchvision.datasets.CIFAR10 public dataset",
                                "requires_download": True,
                            },
                            "implementation": {
                                "type": "public_dataset",
                                "repo_url": "https://github.com/pytorch/vision",
                                "reason": "Public library, not paper source.",
                            },
                            "run_plan": {
                                "repo_url": "https://github.com/pytorch/vision",
                                "config_path": "",
                                "command": "python run_probe.py",
                                "dataset": "CIFAR-10",
                                "expected_artifact": "accuracy",
                            },
                            "gpu_required": True,
                            "estimated_runtime_minutes": 8,
                            "expected_metric": "validation accuracy",
                            "limitations": ["Probe only; not exact paper reproduction."],
                            "approval_question": "Run this bounded GPU probe?",
                        },
                        {
                            "id": "source-window-audit",
                            "title": "Source evidence audit",
                            "kind": "source_bound_probe",
                            "reproduction_level": "probe",
                            "faithfulness": {
                                "level": "probe",
                                "summary": "Evidence check.",
                                "why_not_exact": "No source-listed implementation repo is present.",
                                "paper_targets": ["evidence"],
                                "resource_note": "No GPU needed.",
                            },
                            "is_recommended": False,
                            "recommendation_reason": "Pre-run evidence check.",
                            "hypothesis": "The selected claim can be mapped to nearby source evidence.",
                            "paper_evidence_ids": ["selected"],
                            "paper_evidence_quotes": ["The selected span reports a GPU-backed image-classification result."],
                            "dataset": {
                                "name": "PaperLens indexed evidence window",
                                "source": "selected source rows",
                                "requires_download": False,
                            },
                            "implementation": {"type": "source_bound_probe", "repo_url": "", "reason": "No repo."},
                            "run_plan": {
                                "repo_url": "",
                                "config_path": "",
                                "command": "PaperLens source-bound check",
                                "dataset": "PaperLens evidence",
                                "expected_artifact": "evidence support",
                            },
                            "gpu_required": False,
                            "estimated_runtime_minutes": 1,
                            "expected_metric": "evidence support",
                            "limitations": ["Does not execute the claim."],
                            "approval_question": "Run the evidence check?",
                        },
                    ],
                    "recommended_candidate_id": "gpu-replication-probe",
                }
            )

        gateway = ModelGateway(provider="hf", call_model=candidate_call)
        result = gateway.experiment_candidates(
            paper_title="Image Classification Paper",
            selected_span="The selected span reports a GPU-backed image-classification result.",
            translated_span="",
            source_text="The selected span reports a GPU-backed image-classification result.",
            question="What experiment should we run from this span?",
            locale="en",
            use_model=True,
        )

        self.assertFalse(result.used_fallback, result.error)
        self.assertIsNone(result.error)
        self.assertEqual(len(calls), 2)
        self.assertIn("uses a repo URL that is not listed in the paper source", calls[1])
        self.assertEqual(result.data["candidates"][0]["implementation"]["repo_url"], "")
        self.assertEqual(result.data["candidates"][0]["run_plan"]["repo_url"], "")

    def test_experiment_candidates_reject_invalid_candidate_reproduction_level(self):
        def candidate_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "candidates": [
                        {
                            "id": "invalid-middle-mode",
                            "title": "Invalid middle mode",
                            "kind": "gpu_replication_probe",
                            "reproduction_level": "scaled",
                            "faithfulness": {
                                "level": "scaled",
                                "summary": "Invalid middle mode.",
                                "why_not_exact": "No exact repo.",
                                "paper_targets": ["accuracy"],
                                "resource_note": "short run",
                            },
                            "is_recommended": True,
                            "recommendation_reason": "This should be rejected.",
                            "hypothesis": "A middle mode should not be exposed.",
                            "paper_evidence_ids": ["selected"],
                            "paper_evidence_quotes": ["The selected span reports a GPU-backed training result."],
                            "dataset": {"name": "CIFAR-10", "source": "torchvision.datasets.CIFAR10", "requires_download": True},
                            "implementation": {"type": "public_dataset", "repo_url": "", "reason": "No repo."},
                            "run_plan": {
                                "repo_url": "",
                                "config_path": "",
                                "command": "python run.py",
                                "dataset": "CIFAR-10",
                                "expected_artifact": "accuracy",
                            },
                            "gpu_required": True,
                            "estimated_runtime_minutes": 5,
                            "expected_metric": "accuracy",
                            "limitations": ["Invalid middle mode."],
                            "approval_question": "Run it?",
                        },
                        {
                            "id": "valid-probe",
                            "title": "Valid probe",
                            "kind": "gpu_replication_probe",
                            "reproduction_level": "probe",
                            "faithfulness": {
                                "level": "probe",
                                "summary": "Valid probe.",
                                "why_not_exact": "No exact repo.",
                                "paper_targets": ["accuracy"],
                                "resource_note": "short run",
                            },
                            "is_recommended": False,
                            "recommendation_reason": "Valid fallback.",
                            "hypothesis": "A probe can be exposed.",
                            "paper_evidence_ids": ["selected"],
                            "paper_evidence_quotes": ["The selected span reports a GPU-backed training result."],
                            "dataset": {"name": "CIFAR-10", "source": "torchvision.datasets.CIFAR10", "requires_download": True},
                            "implementation": {"type": "public_dataset", "repo_url": "", "reason": "No repo."},
                            "run_plan": {
                                "repo_url": "",
                                "config_path": "",
                                "command": "python run.py",
                                "dataset": "CIFAR-10",
                                "expected_artifact": "accuracy",
                            },
                            "gpu_required": True,
                            "estimated_runtime_minutes": 5,
                            "expected_metric": "accuracy",
                            "limitations": ["Probe only."],
                            "approval_question": "Run it?",
                        },
                    ],
                    "recommended_candidate_id": "invalid-middle-mode",
                }
            )

        gateway = ModelGateway(provider="hf", call_model=candidate_call)
        result = gateway.experiment_candidates(
            paper_title="GPU Training Paper",
            selected_span="The selected span reports a GPU-backed training result.",
            translated_span="",
            source_text="The selected span reports a GPU-backed training result.",
            question="Find research directions from the paper.",
            locale="en",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIn("invalid reproduction_level scaled", result.error or "")

    def test_experiment_candidates_reject_exact_without_source_repo(self):
        def candidate_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "candidates": [
                        {
                            "id": "exact-reproduction",
                            "title": "Exact reproduction",
                            "kind": "gpu_replication_probe",
                            "reproduction_level": "exact",
                            "faithfulness": {
                                "level": "exact",
                                "summary": "Claims exact reproduction.",
                                "why_not_exact": "",
                                "paper_targets": ["accuracy"],
                                "resource_note": "short run",
                            },
                            "is_recommended": True,
                            "recommendation_reason": "It claims to be exact.",
                            "hypothesis": "Run the selected claim exactly.",
                            "paper_evidence_ids": ["selected"],
                            "paper_evidence_quotes": ["The selected span reports the claim."],
                            "dataset": {"name": "Public benchmark", "source": "public", "requires_download": True},
                            "implementation": {"type": "public_dataset", "repo_url": "", "reason": "No paper repo listed."},
                            "run_plan": {
                                "repo_url": "",
                                "config_path": "",
                                "command": "python run.py",
                                "dataset": "Public benchmark",
                                "expected_artifact": "accuracy",
                            },
                            "gpu_required": True,
                            "estimated_runtime_minutes": 10,
                            "expected_metric": "accuracy",
                            "limitations": ["Short run."],
                            "approval_question": "Run exact reproduction?",
                        },
                        {
                            "id": "probe-reproduction",
                            "title": "Probe reproduction",
                            "kind": "gpu_replication_probe",
                            "reproduction_level": "probe",
                            "faithfulness": {
                                "level": "probe",
                                "summary": "Bounded run.",
                                "why_not_exact": "No source-listed implementation repo is present.",
                                "paper_targets": ["accuracy"],
                                "resource_note": "short run",
                            },
                            "is_recommended": False,
                            "recommendation_reason": "Safer bounded option.",
                            "hypothesis": "Run a bounded public-data reproduction.",
                            "paper_evidence_ids": ["selected"],
                            "paper_evidence_quotes": ["The selected span reports the claim."],
                            "dataset": {"name": "Public benchmark", "source": "public", "requires_download": True},
                            "implementation": {"type": "public_dataset", "repo_url": "", "reason": "No paper repo listed."},
                            "run_plan": {
                                "repo_url": "",
                                "config_path": "",
                                "command": "python run_probe.py",
                                "dataset": "Public benchmark",
                                "expected_artifact": "accuracy",
                            },
                            "gpu_required": True,
                            "estimated_runtime_minutes": 10,
                            "expected_metric": "accuracy",
                            "limitations": ["Not exact."],
                            "approval_question": "Run probe reproduction?",
                        },
                    ],
                    "recommended_candidate_id": "exact-reproduction",
                }
            )

        gateway = ModelGateway(provider="hf", call_model=candidate_call)
        result = gateway.experiment_candidates(
            paper_title="Repo-less Paper",
            selected_span="The selected span reports the claim.",
            translated_span="",
            source_text="The selected span reports the claim.",
            question="Can we reproduce this exactly?",
            locale="en",
            reproduction_level="exact",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIn("exact reproduction without a source-listed paper repo", result.error or "")

    def test_experiment_candidates_reject_exact_with_repo_but_missing_run_plan_details(self):
        repo_url = "https://github.com/example/official-paper-repo"

        def candidate_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "candidates": [
                        {
                            "id": "exact-reproduction",
                            "title": "Exact reproduction",
                            "kind": "gpu_replication_probe",
                            "reproduction_level": "exact",
                            "faithfulness": {
                                "level": "exact",
                                "summary": "Claims exact reproduction.",
                                "why_not_exact": "",
                                "paper_targets": ["accuracy"],
                                "resource_note": "short run",
                            },
                            "is_recommended": True,
                            "recommendation_reason": "It claims to be exact.",
                            "hypothesis": "Run the selected claim exactly.",
                            "paper_evidence_ids": ["selected"],
                            "paper_evidence_quotes": ["The selected span reports the claim."],
                            "dataset": {"name": "Official benchmark", "source": "paper repo", "requires_download": True},
                            "implementation": {"type": "paper_repo", "repo_url": repo_url, "reason": "Source-listed repo."},
                            "run_plan": {
                                "repo_url": repo_url,
                                "config_path": "",
                                "command": "",
                                "dataset": "",
                                "expected_artifact": "accuracy",
                            },
                            "gpu_required": True,
                            "estimated_runtime_minutes": 10,
                            "expected_metric": "accuracy",
                            "limitations": ["Short run."],
                            "approval_question": "Run exact reproduction?",
                        },
                        {
                            "id": "probe-reproduction",
                            "title": "Probe reproduction",
                            "kind": "gpu_replication_probe",
                            "reproduction_level": "probe",
                            "faithfulness": {
                                "level": "probe",
                                "summary": "Bounded run.",
                                "why_not_exact": "No config path is confirmed.",
                                "paper_targets": ["accuracy"],
                                "resource_note": "short run",
                            },
                            "is_recommended": False,
                            "recommendation_reason": "Safer bounded option.",
                            "hypothesis": "Run a bounded repo-adjacent reproduction.",
                            "paper_evidence_ids": ["selected"],
                            "paper_evidence_quotes": ["The selected span reports the claim."],
                            "dataset": {"name": "Official benchmark", "source": "paper repo", "requires_download": True},
                            "implementation": {"type": "paper_repo", "repo_url": repo_url, "reason": "Source-listed repo."},
                            "run_plan": {
                                "repo_url": repo_url,
                                "config_path": "",
                                "command": "python run_probe.py",
                                "dataset": "Official benchmark subset",
                                "expected_artifact": "accuracy",
                            },
                            "gpu_required": True,
                            "estimated_runtime_minutes": 10,
                            "expected_metric": "accuracy",
                            "limitations": ["Not exact."],
                            "approval_question": "Run probe reproduction?",
                        },
                    ],
                    "recommended_candidate_id": "exact-reproduction",
                }
            )

        gateway = ModelGateway(provider="hf", call_model=candidate_call)
        result = gateway.experiment_candidates(
            paper_title="Repo Paper",
            selected_span="The selected span reports the claim.",
            translated_span="",
            source_text=f"The selected span reports the claim. Official implementation: {repo_url}",
            question="Can we reproduce this exactly?",
            locale="en",
            reproduction_level="exact",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIn("without run_plan.config_path", result.error or "")
        self.assertIn("without run_plan.command", result.error or "")
        self.assertIn("without run_plan.dataset", result.error or "")

    def test_gpu_script_repairs_mock_random_dataset_before_service_run(self):
        calls: list[str] = []

        def gpu_script_call(prompt, model_id, max_new_tokens):
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(
                    {
                        "script": (
                            "import torch\n"
                            "from torchtext.data.utils import get_tokenizer\n"
                            "from torch.utils.data import Dataset\n\n"
                            "class MockTranslationDataset(Dataset):\n"
                            "    def __init__(self):\n"
                            "        self.inputs = torch.randint(0, 100, (32, 16))\n"
                            "    def __len__(self):\n"
                            "        return len(self.inputs)\n"
                            "    def __getitem__(self, index):\n"
                            "        return self.inputs[index]\n\n"
                            "def run_paperlens_gpu_probe(config=None):\n"
                            "    return {'passed': True, 'metrics': {}, 'rows': [], 'logs': [], 'hardware': {'cudaAvailable': torch.cuda.is_available()}, 'dataset': {'name': 'MockTranslationDataset'}, 'limitations': [], 'claim_comparison': {}}\n"
                        ),
                        "entrypoint": "run_paperlens_gpu_probe",
                        "dependencies": ["torch"],
                        "hardware": "T4",
                        "dataset": {"name": "MockTranslationDataset", "source": "generated tensors"},
                        "expected_outputs": ["loss"],
                        "paper_claim_comparison_plan": "Compare speed.",
                        "limitations": ["Generated tensor data."],
                    }
                )
            if len(calls) == 2:
                return json.dumps(
                    {
                        "script": (
                            "import torch\n"
                            "from datasets import load_dataset\n\n"
                            "def run_paperlens_gpu_probe(config=None):\n"
                            "    cuda = torch.cuda.is_available()\n"
                            "    records = load_dataset('multi30k', split='train[:8]')\n"
                            "    value = eval('len(records)')\n"
                            "    return {'passed': True, 'metrics': {'rows': value}, 'rows': [{'metric': 'rows', 'value': value}], 'logs': [], 'hardware': {'cudaAvailable': cuda}, 'dataset': {'name': 'Multi30k'}, 'limitations': [], 'claim_comparison': {}}\n"
                        ),
                        "entrypoint": "run_paperlens_gpu_probe",
                        "dependencies": ["torch", "datasets"],
                        "hardware": "T4",
                        "dataset": {"name": "Multi30k", "source": "bentrevett/multi30k train[:8]"},
                        "expected_outputs": ["rows"],
                        "paper_claim_comparison_plan": "Load a bounded public dataset subset.",
                        "limitations": ["Not full WMT14 training."],
                    }
                )
            return json.dumps(
                {
                    "script": (
                        "import torch\n"
                        "from datasets import load_dataset\n\n"
                        "def run_paperlens_gpu_probe(config=None):\n"
                        "    config = config or {}\n"
                        "    cuda = torch.cuda.is_available()\n"
                        "    device = torch.device('cuda' if cuda else 'cpu')\n"
                        "    records = load_dataset('bentrevett/multi30k', split='train[:64]')\n"
                        "    lengths = [len(str(row.get('en') or '').split()) for row in records]\n"
                        "    tensor = torch.tensor(lengths, dtype=torch.float32, device=device)\n"
                        "    mean_length = float(tensor.mean().detach().cpu()) if tensor.numel() else 0.0\n"
                        "    reportHtml = f\"\"\"<section><h1>Multi30k Probe</h1><p><strong>Paper claim</strong>: the paper reports WMT translation quality, so this probe checks a bounded real translation dataset path rather than exact training.</p><p><strong>Paper evidence</strong>: the approved source span mentions WMT 2014 English-to-German and English-to-French results.</p><p><strong>Experiment setup</strong>: load bentrevett/multi30k train[:64] and compute a GPU tensor metric from real English examples.</p><figure><svg viewBox='0 0 240 80' role='img' aria-label='Measured metric mean token length'><rect x='12' y='20' width='{max(4, min(200, mean_length * 8))}' height='24'></rect><text x='12' y='62'>measured metric mean tokens {mean_length:.2f}</text></svg></figure><p><strong>Claim comparison</strong>: directional_probe_only, not a WMT reproduction.</p><p><strong>Limitations</strong>: bounded public dataset probe; next step is exact paper training/eval config.</p></section>\"\"\"\n"
                        "    return {\n"
                        "        'passed': bool(tensor.numel()),\n"
                        "        'metrics': {'mean_english_tokens': mean_length},\n"
                        "        'rows': [{'metric': 'mean_english_tokens', 'value': mean_length, 'split': 'train[:64]'}],\n"
                        "        'logs': ['Loaded a bounded public Multi30k subset through datasets.'],\n"
                        "        'hardware': {'cudaAvailable': cuda, 'device': str(device)},\n"
                        "        'dataset': {'name': 'Multi30k', 'source': 'bentrevett/multi30k train[:64]'},\n"
                        "        'limitations': ['Directional GPU data-loading and metric probe, not full WMT14 training.'],\n"
                        "        'claim_comparison': {'verdict': 'directional_probe_only'},\n"
                        "        'artifacts': {'reportHtml': reportHtml, 'metrics': {'mean_english_tokens': mean_length}, 'manifest': {'reproductionLevel': 'probe'}},\n"
                        "    }\n"
                    ),
                    "entrypoint": "run_paperlens_gpu_probe",
                    "dependencies": ["torch", "datasets"],
                    "hardware": "T4",
                    "dataset": {"name": "Multi30k", "source": "bentrevett/multi30k train[:64]"},
                    "reproduction_level": "probe",
                    "reproduction_plan": {
                        "level": "probe",
                        "repo_url": "",
                        "config_path": "",
                        "command": "",
                        "dataset": "bentrevett/multi30k train[:64]",
                        "expected_artifact": "mean_english_tokens",
                        "faithfulness_note": "Directional probe only.",
                    },
                    "expected_outputs": ["mean_english_tokens"],
                    "paper_claim_comparison_plan": "Use a bounded real translation dataset subset as a directional probe.",
                    "limitations": ["Not full WMT14 training."],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=gpu_script_call)
        result = gateway.gpu_script(
            paper_title="Attention Is All You Need",
            selected_span="Experiments on two machine translation tasks show the Transformer is more parallelizable.",
            source_text=(
                "Experiments on two machine translation tasks show the Transformer is more parallelizable. "
                "The paper reports WMT 2014 English-to-German and English-to-French translation results."
            ),
            candidate={
                "id": "gpu-replication-probe",
                "title": "Translation GPU probe",
                "kind": "gpu_replication_probe",
                "reproduction_level": "probe",
                "paper_evidence_ids": ["selected"],
                "dataset": {"name": "public translation subset", "source": "public dataset"},
                "expected_metric": "bounded translation metric",
            },
            locale="en",
            implementation_repo_manifests=[],
            use_model=True,
        )

        self.assertFalse(result.used_fallback, result.error)
        self.assertIsNone(result.error)
        self.assertEqual(len(calls), 3)
        self.assertIn("load_dataset", result.data["script"])
        self.assertNotIn("MockTranslationDataset", result.data["script"])
        self.assertNotIn("torchtext", result.data["script"])
        self.assertNotIn("torch.randint", result.data["script"])
        self.assertNotIn("eval(", result.data["script"])
        self.assertNotIn("load_dataset('multi30k'", result.data["script"])
        self.assertIn("bentrevett/multi30k", result.data["script"])

    def test_gpu_script_raw_python_requires_model_repair_envelope(self):
        raw_code = """
```python
import torch

def run_paperlens_gpu_probe(config=None):
    cuda = torch.cuda.is_available()
    report = "<figure><svg width='120' height='32'><rect width='100' height='20'></rect></svg><figcaption>Probe metric</figcaption></figure>"
    return {
        "passed": True,
        "metrics": {"probe_metric": 1.0},
        "rows": [{"metric": "probe_metric", "value": 1.0}],
        "logs": [f"cuda={cuda}"],
        "hardware": {"cudaAvailable": cuda},
        "dataset": {"name": "CIFAR-10", "source": "torchvision.datasets.CIFAR10"},
        "limitations": ["bounded probe"],
        "claim_comparison": {"verdict": "directional_probe_only"},
        "artifacts": {
            "reportHtml": report,
            "manifest": {"reproductionLevel": "probe", "dataset": "CIFAR-10"},
            "metrics": {"probe_metric": 1.0},
        },
    }
```
""".strip()

        calls = []

        def repairing_raw_python_call(prompt, model_id, max_new_tokens):
            calls.append(prompt)
            if "repairing a PaperLens Lab GPU replication probe JSON" in prompt:
                return json.dumps(
                    {
                        "script": (
                            "import torch\n\n"
                            "def run_paperlens_gpu_probe(config=None):\n"
                            "    cuda = torch.cuda.is_available()\n"
                            "    report = \"<section><h1>Probe metric</h1><p><strong>Paper claim</strong>: the approved paper claim is tested only as a bounded probe.</p><p><strong>Paper evidence</strong>: source span evidence is the comparison anchor.</p><p><strong>Experiment setup</strong>: run the generated code path on CIFAR-10 style data.</p><figure><svg width='120' height='32'><rect width='100' height='20'></rect></svg><figcaption>Measured metric probe_metric 1.0</figcaption></figure><p><strong>Claim comparison</strong>: directional_probe_only.</p><p><strong>Limitations</strong>: bounded probe; next step is exact reproduction.</p></section>\"\n"
                            "    return {'passed': True, 'metrics': {'probe_metric': 1.0}, "
                            "'rows': [{'metric': 'probe_metric', 'value': 1.0}], "
                            "'logs': [f'cuda={cuda}'], 'hardware': {'cudaAvailable': cuda}, "
                            "'dataset': {'name': 'CIFAR-10', 'source': 'torchvision.datasets.CIFAR10'}, "
                            "'limitations': ['bounded probe'], "
                            "'claim_comparison': {'verdict': 'directional_probe_only'}, "
                            "'artifacts': {'reportHtml': report, 'manifest': {'reproductionLevel': 'probe'}, "
                            "'metrics': {'probe_metric': 1.0}}}\n"
                        ),
                        "entrypoint": "run_paperlens_gpu_probe",
                        "dependencies": ["torch", "torchvision"],
                        "hardware": "T4",
                        "dataset": {"name": "CIFAR-10", "source": "torchvision.datasets.CIFAR10"},
                        "reproduction_level": "probe",
                        "reproduction_plan": {
                            "level": "probe",
                            "repo_url": "",
                            "config_path": "",
                            "command": "python experiment.py",
                            "dataset": "CIFAR-10",
                            "expected_artifact": "reportHtml",
                            "faithfulness_note": "Model repaired the raw Python into the required JSON envelope.",
                        },
                        "expected_outputs": ["metrics", "rows", "reportHtml"],
                        "paper_claim_comparison_plan": "Compare the bounded metric directionally.",
                        "limitations": ["bounded probe"],
                    }
                )
            return raw_code

        gateway = ModelGateway(provider="hf", call_model=repairing_raw_python_call)
        result = gateway.gpu_script(
            paper_title="Deep Residual Learning for Image Recognition",
            selected_span="Residual networks are easier to optimize.",
            source_text="Residual networks are easier to optimize on CIFAR-10.",
            candidate={
                "id": "gpu-probe",
                "title": "Residual probe",
                "reproduction_level": "probe",
                "dataset": {"name": "CIFAR-10", "source": "torchvision.datasets.CIFAR10"},
                "run_plan": {"dataset": "CIFAR-10"},
            },
            locale="en",
            use_model=True,
        )

        self.assertFalse(result.used_fallback, result.error)
        self.assertIsNone(result.error)
        self.assertEqual(len(calls), 2)
        self.assertFalse(result.data.get("recovered_from_non_json", False))
        self.assertIn("run_paperlens_gpu_probe", result.data["script"])

    def test_gpu_script_does_not_treat_truncated_json_as_raw_python(self):
        truncated_json = (
            '{"script": "import torch\\n\\ndef run_paperlens_gpu_probe(config=None):\\n'
            "    return {'passed': True, 'metrics': {}, 'rows': [], 'logs': [], "
            "'hardware': {'cudaAvailable': torch.cuda.is_available()}, "
            "'dataset': {'name': 'CIFAR-10'}, 'limitations': [], "
            "'claim_comparison': {}, 'artifacts': {'reportHtml': '<figure><svg></svg></figure>'}}\", "
            '"reproduction_plan": {'
        )

        gateway = ModelGateway(provider="hf", call_model=lambda *_: truncated_json)
        result = gateway.gpu_script(
            paper_title="Deep Residual Learning for Image Recognition",
            selected_span="Residual networks are easier to optimize.",
            source_text="Residual networks are easier to optimize on CIFAR-10.",
            candidate={
                "id": "gpu-probe",
                "title": "Residual probe",
                "reproduction_level": "probe",
                "dataset": {"name": "CIFAR-10", "source": "torchvision.datasets.CIFAR10"},
                "run_plan": {"dataset": "CIFAR-10"},
            },
            locale="en",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIn("non-JSON", result.error or "")
        self.assertFalse(result.data.get("recovered_from_non_json"))

    def test_gpu_script_validation_failure_does_not_fallback(self):
        def invalid_gpu_script_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "script": (
                        "import torch\n\n"
                        "def run_paperlens_gpu_probe(config=None):\n"
                        "    data = torch.randn(8, 16)\n"
                        "    return {'passed': True, 'metrics': {'rows': 8}, 'rows': [], "
                        "'logs': [], 'hardware': {'cudaAvailable': torch.cuda.is_available()}, "
                        "'dataset': {'name': 'generated tensors'}, 'limitations': [], "
                        "'claim_comparison': {}}\n"
                    ),
                    "entrypoint": "run_paperlens_gpu_probe",
                    "dependencies": ["torch"],
                    "hardware": "T4",
                    "dataset": {"name": "Generated", "source": "torch.randn"},
                    "expected_outputs": ["rows"],
                    "paper_claim_comparison_plan": "Compare throughput.",
                    "limitations": ["Generated tensors."],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=invalid_gpu_script_call)
        result = gateway.gpu_script(
            paper_title="Attention Is All You Need",
            selected_span="Experiments show the Transformer is more parallelizable.",
            source_text="Experiments show the Transformer is more parallelizable on translation tasks.",
            candidate={
                "id": "gpu-replication-probe",
                "title": "Translation GPU probe",
                "kind": "gpu_replication_probe",
                "paper_evidence_ids": ["selected"],
                "dataset": {"name": "public translation subset", "source": "public dataset"},
                "expected_metric": "tokens/sec",
            },
            locale="en",
            implementation_repo_manifests=[],
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIn("torch.randn", result.error or "")
        self.assertNotIn("fallback used", result.error or "")

    def test_exact_gpu_script_requires_repo_config_dataset_and_command_plan(self):
        calls: list[str] = []

        def exact_script_without_plan_call(prompt, model_id, max_new_tokens):
            calls.append(prompt)
            return json.dumps(
                {
                    "script": (
                        "import torch\n\n"
                        "def run_paperlens_gpu_probe(config=None):\n"
                        "    cuda = torch.cuda.is_available()\n"
                        "    model = torch.nn.Linear(4, 2)\n"
                        "    model.eval()\n"
                        "    return {\n"
                        "        'passed': True,\n"
                        "        'metrics': {'rows': 1},\n"
                        "        'rows': [{'metric': 'rows', 'value': 1}],\n"
                        "        'logs': ['repo-backed exact run plan was not executed'],\n"
                        "        'hardware': {'cudaAvailable': cuda},\n"
                        "        'dataset': {'name': 'ImageNet validation', 'source': 'paper repo config'},\n"
                        "        'limitations': ['bounded validation slice'],\n"
                        "        'claim_comparison': {'verdict': 'completed'},\n"
                        "    }\n"
                    ),
                    "entrypoint": "run_paperlens_gpu_probe",
                    "dependencies": ["torch"],
                    "hardware": "T4",
                    "dataset": {"name": "ImageNet validation", "source": "paper repo config"},
                    "reproduction_level": "exact",
                    "reproduction_plan": {
                        "level": "exact",
                        "repo_url": "",
                        "config_path": "",
                        "command": "",
                        "dataset": "",
                        "expected_artifact": "top1 accuracy",
                        "faithfulness_note": "Exact label without executable repo plan must be rejected.",
                    },
                    "expected_outputs": ["top1 accuracy"],
                    "paper_claim_comparison_plan": "Compare with the reported ImageNet table.",
                    "limitations": ["bounded validation slice"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=exact_script_without_plan_call)
        result = gateway.gpu_script(
            paper_title="Deep Residual Learning for Image Recognition",
            selected_span="We provide comprehensive empirical evidence.",
            source_text="Official implementation: https://github.com/KaimingHe/deep-residual-networks",
            candidate={
                "id": "exact-reproduction",
                "title": "Exact ImageNet reproduction",
                "kind": "gpu_replication_probe",
                "reproduction_level": "exact",
                "paper_evidence_ids": ["selected"],
                "implementation": {
                    "type": "paper_repo",
                    "repo_url": "https://github.com/KaimingHe/deep-residual-networks",
                },
                "dataset": {"name": "ImageNet validation", "source": "official config"},
                "expected_metric": "top1 accuracy",
            },
            locale="en",
            implementation_repo_manifests=[
                {"url": "https://github.com/KaimingHe/deep-residual-networks", "status": "inspected"}
            ],
            use_model=True,
        )

        self.assertEqual(len(calls), 4)
        self.assertFalse(result.used_fallback)
        self.assertIn("exact GPU script requires reproduction_plan.repo_url", result.error or "")
        self.assertIn("exact GPU script requires reproduction_plan.config_path", result.error or "")
        self.assertIn("exact GPU script requires reproduction_plan.command", result.error or "")
        self.assertIn("exact GPU script requires reproduction_plan.dataset", result.error or "")
        self.assertNotIn("blocked call eval", result.error or "")

    def test_exact_gpu_script_repo_must_match_inspected_paper_repo(self):
        def exact_script_wrong_repo_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "script": (
                        "import torch\n\n"
                        "def run_paperlens_gpu_probe(config=None):\n"
                        "    cuda = torch.cuda.is_available()\n"
                        "    model = torch.nn.Linear(4, 2)\n"
                        "    model.eval()\n"
                        "    return {\n"
                        "        'passed': True,\n"
                        "        'metrics': {'rows': 1},\n"
                        "        'rows': [{'metric': 'rows', 'value': 1}],\n"
                        "        'logs': ['repo-backed exact run plan was executed'],\n"
                        "        'hardware': {'cudaAvailable': cuda},\n"
                        "        'dataset': {'name': 'ImageNet validation', 'source': 'paper repo config'},\n"
                        "        'limitations': ['bounded validation slice'],\n"
                        "        'claim_comparison': {'verdict': 'completed'},\n"
                        "    }\n"
                    ),
                    "entrypoint": "run_paperlens_gpu_probe",
                    "dependencies": ["torch"],
                    "hardware": "T4",
                    "dataset": {"name": "ImageNet validation", "source": "paper repo config"},
                    "reproduction_level": "exact",
                    "reproduction_plan": {
                        "level": "exact",
                        "repo_url": "https://github.com/other/repo",
                        "config_path": "configs/eval.yaml",
                        "command": "python tools/eval.py --config configs/eval.yaml",
                        "dataset": "ImageNet validation",
                        "expected_artifact": "top1 accuracy",
                        "faithfulness_note": "Exact plan uses the wrong repo.",
                    },
                    "expected_outputs": ["top1 accuracy"],
                    "paper_claim_comparison_plan": "Compare with the reported ImageNet table.",
                    "limitations": ["bounded validation slice"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=exact_script_wrong_repo_call)
        result = gateway.gpu_script(
            paper_title="Deep Residual Learning for Image Recognition",
            selected_span="We provide comprehensive empirical evidence.",
            source_text="Official implementation: https://github.com/KaimingHe/deep-residual-networks",
            candidate={
                "id": "exact-reproduction",
                "title": "Exact ImageNet reproduction",
                "kind": "gpu_replication_probe",
                "reproduction_level": "exact",
                "paper_evidence_ids": ["selected"],
                "implementation": {
                    "type": "paper_repo",
                    "repo_url": "https://github.com/KaimingHe/deep-residual-networks",
                },
                "dataset": {"name": "ImageNet validation", "source": "official config"},
                "expected_metric": "top1 accuracy",
            },
            locale="en",
            implementation_repo_manifests=[
                {"url": "https://github.com/KaimingHe/deep-residual-networks", "status": "inspected"}
            ],
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIn("exact GPU script repo_url must match an inspected paper implementation repo", result.error or "")
        self.assertNotIn("blocked call eval", result.error or "")

    def test_exact_gpu_script_must_match_approved_reproduction_level(self):
        def scaled_script_for_exact_call(prompt, model_id, max_new_tokens):
            return json.dumps(
                {
                    "script": (
                        "import torch\n\n"
                        "def run_paperlens_gpu_probe(config=None):\n"
                        "    cuda = torch.cuda.is_available()\n"
                        "    model = torch.nn.Linear(4, 2)\n"
                        "    model.eval()\n"
                        "    return {\n"
                        "        'passed': True,\n"
                        "        'metrics': {'rows': 1},\n"
                        "        'rows': [{'metric': 'rows', 'value': 1}],\n"
                        "        'logs': ['scaled run'],\n"
                        "        'hardware': {'cudaAvailable': cuda},\n"
                        "        'dataset': {'name': 'ImageNet validation subset', 'source': 'paper repo config'},\n"
                        "        'limitations': ['bounded validation slice'],\n"
                        "        'claim_comparison': {'verdict': 'completed'},\n"
                        "    }\n"
                    ),
                    "entrypoint": "run_paperlens_gpu_probe",
                    "dependencies": ["torch"],
                    "hardware": "T4",
                    "dataset": {"name": "ImageNet validation subset", "source": "paper repo config"},
                    "reproduction_level": "scaled",
                    "reproduction_plan": {
                        "level": "scaled",
                        "repo_url": "",
                        "config_path": "",
                        "command": "python run_scaled.py",
                        "dataset": "ImageNet validation subset",
                        "expected_artifact": "rows",
                        "faithfulness_note": "Scaled script must not satisfy an approved exact candidate.",
                    },
                    "expected_outputs": ["rows"],
                    "paper_claim_comparison_plan": "Compare a scaled subset.",
                    "limitations": ["bounded validation slice"],
                }
            )

        gateway = ModelGateway(provider="hf", call_model=scaled_script_for_exact_call)
        result = gateway.gpu_script(
            paper_title="Deep Residual Learning for Image Recognition",
            selected_span="We provide comprehensive empirical evidence.",
            source_text="Official implementation: https://github.com/KaimingHe/deep-residual-networks",
            candidate={
                "id": "exact-reproduction",
                "title": "Exact ImageNet reproduction",
                "kind": "gpu_replication_probe",
                "reproduction_level": "exact",
                "paper_evidence_ids": ["selected"],
                "implementation": {
                    "type": "paper_repo",
                    "repo_url": "https://github.com/KaimingHe/deep-residual-networks",
                },
                "dataset": {"name": "ImageNet validation", "source": "official config"},
                "expected_metric": "top1 accuracy",
            },
            locale="en",
            implementation_repo_manifests=[
                {"url": "https://github.com/KaimingHe/deep-residual-networks", "status": "inspected"}
            ],
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertIn("reproduction_level must match approved reproduction level", result.error or "")

    def test_growth_ideas_repair_unknown_evidence_ids(self):
        calls = []

        def growth_call(prompt, model_id, max_new_tokens):
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps(
                    {
                        "ideas": [
                            {
                                "idea": "Measure a sharper source-bound contrast.",
                                "source_evidence": ["paper:selected-span", "run:r1", "growth_idea:invented"],
                                "novelty_angle": "Narrow the previous observation.",
                                "testable_next_step": "Reuse indexed evidence rows with a stricter contrast.",
                                "risk": "The effect may be too small.",
                            }
                        ],
                        "fine_tuning_signal": "none",
                        "reason": "",
                    }
                )
            return json.dumps(
                {
                    "ideas": [
                        {
                            "idea": "Measure a sharper source-bound contrast.",
                            "source_evidence": ["paper:selected-span", "run:r1"],
                            "novelty_angle": "Narrow the previous observation.",
                            "testable_next_step": "Reuse indexed evidence rows with a stricter contrast.",
                            "risk": "The effect may be too small.",
                        }
                    ],
                    "fine_tuning_signal": "none",
                    "reason": "No repeated model-output failure pattern.",
                }
            )

        gateway = ModelGateway(provider="hf", call_model=growth_call)
        result = gateway.growth_ideas(
            paper_title="Demo Paper",
            paper_memory=[{"id": "paper:selected-span", "summary": "Source-bound evidence."}],
            mini_lab_result="run:r1 actual source-bound mini-lab execution",
            selected_span="Source-bound evidence.",
            locale="en",
            use_model=True,
        )

        self.assertFalse(result.used_fallback)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.data["ideas"][0]["source_evidence"], ["paper:selected-span", "run:r1"])

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
