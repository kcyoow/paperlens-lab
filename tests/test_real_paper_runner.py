import json
import os
import tempfile
import unittest
from pathlib import Path

from paperlens_lab.ingest import PaperSource
from paperlens_lab.model_adapter import ModelGateway
from paperlens_lab.real_paper_runner import (
    RealPaperCase,
    _selected_spans,
    evaluate_growth_iteration,
    run_real_paper_case,
)


class RealPaperRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        os.environ["PAPERLENS_TRACE_PATH"] = str(root / "traces.jsonl")
        os.environ["PAPERLENS_MEMORY_PATH"] = str(root / "memory.jsonl")
        os.environ["PAPERLENS_SOURCE_INDEX_DIR"] = str(root / "source_index")
        self.output_dir = root / "real_runs"
        self.calls: list[tuple[str, str]] = []

    def tearDown(self):
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
        os.environ.pop("PAPERLENS_MEMORY_PATH", None)
        os.environ.pop("PAPERLENS_SOURCE_INDEX_DIR", None)
        self.tempdir.cleanup()

    def test_real_paper_runner_checks_pdf_spans_litm_and_memory(self):
        source = PaperSource(
            title="A Real PDF-Shaped Paper",
            authors="A. Researcher",
            source_label="arXiv:0000.00000",
            pdf_url="https://arxiv.org/pdf/0000.00000",
            text=self.long_pdf_text(),
        )
        gateway = ModelGateway(provider="hf", call_model=self.fake_call, quality_model_id="test-quality")
        result = run_real_paper_case(
            RealPaperCase(
                name="real_pdf_shaped",
                arxiv="0000.00000",
                question="What does the selected span actually support?",
                idea="Make a source-bound experiment from the selected claim.",
            ),
            source=source,
            gateway=gateway,
            use_model=True,
            max_translate_spans=9,
            max_reader_spans=120,
            output_dir=self.output_dir,
        )

        self.assertTrue(result["passed"], result["evaluations"])
        self.assertGreaterEqual(result["source"]["page_marker_count"], 3)
        self.assertGreaterEqual(result["reader"]["visible_span_count"], 60)
        labels = {item["position_label"] for item in result["reader"]["selected_span_positions"]}
        self.assertIn("middle", labels)
        self.assertGreaterEqual(result["memory"]["records_after_growth"], 3)
        starter_output = result["model_outputs"]["starter_code"]
        self.assertEqual(starter_output["task"], "starter_code")
        self.assertEqual(starter_output["provider"], "hf")
        self.assertFalse(starter_output["used_fallback"])
        self.assertTrue(starter_output["trace_id"])
        self.assertIn("model-generated-starter", starter_output["data"]["code"])
        saved = self.output_dir / "real_pdf_shaped.json"
        self.assertTrue(saved.exists())
        self.assertTrue(any(model_id == "test-quality" for _, model_id in self.calls))

    def test_selected_spans_prefer_informative_middle_span(self):
        spans = [
            {"id": f"P0.S{idx}", "original": "fragment", "position": idx}
            for idx in range(100)
        ]
        spans[50] = {
            "id": "P5.S1",
            "original": "of continuous representations z = (z1, ..., zn).",
            "position": 50,
        }
        spans[53] = {
            "id": "P5.S4",
            "original": "We propose a method that improves F1 by 3.2 points in Table 2 under controlled evidence conditions.",
            "position": 53,
        }

        selected = _selected_spans(spans)

        middle = next(item for item in selected if item["position_label"] == "middle")
        self.assertEqual(middle["id"], "P5.S4")

    def test_fine_tuning_gate_does_not_use_fallback_only_failures(self):
        source = PaperSource(
            title="A Parse-Only Paper",
            authors="A. Researcher",
            source_label="arXiv:0000.00001",
            pdf_url="https://arxiv.org/pdf/0000.00001",
            text=self.long_pdf_text(),
        )

        result = run_real_paper_case(
            RealPaperCase(
                name="parse_only",
                arxiv="0000.00001",
                question="What does the selected span support?",
                idea="Make a source-bound experiment from the selected claim.",
            ),
            source=source,
            use_model=False,
            max_translate_spans=3,
            max_reader_spans=90,
            output_dir=self.output_dir,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["fine_tuning"]["recommendation"], "no")
        self.assertIn("Model-backed validation was not enabled", result["fine_tuning"]["reason"])

    def test_growth_iteration_requires_same_idea_to_cite_memory_run_and_paper(self):
        memories = [
            {"id": "paper:selected-middle", "kind": "paper_span"},
            {"id": "run:r1", "kind": "mini_lab_result"},
            {"id": "growth_idea:abc123", "kind": "growth_idea"},
        ]
        result = evaluate_growth_iteration(
            {
                "ideas": [
                    {
                        "idea": "Use only the prior idea.",
                        "source_evidence": ["growth_idea:abc123"],
                        "novelty_angle": "narrow",
                        "testable_next_step": "run a source-bound split",
                        "risk": "thin evidence",
                    },
                    {
                        "idea": "Use only the paper and run.",
                        "source_evidence": ["paper:selected-middle", "run:r1"],
                        "novelty_angle": "narrow",
                        "testable_next_step": "run a source-bound split",
                        "risk": "thin evidence",
                    },
                ]
            },
            memories,
        )

        self.assertFalse(result.passed)
        self.assertIn("together", " ".join(result.reasons))

    def fake_call(self, prompt: str, model_id: str, max_new_tokens: int):
        self.calls.append((prompt, model_id))
        if '"translations"' in prompt:
            return json.dumps(
                {
                    "translations": [
                        {
                            "span_id": item["span_id"],
                            "translation": f"{item['span_id']} 번역 초안: {item['text']}",
                            "preserved_terms": ["F1", "Table 2", "MIDSTREAM-LITM-427"],
                            "uncertain_phrases": [],
                        }
                        for item in self._span_payloads(prompt)
                    ],
                    "notes": [],
                }
            )
        if "Long evidence packet:" in prompt:
            phrase = self._target_phrase(prompt)
            evidence = self._long_evidence(prompt)
            target = next((item for item in evidence if phrase and phrase in item.get("text", "")), evidence[0])
            return json.dumps(
                {
                    "answer": f"{target['source_id']} contains the middle-only phrase and supports only the local claim.",
                    "evidence": [{"source_id": target["source_id"], "quote": target["text"]}],
                    "confidence": "medium",
                    "needs_more_context": True,
                    "unsupported_assumptions": ["full-paper superiority and fine-tuning need require more context"],
                }
            )
        if '"confidence"' in prompt:
            span_id = self._selected_span_id(prompt)
            quote = self._selected_span(prompt)
            return json.dumps(
                {
                    "answer": f"선택 span {span_id}의 범위 안에서만 답할 수 있다.",
                    "evidence": [{"source_id": span_id, "quote": quote}],
                    "confidence": "medium",
                    "needs_more_context": True,
                    "unsupported_assumptions": [],
                }
            )
        if '"why_this_matches_span"' in prompt and '"limitations"' in prompt:
            return json.dumps(
                {
                    "code": self._starter_code(),
                    "why_this_matches_span": "The starter uses query and context to compare evidence-conditioned candidates from the selected span.",
                    "limitations": ["This source-bound run is not a reproduction of the full paper result."],
                }
            )
        if '"research_question"' in prompt:
            return json.dumps(
                {
                    "research_question": "Does the selected technique improve F1 on indexed paper evidence?",
                    "mini_lab_goal": "Run a 30-minute baseline versus one paper-inspired variant.",
                    "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index rows"},
                    "baseline": "Direct keyword baseline",
                    "metric": "F1",
                    "steps": ["Create examples", "Run baseline", "Run variant", "Compare F1"],
                    "ablation": "Remove only the paper-inspired scoring feature.",
                    "failure_condition": "F1 does not improve over the baseline.",
                    "expected_result": "The variant may improve F1 on examples resembling the selected span.",
                    "faithfulness_notes": ["This is a small learning proxy."],
                    "starter_code_plan": ["baseline", "variant", "score"],
                    "support_span_ids": ["paper:selected-middle"],
                }
            )
        return json.dumps(
            {
                "ideas": [
                    {
                        "idea": "Test whether the F1 gain appears only on evidence-heavy examples.",
                        "source_evidence": self._growth_evidence_ids(prompt),
                        "novelty_angle": "Use the mini-lab result to narrow the paper claim.",
                        "testable_next_step": "Bucket ten examples by evidence density and compare F1 deltas.",
                        "risk": "Manual buckets may be noisy.",
                    }
                ],
                "fine_tuning_signal": "none",
                "reason": "No repeated real model-output failure has been observed.",
            }
        )

    def long_pdf_text(self):
        lines = []
        for page in range(1, 4):
            lines.append(f"[page {page}]")
            for idx in range(1, 31):
                global_idx = (page - 1) * 30 + idx
                if global_idx == 47:
                    lines.append(
                        "The MIDSTREAM-LITM-427 ablation reports that F1 improves by 3.2 points in Table 2 only under controlled evidence conditions and does not establish full-paper superiority."
                    )
                else:
                    lines.append(
                        f"This controlled distractor sentence {global_idx} discusses baselines, metrics, and evidence conditions without the unique middle anchor."
                    )
        return "\n".join(lines)

    def _span_ids(self, prompt):
        import re

        return re.findall(r'"span_id":\s*"(P\d+\.S\d+)"', prompt)

    def _span_payloads(self, prompt):
        import json
        import re

        match = re.search(r"Source spans:\n(.*?)(?:\n\nRules:|\n$)", prompt, re.DOTALL)
        if not match:
            return [{"span_id": span_id, "text": ""} for span_id in self._span_ids(prompt)]
        return json.loads(match.group(1))

    def _target_phrase(self, prompt):
        import re

        match = re.search(r"Exact phrase to locate:\s*(.*?)\nLong evidence packet:", prompt, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _long_evidence(self, prompt):
        import json
        import re

        match = re.search(r"Long evidence packet:\n(.*?)\n\nThe packet intentionally", prompt, re.DOTALL)
        return json.loads(match.group(1)) if match else []

    def _growth_evidence_ids(self, prompt):
        import re

        growth_ids = re.findall(r'"id":\s*"(growth_idea:[^"]+)"', prompt)
        evidence = ["paper:selected-middle", "run:r1"]
        if growth_ids:
            evidence.insert(1, growth_ids[0])
        return evidence

    def _selected_span_id(self, prompt):
        import re

        match = re.search(r"Selected span id:\s*(P\d+\.S\d+)", prompt)
        return match.group(1) if match else "P0.S1"

    def _selected_span(self, prompt):
        import re

        match = re.search(r"Selected span:\s*(.*?)\nAvailable translation:", prompt, re.DOTALL)
        return match.group(1).strip() if match else "The method improves 3.2 points."

    def _starter_code(self):
        return """# model-generated-starter
def baseline(example):
    query = example.get("query", "").lower()
    context = " ".join(example.get("context", [])).lower()
    best = example.get("candidates", [""])[0]
    best_score = -1
    for candidate in example.get("candidates", []):
        lowered = candidate.lower()
        score = int(lowered in query) + int(lowered in context)
        if score > best_score:
            best_score = score
            best = candidate
    return best

def paper_inspired(example):
    query = example.get("query", "").lower()
    context = [chunk.lower() for chunk in example.get("context", [])]
    best = example.get("candidates", [""])[0]
    best_score = -1
    for candidate in example.get("candidates", []):
        lowered = candidate.lower()
        score = 0
        for idx, chunk in enumerate(context):
            weight = len(context) - idx
            if lowered in chunk:
                score += weight
            if query and query.split()[0] in chunk and lowered in chunk:
                score += 1
        if score > best_score:
            best_score = score
            best = candidate
    return best

def score(output, gold):
    return 1.0 if output == gold else 0.0

def run(evidence_rows=None):
    examples = []
    for row in evidence_rows or []:
        examples.append({
            "source_id": row["source_id"],
            "text_hash": row["text_hash"],
            "query": row.get("query", ""),
            "context": [row.get("text", "")],
            "candidates": ["full-paper superiority", "controlled evidence conditions"],
            "gold": "controlled evidence conditions" if row.get("gold") else "full-paper superiority",
        })
    rows = []
    for example in examples:
        base = baseline(example)
        variant = paper_inspired(example)
        baseline_score = score(base, example["gold"])
        prototype_score = score(variant, example["gold"])
        rows.append(
            {
                "baseline_score": baseline_score,
                "prototype_score": prototype_score,
                "metric": "span_proxy_accuracy",
                "source_id": example["source_id"],
                "text_hash": example["text_hash"],
                "failure_condition": prototype_score <= baseline_score,
                "failure_rule": "prototype_score <= baseline_score",
            }
        )
    return rows
"""


if __name__ == "__main__":
    unittest.main()
