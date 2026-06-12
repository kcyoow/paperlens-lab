import json
import os
import tempfile
import unittest
from pathlib import Path

from paperlens_lab.ingest import PaperSource
from paperlens_lab.model_adapter import ModelGateway
from paperlens_lab.real_paper_runner import RealPaperCase, run_real_paper_case


class RealPaperRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        os.environ["PAPERLENS_TRACE_PATH"] = str(root / "traces.jsonl")
        os.environ["PAPERLENS_MEMORY_PATH"] = str(root / "memory.jsonl")
        self.output_dir = root / "real_runs"

    def tearDown(self):
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
        os.environ.pop("PAPERLENS_MEMORY_PATH", None)
        self.tempdir.cleanup()

    def test_real_paper_runner_checks_pdf_spans_litm_and_memory(self):
        source = PaperSource(
            title="A Real PDF-Shaped Paper",
            authors="A. Researcher",
            source_label="arXiv:0000.00000",
            pdf_url="https://arxiv.org/pdf/0000.00000",
            text=self.long_pdf_text(),
        )
        gateway = ModelGateway(provider="hf", call_model=self.fake_call)
        result = run_real_paper_case(
            RealPaperCase(
                name="real_pdf_shaped",
                arxiv="0000.00000",
                question="What does the selected span actually support?",
                idea="Make a tiny experiment from the selected claim.",
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
        saved = self.output_dir / "real_pdf_shaped.json"
        self.assertTrue(saved.exists())

    def fake_call(self, prompt: str, model_id: str, max_new_tokens: int):
        if '"translations"' in prompt:
            return json.dumps(
                {
                    "translations": [
                        {
                            "span_id": span_id,
                            "translation": f"{span_id} 번역은 F1, Table 2, 3.2를 controlled evidence conditions에서만 보존한다.",
                            "preserved_terms": ["F1", "Table 2"],
                            "uncertain_phrases": [],
                        }
                        for span_id in self._span_ids(prompt)
                    ],
                    "notes": [],
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
        if '"research_question"' in prompt:
            return json.dumps(
                {
                    "research_question": "Does the selected technique improve F1 on a toy set?",
                    "mini_lab_goal": "Run a 30-minute baseline versus one paper-inspired variant.",
                    "dataset": {"name": "Toy table", "fallback": "10 hand-built examples from the paper"},
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
                        "source_evidence": ["paper:selected-middle", "run:r1"],
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
                lines.append(
                    "This controlled evidence sentence reports that the method preserves F1 in Table 2 and improves 3.2 points only under controlled evidence conditions."
                )
        return "\n".join(lines)

    def _span_ids(self, prompt):
        import re

        return re.findall(r'"span_id":\s*"(P\d+\.S\d+)"', prompt)

    def _selected_span_id(self, prompt):
        import re

        match = re.search(r"Selected span id:\s*(P\d+\.S\d+)", prompt)
        return match.group(1) if match else "P0.S1"

    def _selected_span(self, prompt):
        import re

        match = re.search(r"Selected span:\s*(.*?)\nAvailable translation:", prompt, re.DOTALL)
        return match.group(1).strip() if match else "The method improves 3.2 points."


if __name__ == "__main__":
    unittest.main()
