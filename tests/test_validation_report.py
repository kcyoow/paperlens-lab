import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from paperlens_lab.server import create_app
from paperlens_lab.validation_report import build_validation_summary


class ValidationReportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.day = self.root / "2026-06-13"
        self.run_dir = self.day / "hf_three_papers_rerun"
        self.run_dir.mkdir(parents=True)
        self._write_validation_tree()
        os.environ["PAPERLENS_VALIDATION_ROOT"] = str(self.root)
        os.environ["PAPERLENS_TRACE_PATH"] = str(self.root / "api_traces.jsonl")
        os.environ["PAPERLENS_MEMORY_PATH"] = str(self.root / "paper_memory.jsonl")

    def tearDown(self):
        os.environ.pop("PAPERLENS_VALIDATION_ROOT", None)
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
        os.environ.pop("PAPERLENS_MEMORY_PATH", None)
        self.tempdir.cleanup()

    def test_build_validation_summary_aggregates_real_paper_artifacts(self):
        summary = build_validation_summary(self.root)

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["realPaperRun"]["paperCount"], 2)
        self.assertEqual(summary["realPaperRun"]["evaluationPassed"], 4)
        self.assertEqual(summary["realPaperRun"]["evaluationTotal"], 4)
        self.assertEqual(summary["realPaperRun"]["fineTuningRecommendation"], "no")
        self.assertEqual(summary["modelTraces"]["total"], 4)
        self.assertEqual(summary["modelTraces"]["fallbackCount"], 0)
        self.assertEqual(summary["modelTraces"]["byTask"]["grounded_qa"], 1)
        self.assertEqual(summary["memory"]["recordCount"], 3)
        self.assertEqual(summary["localDemo"]["selectedSpanId"], "P3.S9")
        self.assertEqual(summary["localDemo"]["evidenceWindow"], "P3.S6-P3.S12")
        self.assertEqual(summary["localDemo"]["sourceHash"], "matching-source-hash")
        self.assertEqual(summary["localDemo"]["sourceIndexHash"], "matching-source-hash")
        self.assertTrue(summary["localDemo"]["sourceIndexConsistent"])
        self.assertFalse(summary["localDemo"]["usedFallback"])

    def test_source_index_mismatch_marks_validation_not_ok(self):
        (self.day / "source_index" / "paper-a.json").write_text(
            json.dumps(
                {
                    "paper_id": "paper:a",
                    "source_text_hash": "stale-source-hash",
                    "source_text_chars": 32005,
                    "spans": [],
                }
            ),
            encoding="utf-8",
        )

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["localDemo"]["sourceIndexConsistent"])
        self.assertIn("source hash differs", " ".join(summary["warnings"]))

    def test_validation_endpoint_returns_summary(self):
        client = TestClient(create_app())
        response = client.get("/api/validation")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["modelTraces"]["modelCount"], 4)
        self.assertEqual(body["localDemo"]["translationStatus"], "ready")

    def _write_validation_tree(self):
        summary = {
            "passed": True,
            "paper_count": 2,
            "fine_tuning": {
                "recommendation": "no",
                "reason": "No repeated real model-output failures.",
                "repeated_failures": [],
            },
            "runs": [
                self._paper_run("attention_is_all_you_need", "1706.03762", "Attention Is All You Need", 19360),
                self._paper_run(
                    "retrieval_augmented_generation",
                    "2005.11401",
                    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    25741,
                ),
            ],
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        trace_records = [
            {"task": "translation", "status": "model", "provider": "hf", "model": "test-small", "error": None},
            {"task": "grounded_qa", "status": "model", "provider": "hf", "model": "test-small", "error": None},
            {"task": "experiment_spec", "status": "model", "provider": "hf", "model": "test-small", "error": None},
            {"task": "research_growth", "status": "model", "provider": "hf", "model": "test-small", "error": None},
        ]
        self._write_jsonl(self.day / "hf_three_papers_rerun_traces.jsonl", trace_records)
        self._write_jsonl(
            self.day / "hf_three_papers_rerun_memory.jsonl",
            [
                {"paper_id": "paper:a", "kind": "paper_span"},
                {"paper_id": "paper:a", "kind": "growth_idea"},
                {"paper_id": "paper:b", "kind": "mini_lab_result"},
            ],
        )
        (self.day / "local_after_source_index_ask_p3s9.json").write_text(
            json.dumps(
                {
                    "confidence": "high",
                    "provider": "hf",
                    "model": "test-small",
                    "usedFallback": False,
                    "evidence": [{"source_id": "P3.S9", "quote": "In this work we propose the Transformer"}],
                    "evidenceWindow": {
                        "paperId": "paper:a",
                        "spanId": "P3.S9",
                        "spanRange": "P3.S6-P3.S12",
                        "sourceHash": "matching-source-hash",
                    },
                }
            ),
            encoding="utf-8",
        )
        (self.day / "local_after_source_index_translate_p3s9.json").write_text(
            json.dumps({"status": "ready", "usedFallback": False}),
            encoding="utf-8",
        )
        (self.day / "local_after_source_index_paper.json").write_text(
            json.dumps(
                {
                    "title": "Attention Is All You Need",
                    "metadata": {"readerSpanCount": 180, "sourceTextChars": 32005},
                }
            ),
            encoding="utf-8",
        )
        source_index_dir = self.day / "source_index"
        source_index_dir.mkdir()
        (source_index_dir / "paper-a.json").write_text(
            json.dumps(
                {
                    "paper_id": "paper:a",
                    "source_text_hash": "matching-source-hash",
                    "source_text_chars": 32005,
                    "spans": [
                        {"span_id": "P3.S9", "position": 99, "text_hash": "span-hash"},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _paper_run(self, name, arxiv, title, source_chars):
        return {
            "case": {"name": name, "arxiv": arxiv},
            "source": {
                "title": title,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv}",
                "page_marker_count": 6,
                "text_chars": source_chars,
            },
            "reader": {
                "visible_span_count": 180,
                "selected_span_positions": [{"span_id": "P3.S9", "position_label": "middle"}],
            },
            "memory": {"records_after_growth": 4},
            "evaluations": [
                {"name": "pdf_parse_and_reader_spans", "passed": True, "reasons": []},
                {"name": "grounded_qa", "passed": True, "reasons": []},
            ],
        }

    def _write_jsonl(self, path, records):
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
