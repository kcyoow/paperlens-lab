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
        self.assertEqual(summary["realPaperRun"]["paperCount"], 3)
        self.assertEqual(summary["realPaperRun"]["evaluationPassed"], 12)
        self.assertEqual(summary["realPaperRun"]["evaluationTotal"], 12)
        self.assertTrue(summary["realPaperRun"]["evidenceConsistencyPassed"])
        self.assertTrue(summary["realPaperRun"]["growthIterationPassed"])
        self.assertEqual(summary["realPaperRun"]["fineTuningRecommendation"], "no")
        self.assertEqual(summary["modelTraces"]["total"], 6)
        self.assertEqual(summary["modelTraces"]["fallbackCount"], 0)
        self.assertEqual(summary["modelTraces"]["byTask"]["grounded_qa"], 1)
        self.assertEqual(summary["modelTraces"]["byTask"]["adversarial_grounded_qa"], 1)
        self.assertEqual(summary["modelTraces"]["byTask"]["research_growth"], 2)
        self.assertEqual(summary["memory"]["recordCount"], 3)
        self.assertEqual(summary["localDemo"]["selectedSpanId"], "P3.S9")
        self.assertEqual(summary["localDemo"]["evidenceWindow"], "P3.S6-P3.S12")
        self.assertEqual(summary["localDemo"]["sourceHash"], "matching-source-hash")
        self.assertEqual(summary["localDemo"]["sourceIndexHash"], "matching-source-hash")
        self.assertTrue(summary["localDemo"]["sourceIndexConsistent"])
        self.assertTrue(summary["localDemo"]["quoteIdsWithinWindow"])
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

    def test_unknown_local_evidence_id_marks_validation_not_ok(self):
        ask_path = self.day / "local_after_source_index_ask_p3s9.json"
        body = json.loads(ask_path.read_text(encoding="utf-8"))
        body["evidence"].append({"source_id": "S3", "quote": "A generated evidence id."})
        ask_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["localDemo"]["quoteIdsWithinWindow"])
        self.assertEqual(summary["localDemo"]["unknownEvidenceIds"], ["S3"])

    def test_split_growth_iteration_evidence_marks_validation_not_ok(self):
        summary_path = self.run_dir / "summary.json"
        body = json.loads(summary_path.read_text(encoding="utf-8"))
        body["runs"][0]["model_outputs"]["growth_iteration"]["data"]["ideas"] = [
            {"source_evidence": ["growth_idea:test"]},
            {"source_evidence": ["paper:selected-middle", "run:r1"]},
        ]
        summary_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["realPaperRun"]["growthIterationPassed"])

    def test_stale_summary_without_source_evidence_is_not_green(self):
        stale_dir = self.day / "hf_three_papers_stale"
        stale_dir.mkdir()
        stale_summary = {
            "passed": True,
            "paper_count": 3,
            "fine_tuning": {"recommendation": "no", "reason": "", "repeated_failures": []},
            "runs": [
                {
                    **self._paper_run("stale", "9999.99999", "Stale Paper", 12000),
                    "model_outputs": {
                        "qa": [
                            {
                                "span": {"id": "P0.S1"},
                                "result": {
                                    "data": {
                                        "evidence": [{"source_id": "S9", "quote": "not bound"}],
                                    }
                                },
                            }
                        ]
                    },
                }
            ],
        }
        (stale_dir / "summary.json").write_text(json.dumps(stale_summary), encoding="utf-8")
        self._write_jsonl(
            self.day / "hf_three_papers_stale_traces.jsonl",
            [
                {"task": "grounded_qa", "status": "model", "provider": "hf", "model": "test-small", "error": None},
            ],
        )

        summary = build_validation_summary(self.root)

        self.assertEqual(summary["realPaperRun"]["runName"], "hf_three_papers_stale")
        self.assertFalse(summary["ok"])
        self.assertFalse(summary["realPaperRun"]["evidenceConsistencyPassed"])

    def test_validation_endpoint_returns_summary(self):
        client = TestClient(create_app())
        response = client.get("/api/validation")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["modelTraces"]["modelCount"], 6)
        self.assertEqual(body["localDemo"]["translationStatus"], "ready")

    def _write_validation_tree(self):
        summary = {
            "passed": True,
            "paper_count": 3,
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
                self._paper_run("lora", "2106.09685", "LoRA", 21000),
            ],
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        trace_records = [
            {"task": "translation", "status": "model", "provider": "hf", "model": "test-small", "error": None},
            {"task": "grounded_qa", "status": "model", "provider": "hf", "model": "test-small", "error": None},
            {"task": "adversarial_grounded_qa", "status": "model", "provider": "hf", "model": "test-small", "error": None},
            {"task": "experiment_spec", "status": "model", "provider": "hf", "model": "test-small", "error": None},
            {"task": "research_growth", "status": "model", "provider": "hf", "model": "test-small", "error": None},
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
                        "spans": [{"spanId": "P3.S9", "textHash": "span-hash", "position": 99}],
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
                "adversarial_litm": {
                    "context_span_count": 80,
                    "context_chars": 9000,
                    "target_span_id": "P3.S9",
                    "target_char_offset_ratio": 0.5,
                    "distractor_count": 79,
                },
            },
            "memory": {"records_after_growth": 4},
            "evaluations": [
                {"name": "pdf_parse_and_reader_spans", "passed": True, "reasons": []},
                {"name": "grounded_qa", "passed": True, "reasons": []},
                {"name": "adversarial_lost_in_the_middle", "passed": True, "reasons": []},
                {"name": "research_growth_iteration", "passed": True, "reasons": []},
            ],
            "model_outputs": {
                "qa": [
                    {
                        "span": {"id": "P3.S9"},
                        "source_evidence": {"P3.S9": "In this work we propose the Transformer"},
                        "result": {
                            "data": {
                                "evidence": [
                                    {
                                        "source_id": "P3.S9",
                                        "quote": "In this work we propose the Transformer",
                                    }
                                ]
                            }
                        },
                    }
                ],
                "growth_iteration": {
                    "data": {
                        "ideas": [
                            {
                                "source_evidence": [
                                    "paper:selected-middle",
                                    "run:r1",
                                    "growth_idea:test",
                                ]
                            }
                        ]
                    }
                },
            },
        }

    def _write_jsonl(self, path, records):
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
