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
        self.assertEqual(summary["realPaperRun"]["evaluationPassed"], 30)
        self.assertEqual(summary["realPaperRun"]["evaluationTotal"], 30)
        self.assertTrue(summary["realPaperRun"]["evidenceConsistencyPassed"])
        self.assertTrue(summary["realPaperRun"]["artifactContractPassed"])
        self.assertTrue(summary["realPaperRun"]["growthIterationPassed"])
        self.assertTrue(summary["realPaperRun"]["starterCodePassed"])
        self.assertEqual(summary["realPaperRun"]["fineTuningRecommendation"], "no")
        self.assertEqual(summary["modelTraces"]["total"], 18)
        self.assertEqual(summary["modelTraces"]["fallbackCount"], 0)
        self.assertTrue(summary["modelTraces"]["traceIdsPassed"])
        self.assertEqual(summary["modelTraces"]["requiredTraceIdCount"], 18)
        self.assertEqual(summary["modelTraces"]["byTask"]["grounded_qa"], 3)
        self.assertEqual(summary["modelTraces"]["byTask"]["adversarial_grounded_qa"], 3)
        self.assertEqual(summary["modelTraces"]["byTask"]["research_growth"], 6)
        self.assertEqual(summary["memory"]["recordCount"], 3)
        self.assertEqual(summary["localDemo"]["selectedSpanId"], "P3.S9")
        self.assertEqual(summary["localDemo"]["evidenceWindow"], "P3.S6-P3.S12")
        self.assertEqual(summary["localDemo"]["sourceHash"], "matching-source-hash")
        self.assertEqual(summary["localDemo"]["sourceIndexHash"], "matching-source-hash")
        self.assertTrue(summary["localDemo"]["sourceIndexConsistent"])
        self.assertTrue(summary["localDemo"]["quoteIdsWithinWindow"])
        self.assertTrue(summary["localDemo"]["quotesInSourceIndex"])
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

    def test_local_quote_missing_from_source_index_marks_validation_not_ok(self):
        ask_path = self.day / "local_after_source_index_ask_p3s9.json"
        body = json.loads(ask_path.read_text(encoding="utf-8"))
        body["evidence"] = [{"source_id": "P3.S9", "quote": "This quote is not in the indexed source span."}]
        ask_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["localDemo"]["quotesInSourceIndex"])
        self.assertEqual(summary["localDemo"]["badQuoteIds"], ["P3.S9"])

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

    def test_missing_starter_code_smoke_marks_validation_not_ok(self):
        summary_path = self.run_dir / "summary.json"
        body = json.loads(summary_path.read_text(encoding="utf-8"))
        for run in body["runs"]:
            run["evaluations"] = [
                item for item in run["evaluations"] if item["name"] != "starter_code_smoke"
            ]
        summary_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["realPaperRun"]["starterCodePassed"])

    def test_passed_starter_eval_with_broken_code_marks_validation_not_ok(self):
        summary_path = self.run_dir / "summary.json"
        body = json.loads(summary_path.read_text(encoding="utf-8"))
        body["runs"][0]["model_outputs"]["starter_code"]["code"] = "def run(:\n"
        summary_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["realPaperRun"]["starterCodePassed"])
        self.assertFalse(summary["realPaperRun"]["papers"][0]["starterCodeRechecked"])

    def test_missing_required_eval_marks_validation_not_ok(self):
        summary_path = self.run_dir / "summary.json"
        body = json.loads(summary_path.read_text(encoding="utf-8"))
        body["runs"][0]["evaluations"] = [
            item for item in body["runs"][0]["evaluations"] if item["name"] != "translation_fidelity"
        ]
        summary_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["realPaperRun"]["artifactContractPassed"])
        self.assertIn("translation_fidelity", summary["realPaperRun"]["papers"][0]["requiredEvalMissing"])

    def test_passed_experiment_eval_with_heavy_spec_marks_validation_not_ok(self):
        summary_path = self.run_dir / "summary.json"
        body = json.loads(summary_path.read_text(encoding="utf-8"))
        body["runs"][0]["model_outputs"]["experiment"]["data"] = {
            "research_question": "Can CUDA P100 full training reproduce WMT14?",
            "mini_lab_goal": "Run multi-day full training.",
            "dataset": {"name": "WMT14", "fallback": "toy examples"},
            "baseline": "PyTorch model",
            "metric": "BLEU",
            "steps": ["Provision CUDA", "Train for 100 epochs", "Evaluate"],
            "failure_condition": "BLEU is lower.",
        }
        summary_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["realPaperRun"]["artifactContractPassed"])
        self.assertFalse(summary["realPaperRun"]["papers"][0]["experimentSpecRechecked"])

    def test_missing_summary_trace_id_marks_validation_not_ok(self):
        summary_path = self.run_dir / "summary.json"
        body = json.loads(summary_path.read_text(encoding="utf-8"))
        body["runs"][0]["model_outputs"]["experiment"]["trace_id"] = "missing-trace-id"
        summary_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["modelTraces"]["traceIdsPassed"])
        self.assertIn("missing-trace-id", " ".join(summary["modelTraces"]["traceIdIssues"]))

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
        self.assertEqual(body["modelTraces"]["modelCount"], 18)
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

        trace_records = self._trace_records_from_summary(summary)
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
                        {
                            "span_id": "P3.S9",
                            "position": 99,
                            "text_hash": "span-hash",
                            "text": "In this work we propose the Transformer",
                        },
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
                {"name": "translation_fidelity", "passed": True, "reasons": []},
                {"name": "grounded_qa", "passed": True, "reasons": []},
                {"name": "middle_selected_span_grounding", "passed": True, "reasons": []},
                {"name": "adversarial_lost_in_the_middle", "passed": True, "reasons": []},
                {"name": "experiment_spec", "passed": True, "reasons": []},
                {"name": "starter_code_smoke", "passed": True, "reasons": []},
                {"name": "growth_ideas", "passed": True, "reasons": []},
                {"name": "research_growth_iteration", "passed": True, "reasons": []},
                {"name": "model_backing", "passed": True, "reasons": []},
            ],
            "model_outputs": {
                "translation": {
                    "task": "translation",
                    "trace_id": f"{name}-translation",
                    "provider": "hf",
                    "model": "test-small",
                    "used_fallback": False,
                    "error": None,
                    "data": {"translations": [{"span_id": "P3.S9", "translation": "번역"}]},
                },
                "qa": [
                    {
                        "span": {"id": "P3.S9"},
                        "source_evidence": {"P3.S9": "In this work we propose the Transformer"},
                        "result": {
                            "task": "grounded_qa",
                            "trace_id": f"{name}-qa",
                            "provider": "hf",
                            "model": "test-small",
                            "used_fallback": False,
                            "error": None,
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
                "adversarial_litm": {
                    "result": {
                        "task": "adversarial_grounded_qa",
                        "trace_id": f"{name}-adversarial",
                        "provider": "hf",
                        "model": "test-small",
                        "used_fallback": False,
                        "error": None,
                        "data": {
                            "evidence": [
                                {
                                    "source_id": "P3.S9",
                                    "quote": "In this work we propose the Transformer",
                                }
                            ]
                        },
                    },
                    "source_evidence": {"P3.S9": "In this work we propose the Transformer"},
                    "stats": {
                        "context_span_count": 80,
                        "context_chars": 9000,
                        "target_span_id": "P3.S9",
                        "target_char_offset_ratio": 0.5,
                        "distractor_count": 79,
                    },
                },
                "experiment": {
                    "task": "experiment_spec",
                    "trace_id": f"{name}-experiment",
                    "provider": "hf",
                    "model": "test-small",
                    "used_fallback": False,
                    "error": None,
                    "data": {
                        "research_question": "Can this idea help on a toy task?",
                        "mini_lab_goal": "Run a small dependency-free comparison.",
                        "dataset": {"name": "Toy examples", "fallback": "10 hand-built examples"},
                        "baseline": "Direct keyword baseline.",
                        "metric": "toy score",
                        "steps": ["Build examples", "Run baseline", "Run variant"],
                        "ablation": "Remove only the paper-inspired heuristic.",
                        "failure_condition": "toy score does not improve.",
                    },
                },
                "starter_code": {"task": "starter_code", "code": self._starter_code()},
                "growth": {
                    "task": "research_growth",
                    "trace_id": f"{name}-growth",
                    "provider": "hf",
                    "model": "test-small",
                    "used_fallback": False,
                    "error": None,
                    "data": {"ideas": [{"source_evidence": ["paper:selected-middle", "run:r1"]}]},
                },
                "growth_iteration": {
                    "task": "research_growth",
                    "trace_id": f"{name}-growth-iteration",
                    "provider": "hf",
                    "model": "test-small",
                    "used_fallback": False,
                    "error": None,
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

    def _trace_records_from_summary(self, summary):
        records = []
        task_by_key = {
            "translation": "translation",
            "experiment": "experiment_spec",
            "growth": "research_growth",
            "growth_iteration": "research_growth",
        }
        for run in summary["runs"]:
            outputs = run["model_outputs"]
            for key, task in task_by_key.items():
                records.append(self._trace_record(outputs[key]["trace_id"], task))
            for qa in outputs["qa"]:
                records.append(self._trace_record(qa["result"]["trace_id"], "grounded_qa"))
            records.append(
                self._trace_record(
                    outputs["adversarial_litm"]["result"]["trace_id"],
                    "adversarial_grounded_qa",
                )
            )
        return records

    def _trace_record(self, trace_id, task):
        return {
            "trace_id": trace_id,
            "task": task,
            "status": "model",
            "provider": "hf",
            "model": "test-small",
            "error": None,
        }

    def _starter_code(self):
        return """def baseline(example):
    return example.get("text", "")

def paper_inspired(example):
    return example.get("text", "") + " Transformer"

def score(output, expected):
    return 1.0 if expected in output else 0.0

def run():
    example = {"text": "In this work we propose the", "expected": "Transformer"}
    base = baseline(example)
    variant = paper_inspired(example)
    return [{
        "baseline_score": score(base, example["expected"]),
        "prototype_score": score(variant, example["expected"]),
        "metric": "toy score",
        "failure_condition": "prototype_score <= baseline_score",
    }]
"""

    def _write_jsonl(self, path, records):
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
