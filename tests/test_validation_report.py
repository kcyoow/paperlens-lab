import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import paperlens_lab.validation_report as validation_report
from paperlens_lab.server import create_app
from paperlens_lab.validation_report import build_validation_summary


class ValidationReportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.day = self.root / "2026-06-13"
        self.run_dir = self.day / "hf_three_papers_rerun"
        self.runtime_source_index_dir = self.root / "runtime_source_index"
        self.frontend_out_dir = self.root / "frontend" / "out"
        self.run_dir.mkdir(parents=True)
        self._write_frontend_static_export()
        self.frontend_out_patch = patch.object(validation_report, "FRONTEND_OUT_DIR", self.frontend_out_dir)
        self.frontend_out_patch.start()
        self._write_validation_tree()
        os.environ["PAPERLENS_VALIDATION_ROOT"] = str(self.root)
        os.environ["PAPERLENS_TRACE_PATH"] = str(self.root / "api_traces.jsonl")
        os.environ["PAPERLENS_MEMORY_PATH"] = str(self.root / "paper_memory.jsonl")
        os.environ["PAPERLENS_SOURCE_INDEX_DIR"] = str(self.runtime_source_index_dir)
        os.environ["PAPERLENS_PROVIDER"] = "hf"
        os.environ["PAPERLENS_MODEL"] = "test-small"
        os.environ["PAPERLENS_TRANSLATION_MODEL"] = "test-small"
        os.environ["PAPERLENS_QUALITY_MODEL"] = "test-quality"

    def tearDown(self):
        os.environ.pop("PAPERLENS_VALIDATION_ROOT", None)
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
        os.environ.pop("PAPERLENS_MEMORY_PATH", None)
        os.environ.pop("PAPERLENS_SOURCE_INDEX_DIR", None)
        os.environ.pop("PAPERLENS_PROVIDER", None)
        os.environ.pop("PAPERLENS_MODEL", None)
        os.environ.pop("PAPERLENS_TRANSLATION_MODEL", None)
        os.environ.pop("PAPERLENS_QUALITY_MODEL", None)
        self.frontend_out_patch.stop()
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
        self.assertEqual(summary["modelTraces"]["total"], 21)
        self.assertEqual(summary["modelTraces"]["fallbackCount"], 0)
        self.assertTrue(summary["modelTraces"]["traceIdsPassed"])
        self.assertTrue(summary["modelTraces"]["currentContractMatched"])
        self.assertEqual(summary["modelTraces"]["requiredTraceIdCount"], 21)
        self.assertEqual(summary["modelTraces"]["byTask"]["grounded_qa"], 3)
        self.assertEqual(summary["modelTraces"]["byTask"]["adversarial_grounded_qa"], 3)
        self.assertEqual(summary["modelTraces"]["byTask"]["starter_code"], 3)
        self.assertEqual(summary["modelTraces"]["byTask"]["research_growth"], 6)
        self.assertEqual(summary["memory"]["recordCount"], 3)
        self.assertEqual(summary["localDemo"]["selectedSpanId"], "P3.S9")
        self.assertEqual(summary["localDemo"]["evidenceWindow"], "P3.S6-P3.S12")
        self.assertEqual(summary["localDemo"]["sourceHash"], "matching-source-hash")
        self.assertEqual(summary["localDemo"]["sourceIndexHash"], "matching-source-hash")
        self.assertEqual(
            summary["localDemo"]["sourceIndexPath"],
            str(self.runtime_source_index_dir / "paper-a.json"),
        )
        self.assertTrue(summary["localDemo"]["sourceIndexRuntimeBound"])
        self.assertTrue(summary["localDemo"]["sourceIndexConsistent"])
        self.assertTrue(summary["localDemo"]["quoteIdsWithinWindow"])
        self.assertTrue(summary["localDemo"]["quotesInSourceIndex"])
        self.assertTrue(summary["localDemo"]["translationSourceConsistent"])
        self.assertEqual(summary["localDemo"]["translationSourceHash"], "e6c340c1ebb19a22")
        self.assertTrue(summary["localDemo"]["traceIdsPassed"])
        self.assertTrue(summary["localDemo"]["currentContractMatched"])
        self.assertFalse(summary["localDemo"]["usedFallback"])
        self.assertTrue(summary["frontendStaticExport"]["ready"])
        self.assertTrue(summary["frontendStaticExport"]["hasIndex"])
        self.assertTrue(summary["frontendStaticExport"]["hasReader"])
        self.assertTrue(summary["frontendStaticExport"]["hasNextStatic"])
        self.assertTrue(summary["frontendStaticExport"]["hasReaderChunk"])
        self.assertGreater(summary["frontendStaticExport"]["fileCount"], 2)

    def test_unrelated_old_fallback_traces_do_not_taint_current_real_paper_run(self):
        trace_path = self.day / "hf_three_papers_rerun_traces.jsonl"
        trace_records = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        trace_records.append(
            {
                "trace_id": "old-qwen-fallback",
                "task": "translation",
                "status": "fallback",
                "provider": "fallback",
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "error": None,
            }
        )
        self._write_jsonl(trace_path, trace_records)

        summary = build_validation_summary(self.root)

        self.assertTrue(summary["ok"])
        self.assertGreater(summary["modelTraces"]["scannedTraceRecordCount"], summary["modelTraces"]["total"])
        self.assertEqual(summary["modelTraces"]["total"], 21)
        self.assertEqual(summary["modelTraces"]["fallbackCount"], 0)
        self.assertTrue(summary["modelTraces"]["traceIdsPassed"])

    def test_stale_model_contract_marks_validation_not_ok(self):
        os.environ["PAPERLENS_MODEL"] = "new-current-model"

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["modelTraces"]["currentContractMatched"])
        self.assertIn("model mismatch", " ".join(summary["modelTraces"]["currentContractIssues"]))

    def test_missing_frontend_static_export_marks_validation_not_ok(self):
        shutil.rmtree(self.frontend_out_dir)

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["frontendStaticExport"]["ready"])
        self.assertIn("frontend static export", " ".join(summary["warnings"]))

    def test_incomplete_frontend_static_export_marks_validation_not_ok(self):
        (self.frontend_out_dir / "reader" / "index.html").unlink()

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["frontendStaticExport"]["ready"])
        self.assertFalse(summary["frontendStaticExport"]["hasReader"])
        self.assertIn("reader/index.html", " ".join(summary["frontendStaticExport"]["issues"]))

    def test_static_export_without_reader_chunk_marks_validation_not_ok(self):
        for path in (self.frontend_out_dir / "_next" / "static" / "chunks" / "app" / "reader").glob("page-*.js"):
            path.unlink()

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["frontendStaticExport"]["ready"])
        self.assertFalse(summary["frontendStaticExport"]["hasReaderChunk"])
        self.assertIn("reader page chunk", " ".join(summary["frontendStaticExport"]["issues"]))

    def test_source_index_mismatch_marks_validation_not_ok(self):
        (self.runtime_source_index_dir / "paper-a.json").write_text(
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

    def test_bundle_source_index_can_prove_local_demo_without_runtime_binding(self):
        (self.runtime_source_index_dir / "paper-a.json").unlink()

        summary = build_validation_summary(self.root)

        self.assertTrue(summary["ok"])
        self.assertTrue(summary["localDemo"]["sourceIndexRuntimeBound"])
        self.assertTrue(summary["localDemo"]["sourceIndexConsistent"])
        self.assertEqual(
            summary["localDemo"]["sourceIndexPath"],
            str(self.day / "source_index" / "paper-a.json"),
        )

    def test_newer_incoherent_local_bundle_does_not_override_coherent_bundle(self):
        newer_day = self.root / "2026-06-14"
        newer_day.mkdir()
        (newer_day / "local_after_source_index_ask_p8s1.json").write_text(
            json.dumps(
                {
                    "traceId": "orphan-ask-trace",
                    "evidenceWindow": {"paperId": "paper:a", "spanId": "P8.S1"},
                    "evidence": [],
                }
            ),
            encoding="utf-8",
        )

        summary = build_validation_summary(self.root)

        self.assertTrue(summary["localDemo"]["artifactBundleCoherent"])
        self.assertEqual(summary["localDemo"]["selectedSpanId"], "P3.S9")
        self.assertEqual(summary["localDemo"]["askPath"], str(self.day / "local_after_source_index_ask_p3s9.json"))

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

    def test_local_translation_hash_mismatch_marks_validation_not_ok(self):
        translate_path = self.day / "local_after_source_index_translate_p3s9.json"
        body = json.loads(translate_path.read_text(encoding="utf-8"))
        body["sourceHash"] = "stale-translation-source"
        translate_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["localDemo"]["translationSourceConsistent"])
        self.assertIn("translation source hash", " ".join(summary["warnings"]))

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

    def test_missing_starter_code_source_run_marks_validation_not_ok(self):
        summary_path = self.run_dir / "summary.json"
        body = json.loads(summary_path.read_text(encoding="utf-8"))
        for run in body["runs"]:
            run["evaluations"] = [
                item for item in run["evaluations"] if item["name"] != "starter_code_source_run"
            ]
        summary_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["realPaperRun"]["starterCodePassed"])

    def test_passed_starter_eval_with_broken_code_marks_validation_not_ok(self):
        summary_path = self.run_dir / "summary.json"
        body = json.loads(summary_path.read_text(encoding="utf-8"))
        body["runs"][0]["model_outputs"]["starter_code"]["data"]["code"] = "def run(:\n"
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
            "dataset": {"name": "WMT14", "fallback": "indexed paper-evidence reduction"},
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

    def test_missing_starter_trace_id_marks_validation_not_ok(self):
        summary_path = self.run_dir / "summary.json"
        body = json.loads(summary_path.read_text(encoding="utf-8"))
        body["runs"][0]["model_outputs"]["starter_code"]["trace_id"] = "missing-starter-trace"
        summary_path.write_text(json.dumps(body), encoding="utf-8")

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["modelTraces"]["traceIdsPassed"])
        self.assertIn("missing-starter-trace", " ".join(summary["modelTraces"]["traceIdIssues"]))

    def test_missing_local_trace_record_marks_validation_not_ok(self):
        trace_path = Path(os.environ["PAPERLENS_TRACE_PATH"])
        traces = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        traces = [record for record in traces if record["trace_id"] != "local-translate-trace"]
        self._write_jsonl(trace_path, traces)

        summary = build_validation_summary(self.root)

        self.assertFalse(summary["ok"])
        self.assertFalse(summary["localDemo"]["traceIdsPassed"])
        self.assertIn("local-translate-trace", " ".join(summary["localDemo"]["traceIdIssues"]))

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
        self.assertEqual(body["modelTraces"]["modelCount"], 21)
        self.assertEqual(body["localDemo"]["translationStatus"], "ready")
        self.assertTrue(body["frontendStaticExport"]["ready"])

    def _write_frontend_static_export(self):
        (self.frontend_out_dir / "reader").mkdir(parents=True, exist_ok=True)
        (self.frontend_out_dir / "_next" / "static" / "chunks").mkdir(parents=True, exist_ok=True)
        (self.frontend_out_dir / "_next" / "static" / "chunks" / "app" / "reader").mkdir(
            parents=True,
            exist_ok=True,
        )
        (self.frontend_out_dir / "index.html").write_text("<main>PaperLens Lab</main>", encoding="utf-8")
        (self.frontend_out_dir / "reader" / "index.html").write_text("<main>Reader</main>", encoding="utf-8")
        (self.frontend_out_dir / "_next" / "static" / "chunks" / "app.js").write_text(
            "self.__next_f=[];",
            encoding="utf-8",
        )
        (self.frontend_out_dir / "_next" / "static" / "chunks" / "app" / "reader" / "page-test.js").write_text(
            "self.__next_reader=[];",
            encoding="utf-8",
        )

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
            self.root / "api_traces.jsonl",
            [
                self._trace_record("local-qa-trace", "grounded_qa", model="test-small"),
                self._trace_record("local-translate-trace", "translation", model="test-small"),
            ],
        )
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
                    "traceId": "local-qa-trace",
                    "usedFallback": False,
                    "evidence": [{"source_id": "P3.S9", "quote": "In this work we propose the Transformer"}],
                    "evidenceWindow": {
                        "paperId": "paper:a",
                        "spanId": "P3.S9",
                        "spanRange": "P3.S6-P3.S12",
                        "sourceHash": "matching-source-hash",
                        "spans": [{"spanId": "P3.S9", "textHash": "e6c340c1ebb19a22", "position": 99}],
                    },
                }
            ),
            encoding="utf-8",
        )
        (self.day / "local_after_source_index_translate_p3s9.json").write_text(
            json.dumps(
                {
                    "status": "ready",
                    "traceId": "local-translate-trace",
                    "usedFallback": False,
                    "sourceHash": "e6c340c1ebb19a22",
                    "sourceIndexBound": True,
                }
            ),
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
        for source_index_dir in (self.day / "source_index", self.runtime_source_index_dir):
            source_index_dir.mkdir(parents=True, exist_ok=True)
            (source_index_dir / "paper-a.json").write_text(
                json.dumps(
                    {
                        "paper_id": "paper:a",
                        "source_text_hash": "matching-source-hash",
                        "source_text_chars": 32005,
                        "spans": [
                            {
                                "span_id": "P3.S8",
                                "position": 98,
                                "text_hash": "control-source-hash",
                                "text": "This control source span describes a control condition",
                            },
                            {
                                "span_id": "P3.S9",
                                "position": 99,
                                "text_hash": "e6c340c1ebb19a22",
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
                "document_id": "paper-a",
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
                {"name": "starter_code_source_run", "passed": True, "reasons": []},
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
                        "research_question": "Can this idea help on the indexed paper evidence?",
                        "mini_lab_goal": "Run a dependency-free comparison over indexed evidence rows.",
                        "dataset": {"name": "Indexed PaperLens evidence", "source": "source-index rows"},
                        "baseline": "Direct keyword baseline.",
                        "metric": "source evidence score",
                        "steps": ["Build examples", "Run baseline", "Run variant"],
                        "ablation": "Remove only the paper-inspired heuristic.",
                        "failure_condition": "source evidence score does not improve.",
                    },
                },
                "starter_code": {
                    "task": "starter_code",
                    "trace_id": f"{name}-starter",
                    "provider": "hf",
                    "model": "test-quality",
                    "used_fallback": False,
                    "error": None,
                    "data": {
                        "code": self._starter_code(),
                        "why_this_matches_span": "A span-grounded candidate comparison over indexed evidence.",
                        "limitations": ["Directional source-evidence probe only."],
                    },
                },
                "growth": {
                    "task": "research_growth",
                    "trace_id": f"{name}-growth",
                    "provider": "hf",
                    "model": "test-quality",
                    "used_fallback": False,
                    "error": None,
                    "data": {"ideas": [{"source_evidence": ["paper:selected-middle", "run:r1"]}]},
                },
                "growth_iteration": {
                    "task": "research_growth",
                    "trace_id": f"{name}-growth-iteration",
                    "provider": "hf",
                    "model": "test-quality",
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
            "starter_code": "starter_code",
            "growth": "research_growth",
            "growth_iteration": "research_growth",
        }
        for run in summary["runs"]:
            outputs = run["model_outputs"]
            for key, task in task_by_key.items():
                records.append(self._trace_record(outputs[key]["trace_id"], task, model=outputs[key].get("model", "test-small")))
            for qa in outputs["qa"]:
                records.append(self._trace_record(qa["result"]["trace_id"], "grounded_qa", model=qa["result"].get("model", "test-small")))
            records.append(
                self._trace_record(
                    outputs["adversarial_litm"]["result"]["trace_id"],
                    "adversarial_grounded_qa",
                    model=outputs["adversarial_litm"]["result"].get("model", "test-small"),
                )
            )
        return records

    def _trace_record(self, trace_id, task, model="test-small"):
        return {
            "trace_id": trace_id,
            "task": task,
            "status": "model",
            "provider": "hf",
            "model": model,
            "error": None,
        }

    def _starter_code(self):
        return """def baseline(example):
    return "" if example.get("gold") else example.get("text", "")

def paper_inspired(example):
    return example.get("text", "") + " Transformer"

def score(output, expected):
    return 1.0 if expected in output else 0.0

def run(evidence_rows=None):
    examples = [
        {
            **row,
            "expected": "Transformer" if row.get("gold") else "control",
        }
        for row in (evidence_rows or [])
    ]
    rows = []
    for example in examples:
        base = baseline(example)
        variant = paper_inspired(example)
        baseline_score = score(base, example["expected"])
        prototype_score = score(variant, example["expected"])
        rows.append({
            "source_id": example["source_id"],
            "text_hash": example["text_hash"],
            "baseline_score": baseline_score,
            "prototype_score": prototype_score,
            "metric": "source evidence score",
            "failure_condition": prototype_score <= baseline_score,
            "failure_rule": "prototype_score <= baseline_score",
        })
    return rows
"""

    def _write_jsonl(self, path, records):
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
