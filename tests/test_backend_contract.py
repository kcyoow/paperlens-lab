import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from paperlens_lab.gpu_lab import _validate_gpu_script_contract, _validated_gpu_result
from paperlens_lab.ingest import PaperSource
from paperlens_lab.mini_lab import code_hash
from paperlens_lab.scenario_eval import evaluate_starter_code
from paperlens_lab.server import (
    _CANDIDATE_SETS,
    _EXPERIMENT_RUNS,
    _GPU_PROBE_RUNS,
    _issue_experiment_run,
    create_app,
    paper_document_from_source,
)
from paperlens_lab.source_index import load_source_index, text_hash


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["PAPERLENS_TRACE_PATH"] = str(Path(self.tempdir.name) / "api_traces.jsonl")
        os.environ["PAPERLENS_MEMORY_PATH"] = str(Path(self.tempdir.name) / "paper_memory.jsonl")
        os.environ["PAPERLENS_SOURCE_INDEX_DIR"] = str(Path(self.tempdir.name) / "source_index")
        os.environ["PAPERLENS_TRANSLATION_CACHE_DIR"] = str(Path(self.tempdir.name) / "translation_cache")
        os.environ["PAPERLENS_ENABLE_DIAGNOSTIC_STARTER"] = "1"
        _EXPERIMENT_RUNS.clear()
        _CANDIDATE_SETS.clear()
        _GPU_PROBE_RUNS.clear()
        self.client = TestClient(create_app())

    def indexed_starter_payload(self, code: str, *, selected_fragment: str | None = None) -> dict:
        source = PaperSource(
            title="Source Bound Starter Paper",
            authors="A. Author",
            source_label="unit-source-bound",
            text=(
                "First sentence introduces a compact evidence mechanism. "
                "Second sentence says the source evidence mechanism improves precision in retrieved citations. "
                "Third sentence warns that the result may not generalize."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        selected = document["sections"][0]["paragraphs"][0]["spans"][1]
        return {
            "code": code,
            "paper_id": document["id"],
            "paper_title": document["title"],
            "span_id": selected["id"],
            "selected_span": selected_fragment or selected["original"],
        }

    def authorized_mini_lab_payload(self, payload: dict) -> dict:
        run = _issue_experiment_run(
            paper_id=payload["paper_id"],
            paper_title=payload["paper_title"],
            span_id=payload["span_id"],
            selected_span=payload["selected_span"],
            code=payload["code"],
            experiment_trace_id="test_experiment_trace",
            starter_trace_id="test_starter_trace",
            provider="hf",
            model="test-model",
            starter_provider="hf",
            starter_model="test-model",
        )
        return {**payload, "experiment_run_id": run["id"]}

    def indexed_experiment_payload(self, *, idea: str = "Try source-bound evidence reranking", locale: str = "ko") -> dict:
        source = PaperSource(
            title="Evidence Reranking",
            authors="A. Author",
            source_label="unit-experiment",
            text=(
                "First sentence defines a retrieval baseline. "
                "Second sentence says the method improves retrieval with evidence-linked reranking. "
                "Third sentence limits the claim to indexed paper evidence."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        selected = document["sections"][0]["paragraphs"][0]["spans"][1]
        return {
            "paper_id": document["id"],
            "span_id": selected["id"],
            "paper_title": document["title"],
            "selected_span": selected["original"],
            "source_text": source.text,
            "idea": idea,
            "locale": locale,
            "use_model": True,
        }

    def tearDown(self):
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
        os.environ.pop("PAPERLENS_MEMORY_PATH", None)
        os.environ.pop("PAPERLENS_SOURCE_INDEX_DIR", None)
        os.environ.pop("PAPERLENS_TRANSLATION_CACHE_DIR", None)
        os.environ.pop("PAPERLENS_ENABLE_DIAGNOSTIC_STARTER", None)
        os.environ.pop("PAPERLENS_MINILAB_PROVIDER", None)
        self.tempdir.cleanup()

    def test_health_exposes_runtime_switches(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("provider", body)
        self.assertIn("forceModel", body)
        self.assertIn("traceContent", body)

    def test_paper_endpoint_preserves_reader_shape(self):
        response = self.client.post(
            "/api/paper",
            json={
                "pasted_text": (
                    "Title: Evidence Reranking\n\n"
                    "We propose evidence-linked reranking for retrieval augmented generation. "
                    "The method improves top-5 precision by 3.2 points over a relevance-only baseline. "
                    "Limitations include weak performance on ambiguous questions."
                )
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("sections", body)
        first_span = body["sections"][0]["paragraphs"][0]["spans"][0]
        self.assertIn("id", first_span)
        self.assertIn("original", first_span)
        self.assertIn("translated", first_span)
        self.assertTrue(first_span["id"].startswith("P0.S"))

    def test_manual_papers_get_distinct_content_bound_ids(self):
        first = self.client.post(
            "/api/paper",
            json={"pasted_text": "First paper sentence one. First paper sentence two."},
        )
        second = self.client.post(
            "/api/paper",
            json={"pasted_text": "Second paper alpha. Second paper beta."},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(first.json()["id"], second.json()["id"])
        self.assertTrue(first.json()["id"].startswith("manual-input-"))
        self.assertTrue(second.json()["id"].startswith("manual-input-"))

    def test_arxiv_endpoint_does_not_mask_pdf_path_with_example_text(self):
        captured = {}

        def fake_build_source(**kwargs):
            captured.update(kwargs)
            return PaperSource(
                title="Real arXiv Paper",
                authors="A. Author",
                source_label="arXiv:1706.03762",
                text=(
                    "Title: Real arXiv Paper. "
                    "This actual paper text should come from the PDF branch, not from the bundled example."
                ),
                pdf_url="https://arxiv.org/pdf/1706.03762",
            )

        with patch("paperlens_lab.server.build_source", side_effect=fake_build_source):
            response = self.client.post(
                "/api/paper",
                json={"arxiv_or_url": "1706.03762", "max_pdf_pages": 2},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["pasted_text"], "")
        self.assertEqual(captured["arxiv_or_url"], "1706.03762")
        body = response.json()
        self.assertEqual(body["source"], "arXiv:1706.03762")
        self.assertIn("metadata", body)

    def test_paper_endpoint_rejects_empty_input_instead_of_loading_sample(self):
        response = self.client.post("/api/paper", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Add a PDF, arXiv URL/ID, or paper text", response.json()["detail"])

    def test_paper_document_writes_source_index_and_ask_uses_evidence_window(self):
        source = PaperSource(
            title="Windowed Evidence Paper",
            authors="A. Author",
            source_label="arXiv:1234.56789",
            text=(
                "Sentence one defines the retrieval task. "
                "Sentence two describes the student query. "
                "Sentence three introduces a compact reranker. "
                "Sentence four reports that the compact reranker improves top five precision. "
                "Sentence five limits the claim to controlled evidence settings. "
                "Sentence six warns that broad deployment was not tested. "
                "Sentence seven describes the ablation."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        paper_id = document["id"]
        record = load_source_index(paper_id)

        self.assertIsNotNone(record)
        self.assertEqual(record["paper_id"], paper_id)
        self.assertGreaterEqual(len(record["spans"]), 7)

        selected = document["sections"][0]["paragraphs"][0]["spans"][3]
        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.answer_span.return_value = SimpleNamespace(
                data={
                    "answer": "The claim is limited to the evidence window.",
                    "evidence": [
                        {
                            "source_id": selected["id"],
                            "quote": "Sentence four reports that the compact reranker improves top five precision.",
                        }
                    ],
                    "confidence": "medium",
                    "needs_more_context": True,
                },
                text="The claim is limited to the evidence window.",
                model="test-model",
                provider="hf",
                trace_id="qa_window_test",
                error=None,
                used_fallback=False,
            )

            response = self.client.post(
                "/api/ask",
                json={
                    "paper_id": paper_id,
                    "span_id": selected["id"],
                    "question": "What exactly does this support?",
                    "original": "FORGED CLIENT TEXT SHOULD NOT REPLACE THE INDEXED SPAN",
                    "paper_title": document["title"],
                    "source_text": "CLIENT TEXT SHOULD NOT BE USED WHEN INDEX EXISTS",
                    "locale": "en",
                    "use_model": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["usedFallback"])
        self.assertEqual(body["evidenceWindow"]["spanId"], selected["id"])
        self.assertIn("P0.S1-P0.S7", body["evidenceWindow"]["spanRange"])
        model_source_text = gateway.answer_span.call_args.kwargs["source_text"]
        self.assertIn("Sentence one defines the retrieval task.", model_source_text)
        self.assertIn("Sentence seven describes the ablation.", model_source_text)
        self.assertNotIn("CLIENT TEXT SHOULD NOT BE USED", model_source_text)
        self.assertEqual(gateway.answer_span.call_args.kwargs["selected_span"], selected["original"])
        evidence_items = gateway.answer_span.call_args.kwargs["evidence_items_override"]
        self.assertEqual({item["source_id"] for item in evidence_items}, {span["span_id"] for span in record["spans"]})

    def test_ask_endpoint_rejects_missing_indexed_span(self):
        response = self.client.post(
            "/api/ask",
            json={
                "paper_id": "missing-paper",
                "span_id": "P0.S1",
                "question": "What does this say?",
                "original": "FORGED CLIENT TEXT",
                "paper_title": "Missing",
                "source_text": "Client text should not be accepted when paper_id is indexed.",
                "locale": "en",
                "use_model": True,
            },
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("Selected span was not found", response.json()["detail"])

    def test_ask_endpoint_accepts_free_text_inside_indexed_evidence_window(self):
        source = PaperSource(
            title="Free Selection Paper",
            authors="A. Author",
            source_label="manual-free-selection",
            text=(
                "First sentence defines the setup. "
                "Second sentence says the compact reranker improves top five precision in indexed source evidence. "
                "Third sentence limits the claim to controlled evidence settings."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        selected = document["sections"][0]["paragraphs"][0]["spans"][1]
        selected_fragment = "compact reranker improves top five precision"

        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.answer_span.return_value = SimpleNamespace(
                data={
                    "answer": "The selected fragment supports only the local precision claim.",
                    "evidence": [{"source_id": selected["id"], "quote": selected_fragment}],
                    "confidence": "medium",
                    "needs_more_context": True,
                },
                text="The selected fragment supports only the local precision claim.",
                model="test-model",
                provider="hf",
                trace_id="qa_free_selection_test",
                error=None,
                used_fallback=False,
            )

            response = self.client.post(
                "/api/ask",
                json={
                    "paper_id": document["id"],
                    "span_id": selected["id"],
                    "question": "What does this highlighted phrase support?",
                    "original": selected_fragment,
                    "paper_title": document["title"],
                    "source_text": "CLIENT TEXT SHOULD NOT BE USED WHEN INDEX EXISTS",
                    "locale": "en",
                    "use_model": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["usedFallback"])
        self.assertEqual(gateway.answer_span.call_args.kwargs["selected_span"], selected_fragment)
        self.assertIn(selected["id"], response.json()["supportSpanIds"])

    def test_ask_endpoint_accepts_multi_span_selected_ranges(self):
        source = PaperSource(
            title="Multi Selection Paper",
            authors="A. Author",
            source_label="manual-multi-selection",
            text=(
                "First sentence defines the setup. "
                "Second sentence says the compact reranker improves top five precision in indexed source evidence. "
                "Third sentence says the same reranker fails on ambiguous questions. "
                "Fourth sentence limits the claim to controlled evidence settings."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        spans = document["sections"][0]["paragraphs"][0]["spans"]
        first_segment = "compact reranker improves top five precision"
        second_segment = "same reranker fails on ambiguous questions"
        first_selected = spans[1]
        second_selected = spans[2]

        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.answer_span.return_value = SimpleNamespace(
                data={
                    "answer": "The selected ranges include both an improvement claim and a failure limitation.",
                    "evidence": [
                        {"source_id": first_selected["id"], "quote": first_segment},
                        {"source_id": second_selected["id"], "quote": second_segment},
                    ],
                    "confidence": "medium",
                    "needs_more_context": True,
                },
                text="The selected ranges include both an improvement claim and a failure limitation.",
                model="test-model",
                provider="hf",
                trace_id="qa_multi_selection_test",
                error=None,
                used_fallback=False,
            )

            response = self.client.post(
                "/api/ask",
                json={
                    "paper_id": document["id"],
                    "span_id": first_selected["id"],
                    "question": "What do these highlighted ranges support?",
                    "original": f"{first_segment} {second_segment}",
                    "paper_title": document["title"],
                    "source_text": "CLIENT TEXT SHOULD NOT BE USED WHEN INDEX EXISTS",
                    "locale": "en",
                    "use_model": True,
                    "selected_spans": [
                        {
                            "span_id": first_selected["id"],
                            "surface": "original",
                            "text": first_segment,
                            "start_offset": first_selected["original"].index(first_segment),
                            "end_offset": first_selected["original"].index(first_segment) + len(first_segment),
                        },
                        {
                            "span_id": second_selected["id"],
                            "surface": "original",
                            "text": second_segment,
                            "start_offset": second_selected["original"].index(second_segment),
                            "end_offset": second_selected["original"].index(second_segment) + len(second_segment),
                        },
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["usedFallback"])
        self.assertEqual(
            gateway.answer_span.call_args.kwargs["selected_span"],
            f"{first_segment} {second_segment}",
        )
        self.assertEqual(
            {item["source_id"] for item in gateway.answer_span.call_args.kwargs["evidence_items_override"]},
            {span["id"] for span in spans},
        )
        self.assertIn(first_selected["id"], body["supportSpanIds"])
        self.assertIn(second_selected["id"], body["supportSpanIds"])

    def test_ask_endpoint_splits_model_combined_quote_for_multi_span_selection(self):
        source = PaperSource(
            title="Combined Quote Multi Selection Paper",
            authors="A. Author",
            source_label="manual-combined-quote-selection",
            text=(
                "First sentence defines the setup. "
                "Second sentence says the compact reranker improves top five precision in indexed source evidence. "
                "Third sentence says the same reranker fails on ambiguous questions."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        spans = document["sections"][0]["paragraphs"][0]["spans"]
        first_segment = "compact reranker improves top five precision"
        second_segment = "same reranker fails on ambiguous questions"
        combined_quote = f"{first_segment} {second_segment}"

        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.answer_span.return_value = SimpleNamespace(
                data={
                    "answer": "The selected ranges connect an improvement claim with a limitation.",
                    "evidence": [{"source_id": spans[1]["id"], "quote": combined_quote}],
                    "confidence": "medium",
                    "needs_more_context": True,
                },
                text="The selected ranges connect an improvement claim with a limitation.",
                model="test-model",
                provider="hf",
                trace_id="qa_multi_combined_quote_test",
                error=None,
                used_fallback=False,
            )

            response = self.client.post(
                "/api/ask",
                json={
                    "paper_id": document["id"],
                    "span_id": spans[1]["id"],
                    "question": "How do these selected ranges connect?",
                    "original": combined_quote,
                    "paper_title": document["title"],
                    "locale": "en",
                    "use_model": True,
                    "selected_spans": [
                        {
                            "span_id": spans[1]["id"],
                            "surface": "original",
                            "text": first_segment,
                            "start_offset": spans[1]["original"].index(first_segment),
                            "end_offset": spans[1]["original"].index(first_segment) + len(first_segment),
                        },
                        {
                            "span_id": spans[2]["id"],
                            "surface": "original",
                            "text": second_segment,
                            "start_offset": spans[2]["original"].index(second_segment),
                            "end_offset": spans[2]["original"].index(second_segment) + len(second_segment),
                        },
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["usedFallback"])
        self.assertEqual(
            body["evidence"],
            [
                {"source_id": spans[1]["id"], "quote": first_segment},
                {"source_id": spans[2]["id"], "quote": second_segment},
            ],
        )
        self.assertEqual(body["supportSpanIds"], [spans[1]["id"], spans[2]["id"]])

    def test_ask_endpoint_rejects_forged_multi_span_selected_range(self):
        source = PaperSource(
            title="Forged Multi Selection Paper",
            authors="A. Author",
            source_label="manual-forged-multi-selection",
            text=(
                "First sentence defines the setup. "
                "Second sentence says the compact reranker improves top five precision."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        selected = document["sections"][0]["paragraphs"][0]["spans"][1]

        response = self.client.post(
            "/api/ask",
            json={
                "paper_id": document["id"],
                "span_id": selected["id"],
                "question": "What does this forged range support?",
                "original": "compact reranker secretly proves deployment readiness",
                "paper_title": document["title"],
                "locale": "en",
                "use_model": True,
                "selected_spans": [
                    {
                        "span_id": selected["id"],
                        "surface": "original",
                        "text": "secretly proves deployment readiness",
                        "start_offset": 0,
                        "end_offset": 20,
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("does not match the paper index", response.json()["detail"])

    def test_ask_endpoint_rejects_source_id_outside_evidence_window(self):
        source = PaperSource(
            title="Window Boundary Paper",
            authors="A. Author",
            source_label="manual-window",
            text=(
                "First sentence defines the setup. "
                "Second sentence carries the selected claim. "
                "Third sentence gives the local limitation."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        paper_id = document["id"]
        selected = document["sections"][0]["paragraphs"][0]["spans"][1]

        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.answer_span.return_value = SimpleNamespace(
                data={
                    "answer": "This cites a generated id that is not in the source-index window.",
                    "evidence": [{"source_id": "S3", "quote": selected["original"]}],
                    "confidence": "high",
                    "needs_more_context": False,
                },
                text="This cites a generated id that is not in the source-index window.",
                model="test-model",
                provider="hf",
                trace_id="qa_window_id_test",
                error=None,
                used_fallback=False,
            )

            response = self.client.post(
                "/api/ask",
                json={
                    "paper_id": paper_id,
                    "span_id": selected["id"],
                    "question": "What exactly does this support?",
                    "original": selected["original"],
                    "paper_title": document["title"],
                    "locale": "en",
                    "use_model": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["usedFallback"])
        self.assertEqual(body["confidence"], "low")
        self.assertIn("outside the selected evidence window", body["error"])

    def test_ask_endpoint_rejects_quote_mismatched_to_cited_source_id(self):
        source = PaperSource(
            title="Window Quote Attribution Paper",
            authors="A. Author",
            source_label="manual-window-quote",
            text=(
                "First sentence defines the setup. "
                "Second sentence carries the selected claim. "
                "Third sentence gives the local limitation."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        paper_id = document["id"]
        selected = document["sections"][0]["paragraphs"][0]["spans"][1]
        first = document["sections"][0]["paragraphs"][0]["spans"][0]

        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.answer_span.return_value = SimpleNamespace(
                data={
                    "answer": "This cites the wrong source id for the quote.",
                    "evidence": [{"source_id": selected["id"], "quote": first["original"]}],
                    "confidence": "high",
                    "needs_more_context": False,
                },
                text="This cites the wrong source id for the quote.",
                model="test-model",
                provider="hf",
                trace_id="qa_window_quote_test",
                error=None,
                used_fallback=False,
            )

            response = self.client.post(
                "/api/ask",
                json={
                    "paper_id": paper_id,
                    "span_id": selected["id"],
                    "question": "What exactly does this support?",
                    "original": selected["original"],
                    "paper_title": document["title"],
                    "locale": "en",
                    "use_model": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["usedFallback"])
        self.assertEqual(body["confidence"], "low")
        self.assertIn("does not match the cited source id", body["error"])

    def test_translate_span_uses_source_index_and_cache(self):
        source = PaperSource(
            title="Translation Cache Paper",
            authors="A. Author",
            source_label="manual-cache",
            text=(
                "First sentence defines the setup. "
                "Second sentence carries the metric. "
                "Third sentence is selected for translation. "
                "Fourth sentence gives the limitation."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        paper_id = document["id"]
        selected = document["sections"][0]["paragraphs"][0]["spans"][2]

        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.model_id = "test-model"
            gateway.translation_model_id = "test-model"
            gateway.provider = "hf"
            gateway.translate_spans.return_value = SimpleNamespace(
                data={
                    "translations": [
                        {
                            "span_id": selected["id"],
                            "translation": "세 번째 문장은 번역 캐시 검증을 위해 선택된다.",
                        }
                    ]
                },
                model="test-model",
                provider="hf",
                trace_id="translate_span_test",
                error=None,
                used_fallback=False,
            )

            first = self.client.post(
                "/api/translate-span",
                json={
                    "paper_id": paper_id,
                    "paper_title": document["title"],
                    "span_id": selected["id"],
                    "source_text": "CLIENT TEXT SHOULD NOT BE TRANSLATED",
                    "locale": "ko",
                    "use_model": True,
                },
            )
            second = self.client.post(
                "/api/translate-span",
                json={
                    "paper_id": paper_id,
                    "paper_title": document["title"],
                    "span_id": selected["id"],
                    "locale": "ko",
                    "use_model": True,
                },
            )
            third = self.client.post(
                "/api/translate-span",
                json={
                    "paper_id": paper_id,
                    "paper_title": document["title"],
                    "span_id": selected["id"],
                    "locale": "ko",
                    "use_model": True,
                    "force_refresh": True,
                },
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 200)
        self.assertEqual(first.json()["translation"], "세 번째 문장은 번역 캐시 검증을 위해 선택된다.")
        self.assertEqual(first.json()["sourceHash"], text_hash(selected["original"]))
        self.assertTrue(first.json()["sourceIndexBound"])
        self.assertEqual(second.json()["status"], "cached")
        self.assertEqual(third.json()["status"], "ready")
        self.assertEqual(second.json()["sourceHash"], text_hash(selected["original"]))
        self.assertEqual(gateway.translate_spans.call_count, 2)
        translated_payload = gateway.translate_spans.call_args.args[1][0]
        self.assertEqual(translated_payload["text"], selected["original"])
        self.assertNotIn("CLIENT TEXT SHOULD NOT BE TRANSLATED", translated_payload["text"])

    def test_translate_span_rejects_index_miss_even_with_client_text(self):
        response = self.client.post(
            "/api/translate-span",
            json={
                "paper_id": "missing-paper",
                "paper_title": "Missing",
                "span_id": "P9.S9",
                "source_text": "Client text should not be accepted for an indexed request.",
                "locale": "ko",
                "use_model": True,
            },
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("Selected span was not found", response.json()["detail"])

    def test_translate_endpoint_uses_source_index_and_batch_cache(self):
        source = PaperSource(
            title="Batch Cache Paper",
            authors="A. Author",
            source_label="manual-batch-cache",
            text=(
                "First indexed sentence is available for the batch translator. "
                "Second indexed sentence should also be resolved through the source index. "
                "Third indexed sentence gives a control example."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        spans = document["sections"][0]["paragraphs"][0]["spans"][:2]
        paper_id = document["id"]

        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.model_id = "test-model"
            gateway.translation_model_id = "test-model"
            gateway.provider = "hf"
            gateway.translate_spans.return_value = SimpleNamespace(
                data={
                    "translations": [
                        {"span_id": spans[0]["id"], "translation": "첫 번째 문장 배치 번역"},
                        {"span_id": spans[1]["id"], "translation": "두 번째 문장 배치 번역"},
                    ],
                    "notes": ["batched"],
                },
                model="test-model",
                provider="hf",
                trace_id="translate_batch_test",
                error=None,
                used_fallback=False,
            )

            first = self.client.post(
                "/api/translate",
                json={
                    "paper_id": paper_id,
                    "paper_title": document["title"],
                    "spans": [
                        {
                            "span_id": spans[0]["id"],
                            "text": "CLIENT TEXT SHOULD NOT BE USED FOR THE FIRST SPAN",
                        },
                        {
                            "span_id": spans[1]["id"],
                            "text": "CLIENT TEXT SHOULD NOT BE USED FOR THE SECOND SPAN",
                        },
                    ],
                    "locale": "ko",
                    "use_model": True,
                },
            )
            second = self.client.post(
                "/api/translate",
                json={
                    "paper_id": paper_id,
                    "paper_title": document["title"],
                    "spans": [
                        {"span_id": spans[0]["id"]},
                        {"span_id": spans[1]["id"]},
                    ],
                    "locale": "ko",
                    "use_model": True,
                },
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        first_body = first.json()
        second_body = second.json()
        self.assertEqual(gateway.translate_spans.call_count, 1)
        translated_payload = gateway.translate_spans.call_args.args[1]
        self.assertEqual(
            [item["text"] for item in translated_payload],
            [spans[0]["original"], spans[1]["original"]],
        )
        self.assertNotIn("CLIENT TEXT SHOULD NOT BE USED", translated_payload[0]["text"])
        self.assertEqual(
            [item["translation"] for item in first_body["translations"]],
            ["첫 번째 문장 배치 번역", "두 번째 문장 배치 번역"],
        )
        self.assertEqual(
            [item["status"] for item in second_body["translations"]],
            ["cached", "cached"],
        )
        self.assertTrue(all(item["sourceIndexBound"] for item in first_body["translations"]))
        self.assertEqual(
            [item["sourceHash"] for item in first_body["translations"]],
            [text_hash(spans[0]["original"]), text_hash(spans[1]["original"])],
        )

    def test_paper_document_translates_in_small_batches(self):
        os.environ["PAPERLENS_TRANSLATION_BATCH_SIZE"] = "2"
        source = PaperSource(
            title="Batch Paper",
            authors="A. Author",
            source_label="manual",
            text=(
                "First sentence has enough content to become a reader span. "
                "Second sentence has enough content to become a reader span. "
                "Third sentence has enough content to become a reader span. "
                "Fourth sentence has enough content to become a reader span. "
                "Fifth sentence has enough content to become a reader span."
            ),
        )

        def fake_translate(title, spans, locale, use_model):
            return type(
                "Result",
                (),
                {
                    "data": {
                        "translations": [
                            {"span_id": item["span_id"], "translation": f"ko {item['span_id']}"}
                            for item in spans
                        ]
                    }
                },
            )()

        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway_cls.return_value.translate_spans.side_effect = fake_translate
            document = paper_document_from_source(
                source,
                use_model=True,
                max_translate_spans=5,
                max_reader_spans=12,
            )

        os.environ.pop("PAPERLENS_TRANSLATION_BATCH_SIZE", None)
        self.assertEqual(gateway_cls.return_value.translate_spans.call_count, 3)
        translated = [
            span["translated"]
            for section in document["sections"]
            for paragraph in section["paragraphs"]
            for span in paragraph["spans"]
        ]
        self.assertEqual(translated[:5], ["ko P0.S1", "ko P0.S2", "ko P0.S3", "ko P0.S4", "ko P0.S5"])

    def test_ask_endpoint_keeps_frontend_shape_and_adds_trace(self):
        response = self.client.post(
            "/api/ask",
            json={
                "span_id": "P0.S1",
                "question": "이게 무슨 뜻이야?",
                "original": "The method improves top-5 precision by 3.2 points.",
                "translated": "이 방법은 top-5 precision을 3.2점 향상시킨다.",
                "paper_title": "Evidence Reranking",
                "source_text": "The method improves top-5 precision by 3.2 points. Limitations include ambiguity.",
                "locale": "ko",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["role"], "assistant")
        self.assertIn("content", body)
        self.assertIn("P0.S1", body["supportSpanIds"])
        self.assertIn("traceId", body)
        self.assertTrue(body["usedFallback"])

    def test_ask_endpoint_rejects_model_quote_not_in_source(self):
        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.answer_span.return_value.task = "grounded_qa"
            gateway.answer_span.return_value.text = "Unsupported model answer"
            gateway.answer_span.return_value.data = {
                "answer": "Unsupported model answer",
                "evidence": [{"source_id": "P0.S1", "quote": "This quote is not in the paper."}],
                "confidence": "high",
                "needs_more_context": False,
            }
            gateway.answer_span.return_value.model = "test-model"
            gateway.answer_span.return_value.provider = "hf"
            gateway.answer_span.return_value.trace_id = "qa_test"
            gateway.answer_span.return_value.error = None
            gateway.answer_span.return_value.used_fallback = False

            response = self.client.post(
                "/api/ask",
                json={
                    "span_id": "P0.S1",
                    "question": "What does this say?",
                    "original": "The actual paper says only this sentence.",
                    "paper_title": "Evidence Reranking",
                    "source_text": "The actual paper says only this sentence.",
                    "locale": "en",
                    "use_model": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["usedFallback"])
        self.assertEqual(body["confidence"], "low")
        self.assertTrue(body["needsMoreContext"])
        self.assertIn("not present", body["error"])

    def test_experiment_and_growth_endpoints_are_structured(self):
        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.experiment_spec.return_value = SimpleNamespace(
                data={
                    "research_question": "Can evidence-linked reranking improve the selected source-evidence behavior?",
                    "mini_lab_goal": "Run a source-bound comparison over indexed evidence rows.",
                    "dataset": {"name": "Indexed PaperLens evidence", "source": "source-index rows"},
                    "baseline": "Direct lexical retrieval baseline",
                    "metric": "source evidence score",
                    "steps": [
                        "Load indexed paper evidence rows.",
                        "Run the direct baseline.",
                        "Run the paper-inspired reranker.",
                    ],
                    "ablation": "Remove only the evidence-linked reranking mechanism.",
                    "failure_condition": "source evidence score does not improve.",
                },
                text="card",
                model="test-model",
                provider="hf",
                trace_id="experiment_contract",
                error=None,
                used_fallback=False,
            )
            gateway.starter_code.return_value = SimpleNamespace(
                data={
                    "code": (
                        "METRIC = 'source evidence score'\n"
                        "FAILURE_CONDITION = 'source evidence score does not improve.'\n"
                        "def baseline(example):\n    return example['text']\n"
                        "def paper_inspired(example):\n    return example['text']\n"
                        "def score(output, expected):\n    return 1.0 if output else 0.0\n"
                        "def run(evidence_rows=None):\n    return []\n"
                    )
                },
                text="",
                model="test-model",
                provider="hf",
                trace_id="starter_contract",
                error=None,
                used_fallback=False,
            )
            gateway.growth_ideas.return_value = SimpleNamespace(
                data={
                    "ideas": [
                        {
                            "idea": "Compare the reranker on source-index evidence rows.",
                            "source_evidence": ["paper:s1", "run:r1"],
                            "novelty_angle": "Keep the next step bound to paper evidence.",
                            "testable_next_step": "Run the source-bound mini-lab again with a harder evidence window.",
                            "risk": "The signal may not generalize beyond indexed evidence.",
                        }
                    ],
                    "fine_tuning_signal": "none",
                    "reason": "No repeated model failures were observed in this contract test.",
                },
                text="growth",
                model="test-model",
                provider="hf",
                trace_id="growth_contract",
                error=None,
                used_fallback=False,
            )

            experiment = self.client.post("/api/experiment", json=self.indexed_experiment_payload(locale="en"))
            growth = self.client.post(
                "/api/growth",
                json={
                    "paper_title": "Evidence Reranking",
                    "selected_span": "The method improves retrieval with evidence-linked reranking.",
                    "paper_memory": [{"id": "paper:s1", "summary": "Evidence reranking may improve precision."}],
                    "mini_lab_result": "run:r1 improved precision but failed on ambiguous examples.",
                    "locale": "en",
                    "use_model": True,
                },
            )
        self.assertEqual(experiment.status_code, 200)
        exp_body = experiment.json()
        self.assertIn("card", exp_body)
        self.assertIn("starter", exp_body)
        self.assertIn("experimentRunId", exp_body)
        self.assertEqual(exp_body["experimentRun"]["codeHash"], code_hash(exp_body["starter"]))
        self.assertEqual(exp_body["experimentRun"]["spanId"], self.indexed_experiment_payload(locale="en")["span_id"])
        self.assertIn("spec", exp_body)
        self.assertIn("metric", exp_body["spec"])
        self.assertIn("source", str(exp_body["spec"]).lower())
        self.assertIn(exp_body["spec"]["metric"], exp_body["starter"])
        self.assertIn(exp_body["spec"]["failure_condition"], exp_body["starter"])

        self.assertEqual(growth.status_code, 200)
        growth_body = growth.json()
        self.assertGreaterEqual(len(growth_body["ideas"]), 1)
        self.assertIn("fineTuningSignal", growth_body)

    def test_ui_originated_gpu_candidate_approval_and_run_contract(self):
        payload = self.indexed_experiment_payload(locale="en")
        candidate = {
            "id": "gpu-replication-probe",
            "title": "GPU replication probe",
            "kind": "gpu_replication_probe",
            "reproduction_level": "scaled",
            "faithfulness": {
                "level": "scaled",
                "summary": "Bounded reproduction of the paper claim direction.",
                "why_not_exact": "The fixture does not provide a source-listed official repo.",
                "paper_targets": ["test_accuracy"],
                "resource_note": "Short GPU run.",
            },
            "is_recommended": True,
            "recommendation_reason": "It gives the clearest short GPU-backed signal.",
            "hypothesis": "Adam-style optimization should improve early accuracy over SGD on a public image benchmark.",
            "paper_evidence_ids": ["P0.S2"],
            "paper_evidence_quotes": ["The selected span describes an optimization mechanism."],
            "dataset": {"name": "MNIST", "source": "torchvision.datasets.MNIST", "requires_download": True},
            "implementation": {"type": "public_dataset", "repo_url": "", "reason": "No source repo is required."},
            "run_plan": {
                "repo_url": "",
                "config_path": "",
                "command": "python run_probe.py",
                "dataset": "MNIST",
                "expected_artifact": "test_accuracy",
            },
            "gpu_required": True,
            "estimated_runtime_minutes": 8,
            "expected_metric": "test_accuracy",
            "limitations": ["Directional probe only, not full paper reproduction."],
            "approval_question": "Run the GPU replication probe?",
        }
        alternate = {
            **candidate,
            "id": "source-window-probe",
            "title": "Source evidence window probe",
            "kind": "source_bound_probe",
            "is_recommended": False,
            "gpu_required": False,
        }
        script = (
            "import torch\n\n"
            "def run_paperlens_gpu_probe(config=None):\n"
            "    cuda = torch.cuda.is_available()\n"
            "    return {\n"
            "        'passed': True,\n"
            "        'metrics': {'test_accuracy': 0.95},\n"
            "        'rows': [{'metric': 'test_accuracy', 'value': 0.95}],\n"
            "        'logs': [f'cuda={cuda}'],\n"
            "        'hardware': {'cudaAvailable': cuda},\n"
            "        'dataset': {'name': 'MNIST', 'source': 'torchvision.datasets.MNIST'},\n"
            "        'limitations': ['short directional run'],\n"
            "        'claim_comparison': {'verdict': 'directionally_consistent'},\n"
            "    }\n"
        )
        with patch("paperlens_lab.server.ModelGateway") as gateway_cls, patch(
            "paperlens_lab.server.run_gpu_probe_job"
        ) as run_gpu:
            gateway = gateway_cls.return_value
            gateway.experiment_candidates.return_value = SimpleNamespace(
                data={
                    "candidates": [candidate, alternate],
                    "recommended_candidate_id": "gpu-replication-probe",
                },
                text="candidates",
                model="test-model",
                provider="hf",
                trace_id="candidate_trace",
                error=None,
                used_fallback=False,
            )
            gateway.gpu_script.return_value = SimpleNamespace(
                data={
                    "script": script,
                    "entrypoint": "run_paperlens_gpu_probe",
                    "dependencies": ["torch"],
                    "hardware": "T4",
                    "dataset": {"name": "MNIST", "source": "torchvision.datasets.MNIST"},
                    "reproduction_level": "scaled",
                    "reproduction_plan": {
                        "level": "scaled",
                        "repo_url": "",
                        "config_path": "",
                        "command": "python run_probe.py",
                        "dataset": "MNIST",
                        "expected_artifact": "test_accuracy",
                        "faithfulness_note": "Bounded reproduction.",
                    },
                    "expected_outputs": ["test_accuracy"],
                    "paper_claim_comparison_plan": "Compare early Adam-style convergence directionally.",
                    "limitations": ["short directional run"],
                },
                text=script,
                model="test-model",
                provider="hf",
                trace_id="gpu_script_trace",
                error=None,
                used_fallback=False,
            )
            run_gpu.return_value = {
                "passed": True,
                "reasons": [],
                "provider": "modal",
                "executionMode": "modal-gpu-replication-probe",
                "runner": "paperlens-modal-gpu-probe",
                "gpuRequested": True,
                "hardware": {"cudaAvailable": True, "gpuName": "Tesla T4"},
                "paperId": payload["paper_id"],
                "paperTitle": payload["paper_title"],
                "spanId": payload["span_id"],
                "candidateSetId": "filled-by-binding",
                "candidateId": "gpu-replication-probe",
                "sourceHash": "source",
                "codeHash": "code",
                "evidenceHash": "evidence",
                "evidenceRowCount": 3,
                "reproductionLevel": "scaled",
                "requestedReproductionLevel": "scaled",
                "validation": {"providerIsModal": True, "gpuRequested": True},
                "dataset": {"name": "MNIST"},
                "metrics": {"test_accuracy": 0.95},
                "rows": [{"metric": "test_accuracy", "value": 0.95}],
                "logs": ["modal gpu run"],
                "claimComparison": {"verdict": "directionally_consistent"},
                "limitations": ["short directional run"],
                "durationMs": 123,
            }

            candidates_response = self.client.post(
                "/api/experiment/candidates",
                json={
                    **payload,
                    "question": "What experiment should we run for this span?",
                    "reproduction_level": "scaled",
                    "use_model": True,
                },
            )
            self.assertEqual(candidates_response.status_code, 200)
            candidates_body = candidates_response.json()
            self.assertEqual(len(candidates_body["candidates"]), 2)
            self.assertEqual(candidates_body["recommendedCandidateId"], "gpu-replication-probe")
            self.assertEqual(candidates_body["reproductionLevel"], "scaled")

            unapproved_run = self.client.post("/api/gpu-lab/run", json={"gpu_run_id": ""})
            self.assertEqual(unapproved_run.status_code, 403)

            script_response = self.client.post(
                "/api/experiment/gpu-script",
                json={
                    "candidate_set_id": candidates_body["candidateSetId"],
                    "candidate_id": "gpu-replication-probe",
                    "paper_id": payload["paper_id"],
                    "span_id": payload["span_id"],
                    "selected_span": payload["selected_span"],
                    "reproduction_level": "scaled",
                    "locale": "en",
                    "use_model": True,
                },
            )
            self.assertEqual(script_response.status_code, 200)
            script_body = script_response.json()
            self.assertIn("gpuRunId", script_body)
            self.assertIn("run_paperlens_gpu_probe", script_body["script"])
            self.assertEqual(script_body["reproductionLevel"], "scaled")
            self.assertEqual(script_body["gpuRun"]["reproductionLevel"], "scaled")

            run_response = self.client.post("/api/gpu-lab/run", json={"gpu_run_id": script_body["gpuRunId"]})
            self.assertEqual(run_response.status_code, 200)
            run_body = run_response.json()
            self.assertEqual(run_body["provider"], "modal")
            self.assertEqual(run_body["reproductionLevel"], "scaled")
            self.assertTrue(run_body["gpuRequested"])
            self.assertEqual(run_body["metrics"]["test_accuracy"], 0.95)
            run_gpu.assert_called_once()

    def test_gpu_script_failure_returns_public_product_message(self):
        payload = self.indexed_experiment_payload(locale="en")
        candidate = {
            "id": "gpu-replication-probe",
            "title": "GPU replication probe",
            "kind": "gpu_replication_probe",
            "is_recommended": True,
            "recommendation_reason": "It gives the clearest short GPU-backed signal.",
            "hypothesis": "A short GPU probe should use real paper-linked data.",
            "paper_evidence_ids": ["P0.S2"],
            "paper_evidence_quotes": ["The selected span describes an optimization mechanism."],
            "dataset": {"name": "MNIST", "source": "torchvision.datasets.MNIST", "requires_download": True},
            "implementation": {"type": "public_dataset", "repo_url": "", "reason": "No source repo is required."},
            "gpu_required": True,
            "estimated_runtime_minutes": 8,
            "expected_metric": "test_accuracy",
            "limitations": ["Directional probe only, not full paper reproduction."],
            "approval_question": "Run the GPU replication probe?",
        }
        alternate = {
            **candidate,
            "id": "source-window-probe",
            "title": "Source evidence window probe",
            "kind": "source_bound_probe",
            "is_recommended": False,
            "gpu_required": False,
        }
        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.experiment_candidates.return_value = SimpleNamespace(
                data={
                    "candidates": [candidate, alternate],
                    "recommended_candidate_id": "gpu-replication-probe",
                },
                text="candidates",
                model="test-model",
                provider="hf",
                trace_id="candidate_trace",
                error=None,
                used_fallback=False,
            )
            gateway.gpu_script.return_value = SimpleNamespace(
                data={},
                text="",
                model="test-model",
                provider="hf",
                trace_id="gpu_script_trace",
                error="generated GPU script failed checks: torch.randn; fallback used",
                used_fallback=False,
            )

            candidates_response = self.client.post(
                "/api/experiment/candidates",
                json={
                    **payload,
                    "question": "What experiment should we run for this span?",
                    "use_model": True,
                },
            )
            self.assertEqual(candidates_response.status_code, 200)
            candidates_body = candidates_response.json()

            script_response = self.client.post(
                "/api/experiment/gpu-script",
                json={
                    "candidate_set_id": candidates_body["candidateSetId"],
                    "candidate_id": "gpu-replication-probe",
                    "paper_id": payload["paper_id"],
                    "span_id": payload["span_id"],
                    "selected_span": payload["selected_span"],
                    "locale": "en",
                    "use_model": True,
                },
            )

        self.assertEqual(script_response.status_code, 503)
        detail = script_response.json()["detail"]
        self.assertIn("did not pass service execution checks", detail)
        self.assertNotIn("fallback", detail.lower())
        self.assertNotIn("torch.randn", detail)

    def test_exact_gpu_script_requires_inspected_paper_repo(self):
        payload = self.indexed_experiment_payload(locale="en")
        candidate = {
            "id": "exact-reproduction",
            "title": "Exact repo reproduction",
            "kind": "gpu_replication_probe",
            "reproduction_level": "exact",
            "is_recommended": True,
            "recommendation_reason": "Use the paper implementation when it is available.",
            "hypothesis": "The official implementation should reproduce the reported metric.",
            "paper_evidence_ids": ["P0.S2"],
            "paper_evidence_quotes": ["The selected span describes the paper result."],
            "dataset": {"name": "ImageNet validation", "source": "official repo config"},
            "implementation": {
                "type": "paper_repo",
                "repo_url": "https://github.com/example/paperlens-missing-official-repo",
                "reason": "Paper source-listed implementation.",
            },
            "run_plan": {
                "repo_url": "https://github.com/example/paperlens-missing-official-repo",
                "config_path": "configs/eval.yaml",
                "command": "python tools/eval.py --config configs/eval.yaml",
                "dataset": "ImageNet validation",
                "expected_artifact": "top1 accuracy",
            },
            "gpu_required": True,
            "estimated_runtime_minutes": 30,
            "expected_metric": "top1 accuracy",
            "limitations": ["Exact requires the repo to be inspected before execution."],
            "approval_question": "Run the exact reproduction?",
        }
        with patch("paperlens_lab.server.ModelGateway") as gateway_cls, patch(
            "paperlens_lab.server.inspect_implementation_repositories"
        ) as inspect_repos:
            gateway = gateway_cls.return_value
            gateway.experiment_candidates.return_value = SimpleNamespace(
                data={
                    "candidates": [candidate],
                    "recommended_candidate_id": "exact-reproduction",
                },
                text="candidates",
                model="test-model",
                provider="hf",
                trace_id="candidate_trace",
                error=None,
                used_fallback=False,
            )
            inspect_repos.return_value = [
                {
                    "url": "https://github.com/example/paperlens-missing-official-repo",
                    "status": "unavailable",
                    "error": "clone failed",
                }
            ]

            candidates_response = self.client.post(
                "/api/experiment/candidates",
                json={
                    **payload,
                    "question": "Can we reproduce this exactly?",
                    "reproduction_level": "exact",
                    "use_model": True,
                },
            )
            self.assertEqual(candidates_response.status_code, 200)
            candidates_body = candidates_response.json()
            _CANDIDATE_SETS[candidates_body["candidateSetId"]]["implementationLinks"] = [
                {
                    "url": "https://github.com/example/paperlens-missing-official-repo",
                    "source_url": "https://github.com/example/paperlens-missing-official-repo",
                    "source_id": "implementation:github:1",
                    "usage": "Paper source-listed implementation.",
                }
            ]

            script_response = self.client.post(
                "/api/experiment/gpu-script",
                json={
                    "candidate_set_id": candidates_body["candidateSetId"],
                    "candidate_id": "exact-reproduction",
                    "paper_id": payload["paper_id"],
                    "span_id": payload["span_id"],
                    "selected_span": payload["selected_span"],
                    "reproduction_level": "exact",
                    "locale": "en",
                    "use_model": True,
                },
            )

        self.assertEqual(script_response.status_code, 503)
        detail = script_response.json()["detail"]
        self.assertIn("Exact reproduction requires an inspected implementation repository", detail)
        gateway.gpu_script.assert_not_called()

    def test_gpu_script_rejects_script_level_mismatch_before_run_binding(self):
        payload = self.indexed_experiment_payload(locale="en")
        candidate = {
            "id": "exact-reproduction",
            "title": "Exact repo reproduction",
            "kind": "gpu_replication_probe",
            "reproduction_level": "exact",
            "is_recommended": True,
            "recommendation_reason": "Use the paper implementation when it is available.",
            "hypothesis": "The official implementation should reproduce the reported metric.",
            "paper_evidence_ids": ["P0.S2"],
            "paper_evidence_quotes": ["The selected span describes the paper result."],
            "dataset": {"name": "ImageNet validation", "source": "official repo config"},
            "implementation": {
                "type": "paper_repo",
                "repo_url": "https://github.com/example/paperlens-official-repo",
                "reason": "Paper source-listed implementation.",
            },
            "run_plan": {
                "repo_url": "https://github.com/example/paperlens-official-repo",
                "config_path": "configs/eval.yaml",
                "command": "python tools/eval.py --config configs/eval.yaml",
                "dataset": "ImageNet validation",
                "expected_artifact": "top1 accuracy",
            },
            "gpu_required": True,
            "estimated_runtime_minutes": 30,
            "expected_metric": "top1 accuracy",
            "limitations": ["Exact repo path must remain exact through script generation."],
            "approval_question": "Run the exact reproduction?",
        }
        script = (
            "import torch\n\n"
            "def run_paperlens_gpu_probe(config=None):\n"
            "    cuda = torch.cuda.is_available()\n"
            "    return {\n"
            "        'passed': True,\n"
            "        'metrics': {'rows': 1},\n"
            "        'rows': [{'metric': 'rows', 'value': 1}],\n"
            "        'logs': [f'cuda={cuda}'],\n"
            "        'hardware': {'cudaAvailable': cuda},\n"
            "        'dataset': {'name': 'ImageNet validation', 'source': 'official repo config'},\n"
            "        'limitations': ['bounded validation slice'],\n"
            "        'claim_comparison': {'verdict': 'completed'},\n"
            "    }\n"
        )
        with patch("paperlens_lab.server.ModelGateway") as gateway_cls, patch(
            "paperlens_lab.server.inspect_implementation_repositories"
        ) as inspect_repos:
            gateway = gateway_cls.return_value
            gateway.experiment_candidates.return_value = SimpleNamespace(
                data={
                    "candidates": [candidate],
                    "recommended_candidate_id": "exact-reproduction",
                },
                text="candidates",
                model="test-model",
                provider="hf",
                trace_id="candidate_trace",
                error=None,
                used_fallback=False,
            )
            gateway.gpu_script.return_value = SimpleNamespace(
                data={
                    "script": script,
                    "entrypoint": "run_paperlens_gpu_probe",
                    "dependencies": ["torch"],
                    "hardware": "T4",
                    "dataset": {"name": "ImageNet validation", "source": "official repo config"},
                    "reproduction_level": "scaled",
                    "reproduction_plan": {
                        "level": "scaled",
                        "repo_url": "",
                        "config_path": "",
                        "command": "python run_scaled.py",
                        "dataset": "ImageNet validation subset",
                        "expected_artifact": "rows",
                        "faithfulness_note": "Incorrectly downgraded script.",
                    },
                    "expected_outputs": ["rows"],
                    "paper_claim_comparison_plan": "Compare a scaled subset.",
                    "limitations": ["scaled run"],
                },
                text=script,
                model="test-model",
                provider="hf",
                trace_id="gpu_script_trace",
                error=None,
                used_fallback=False,
            )
            inspect_repos.return_value = [
                {
                    "url": "https://github.com/example/paperlens-official-repo",
                    "status": "inspected",
                }
            ]

            candidates_response = self.client.post(
                "/api/experiment/candidates",
                json={
                    **payload,
                    "question": "Can we reproduce this exactly?",
                    "reproduction_level": "exact",
                    "use_model": True,
                },
            )
            self.assertEqual(candidates_response.status_code, 200)
            candidates_body = candidates_response.json()
            _CANDIDATE_SETS[candidates_body["candidateSetId"]]["implementationLinks"] = [
                {
                    "url": "https://github.com/example/paperlens-official-repo",
                    "source_url": "https://github.com/example/paperlens-official-repo",
                    "source_id": "implementation:github:1",
                    "usage": "Paper source-listed implementation.",
                }
            ]

            script_response = self.client.post(
                "/api/experiment/gpu-script",
                json={
                    "candidate_set_id": candidates_body["candidateSetId"],
                    "candidate_id": "exact-reproduction",
                    "paper_id": payload["paper_id"],
                    "span_id": payload["span_id"],
                    "selected_span": payload["selected_span"],
                    "reproduction_level": "exact",
                    "locale": "en",
                    "use_model": True,
                },
            )

        self.assertEqual(script_response.status_code, 503)
        detail = script_response.json()["detail"]
        self.assertIn("did not pass service execution checks", detail)
        self.assertEqual(len(_GPU_PROBE_RUNS), 0)

    def test_experiment_candidates_reject_invalid_reproduction_level(self):
        payload = self.indexed_experiment_payload(locale="en")

        response = self.client.post(
            "/api/experiment/candidates",
            json={
                **payload,
                "question": "What experiment should we run for this span?",
                "reproduction_level": "demo-only",
                "use_model": True,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Reproduction level", response.json()["detail"])

    def test_gpu_result_treats_falsified_hypothesis_as_completed_execution(self):
        job = {
            "paperId": "paper-1",
            "paperTitle": "Real Paper",
            "spanId": "P0.S5",
            "candidateSetId": "candidate-set-1",
            "candidateId": "gpu-replication-probe",
            "sourceHash": "source-hash",
            "codeHash": "code-hash",
            "evidenceHash": "evidence-hash",
            "evidenceRowCount": 3,
            "reproductionLevel": "scaled",
            "requestedReproductionLevel": "scaled",
        }
        result = {
            "passed": False,
            "reasons": [],
            "provider": "modal",
            "executionMode": "modal-gpu-replication-probe",
            "runner": "paperlens-modal-gpu-probe",
            "gpuRequested": True,
            "hardware": {"cudaAvailable": True, "gpuName": "Tesla T4"},
            "paperId": "paper-1",
            "spanId": "P0.S5",
            "candidateId": "gpu-replication-probe",
            "codeHash": "code-hash",
            "evidenceHash": "evidence-hash",
            "metrics": {
                "transformer_tokens_per_sec": 110160.78,
                "lstm_tokens_per_sec": 618125.88,
            },
            "rows": [
                {"metric": "transformer_tokens_per_sec", "value": 110160.78},
                {"metric": "lstm_tokens_per_sec", "value": 618125.88},
            ],
            "logs": ["cuda=True", "gpu=Tesla T4"],
            "claimComparison": {
                "verdict": "not_supported",
                "generatedPassed": False,
                "summary": "Measured throughput did not support the generated hypothesis.",
            },
            "limitations": ["short directional run"],
            "durationMs": 1200,
        }

        body = _validated_gpu_result(result, job)

        self.assertTrue(body["passed"], body)
        self.assertEqual(body["reasons"], [])
        self.assertFalse(body["claimComparison"]["generatedPassed"])
        self.assertEqual(body["claimComparison"]["verdict"], "not_supported")
        self.assertEqual(body["reproductionLevel"], "scaled")

    def test_gpu_result_still_fails_real_execution_errors(self):
        job = {
            "paperId": "paper-1",
            "paperTitle": "Real Paper",
            "spanId": "P0.S5",
            "candidateSetId": "candidate-set-1",
            "candidateId": "gpu-replication-probe",
            "sourceHash": "source-hash",
            "codeHash": "code-hash",
            "evidenceHash": "evidence-hash",
            "evidenceRowCount": 3,
            "reproductionLevel": "probe",
            "requestedReproductionLevel": "probe",
        }
        result = {
            "passed": False,
            "reasons": ["RuntimeError: dataset download failed"],
            "provider": "modal",
            "executionMode": "modal-gpu-replication-probe",
            "runner": "paperlens-modal-gpu-probe",
            "gpuRequested": True,
            "hardware": {"cudaAvailable": True, "gpuName": "Tesla T4"},
            "paperId": "paper-1",
            "spanId": "P0.S5",
            "candidateId": "gpu-replication-probe",
            "codeHash": "code-hash",
            "evidenceHash": "evidence-hash",
            "metrics": {},
            "rows": [],
            "logs": ["Generated GPU probe raised an exception inside Modal."],
            "claimComparison": {"verdict": "failed_to_execute"},
            "limitations": [],
            "durationMs": 1200,
        }

        body = _validated_gpu_result(result, job)

        self.assertFalse(body["passed"], body)
        self.assertEqual(body["reproductionLevel"], "probe")
        self.assertIn("RuntimeError: dataset download failed", body["reasons"])
        self.assertIn("GPU probe returned no rows or metrics", body["reasons"])

    def test_gpu_script_contract_rejects_counter_most_common_items_bug(self):
        code = """
import torch
from collections import Counter
from datasets import load_dataset

def run_paperlens_gpu_probe(config=None):
    records = load_dataset("bentrevett/multi30k", split="train[:8]")
    counts = Counter(" ".join(row["en"] for row in records).split())
    vocab = {word: i for i, (word, count) in enumerate(counts.most_common(1000).items())}
    return {
        "passed": bool(vocab),
        "metrics": {"vocab": len(vocab)},
        "rows": [{"metric": "vocab", "value": len(vocab)}],
        "logs": [],
        "hardware": {"cudaAvailable": torch.cuda.is_available()},
        "dataset": {"name": "Multi30k", "source": "bentrevett/multi30k"},
        "limitations": ["bounded subset"],
        "claim_comparison": {"verdict": "directional_probe_only"},
    }
"""

        errors = _validate_gpu_script_contract(code)

        self.assertTrue(any("most_common" in reason for reason in errors), errors)

    def test_gpu_script_contract_allows_unrelated_dict_items_with_counter_most_common(self):
        code = """
import torch
from collections import Counter

def run_paperlens_gpu_probe(config=None):
    cuda = torch.cuda.is_available()
    counts = Counter(["residual", "depth", "residual"]).most_common(2)
    metadata = {"dataset": "bentrevett/multi30k", "split": "train[:8]"}
    metadata_pairs = list(metadata.items())
    return {
        "passed": True,
        "metrics": {"top_count": counts[0][1], "metadata_pairs": len(metadata_pairs)},
        "rows": [{"token": token, "count": count} for token, count in counts],
        "logs": [f"cuda={cuda}"],
        "hardware": {"cudaAvailable": cuda},
        "dataset": {"name": "Multi30k", "source": "bentrevett/multi30k train[:8]"},
        "limitations": ["bounded public dataset subset"],
        "claim_comparison": {"verdict": "directional_probe_only"},
    }
"""

        errors = _validate_gpu_script_contract(code)

        self.assertFalse(any("most_common" in reason for reason in errors), errors)

    def test_gpu_script_contract_allows_pytorch_model_eval_method(self):
        code = """
import torch
import torch.nn as nn

def run_paperlens_gpu_probe(config=None):
    model = nn.Linear(4, 2)
    model.eval()
    cuda = torch.cuda.is_available()
    return {
        "passed": True,
        "metrics": {"rows": 1},
        "rows": [{"metric": "rows", "value": 1}],
        "logs": [f"cuda={cuda}"],
        "hardware": {"cudaAvailable": cuda},
        "dataset": {"name": "real public dataset", "source": "torchvision/datasets path"},
        "limitations": ["contract-only test"],
        "claim_comparison": {"verdict": "directional_probe_only"},
    }
"""

        errors = _validate_gpu_script_contract(code)

        self.assertFalse(any("blocked call eval" in reason for reason in errors), errors)
        self.assertFalse(any("blocked operation eval" in reason for reason in errors), errors)

    def test_gpu_script_contract_rejects_builtin_eval_call(self):
        code = """
import torch

def run_paperlens_gpu_probe(config=None):
    cuda = torch.cuda.is_available()
    value = eval("1 + 1")
    return {
        "passed": True,
        "metrics": {"value": value},
        "rows": [{"metric": "value", "value": value}],
        "logs": [f"cuda={cuda}"],
        "hardware": {"cudaAvailable": cuda},
        "dataset": {"name": "real public dataset", "source": "torchvision/datasets path"},
        "limitations": ["contract-only test"],
        "claim_comparison": {"verdict": "directional_probe_only"},
    }
"""

        errors = _validate_gpu_script_contract(code)

        self.assertTrue(any("blocked call eval" in reason for reason in errors), errors)

    def test_growth_endpoint_rejects_unbound_or_toy_followup_ideas(self):
        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.growth_ideas.return_value = SimpleNamespace(
                data={
                    "ideas": [
                        {
                            "idea": "Use a toy setup before any real mini-lab run.",
                            "source_evidence": ["paper:s1"],
                            "novelty_angle": "Unbound setup",
                            "testable_next_step": "Run a toy setup.",
                            "risk": "Not tied to the actual run.",
                        }
                    ],
                    "fine_tuning_signal": "none",
                    "reason": "",
                },
                text="growth",
                model="test-model",
                provider="hf",
                trace_id="growth_contract_bad",
                error=None,
                used_fallback=False,
            )

            response = self.client.post(
                "/api/growth",
                json={
                    "paper_title": "Evidence Reranking",
                    "selected_span": "The method improves retrieval with evidence-linked reranking.",
                    "paper_memory": [{"id": "paper:s1", "summary": "Evidence reranking may improve precision."}],
                    "mini_lab_result": "experiment card only; no actual run:r1 result.",
                    "locale": "en",
                    "use_model": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["ideas"], [])
        self.assertTrue(body["usedFallback"])
        self.assertIn("toy", body["error"])

    def test_experiment_endpoint_attaches_source_repo_manifest_to_run(self):
        manifest = {
            "source_id": "implementation:github:1",
            "url": "https://github.com/microsoft/LoRA",
            "source_url": "https://github.com/microsoft/LoRA",
            "status": "inspected",
            "execution": "none",
            "commit": "abc123",
            "default_branch": "main",
            "file_count": 4,
            "total_bytes": 1200,
            "truncated": False,
            "files": [{"path": "README.md", "bytes": 20, "kind": "readme"}],
            "readme": {"path": "README.md", "excerpt": "LoRA"},
            "license": {"path": "LICENSE", "excerpt": "MIT"},
            "error": "",
        }
        with (
            patch("paperlens_lab.server.ModelGateway") as gateway_cls,
            patch("paperlens_lab.server.inspect_implementation_repositories", return_value=[manifest]) as inspect_repos,
        ):
            gateway = gateway_cls.return_value
            gateway.experiment_spec.return_value = SimpleNamespace(
                data={
                    "research_question": "Can LoRA implementation evidence be inspected before a source-bound run?",
                    "mini_lab_goal": "Inspect source-listed implementation metadata and run indexed evidence rows.",
                    "dataset": {"name": "Indexed PaperLens evidence", "source": "source-index rows"},
                    "baseline": "Direct source-evidence baseline",
                    "metric": "source evidence score",
                    "steps": ["Load indexed rows", "Inspect source-listed repo manifest", "Run source-bound variant"],
                    "ablation": "Remove only the source-listed implementation manifest context.",
                    "failure_condition": "source evidence score does not improve.",
                    "implementation_repositories": [
                        {
                            "source_id": "implementation:github:1",
                            "url": "https://github.com/microsoft/LoRA",
                            "source_url": "https://github.com/microsoft/LoRA",
                        }
                    ],
                },
                text="card",
                model="test-model",
                provider="hf",
                trace_id="experiment_repo_contract",
                error=None,
                used_fallback=False,
            )
            gateway.starter_code.return_value = SimpleNamespace(
                data={"code": _valid_starter_code()},
                text="",
                model="test-model",
                provider="hf",
                trace_id="starter_repo_contract",
                error=None,
                used_fallback=False,
            )
            gateway.translate_experiment_spec_display.return_value = None

            response = self.client.post("/api/experiment", json=self.indexed_experiment_payload(locale="en"))

        self.assertEqual(response.status_code, 200)
        body = response.json()
        inspect_repos.assert_called_once()
        self.assertEqual(inspect_repos.call_args.args[0][0]["url"], "https://github.com/microsoft/LoRA")
        self.assertEqual(
            gateway.starter_code.call_args.kwargs["implementation_repo_manifests"][0]["url"],
            "https://github.com/microsoft/LoRA",
        )
        self.assertEqual(
            gateway.starter_code.call_args.kwargs["implementation_repo_manifests"][0]["execution"],
            "none",
        )
        self.assertEqual(body["implementationRepoManifests"][0]["status"], "inspected")
        self.assertEqual(body["implementationRepoManifests"][0]["execution"], "none")
        self.assertEqual(body["experimentRun"]["implementationRepoManifests"][0]["commit"], "abc123")

    def test_experiment_and_growth_include_display_localization_when_translation_succeeds(self):
        with patch("paperlens_lab.server.ModelGateway") as gateway_cls:
            gateway = gateway_cls.return_value
            gateway.experiment_spec.return_value = SimpleNamespace(
                data={
                    "research_question": "How is the output of attention computed as a weighted sum of values?",
                    "mini_lab_goal": "Demonstrate the weighted-sum property on indexed paper evidence.",
                    "dataset": {
                        "name": "Indexed PaperLens evidence window",
                        "source": "source-index spans",
                    },
                    "baseline": "Scaled dot-product attention with fixed vectors",
                    "metric": "output_vector_similarity_to_weighted_sum",
                    "ablation": "Remove only the weighted sum stage.",
                    "failure_condition": "The mini-lab fails if the similarity metric drops below 0.8.",
                    "expected_result": "The output should remain close to the weighted sum.",
                    "steps": [
                        "Load indexed paper evidence rows.",
                        "Compute attention weights on the selected span.",
                        "Compare the source-bound metric.",
                    ],
                    "faithfulness_notes": ["Keep the run tied to the selected paper span."],
                },
                text="card",
                model="test-model",
                provider="hf",
                trace_id="exp_test",
                error=None,
                used_fallback=False,
            )
            gateway.starter_code.return_value = SimpleNamespace(
                data={"code": "def run():\n    return []\n"},
                text="",
                model="test-model",
                provider="hf",
                trace_id="starter_test",
                error=None,
                used_fallback=False,
            )
            gateway.growth_ideas.return_value = SimpleNamespace(
                data={
                    "ideas": [
                        {
                            "idea": "Test whether removing the scaling factor changes the weighted-sum behavior.",
                            "source_evidence": ["paper:s1", "run:r1"],
                            "novelty_angle": "",
                            "testable_next_step": "Run the source-bound mini-lab after removing the scaling factor.",
                            "risk": "The source-evidence metric may not generalize beyond this paper window.",
                        }
                    ],
                    "fine_tuning_signal": "none",
                    "reason": "",
                },
                text="growth",
                model="test-model",
                provider="hf",
                trace_id="growth_test",
                error=None,
                used_fallback=False,
            )
            def fake_translate_spans(_paper_title, spans, locale, use_model):
                translations = []
                mapping = {
                    "research_question": "어텐션의 출력은 값들의 가중합으로 어떻게 계산되는가?",
                    "metric": "출력 벡터와 가중합 사이의 유사도",
                    "dataset:name": "인덱싱된 논문 근거 창",
                    "dataset:source": "source-index spans",
                    "idea:0": "스케일링 팩터를 제거하면 가중합 동작이 달라지는지 테스트합니다.",
                }
                for item in spans:
                    span_id = item["span_id"]
                    if span_id in mapping:
                        translations.append({"span_id": span_id, "translation": mapping[span_id]})
                return SimpleNamespace(
                    data={"translations": translations},
                    text="",
                    model="test-model",
                    provider="hf",
                    trace_id="translate_mock",
                    error=None,
                    used_fallback=False,
                )

            gateway.translate_spans.side_effect = fake_translate_spans

            experiment = self.client.post("/api/experiment", json=self.indexed_experiment_payload())
            growth = self.client.post(
                "/api/growth",
                json={
                    "paper_title": "Evidence Reranking",
                    "selected_span": "The method improves retrieval with evidence-linked reranking.",
                    "paper_memory": [{"id": "paper:s1", "summary": "Evidence reranking may improve precision."}],
                    "mini_lab_result": "run:r1 improved precision but failed on ambiguous examples.",
                    "locale": "ko",
                    "use_model": True,
                },
            )

        self.assertEqual(experiment.status_code, 200)
        exp_body = experiment.json()
        self.assertEqual(
            exp_body["specDisplay"]["research_question"],
            "어텐션의 출력은 값들의 가중합으로 어떻게 계산되는가?",
        )
        self.assertEqual(exp_body["specDisplay"]["metric"], "출력 벡터와 가중합 사이의 유사도")
        self.assertEqual(
            exp_body["specDisplay"]["dataset"],
            {
                "name": "인덱싱된 논문 근거 창",
                "source": "source-index spans",
            },
        )
        self.assertEqual(exp_body["spec"]["metric"], "output_vector_similarity_to_weighted_sum")
        self.assertEqual(growth.status_code, 200)
        growth_body = growth.json()
        self.assertEqual(
            growth_body["ideas"][0]["displayIdea"],
            "스케일링 팩터를 제거하면 가중합 동작이 달라지는지 테스트합니다.",
        )

    def test_starter_run_endpoint_executes_source_rows(self):
        code = """
def baseline(example):
    return example["input"]

def paper_inspired(example):
    return example["input"] + " paper"

def score(output, expected):
    return 1.0 if expected in output else 0.0

def run(evidence_rows=None):
    examples = evidence_rows or []
    rows = []
    for example in examples:
        row = {"input": example["text"], "expected": "source" if example.get("gold") else "control"}
        base = baseline(row)
        proto = paper_inspired(row)
        baseline_score = score(base, row["expected"])
        prototype_score = score(proto, row["expected"])
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
        payload = self.indexed_starter_payload(code)
        response = self.client.post("/api/starter/run", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["passed"])
        self.assertEqual(body["reasons"], [])
        selected_row = next(row for row in body["rows"] if row["source_id"] == payload["span_id"])
        self.assertEqual(selected_row["prototype_score"], 1.0)

    def test_starter_run_endpoint_reports_syntax_error(self):
        response = self.client.post("/api/starter/run", json=self.indexed_starter_payload("def run(:\n"))

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["passed"])
        self.assertIn("syntax error", body["reasons"][0])

    def test_starter_run_endpoint_rejects_forged_evidence_hashes(self):
        code = """
def baseline(example):
    return example

def paper_inspired(example):
    return example

def score(output, expected):
    return 1.0

def run(evidence_rows=None):
    rows = []
    for example in evidence_rows or []:
        rows.append({
            "source_id": example["source_id"],
            "text_hash": "forged-hash",
            "baseline_score": 0.0,
            "prototype_score": 1.0,
            "metric": "source evidence score",
            "failure_condition": False,
            "failure_rule": "prototype_score <= baseline_score",
        })
    return rows
"""
        response = self.client.post("/api/starter/run", json=self.indexed_starter_payload(code))

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["passed"])
        self.assertTrue(any("text_hash" in reason for reason in body["reasons"]))

    def test_starter_run_endpoint_rejects_unsafe_code(self):
        code = """
import os

def baseline(example):
    return example

def paper_inspired(example):
    return example

def score(output, expected):
    return 0.0

def run(evidence_rows=None):
    open("/tmp/paperlens-unsafe-source-run", "w").write("nope")
    return [{
        "source_id": "P0.S1",
        "baseline_score": 0.0,
        "prototype_score": 0.0,
        "metric": "source evidence score",
        "failure_condition": True,
    }]
"""
        payload = self.indexed_starter_payload(code)
        response = self.client.post("/api/starter/run", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["passed"])
        self.assertIn("starter code may only import json, re, or math", body["reasons"])
        self.assertIn("starter code uses unsafe name open", body["reasons"])

    def test_starter_run_endpoint_times_out_infinite_loop(self):
        code = """
def baseline(example):
    return example

def paper_inspired(example):
    return example

def score(output, expected):
    return 0.0

def run(evidence_rows=None):
    while True:
        pass
"""
        payload = self.indexed_starter_payload(code)
        response = self.client.post("/api/starter/run", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["passed"])
        self.assertIn("starter subprocess timed out", body["reasons"])

    def test_starter_run_endpoint_allows_ord_chr_and_hash_helpers(self):
        code = """
def baseline(example):
    return {"prediction": example["candidates"][0]}

def paper_inspired(example):
    target = chr(ord("a"))
    hashed = hash(example["gold"]) != 0
    prediction = "attention" if target in example["gold"] and hashed else example["candidates"][0]
    return {"prediction": prediction, "hashed": hashed}

def score(output, expected):
    return 1.0 if output["prediction"] == expected else 0.0

def run(evidence_rows=None):
    examples = evidence_rows or []
    rows = []
    for example in examples:
        example = {
            **example,
            "candidates": ["recurrence", "attention"],
            "gold": "attention" if example.get("gold") else "recurrence",
        }
        base = baseline(example)
        proto = paper_inspired(example)
        baseline_score = score(base, example["gold"])
        prototype_score = score(proto, example["gold"])
        rows.append({
            "source_id": example["source_id"],
            "text_hash": example["text_hash"],
            "baseline_score": baseline_score,
            "prototype_score": prototype_score,
            "metric": "source evidence accuracy",
            "failure_condition": prototype_score <= baseline_score,
            "failure_rule": "prototype_score <= baseline_score",
        })
    return rows
"""
        payload = self.indexed_starter_payload(code)
        response = self.client.post("/api/starter/run", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["passed"])
        self.assertEqual(body["reasons"], [])
        selected_row = next(row for row in body["rows"] if row["source_id"] == payload["span_id"])
        self.assertEqual(selected_row["prototype_score"], 1.0)

    def test_mini_lab_run_endpoint_executes_source_bound_local_job(self):
        source = PaperSource(
            title="Mini Lab Paper",
            authors="A. Author",
            source_label="manual-mini-lab",
            text=(
                "First sentence introduces a compact evidence mechanism. "
                "Second sentence says the mechanism improves precision in indexed source evidence. "
                "Third sentence warns that the result may not generalize."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        selected = document["sections"][0]["paragraphs"][0]["spans"][1]
        code = _valid_starter_code()
        payload = self.authorized_mini_lab_payload(
            {
                "code": code,
                "paper_id": document["id"],
                "paper_title": document["title"],
                "span_id": selected["id"],
                "selected_span": selected["original"],
            }
        )

        response = self.client.post("/api/mini-lab/run", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["passed"])
        self.assertEqual(body["provider"], "local")
        self.assertEqual(body["sourceHash"], text_hash(selected["original"]))
        self.assertEqual(body["codeHash"], code_hash(code))
        self.assertTrue(body["sourceIndexBound"])
        self.assertTrue(body["validation"]["sourceHashMatches"])
        self.assertTrue(body["validation"]["codeHashMatches"])
        selected_row = next(row for row in body["rows"] if row["source_id"] == selected["id"])
        self.assertEqual(selected_row["prototype_score"], 1.0)

    def test_mini_lab_run_requires_generated_experiment_run_id(self):
        payload = self.indexed_starter_payload(_valid_starter_code())

        response = self.client.post("/api/mini-lab/run", json=payload)

        self.assertEqual(response.status_code, 403)
        self.assertIn("experiment run id", response.json()["detail"])

    def test_mini_lab_run_rejects_code_that_differs_from_generated_run(self):
        payload = self.authorized_mini_lab_payload(self.indexed_starter_payload(_valid_starter_code()))
        payload["code"] = payload["code"] + "\n# client-side mutation\n"

        response = self.client.post("/api/mini-lab/run", json=payload)

        self.assertEqual(response.status_code, 403)
        self.assertIn("starter code", response.json()["detail"])

    def test_mini_lab_run_accepts_free_text_fragment_from_indexed_span(self):
        source = PaperSource(
            title="Mini Lab Free Selection Paper",
            authors="A. Author",
            source_label="manual-mini-lab-free-selection",
            text=(
                "First sentence introduces a compact evidence mechanism. "
                "Second sentence says the mechanism improves precision in indexed source evidence. "
                "Third sentence warns that the result may not generalize."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        selected = document["sections"][0]["paragraphs"][0]["spans"][1]
        selected_fragment = "mechanism improves precision"
        code = _valid_starter_code()
        payload = self.authorized_mini_lab_payload(
            {
                "code": code,
                "paper_id": document["id"],
                "paper_title": document["title"],
                "span_id": selected["id"],
                "selected_span": selected_fragment,
            }
        )

        response = self.client.post("/api/mini-lab/run", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["passed"])
        self.assertEqual(body["sourceHash"], text_hash(selected_fragment))
        self.assertEqual(body["selectedSpanHash"], text_hash(selected_fragment))
        self.assertTrue(body["validation"]["sourceHashMatches"])

    def test_mini_lab_run_rejects_selected_span_mismatch(self):
        source = PaperSource(
            title="Mini Lab Mismatch Paper",
            authors="A. Author",
            source_label="manual-mini-lab-mismatch",
            text=(
                "First sentence defines the setup. "
                "Second sentence is the indexed span for the mini lab. "
                "Third sentence gives a limitation."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        selected = document["sections"][0]["paragraphs"][0]["spans"][1]
        payload = self.authorized_mini_lab_payload(
            {
                "code": _valid_starter_code(),
                "paper_id": document["id"],
                "paper_title": document["title"],
                "span_id": selected["id"],
                "selected_span": selected["original"],
            }
        )
        payload["selected_span"] = "A different client-side sentence should not be accepted."

        response = self.client.post("/api/mini-lab/run", json=payload)

        self.assertEqual(response.status_code, 403)
        self.assertIn("does not match", response.json()["detail"])

    def test_mini_lab_modal_result_must_match_code_hash(self):
        os.environ["PAPERLENS_MINILAB_PROVIDER"] = "modal"
        source = PaperSource(
            title="Mini Lab Modal Paper",
            authors="A. Author",
            source_label="manual-mini-lab-modal",
            text=(
                "First sentence defines the setup. "
                "Second sentence is selected for remote Modal execution. "
                "Third sentence gives a limitation."
            ),
        )
        document = paper_document_from_source(source, max_reader_spans=12)
        selected = document["sections"][0]["paragraphs"][0]["spans"][1]
        code = _valid_starter_code()

        def fake_modal(job):
            selected_row = next(row for row in job["evidenceRows"] if row["source_id"] == job["spanId"])
            return {
                "provider": "modal",
                "executionMode": "modal-remote-function",
                "runner": "paperlens-modal-minilab",
                "paperId": job["paperId"],
                "paperTitle": job["paperTitle"],
                "spanId": job["spanId"],
                "sourceHash": job["sourceHash"],
                "selectedSpanHash": job["selectedSpanHash"],
                "codeHash": "wrong-code-hash",
                "evidenceHash": job["evidenceHash"],
                "sourceIndexBound": True,
                "passed": True,
                "reasons": [],
                "rows": [
                    {
                        "source_id": selected_row["source_id"],
                        "text_hash": selected_row["text_hash"],
                        "baseline_score": 0.0,
                        "prototype_score": 1.0,
                        "metric": "source evidence score",
                        "failure_condition": False,
                    }
                ],
                "logs": ["fake modal"],
            }

        with patch("paperlens_lab.mini_lab._run_modal_mini_lab_with_cli", side_effect=fake_modal):
            payload = self.authorized_mini_lab_payload(
                {
                    "code": code,
                    "paper_id": document["id"],
                    "paper_title": document["title"],
                    "span_id": selected["id"],
                    "selected_span": selected["original"],
                }
            )
            response = self.client.post(
                "/api/mini-lab/run",
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["passed"])
        self.assertFalse(body["validation"]["codeHashMatches"])
        self.assertIn("mini-lab validation failed: codeHashMatches", body["reasons"])


def _valid_starter_code() -> str:
    return """
def baseline(example):
    return example["input"]

def paper_inspired(example):
    return example["input"] + " paper"

def score(output, expected):
    return 1.0 if expected in output else 0.0

def run(evidence_rows=None):
    examples = evidence_rows or []
    rows = []
    for example in examples:
        row = {"input": example["text"], "expected": "paper" if example.get("gold") else "control"}
        base = baseline(row)
        proto = paper_inspired(row)
        baseline_score = score(base, row["expected"])
        prototype_score = score(proto, row["expected"])
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


if __name__ == "__main__":
    unittest.main()
