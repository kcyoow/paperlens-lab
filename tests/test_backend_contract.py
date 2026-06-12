import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from paperlens_lab.ingest import PaperSource
from paperlens_lab.server import create_app, paper_document_from_source
from paperlens_lab.source_index import load_source_index


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["PAPERLENS_TRACE_PATH"] = str(Path(self.tempdir.name) / "api_traces.jsonl")
        os.environ["PAPERLENS_MEMORY_PATH"] = str(Path(self.tempdir.name) / "paper_memory.jsonl")
        os.environ["PAPERLENS_SOURCE_INDEX_DIR"] = str(Path(self.tempdir.name) / "source_index")
        os.environ["PAPERLENS_TRANSLATION_CACHE_DIR"] = str(Path(self.tempdir.name) / "translation_cache")
        self.client = TestClient(create_app())

    def tearDown(self):
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
        os.environ.pop("PAPERLENS_MEMORY_PATH", None)
        os.environ.pop("PAPERLENS_SOURCE_INDEX_DIR", None)
        os.environ.pop("PAPERLENS_TRANSLATION_CACHE_DIR", None)
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
                    "original": selected["original"],
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
        evidence_items = gateway.answer_span.call_args.kwargs["evidence_items_override"]
        self.assertEqual({item["source_id"] for item in evidence_items}, {span["span_id"] for span in record["spans"]})

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

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["translation"], "세 번째 문장은 번역 캐시 검증을 위해 선택된다.")
        self.assertEqual(second.json()["status"], "cached")
        self.assertEqual(gateway.translate_spans.call_count, 1)

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
        experiment = self.client.post(
            "/api/experiment",
            json={
                "paper_title": "Evidence Reranking",
                "selected_span": "The method improves retrieval with evidence-linked reranking.",
                "source_text": "The method improves retrieval with evidence-linked reranking.",
                "idea": "Try evidence reranking",
                "locale": "ko",
            },
        )
        self.assertEqual(experiment.status_code, 200)
        exp_body = experiment.json()
        self.assertIn("card", exp_body)
        self.assertIn("starter", exp_body)
        self.assertIn("spec", exp_body)
        self.assertIn("metric", exp_body["spec"])

        growth = self.client.post(
            "/api/growth",
            json={
                "paper_title": "Evidence Reranking",
                "selected_span": "The method improves retrieval with evidence-linked reranking.",
                "paper_memory": [{"id": "paper:s1", "summary": "Evidence reranking may improve precision."}],
                "mini_lab_result": "run:r1 improved precision but failed on ambiguous examples.",
                "locale": "ko",
            },
        )
        self.assertEqual(growth.status_code, 200)
        growth_body = growth.json()
        self.assertGreaterEqual(len(growth_body["ideas"]), 1)
        self.assertIn("fineTuningSignal", growth_body)


if __name__ == "__main__":
    unittest.main()
