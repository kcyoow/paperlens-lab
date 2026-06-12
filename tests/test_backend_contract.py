import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from paperlens_lab.ingest import PaperSource
from paperlens_lab.server import create_app


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["PAPERLENS_TRACE_PATH"] = str(Path(self.tempdir.name) / "api_traces.jsonl")
        os.environ["PAPERLENS_MEMORY_PATH"] = str(Path(self.tempdir.name) / "paper_memory.jsonl")
        self.client = TestClient(create_app())

    def tearDown(self):
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
        os.environ.pop("PAPERLENS_MEMORY_PATH", None)
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
