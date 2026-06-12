import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from paperlens_lab.server import create_app


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["PAPERLENS_TRACE_PATH"] = str(Path(self.tempdir.name) / "api_traces.jsonl")
        self.client = TestClient(create_app())

    def tearDown(self):
        os.environ.pop("PAPERLENS_TRACE_PATH", None)
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
