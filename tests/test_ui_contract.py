import unittest

from fastapi.testclient import TestClient

from paperlens_lab.server import create_app


class ServiceUiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_health_does_not_mount_legacy_sample_ui(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["runtime"], "react-fastapi-service")
        self.assertIsNone(body["gradio_path"])

    def test_sample_paper_api_is_disabled_in_service_mode(self):
        response = self.client.get("/api/sample-paper")

        self.assertEqual(response.status_code, 404)
        self.assertIn("Load a real PDF or arXiv paper", response.json()["detail"])

    def test_empty_paper_input_is_rejected(self):
        response = self.client.post("/api/paper", json={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Add a PDF, arXiv URL/ID, or paper text", response.json()["detail"])

    def test_diagnostic_starter_runner_is_disabled_in_service_mode(self):
        response = self.client.post(
            "/api/starter/run",
            json={
                "code": "def run(evidence_rows=None):\n    return []\n",
                "paper_id": "arxiv-1706.03762",
                "paper_title": "Attention Is All You Need",
                "span_id": "P0.S4",
                "selected_span": "We propose a new simple network architecture, the Transformer.",
            },
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("bound mini-lab", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
