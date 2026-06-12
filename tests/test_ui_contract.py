import unittest

from paperlens_lab.ui import EXAMPLE_TEXT, _effective_pasted_text


class GradioUiContractTests(unittest.TestCase):
    def test_arxiv_input_does_not_use_default_sample_text(self):
        self.assertEqual(_effective_pasted_text(None, "1706.03762", EXAMPLE_TEXT), "")

    def test_pdf_upload_does_not_use_default_sample_text(self):
        self.assertEqual(_effective_pasted_text("/tmp/paper.pdf", "", EXAMPLE_TEXT), "")

    def test_empty_inputs_keep_sample_text_for_demo_mode(self):
        self.assertEqual(_effective_pasted_text(None, "", ""), EXAMPLE_TEXT)


if __name__ == "__main__":
    unittest.main()
