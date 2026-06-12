import unittest

from paperlens_lab.analysis import (
    analyze_paper,
    experiment_card,
    split_sentences,
    starter_code_from_spec,
    top_sentences,
)
from paperlens_lab.scenario_eval import evaluate_starter_code
from paperlens_lab.ingest import PaperSource


SAMPLE_TEXT = """
Title: Grounded Paper Explanation

Abstract: We propose a small research assistant that explains papers with cited evidence.
The method separates direct paper claims from interpretation and builds experiment cards.
We evaluate the assistant on a small benchmark of scientific abstracts and show fewer unsupported claims.
Limitations include short context windows and weak mathematical reasoning.
"""


class AnalysisTests(unittest.TestCase):
    def source(self):
        return PaperSource(
            title="Grounded Paper Explanation",
            authors="PaperLens Team",
            source_label="unit-test",
            text=SAMPLE_TEXT,
        )

    def test_sentence_splitting(self):
        sentences = split_sentences(SAMPLE_TEXT)
        self.assertGreaterEqual(len(sentences), 3)

    def test_top_sentences_prioritize_claims(self):
        ranked = top_sentences(SAMPLE_TEXT, limit=3)
        joined = " ".join(sentence.text for sentence in ranked)
        self.assertIn("propose", joined.lower())

    def test_analyze_paper_outputs_evidence(self):
        overview, evidence, structured, raw = analyze_paper(
            self.source(),
            audience="Practitioner",
            focus="Core idea",
            use_model=False,
        )
        self.assertIn("Grounded Paper Explanation", overview)
        self.assertIn("| Ref | Evidence |", evidence)
        self.assertIn("Claims", structured)
        self.assertTrue(raw.strip())

    def test_experiment_card_outputs_starter(self):
        card, code = experiment_card(
            self.source(),
            idea="test evidence-linked explanations",
            audience="Practitioner",
            use_model=False,
        )
        self.assertIn("Experiment Card", card)
        self.assertIn("def baseline", code)

    def test_starter_code_from_spec_smoke_runs(self):
        code = starter_code_from_spec(
            "Grounded Paper Explanation",
            {
                "research_question": "Do cited explanations reduce unsupported claims?",
                "mini_lab_goal": "Compare a baseline with evidence-linked explanations.",
                "dataset": {"name": "Toy abstracts", "fallback": "hand-built examples"},
                "baseline": "Plain explanation",
                "metric": "unsupported claim count",
                "ablation": "Remove evidence links.",
                "failure_condition": "Prototype score does not improve the metric.",
                "expected_result": "A small reduction in unsupported claims.",
            },
            selected_span="We evaluate the assistant on scientific abstracts.",
        )

        self.assertIn("def paper_inspired", code)
        self.assertTrue(evaluate_starter_code(code).passed)


if __name__ == "__main__":
    unittest.main()
