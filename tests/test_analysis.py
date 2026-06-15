import unittest

from paperlens_lab.analysis import (
    analyze_paper,
    experiment_card,
    split_sentences,
    starter_code_from_spec,
    top_sentences,
)
from paperlens_lab.scenario_eval import evaluate_starter_code, evaluate_starter_grounding
from paperlens_lab.ingest import PaperSource


SOURCE_EVIDENCE_ROWS = [
    {
        "source_id": "P0.S1",
        "text": "We evaluate the assistant on scientific abstracts and show fewer unsupported claims.",
        "text_hash": "row1",
        "label": "selected",
        "gold": True,
        "query": "We evaluate the assistant on scientific abstracts.",
    },
    {
        "source_id": "P0.S2",
        "text": "Limitations include short context windows and weak mathematical reasoning.",
        "text_hash": "row2",
        "label": "context_control",
        "gold": False,
        "query": "We evaluate the assistant on scientific abstracts.",
    },
]


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

    def test_starter_code_from_spec_source_run(self):
        code = starter_code_from_spec(
            "Grounded Paper Explanation",
            {
                "research_question": "Do cited explanations reduce unsupported claims?",
                "mini_lab_goal": "Compare a baseline with evidence-linked explanations.",
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "source-index spans"},
                "baseline": "Plain explanation",
                "metric": "unsupported claim count",
                "ablation": "Remove evidence links.",
                "failure_condition": "Prototype score does not improve the metric.",
                "expected_result": "A small reduction in unsupported claims.",
            },
            selected_span="We evaluate the assistant on scientific abstracts.",
        )

        self.assertIn("def paper_inspired", code)
        self.assertTrue(
            evaluate_starter_code(code, evidence_rows=SOURCE_EVIDENCE_ROWS, require_evidence_rows=True).passed
        )

    def test_attention_only_starter_source_run_and_stays_grounded(self):
        selected_span = (
            "We propose a new simple network architecture, the Transformer, based solely on "
            "attention mechanisms, dispensing with recurrence and convolutions entirely."
        )
        code = starter_code_from_spec(
            "Attention Is All You Need",
            {
                "research_question": "Can an attention-style global scorer recover the selected claim better than a local baseline?",
                "mini_lab_goal": "Compare a local baseline against an attention-style scorer on indexed paper evidence.",
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "selected span plus contrast spans"},
                "baseline": "Local or first-match heuristic without the attention-style bonus.",
                "metric": "label accuracy on indexed paper evidence",
                "ablation": "Remove only the attention-style global scoring bonus and keep everything else fixed.",
                "failure_condition": "The mini-lab fails if label accuracy on indexed paper evidence does not improve.",
                "expected_result": "A small directional gain on long-range or distractor-heavy examples.",
            },
            selected_span=selected_span,
        )

        rows = [
            {**SOURCE_EVIDENCE_ROWS[0], "text": selected_span, "query": selected_span},
            {
                **SOURCE_EVIDENCE_ROWS[1],
                "text": "A recurrent baseline processes tokens locally and can miss global dependencies.",
                "query": selected_span,
            },
        ]
        self.assertTrue(evaluate_starter_code(code, evidence_rows=rows, require_evidence_rows=True).passed)
        self.assertTrue(evaluate_starter_grounding(code, selected_span).passed)

    def test_attention_grounding_allows_ranked_candidate_prediction(self):
        selected_span = (
            "We propose a new simple network architecture, the Transformer, based solely on "
            "attention mechanisms, dispensing with recurrence and convolutions entirely."
        )
        code = """
def baseline(example):
    for candidate in example["candidates"][1:] + example["candidates"][:1]:
        if candidate in example["context"]:
            return candidate
    return example["candidates"][0]

def paper_inspired(example):
    scores = []
    for candidate in example["candidates"]:
        value = 1 if candidate == example["gold"] else 0
        scores.append((candidate, value))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[0][0]

def score(output, gold):
    return 1.0 if output == gold else 0.0

def run():
    examples = [
        {"query": "q1", "context": "attention removes recurrence and convolutions", "candidates": ["attention", "recurrence", "convolutions"], "gold": "attention", "mode": "central"},
        {"query": "q2", "context": "removes recurrence", "candidates": ["attention", "recurrence", "convolutions"], "gold": "recurrence", "mode": "removed"},
        {"query": "q3", "context": "late attention cue", "candidates": ["attention", "recurrence", "convolutions"], "gold": "attention", "mode": "global"},
    ]
    rows = []
    for example in examples:
        base = baseline(example)
        proto = paper_inspired(example)
        rows.append({
            "baseline_score": score(base, example["gold"]),
            "prototype_score": score(proto, example["gold"]),
            "metric": "label accuracy on indexed paper evidence",
            "failure_condition": "prototype_score must beat baseline_score",
        })
    return rows
""".strip()

        self.assertTrue(evaluate_starter_grounding(code, selected_span).passed)

    def test_attention_grounding_rejects_context_free_first_candidate_baseline(self):
        selected_span = (
            "We propose a new simple network architecture, the Transformer, based solely on "
            "attention mechanisms, dispensing with recurrence and convolutions entirely."
        )
        code = """
def baseline(example):
    return example["candidates"][0]

def paper_inspired(example):
    return example["gold"]

def score(output, gold):
    return 1.0 if output == gold else 0.0

def run():
    examples = [
        {"query": "q1", "context": "attention removes recurrence and convolutions", "candidates": ["attention", "recurrence", "convolutions"], "gold": "attention", "mode": "central"},
        {"query": "q2", "context": "removes recurrence", "candidates": ["attention", "recurrence", "convolutions"], "gold": "recurrence", "mode": "removed"},
        {"query": "q3", "context": "late attention cue", "candidates": ["attention", "recurrence", "convolutions"], "gold": "attention", "mode": "global"},
    ]
    rows = []
    for example in examples:
        base = baseline(example)
        proto = paper_inspired(example)
        rows.append({
            "baseline_score": score(base, example["gold"]),
            "prototype_score": score(proto, example["gold"]),
            "metric": "label accuracy on indexed paper evidence",
            "failure_condition": "prototype_score must beat baseline_score",
        })
    return rows
""".strip()

        result = evaluate_starter_grounding(code, selected_span)
        self.assertFalse(result.passed)
        self.assertIn("starter baseline is a trivial first-candidate selector", result.reasons)

    def test_starter_source_run_allows_safe_any_builtin(self):
        code = """
def baseline(example):
    for candidate in example["candidates"][1:] + example["candidates"][:1]:
        if candidate in example["context"]:
            return candidate
    return example["candidates"][0]

def paper_inspired(example):
    return example["gold"]

def score(output, gold):
    return 1.0 if output == gold else 0.0

def run(evidence_rows=None):
    examples = evidence_rows or []
    rows = []
    for example in examples:
        example = {
            **example,
            "context": example["text"],
            "candidates": ["attention", "recurrence"],
        }
        base = baseline(example)
        proto = paper_inspired(example)
        expected = "attention" if example.get("gold") else "recurrence"
        baseline_score = score(base, expected)
        prototype_score = score(proto, expected)
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
""".strip()

        rows = [
            {
                "source_id": "P0.S1",
                "text": "The Transformer uses attention mechanisms instead of recurrence.",
                "text_hash": "row1",
                "label": "selected",
                "gold": True,
                "query": "attention mechanisms",
            },
            {
                "source_id": "P0.S2",
                "text": "The recurrent baseline processes tokens in order.",
                "text_hash": "row2",
                "label": "context_control",
                "gold": False,
                "query": "attention mechanisms",
            },
        ]
        self.assertTrue(evaluate_starter_code(code, evidence_rows=rows, require_evidence_rows=True).passed)


if __name__ == "__main__":
    unittest.main()
