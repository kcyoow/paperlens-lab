from __future__ import annotations

import math
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from typing import Any
from typing import Iterable

from .ingest import PaperSource, clean_text
from .model_adapter import DEFAULT_MODEL, generate_with_hf_inference


CLAIM_WORDS = {
    "propose",
    "introduce",
    "present",
    "show",
    "demonstrate",
    "improve",
    "achieve",
    "outperform",
    "evaluate",
    "train",
    "fine-tune",
    "benchmark",
    "dataset",
    "method",
    "model",
    "loss",
    "objective",
    "retrieval",
    "translation",
    "explanation",
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "can",
    "into",
    "their",
    "our",
    "using",
    "use",
    "used",
    "paper",
    "model",
    "models",
    "method",
    "results",
}


@dataclass
class Sentence:
    pid: int
    text: str
    score: float


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [piece.strip() for piece in pieces if len(piece.strip()) > 24]


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text.lower())


def score_sentences(text: str) -> list[Sentence]:
    raw_sentences = split_sentences(text)
    counts = Counter(token for token in words(text) if token not in STOPWORDS)
    if not raw_sentences:
        return []

    scored = []
    for idx, sentence in enumerate(raw_sentences, start=1):
        tokens = [token for token in words(sentence) if token not in STOPWORDS]
        lexical = sum(math.log(1 + counts[token]) for token in tokens[:60])
        claim_bonus = sum(1.4 for token in tokens if token in CLAIM_WORDS)
        length_penalty = 0.015 * max(0, len(sentence) - 260)
        score = lexical + claim_bonus - length_penalty
        scored.append(Sentence(pid=idx, text=sentence, score=score))
    return scored


def top_sentences(text: str, limit: int = 7) -> list[Sentence]:
    ranked = sorted(score_sentences(text), key=lambda item: item.score, reverse=True)
    return sorted(ranked[:limit], key=lambda item: item.pid)


def extract_terms(text: str, limit: int = 10) -> list[str]:
    candidates = []
    candidates.extend(re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z]?[A-Za-z0-9]+){0,4}\b", text))
    candidates.extend(re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b", text))
    counts = Counter(
        clean_text(candidate).strip(" ,.;:")
        for candidate in candidates
        if 2 < len(clean_text(candidate)) < 55
    )
    return [term for term, _ in counts.most_common(limit)]


def evidence_table(sentences: Iterable[Sentence]) -> str:
    rows = ["| Ref | Evidence |", "| --- | --- |"]
    for sentence in sentences:
        rows.append(f"| S{sentence.pid} | {sentence.text[:420]} |")
    return "\n".join(rows)


def make_claims(sentences: list[Sentence]) -> list[str]:
    claims = []
    for sentence in sentences:
        text = sentence.text
        if any(word in text.lower() for word in CLAIM_WORDS):
            claims.append(f"S{sentence.pid}: {text}")
    return claims[:5] or [f"S{sentence.pid}: {sentence.text}" for sentence in sentences[:3]]


def korean_reading_guide(source: PaperSource, audience: str, focus: str, use_model: bool) -> str:
    evidence = top_sentences(source.text, limit=6)
    terms = extract_terms(source.text, limit=8)
    claims = make_claims(evidence)

    prompt = f"""You are PaperLens Lab. Explain this research paper in Korean.
Audience: {audience}
Focus: {focus}
Rules:
- Separate paper claims from interpretation.
- Keep citations as S-number references.
- Do not add facts not supported by the evidence.

Title: {source.title}
Evidence:
{chr(10).join(claims)}

Terms: {", ".join(terms)}
"""
    model_text = generate_with_hf_inference(prompt) if use_model else None
    if model_text:
        return model_text

    bullets = [
        f"- 이 논문은 `{source.title}`에 대해 다룹니다.",
        f"- 읽는 대상: {audience}. 초점: {focus}.",
        "- 아래 요약은 입력 원문에서 점수가 높은 문장을 근거로 만든 extractive reading guide입니다.",
    ]
    claim_lines = [f"- {claim}" for claim in claims]
    term_lines = [f"- `{term}`: 논문 안에서 반복되거나 제목처럼 쓰이는 핵심 표현입니다." for term in terms[:6]]
    return "\n".join(
        [
            "### Korean Reading Guide",
            *bullets,
            "",
            "### Paper Claims",
            *claim_lines,
            "",
            "### Key Terms",
            *(term_lines or ["- 핵심 용어를 충분히 추출하지 못했습니다."]),
        ]
    )


def analyze_paper(
    source: PaperSource,
    audience: str,
    focus: str,
    use_model: bool,
) -> tuple[str, str, str, str]:
    evidence = top_sentences(source.text, limit=8)
    terms = extract_terms(source.text, limit=10)
    claims = make_claims(evidence)
    token_estimate = len(source.text.split())

    reader = korean_reading_guide(source, audience, focus, use_model)
    overview = textwrap.dedent(
        f"""
        # {source.title}

        **Source:** {source.source_label}  
        **Authors:** {source.authors or "Not detected"}  
        **Approx. words loaded:** {token_estimate:,}  
        **Model adapter:** {"enabled" if use_model else "fallback mode"}  
        **Target model:** `{DEFAULT_MODEL}`

        {reader}

        ### Faithfulness Notes
        - Paper claims are tied to evidence IDs such as S1 and S2.
        - Interpretation is kept separate from direct claims.
        - Full-paper translation quality depends on the model adapter; fallback mode is an extractive guide.
        """
    ).strip()

    evidence_md = evidence_table(evidence)
    claims_md = "\n".join(f"- {claim}" for claim in claims)
    terms_md = "\n".join(f"- {term}" for term in terms) or "- No strong terms detected."
    raw_extract = "\n\n".join(sentence.text for sentence in evidence)

    structured = textwrap.dedent(
        f"""
        ## Claims
        {claims_md}

        ## Terms
        {terms_md}
        """
    ).strip()
    return overview, evidence_md, structured, raw_extract


def experiment_card(source: PaperSource, idea: str, audience: str, use_model: bool) -> tuple[str, str]:
    evidence = top_sentences(source.text, limit=5)
    claims = make_claims(evidence)
    terms = extract_terms(source.text, limit=8)
    idea_text = clean_text(idea) or "Test the paper's core idea on a small public dataset."

    prompt = f"""Create a small reproducible experiment plan from a research paper.
Audience: {audience}
Idea: {idea_text}
Evidence:
{chr(10).join(claims)}
Terms: {", ".join(terms)}
Return: hypothesis, baseline, toy dataset, metric, steps, risks, and a short Python starter.
"""
    model_text = generate_with_hf_inference(prompt) if use_model else None
    if model_text:
        starter = starter_code(source.title, terms, idea_text)
        return model_text, starter

    top_term = terms[0] if terms else "paper method"
    card = textwrap.dedent(
        f"""
        # Experiment Card

        **Idea:** {idea_text}

        **Hypothesis:** If the paper's `{top_term}` idea is useful, a tiny prototype should improve one measurable behavior over a simple baseline.

        **Baseline:** Use a direct prompt or keyword heuristic without the paper-inspired component.

        **Prototype:** Add the smallest version of the paper-inspired component and run it on 10-50 examples.

        **Dataset:** Start with a hand-built mini set from the paper's examples, then move to a public benchmark once the behavior is visible.

        **Metric:** Use exact match, pairwise preference, latency, cost, or an error tag count depending on the task.

        **Evidence Links:** {", ".join(f"S{s.pid}" for s in evidence)}

        **Risks:**
        - The prototype may reproduce surface wording rather than the actual method.
        - A tiny dataset can make gains look larger than they are.
        - The paper may rely on training scale that is not available in a Space demo.

        **Next Step:** Replace the baseline function in the starter code with the smallest paper-inspired operation.
        """
    ).strip()
    return card, starter_code(source.title, terms, idea_text)


def starter_code(title: str, terms: list[str], idea: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_]+", "_", title.lower()).strip("_")[:40] or "paper_experiment"
    term_list = ", ".join(repr(term) for term in terms[:6])
    return textwrap.dedent(
        f"""
        \"\"\"Starter experiment for {title}.\"\"\"

        PAPER_TERMS = [{term_list}]


        def baseline(example: str) -> dict:
            return {{
                "prediction": example[:240],
                "notes": "Direct baseline. Replace with a real baseline for your task.",
            }}


        def paper_inspired(example: str) -> dict:
            # Idea under test: {idea[:180]}
            term_hits = [term for term in PAPER_TERMS if term.lower() in example.lower()]
            return {{
                "prediction": example[:240],
                "term_hits": term_hits,
                "notes": "Add the smallest version of the paper-inspired component here.",
            }}


        def score(output: dict, expected: str) -> float:
            prediction = output.get("prediction", "").lower()
            expected = expected.lower()
            return float(expected in prediction) if expected else 0.0


        def run():
            examples = [
                ("A short paper example goes here.", "expected phrase"),
            ]
            rows = []
            for example, expected in examples:
                base = baseline(example)
                proto = paper_inspired(example)
                rows.append({{
                    "example": example,
                    "baseline_score": score(base, expected),
                    "prototype_score": score(proto, expected),
                    "prototype_notes": proto,
                }})
            return rows


        if __name__ == "__main__":
            for row in run():
                print(row)
        """
    ).strip() + "\n"


def starter_code_from_spec(
    title: str,
    spec: dict[str, Any],
    *,
    selected_span: str = "",
) -> str:
    """Build a dependency-free starter that can be compiled and smoke-run."""

    safe_title = repr(title or "Untitled paper")
    question = repr(str(spec.get("research_question") or "Does the paper-inspired variant help?"))
    metric = repr(str(spec.get("metric") or "score"))
    baseline = repr(str(spec.get("baseline") or "Direct baseline"))
    ablation = repr(str(spec.get("ablation") or "Remove the paper-inspired component."))
    failure = repr(str(spec.get("failure_condition") or "Prototype score does not beat baseline score."))
    expected = repr(str(spec.get("expected_result") or "A modest measurable change, if any."))
    span = repr(selected_span or str(spec.get("mini_lab_goal") or "A short paper example goes here."))
    keywords = _starter_keywords(spec, selected_span)
    keyword_list = repr(keywords)
    dataset_name = ""
    dataset = spec.get("dataset")
    if isinstance(dataset, dict):
        dataset_name = str(dataset.get("name") or dataset.get("fallback") or "")
    else:
        dataset_name = str(dataset or "")
    dataset_repr = repr(dataset_name or "hand-built mini set")

    return textwrap.dedent(
        f"""
        \"\"\"Runnable PaperLens mini-lab starter.

        Paper: {title}
        This file is intentionally dependency-free so it can run in a clean Space,
        notebook, or local Python process before a student replaces the toy data.
        \"\"\"

        PAPER_TITLE = {safe_title}
        RESEARCH_QUESTION = {question}
        DATASET = {dataset_repr}
        METRIC = {metric}
        BASELINE = {baseline}
        ABLATION = {ablation}
        FAILURE_CONDITION = {failure}
        EXPECTED_RESULT = {expected}
        KEYWORDS = {keyword_list}


        EXAMPLES = [
            {{
                "id": "paper-span-1",
                "input": {span},
                "expected_terms": KEYWORDS[:3],
            }},
            {{
                "id": "contrast-1",
                "input": "A control example that omits most paper-specific cues.",
                "expected_terms": [],
            }},
        ]


        def baseline(example):
            text = example["input"].lower()
            hits = [term for term in KEYWORDS if term.lower() in text]
            return {{
                "method": "baseline",
                "prediction": "paper-related" if hits else "control",
                "hits": hits[:1],
            }}


        def paper_inspired(example):
            text = example["input"].lower()
            hits = [term for term in KEYWORDS if term.lower() in text]
            return {{
                "method": "paper_inspired",
                "prediction": "paper-related" if len(hits) >= 1 else "control",
                "hits": hits,
                "ablation": ABLATION,
            }}


        def score(output, expected_terms):
            if not expected_terms:
                return 1.0 if output["prediction"] == "control" else 0.0
            matched = set(term.lower() for term in output.get("hits", []))
            expected = set(term.lower() for term in expected_terms)
            return len(matched & expected) / max(1, len(expected))


        def run(examples=None):
            rows = []
            for example in examples or EXAMPLES:
                base = baseline(example)
                proto = paper_inspired(example)
                rows.append({{
                    "id": example["id"],
                    "baseline_score": score(base, example["expected_terms"]),
                    "prototype_score": score(proto, example["expected_terms"]),
                    "baseline": base,
                    "prototype": proto,
                    "metric": METRIC,
                    "failure_condition": FAILURE_CONDITION,
                }})
            return rows


        if __name__ == "__main__":
            import json

            print(json.dumps(run(), indent=2, ensure_ascii=False))
        """
    ).strip() + "\n"


def _starter_keywords(spec: dict[str, Any], selected_span: str) -> list[str]:
    values: list[str] = []
    for key in ("research_question", "mini_lab_goal", "baseline", "metric", "ablation", "expected_result"):
        values.append(str(spec.get(key) or ""))
    dataset = spec.get("dataset")
    if isinstance(dataset, dict):
        values.extend(str(item or "") for item in dataset.values())
    elif dataset:
        values.append(str(dataset))
    values.append(selected_span)
    text = " ".join(values)
    candidates = extract_terms(text, limit=8)
    if len(candidates) < 3:
        candidates.extend(token for token in words(text) if token not in STOPWORDS and len(token) > 4)
    deduped = []
    for candidate in candidates:
        cleaned = clean_text(candidate).strip(" ,.;:")
        if cleaned and cleaned.lower() not in {item.lower() for item in deduped}:
            deduped.append(cleaned)
    return deduped[:6] or ["paper", "baseline", "metric"]
