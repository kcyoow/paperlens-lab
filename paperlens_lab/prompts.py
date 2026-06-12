from __future__ import annotations

import json
from typing import Any


def translation_prompt(title: str, spans: list[dict[str, str]], locale: str = "ko") -> str:
    return f"""You are PaperLens Lab, helping a non-native undergraduate read an English research paper.
Translate each source span into Korean while preserving technical terms, variables, dataset names, numbers, citations, uncertainty, and result direction.
Return only valid JSON with this shape:
{{
  "translations": [
    {{"span_id": "P0.S1", "translation": "...", "preserved_terms": ["..."], "uncertain_phrases": []}}
  ],
  "notes": ["..."]
}}

Paper title: {title}
Target locale: {locale}
Source spans:
{json.dumps(spans, ensure_ascii=False, indent=2)}
"""


def qa_prompt(
    paper_title: str,
    span_id: str,
    selected_span: str,
    translated_span: str,
    question: str,
    evidence: list[dict[str, str]],
    locale: str,
) -> str:
    return f"""You are PaperLens Lab. Answer the student's question using only the selected span and supplied neighboring evidence.
Use Korean when locale is ko. Cite source IDs for every substantive claim. If the evidence is insufficient, say what is missing.
Return only valid JSON:
{{
  "answer": "...",
  "evidence": [{{"source_id": "{span_id}", "quote": "..."}}],
  "confidence": "high|medium|low",
  "needs_more_context": false,
  "unsupported_assumptions": []
}}

Paper: {paper_title}
Locale: {locale}
Selected span id: {span_id}
Selected span: {selected_span}
Available translation: {translated_span}
Question: {question}
Neighbor evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Rules:
- Avoid strong words such as proves, guarantees, SOTA, 증명, 입증, 최고 unless the evidence literally says that.
- Prefer "suggests", "shows within this evidence", or Korean equivalents such as "보여준다" / "시사한다".
- If the student's question asks for a broader claim than the evidence supports, set needs_more_context to true and confidence to low or medium.
"""


def experiment_prompt(
    paper_title: str,
    selected_span: str,
    translated_span: str,
    source_evidence: list[dict[str, str]],
    idea: str,
    locale: str,
) -> str:
    return f"""You are PaperLens Lab. Convert a highlighted paper span into a 30-60 minute undergraduate mini-lab.
Keep it faithful, runnable, small, and honest about limitations. Do not claim full paper reproduction.
Return only valid JSON:
{{
  "research_question": "...",
  "mini_lab_goal": "...",
  "dataset": {{"name": "...", "fallback": "..."}},
  "baseline": "...",
  "metric": "...",
  "steps": ["..."],
  "ablation": "...",
  "failure_condition": "...",
  "expected_result": "...",
  "faithfulness_notes": ["..."],
  "starter_code_plan": ["..."],
  "support_span_ids": ["..."]
}}

Paper: {paper_title}
Locale: {locale}
Student idea: {idea}
Selected span: {selected_span}
Available translation: {translated_span}
Evidence:
{json.dumps(source_evidence, ensure_ascii=False, indent=2)}

Rules:
- The dataset must be a public, toy, hand-built, or fallback dataset a student can use in 30-60 minutes.
- The failure_condition must explicitly name the metric and say what metric outcome would falsify the mini-lab.
- The expected_result must be modest; do not promise that the paper's original delta will reproduce.
- The ablation should isolate one variable.
"""


def growth_prompt(
    paper_title: str,
    paper_memory: list[dict[str, Any]],
    mini_lab_result: str,
    selected_span: str,
    locale: str,
) -> str:
    return f"""You are PaperLens Lab Research Growth Mode.
Given paper memories and a mini-lab result, generate next testable research ideas for a student.
Each idea must cite paper/result evidence, state novelty angle, and propose a low-cost next step.
Return only valid JSON:
{{
  "ideas": [
    {{
      "idea": "...",
      "source_evidence": ["paper:s1", "run:r1"],
      "novelty_angle": "...",
      "testable_next_step": "...",
      "risk": "..."
    }}
  ],
  "fine_tuning_signal": "none|maybe|recommended",
  "reason": "..."
}}

Paper: {paper_title}
Locale: {locale}
Selected span: {selected_span}
Mini-lab result evidence id: run:r1
Mini-lab result:
{mini_lab_result}
Paper memories:
{json.dumps(paper_memory, ensure_ascii=False, indent=2)}

Rules:
- Each idea must include at least one paper memory id and `run:r1` in source_evidence.
- Do not call fine-tuning recommended just because an idea is promising; use recommended only for repeated observed model-output failures.
- Keep each next step low-cost and directly testable.
"""
