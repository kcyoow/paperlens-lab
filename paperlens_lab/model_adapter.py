from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

from .ingest import clean_text
from .prompts import experiment_prompt, growth_prompt, qa_prompt, translation_prompt
from .tracing import TraceRecord, new_trace_id, trace_content_enabled, write_trace


DEFAULT_MODEL = os.getenv("PAPERLENS_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
DEFAULT_PROVIDER = os.getenv("PAPERLENS_PROVIDER", "fallback")
QUALITY_MODEL = os.getenv("PAPERLENS_QUALITY_MODEL", DEFAULT_MODEL)


@dataclass
class ModelResult:
    task: str
    text: str
    data: dict[str, Any]
    provider: str
    model: str
    trace_id: str
    error: str | None = None
    raw: str = ""
    used_fallback: bool = False


@dataclass
class ModelGateway:
    provider: str = DEFAULT_PROVIDER
    model_id: str = DEFAULT_MODEL
    quality_model_id: str = QUALITY_MODEL
    call_model: Callable[[str, str, int], str | None] | None = None
    trace_inputs: bool = field(default_factory=trace_content_enabled)
    trace_outputs: bool = field(default_factory=trace_content_enabled)
    last_errors: list[str] = field(default_factory=list)

    def translate_spans(
        self,
        title: str,
        spans: list[dict[str, str]],
        locale: str = "ko",
        use_model: bool = False,
    ) -> ModelResult:
        task = "translation"
        prompt = translation_prompt(title, spans, locale)

        def fallback() -> tuple[str, dict[str, Any]]:
            translations = []
            for item in spans:
                source = clean_text(item.get("text", ""))
                terms = _extract_terms(source, limit=5)
                translations.append(
                    {
                        "span_id": item.get("span_id", ""),
                        "translation": _fallback_translate(source, locale),
                        "preserved_terms": terms,
                        "uncertain_phrases": [],
                    }
                )
            data = {
                "translations": translations,
                "notes": ["fallback translation keeps source wording visible until a small model is enabled."],
            }
            return _format_translation_text(data), data

        return self._run(task, prompt, fallback, use_model=use_model, max_new_tokens=900)

    def answer_span(
        self,
        paper_title: str,
        span_id: str,
        selected_span: str,
        translated_span: str,
        question: str,
        source_text: str,
        locale: str,
        use_model: bool = False,
    ) -> ModelResult:
        task = "grounded_qa"
        evidence = _evidence_items(source_text or selected_span, selected_span=selected_span, span_id=span_id)
        prompt = qa_prompt(
            paper_title,
            span_id,
            selected_span,
            translated_span,
            question,
            evidence,
            locale,
        )

        def fallback() -> tuple[str, dict[str, Any]]:
            answer = (
                f"선택한 문장 `{span_id}`는 논문의 직접 근거로 보면 \"{selected_span[:220]}\"입니다. "
                f"질문 \"{question}\"에 대해서는 이 문장이 주장하는 범위 안에서만 답할 수 있고, "
                "더 넓은 성능 비교나 원인 설명은 주변 실험/방법 문단이 더 필요합니다."
                if locale == "ko"
                else (
                    f"Selected span `{span_id}` says: \"{selected_span[:220]}\". "
                    f"For \"{question}\", PaperLens can answer only inside that evidence; broader causes or comparisons need more context."
                )
            )
            data = {
                "answer": answer,
                "evidence": [{"source_id": span_id, "quote": selected_span[:420]}],
                "confidence": "medium",
                "needs_more_context": len(source_text.split()) < 60,
                "unsupported_assumptions": [],
            }
            return answer, data

        return self._run(task, prompt, fallback, use_model=use_model, max_new_tokens=700)

    def experiment_spec(
        self,
        paper_title: str,
        selected_span: str,
        translated_span: str,
        source_text: str,
        idea: str,
        locale: str,
        use_model: bool = False,
    ) -> ModelResult:
        task = "experiment_spec"
        evidence = _evidence_items(source_text or selected_span, selected_span=selected_span)
        prompt = experiment_prompt(paper_title, selected_span, translated_span, evidence, idea, locale)

        def fallback() -> tuple[str, dict[str, Any]]:
            terms = _extract_terms(source_text or selected_span, limit=6)
            top_term = terms[0] if terms else "the selected method"
            data = {
                "research_question": f"Does a tiny prototype of {top_term} improve a measurable behavior?",
                "mini_lab_goal": "Build a 30-minute toy comparison between a direct baseline and one paper-inspired variant.",
                "dataset": {
                    "name": "Hand-built 10-20 example toy set",
                    "fallback": "Use 5 examples copied or paraphrased from the selected paper span.",
                },
                "baseline": "Direct heuristic or prompt without the paper-inspired component.",
                "metric": "Exact match, pairwise preference, error count, or top-k precision depending on the task.",
                "steps": [
                    "Create a tiny input/output table.",
                    "Run the baseline.",
                    "Add the smallest paper-inspired operation.",
                    "Compare metric and inspect failures.",
                ],
                "ablation": "Remove the paper-inspired operation and keep all other inputs fixed.",
                "failure_condition": "The variant does not improve the metric or only improves by changing the task.",
                "expected_result": "The prototype may reveal whether the idea is visible at toy scale, not whether the full paper is reproduced.",
                "faithfulness_notes": [
                    "This mini-lab is a learning proxy, not a claim that the original paper result was reproduced.",
                    "Use source spans as constraints when explaining any result.",
                ],
                "starter_code_plan": ["baseline(example)", "paper_inspired(example)", "score(output, expected)", "run()"],
                "support_span_ids": [item["source_id"] for item in evidence[:3]],
            }
            return format_experiment_spec(data), data

        return self._run(task, prompt, fallback, use_model=use_model, max_new_tokens=950, quality=True)

    def growth_ideas(
        self,
        paper_title: str,
        paper_memory: list[dict[str, Any]],
        mini_lab_result: str,
        selected_span: str,
        locale: str,
        use_model: bool = False,
    ) -> ModelResult:
        task = "research_growth"
        prompt = growth_prompt(paper_title, paper_memory, mini_lab_result, selected_span, locale)

        def fallback() -> tuple[str, dict[str, Any]]:
            memories = paper_memory or [{"id": "paper:s1", "summary": selected_span[:220]}]
            first_memory = memories[0]
            evidence_id = str(first_memory.get("id", "paper:s1"))
            data = {
                "ideas": [
                    {
                        "idea": "Turn the selected claim into a smaller ablation that changes only one variable.",
                        "source_evidence": [evidence_id, "run:r1"],
                        "novelty_angle": "Use the mini-lab failure/success pattern as a lens for a narrower next test.",
                        "testable_next_step": "Run the same toy dataset with one component removed and compare the failure tags.",
                        "risk": "The toy result may be too small to generalize beyond the learning exercise.",
                    },
                    {
                        "idea": "Compare whether the paper-inspired component helps more on hard examples than easy examples.",
                        "source_evidence": [evidence_id, "run:r1"],
                        "novelty_angle": "This checks when the idea matters instead of only whether it helps on average.",
                        "testable_next_step": "Split 10 examples into easy/hard buckets and report metric deltas separately.",
                        "risk": "Manual difficulty labels can bias the interpretation.",
                    },
                ],
                "fine_tuning_signal": "maybe" if "json" in mini_lab_result.lower() else "none",
                "reason": "Fine-tune only after repeated schema/style failures survive prompt and retrieval fixes.",
            }
            return format_growth_ideas(data), data

        return self._run(task, prompt, fallback, use_model=use_model, max_new_tokens=900, quality=True)

    def _run(
        self,
        task: str,
        prompt: str,
        fallback: Callable[[], tuple[str, dict[str, Any]]],
        *,
        use_model: bool,
        max_new_tokens: int,
        quality: bool = False,
    ) -> ModelResult:
        started = time.perf_counter()
        trace_id = new_trace_id(task[:2])
        provider = self.provider if use_model else "fallback"
        model_id = self.quality_model_id if quality and use_model else self.model_id
        raw = ""
        error = None
        used_fallback = not use_model

        if use_model:
            try:
                raw = self._call(prompt, model_id, max_new_tokens) or ""
            except Exception as exc:  # pragma: no cover - defensive fallback path
                error = f"{type(exc).__name__}: {exc}"
                self.last_errors.append(error)

        if raw:
            data = _parse_json_object(raw)
            if data:
                schema_errors = _validate_task_data(task, data)
                if schema_errors:
                    text, data = fallback()
                    used_fallback = True
                    error = f"invalid model output for {task}: {', '.join(schema_errors)}; fallback used"
                else:
                    text = _format_task_text(task, data)
            else:
                text, data = fallback()
                used_fallback = True
                error = f"model returned non-JSON output for {task}; fallback used"
        else:
            text, data = fallback()
            used_fallback = True
            if error is None and use_model:
                error = "model returned empty output; fallback used"

        latency_ms = int((time.perf_counter() - started) * 1000)
        result = ModelResult(
            task=task,
            text=text,
            data=data,
            provider=provider,
            model=model_id if use_model else "fallback-extractive",
            trace_id=trace_id,
            error=error,
            raw=raw,
            used_fallback=used_fallback,
        )
        write_trace(
            TraceRecord(
                schema_version="0.1",
                trace_id=trace_id,
                task=task,
                provider=result.provider,
                model=result.model,
                status="fallback" if result.used_fallback else "model",
                latency_ms=latency_ms,
                input=_trace_input(prompt, self.trace_inputs),
                output=_trace_output(result, self.trace_outputs),
                error=error,
            )
        )
        return result

    def _call(self, prompt: str, model_id: str, max_new_tokens: int) -> str | None:
        if self.call_model is not None:
            return self.call_model(prompt, model_id, max_new_tokens)
        if self.provider == "modal":
            return generate_with_modal(prompt, model_id=model_id, max_new_tokens=max_new_tokens)
        if self.provider == "hf":
            return generate_with_hf_inference(
                prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                raise_errors=True,
            )
        return None


def generate_with_hf_inference(
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    max_new_tokens: int = 420,
    raise_errors: bool = False,
) -> Optional[str]:
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        if raise_errors:
            raise
        return None

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or True
    first_error: Exception | None = None
    try:
        client = InferenceClient(model=model_id, token=token)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.15,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        first_error = exc
        try:
            client = InferenceClient(model=model_id, token=token)
            return client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.15,
                top_p=0.9,
                do_sample=False,
            ).strip()
        except Exception as exc:
            if raise_errors:
                raise RuntimeError(f"HF inference failed: chat={first_error}; text_generation={exc}") from exc
            return None


def generate_with_modal(
    prompt: str,
    model_id: str = QUALITY_MODEL,
    max_new_tokens: int = 700,
) -> Optional[str]:
    endpoint = os.getenv("PAPERLENS_MODAL_URL", "").strip()
    if not endpoint:
        return None
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.15,
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("PAPERLENS_MODAL_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    body = response.json()
    if "choices" in body:
        return body["choices"][0]["message"]["content"].strip()
    if "text" in body:
        return str(body["text"]).strip()
    if "output" in body:
        return str(body["output"]).strip()
    return None


def format_experiment_spec(data: dict[str, Any]) -> str:
    lines = [
        "# ExperimentSpec",
        "",
        f"**Research question:** {data.get('research_question', '')}",
        f"**Mini-lab goal:** {data.get('mini_lab_goal', '')}",
        f"**Dataset:** {_dataset_text(data.get('dataset'))}",
        f"**Baseline:** {data.get('baseline', '')}",
        f"**Metric:** {data.get('metric', '')}",
        f"**Ablation:** {data.get('ablation', '')}",
        f"**Failure condition:** {data.get('failure_condition', '')}",
        f"**Expected result:** {data.get('expected_result', '')}",
        "",
        "## Steps",
        *[f"- {item}" for item in data.get("steps", [])],
        "",
        "## Faithfulness Notes",
        *[f"- {item}" for item in data.get("faithfulness_notes", [])],
    ]
    return "\n".join(line for line in lines if line is not None).strip()


def format_growth_ideas(data: dict[str, Any]) -> str:
    lines = ["# Research Growth Ideas", ""]
    for idx, idea in enumerate(data.get("ideas", []), start=1):
        lines.extend(
            [
                f"## Idea {idx}: {idea.get('idea', '')}",
                f"- Evidence: {', '.join(idea.get('source_evidence', []))}",
                f"- Novelty angle: {idea.get('novelty_angle', '')}",
                f"- Next test: {idea.get('testable_next_step', '')}",
                f"- Risk: {idea.get('risk', '')}",
                "",
            ]
        )
    lines.append(f"Fine-tuning signal: {data.get('fine_tuning_signal', 'none')} - {data.get('reason', '')}")
    return "\n".join(lines).strip()


def _format_translation_text(data: dict[str, Any]) -> str:
    return "\n".join(
        f"{item.get('span_id', '')}: {item.get('translation', '')}"
        for item in data.get("translations", [])
    )


def _format_task_text(task: str, data: dict[str, Any]) -> str:
    if task == "translation":
        return _format_translation_text(data)
    if task == "grounded_qa":
        return str(data.get("answer") or data.get("answer_ko") or data)
    if task == "experiment_spec":
        return format_experiment_spec(data)
    if task == "research_growth":
        return format_growth_ideas(data)
    return json.dumps(data, ensure_ascii=False)


def _validate_task_data(task: str, data: dict[str, Any]) -> list[str]:
    if task == "translation":
        translations = data.get("translations")
        if not isinstance(translations, list) or not translations:
            return ["missing translations"]
        errors = []
        for idx, item in enumerate(translations, start=1):
            if not isinstance(item, dict):
                errors.append(f"translation {idx} is not an object")
                continue
            if not item.get("span_id"):
                errors.append(f"translation {idx} missing span_id")
            if not item.get("translation"):
                errors.append(f"translation {idx} missing translation")
        return errors
    if task == "grounded_qa":
        errors = []
        if not data.get("answer"):
            errors.append("missing answer")
        if data.get("confidence") not in {"high", "medium", "low"}:
            errors.append("missing confidence")
        evidence = data.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append("missing evidence")
        return errors
    if task == "experiment_spec":
        required = ("research_question", "mini_lab_goal", "dataset", "baseline", "metric", "steps")
        errors = [f"missing {key}" for key in required if not data.get(key)]
        if data.get("steps") and not isinstance(data["steps"], list):
            errors.append("steps must be a list")
        return errors
    if task == "research_growth":
        ideas = data.get("ideas")
        if not isinstance(ideas, list) or not ideas:
            return ["missing ideas"]
        errors = []
        for idx, idea in enumerate(ideas, start=1):
            if not isinstance(idea, dict):
                errors.append(f"idea {idx} is not an object")
                continue
            for key in ("idea", "source_evidence", "testable_next_step", "risk"):
                if not idea.get(key):
                    errors.append(f"idea {idx} missing {key}")
        return errors
    return []


def _trace_input(prompt: str, include_content: bool) -> dict[str, Any]:
    if include_content:
        return {"prompt": prompt, "content_logged": True}
    return {"prompt_chars": len(prompt), "content_logged": False}


def _trace_output(result: ModelResult, include_content: bool) -> dict[str, Any]:
    if include_content:
        return {"text": result.text[:4000], "data": result.data, "content_logged": True}
    return {
        "text_chars": len(result.text),
        "data_keys": sorted(str(key) for key in result.data.keys()),
        "used_fallback": result.used_fallback,
        "content_logged": False,
    }


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _fallback_translate(source: str, locale: str) -> str:
    if locale != "ko":
        return source
    if not source:
        return ""
    terms = _extract_terms(source, limit=4)
    term_hint = f" 핵심 용어: {', '.join(terms)}." if terms else ""
    return f"[초안 번역] {source}{term_hint}"


def _evidence_items(text: str, selected_span: str, span_id: str = "selected") -> list[dict[str, str]]:
    ranked = _top_sentences(text, limit=5)
    items = [{"source_id": span_id, "text": selected_span}]
    for sentence in ranked:
        sid = f"S{sentence['pid']}"
        if sentence["text"].strip() and sentence["text"].strip() != selected_span.strip():
            items.append({"source_id": sid, "text": sentence["text"]})
    return items[:6]


def _dataset_text(value: Any) -> str:
    if isinstance(value, dict):
        return f"{value.get('name', '')} / fallback: {value.get('fallback', '')}"
    return str(value or "")


def _extract_terms(text: str, limit: int = 8) -> list[str]:
    candidates = []
    candidates.extend(re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z]?[A-Za-z0-9]+){0,4}\b", text))
    candidates.extend(re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b", text))
    counts = Counter(
        clean_text(candidate).strip(" ,.;:")
        for candidate in candidates
        if 2 < len(clean_text(candidate)) < 55
    )
    return [term for term, _ in counts.most_common(limit)]


def _top_sentences(text: str, limit: int = 5) -> list[dict[str, Any]]:
    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", clean_text(text))
    sentences = [piece.strip() for piece in pieces if len(piece.strip()) > 24]
    scored = []
    for idx, sentence in enumerate(sentences, start=1):
        lower = sentence.lower()
        score = sum(
            1.5
            for word in (
                "propose",
                "show",
                "evaluate",
                "method",
                "baseline",
                "metric",
                "limitation",
                "experiment",
                "result",
            )
            if word in lower
        ) + min(len(sentence), 240) / 240
        scored.append({"pid": idx, "text": sentence, "score": score})
    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]
    return sorted(ranked, key=lambda item: item["pid"])
