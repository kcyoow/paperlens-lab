from __future__ import annotations

import json
import hashlib
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

from .ingest import clean_text
from .gpu_lab import _validate_gpu_script_contract
from .prompts import (
    evidence_probe_prompt,
    experiment_candidates_prompt,
    experiment_candidates_repair_prompt,
    experiment_prompt,
    experiment_repair_prompt,
    gpu_script_prompt,
    gpu_script_repair_prompt,
    growth_prompt,
    growth_repair_prompt,
    qa_prompt,
    starter_code_repair_prompt,
    starter_code_prompt,
    translation_prompt,
    translation_repair_prompt,
)
from .scenario_eval import (
    evaluate_experiment_spec,
    evaluate_growth_ideas,
    evaluate_starter_code,
    evaluate_starter_grounding,
    experiment_heavy_terms,
)
from .tracing import TraceRecord, new_trace_id, trace_content_enabled, write_trace


DEFAULT_SMALL_MULTILINGUAL_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_MODEL = os.getenv("PAPERLENS_MODEL", DEFAULT_SMALL_MULTILINGUAL_MODEL)
DEFAULT_PROVIDER = os.getenv("PAPERLENS_PROVIDER", "hf")
QUALITY_MODEL = os.getenv("PAPERLENS_QUALITY_MODEL", DEFAULT_MODEL)
TRANSLATION_MODEL = os.getenv("PAPERLENS_TRANSLATION_MODEL", os.getenv("PAPERLENS_MODEL", DEFAULT_SMALL_MULTILINGUAL_MODEL))
STRICT_MODEL_PROOF_TASKS = {"experiment_candidates", "gpu_script"}
REPRODUCTION_LEVELS = {"probe", "exact"}


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default
    return max(1, value)


def _normalize_reproduction_level(value: str, default: str = "probe") -> str:
    level = clean_text(value).lower().replace(" ", "_").replace("-", "_")
    return level if level in REPRODUCTION_LEVELS else default


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
    translation_model_id: str = TRANSLATION_MODEL
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

        return self._run(
            task,
            prompt,
            fallback,
            use_model=use_model,
            max_new_tokens=900,
            model_id_override=self.translation_model_id,
            context={"spans": spans, "locale": locale},
        )

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
        evidence_items_override: list[dict[str, str]] | None = None,
    ) -> ModelResult:
        task = "grounded_qa"
        evidence = evidence_items_override or _evidence_items(
            source_text or selected_span,
            selected_span=selected_span,
            span_id=span_id,
        )
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

    def answer_evidence_probe(
        self,
        paper_title: str,
        question: str,
        target_span_id: str,
        target_phrase: str,
        evidence_items: list[dict[str, str]],
        locale: str,
        use_model: bool = False,
    ) -> ModelResult:
        task = "adversarial_grounded_qa"
        prompt = evidence_probe_prompt(paper_title, question, target_phrase, evidence_items, locale)
        target_item = next(
            (item for item in evidence_items if item.get("source_id") == target_span_id),
            {"source_id": target_span_id, "text": target_phrase},
        )

        def fallback() -> tuple[str, dict[str, Any]]:
            quote = clean_text(str(target_item.get("text", "")))[:420]
            answer = (
                f"긴 근거 묶음 안에서 `{target_span_id}`가 exact phrase를 포함한다. "
                "이 근거만으로는 더 넓은 논문 전체 결론이나 fine-tuning 필요성까지 단정할 수 없다."
                if locale == "ko"
                else (
                    f"In the long evidence packet, `{target_span_id}` contains the exact phrase. "
                    "This evidence alone does not justify a broader full-paper conclusion or a fine-tuning decision."
                )
            )
            data = {
                "answer": answer,
                "evidence": [{"source_id": target_span_id, "quote": quote}],
                "confidence": "medium",
                "needs_more_context": True,
                "unsupported_assumptions": ["broader paper-level claims need more evidence"],
            }
            return answer, data

        return self._run(task, prompt, fallback, use_model=use_model, max_new_tokens=800)

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
        implementation_links = extract_implementation_links(source_text or selected_span)
        prompt = experiment_prompt(
            paper_title,
            selected_span,
            translated_span,
            evidence,
            idea,
            locale,
            implementation_links=implementation_links,
        )

        def fallback() -> tuple[str, dict[str, Any]]:
            focus_label = _experiment_focus_label(source_text or selected_span)
            metric = _experiment_metric_hint(source_text or selected_span)
            if _is_attention_mechanism_span(source_text or selected_span):
                data = {
                    "research_question": (
                        f"Can a source-bound {focus_label} run recover the selected claim better than a local sequential baseline?"
                    ),
                    "mini_lab_goal": (
                        "Compare a local-first baseline against an attention-style global scorer on 6-10 "
                        "indexed evidence rows around the selected span."
                    ),
                    "dataset": {
                        "name": "Indexed PaperLens evidence window",
                        "source": "Require source-index spans around the selected paper evidence.",
                    },
                    "baseline": "Use a local or first-match heuristic that cannot aggregate the whole sequence.",
                    "metric": metric,
                    "steps": [
                        "Load the indexed evidence rows around the selected span.",
                        "Run a local baseline that chooses the first or nearest matching candidate.",
                        "Run an attention-style global scoring rule over the full context while keeping the examples fixed.",
                        "Compare the metric and inspect where the global scorer helps or fails.",
                    ],
                    "ablation": (
                        "Remove only the global attention-style scoring bonus and keep the examples, candidates, "
                        "and metric fixed."
                    ),
                    "failure_condition": (
                        f"The mini-lab fails if {metric} does not improve after adding the attention-style global scorer."
                    ),
                    "expected_result": (
                        "A directional difference may appear on long-range or distractor-heavy evidence rows, but this does not "
                        "reproduce the original Transformer benchmark."
                    ),
                    "faithfulness_notes": [
                        "This mini-lab tests the selected evidence window, not the full paper training setup.",
                        "Interpret results only as directional evidence for the highlighted mechanism.",
                    ],
                    "implementation_repositories": implementation_links,
                    "starter_code_plan": [
                        "baseline(example)",
                        "paper_inspired(example)",
                        "score(output, gold)",
                        "run()",
                    ],
                    "support_span_ids": [item["source_id"] for item in evidence[:3]] or ["selected"],
                }
                return format_experiment_spec(data), data

            data = {
                "research_question": f"Can a source-bound {focus_label} run improve a measurable behavior on indexed paper evidence?",
                "mini_lab_goal": "Compare a direct baseline and one paper-inspired variant over the selected paper evidence window.",
                "dataset": {
                    "name": "Indexed PaperLens evidence window",
                    "source": "Require source-index spans around the selected paper evidence.",
                },
                "baseline": "Direct heuristic or prompt without the paper-inspired component.",
                "metric": metric,
                "steps": [
                    "Load the indexed source spans around the selected evidence.",
                    "Run the baseline.",
                    "Add the smallest paper-inspired operation.",
                    "Compare metric and inspect failures.",
                ],
                "ablation": "Remove only the paper-inspired operation and keep all other inputs fixed.",
                "failure_condition": f"The mini-lab fails if {metric} does not improve or only improves by changing the task.",
                "expected_result": "The run may reveal whether the idea is visible in the selected evidence window, not whether the full paper is reproduced.",
                "faithfulness_notes": [
                    "This mini-lab is a learning proxy, not a claim that the original paper result was reproduced.",
                    "Use source spans as constraints when explaining any result.",
                ],
                "implementation_repositories": implementation_links,
                "starter_code_plan": ["baseline(example)", "paper_inspired(example)", "score(output, expected)", "run()"],
                "support_span_ids": [item["source_id"] for item in evidence[:3]] or ["selected"],
            }
            return format_experiment_spec(data), data

        return self._run(
            task,
            prompt,
            fallback,
            use_model=use_model,
            max_new_tokens=950,
            context={
                "selected_span": selected_span,
                "locale": locale,
                "implementation_links": implementation_links,
            },
        )

    def experiment_candidates(
        self,
        paper_title: str,
        selected_span: str,
        translated_span: str,
        source_text: str,
        question: str,
        locale: str,
        reproduction_level: str = "probe",
        use_model: bool = False,
    ) -> ModelResult:
        task = "experiment_candidates"
        reproduction_level = _normalize_reproduction_level(reproduction_level)
        evidence = _evidence_items(source_text or selected_span, selected_span=selected_span)
        implementation_links = extract_implementation_links(source_text or selected_span)
        prompt = experiment_candidates_prompt(
            paper_title,
            selected_span,
            translated_span,
            evidence,
            question,
            reproduction_level,
            locale,
            implementation_links=implementation_links,
        )

        def fallback() -> tuple[str, dict[str, Any]]:
            candidate = {
                "id": "source-bound-probe",
                "title": "Source-bound evidence probe",
                "kind": "source_bound_probe",
                "reproduction_level": "probe",
                "faithfulness": {
                    "level": "probe",
                    "summary": "Fallback only; not product proof.",
                    "why_not_exact": "Fallback candidates cannot establish exact paper reproduction.",
                    "paper_targets": [],
                    "resource_note": "Fallback path.",
                },
                "is_recommended": True,
                "recommendation_reason": "Fallback only; live model candidates are required for service proof.",
                "hypothesis": "Check whether the selected paper idea creates a measurable signal in indexed evidence rows.",
                "paper_evidence_ids": [item["source_id"] for item in evidence[:2]] or ["selected"],
                "paper_evidence_quotes": [item["text"][:240] for item in evidence[:2]],
                "dataset": {"name": "Indexed PaperLens evidence window", "source": "PaperLens source index", "requires_download": False},
                "implementation": {"type": "source_bound_probe", "repo_url": "", "reason": "No source-listed implementation was used."},
                "run_plan": {"repo_url": "", "config_path": "", "command": "", "dataset": "Indexed PaperLens evidence window", "expected_artifact": "source-bound score rows"},
                "why_not_exact": "Fallback candidates cannot establish exact paper reproduction.",
                "gpu_required": False,
                "estimated_runtime_minutes": 1,
                "expected_metric": "source-bound directional score",
                "limitations": ["Fallback candidates cannot be used as product proof."],
                "approval_question": "Approve this source-bound probe?",
            }
            data = {"candidates": [candidate], "recommended_candidate_id": candidate["id"]}
            return format_experiment_candidates(data), data

        return self._run(
            task,
            prompt,
            fallback,
            use_model=use_model,
            max_new_tokens=1500,
            quality=True,
            context={
                "source_evidence": evidence,
                "locale": locale,
                "implementation_links": implementation_links,
                "reproduction_level": reproduction_level,
            },
        )

    def gpu_script(
        self,
        paper_title: str,
        selected_span: str,
        source_text: str,
        candidate: dict[str, Any],
        locale: str,
        implementation_repo_manifests: list[dict[str, Any]] | None = None,
        use_model: bool = False,
    ) -> ModelResult:
        task = "gpu_script"
        reproduction_level = _normalize_reproduction_level(str(candidate.get("reproduction_level") or "probe"))
        evidence = _evidence_items(source_text or selected_span, selected_span=selected_span)
        prompt = gpu_script_prompt(
            paper_title,
            selected_span,
            evidence,
            candidate,
            reproduction_level,
            locale,
            implementation_repo_manifests=implementation_repo_manifests,
        )

        def fallback() -> tuple[str, dict[str, Any]]:
            data = {
                "script": "",
                "entrypoint": "run_paperlens_gpu_probe",
                "dependencies": [],
                "hardware": "T4",
                "dataset": {"name": "", "source": ""},
                "reproduction_level": reproduction_level,
                "reproduction_plan": {"level": reproduction_level, "repo_url": "", "config_path": "", "command": "", "dataset": "", "expected_artifact": "", "faithfulness_note": ""},
                "expected_outputs": [],
                "paper_claim_comparison_plan": "",
                "limitations": ["Fallback GPU scripts cannot be used as product proof."],
            }
            return "", data

        return self._run(
            task,
            prompt,
            fallback,
            use_model=use_model,
            max_new_tokens=_env_int("PAPERLENS_GPU_SCRIPT_MAX_NEW_TOKENS", 2100),
            quality=True,
            context={
                "selected_span": selected_span,
                "source_evidence": evidence,
                "candidate": candidate,
                "locale": locale,
                "reproduction_level": reproduction_level,
                "implementation_repo_manifests": implementation_repo_manifests or [],
            },
        )

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
            paper_memory_id = str(
                next((memory.get("id") for memory in memories if str(memory.get("id", "")).startswith("paper:")), "")
                or memories[0].get("id", "paper:s1")
            )
            prior_growth_id = str(
                next((memory.get("id") for memory in memories if str(memory.get("id", "")).startswith("growth_idea:")), "")
                or ""
            )
            evidence_ids = [paper_memory_id, "run:r1"]
            if prior_growth_id:
                evidence_ids.insert(1, prior_growth_id)
            data = {
                "ideas": [
                    {
                        "idea": "Turn the selected claim into a smaller ablation that changes only one variable.",
                        "source_evidence": evidence_ids,
                        "novelty_angle": "Use the mini-lab failure/success pattern as a lens for a narrower next test.",
                        "testable_next_step": "Run the same indexed evidence rows with one component removed and compare the failure tags.",
                        "risk": "The selected evidence window may be too narrow to generalize beyond the reading task.",
                    },
                    {
                        "idea": "Compare whether the paper-inspired component helps more on hard examples than easy examples.",
                        "source_evidence": evidence_ids,
                        "novelty_angle": "This checks when the idea matters instead of only whether it helps on average.",
                        "testable_next_step": "Split 10 examples into easy/hard buckets and report metric deltas separately.",
                        "risk": "Manual difficulty labels can bias the interpretation.",
                    },
                ],
                "fine_tuning_signal": "maybe" if "json" in mini_lab_result.lower() else "none",
                "reason": "Fine-tune only after repeated schema/style failures survive prompt and retrieval fixes.",
            }
            return format_growth_ideas(data), data

        return self._run(
            task,
            prompt,
            fallback,
            use_model=use_model,
            max_new_tokens=900,
            quality=True,
            context={"known_evidence_ids": _known_growth_evidence_ids(paper_memory), "locale": locale},
        )

    def starter_code(
        self,
        paper_title: str,
        selected_span: str,
        source_text: str,
        experiment_spec: dict[str, Any],
        locale: str,
        implementation_repo_manifests: list[dict[str, Any]] | None = None,
        use_model: bool = False,
    ) -> ModelResult:
        task = "starter_code"
        evidence = _evidence_items(source_text or selected_span, selected_span=selected_span)
        prompt = starter_code_prompt(
            paper_title,
            selected_span,
            evidence,
            experiment_spec,
            locale,
            implementation_repo_manifests=implementation_repo_manifests,
        )
        from .analysis import starter_code_from_spec

        fallback_code = starter_code_from_spec(
            paper_title,
            experiment_spec,
            selected_span=selected_span,
        )

        def fallback() -> tuple[str, dict[str, Any]]:
            data = {
                "code": fallback_code,
                "why_this_matches_span": "Fallback starter uses the selected span and experiment spec when model code generation is unavailable.",
                "limitations": [
                    "Fallback starter is a conservative source-bound scaffold rather than a paper-specific coded mechanism.",
                    "Replace the heuristic body with a tighter paper-grounded operation before claiming experiment realism.",
                ],
            }
            return fallback_code, data

        return self._run(
            task,
            prompt,
            fallback,
            use_model=use_model,
            max_new_tokens=1800,
            quality=True,
            context={
                "selected_span": selected_span,
                "source_evidence": evidence,
                "locale": locale,
                "implementation_repo_manifests": implementation_repo_manifests or [],
            },
        )

    def _run(
        self,
        task: str,
        prompt: str,
        fallback: Callable[[], tuple[str, dict[str, Any]]],
        *,
        use_model: bool,
        max_new_tokens: int,
        quality: bool = False,
        model_id_override: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ModelResult:
        started = time.perf_counter()
        trace_id = new_trace_id(task[:2])
        provider = self.provider if use_model else "fallback"
        model_id = model_id_override or (self.quality_model_id if quality and use_model else self.model_id)
        raw = ""
        error = None
        used_fallback = not use_model
        strict_model_proof = use_model and task in STRICT_MODEL_PROOF_TASKS

        if use_model:
            try:
                raw = self._call(prompt, model_id, max_new_tokens) or ""
            except Exception as exc:  # pragma: no cover - defensive fallback path
                error = f"{type(exc).__name__}: {exc}"
                self.last_errors.append(error)

        if raw:
            error, data = _materialize_task_output(task, raw, context=context)
            if error and use_model:
                repair_attempts = 3 if task in {"starter_code", "gpu_script"} else 2 if task == "research_growth" else 1
                latest_raw = raw
                latest_error = error
                latest_data = data
                while latest_error and repair_attempts > 0:
                    repair_prompt = _repair_task_output_prompt(task, latest_raw, latest_error, context=context)
                    if not repair_prompt:
                        break
                    try:
                        repaired_raw = self._call(repair_prompt, model_id, max_new_tokens) or ""
                    except Exception as exc:  # pragma: no cover - defensive fallback path
                        repaired_raw = ""
                        self.last_errors.append(f"{type(exc).__name__}: {exc}")
                    if not repaired_raw:
                        break
                    raw = repaired_raw
                    latest_raw = repaired_raw
                    latest_error, latest_data = _materialize_task_output(task, repaired_raw, context=context)
                    repair_attempts -= 1
                error = latest_error
                data = latest_data
            if error is None:
                used_fallback = False
                text = _format_task_text(task, data)
            elif strict_model_proof:
                text = ""
                used_fallback = False
            else:
                text, data = fallback()
                used_fallback = True
                error = f"{error}; fallback used"
        else:
            if strict_model_proof:
                text, data = "", {}
                used_fallback = False
                if error is None:
                    error = "model returned empty output"
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
        client = InferenceClient(
            model=model_id,
            token=token,
            timeout=_env_int("PAPERLENS_HF_TIMEOUT_SECONDS", 120),
        )
        chat_kwargs: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.15,
        }
        if _hf_json_mode_enabled() and "return only valid json" in prompt.lower():
            try:
                response = client.chat.completions.create(
                    **chat_kwargs,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                first_error = exc
        response = client.chat.completions.create(**chat_kwargs)
        return response.choices[0].message.content.strip()
    except Exception as exc:
        first_error = first_error or exc
        try:
            client = InferenceClient(
                model=model_id,
                token=token,
                timeout=_env_int("PAPERLENS_HF_TIMEOUT_SECONDS", 120),
            )
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


def _hf_json_mode_enabled() -> bool:
    return os.getenv("PAPERLENS_HF_JSON_MODE", "1").lower() not in {"0", "false", "no"}


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
    implementation_repositories = _format_implementation_repositories(data.get("implementation_repositories"))
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
        f"**Implementation repositories:** {implementation_repositories}" if implementation_repositories else None,
        "",
        "## Steps",
        *[f"- {item}" for item in data.get("steps", [])],
        "",
        "## Faithfulness Notes",
        *[f"- {item}" for item in data.get("faithfulness_notes", [])],
    ]
    return "\n".join(line for line in lines if line is not None).strip()


def format_experiment_candidates(data: dict[str, Any]) -> str:
    lines = ["# Experiment Candidates", ""]
    recommended_id = str(data.get("recommended_candidate_id") or "")
    for idx, candidate in enumerate(data.get("candidates", []), start=1):
        if not isinstance(candidate, dict):
            continue
        marker = " recommended" if candidate.get("id") == recommended_id or candidate.get("is_recommended") else ""
        lines.extend(
            [
                f"## {idx}. {candidate.get('title', '')}{marker}",
                f"- Kind: {candidate.get('kind', '')}",
                f"- Reproduction level: {candidate.get('reproduction_level', '')}",
                f"- Hypothesis: {candidate.get('hypothesis', '')}",
                f"- Evidence: {', '.join(str(item) for item in candidate.get('paper_evidence_ids', []))}",
                f"- Dataset: {_dataset_text(candidate.get('dataset'))}",
                f"- GPU required: {bool(candidate.get('gpu_required'))}",
                f"- Metric: {candidate.get('expected_metric', '')}",
                f"- Limitations: {'; '.join(str(item) for item in candidate.get('limitations', []))}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def format_gpu_script(data: dict[str, Any]) -> str:
    return str(data.get("script") or "")


def _format_implementation_repositories(value: Any) -> str:
    repos = _implementation_links_for_spec(value)
    if not repos:
        return ""
    return "; ".join(
        f"{item.get('url', '')} ({item.get('usage', 'optional inspected implementation path')})"
        for item in repos
        if item.get("url")
    )


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


def _known_growth_evidence_ids(paper_memory: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for memory in paper_memory:
        if not isinstance(memory, dict):
            continue
        for key in ("id", "span_id", "source_id"):
            value = str(memory.get(key) or "").strip()
            if value and value not in ids:
                ids.append(value)
    if "run:r1" not in ids:
        ids.append("run:r1")
    return ids


def extract_implementation_links(text: str, limit: int = 3) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"(?i)(?:https?://)?(?:www\.)?github\.com/"
        r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
        r"(?:/[A-Za-z0-9_./?=&%#:+-]*)?"
    )
    for match in pattern.finditer(text or ""):
        owner, repo = match.group(1), match.group(2)
        repo = repo.strip().strip(".,;:)】}>")
        owner = owner.strip().strip(".,;:)】}>")
        if not owner or not repo or repo.lower() in {"tree", "blob", "issues", "pulls"}:
            continue
        url = f"https://github.com/{owner}/{repo}"
        source_url = _normalize_github_source_url(match.group(0), owner=owner, repo=repo)
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        links.append(
            {
                "source_id": f"implementation:github:{len(links) + 1}",
                "url": url,
                "source_url": source_url,
                "host": "github.com",
                "usage": "optional inspected implementation path before expanding beyond the source-bound mini-lab",
            }
        )
        if len(links) >= limit:
            break
    return links


def _normalize_github_source_url(raw_url: str, *, owner: str, repo: str) -> str:
    value = str(raw_url or "").strip().strip(".,;:)】}>")
    if not value:
        return f"https://github.com/{owner}/{repo}"
    if value.lower().startswith("www."):
        value = f"https://{value}"
    elif value.lower().startswith("github.com/"):
        value = f"https://{value}"
    elif not re.match(r"(?i)^https?://", value):
        value = f"https://github.com/{owner}/{repo}"
    return re.sub(r"(?i)^http://", "https://", value)


def _implementation_links_for_spec(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        match = re.match(r"(?i)^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", url)
        if not match:
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        links.append(
            {
                "source_id": str(item.get("source_id") or f"implementation:github:{len(links) + 1}"),
                "url": url,
                "source_url": str(item.get("source_url") or url).strip(),
                "host": "github.com",
                "usage": str(
                    item.get("usage")
                    or "optional inspected implementation path before expanding beyond the source-bound mini-lab"
                ),
            }
        )
    return links


def _approved_implementation_repo_roots(context: dict[str, Any] | None) -> set[str]:
    return {
        item["url"].lower()
        for item in _implementation_links_for_spec(
            context.get("implementation_links") if isinstance(context, dict) else None
        )
        if item.get("url")
    }


def _approved_inspected_repo_roots(context: dict[str, Any] | None) -> set[str]:
    manifests = context.get("implementation_repo_manifests") if isinstance(context, dict) else None
    if not isinstance(manifests, list):
        return set()
    roots: set[str] = set()
    for manifest in manifests:
        if not isinstance(manifest, dict) or manifest.get("status") != "inspected":
            continue
        repo_url = _canonical_github_repo_url(str(manifest.get("url") or ""))
        if repo_url:
            roots.add(repo_url.lower())
    return roots


def _canonical_github_repo_url(raw_url: str) -> str | None:
    match = re.search(
        r"(?i)(?:https?://)?(?:www\.)?github\.com/"
        r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
        str(raw_url or ""),
    )
    if not match:
        return None
    owner = match.group(1).strip().strip(".,;:)】}>")
    repo = match.group(2).strip().strip(".,;:)】}>")
    if not owner or not repo or repo.lower() in {"tree", "blob", "issues", "pulls"}:
        return None
    return f"https://github.com/{owner}/{repo}"


def _scrub_unapproved_github_urls(value: Any, approved_repo_roots: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub_unapproved_github_urls(item, approved_repo_roots)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scrub_unapproved_github_urls(item, approved_repo_roots) for item in value]
    if not isinstance(value, str):
        return value

    pattern = re.compile(
        r"(?i)(?:https?://)?(?:www\.)?github\.com/"
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
        r"(?:/[A-Za-z0-9_./?=&%#:+-]*)?"
    )

    def replacement(match: re.Match[str]) -> str:
        repo_root = _canonical_github_repo_url(match.group(0))
        if repo_root and repo_root.lower() in approved_repo_roots:
            return match.group(0)
        return "the source-listed implementation repository" if approved_repo_roots else "the paper source"

    return pattern.sub(replacement, value)


def _format_translation_text(data: dict[str, Any]) -> str:
    return "\n".join(
        f"{item.get('span_id', '')}: {item.get('translation', '')}"
        for item in data.get("translations", [])
    )


def _format_task_text(task: str, data: dict[str, Any]) -> str:
    if task == "translation":
        return _format_translation_text(data)
    if task in {"grounded_qa", "adversarial_grounded_qa"}:
        return str(data.get("answer") or data.get("answer_ko") or data)
    if task == "experiment_spec":
        return format_experiment_spec(data)
    if task == "experiment_candidates":
        return format_experiment_candidates(data)
    if task == "starter_code":
        return str(data.get("code") or "")
    if task == "gpu_script":
        return format_gpu_script(data)
    if task == "research_growth":
        return format_growth_ideas(data)
    return json.dumps(data, ensure_ascii=False)


def _postprocess_task_data(
    task: str,
    data: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if task == "experiment_spec":
        return _lightweight_experiment_spec(data, context=context)
    if task == "starter_code":
        return _repair_starter_code_runtime_contract(data)
    if task == "experiment_candidates":
        return _normalize_experiment_candidates(data, context=context)
    return data


def _repair_starter_code_runtime_contract(data: dict[str, Any]) -> dict[str, Any]:
    return _repair_safe_starter_imports(data)


def _normalize_experiment_candidates(
    data: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        return data
    normalized: list[dict[str, Any]] = []
    recommended_id = str(data.get("recommended_candidate_id") or "").strip()
    requested_level = _normalize_reproduction_level(str((context or {}).get("reproduction_level") or "probe"))
    for index, item in enumerate(candidates[:3], start=1):
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        candidate_id = clean_text(str(candidate.get("id") or ""))
        if not candidate_id:
            raw_title = clean_text(str(candidate.get("title") or f"candidate-{index}")).lower()
            candidate_id = re.sub(r"[^a-z0-9]+", "-", raw_title).strip("-") or f"candidate-{index}"
        candidate["id"] = candidate_id
        if not isinstance(candidate.get("paper_evidence_ids"), list):
            candidate["paper_evidence_ids"] = []
        if not isinstance(candidate.get("paper_evidence_quotes"), list):
            candidate["paper_evidence_quotes"] = []
        if not isinstance(candidate.get("limitations"), list):
            candidate["limitations"] = []
        if not isinstance(candidate.get("dataset"), dict):
            candidate["dataset"] = {"name": str(candidate.get("dataset") or ""), "source": ""}
        if not isinstance(candidate.get("implementation"), dict):
            candidate["implementation"] = {"type": "source_bound_probe", "repo_url": "", "reason": ""}
        raw_level = clean_text(str(candidate.get("reproduction_level") or "")).lower().replace(" ", "_").replace("-", "_")
        candidate["reproduction_level"] = raw_level if raw_level else requested_level
        faithfulness = candidate.get("faithfulness") if isinstance(candidate.get("faithfulness"), dict) else {}
        raw_faithfulness_level = clean_text(str(faithfulness.get("level") or "")).lower().replace(" ", "_").replace("-", "_")
        candidate["faithfulness"] = {
            **faithfulness,
            "level": raw_faithfulness_level or candidate["reproduction_level"],
        }
        if not isinstance(candidate.get("run_plan"), dict):
            dataset = candidate.get("dataset") if isinstance(candidate.get("dataset"), dict) else {}
            candidate["run_plan"] = {
                "repo_url": str(candidate["implementation"].get("repo_url") or ""),
                "config_path": "",
                "command": "",
                "dataset": str(dataset.get("name") or ""),
                "expected_artifact": str(candidate.get("expected_metric") or ""),
            }
        normalized.append(candidate)
    if not recommended_id and normalized:
        recommended = next((item for item in normalized if item.get("is_recommended") is True), normalized[0])
        recommended_id = str(recommended.get("id") or "")
    for item in normalized:
        item["is_recommended"] = bool(item.get("id") == recommended_id)
    return {**data, "candidates": normalized, "recommended_candidate_id": recommended_id}


def _repair_safe_starter_imports(data: dict[str, Any]) -> dict[str, Any]:
    code = str(data.get("code") or "")
    if not code.strip():
        return data
    imports_to_add: list[str] = []
    for module in ("json", "re", "math"):
        uses_module = bool(re.search(rf"(?<![A-Za-z0-9_]){module}\s*\.", code))
        imports_module = bool(
            re.search(rf"^\s*import\s+{module}\b", code, flags=re.MULTILINE)
            or re.search(rf"^\s*from\s+{module}\s+import\s+", code, flags=re.MULTILINE)
        )
        if uses_module and not imports_module:
            imports_to_add.append(f"import {module}")
    if not imports_to_add:
        return data
    repaired = dict(data)
    repaired["code"] = "\n".join(imports_to_add) + "\n\n" + code.lstrip()
    repaired["runtime_repairs"] = [
        *list(data.get("runtime_repairs") or []),
        f"added missing safe import: {', '.join(module.split()[1] for module in imports_to_add)}",
    ]
    return repaired


def _lightweight_experiment_spec(
    data: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    selected_span = clean_text(str(context.get("selected_span") or ""))
    approved_repo_roots = _approved_implementation_repo_roots(context)
    flattened = json.dumps(data, ensure_ascii=False).lower()
    repaired = dict(data)
    heavy_terms = experiment_heavy_terms(flattened)
    focus_label = _experiment_focus_label(selected_span or flattened)
    metric = _experiment_metric_hint(selected_span or flattened) if heavy_terms else str(
        repaired.get("metric") or _experiment_metric_hint(selected_span or flattened)
    )

    if heavy_terms and experiment_heavy_terms(str(repaired.get("research_question", ""))):
        repaired["research_question"] = (
            f"Can a source-bound {focus_label} run produce a measurable signal in a dependency-free mini-lab?"
        )
    if heavy_terms:
        if _is_attention_mechanism_span(selected_span or flattened):
            repaired.update(
                {
                    "mini_lab_goal": (
                        "Run a dependency-free comparison between a local-first baseline and an attention-style "
                        "global scorer on indexed paper evidence rows."
                    ),
                    "dataset": {
                        "name": "Indexed PaperLens evidence window",
                        "source": "Require source-index spans around the selected paper evidence.",
                    },
                    "baseline": "Local or first-match heuristic without the attention-style global scoring bonus.",
                    "metric": metric,
                    "steps": [
                        "Load the indexed evidence rows around the selected span.",
                        "Run the local baseline on every example.",
                        "Run an attention-style global scoring rule while keeping the examples fixed.",
                        "Compare the metric and inspect the long-range or distractor-heavy failures.",
                    ],
                    "ablation": (
                        "Disable only the attention-style global scoring bonus and keep the examples, candidates, "
                        "and metric fixed."
                    ),
                    "failure_condition": (
                        f"The mini-lab fails if {metric} does not improve after adding the attention-style global scorer."
                    ),
                    "expected_result": (
                        "A small directional signal may appear on long-range or distractor-heavy examples, but this does "
                        "not reproduce the original benchmark or training setup."
                    ),
                    "starter_code_plan": [
                        "baseline(example)",
                        "paper_inspired(example)",
                        "score(output, gold)",
                        "run()",
                    ],
                }
            )
        else:
            repaired.update(
                {
                    "mini_lab_goal": (
                        "Run a dependency-free comparison between a direct baseline "
                        "and one paper-inspired heuristic using indexed paper evidence."
                    ),
                    "dataset": {
                        "name": "Indexed PaperLens evidence window",
                        "source": "Require source-index spans around the selected paper evidence.",
                    },
                    "baseline": "Direct keyword or error-tag heuristic without the paper-inspired operation.",
                    "metric": metric,
                    "steps": [
                        "Load the source-index evidence rows around the selected span.",
                        "Run the direct baseline on every example.",
                        "Run one paper-inspired heuristic while keeping the examples fixed.",
                        "Compare the metric and inspect failure tags.",
                    ],
                    "ablation": "Disable only the paper-inspired heuristic and keep the examples, scoring, and prompts fixed.",
                    "failure_condition": (
                        f"The mini-lab fails if {metric} does not improve or only improves by changing the task."
                    ),
                    "expected_result": (
                        "A directional signal may appear in the selected evidence window; it does not reproduce "
                        "the original paper benchmark or training setup."
                    ),
                    "starter_code_plan": [
                        "baseline(example)",
                        "paper_inspired(example)",
                        "score(output, expected_terms)",
                        "run()",
                    ],
                }
            )

    if _is_attention_mechanism_span(selected_span or flattened):
        metric_text = str(repaired.get("metric") or "")
        if any(term in metric_text.lower() for term in ("bleu", "benchmark", "translation")):
            repaired.update(
                {
                    "metric": _experiment_metric_hint(selected_span or flattened),
                    "mini_lab_goal": (
                        "Compare a local-first baseline against an attention-style global scorer on indexed "
                        "evidence rows around the highlighted span."
                    ),
                    "baseline": "Local or first-match heuristic without the attention-style global scoring bonus.",
                    "ablation": (
                        "Remove only the attention-style global scoring bonus and keep the examples, candidates, "
                        "and metric fixed."
                    ),
                    "failure_condition": (
                        "The mini-lab fails if the selected source-bound metric does not improve "
                        "after adding the attention-style global scorer."
                    ),
                }
            )

    dataset = repaired.get("dataset")
    if not isinstance(dataset, dict):
        repaired["dataset"] = {
            "name": str(dataset or "Indexed PaperLens evidence window"),
            "source": "Indexed PaperLens evidence rows around the selected span.",
        }
    else:
        dataset = {str(key): value for key, value in dataset.items()}
        dataset_text = json.dumps(dataset, ensure_ascii=False).lower()
        service_dataset = {
            "name": "Indexed PaperLens evidence window",
            "source": "Indexed PaperLens evidence rows around the selected span.",
        }
        service_blockers = (
            "toy",
            "hand-built",
            "built-in example",
            "sample dataset",
            "synthetic",
            "simulated",
            "pseudo",
            "randomly initialized",
            "random initialization",
            "random vector",
            "random vectors",
            "random-vector",
            "random-vectors",
            "controlled sequence",
            "controlled-sequence",
            "small sequence",
            "small-sequence",
            "generated sequence",
            "generated-sequence",
            "generated inputs",
            "generated-inputs",
        )
        if any(term in dataset_text for term in service_blockers):
            repaired["dataset"] = dataset
        elif not any(term in dataset_text for term in ("indexed", "source", "evidence", "paperlens", "paper")):
            repaired["dataset"] = {**dataset, **service_dataset}
        else:
            dataset.pop("fallback", None)
            dataset.setdefault("name", "Indexed PaperLens evidence window")
            dataset.setdefault("source", "Indexed PaperLens evidence rows around the selected span.")
            repaired["dataset"] = dataset

    if not repaired.get("metric"):
        repaired["metric"] = metric
    if not _ablation_isolating(str(repaired.get("ablation", ""))):
        repaired["ablation"] = "Remove only the paper-inspired operation and keep the examples, prompts, and metric fixed."
    if not _failure_mentions_metric(str(repaired.get("metric", "")), str(repaired.get("failure_condition", ""))):
        repaired["failure_condition"] = (
            f"The mini-lab fails if {repaired['metric']} does not improve or only improves by changing the task."
        )
    steps = repaired.get("steps")
    if not isinstance(steps, list) or len(steps) < 3:
        repaired["steps"] = [
            "Load the indexed evidence rows around the selected span.",
            "Run the baseline on every example.",
            "Run the paper-inspired variant while keeping the examples fixed.",
            "Compare the metric and inspect where the variant helps or fails.",
        ]
    support_ids = repaired.get("support_span_ids")
    if not isinstance(support_ids, list) or not support_ids:
        repaired["support_span_ids"] = ["selected"]
    elif "selected" not in support_ids:
        repaired["support_span_ids"] = ["selected", *support_ids]

    notes = repaired.get("faithfulness_notes")
    if not isinstance(notes, list):
        notes = []
    notes = [str(note) for note in notes if not experiment_heavy_terms(str(note))]
    repaired["faithfulness_notes"] = list(dict.fromkeys(
        [
            *notes,
            "This mini-lab is a source-bound run, not a full reproduction of the original paper benchmark.",
            "Keep every row dependency-free and explicitly tied to indexed paper evidence.",
        ]
    ))
    repaired["implementation_repositories"] = _implementation_links_for_spec(
        context.get("implementation_links") if isinstance(context, dict) else None
    )
    repaired = _scrub_unapproved_github_urls(repaired, approved_repo_roots)
    repaired["implementation_repositories"] = _implementation_links_for_spec(
        context.get("implementation_links") if isinstance(context, dict) else None
    )
    if heavy_terms:
        repaired["repair_notes"] = [
            "reduced_heavy_experiment_plan_to_dependency_free_source_bound_run",
            "preserved_selected_mechanism_but_replaced_training_plan_with_indexed_evidence_contract",
        ]
    return repaired


def _materialize_task_output(
    task: str,
    raw: str,
    *,
    context: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    data = _parse_json_object(raw)
    if data is None:
        data = _coerce_non_json_task_output(task, raw)
    if data is None:
        return f"model returned non-JSON output for {task}", {}

    schema_errors = _validate_task_data(task, data)
    if schema_errors and task == "starter_code":
        coerced = _coerce_non_json_task_output(task, raw)
        if coerced is not None:
            data = coerced
            schema_errors = _validate_task_data(task, data)
    if schema_errors:
        return f"invalid model output for {task}: {', '.join(schema_errors)}", {}

    data = _postprocess_task_data(task, data, context=context)
    if task == "experiment_spec":
        spec_eval = evaluate_experiment_spec(data)
        if not spec_eval.passed:
            return (
                f"postprocessed model output for {task} failed mini-lab constraints: {', '.join(spec_eval.reasons)}",
                data,
            )
    if task == "experiment_candidates":
        candidate_errors = _experiment_candidate_contract_errors(data, context or {})
        if candidate_errors:
            return f"generated experiment candidates failed checks: {', '.join(candidate_errors)}", data
    if task == "starter_code":
        reasons: list[str] = []
        code = str(data.get("code", ""))
        code_eval = evaluate_starter_code(
            code,
            evidence_rows=_starter_validation_rows(context or {}),
            require_evidence_rows=True,
        )
        if not code_eval.passed:
            reasons.extend(code_eval.reasons)
        grounding_eval = evaluate_starter_grounding(code, str((context or {}).get("selected_span", "")))
        if not grounding_eval.passed:
            reasons.extend(grounding_eval.reasons)
        if reasons:
            return f"generated starter code failed checks for {task}: {', '.join(reasons)}", data
    if task == "gpu_script":
        script_errors = _gpu_script_contract_errors(data, context=context or {})
        if script_errors:
            return f"generated GPU script failed checks: {', '.join(script_errors)}", data
    if task == "research_growth":
        known_evidence_ids = set(str(item) for item in (context or {}).get("known_evidence_ids", []) if str(item))
        growth_eval = evaluate_growth_ideas(
            data,
            known_evidence_ids=known_evidence_ids or None,
            require_multiple_sources=bool(known_evidence_ids),
        )
        if not growth_eval.passed:
            return f"generated research growth failed checks: {', '.join(growth_eval.reasons)}", data
    return None, data


def _experiment_candidate_contract_errors(data: dict[str, Any], context: dict[str, Any]) -> list[str]:
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        return ["candidates must be a list"]
    errors: list[str] = []
    if len(candidates) not in {2, 3}:
        errors.append("must return 2 or 3 candidates")
    recommended_id = str(data.get("recommended_candidate_id") or "")
    recommended_count = sum(1 for item in candidates if isinstance(item, dict) and item.get("is_recommended") is True)
    if not recommended_id:
        errors.append("missing recommended_candidate_id")
    if recommended_count != 1:
        errors.append("exactly one candidate must be recommended")
    allowed_ids = {
        str(item.get("source_id") or "")
        for item in context.get("source_evidence", [])
        if isinstance(item, dict) and item.get("source_id")
    }
    approved_repo_roots = _approved_implementation_repo_roots(context)
    ids: set[str] = set()
    has_gpu_candidate = False
    blocked_terms = ("toy", "fake", "placeholder", "random vector", "random-vector", "simulated", "pseudo")
    for index, item in enumerate(candidates, start=1):
        if not isinstance(item, dict):
            errors.append(f"candidate {index} is not an object")
            continue
        candidate_id = str(item.get("id") or "")
        if not candidate_id:
            errors.append(f"candidate {index} missing id")
        elif candidate_id in ids:
            errors.append(f"candidate {index} duplicate id")
        ids.add(candidate_id)
        for key in ("title", "kind", "hypothesis", "expected_metric", "approval_question"):
            if not item.get(key):
                errors.append(f"candidate {index} missing {key}")
        evidence_ids = item.get("paper_evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            errors.append(f"candidate {index} missing paper_evidence_ids")
        elif allowed_ids:
            for evidence_id in evidence_ids:
                if str(evidence_id) not in allowed_ids:
                    errors.append(f"candidate {index} cites outside evidence id {evidence_id}")
        if not isinstance(item.get("paper_evidence_quotes"), list) or not item.get("paper_evidence_quotes"):
            errors.append(f"candidate {index} missing paper_evidence_quotes")
        if not isinstance(item.get("dataset"), dict):
            errors.append(f"candidate {index} missing dataset")
        implementation = item.get("implementation")
        raw_reproduction_level = clean_text(str(item.get("reproduction_level") or "")).lower().replace(" ", "_").replace("-", "_")
        if not raw_reproduction_level:
            errors.append(f"candidate {index} missing reproduction_level")
            reproduction_level = ""
        elif raw_reproduction_level not in REPRODUCTION_LEVELS:
            errors.append(f"candidate {index} invalid reproduction_level {raw_reproduction_level}")
            reproduction_level = raw_reproduction_level
        else:
            reproduction_level = raw_reproduction_level
        if isinstance(implementation, dict):
            repo_url = _canonical_github_repo_url(str(implementation.get("repo_url") or ""))
            if repo_url and repo_url.lower() not in approved_repo_roots:
                errors.append(f"candidate {index} uses a repo URL that is not listed in the paper source")
            if reproduction_level == "exact":
                implementation_type = str(implementation.get("type") or "").strip().lower()
                if implementation_type != "paper_repo" or not repo_url:
                    errors.append(f"candidate {index} labels exact reproduction without a source-listed paper repo")
                elif repo_url.lower() not in approved_repo_roots:
                    errors.append(f"candidate {index} labels exact reproduction with an unapproved repo URL")
                run_plan = item.get("run_plan") if isinstance(item.get("run_plan"), dict) else {}
                plan_repo_url = _canonical_github_repo_url(str(run_plan.get("repo_url") or ""))
                if not plan_repo_url:
                    errors.append(f"candidate {index} labels exact reproduction without run_plan.repo_url")
                elif repo_url and plan_repo_url.lower() != repo_url.lower():
                    errors.append(f"candidate {index} exact run_plan.repo_url must match implementation repo_url")
                for required_key in ("config_path", "command", "dataset"):
                    if not str(run_plan.get(required_key) or "").strip():
                        errors.append(f"candidate {index} labels exact reproduction without run_plan.{required_key}")
        elif reproduction_level == "exact":
            errors.append(f"candidate {index} labels exact reproduction without implementation metadata")
        if not isinstance(item.get("limitations"), list) or not item.get("limitations"):
            errors.append(f"candidate {index} missing limitations")
        if item.get("gpu_required") is True or str(item.get("kind") or "").lower() == "gpu_replication_probe":
            has_gpu_candidate = True
        flattened = json.dumps(item, ensure_ascii=False).lower()
        if any(term in flattened for term in blocked_terms):
            errors.append(f"candidate {index} uses blocked toy/synthetic wording")
    if recommended_id and recommended_id not in ids:
        errors.append("recommended_candidate_id does not match any candidate")
    if not has_gpu_candidate:
        errors.append("missing GPU replication probe candidate")
    return errors


def _gpu_script_contract_errors(data: dict[str, Any], context: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    script = str(data.get("script") or "")
    if not script.strip():
        errors.append("missing script")
    if data.get("entrypoint") != "run_paperlens_gpu_probe":
        errors.append("entrypoint must be run_paperlens_gpu_probe")
    if "def run_paperlens_gpu_probe" not in script:
        errors.append("script must define run_paperlens_gpu_probe")
    errors.extend(_validate_gpu_script_contract(script))
    lowered = script.lower()
    if "torch" not in lowered:
        errors.append("script must use torch for GPU execution")
    blocked = ("subprocess", "socket", "requests", "httpx", "open(", "exec(")
    for term in blocked:
        if term in lowered:
            errors.append(f"script uses blocked operation {term}")
    if 'load_dataset("multi30k"' in lowered or "load_dataset('multi30k'" in lowered:
        errors.append("script must use the exact Hugging Face dataset id bentrevett/multi30k for Multi30k")
    if "bentrevett/multi30k" in lowered and ("['translation']" in lowered or '["translation"]' in lowered):
        errors.append("bentrevett/multi30k rows use en/de fields, not a translation field")
    if "bentrevett/multi30k" in lowered and ("dataloader" in lowered or "torch.utils.data" in lowered):
        errors.append("Multi30k GPU probes must build fixed-shape tensors directly instead of using DataLoader/custom Dataset")
    if "transformerencoder" in lowered and (
        "transformer_model(src_tensor" in lowered
        or "transformer_model(tgt_tensor" in lowered
        or "transformer(src_tensor" in lowered
        or "transformer(tgt_tensor" in lowered
    ):
        errors.append("Transformer throughput probes must pass embedded float tensors with shape [batch, seq_len, d_model], not raw token id tensors")
    blocked_data_terms = (
        "mock",
        "dummy",
        "fake",
        "placeholder",
        "synthetic",
        "simulated",
        "toy",
        "random-vector",
        "random vector",
        "randomly generated",
        "torch.randint",
        "torch.randn",
        "torch.rand(",
        "np.random",
        "numpy.random",
        "random.random",
        "random.randint",
        "random.choice",
    )
    script_and_metadata = f"{script}\n{json.dumps(data.get('dataset') or {}, ensure_ascii=False)}".lower()
    for term in blocked_data_terms:
        if term in script_and_metadata:
            errors.append(f"script uses blocked generated/mock data term {term}")
    if not isinstance(data.get("dataset"), dict) or not data.get("dataset"):
        errors.append("missing dataset")
    raw_script_level = str(data.get("reproduction_level") or "").strip()
    script_level = _normalize_reproduction_level(raw_script_level)
    if not raw_script_level or raw_script_level.lower().replace(" ", "_").replace("-", "_") not in REPRODUCTION_LEVELS:
        errors.append("missing reproduction_level")
    approved_level_raw = str((context or {}).get("reproduction_level") or "").strip()
    if approved_level_raw:
        approved_level = _normalize_reproduction_level(approved_level_raw, default=script_level)
        if script_level != approved_level:
            errors.append("reproduction_level must match approved reproduction level")
    reproduction_plan = data.get("reproduction_plan")
    if not isinstance(reproduction_plan, dict) or not reproduction_plan:
        errors.append("missing reproduction_plan")
    elif _normalize_reproduction_level(str(reproduction_plan.get("level") or ""), default=script_level) != script_level:
        errors.append("reproduction_plan.level must match reproduction_level")
    else:
        repo_url = _canonical_github_repo_url(str(reproduction_plan.get("repo_url") or ""))
        if repo_url and repo_url.lower() not in _approved_inspected_repo_roots(context):
            errors.append("reproduction_plan.repo_url must match an inspected paper implementation repo")
    if isinstance(reproduction_plan, dict) and script_level == "exact":
        repo_url = _canonical_github_repo_url(str(reproduction_plan.get("repo_url") or ""))
        if not repo_url:
            errors.append("exact GPU script requires reproduction_plan.repo_url")
        elif repo_url.lower() not in _approved_inspected_repo_roots(context):
            errors.append("exact GPU script repo_url must match an inspected paper implementation repo")
        for required_key in ("config_path", "command", "dataset"):
            if not str(reproduction_plan.get(required_key) or "").strip():
                errors.append(f"exact GPU script requires reproduction_plan.{required_key}")
    if not isinstance(data.get("limitations"), list) or not data.get("limitations"):
        errors.append("missing limitations")
    return errors


def _starter_validation_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    selected_span = clean_text(str(context.get("selected_span") or ""))
    source_evidence = context.get("source_evidence")
    if not selected_span or not isinstance(source_evidence, list):
        return []
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(source_evidence):
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or f"evidence:{idx}")
        text = clean_text(str(item.get("text") or ""))
        if not source_id or not text:
            continue
        is_selected = idx == 0 or text == selected_span
        rows.append(
            {
                "source_id": source_id,
                "text": text,
                "text_hash": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
                "label": "selected" if is_selected else "context_control",
                "gold": is_selected,
                "query": selected_span,
            }
        )
    return rows


def _repair_task_output_prompt(
    task: str,
    raw: str,
    error: str,
    *,
    context: dict[str, Any] | None = None,
) -> str | None:
    context = context or {}
    if task == "translation":
        spans = context.get("spans")
        if not isinstance(spans, list):
            spans = []
        return translation_repair_prompt(
            raw,
            error,
            spans,
            str(context.get("locale") or "ko"),
        )
    if task == "experiment_spec":
        return experiment_repair_prompt(raw, error)
    if task == "experiment_candidates":
        source_evidence = context.get("source_evidence") if isinstance(context.get("source_evidence"), list) else []
        implementation_links = (
            context.get("implementation_links") if isinstance(context.get("implementation_links"), list) else []
        )
        return experiment_candidates_repair_prompt(
            raw,
            error,
            source_evidence,
            str(context.get("reproduction_level") or "probe"),
            str(context.get("locale") or "en"),
            implementation_links=implementation_links,
        )
    if task == "starter_code":
        return starter_code_repair_prompt(
            raw,
            error,
            str(context.get("selected_span") or ""),
            str(context.get("locale") or "en"),
        )
    if task == "gpu_script":
        candidate = context.get("candidate") if isinstance(context.get("candidate"), dict) else {}
        source_evidence = context.get("source_evidence") if isinstance(context.get("source_evidence"), list) else []
        return gpu_script_repair_prompt(
            raw,
            error,
            candidate,
            source_evidence,
            str(context.get("reproduction_level") or candidate.get("reproduction_level") or "probe"),
            str(context.get("locale") or "en"),
        )
    if task == "research_growth":
        return growth_repair_prompt(
            raw,
            error,
            [str(item) for item in context.get("known_evidence_ids", []) if str(item)],
            str(context.get("locale") or "en"),
        )
    return None


def _is_missing_required_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


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
    if task in {"grounded_qa", "adversarial_grounded_qa"}:
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
    if task == "experiment_candidates":
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return ["missing candidates"]
        errors = []
        if not data.get("recommended_candidate_id"):
            errors.append("missing recommended_candidate_id")
        for idx, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                errors.append(f"candidate {idx} is not an object")
                continue
            for key in ("id", "title", "kind", "hypothesis", "paper_evidence_ids", "dataset", "gpu_required", "expected_metric", "limitations"):
                if key not in candidate or _is_missing_required_value(candidate.get(key)):
                    errors.append(f"candidate {idx} missing {key}")
        return errors
    if task == "starter_code":
        code = data.get("code")
        if not isinstance(code, str) or not code.strip():
            return ["missing code"]
        return []
    if task == "gpu_script":
        script = data.get("script")
        if not isinstance(script, str) or not script.strip():
            return ["missing script"]
        if data.get("entrypoint") != "run_paperlens_gpu_probe":
            return ["missing run_paperlens_gpu_probe entrypoint"]
        return []
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
    cleaned = raw.strip().lstrip("\ufeff")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    candidates = [cleaned]
    balanced = _first_balanced_json_object(cleaned)
    if balanced and balanced != cleaned:
        candidates.append(balanced)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def _coerce_non_json_task_output(task: str, raw: str) -> dict[str, Any] | None:
    if task == "gpu_script":
        # GPU scripts must keep their model-authored JSON envelope. If the model
        # emits raw Python, the repair prompt can wrap its own code; PaperLens
        # must not infer experiment metadata and turn it into a product success.
        return None
    if task != "starter_code":
        return None
    code = _extract_probable_python_code(raw)
    if not code:
        return None
    return {
        "code": code,
        "why_this_matches_span": (
            "Recovered starter code from non-JSON model output. "
            "The Python body passed through the normal starter validation checks."
        ),
        "limitations": [
            "Wrapper metadata was inferred because the model returned raw code instead of the requested JSON envelope.",
        ],
        "recovered_from_non_json": True,
    }


def _extract_probable_python_code(raw: str) -> str:
    cleaned = raw.strip().lstrip("\ufeff")
    fenced_match = re.search(r"```(?:python|py)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        if _looks_like_python_script(candidate):
            return candidate

    lines = cleaned.splitlines()
    start_index = _first_python_line_index(lines)
    if start_index is not None:
        candidate = "\n".join(lines[start_index:]).strip()
        if _looks_like_python_script(candidate):
            return candidate

    return cleaned if _looks_like_python_script(cleaned) else ""


def _first_python_line_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(
            (
                '"""',
                "'''",
                "import ",
                "from ",
                "PAPERLENS_EVIDENCE_ROWS",
                "def ",
                "if __name__ ==",
            )
        ):
            return index
    return None


def _looks_like_python_script(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    score = 0
    markers = (
        "def baseline(",
        "def paper_inspired(",
        "def score(",
        "def run(",
        "if __name__ ==",
        "PAPERLENS_EVIDENCE_ROWS",
        "rows = []",
        "return rows",
        "import json",
        "import re",
    )
    for marker in markers:
        if marker in stripped:
            score += 1
    return score >= 2


def _first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _fallback_translate(source: str, locale: str) -> str:
    if locale != "ko":
        return source
    if not source:
        return ""
    terms = _extract_terms(source, limit=4)
    term_hint = f" 핵심 용어: {', '.join(terms)}." if terms else ""
    return f"[초안 번역] {source}{term_hint}"


def evidence_map(text: str, selected_span: str, span_id: str = "selected") -> dict[str, str]:
    return {item["source_id"]: item["text"] for item in evidence_items(text, selected_span, span_id=span_id)}


def evidence_items(text: str, selected_span: str, span_id: str = "selected") -> list[dict[str, str]]:
    ranked = _top_sentences(text, limit=5)
    items = [{"source_id": span_id, "text": selected_span}]
    for sentence in ranked:
        sid = f"S{sentence['pid']}"
        if sentence["text"].strip() and sentence["text"].strip() != selected_span.strip():
            items.append({"source_id": sid, "text": sentence["text"]})
    return items[:6]


def _evidence_items(text: str, selected_span: str, span_id: str = "selected") -> list[dict[str, str]]:
    return evidence_items(text, selected_span, span_id=span_id)


def _dataset_text(value: Any) -> str:
    if isinstance(value, dict):
        return f"{value.get('name', '')} / source: {value.get('source', '')}"
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


def _is_attention_mechanism_span(text: str) -> bool:
    lowered = clean_text(text).lower()
    return "attention" in lowered and any(token in lowered for token in ("recurr", "convol"))


def _experiment_focus_label(text: str) -> str:
    lowered = clean_text(text).lower()
    if _is_attention_mechanism_span(lowered):
        return "attention-style global scoring"
    if "rerank" in lowered or ("retrieval" in lowered and "evidence" in lowered):
        return "evidence-linked reranking"
    if "low-rank" in lowered or "lora" in lowered:
        return "low-rank adaptation"
    if "adapter" in lowered:
        return "adapter-style update"
    terms = _extract_terms(text, limit=4)
    if terms:
        return terms[0]
    return "selected paper mechanism"


def _experiment_metric_hint(text: str) -> str:
    lowered = clean_text(text).lower()
    if _is_attention_mechanism_span(lowered):
        return "label accuracy on indexed paper evidence rows"
    if "retrieval" in lowered or "rerank" in lowered:
        return "top-k evidence hit rate on indexed retrieval evidence"
    return "source-bound label accuracy"


def _ablation_isolating(text: str) -> bool:
    lowered = clean_text(text).lower()
    return any(term in lowered for term in ("remove only", "disable only", "without only", "isolate one"))


def _failure_mentions_metric(metric: str, failure_condition: str) -> bool:
    metric_head = _normalize_metric_token(str(metric).split(",")[0].split()[0])
    failure_text = _normalize_metric_token(failure_condition)
    return bool(metric_head and metric_head in failure_text) or "metric" in failure_text


def _normalize_metric_token(text: str) -> str:
    return str(text).lower().replace("_", " ").replace("-", " ")


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
