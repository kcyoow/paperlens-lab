from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analysis import split_sentences, top_sentences
from .gpu_lab import GpuLabError, gpu_code_hash, run_gpu_probe_job
from .implementation_repo import inspect_implementation_repositories
from .ingest import PaperSource, build_source, clean_text
from .memory_store import append_memory, load_memories, paper_key
from .mini_lab import MiniLabError, code_hash, mini_lab_provider, run_mini_lab_job
from .model_adapter import DEFAULT_MODEL, DEFAULT_PROVIDER, TRANSLATION_MODEL, ModelGateway, extract_implementation_links
from .scenario_eval import evaluate_growth_ideas, source_contains_quote
from .source_index import (
    evidence_window,
    get_cached_translation,
    get_span_text,
    load_source_index,
    save_cached_translation,
    save_source_index,
    text_hash,
)
from .tracing import trace_content_enabled
from .validation_report import build_validation_summary


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_OUT_DIR = FRONTEND_DIR / "out"
DEFAULT_SANDBOX_WORKSPACE_DIR = Path("outputs/sandbox_workspaces")
REPRODUCTION_LEVELS = {"probe", "exact"}


class PaperInput(BaseModel):
    arxiv_or_url: str = ""
    pasted_text: str = ""
    max_pdf_pages: int = Field(default=64, ge=1, le=96)
    use_model: bool = False
    max_translate_spans: int = Field(default=24, ge=1, le=96)
    max_reader_spans: int = Field(default=800, ge=12, le=1200)


class AskInput(BaseModel):
    paper_id: str = ""
    span_id: str = ""
    scope: str = "span"
    question: str = ""
    original: str
    translated: str = ""
    selected_spans: list[dict[str, Any]] = Field(default_factory=list)
    paper_title: str = "Untitled paper"
    source_text: str = ""
    locale: str = "en"
    use_model: bool = False


class ExperimentInput(BaseModel):
    paper_id: str = ""
    span_id: str = ""
    paper_title: str = "Untitled paper"
    selected_span: str
    translated_span: str = ""
    source_text: str = ""
    idea: str = ""
    locale: str = "en"
    use_model: bool = False
    session_id: str = ""
    workspace_id: str = ""


class ExperimentCandidatesInput(ExperimentInput):
    question: str = ""
    reproduction_level: str = "probe"
    session_id: str = ""
    workspace_id: str = ""


class GpuScriptInput(BaseModel):
    candidate_set_id: str
    candidate_id: str
    paper_id: str = ""
    span_id: str = ""
    selected_span: str = ""
    reproduction_level: str = "probe"
    locale: str = "en"
    use_model: bool = False
    session_id: str = ""
    workspace_id: str = ""


class GpuProbeRunInput(BaseModel):
    gpu_run_id: str
    session_id: str = ""
    workspace_id: str = ""


class TranslationInput(BaseModel):
    paper_id: str = ""
    paper_title: str = "Untitled paper"
    spans: list[dict[str, str]]
    locale: str = "ko"
    use_model: bool = False


class TranslateSpanInput(BaseModel):
    paper_id: str
    paper_title: str = "Untitled paper"
    span_id: str
    source_text: str = ""
    locale: str = "ko"
    use_model: bool = False
    force_refresh: bool = False


class GrowthInput(BaseModel):
    paper_id: str = ""
    paper_title: str = "Untitled paper"
    selected_span: str = ""
    paper_memory: list[dict[str, Any]] = Field(default_factory=list)
    mini_lab_result: str = ""
    locale: str = "ko"
    use_model: bool = False
    persist_memory: bool = True


class StarterRunInput(BaseModel):
    code: str
    paper_id: str = ""
    paper_title: str = "Untitled paper"
    span_id: str
    selected_span: str = ""


class MiniLabRunInput(BaseModel):
    code: str
    experiment_run_id: str = ""
    paper_id: str = ""
    paper_title: str = "Untitled paper"
    span_id: str
    selected_span: str = ""
    session_id: str = ""
    workspace_id: str = ""


def _reproduction_level(value: str, default: str = "probe") -> str:
    level = clean_text(value).lower().replace(" ", "_").replace("-", "_")
    if level in REPRODUCTION_LEVELS:
        return level
    return default


def _validated_reproduction_level(value: str) -> str:
    level = clean_text(value).lower().replace(" ", "_").replace("-", "_")
    if level in REPRODUCTION_LEVELS:
        return level
    raise HTTPException(status_code=400, detail="Reproduction level must be one of: probe, exact.")


def _candidate_reproduction_level(candidate: dict[str, Any], fallback: str = "probe") -> str:
    return _reproduction_level(str(candidate.get("reproduction_level") or ""), default=fallback)


_EXPERIMENT_RUN_TTL_SECONDS = 60 * 60
_MAX_EXPERIMENT_RUNS = 128
_EXPERIMENT_RUNS: dict[str, dict[str, Any]] = {}
_CANDIDATE_SETS: dict[str, dict[str, Any]] = {}
_GPU_PROBE_RUNS: dict[str, dict[str, Any]] = {}


def create_app() -> FastAPI:
    app = FastAPI(
        title="PaperLens Lab",
        description="React reader frontend with a Python model backend.",
        version="0.2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _maybe_mount_frontend_assets(app)
    _register_api(app)

    _register_frontend_routes(app)
    return app


def _maybe_mount_frontend_assets(app: FastAPI) -> None:
    next_dir = FRONTEND_OUT_DIR / "_next"
    if next_dir.exists():
        app.mount("/_next", StaticFiles(directory=next_dir), name="next-static")


def _register_api(app: FastAPI) -> None:
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "frontend_ready": _frontend_ready(),
            "frontend_dir": str(FRONTEND_OUT_DIR),
            "gradio_path": None,
            "model": DEFAULT_MODEL,
            "translationModel": TRANSLATION_MODEL,
            "provider": DEFAULT_PROVIDER,
            "miniLabProvider": mini_lab_provider(),
            "forceModel": _force_model_enabled(),
            "traceContent": trace_content_enabled(),
            "runtime": "react-fastapi-service",
        }

    @app.get("/api/validation")
    def validation() -> dict[str, Any]:
        return build_validation_summary()

    @app.get("/api/sample-paper")
    def sample_paper() -> dict[str, Any]:
        raise HTTPException(status_code=404, detail="Sample papers are disabled in service mode. Load a real PDF or arXiv paper.")

    @app.post("/api/paper")
    def load_paper(payload: PaperInput) -> dict[str, Any]:
        try:
            source = build_source(
                uploaded_pdf=None,
                arxiv_or_url=payload.arxiv_or_url,
                pasted_text=_paper_payload_text(payload),
                max_pdf_pages=payload.max_pdf_pages,
            )
            return paper_document_from_source(
                source,
                use_model=_should_use_model(payload.use_model),
                max_translate_spans=payload.max_translate_spans,
                max_reader_spans=payload.max_reader_spans,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/paper/upload")
    async def upload_paper(
        pdf: UploadFile = File(...),
        max_pdf_pages: int = Form(10),
        arxiv_or_url: str = Form(""),
        use_model: bool = Form(False),
        max_translate_spans: int = Form(24),
        max_reader_spans: int = Form(800),
    ) -> dict[str, Any]:
        if not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Upload a PDF file.")

        suffix = Path(pdf.filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
            shutil.copyfileobj(pdf.file, handle)
            handle.flush()
            try:
                source = build_source(
                    uploaded_pdf=handle.name,
                    arxiv_or_url=arxiv_or_url,
                    pasted_text="",
                    max_pdf_pages=max(1, min(max_pdf_pages, 96)),
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return paper_document_from_source(
            source,
            use_model=_should_use_model(use_model),
            max_translate_spans=max(1, min(max_translate_spans, 96)),
            max_reader_spans=max(12, min(max_reader_spans, 1200)),
        )

    @app.post("/api/translate")
    def translate_spans(payload: TranslationInput) -> dict[str, Any]:
        gateway = ModelGateway()
        prepared = _prepare_translation_requests(payload, gateway)
        uncached = [item for item in prepared if not item["cached_translation"]]
        translations_by_key: dict[int, dict[str, Any]] = {}
        model = gateway.translation_model_id
        provider = gateway.provider
        trace_id = ""
        error = None
        used_fallback = False

        for item in prepared:
            if item["cached_translation"]:
                translations_by_key[item["request_index"]] = {
                    "span_id": item["span_id"],
                    "translation": item["cached_translation"],
                    "status": "cached",
                    "sourceHash": item["source_hash"],
                    "sourceIndexBound": item["source_index_bound"],
                }

        if uncached:
            result = gateway.translate_spans(
                payload.paper_title,
                [{"span_id": item["span_id"], "text": item["source_text"]} for item in uncached],
                locale=payload.locale,
                use_model=_should_use_model(payload.use_model),
            )
            model = result.model
            provider = result.provider
            trace_id = result.trace_id
            error = result.error
            used_fallback = result.used_fallback
            translated_by_id = {
                str(item.get("span_id", "")): str(item.get("translation", ""))
                for item in result.data.get("translations", [])
                if isinstance(item, dict) and item.get("span_id")
            }
            for item in uncached:
                translation = translated_by_id.get(item["span_id"], "")
                status = _translation_status(translation, result.used_fallback)
                if status == "ready" and item["paper_id"]:
                    save_cached_translation(
                        item["paper_id"],
                        item["span_id"],
                        item["source_text"],
                        translation,
                        locale=payload.locale,
                        model=result.model,
                    )
                translations_by_key[item["request_index"]] = {
                    "span_id": item["span_id"],
                    "translation": translation,
                    "status": status,
                    "sourceHash": item["source_hash"],
                    "sourceIndexBound": item["source_index_bound"],
                }
        return {
            "translations": [translations_by_key[index] for index in sorted(translations_by_key)],
            "notes": [] if not uncached else result.data.get("notes", []),
            "model": model,
            "provider": provider,
            "traceId": trace_id,
            "error": error,
            "usedFallback": used_fallback,
        }

    @app.post("/api/translate-span")
    def translate_span(payload: TranslateSpanInput) -> dict[str, Any]:
        indexed_text = get_span_text(payload.paper_id, payload.span_id) if payload.paper_id else ""
        if payload.paper_id and not indexed_text:
            raise HTTPException(status_code=404, detail="Selected span was not found in the paper index.")
        source_text = clean_text(indexed_text or payload.source_text)
        if not source_text:
            raise HTTPException(status_code=404, detail="Selected span was not found in the paper index.")
        source_hash = text_hash(source_text)
        source_index_bound = bool(indexed_text)
        gateway = ModelGateway()
        cached = ""
        if not payload.force_refresh:
            cached = get_cached_translation(
                payload.paper_id,
                payload.span_id,
                source_text,
                locale=payload.locale,
                model=gateway.translation_model_id,
            )
        if cached:
            return {
                "spanId": payload.span_id,
                "translation": cached,
                "status": "cached",
                "model": gateway.translation_model_id,
                "provider": gateway.provider,
                "usedFallback": False,
                "sourceHash": source_hash,
                "sourceIndexBound": source_index_bound,
            }
        result = gateway.translate_spans(
            payload.paper_title,
            [{"span_id": payload.span_id, "text": source_text}],
            locale=payload.locale,
            use_model=_should_use_model(payload.use_model),
        )
        translation = ""
        translations = result.data.get("translations", [])
        if translations and isinstance(translations[0], dict):
            translation = translations[0].get("translation", "")
        status = _translation_status(translation, result.used_fallback)
        if status == "ready":
            save_cached_translation(
                payload.paper_id,
                payload.span_id,
                source_text,
                translation,
                locale=payload.locale,
                model=result.model,
            )
        return {
            "spanId": payload.span_id,
            "translation": translation,
            "status": status,
            "model": result.model,
            "provider": result.provider,
            "traceId": result.trace_id,
            "error": result.error,
            "usedFallback": result.used_fallback,
            "sourceHash": source_hash,
            "sourceIndexBound": source_index_bound,
        }

    @app.post("/api/ask")
    def ask_question(payload: AskInput) -> dict[str, Any]:
        question = clean_text(payload.question)
        if not question:
            question = (
                "Explain this selected sentence using only the surrounding paper evidence."
                if payload.locale == "en"
                else "선택한 문장을 논문 근거에 맞춰 설명해줘."
            )

        gateway = ModelGateway()
        scope = "paper" if payload.scope == "paper" else "span"
        if scope == "paper":
            source_text = clean_text(payload.source_text)
            if not source_text:
                raise HTTPException(status_code=400, detail="Paper-level questions require source text.")
            paper_evidence = _paper_evidence_items(source_text)
            allowed_source_ids = {item["source_id"] for item in paper_evidence}
            source_text_by_id = {item["source_id"]: item["text"] for item in paper_evidence}
            selected_span_text = (
                "Paper-level question. Use the supplied evidence items as the bounded paper context."
            )
            result = gateway.answer_span(
                paper_title=payload.paper_title,
                span_id="paper",
                selected_span=selected_span_text,
                translated_span="",
                question=question,
                source_text=source_text,
                locale=payload.locale,
                use_model=_should_use_model(payload.use_model),
                evidence_items_override=paper_evidence,
            )
            answer_data, validation_error = _validated_answer_data(
                result.data,
                payload,
                evidence_text=source_text,
                allowed_source_ids=allowed_source_ids,
                selected_span_text=selected_span_text,
                source_text_by_id=source_text_by_id,
            )
            fallback_content = _fallback_answer(payload, question, selected_span_text=selected_span_text, evidence_text=source_text)
            return {
                "role": "assistant",
                "content": fallback_content if validation_error else answer_data.get("answer") or result.text or fallback_content,
                "supportSpanIds": _support_ids(answer_data, "paper"),
                "evidence": answer_data.get("evidence", []),
                "evidenceWindow": None,
                "confidence": answer_data.get("confidence", "low"),
                "needsMoreContext": answer_data.get("needs_more_context", True),
                "model": result.model,
                "provider": result.provider,
                "traceId": result.trace_id,
                "error": validation_error or result.error,
                "usedFallback": result.used_fallback or bool(validation_error),
            }

        indexed_text = get_span_text(payload.paper_id, payload.span_id) if payload.paper_id else ""
        if payload.paper_id and not indexed_text:
            raise HTTPException(status_code=404, detail="Selected span was not found in the paper index.")
        selected_segments = _validated_selected_segments(payload)
        window = (
            selected_evidence_window(payload.paper_id, selected_segments)
            if selected_segments
            else evidence_window(payload.paper_id, payload.span_id) if payload.paper_id else None
        )
        if payload.paper_id and not window:
            raise HTTPException(status_code=404, detail="Selected span was not found in the paper index.")
        source_text = clean_text((window["text"] if window else payload.source_text) or indexed_text or payload.original)
        selected_span_text = (
            clean_text(" ".join(segment["text"] for segment in selected_segments))
            if selected_segments
            else _selected_text_from_payload(
                payload.original,
                indexed_text=indexed_text,
                source_text=source_text,
            )
        )
        window_evidence = _selected_segment_evidence_items(selected_segments) + _window_evidence_items(window)
        allowed_source_ids = {item["source_id"] for item in window_evidence} if window_evidence else None
        source_text_by_id = _source_text_by_evidence_id(window_evidence)
        result = gateway.answer_span(
            paper_title=payload.paper_title,
            span_id=payload.span_id,
            selected_span=selected_span_text,
            translated_span=payload.translated,
            question=question,
            source_text=source_text,
            locale=payload.locale,
            use_model=_should_use_model(payload.use_model),
            evidence_items_override=window_evidence or None,
        )
        answer_data, validation_error = _validated_answer_data(
            result.data,
            payload,
            evidence_text=source_text,
            allowed_source_ids=allowed_source_ids,
            selected_span_text=selected_span_text,
            source_text_by_id=source_text_by_id,
            selected_segments=selected_segments,
        )
        fallback_content = _fallback_answer(payload, question, selected_span_text=selected_span_text, evidence_text=source_text)
        return {
            "role": "assistant",
            "content": fallback_content if validation_error else answer_data.get("answer") or result.text or fallback_content,
            "supportSpanIds": _support_ids(answer_data, payload.span_id),
            "evidence": answer_data.get("evidence", []),
            "evidenceWindow": _public_evidence_window(window),
            "confidence": answer_data.get("confidence", "low"),
            "needsMoreContext": answer_data.get("needs_more_context", True),
            "model": result.model,
            "provider": result.provider,
            "traceId": result.trace_id,
            "error": validation_error or result.error,
            "usedFallback": result.used_fallback or bool(validation_error),
        }

    @app.post("/api/experiment/candidates")
    def build_experiment_candidates(payload: ExperimentCandidatesInput) -> dict[str, Any]:
        _client_owner(payload.session_id, payload.workspace_id)
        context = _experiment_context(payload)
        question = clean_text(payload.question or payload.idea) or (
            "What experiment should we run from this selected paper evidence?"
        )
        reproduction_level = _validated_reproduction_level(payload.reproduction_level)
        gateway = ModelGateway()
        use_model = _should_use_model(payload.use_model)
        if not use_model:
            raise HTTPException(status_code=400, detail="Experiment candidate generation requires a live model path.")
        result = gateway.experiment_candidates(
            paper_title=payload.paper_title,
            selected_span=context["selectedSpan"],
            translated_span=payload.translated_span,
            source_text=context["sourceText"],
            question=question,
            reproduction_level=reproduction_level,
            locale=payload.locale,
            use_model=use_model,
        )
        if result.used_fallback or result.error:
            raise HTTPException(
                status_code=503,
                detail=_public_experiment_candidates_error(
                    payload.locale,
                    result.error or "Experiment candidates were unavailable.",
                ),
            )
        candidate_set = _issue_candidate_set(
            paper_id=payload.paper_id,
            paper_title=payload.paper_title,
            span_id=payload.span_id,
            selected_span=context["selectedSpan"],
            source_text=context["sourceText"],
            question=question,
            candidates=list(result.data.get("candidates") or []),
            recommended_candidate_id=str(result.data.get("recommended_candidate_id") or ""),
            reproduction_level=reproduction_level,
            trace_id=result.trace_id,
            provider=result.provider,
            model=result.model,
            implementation_links=extract_implementation_links(context["sourceText"]),
            session_id=payload.session_id,
            workspace_id=payload.workspace_id,
        )
        return {
            "candidateSetId": candidate_set["id"],
            "candidateSet": candidate_set,
            "candidates": candidate_set["candidates"],
            "recommendedCandidateId": candidate_set["recommendedCandidateId"],
            "question": question,
            "reproductionLevel": candidate_set["reproductionLevel"],
            "model": result.model,
            "provider": result.provider,
            "traceId": result.trace_id,
            "usedFallback": result.used_fallback,
            "error": result.error,
        }

    @app.post("/api/experiment/gpu-script")
    def build_gpu_script(payload: GpuScriptInput) -> dict[str, Any]:
        candidate_set = _validated_candidate_set(payload)
        candidate = _candidate_from_set(candidate_set, payload.candidate_id)
        reproduction_level = _candidate_reproduction_level(candidate, fallback=candidate_set.get("reproductionLevel", "probe"))
        gateway = ModelGateway()
        use_model = _should_use_model(payload.use_model)
        if not use_model:
            raise HTTPException(status_code=400, detail="GPU script generation requires a live model path.")
        implementation_repo_manifests = inspect_implementation_repositories(
            _candidate_implementation_repositories(candidate, candidate_set)
        )
        exact_blocker = _exact_reproduction_blocker(reproduction_level, implementation_repo_manifests, payload.locale)
        if exact_blocker:
            raise HTTPException(status_code=503, detail=exact_blocker)
        result = gateway.gpu_script(
            paper_title=candidate_set["paperTitle"],
            selected_span=candidate_set["selectedSpan"],
            source_text=candidate_set["sourceText"],
            candidate=candidate,
            locale=payload.locale,
            implementation_repo_manifests=implementation_repo_manifests,
            use_model=use_model,
        )
        script = str(result.data.get("script") or result.text or "")
        if result.used_fallback or result.error or not script:
            raise HTTPException(
                status_code=503,
                detail=_public_gpu_script_error(payload.locale, result.error or "GPU script was unavailable."),
            )
        script_reproduction_level = _reproduction_level(str(result.data.get("reproduction_level") or ""), default="")
        if script_reproduction_level != reproduction_level:
            raise HTTPException(
                status_code=503,
                detail=_public_gpu_script_error(payload.locale, "GPU script reproduction level did not match the approved level."),
            )
        gpu_run = _issue_gpu_probe_run(
            candidate_set=candidate_set,
            candidate=candidate,
            code=script,
            gpu_trace_id=result.trace_id,
            provider=result.provider,
            model=result.model,
            script_data=result.data,
            implementation_repo_manifests=implementation_repo_manifests,
        )
        return {
            "gpuRunId": gpu_run["id"],
            "gpuRun": gpu_run,
            "workspaceId": gpu_run.get("workspaceId", gpu_run["id"]),
            "workspace": gpu_run.get("workspace", {}),
            "candidate": candidate,
            "reproductionLevel": reproduction_level,
            "requestedReproductionLevel": candidate_set.get("reproductionLevel", reproduction_level),
            "script": script,
            "entrypoint": result.data.get("entrypoint", "run_paperlens_gpu_probe"),
            "dependencies": result.data.get("dependencies", []),
            "hardware": result.data.get("hardware", "T4"),
            "dataset": result.data.get("dataset", {}),
            "reproductionPlan": result.data.get("reproduction_plan", {}),
            "expectedOutputs": result.data.get("expected_outputs", []),
            "paperClaimComparisonPlan": result.data.get("paper_claim_comparison_plan", ""),
            "limitations": result.data.get("limitations", []),
            "implementationRepoManifests": implementation_repo_manifests,
            "model": result.model,
            "provider": result.provider,
            "traceId": result.trace_id,
            "usedFallback": result.used_fallback,
            "error": result.error,
        }

    @app.post("/api/gpu-lab/run")
    def run_gpu_lab(payload: GpuProbeRunInput) -> dict[str, Any]:
        binding = _validated_gpu_probe_run(payload)
        try:
            result = run_gpu_probe_job(binding)
        except GpuLabError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        binding["lastResult"] = result
        binding["lastRunAt"] = time.time()
        _persist_sandbox_result(binding, result)
        return result

    @app.post("/api/experiment")
    def build_experiment(payload: ExperimentInput) -> dict[str, Any]:
        _client_owner(payload.session_id, payload.workspace_id)
        if not payload.paper_id or not payload.span_id:
            raise HTTPException(status_code=400, detail="Experiment requires an indexed paper id and selected span id.")
        indexed_text = get_span_text(payload.paper_id, payload.span_id) if payload.paper_id and payload.span_id else ""
        if not indexed_text:
            raise HTTPException(status_code=404, detail="Selected span was not found in the paper index.")
        window = evidence_window(payload.paper_id, payload.span_id) if payload.paper_id and payload.span_id else None
        if not window:
            raise HTTPException(status_code=404, detail="Selected span was not found in the paper index.")
        source_text = clean_text((window["text"] if window else payload.source_text) or indexed_text or payload.selected_span)
        selected_span = _selected_text_from_payload(
            payload.selected_span,
            indexed_text=indexed_text,
            source_text=source_text,
        )
        idea = clean_text(payload.idea) or (
            "Test whether the selected paper idea improves a measurable source-evidence behavior."
        )
        gateway = ModelGateway()
        use_model = _should_use_model(payload.use_model)
        if not use_model:
            raise HTTPException(status_code=400, detail="Experiment generation requires a live model path.")
        result = gateway.experiment_spec(
            paper_title=payload.paper_title,
            selected_span=selected_span,
            translated_span=payload.translated_span,
            source_text=source_text,
            idea=idea,
            locale=payload.locale,
            use_model=use_model,
        )
        if result.used_fallback or result.error or not result.text:
            raise HTTPException(status_code=503, detail=result.error or "Experiment model output was unavailable.")
        implementation_repo_manifests = inspect_implementation_repositories(
            result.data.get("implementation_repositories")
            if isinstance(result.data.get("implementation_repositories"), list)
            else []
        )
        starter_result = gateway.starter_code(
            paper_title=payload.paper_title,
            selected_span=selected_span,
            source_text=source_text,
            experiment_spec=result.data,
            locale=payload.locale,
            implementation_repo_manifests=implementation_repo_manifests,
            use_model=use_model,
        )
        starter = starter_result.data.get("code") or starter_result.text
        if starter_result.used_fallback or starter_result.error or not starter:
            raise HTTPException(status_code=503, detail=starter_result.error or "Starter model output was unavailable.")
        spec_display = _experiment_spec_display(
            gateway,
            paper_title=payload.paper_title,
            spec=result.data,
            locale=payload.locale,
            use_model=use_model,
        )
        experiment_run = _issue_experiment_run(
            paper_id=payload.paper_id,
            paper_title=payload.paper_title,
            span_id=payload.span_id,
            selected_span=selected_span,
            code=starter,
            experiment_trace_id=result.trace_id,
            starter_trace_id=starter_result.trace_id,
            provider=result.provider,
            model=result.model,
            starter_provider=starter_result.provider,
            starter_model=starter_result.model,
            implementation_repo_manifests=implementation_repo_manifests,
            session_id=payload.session_id,
            workspace_id=payload.workspace_id,
        )
        return {
            "card": result.text,
            "starter": starter,
            "experimentRunId": experiment_run["id"],
            "experimentRun": experiment_run,
            "implementationRepoManifests": implementation_repo_manifests,
            "spec": result.data,
            "specDisplay": spec_display,
            "model": result.model,
            "provider": result.provider,
            "traceId": result.trace_id,
            "error": result.error,
            "usedFallback": result.used_fallback,
            "starterModel": starter_result.model,
            "starterProvider": starter_result.provider,
            "starterTraceId": starter_result.trace_id,
            "starterError": starter_result.error,
            "starterUsedFallback": starter_result.used_fallback,
        }

    @app.post("/api/starter/run")
    def run_starter(payload: StarterRunInput) -> dict[str, Any]:
        if os.getenv("PAPERLENS_ENABLE_DIAGNOSTIC_STARTER", "0").lower() not in {"1", "true", "yes"}:
            raise HTTPException(
                status_code=404,
                detail="Diagnostic starter runner is disabled in service mode. Use the bound mini-lab run endpoint.",
            )
        try:
            result = run_mini_lab_job(
                code=payload.code,
                paper_id=payload.paper_id,
                paper_title=payload.paper_title,
                span_id=payload.span_id,
                selected_span=payload.selected_span,
                provider="local",
            )
        except MiniLabError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return {
            "passed": bool(result.get("passed")),
            "reasons": result.get("reasons", []),
            "rows": result.get("rows", []),
            "provider": result.get("provider", "local"),
            "executionMode": result.get("executionMode", ""),
            "evidenceRowCount": result.get("evidenceRowCount", 0),
        }

    @app.post("/api/mini-lab/run")
    def run_mini_lab(payload: MiniLabRunInput) -> dict[str, Any]:
        bound_run = _validated_experiment_run(payload)
        try:
            return run_mini_lab_job(
                code=payload.code,
                paper_id=bound_run["paperId"],
                paper_title=bound_run["paperTitle"],
                span_id=bound_run["spanId"],
                selected_span=bound_run["selectedSpan"],
            )
        except MiniLabError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/api/growth")
    def growth_ideas(payload: GrowthInput) -> dict[str, Any]:
        resolved_paper_id = payload.paper_id or paper_key(payload.paper_title)
        persisted_memory = load_memories(resolved_paper_id) if payload.persist_memory else []
        paper_memory = [*persisted_memory, *payload.paper_memory]
        gateway = ModelGateway()
        use_model = _should_use_model(payload.use_model)
        result = gateway.growth_ideas(
            paper_title=payload.paper_title,
            paper_memory=paper_memory,
            mini_lab_result=payload.mini_lab_result,
            selected_span=payload.selected_span,
            locale=payload.locale,
            use_model=use_model,
        )
        raw_ideas = result.data.get("ideas", [])
        known_evidence_ids = _known_growth_evidence_ids(paper_memory)
        growth_eval = evaluate_growth_ideas(
            result.data,
            known_evidence_ids=known_evidence_ids,
            require_multiple_sources=True,
        )
        validation_error = "" if growth_eval.passed else "; ".join(growth_eval.reasons)
        usable_growth = (
            not result.used_fallback
            and not result.error
            and isinstance(raw_ideas, list)
            and bool(raw_ideas)
            and growth_eval.passed
        )
        if payload.persist_memory and usable_growth and payload.selected_span:
            append_memory(
                resolved_paper_id,
                kind="paper_span",
                payload={
                    "paper_title": payload.paper_title,
                    "summary": payload.selected_span[:800],
                },
            )
        if payload.persist_memory and usable_growth and payload.mini_lab_result:
            append_memory(
                resolved_paper_id,
                kind="mini_lab_result",
                payload={
                    "paper_title": payload.paper_title,
                    "summary": payload.mini_lab_result[:1200],
                },
            )
        if payload.persist_memory and usable_growth:
            for idea in raw_ideas:
                append_memory(
                    resolved_paper_id,
                    kind="growth_idea",
                    payload={
                        "paper_title": payload.paper_title,
                        "idea": idea,
                    },
                )
        display_ideas = _growth_ideas_display(
            gateway,
            paper_title=payload.paper_title,
            ideas=raw_ideas if usable_growth else [],
            locale=payload.locale,
            use_model=use_model,
        )
        return {
            "ideas": display_ideas,
            "fineTuningSignal": result.data.get("fine_tuning_signal", "none"),
            "reason": result.data.get("reason", ""),
            "paperId": resolved_paper_id,
            "memoryCount": len(load_memories(resolved_paper_id)) if payload.persist_memory else len(payload.paper_memory),
            "model": result.model,
            "provider": result.provider,
            "traceId": result.trace_id,
            "error": result.error or validation_error or None,
            "usedFallback": result.used_fallback or bool(validation_error),
        }


def _register_frontend_routes(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    def index():
        return _frontend_response("")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path.startswith(("api/", "_next/")):
            raise HTTPException(status_code=404, detail="Not found")
        return _frontend_response(path)


def _issue_experiment_run(
    *,
    paper_id: str,
    paper_title: str,
    span_id: str,
    selected_span: str,
    code: str,
    experiment_trace_id: str,
    starter_trace_id: str,
    provider: str,
    model: str,
    starter_provider: str,
    starter_model: str,
    implementation_repo_manifests: list[dict[str, Any]] | None = None,
    session_id: str,
    workspace_id: str,
) -> dict[str, Any]:
    owner = _client_owner(session_id, workspace_id)
    _prune_experiment_runs()
    run_id = f"exp_{secrets.token_urlsafe(18)}"
    now = time.time()
    binding = {
        "id": run_id,
        "createdAt": now,
        "expiresAt": now + _EXPERIMENT_RUN_TTL_SECONDS,
        **owner,
        "paperId": paper_id,
        "paperTitle": paper_title or "Untitled paper",
        "spanId": span_id,
        "selectedSpan": selected_span,
        "selectedSpanHash": text_hash(selected_span),
        "codeHash": code_hash(code),
        "experimentTraceId": experiment_trace_id,
        "starterTraceId": starter_trace_id,
        "provider": provider,
        "model": model,
        "starterProvider": starter_provider,
        "starterModel": starter_model,
        "implementationRepoManifests": implementation_repo_manifests or [],
        "usedFallback": False,
    }
    _EXPERIMENT_RUNS[run_id] = binding
    return _public_experiment_run(binding)


def _experiment_context(payload: ExperimentInput) -> dict[str, Any]:
    if not payload.paper_id or not payload.span_id:
        raise HTTPException(status_code=400, detail="Experiment requires an indexed paper id and selected span id.")
    indexed_text = get_span_text(payload.paper_id, payload.span_id)
    if not indexed_text:
        raise HTTPException(status_code=404, detail="Selected span was not found in the paper index.")
    window = evidence_window(payload.paper_id, payload.span_id)
    if not window:
        raise HTTPException(status_code=404, detail="Selected span was not found in the paper index.")
    source_text = clean_text(_indexed_paper_text(payload.paper_id) or payload.source_text or indexed_text or payload.selected_span)
    selected_span = _selected_text_from_payload(
        payload.selected_span,
        indexed_text=indexed_text,
        source_text=source_text,
    )
    return {
        "indexedText": indexed_text,
        "window": window,
        "sourceText": source_text,
        "selectedSpan": selected_span,
    }


def _indexed_paper_text(paper_id: str) -> str:
    record = load_source_index(paper_id)
    if not record:
        return ""
    spans = record.get("spans", [])
    if not isinstance(spans, list):
        return ""
    ordered_spans = sorted(
        (span for span in spans if isinstance(span, dict)),
        key=lambda span: int(span.get("position") or 0),
    )
    return clean_text(" ".join(str(span.get("text") or "") for span in ordered_spans))


def _client_owner(session_id: str, workspace_id: str) -> dict[str, str]:
    session = clean_text(session_id)
    workspace = clean_text(workspace_id)
    if not session or not workspace:
        raise HTTPException(status_code=403, detail="Lab workspace session is required. Reload the reader and try again.")
    if len(session) > 120 or len(workspace) > 120:
        raise HTTPException(status_code=403, detail="Lab workspace session is invalid. Reload the reader and try again.")
    return {"sessionId": session, "workspaceOwnerId": workspace}


def _assert_client_owner(binding: dict[str, Any], session_id: str, workspace_id: str) -> None:
    owner = _client_owner(session_id, workspace_id)
    if binding.get("sessionId") != owner["sessionId"] or binding.get("workspaceOwnerId") != owner["workspaceOwnerId"]:
        raise HTTPException(status_code=403, detail="This Lab workspace belongs to another browser session. Regenerate it in this tab.")


def _issue_candidate_set(
    *,
    paper_id: str,
    paper_title: str,
    span_id: str,
    selected_span: str,
    source_text: str,
    question: str,
    candidates: list[dict[str, Any]],
    recommended_candidate_id: str,
    reproduction_level: str,
    trace_id: str,
    provider: str,
    model: str,
    implementation_links: list[dict[str, str]] | None = None,
    session_id: str,
    workspace_id: str,
) -> dict[str, Any]:
    _prune_experiment_runs()
    owner = _client_owner(session_id, workspace_id)
    set_id = f"cand_{secrets.token_urlsafe(18)}"
    now = time.time()
    binding = {
        "id": set_id,
        "createdAt": now,
        "expiresAt": now + _EXPERIMENT_RUN_TTL_SECONDS,
        **owner,
        "paperId": paper_id,
        "paperTitle": paper_title or "Untitled paper",
        "spanId": span_id,
        "selectedSpan": selected_span,
        "selectedSpanHash": text_hash(selected_span),
        "sourceText": source_text,
        "sourceHash": text_hash(source_text),
        "question": question,
        "reproductionLevel": reproduction_level,
        "candidates": candidates,
        "recommendedCandidateId": recommended_candidate_id,
        "candidateTraceId": trace_id,
        "provider": provider,
        "model": model,
        "implementationLinks": implementation_links or [],
    }
    _CANDIDATE_SETS[set_id] = binding
    return _public_candidate_set(binding)


def _public_candidate_set(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": binding["id"],
        "paperId": binding["paperId"],
        "paperTitle": binding["paperTitle"],
        "spanId": binding["spanId"],
        "selectedSpanHash": binding["selectedSpanHash"],
        "sourceHash": binding["sourceHash"],
        "question": binding["question"],
        "reproductionLevel": binding.get("reproductionLevel", "probe"),
        "candidates": binding["candidates"],
        "recommendedCandidateId": binding["recommendedCandidateId"],
        "candidateTraceId": binding["candidateTraceId"],
        "provider": binding["provider"],
        "model": binding["model"],
        "implementationLinks": binding.get("implementationLinks", []),
        "expiresAt": binding["expiresAt"],
    }


def _validated_candidate_set(payload: GpuScriptInput) -> dict[str, Any]:
    _prune_experiment_runs()
    binding = _CANDIDATE_SETS.get(payload.candidate_set_id.strip())
    if not binding:
        raise HTTPException(status_code=403, detail="Experiment candidate set was not found or expired. Regenerate candidates.")
    _assert_client_owner(binding, payload.session_id, payload.workspace_id)
    mismatch_reasons = []
    if payload.paper_id != binding["paperId"]:
        mismatch_reasons.append("paper id")
    if payload.span_id != binding["spanId"]:
        mismatch_reasons.append("span id")
    if clean_text(payload.selected_span) != binding["selectedSpan"]:
        mismatch_reasons.append("selected span")
    if _validated_reproduction_level(payload.reproduction_level) != binding.get("reproductionLevel", "probe"):
        mismatch_reasons.append("reproduction level")
    if mismatch_reasons:
        raise HTTPException(
            status_code=403,
            detail=f"GPU script approval does not match the generated candidates: {', '.join(mismatch_reasons)}.",
        )
    return binding


def _candidate_from_set(candidate_set: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in candidate_set.get("candidates", []):
        if isinstance(candidate, dict) and str(candidate.get("id") or "") == candidate_id:
            return candidate
    raise HTTPException(status_code=404, detail="Approved experiment candidate was not found.")


def _candidate_implementation_repositories(
    candidate: dict[str, Any],
    candidate_set: dict[str, Any],
) -> list[dict[str, str]]:
    implementation = candidate.get("implementation") if isinstance(candidate.get("implementation"), dict) else {}
    repo_url = str(implementation.get("repo_url") or "").strip()
    if not repo_url:
        return []
    approved_links = [
        item
        for item in candidate_set.get("implementationLinks", [])
        if isinstance(item, dict) and str(item.get("url") or "").strip().lower() == repo_url.lower()
    ]
    if not approved_links:
        return []
    approved = dict(approved_links[0])
    approved["usage"] = str(implementation.get("reason") or approved.get("usage") or "approved GPU probe implementation context")
    return [approved]


def _exact_reproduction_blocker(
    reproduction_level: str,
    implementation_repo_manifests: list[dict[str, Any]] | None,
    locale: str,
) -> str:
    if reproduction_level != "exact":
        return ""
    if os.getenv("PAPERLENS_ENABLE_EXACT_REPO_RUNNER", "").strip().lower() not in {"1", "true", "yes"}:
        if locale == "ko":
            return "Exact 재현은 repo/config/dataset을 실제로 실행하는 별도 sandbox runner가 필요합니다. 현재 runner에서는 Probe로 진행하세요."
        return "Exact reproduction requires the repo/config/dataset sandbox runner. Use Probe with the current GPU runner."
    inspected = [
        manifest
        for manifest in implementation_repo_manifests or []
        if isinstance(manifest, dict) and manifest.get("status") == "inspected" and str(manifest.get("url") or "")
    ]
    if inspected:
        return ""
    if locale == "ko":
        return "Exact 재현은 논문에 나온 구현 저장소를 실제로 확인한 뒤에만 실행할 수 있습니다. 지금은 Probe로 진행하세요."
    return "Exact reproduction requires an inspected implementation repository from the paper. Use Probe for this paper direction."


def _issue_gpu_probe_run(
    *,
    candidate_set: dict[str, Any],
    candidate: dict[str, Any],
    code: str,
    gpu_trace_id: str,
    provider: str,
    model: str,
    script_data: dict[str, Any],
    implementation_repo_manifests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _prune_experiment_runs()
    run_id = f"gpu_{secrets.token_urlsafe(18)}"
    now = time.time()
    binding = {
        "id": run_id,
        "createdAt": now,
        "expiresAt": now + _EXPERIMENT_RUN_TTL_SECONDS,
        "sessionId": candidate_set["sessionId"],
        "workspaceOwnerId": candidate_set["workspaceOwnerId"],
        "candidateSetId": candidate_set["id"],
        "candidateId": candidate["id"],
        "candidate": candidate,
        "paperId": candidate_set["paperId"],
        "paperTitle": candidate_set["paperTitle"],
        "spanId": candidate_set["spanId"],
        "selectedSpan": candidate_set["selectedSpan"],
        "selectedSpanHash": candidate_set["selectedSpanHash"],
        "sourceHash": candidate_set["sourceHash"],
        "code": code,
        "codeHash": gpu_code_hash(code),
        "candidateTraceId": candidate_set["candidateTraceId"],
        "gpuTraceId": gpu_trace_id,
        "provider": provider,
        "model": model,
        "reproductionLevel": _candidate_reproduction_level(candidate, fallback=candidate_set.get("reproductionLevel", "probe")),
        "requestedReproductionLevel": candidate_set.get("reproductionLevel", "probe"),
        "scriptData": script_data,
        "implementationRepoManifests": implementation_repo_manifests or [],
    }
    binding["workspaceId"] = f"sandbox_{run_id.removeprefix('gpu_')}"
    binding["workspace"] = _gpu_script_workspace(binding)
    _GPU_PROBE_RUNS[run_id] = binding
    _persist_sandbox_workspace(binding["workspace"])
    return _public_gpu_probe_run(binding)


def _public_gpu_probe_run(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": binding["id"],
        "candidateSetId": binding["candidateSetId"],
        "candidateId": binding["candidateId"],
        "paperId": binding["paperId"],
        "paperTitle": binding["paperTitle"],
        "spanId": binding["spanId"],
        "selectedSpanHash": binding["selectedSpanHash"],
        "sourceHash": binding["sourceHash"],
        "codeHash": binding["codeHash"],
        "candidateTraceId": binding["candidateTraceId"],
        "gpuTraceId": binding["gpuTraceId"],
        "provider": binding["provider"],
        "model": binding["model"],
        "reproductionLevel": binding.get("reproductionLevel", "probe"),
        "requestedReproductionLevel": binding.get("requestedReproductionLevel", binding.get("reproductionLevel", "probe")),
        "implementationRepoManifests": binding.get("implementationRepoManifests", []),
        "workspaceId": binding.get("workspaceId", binding["id"]),
        "workspace": binding.get("workspace", {}),
        "expiresAt": binding["expiresAt"],
    }


def _gpu_script_workspace(binding: dict[str, Any]) -> dict[str, Any]:
    script_data = binding.get("scriptData") if isinstance(binding.get("scriptData"), dict) else {}
    candidate = binding.get("candidate") if isinstance(binding.get("candidate"), dict) else {}
    plan = script_data.get("reproduction_plan") if isinstance(script_data.get("reproduction_plan"), dict) else {}
    dataset = script_data.get("dataset") if isinstance(script_data.get("dataset"), dict) else {}
    config_payload = {
        "paperTitle": binding.get("paperTitle", ""),
        "spanId": binding.get("spanId", ""),
        "candidateId": binding.get("candidateId", ""),
        "candidateTitle": candidate.get("title", ""),
        "reproductionLevel": binding.get("reproductionLevel", "probe"),
        "requestedReproductionLevel": binding.get("requestedReproductionLevel", "probe"),
        "dataset": dataset,
        "reproductionPlan": plan,
        "expectedOutputs": script_data.get("expected_outputs", []),
        "paperClaimComparisonPlan": script_data.get("paper_claim_comparison_plan", ""),
        "limitations": script_data.get("limitations", []),
        "codeHash": binding.get("codeHash", ""),
        "evidenceHash": binding.get("sourceHash", ""),
    }
    manifest_payload = {
        "workspaceId": binding.get("workspaceId", binding.get("id", "")),
        "gpuRunId": binding.get("id", ""),
        "candidateSetId": binding.get("candidateSetId", ""),
        "provider": binding.get("provider", ""),
        "model": binding.get("model", ""),
        "implementationRepoManifests": binding.get("implementationRepoManifests", []),
        **config_payload,
    }
    run_command = clean_text(str(plan.get("command") or "Run from PaperLens with the approved GPU run binding."))
    files = [
        {
            "path": "experiment.py",
            "language": "python",
            "role": "entrypoint",
            "content": str(binding.get("code") or ""),
        },
        {
            "path": "config.json",
            "language": "json",
            "role": "configuration",
            "content": json.dumps(config_payload, ensure_ascii=False, indent=2, sort_keys=True),
        },
        {
            "path": "run.sh",
            "language": "shell",
            "role": "command",
            "content": "# Executed through the approved PaperLens Modal GPU binding.\n" f"# Paper/source-bound command: {run_command}\n",
        },
        {
            "path": "manifest.json",
            "language": "json",
            "role": "provenance",
            "content": json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True),
        },
    ]
    return {
        "id": binding.get("workspaceId", binding.get("id", "")),
        "status": "script_ready",
        "title": candidate.get("title") or "PaperLens sandbox workspace",
        "paperTitle": binding.get("paperTitle", ""),
        "reproductionLevel": binding.get("reproductionLevel", "probe"),
        "requestedReproductionLevel": binding.get("requestedReproductionLevel", "probe"),
        "plan": plan,
        "dataset": dataset,
        "files": files,
        "provenance": {
            "codeHash": binding.get("codeHash", ""),
            "sourceHash": binding.get("sourceHash", ""),
            "candidateTraceId": binding.get("candidateTraceId", ""),
            "gpuTraceId": binding.get("gpuTraceId", ""),
            "implementationRepoManifests": binding.get("implementationRepoManifests", []),
        },
    }


def _validated_gpu_probe_run(payload: GpuProbeRunInput) -> dict[str, Any]:
    _prune_experiment_runs()
    run_id = payload.gpu_run_id.strip()
    if not run_id:
        raise HTTPException(status_code=403, detail="GPU execution requires an approved GPU run id.")
    binding = _GPU_PROBE_RUNS.get(run_id)
    if not binding:
        raise HTTPException(status_code=403, detail="GPU run id was not found or expired. Regenerate the script.")
    _assert_client_owner(binding, payload.session_id, payload.workspace_id)
    return binding


def _persist_sandbox_workspace(workspace: dict[str, Any]) -> None:
    workspace_id = clean_text(str(workspace.get("id") or ""))
    if not workspace_id:
        return
    try:
        workspace_dir = _sandbox_workspace_dir() / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "workspace.json").write_text(
            json.dumps(workspace, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        return


def _persist_sandbox_result(binding: dict[str, Any], result: dict[str, Any]) -> None:
    workspace_id = clean_text(str(binding.get("workspaceId") or ""))
    if not workspace_id:
        return
    try:
        workspace_dir = _sandbox_workspace_dir() / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        return


def _sandbox_workspace_dir() -> Path:
    configured = os.getenv("PAPERLENS_SANDBOX_WORKSPACE_DIR", "").strip()
    return Path(configured) if configured else DEFAULT_SANDBOX_WORKSPACE_DIR


def _validated_experiment_run(payload: MiniLabRunInput) -> dict[str, Any]:
    _prune_experiment_runs()
    run_id = payload.experiment_run_id.strip()
    if not run_id:
        raise HTTPException(status_code=403, detail="Mini-lab execution requires a generated experiment run id.")
    binding = _EXPERIMENT_RUNS.get(run_id)
    if not binding:
        raise HTTPException(status_code=403, detail="Experiment run id was not found or expired. Regenerate the experiment.")
    _assert_client_owner(binding, payload.session_id, payload.workspace_id)

    mismatch_reasons = []
    if payload.paper_id != binding["paperId"]:
        mismatch_reasons.append("paper id")
    if payload.span_id != binding["spanId"]:
        mismatch_reasons.append("span id")
    if clean_text(payload.selected_span) != binding["selectedSpan"]:
        mismatch_reasons.append("selected span")
    if code_hash(payload.code) != binding["codeHash"]:
        mismatch_reasons.append("starter code")
    if mismatch_reasons:
        raise HTTPException(
            status_code=403,
            detail=f"Mini-lab execution does not match the generated experiment run: {', '.join(mismatch_reasons)}.",
        )
    return binding


def _public_experiment_run(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": binding["id"],
        "paperId": binding["paperId"],
        "paperTitle": binding["paperTitle"],
        "spanId": binding["spanId"],
        "selectedSpanHash": binding["selectedSpanHash"],
        "codeHash": binding["codeHash"],
        "experimentTraceId": binding["experimentTraceId"],
        "starterTraceId": binding["starterTraceId"],
        "provider": binding["provider"],
        "model": binding["model"],
        "starterProvider": binding["starterProvider"],
        "starterModel": binding["starterModel"],
        "implementationRepoManifests": binding.get("implementationRepoManifests", []),
        "expiresAt": binding["expiresAt"],
    }


def _prune_experiment_runs() -> None:
    now = time.time()
    expired = [run_id for run_id, binding in _EXPERIMENT_RUNS.items() if float(binding.get("expiresAt") or 0) <= now]
    for run_id in expired:
        _EXPERIMENT_RUNS.pop(run_id, None)
    expired_candidate_sets = [
        set_id for set_id, binding in _CANDIDATE_SETS.items() if float(binding.get("expiresAt") or 0) <= now
    ]
    for set_id in expired_candidate_sets:
        _CANDIDATE_SETS.pop(set_id, None)
    expired_gpu_runs = [
        run_id for run_id, binding in _GPU_PROBE_RUNS.items() if float(binding.get("expiresAt") or 0) <= now
    ]
    for run_id in expired_gpu_runs:
        _GPU_PROBE_RUNS.pop(run_id, None)
    if len(_EXPERIMENT_RUNS) <= _MAX_EXPERIMENT_RUNS:
        experiment_overflow = 0
    else:
        experiment_overflow = max(0, len(_EXPERIMENT_RUNS) - _MAX_EXPERIMENT_RUNS)
    ordered = sorted(
        _EXPERIMENT_RUNS.items(),
        key=lambda item: float(item[1].get("createdAt") or 0),
    )
    for run_id, _binding in ordered[:experiment_overflow]:
        _EXPERIMENT_RUNS.pop(run_id, None)
    for store in (_CANDIDATE_SETS, _GPU_PROBE_RUNS):
        if len(store) <= _MAX_EXPERIMENT_RUNS:
            continue
        ordered_store = sorted(
            store.items(),
            key=lambda item: float(item[1].get("createdAt") or 0),
        )
        for key, _binding in ordered_store[: max(0, len(store) - _MAX_EXPERIMENT_RUNS)]:
            store.pop(key, None)


def _known_growth_evidence_ids(paper_memory: list[dict[str, Any]]) -> set[str]:
    ids = {"run:r1"}
    for item in paper_memory:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("id") or "").strip()
        if evidence_id:
            ids.add(evidence_id)
    return ids


def _frontend_ready() -> bool:
    return (FRONTEND_OUT_DIR / "index.html").exists()


def _frontend_response(path: str) -> FileResponse | HTMLResponse:
    if not _frontend_ready():
        return HTMLResponse(_missing_frontend_html(), status_code=503)

    requested = path.strip("/")
    candidates = []
    if not requested:
        candidates.append(FRONTEND_OUT_DIR / "index.html")
    else:
        candidates.extend(
            [
                FRONTEND_OUT_DIR / requested,
                FRONTEND_OUT_DIR / requested / "index.html",
                FRONTEND_OUT_DIR / f"{requested}.html",
            ]
        )

    for candidate in candidates:
        try:
            candidate.relative_to(FRONTEND_OUT_DIR)
        except ValueError:
            continue
        if candidate.is_file():
            return FileResponse(candidate)

    return FileResponse(FRONTEND_OUT_DIR / "index.html")


def _missing_frontend_html() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>PaperLens Lab</title>
        <style>
          body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #0f172a; }
          main { min-height: 100vh; display: grid; place-items: center; padding: 24px; }
          section { max-width: 640px; border: 1px solid #e2e8f0; border-radius: 12px; background: white; padding: 28px; box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08); }
          a { color: #0f766e; font-weight: 700; }
          code { border-radius: 6px; background: #f1f5f9; padding: 2px 6px; }
        </style>
      </head>
      <body>
        <main>
          <section>
            <h1>PaperLens Lab frontend is not built yet</h1>
            <p>Run <code>cd frontend && npm ci && npm run build</code>, then restart <code>python app.py</code>. For the Hugging Face Space, sync the generated <code>frontend/out/</code> files with the app.</p>
          </section>
        </main>
      </body>
    </html>
    """


def paper_document_from_source(
    source: PaperSource,
    *,
    use_model: bool = False,
    max_translate_spans: int = 24,
    max_reader_spans: int = 180,
    gateway: ModelGateway | None = None,
) -> dict[str, Any]:
    sentences = split_sentences(source.text)
    if not sentences:
        sentences = [source.text]
    total_sentences = len(sentences)
    reader_limit = max(12, min(max_reader_spans, 1200))
    sentences = sentences[:reader_limit]

    section_count = min(10, max(1, (len(sentences) + 17) // 18))
    section_plans = []
    sections = []
    cursor = 0
    for section_index in range(section_count):
        remaining = len(sentences) - cursor
        take = max(1, (remaining + section_count - section_index - 1) // (section_count - section_index))
        section_sentences = sentences[cursor : cursor + take]
        cursor += take
        section_plans.append((section_index, section_sentences))

    span_sources = [
        {"span_id": _span_id(section_index, span_index + 1), "text": sentence}
        for section_index, section_sentences in section_plans
        for span_index, sentence in enumerate(section_sentences)
    ][: max(1, min(max_translate_spans, 96))]
    translation_records = _translation_records(source.title, span_sources, use_model, gateway=gateway)

    for section_index, section_sentences in section_plans:
        paragraph_spans = []
        for span_index, sentence in enumerate(section_sentences):
            span_id = _span_id(section_index, span_index + 1)
            record = translation_records.get(span_id, {})
            translated = str(record.get("translation") or "") if record else ""
            paragraph_spans.append(
                {
                    "id": span_id,
                    "original": sentence,
                    "translated": translated or _translation_placeholder(sentence),
                    "translationStatus": str(record.get("status") or "draft") if record else "draft",
                }
            )
        sections.append(
            {
                "id": f"sec-{section_index + 1}",
                "title": "Loaded Paper" if section_index == 0 else f"Source Extract {section_index + 1}",
                "titleKo": "불러온 논문" if section_index == 0 else f"원문 추출 {section_index + 1}",
                "paragraphs": [{"id": f"P{section_index}", "spans": paragraph_spans}],
            }
        )

    document = {
        "id": _document_id_from_source(source),
        "title": source.title or "Untitled paper",
        "titleKo": "제목 없는 논문" if not source.title or source.title == "Untitled paper" else source.title,
        "authors": [item.strip() for item in source.authors.split(",") if item.strip()] or ["Unknown authors"],
        "source": source.source_label,
        "sections": sections,
        "model": DEFAULT_MODEL if use_model else "fallback-extractive",
        "translationModel": TRANSLATION_MODEL if use_model else "fallback-extractive",
        "provider": DEFAULT_PROVIDER if use_model else "fallback",
        "metadata": {
            "pdfUrl": source.pdf_url,
            "warnings": list(source.warnings),
            "totalSentenceCount": total_sentences,
            "readerSpanCount": len(sentences),
            "readerSpanLimit": reader_limit,
            "translatedSpanCount": sum(1 for record in translation_records.values() if record.get("status") == "ready"),
            "sourceTextChars": len(source.text),
        },
    }
    save_source_index(
        document["id"],
        title=document["title"],
        source_label=source.source_label,
        pdf_url=source.pdf_url,
        source_text=source.text,
        sections=sections,
    )
    return document


def _paper_payload_text(payload: PaperInput) -> str:
    if payload.pasted_text.strip():
        return payload.pasted_text
    return ""


def _translation_placeholder(sentence: str) -> str:
    return f"[초안 번역] {sentence}"


def _translation_records(
    title: str,
    spans: list[dict[str, str]],
    use_model: bool,
    *,
    gateway: ModelGateway | None = None,
) -> dict[str, dict[str, Any]]:
    if not spans:
        return {}
    records: dict[str, dict[str, Any]] = {}
    gateway = gateway or ModelGateway()
    batch_size = _translation_batch_size()
    for index in range(0, len(spans), batch_size):
        batch = spans[index : index + batch_size]
        result = gateway.translate_spans(title, batch, locale="ko", use_model=use_model)
        for item in result.data.get("translations", []):
            span_id = item.get("span_id", "")
            if span_id:
                translation = str(item.get("translation", ""))
                records[str(span_id)] = {
                    "translation": translation,
                    "status": _translation_status(translation, bool(getattr(result, "used_fallback", False))),
                }
    return records


def _prepare_translation_requests(payload: TranslationInput, gateway: ModelGateway) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for request_index, item in enumerate(payload.spans):
        span_id = str(item.get("span_id", "")).strip()
        indexed_text = get_span_text(payload.paper_id, span_id) if payload.paper_id and span_id else ""
        if payload.paper_id and span_id and not indexed_text:
            raise HTTPException(status_code=404, detail=f"Selected span was not found in the paper index: {span_id}")
        source_text = clean_text(indexed_text or str(item.get("text", "")))
        if not source_text:
            raise HTTPException(status_code=400, detail=f"Translation source text is missing for span: {span_id or request_index}")
        prepared.append(
            {
                "request_index": request_index,
                "paper_id": payload.paper_id,
                "span_id": span_id or f"req-{request_index}",
                "source_text": source_text,
                "source_hash": text_hash(source_text),
                "source_index_bound": bool(indexed_text),
                "cached_translation": get_cached_translation(
                    payload.paper_id,
                    span_id,
                    source_text,
                    locale=payload.locale,
                    model=gateway.translation_model_id,
                )
                if payload.paper_id and span_id
                else "",
            }
        )
    return prepared


def _translation_batch_size() -> int:
    raw = os.getenv("PAPERLENS_TRANSLATION_BATCH_SIZE", "4")
    try:
        value = int(raw)
    except ValueError:
        value = 4
    return max(1, min(value, 12))


def _translation_status(translation: str, used_fallback: bool) -> str:
    if not translation or used_fallback or _is_draft_translation(translation):
        return "fallback"
    return "ready"


def _selected_text_from_payload(
    requested_text: str,
    *,
    indexed_text: str,
    source_text: str,
) -> str:
    requested = clean_text(requested_text)
    indexed = clean_text(indexed_text)
    source = clean_text(source_text)
    if requested and indexed and requested != indexed:
        if source_contains_quote(source or indexed, requested):
            return requested
        if source_contains_quote(indexed, requested):
            return requested
    return indexed or requested


def _validated_selected_segments(payload: AskInput) -> list[dict[str, Any]]:
    if not payload.selected_spans:
        return []
    if not payload.paper_id:
        raise HTTPException(status_code=400, detail="Selected span ranges require an indexed paper.")
    record = load_source_index(payload.paper_id)
    if not record:
        raise HTTPException(status_code=404, detail="Selected paper index was not found.")
    spans_by_id = {
        str(span.get("span_id") or ""): clean_text(str(span.get("text") or ""))
        for span in record.get("spans", [])
        if span.get("span_id")
    }
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(payload.selected_spans, start=1):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"Selected span range {index} is invalid.")
        span_id = clean_text(str(item.get("span_id") or item.get("spanId") or ""))
        selected_text = clean_text(str(item.get("text") or ""))
        surface = clean_text(str(item.get("surface") or "original")) or "original"
        if surface != "original":
            continue
        indexed_text = spans_by_id.get(span_id, "")
        if not span_id or not indexed_text:
            raise HTTPException(status_code=400, detail=f"Selected span range {index} does not match the paper index.")
        start = _optional_int(item.get("start_offset", item.get("startOffset")))
        end = _optional_int(item.get("end_offset", item.get("endOffset")))
        if start is not None and end is not None and 0 <= start < end <= len(indexed_text):
            exact = clean_text(indexed_text[start:end])
            if selected_text and not source_contains_quote(exact, selected_text):
                raise HTTPException(status_code=400, detail=f"Selected span range {index} text does not match the paper index.")
            selected_text = exact
        elif selected_text and not source_contains_quote(indexed_text, selected_text):
            raise HTTPException(status_code=400, detail=f"Selected span range {index} text does not match the paper index.")
        if selected_text:
            validated.append(
                {
                    "span_id": span_id,
                    "text": selected_text,
                    "surface": surface,
                    "start_offset": start,
                    "end_offset": end,
                }
            )
    return validated


def selected_evidence_window(paper_id: str, selected_segments: list[dict[str, Any]], *, radius: int = 3) -> dict[str, Any] | None:
    record = load_source_index(paper_id)
    if not record or not selected_segments:
        return None
    spans = record.get("spans", [])
    position_by_id = {
        str(span.get("span_id") or ""): idx
        for idx, span in enumerate(spans)
        if span.get("span_id")
    }
    selected_positions = [
        position_by_id[segment["span_id"]]
        for segment in selected_segments
        if segment.get("span_id") in position_by_id
    ]
    if not selected_positions:
        return None
    start = max(0, min(selected_positions) - radius)
    end = min(len(spans), max(selected_positions) + radius + 1)
    window_spans = spans[start:end]
    return {
        "paper_id": paper_id,
        "span_id": selected_segments[0]["span_id"],
        "span_range": f"{window_spans[0]['span_id']}-{window_spans[-1]['span_id']}" if window_spans else selected_segments[0]["span_id"],
        "source_hash": record.get("source_text_hash", ""),
        "text": " ".join(span.get("text", "") for span in window_spans),
        "spans": window_spans,
        "selected_spans": selected_segments,
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _span_id(section_index: int, span_index: int) -> str:
    return f"P{section_index}.S{span_index}"


def _should_use_model(requested: bool) -> bool:
    return requested or _force_model_enabled()


def _public_gpu_script_error(locale: str, error: str) -> str:
    if locale == "ko":
        return "모델이 생성한 GPU 스크립트가 서비스 실행 검증을 통과하지 못했습니다. 후보를 다시 생성하거나 다른 후보를 승인해 주세요."
    return "The model-generated GPU script did not pass service execution checks. Regenerate candidates or approve a different candidate."


def _public_experiment_candidates_error(locale: str, error: str) -> str:
    if locale == "ko":
        return "모델이 이 논문에서 실행 가능한 연구 방향을 확정하지 못했습니다. 질문을 좁히거나 다시 시도해 주세요."
    return "The model could not finalize paper-grounded research directions. Refine the question or try again."


def _force_model_enabled() -> bool:
    return os.getenv("PAPERLENS_FORCE_MODEL", "").lower() in {"1", "true", "yes"}


def _is_draft_translation(text: str) -> bool:
    stripped = text.strip()
    return not stripped or stripped.startswith("[초안 번역]") or stripped.startswith("[Korean draft pending]")


def _display_text_needs_translation(value: str) -> bool:
    stripped = clean_text(value)
    if not stripped or not re.search(r"[A-Za-z]", stripped):
        return False
    return any(token in stripped for token in (" ", "·", ":", ",", ".", "-", "/", "(", ")", "_"))


def _translate_display_fragments(
    gateway: ModelGateway,
    *,
    paper_title: str,
    entries: list[tuple[str, str]],
    locale: str,
    use_model: bool,
) -> dict[str, str]:
    if locale != "ko" or not use_model or not entries:
        return {}
    translated_by_id: dict[str, str] = {}
    for start in range(0, len(entries), 6):
        batch = entries[start : start + 6]
        result = gateway.translate_spans(
            paper_title,
            [{"span_id": key, "text": value} for key, value in batch],
            locale=locale,
            use_model=use_model,
        )
        if result.used_fallback:
            continue
        for item in result.data.get("translations", []):
            if not isinstance(item, dict):
                continue
            span_id = str(item.get("span_id", ""))
            translation = clean_text(str(item.get("translation", "")))
            if span_id and _translation_status(translation, False) == "ready":
                translated_by_id[span_id] = translation
    return translated_by_id


def _experiment_spec_display(
    gateway: ModelGateway,
    *,
    paper_title: str,
    spec: dict[str, Any],
    locale: str,
    use_model: bool,
) -> dict[str, Any] | None:
    if locale != "ko" or not use_model or not isinstance(spec, dict):
        return None
    display = dict(spec)
    dataset_value = spec.get("dataset")
    if isinstance(dataset_value, dict):
        display["dataset"] = dict(dataset_value)
    display["steps"] = list(spec.get("steps", [])) if isinstance(spec.get("steps"), list) else []
    display["faithfulness_notes"] = (
        list(spec.get("faithfulness_notes", [])) if isinstance(spec.get("faithfulness_notes"), list) else []
    )
    entries: list[tuple[str, str]] = []
    for key in (
        "research_question",
        "mini_lab_goal",
        "metric",
        "baseline",
        "ablation",
        "failure_condition",
        "expected_result",
    ):
        value = str(spec.get(key, ""))
        if _display_text_needs_translation(value):
            entries.append((key, value))
    if isinstance(dataset_value, dict):
        dataset_name = str(dataset_value.get("name", ""))
        dataset_source = str(dataset_value.get("source") or dataset_value.get("fallback") or "")
        if _display_text_needs_translation(dataset_name):
            entries.append(("dataset:name", dataset_name))
        if _display_text_needs_translation(dataset_source):
            entries.append(("dataset:source", dataset_source))
    else:
        dataset_text = str(dataset_value or "")
        if _display_text_needs_translation(dataset_text):
            entries.append(("dataset", dataset_text))
    for index, step in enumerate(display["steps"]):
        if isinstance(step, str) and _display_text_needs_translation(step):
            entries.append((f"steps:{index}", step))
    for index, note in enumerate(display["faithfulness_notes"]):
        if isinstance(note, str) and _display_text_needs_translation(note):
            entries.append((f"faithfulness_notes:{index}", note))
    translations = _translate_display_fragments(
        gateway,
        paper_title=paper_title,
        entries=entries,
        locale=locale,
        use_model=use_model,
    )
    if not translations:
        return None
    for key, translation in translations.items():
        if key.startswith("steps:"):
            index = int(key.split(":", 1)[1])
            if 0 <= index < len(display["steps"]):
                display["steps"][index] = translation
            continue
        if key.startswith("faithfulness_notes:"):
            index = int(key.split(":", 1)[1])
            if 0 <= index < len(display["faithfulness_notes"]):
                display["faithfulness_notes"][index] = translation
            continue
        if key.startswith("dataset:") and isinstance(display.get("dataset"), dict):
            dataset_key = key.split(":", 1)[1]
            if dataset_key in {"name", "source", "fallback"}:
                display["dataset"][dataset_key] = translation
            continue
        display[key] = translation
    return display


def _growth_ideas_display(
    gateway: ModelGateway,
    *,
    paper_title: str,
    ideas: list[dict[str, Any]],
    locale: str,
    use_model: bool,
) -> list[dict[str, Any]]:
    normalized = [dict(item) for item in ideas if isinstance(item, dict)]
    if locale != "ko" or not use_model or not normalized:
        return normalized
    entries: list[tuple[str, str]] = []
    for index, idea in enumerate(normalized):
        text = str(idea.get("idea", ""))
        if _display_text_needs_translation(text):
            entries.append((f"idea:{index}", text))
    translations = _translate_display_fragments(
        gateway,
        paper_title=paper_title,
        entries=entries,
        locale=locale,
        use_model=use_model,
    )
    if not translations:
        return normalized
    for key, translation in translations.items():
        if not key.startswith("idea:"):
            continue
        index = int(key.split(":", 1)[1])
        if 0 <= index < len(normalized):
            normalized[index]["displayIdea"] = translation
    return normalized


def _support_ids(data: dict[str, Any], fallback_span_id: str) -> list[str]:
    evidence = data.get("evidence", [])
    ids = [item.get("source_id") for item in evidence if isinstance(item, dict) and item.get("source_id")]
    support_span_ids = data.get("support_span_ids", [])
    ids.extend(item for item in support_span_ids if isinstance(item, str))
    return list(dict.fromkeys(ids or [fallback_span_id]))


def _validated_answer_data(
    data: dict[str, Any],
    payload: AskInput,
    *,
    evidence_text: str = "",
    allowed_source_ids: set[str] | None = None,
    selected_span_text: str = "",
    source_text_by_id: dict[str, str] | None = None,
    selected_segments: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str | None]:
    if not isinstance(data, dict):
        return _insufficient_answer(payload, selected_span_text=selected_span_text), "answer payload is not structured JSON"

    evidence = data.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        return _insufficient_answer(payload, selected_span_text=selected_span_text), "answer evidence is missing"

    source_pool = f"{selected_span_text or payload.original}\n\n{evidence_text or payload.source_text}"
    for item in evidence:
        if not isinstance(item, dict):
            return _insufficient_answer(payload, selected_span_text=selected_span_text), "answer evidence is not structured"
        source_id = str(item.get("source_id", ""))
        if allowed_source_ids is not None and source_id not in allowed_source_ids:
            return (
                _insufficient_answer(payload, selected_span_text=selected_span_text),
                f"answer source id is outside the selected evidence window: {source_id}",
            )
        quote = clean_text(str(item.get("quote", "")))
        quote_source = (
            (source_text_by_id or {}).get(source_id, "")
            if source_id
            else ""
        )
        if quote and quote_source and not source_contains_quote(quote_source, quote):
            if selected_segments and source_contains_quote(selected_span_text, quote):
                repaired = dict(data)
                repaired["evidence"] = _selected_segment_answer_evidence_items(selected_segments)
                repaired["support_span_ids"] = [
                    segment["span_id"]
                    for segment in selected_segments
                    if segment.get("span_id")
                ]
                return repaired, None
            return (
                _insufficient_answer(payload, selected_span_text=selected_span_text),
                f"answer quote does not match the cited source id: {source_id}",
            )
        if quote and not quote_source and not source_contains_quote(source_pool, quote):
            return (
                _insufficient_answer(payload, selected_span_text=selected_span_text),
                f"answer quote is not present in source evidence: {item.get('source_id', '')}",
            )
    return data, None


def _window_evidence_items(window: dict[str, Any] | None) -> list[dict[str, str]]:
    if not window:
        return []
    items = []
    for span in window.get("spans", []):
        span_id = str(span.get("span_id", ""))
        text = clean_text(str(span.get("text", "")))
        if span_id and text:
            items.append({"source_id": span_id, "text": text})
    return items


def _selected_segment_evidence_items(selected_segments: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    items = []
    for segment in selected_segments or []:
        span_id = clean_text(str(segment.get("span_id") or ""))
        text = clean_text(str(segment.get("text") or ""))
        if span_id and text:
            items.append({"source_id": span_id, "text": text})
    return items


def _selected_segment_answer_evidence_items(selected_segments: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    items = []
    for segment in selected_segments or []:
        span_id = clean_text(str(segment.get("span_id") or ""))
        text = clean_text(str(segment.get("text") or ""))
        if span_id and text:
            items.append({"source_id": span_id, "quote": text})
    return items


def _source_text_by_evidence_id(items: list[dict[str, str]]) -> dict[str, str]:
    text_by_id: dict[str, list[str]] = {}
    for item in items:
        source_id = str(item.get("source_id") or "")
        text = clean_text(str(item.get("text") or ""))
        if source_id and text:
            text_by_id.setdefault(source_id, []).append(text)
    return {
        source_id: "\n\n".join(dict.fromkeys(parts))
        for source_id, parts in text_by_id.items()
    }


def _paper_evidence_items(source_text: str) -> list[dict[str, str]]:
    items = []
    for item in top_sentences(source_text, limit=10):
        text = clean_text(item.text)
        if text:
            items.append({"source_id": f"paper.S{item.pid}", "text": text})
    if not items and source_text.strip():
        items.append({"source_id": "paper.S1", "text": clean_text(source_text)[:1200]})
    return items


def _public_evidence_window(window: dict[str, Any] | None) -> dict[str, Any] | None:
    if not window:
        return None
    spans = []
    for item in window.get("spans", []):
        spans.append(
            {
                "spanId": item.get("span_id", ""),
                "textHash": item.get("text_hash", ""),
                "position": item.get("position"),
            }
        )
    return {
        "paperId": window.get("paper_id", ""),
        "spanId": window.get("span_id", ""),
        "spanRange": window.get("span_range", ""),
        "sourceHash": window.get("source_hash", ""),
        "spans": spans,
    }


def _insufficient_answer(payload: AskInput, *, selected_span_text: str = "") -> dict[str, Any]:
    quote = (selected_span_text or payload.original)[:420]
    if payload.locale == "ko":
        answer = (
            "이 질문은 현재 확인된 원문 근거만으로는 충분히 답하기 어렵습니다. "
            "선택 문장과 실제 원문 quote가 맞는 범위 안에서 다시 확인해야 합니다."
        )
    else:
        answer = (
            "The current evidence is not sufficient to answer safely. "
            "PaperLens needs a quote that is present in the selected span or source text."
        )
    return {
        "answer": answer,
        "evidence": [{"source_id": payload.span_id or "paper", "quote": quote}],
        "confidence": "low",
        "needs_more_context": True,
        "unsupported_assumptions": ["model evidence quote failed source-substring validation"],
    }


def _ask_prompt(payload: AskInput, question: str) -> str:
    evidence = "\n".join(sentence.text for sentence in top_sentences(payload.source_text, limit=5))
    return f"""You are PaperLens Lab. Answer the user's question about a selected paper sentence.
Rules:
- Use the selected sentence and source evidence first.
- Say when something is an interpretation beyond the paper.
- Keep the answer concise.

Paper: {payload.paper_title}
Selected sentence ({payload.span_id}): {payload.original}
Available translation: {payload.translated}
Question: {question}
Evidence:
{evidence}
"""


def _fallback_answer(
    payload: AskInput,
    question: str,
    *,
    selected_span_text: str = "",
    evidence_text: str = "",
) -> str:
    resolved_span = selected_span_text or payload.original
    evidence = top_sentences(evidence_text or payload.source_text or resolved_span, limit=3)
    evidence_hint = " ".join(f"S{item.pid}" for item in evidence) or payload.span_id
    if payload.locale == "ko":
        return (
            f"모델 답변을 근거 검증까지 확정하지 못해 원문 근거만 표시합니다. "
            f"질문은 \"{question}\"이고, 핵심 원문은 \"{resolved_span[:180]}\"입니다. "
            f"근거 후보는 {evidence_hint}입니다."
        )
    return (
        "The model answer could not be confirmed against the paper evidence, so PaperLens is showing source evidence only. "
        f"For \"{question}\", the key source sentence is \"{resolved_span[:180]}\". "
        f"Candidate evidence: {evidence_hint}."
    )


def _document_id_from_source(source: PaperSource) -> str:
    base = source.source_label.replace(":", "-").replace("/", "-").lower() or "paper"
    safe_base = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in base).strip("-") or "paper"
    normalized_label = source.source_label.strip().lower()
    if normalized_label.startswith("arxiv:") or normalized_label in {"sample", "frontend-reader", "error"}:
        return safe_base
    return f"{safe_base[:80]}-{text_hash(source.text or source.pdf_url or safe_base)}"
