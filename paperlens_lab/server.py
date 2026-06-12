from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analysis import experiment_card, split_sentences, top_sentences
from .ingest import PaperSource, build_source, clean_text
from .memory_store import append_memory, load_memories, paper_key
from .model_adapter import DEFAULT_MODEL, DEFAULT_PROVIDER, ModelGateway
from .tracing import trace_content_enabled
from .ui import EXAMPLE_TEXT, build_demo


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_OUT_DIR = FRONTEND_DIR / "out"


class PaperInput(BaseModel):
    arxiv_or_url: str = ""
    pasted_text: str = ""
    max_pdf_pages: int = Field(default=10, ge=1, le=32)
    use_model: bool = False
    max_translate_spans: int = Field(default=24, ge=1, le=96)
    max_reader_spans: int = Field(default=180, ge=12, le=320)


class AskInput(BaseModel):
    span_id: str
    question: str = ""
    original: str
    translated: str = ""
    paper_title: str = "Untitled paper"
    source_text: str = ""
    locale: str = "en"
    use_model: bool = False


class ExperimentInput(BaseModel):
    paper_title: str = "Untitled paper"
    selected_span: str
    translated_span: str = ""
    source_text: str = ""
    idea: str = ""
    locale: str = "en"
    use_model: bool = False


class TranslationInput(BaseModel):
    paper_title: str = "Untitled paper"
    spans: list[dict[str, str]]
    locale: str = "ko"
    use_model: bool = False


class GrowthInput(BaseModel):
    paper_id: str = ""
    paper_title: str = "Untitled paper"
    selected_span: str = ""
    paper_memory: list[dict[str, Any]] = Field(default_factory=list)
    mini_lab_result: str = ""
    locale: str = "ko"
    use_model: bool = False
    persist_memory: bool = True


def create_app() -> FastAPI:
    app = FastAPI(
        title="PaperLens Lab",
        description="React reader frontend with a Gradio/Python model backend.",
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

    demo = build_demo()
    app = gr.mount_gradio_app(app, demo, path="/gradio")
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
            "gradio_path": "/gradio",
            "model": DEFAULT_MODEL,
            "provider": DEFAULT_PROVIDER,
            "forceModel": _force_model_enabled(),
            "traceContent": trace_content_enabled(),
            "runtime": "react-fastapi-gradio-hybrid",
        }

    @app.get("/api/sample-paper")
    def sample_paper() -> dict[str, Any]:
        source = PaperSource(
            title="Improving Retrieval-Augmented Generation with Evidence-Linked Reranking",
            authors="J. Kim, S. Park, M. Lee",
            source_label="sample",
            text=EXAMPLE_TEXT,
        )
        return paper_document_from_source(source)

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
        max_reader_spans: int = Form(180),
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
                    max_pdf_pages=max(1, min(max_pdf_pages, 32)),
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return paper_document_from_source(
            source,
            use_model=_should_use_model(use_model),
            max_translate_spans=max(1, min(max_translate_spans, 96)),
            max_reader_spans=max(12, min(max_reader_spans, 320)),
        )

    @app.post("/api/translate")
    def translate_spans(payload: TranslationInput) -> dict[str, Any]:
        gateway = ModelGateway()
        result = gateway.translate_spans(
            payload.paper_title,
            payload.spans,
            locale=payload.locale,
            use_model=_should_use_model(payload.use_model),
        )
        return {
            "translations": result.data.get("translations", []),
            "notes": result.data.get("notes", []),
            "model": result.model,
            "provider": result.provider,
            "traceId": result.trace_id,
            "error": result.error,
            "usedFallback": result.used_fallback,
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
        result = gateway.answer_span(
            paper_title=payload.paper_title,
            span_id=payload.span_id,
            selected_span=payload.original,
            translated_span=payload.translated,
            question=question,
            source_text=payload.source_text,
            locale=payload.locale,
            use_model=_should_use_model(payload.use_model),
        )
        answer_data, validation_error = _validated_answer_data(result.data, payload)
        return {
            "role": "assistant",
            "content": answer_data.get("answer") or result.text or _fallback_answer(payload, question),
            "supportSpanIds": _support_ids(answer_data, payload.span_id),
            "evidence": answer_data.get("evidence", []),
            "confidence": answer_data.get("confidence", "low"),
            "needsMoreContext": answer_data.get("needs_more_context", True),
            "model": result.model,
            "provider": result.provider,
            "traceId": result.trace_id,
            "error": validation_error or result.error,
            "usedFallback": result.used_fallback or bool(validation_error),
        }

    @app.post("/api/experiment")
    def build_experiment(payload: ExperimentInput) -> dict[str, Any]:
        source_text = clean_text(payload.source_text or payload.selected_span)
        idea = clean_text(payload.idea) or (
            "Test whether the selected paper idea improves a small measurable behavior."
        )
        source = PaperSource(
            title=payload.paper_title,
            authors="",
            source_label="frontend-reader",
            text=f"{payload.selected_span}\n\n{source_text}",
        )
        gateway = ModelGateway()
        result = gateway.experiment_spec(
            paper_title=payload.paper_title,
            selected_span=payload.selected_span,
            translated_span=payload.translated_span,
            source_text=source_text,
            idea=idea,
            locale=payload.locale,
            use_model=_should_use_model(payload.use_model),
        )
        fallback_card, starter = experiment_card(source, idea, "Research prototype builder", use_model=False)
        card = result.text if result.text else fallback_card
        return {
            "card": card,
            "starter": starter,
            "spec": result.data,
            "model": result.model,
            "provider": result.provider,
            "traceId": result.trace_id,
            "error": result.error,
            "usedFallback": result.used_fallback,
        }

    @app.post("/api/growth")
    def growth_ideas(payload: GrowthInput) -> dict[str, Any]:
        resolved_paper_id = payload.paper_id or paper_key(payload.paper_title)
        persisted_memory = load_memories(resolved_paper_id) if payload.persist_memory else []
        paper_memory = [*persisted_memory, *payload.paper_memory]
        if payload.persist_memory and payload.selected_span:
            append_memory(
                resolved_paper_id,
                kind="paper_span",
                payload={
                    "paper_title": payload.paper_title,
                    "summary": payload.selected_span[:800],
                },
            )
        if payload.persist_memory and payload.mini_lab_result:
            append_memory(
                resolved_paper_id,
                kind="mini_lab_result",
                payload={
                    "paper_title": payload.paper_title,
                    "summary": payload.mini_lab_result[:1200],
                },
            )
        gateway = ModelGateway()
        result = gateway.growth_ideas(
            paper_title=payload.paper_title,
            paper_memory=paper_memory,
            mini_lab_result=payload.mini_lab_result,
            selected_span=payload.selected_span,
            locale=payload.locale,
            use_model=_should_use_model(payload.use_model),
        )
        if payload.persist_memory:
            for idea in result.data.get("ideas", []):
                append_memory(
                    resolved_paper_id,
                    kind="growth_idea",
                    payload={
                        "paper_title": payload.paper_title,
                        "idea": idea,
                    },
                )
        return {
            "ideas": result.data.get("ideas", []),
            "fineTuningSignal": result.data.get("fine_tuning_signal", "none"),
            "reason": result.data.get("reason", ""),
            "paperId": resolved_paper_id,
            "memoryCount": len(load_memories(resolved_paper_id)) if payload.persist_memory else len(payload.paper_memory),
            "model": result.model,
            "provider": result.provider,
            "traceId": result.trace_id,
            "error": result.error,
            "usedFallback": result.used_fallback,
        }


def _register_frontend_routes(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    def index():
        return _frontend_response("")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path.startswith(("api/", "gradio", "_next/")):
            raise HTTPException(status_code=404, detail="Not found")
        return _frontend_response(path)


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
            <p>Run <code>cd frontend && npm ci && npm run build</code>, then restart <code>python app.py</code>.</p>
            <p>The Gradio fallback remains available at <a href="/gradio">/gradio</a>.</p>
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
) -> dict[str, Any]:
    sentences = split_sentences(source.text)
    if not sentences:
        sentences = [source.text]
    total_sentences = len(sentences)
    reader_limit = max(12, min(max_reader_spans, 320))
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
    translation_map = _translation_map(source.title, span_sources, use_model)

    for section_index, section_sentences in section_plans:
        paragraph_spans = [
            {
                "id": _span_id(section_index, span_index + 1),
                "original": sentence,
                "translated": translation_map.get(_span_id(section_index, span_index + 1))
                or _translation_placeholder(sentence),
            }
            for span_index, sentence in enumerate(section_sentences)
        ]
        sections.append(
            {
                "id": f"sec-{section_index + 1}",
                "title": "Loaded Paper" if section_index == 0 else f"Source Extract {section_index + 1}",
                "titleKo": "불러온 논문" if section_index == 0 else f"원문 추출 {section_index + 1}",
                "paragraphs": [{"id": f"P{section_index}", "spans": paragraph_spans}],
            }
        )

    return {
        "id": source.source_label.replace(":", "-").replace("/", "-").lower() or "paper",
        "title": source.title or "Untitled paper",
        "titleKo": source.title or "번역 제목 생성 대기",
        "authors": [item.strip() for item in source.authors.split(",") if item.strip()] or ["Unknown authors"],
        "source": source.source_label,
        "sections": sections,
        "model": DEFAULT_MODEL if use_model else "fallback-extractive",
        "provider": DEFAULT_PROVIDER if use_model else "fallback",
        "metadata": {
            "pdfUrl": source.pdf_url,
            "warnings": list(source.warnings),
            "totalSentenceCount": total_sentences,
            "readerSpanCount": len(sentences),
            "readerSpanLimit": reader_limit,
            "translatedSpanCount": len(translation_map),
            "sourceTextChars": len(source.text),
        },
    }


def _paper_payload_text(payload: PaperInput) -> str:
    if payload.pasted_text.strip():
        return payload.pasted_text
    if payload.arxiv_or_url.strip():
        return ""
    return EXAMPLE_TEXT


def _translation_placeholder(sentence: str) -> str:
    return f"[초안 번역] {sentence}"


def _translation_map(title: str, spans: list[dict[str, str]], use_model: bool) -> dict[str, str]:
    if not spans:
        return {}
    result = ModelGateway().translate_spans(title, spans, locale="ko", use_model=use_model)
    return {
        item.get("span_id", ""): item.get("translation", "")
        for item in result.data.get("translations", [])
        if item.get("span_id")
    }


def _span_id(section_index: int, span_index: int) -> str:
    return f"P{section_index}.S{span_index}"


def _should_use_model(requested: bool) -> bool:
    return requested or _force_model_enabled()


def _force_model_enabled() -> bool:
    return os.getenv("PAPERLENS_FORCE_MODEL", "").lower() in {"1", "true", "yes"}


def _support_ids(data: dict[str, Any], fallback_span_id: str) -> list[str]:
    evidence = data.get("evidence", [])
    ids = [item.get("source_id") for item in evidence if isinstance(item, dict) and item.get("source_id")]
    support_span_ids = data.get("support_span_ids", [])
    ids.extend(item for item in support_span_ids if isinstance(item, str))
    return list(dict.fromkeys(ids or [fallback_span_id]))


def _validated_answer_data(data: dict[str, Any], payload: AskInput) -> tuple[dict[str, Any], str | None]:
    evidence = data.get("evidence", []) if isinstance(data, dict) else []
    source_pool = f"{payload.original}\n\n{payload.source_text}"
    for item in evidence:
        if not isinstance(item, dict):
            return _insufficient_answer(payload), "answer evidence is not structured"
        quote = clean_text(str(item.get("quote", "")))
        if quote and quote not in source_pool:
            return _insufficient_answer(payload), f"answer quote is not present in source evidence: {item.get('source_id', '')}"
    return data, None


def _insufficient_answer(payload: AskInput) -> dict[str, Any]:
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
        "evidence": [{"source_id": payload.span_id, "quote": payload.original[:420]}],
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


def _fallback_answer(payload: AskInput, question: str) -> str:
    evidence = top_sentences(payload.source_text or payload.original, limit=3)
    evidence_hint = " ".join(f"S{item.pid}" for item in evidence) or payload.span_id
    if payload.locale == "ko":
        return (
            f"백엔드가 선택 문장 `{payload.span_id}`를 기준으로 답했습니다. "
            f"질문은 \"{question}\"이고, 핵심 원문은 \"{payload.original[:180]}\"입니다. "
            f"현재는 소형 모델 토큰이 없어 fallback extractive 모드이며, 근거 후보는 {evidence_hint}입니다."
        )
    return (
        f"The backend answered from selected span `{payload.span_id}`. "
        f"For \"{question}\", the key source sentence is \"{payload.original[:180]}\". "
        f"This is fallback extractive mode until model inference is enabled; candidate evidence: {evidence_hint}."
    )
