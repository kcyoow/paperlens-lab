from __future__ import annotations

import html
from dataclasses import dataclass

import gradio as gr

from .analysis import analyze_paper, experiment_card, top_sentences
from .ingest import PaperSource, build_source


CSS = """
:root {
  --pl-primary-50: #f0fdfa;
  --pl-primary-100: #ccfbf1;
  --pl-primary-200: #99f6e4;
  --pl-primary-400: #2dd4bf;
  --pl-primary-500: #14b8a6;
  --pl-primary-600: #0d9488;
  --pl-primary-700: #0f766e;
  --pl-surface: #ffffff;
  --pl-surface-secondary: #f8fafc;
  --pl-surface-hover: #f1f5f9;
  --pl-border: #e2e8f0;
  --pl-border-strong: #cbd5e1;
  --pl-text-primary: #0f172a;
  --pl-text-secondary: #475569;
  --pl-text-muted: #94a3b8;
  --pl-yellow: #fef9c3;
  --pl-yellow-strong: #fde047;
  --pl-blue: #dbeafe;
  --pl-blue-strong: #60a5fa;
  --pl-red: #fecaca;
  --pl-red-strong: #f87171;
  --pl-purple: #e9d5ff;
  --pl-purple-strong: #a78bfa;
  --pl-orange: #ffedd5;
  --pl-orange-strong: #fb923c;
}

.gradio-container {
  max-width: none !important;
  color: var(--pl-text-primary);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR", system-ui, sans-serif;
}

.paperlens-host {
  max-width: 1440px;
  margin: 0 auto;
}

.paperlens-load-panel {
  border: 1px solid var(--pl-border) !important;
  border-radius: 12px !important;
  background: var(--pl-surface) !important;
}

.paperlens-reader {
  min-height: 760px;
  overflow: hidden;
  border: 1px solid var(--pl-border);
  border-radius: 12px;
  background: var(--pl-surface);
  color: var(--pl-text-primary);
  box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
}

.paperlens-reader * {
  box-sizing: border-box;
}

.pl-topbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--pl-border);
  padding: 12px 20px;
}

.pl-brand-row {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 12px;
}

.pl-brand {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--pl-primary-700);
  font-size: 14px;
  font-weight: 700;
  white-space: nowrap;
}

.pl-search-icon {
  width: 16px;
  height: 16px;
  border: 2px solid currentColor;
  border-radius: 999px;
  position: relative;
}

.pl-search-icon::after {
  content: "";
  position: absolute;
  right: -5px;
  bottom: -4px;
  width: 7px;
  height: 2px;
  border-radius: 999px;
  background: currentColor;
  transform: rotate(45deg);
}

.pl-divider {
  color: var(--pl-text-muted);
}

.pl-paper-title {
  min-width: 0;
}

.pl-paper-title h1 {
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--pl-text-primary);
  font-size: 14px;
  font-weight: 600;
}

.pl-paper-title p {
  margin: 2px 0 0;
  color: var(--pl-text-muted);
  font-size: 12px;
}

.pl-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.pl-segment {
  display: flex;
  gap: 4px;
  border-radius: 8px;
  background: var(--pl-surface-secondary);
  padding: 4px;
}

.pl-segment span {
  border-radius: 6px;
  padding: 6px 10px;
  color: var(--pl-text-secondary);
  font-size: 12px;
  font-weight: 600;
}

.pl-segment .active {
  background: var(--pl-surface);
  color: var(--pl-text-primary);
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
}

.pl-workspace {
  display: grid;
  min-height: 690px;
  grid-template-columns: 192px minmax(0, 1fr) 288px;
}

.pl-sidebar {
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--pl-border);
  background: var(--pl-surface-secondary);
}

.pl-sidebar-section {
  flex: 1;
  padding: 12px;
}

.pl-eyebrow {
  margin: 0 0 8px;
  padding: 0 8px;
  color: var(--pl-text-muted);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.pl-nav {
  display: grid;
  gap: 2px;
}

.pl-nav a {
  display: block;
  border-radius: 8px;
  padding: 7px 10px;
  color: var(--pl-text-secondary);
  font-size: 12px;
  line-height: 1.25;
  text-decoration: none;
}

.pl-nav a.active {
  background: var(--pl-primary-50);
  color: var(--pl-primary-700);
  font-weight: 700;
}

.pl-marks {
  border-top: 1px solid var(--pl-border);
  padding: 12px;
}

.pl-mark-empty {
  margin: 0;
  padding: 0 8px;
  color: var(--pl-text-muted);
  font-size: 11px;
  line-height: 1.45;
}

.pl-reader-main {
  overflow: auto;
  padding: 24px;
}

.pl-paper {
  max-width: none;
}

.pl-paper-heading {
  margin-bottom: 30px;
}

.pl-paper-heading h2 {
  margin: 0 0 8px;
  color: var(--pl-text-primary);
  font-size: 20px;
  font-weight: 800;
  line-height: 1.25;
  word-break: keep-all;
}

.pl-paper-heading p {
  margin: 0;
  color: var(--pl-text-secondary);
  font-size: 14px;
}

.pl-section {
  margin-bottom: 32px;
}

.pl-section h3 {
  margin: 0 0 16px;
  border-bottom: 1px solid var(--pl-border);
  padding-bottom: 9px;
  color: var(--pl-text-primary);
  font-size: 18px;
  font-weight: 700;
}

.pl-paragraph {
  margin: 0 0 20px;
  color: var(--pl-text-primary);
  font-size: 15px;
  line-height: 1.75;
}

.pl-side-by-side {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 24px;
}

.pl-translation-column {
  border-left: 1px solid var(--pl-border);
  padding-left: 24px;
  color: var(--pl-text-secondary);
  font-size: 14px;
}

.pl-span {
  border-radius: 3px;
  padding: 0 2px;
  color: var(--pl-text-primary) !important;
}

.pl-span.highlight {
  border-bottom: 2px solid var(--pl-yellow-strong);
  background: var(--pl-yellow);
}

.pl-span.underline {
  border-bottom: 2px solid var(--pl-blue-strong);
}

.pl-span.question {
  border-bottom: 2px solid var(--pl-orange-strong);
  background: var(--pl-orange);
}

.pl-span.experiment {
  border-bottom: 2px solid var(--pl-purple-strong);
  background: var(--pl-purple);
}

.pl-span.limitation {
  border-bottom: 2px solid var(--pl-red-strong);
  background: var(--pl-red);
}

.pl-span.active-source {
  outline: 2px solid var(--pl-primary-400);
  outline-offset: 1px;
  background: var(--pl-primary-100);
}

.pl-right {
  display: flex;
  flex-direction: column;
  border-left: 1px solid var(--pl-border);
  background: var(--pl-surface);
}

.pl-tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  border-bottom: 1px solid var(--pl-border);
}

.pl-tab {
  padding: 10px 8px;
  text-align: center;
  color: var(--pl-text-muted);
  font-size: 12px;
  font-weight: 700;
}

.pl-tab.active {
  border-bottom: 2px solid var(--pl-primary-500);
  color: var(--pl-primary-700);
}

.pl-panel {
  padding: 16px;
}

.pl-pill {
  display: inline-block;
  margin: 2px 0 12px;
  border-radius: 4px;
  background: var(--pl-primary-50);
  padding: 3px 7px;
  color: var(--pl-primary-700);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 10px;
  font-weight: 700;
}

.pl-source-card {
  margin-bottom: 12px;
  border: 1px solid var(--pl-primary-200);
  border-radius: 10px;
  background: rgba(240, 253, 250, .55);
  padding: 12px;
}

.pl-translation-card {
  border: 1px solid var(--pl-border);
  border-radius: 10px;
  padding: 12px;
}

.pl-card-label {
  margin: 0 0 6px;
  color: var(--pl-primary-700);
  font-size: 10px;
  font-weight: 800;
}

.pl-translation-card .pl-card-label {
  color: var(--pl-text-muted);
}

.pl-card-text {
  margin: 0;
  color: var(--pl-text-primary);
  font-size: 13px;
  line-height: 1.55;
}

.pl-actions {
  display: flex;
  gap: 8px;
  margin-top: 14px;
}

.pl-ghost-btn {
  border: 1px solid var(--pl-border);
  border-radius: 8px;
  background: var(--pl-surface);
  padding: 7px 10px;
  color: var(--pl-text-secondary);
  font-size: 11px;
  font-weight: 700;
}

.pl-qa-card {
  margin-top: 18px;
  border: 1px solid var(--pl-border);
  border-radius: 10px;
  padding: 12px;
}

.pl-qa-card p {
  margin: 0;
  color: var(--pl-text-secondary);
  font-size: 13px;
  line-height: 1.55;
}

.pl-floating-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  border-top: 1px solid var(--pl-border);
  padding: 12px;
  background: rgba(255, 255, 255, .96);
}

.pl-tool,
.pl-primary-tool {
  border: 0;
  border-radius: 12px;
  padding: 9px 11px;
  font-size: 12px;
  font-weight: 800;
}

.pl-tool.yellow { background: #fefce8; color: #a16207; }
.pl-tool.blue { background: #eff6ff; color: #2563eb; }
.pl-tool.orange { background: #fff7ed; color: #ea580c; }
.pl-tool.red { background: #fef2f2; color: #dc2626; }
.pl-tool.ai { background: var(--pl-primary-50); color: var(--pl-primary-700); }
.pl-primary-tool { background: var(--pl-primary-600); color: white; }
.pl-tool-divider { width: 1px; height: 24px; background: var(--pl-border); margin: 0 4px; }

.paperlens-code textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
}

@media (max-width: 1100px) {
  .pl-workspace {
    grid-template-columns: 160px minmax(0, 1fr);
  }
  .pl-right {
    display: none;
  }
}

@media (max-width: 760px) {
  .pl-workspace {
    display: block;
  }
  .pl-sidebar {
    display: none;
  }
  .pl-side-by-side {
    grid-template-columns: 1fr;
  }
  .pl-translation-column {
    border-left: 0;
    border-top: 1px solid var(--pl-border);
    padding-left: 0;
    padding-top: 16px;
  }
}
"""


EXAMPLE_TEXT = """Abstract: We propose a retrieval-augmented reading assistant for scientific papers. The system grounds every explanation in cited source spans, separates paper claims from interpretation, and produces small experiment cards that help readers test an idea before investing in a full implementation.

Introduction: Non-native English readers often need to compare a translation with the source sentence while preserving technical terms. PaperLens Lab keeps the source text visible, lets readers mark claims, and turns promising highlighted spans into small experiments.

Method: The reader workspace parses a paper into source spans, creates a Korean translation draft, links every explanation to the original sentence, and keeps unsupported interpretation separate from paper claims.

Limitations: The current demo uses lightweight extraction when no model token is configured. Full translation, PDF layout recovery, and agentic experiment execution are planned backend work."""


MOCK_TRANSLATIONS = [
    "검색 증강 독서 도우미는 인용된 원문 span에 모든 설명을 연결하고, 논문의 주장과 해석을 분리하며, 아이디어를 작은 실험 카드로 바꾸도록 설계된다.",
    "비영어권 독자는 기술 용어를 유지하면서 번역문과 원문 문장을 함께 확인해야 하는 경우가 많다.",
    "PaperLens Lab은 원문을 계속 보여주고, 독자가 중요한 주장을 표시하며, 선택한 span을 작은 실험으로 전환할 수 있게 한다.",
    "이 reader workspace는 논문을 source span으로 파싱하고, 한국어 번역 초안을 만들며, 모든 설명을 원문 문장에 연결한다.",
    "현재 데모는 모델 토큰이 없을 때 가벼운 추출 방식을 사용한다. 전체 번역, PDF 레이아웃 복원, 에이전트 실험 실행은 이후 백엔드 작업이다.",
]


@dataclass
class ReaderSpan:
    sid: str
    original: str
    translated: str
    style: str


@dataclass
class ReaderSection:
    anchor: str
    title: str
    title_ko: str
    spans: list[ReaderSpan]


def _source_from_inputs(uploaded_pdf, arxiv_or_url, pasted_text, max_pdf_pages) -> PaperSource:
    return build_source(uploaded_pdf, arxiv_or_url or "", pasted_text or "", int(max_pdf_pages))


def _effective_pasted_text(uploaded_pdf, arxiv_or_url, pasted_text) -> str:
    if uploaded_pdf or str(arxiv_or_url or "").strip():
        return ""
    return pasted_text or EXAMPLE_TEXT


def _escape(value: str) -> str:
    return html.escape(value or "", quote=True)


def _split_for_reader(source: PaperSource) -> list[ReaderSpan]:
    sentences = top_sentences(source.text, limit=8)
    if not sentences:
        return [
            ReaderSpan(
                sid="P0.S1",
                original="Upload a PDF, paste paper text, or enter an arXiv URL to load the reader workspace.",
                translated="PDF를 업로드하거나 논문 텍스트/arXiv URL을 입력하면 reader workspace가 구성됩니다.",
                style="highlight active-source",
            )
        ]

    styles = ["highlight active-source", "underline", "question", "experiment", "limitation", "", "", ""]
    spans: list[ReaderSpan] = []
    for index, sentence in enumerate(sentences):
        translated = MOCK_TRANSLATIONS[index % len(MOCK_TRANSLATIONS)]
        if source.source_label != "sample":
            translated = f"번역 초안 {index + 1}: {_escape(sentence.text[:150])}"
        spans.append(
            ReaderSpan(
                sid=f"P{index}.S1",
                original=sentence.text,
                translated=translated,
                style=styles[index] if index < len(styles) else "",
            )
        )
    return spans


def _sections_for_reader(source: PaperSource) -> list[ReaderSection]:
    spans = _split_for_reader(source)
    midpoint = max(1, len(spans) // 2)
    return [
        ReaderSection("sec-abstract", "Abstract", "초록", spans[:midpoint]),
        ReaderSection("sec-method", "Reader Notes", "리더 노트", spans[midpoint:]),
    ]


def _segment(label: str, active: bool) -> str:
    return f"<span class=\"{'active' if active else ''}\">{_escape(label)}</span>"


def _nav_html(sections: list[ReaderSection], locale: str) -> str:
    links = []
    for index, section in enumerate(sections):
        title = section.title_ko if locale == "ko" else section.title
        links.append(
            f'<a class="{"active" if index == 0 else ""}" href="#{_escape(section.anchor)}">{_escape(title)}</a>'
        )
    return "\n".join(links)


def _spans_inline(spans: list[ReaderSpan], use_translation: bool) -> str:
    chunks = []
    for span in spans:
        text = span.translated if use_translation else span.original
        chunks.append(f'<span class="pl-span {span.style}">{_escape(text)}</span>')
    return " ".join(chunks)


def _paper_html(sections: list[ReaderSection], view_mode: str, locale: str) -> str:
    rendered = []
    for section in sections:
        title = section.title_ko if view_mode == "translated" else section.title
        subtitle = f'<span style="margin-left:10px;color:var(--pl-text-muted);font-size:13px;font-weight:500;">{_escape(section.title_ko)}</span>' if view_mode == "side-by-side" else ""
        if view_mode == "side-by-side":
            body = (
                '<div class="pl-side-by-side">'
                f'<p class="pl-paragraph">{_spans_inline(section.spans, False)}</p>'
                f'<p class="pl-paragraph pl-translation-column">{_spans_inline(section.spans, True)}</p>'
                "</div>"
            )
        else:
            body = f'<p class="pl-paragraph">{_spans_inline(section.spans, view_mode == "translated")}</p>'
        rendered.append(
            f"""
            <section class="pl-section" id="{_escape(section.anchor)}">
              <h3>{_escape(title)}{subtitle}</h3>
              {body}
            </section>
            """
        )
    return "\n".join(rendered)


def render_reader(source: PaperSource, view_mode: str = "original", locale: str = "en") -> str:
    sections = _sections_for_reader(source)
    selected = sections[0].spans[0]
    title = source.title or "Improving Retrieval-Augmented Generation with Evidence-Linked Reranking"
    subtitle = source.authors or "PaperLens Team"
    source_label = source.source_label or "sample"
    ko_title = "논문 번역 및 원문 대조 workspace"
    mode_labels = {
        "original": "Original" if locale == "en" else "원문",
        "translated": "Translation" if locale == "en" else "번역",
        "side-by-side": "Side by side" if locale == "en" else "나란히",
    }
    source_tab = "Source check" if locale == "en" else "원문 대조"
    qa_tab = "Ask AI" if locale == "en" else "AI 질문"
    selected_label = "Selected sentence" if locale == "en" else "선택 문장"
    english_label = "English source" if locale == "en" else "영어 원문"
    korean_label = "Korean translation" if locale == "en" else "한국어 번역"
    report = "Report translation" if locale == "en" else "번역 문제 표시"
    retranslate = "Retranslate" if locale == "en" else "다시 번역"
    qa_text = (
        "This span is linked to the source sentence. PaperLens should answer with evidence IDs before making an interpretation."
        if locale == "en"
        else "이 span은 원문 문장과 연결되어 있습니다. PaperLens는 해석을 덧붙이기 전에 evidence ID를 먼저 보여줘야 합니다."
    )

    return f"""
    <div class="paperlens-reader">
      <header class="pl-topbar">
        <div class="pl-brand-row">
          <div class="pl-brand"><span class="pl-search-icon"></span>PaperLens Lab</div>
          <span class="pl-divider">|</span>
          <div class="pl-paper-title">
            <h1>{_escape(ko_title if view_mode == "translated" else title)}</h1>
            <p>{_escape(subtitle)} · {_escape(source_label)}</p>
          </div>
        </div>
        <div class="pl-controls">
          <div class="pl-segment">
            {_segment("EN", locale == "en")}
            {_segment("KO", locale == "ko")}
          </div>
          <div class="pl-segment">
            {_segment(mode_labels["original"], view_mode == "original")}
            {_segment(mode_labels["translated"], view_mode == "translated")}
            {_segment(mode_labels["side-by-side"], view_mode == "side-by-side")}
          </div>
        </div>
      </header>
      <div class="pl-workspace">
        <aside class="pl-sidebar">
          <div class="pl-sidebar-section">
            <p class="pl-eyebrow">{"Contents" if locale == "en" else "목차"}</p>
            <nav class="pl-nav">{_nav_html(sections, locale)}</nav>
          </div>
          <div class="pl-marks">
            <p class="pl-eyebrow">{"My marks" if locale == "en" else "내 표시"} (4)</p>
            <p class="pl-mark-empty">{"Highlight, underline, question, and limitation marks stay visible beside the paper." if locale == "en" else "형광펜, 밑줄, 질문, 한계 표시가 논문 옆에 유지됩니다."}</p>
          </div>
        </aside>
        <main class="pl-reader-main">
          <article class="pl-paper">
            <div class="pl-paper-heading">
              <h2>{_escape(title)}</h2>
              <p>{_escape(subtitle)}</p>
            </div>
            {_paper_html(sections, view_mode, locale)}
          </article>
        </main>
        <aside class="pl-right">
          <div class="pl-tabs">
            <div class="pl-tab active">{_escape(source_tab)}</div>
            <div class="pl-tab">{_escape(qa_tab)}</div>
          </div>
          <div class="pl-panel">
            <p class="pl-eyebrow" style="padding:0;">{_escape(selected_label)}</p>
            <span class="pl-pill">{_escape(selected.sid)}</span>
            <div class="pl-source-card">
              <p class="pl-card-label">{_escape(english_label)}</p>
              <p class="pl-card-text">{_escape(selected.original)}</p>
            </div>
            <div class="pl-translation-card">
              <p class="pl-card-label">{_escape(korean_label)}</p>
              <p class="pl-card-text">{_escape(selected.translated)}</p>
            </div>
            <div class="pl-actions">
              <button class="pl-ghost-btn">{_escape(report)}</button>
              <button class="pl-ghost-btn">{_escape(retranslate)}</button>
            </div>
            <div class="pl-qa-card">
              <p>{_escape(qa_text)}</p>
            </div>
          </div>
        </aside>
      </div>
      <div class="pl-floating-bar">
        <button class="pl-tool yellow">{"Highlight" if locale == "en" else "형광펜"}</button>
        <button class="pl-tool blue">{"Underline" if locale == "en" else "밑줄"}</button>
        <button class="pl-tool orange">{"Question" if locale == "en" else "질문"}</button>
        <button class="pl-tool red">{"Limitation" if locale == "en" else "한계"}</button>
        <span class="pl-tool-divider"></span>
        <button class="pl-tool ai">{"Ask AI" if locale == "en" else "AI 질문"}</button>
        <span class="pl-tool-divider"></span>
        <button class="pl-primary-tool">{"Try Experiment" if locale == "en" else "실험해보기"}</button>
      </div>
    </div>
    """


def load_reader(uploaded_pdf, arxiv_or_url, pasted_text, view_mode, locale, max_pdf_pages):
    try:
        source = _source_from_inputs(
            uploaded_pdf,
            arxiv_or_url,
            _effective_pasted_text(uploaded_pdf, arxiv_or_url, pasted_text),
            max_pdf_pages,
        )
        if not uploaded_pdf and not arxiv_or_url and not pasted_text:
            source = PaperSource(
                title="Improving Retrieval-Augmented Generation with Evidence-Linked Reranking",
                authors="J. Kim, S. Park, M. Lee",
                source_label="sample",
                text=EXAMPLE_TEXT,
            )
        return render_reader(source, view_mode, locale)
    except Exception as exc:
        fallback = PaperSource(
            title="Paper could not be loaded",
            authors="PaperLens Lab",
            source_label="error",
            text=f"Could not load the paper. {exc}",
        )
        return render_reader(fallback, view_mode, locale)


def run_analysis(uploaded_pdf, arxiv_or_url, pasted_text, max_pdf_pages, use_model):
    try:
        source = _source_from_inputs(
            uploaded_pdf,
            arxiv_or_url,
            _effective_pasted_text(uploaded_pdf, arxiv_or_url, pasted_text),
            max_pdf_pages,
        )
        return analyze_paper(source, "Research reader", "Source-grounded translation and experiment design", use_model)
    except Exception as exc:
        message = f"## Could not analyze paper\n\n{exc}"
        return message, "", "", ""


def run_experiment(uploaded_pdf, arxiv_or_url, pasted_text, idea, max_pdf_pages, use_model):
    try:
        source = _source_from_inputs(
            uploaded_pdf,
            arxiv_or_url,
            _effective_pasted_text(uploaded_pdf, arxiv_or_url, pasted_text),
            max_pdf_pages,
        )
        return experiment_card(source, idea, "Research reader", use_model)
    except Exception as exc:
        return f"## Could not build experiment card\n\n{exc}", ""


def build_demo() -> gr.Blocks:
    sample_source = PaperSource(
        title="Improving Retrieval-Augmented Generation with Evidence-Linked Reranking",
        authors="J. Kim, S. Park, M. Lee",
        source_label="sample",
        text=EXAMPLE_TEXT,
    )

    with gr.Blocks(
        title="PaperLens Lab",
        css=CSS,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    ) as demo:
        with gr.Column(elem_classes=["paperlens-host"]):
            reader_html = gr.HTML(
                value=render_reader(sample_source, "original", "en"),
                elem_classes=["paperlens-reader-output"],
            )

            with gr.Accordion("Load paper and controls", open=False, elem_classes=["paperlens-load-panel"]):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        uploaded_pdf = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
                        arxiv_or_url = gr.Textbox(
                            label="arXiv ID or URL",
                            placeholder="2505.09388 or https://arxiv.org/abs/2505.09388",
                        )
                    with gr.Column(scale=4):
                        pasted_text = gr.Textbox(
                            label="Paper text",
                            value=EXAMPLE_TEXT,
                            lines=7,
                            max_lines=14,
                        )
                    with gr.Column(scale=2):
                        locale = gr.Radio(
                            [("English UI", "en"), ("Korean UI", "ko")],
                            value="en",
                            label="Language",
                        )
                        view_mode = gr.Radio(
                            [("Original", "original"), ("Translation", "translated"), ("Side by side", "side-by-side")],
                            value="original",
                            label="Reader view",
                        )
                        max_pdf_pages = gr.Slider(2, 24, value=10, step=1, label="PDF pages")
                        use_model = gr.Checkbox(value=False, label="Use HF model adapter")
                        load_btn = gr.Button("Open Reader", variant="primary")

            with gr.Accordion("Analysis and Lab outputs", open=False):
                analyze_btn = gr.Button("Run source-grounded analysis", variant="secondary")
                with gr.Tabs():
                    with gr.Tab("Reading guide"):
                        reader_output = gr.Markdown()
                    with gr.Tab("Evidence"):
                        evidence_output = gr.Markdown()
                    with gr.Tab("Claims and terms"):
                        claims_output = gr.Markdown()
                    with gr.Tab("Raw extract"):
                        raw_output = gr.Textbox(lines=16, label="Top source spans")

                idea = gr.Textbox(
                    label="Paper idea to test",
                    placeholder="Example: test whether source-linked translation reduces unsupported claims",
                    lines=3,
                )
                experiment_btn = gr.Button("Build Experiment Card", variant="secondary")
                with gr.Row(equal_height=False):
                    experiment_output = gr.Markdown(label="Experiment card")
                    starter_output = gr.Code(
                        label="starter.py",
                        language="python",
                        lines=20,
                        elem_classes=["paperlens-code"],
                    )

        reader_inputs = [uploaded_pdf, arxiv_or_url, pasted_text, view_mode, locale, max_pdf_pages]
        load_btn.click(fn=load_reader, inputs=reader_inputs, outputs=reader_html)
        view_mode.change(fn=load_reader, inputs=reader_inputs, outputs=reader_html)
        locale.change(fn=load_reader, inputs=reader_inputs, outputs=reader_html)
        pasted_text.submit(fn=load_reader, inputs=reader_inputs, outputs=reader_html)

        analyze_btn.click(
            fn=run_analysis,
            inputs=[uploaded_pdf, arxiv_or_url, pasted_text, max_pdf_pages, use_model],
            outputs=[reader_output, evidence_output, claims_output, raw_output],
        )
        experiment_btn.click(
            fn=run_experiment,
            inputs=[uploaded_pdf, arxiv_or_url, pasted_text, idea, max_pdf_pages, use_model],
            outputs=[experiment_output, starter_output],
        )

    return demo
