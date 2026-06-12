"use client";

import { useState, useCallback, useEffect } from "react";
import { MOCK_PAPER } from "@/lib/mock-data";
import {
  askAboutSpan,
  loadPaperFromSession,
  savePaperToSession,
  translateSelectedSpan,
} from "@/lib/api";
import {
  ViewMode,
  Annotation,
  AnnotationType,
  QAMessage,
  Span,
} from "@/lib/types";
import { getInitialLocale, Locale, UI_TEXT } from "@/lib/i18n";
import SectionNav from "@/components/SectionNav";
import ReadingPane from "@/components/ReadingPane";
import RightPanel from "@/components/RightPanel";
import AnnotationBar from "@/components/AnnotationBar";
import LabModal from "@/components/LabModal";

export default function ReaderPage() {
  const [paper, setPaper] = useState(MOCK_PAPER);
  const [locale, setLocale] = useState<Locale>("en");
  const [viewMode, setViewMode] = useState<ViewMode>("original");
  const [selectedSpanId, setSelectedSpanId] = useState<string | null>(null);
  const [activeSourceSpanId, setActiveSourceSpanId] = useState<string | null>(
    null,
  );
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [qaMessages, setQaMessages] = useState<QAMessage[]>([]);
  const [showQA, setShowQA] = useState(false);
  const [labSpan, setLabSpan] = useState<Span | null>(null);
  const [activeSectionId, setActiveSectionId] = useState<string>(
    paper.sections[0]?.id ?? "",
  );
  const text = UI_TEXT[locale];

  useEffect(() => {
    setLocale(getInitialLocale(new URLSearchParams(window.location.search).get("lang")));
    const sessionPaper = loadPaperFromSession();
    if (sessionPaper) {
      setPaper(sessionPaper);
      setActiveSectionId(sessionPaper.sections[0]?.id ?? "");
    }
  }, []);

  const allSpans = paper.sections.flatMap((s) =>
    s.paragraphs.flatMap((p) => p.spans),
  );

  const findSpan = useCallback(
    (id: string) => allSpans.find((s) => s.id === id) ?? null,
    [allSpans],
  );
  const sourceText = allSpans.map((span) => span.original).join(" ");

  function handleSpanClick(spanId: string) {
    setSelectedSpanId(spanId);
    setActiveSourceSpanId(spanId);
    void ensureSpanTranslation(spanId);
  }

  function updateSpanTranslation(spanId: string, translated: string) {
    setPaper((currentPaper) => {
      const nextPaper = {
        ...currentPaper,
        sections: currentPaper.sections.map((section) => ({
          ...section,
          paragraphs: section.paragraphs.map((paragraph) => ({
            ...paragraph,
            spans: paragraph.spans.map((span) =>
              span.id === spanId ? { ...span, translated } : span,
            ),
          })),
        })),
      };
      savePaperToSession(nextPaper);
      return nextPaper;
    });
  }

  async function ensureSpanTranslation(spanId: string) {
    const span = findSpan(spanId);
    if (!span || !isDraftTranslation(span.translated)) return;

    updateSpanTranslation(
      spanId,
      locale === "ko" ? "모델 번역 생성 중..." : "Generating model translation...",
    );
    try {
      const result = await translateSelectedSpan({
        paperId: paper.id,
        paperTitle: paper.title,
        span,
        locale: "ko",
      });
      if (result.translation) {
        updateSpanTranslation(spanId, result.translation);
      }
    } catch {
      updateSpanTranslation(spanId, span.translated);
    }
  }

  function handleAnnotate(type: AnnotationType) {
    if (!selectedSpanId) return;
    const exists = annotations.find(
      (a) => a.spanId === selectedSpanId && a.type === type,
    );
    if (exists) {
      setAnnotations(annotations.filter((a) => a.id !== exists.id));
    } else {
      setAnnotations([
        ...annotations,
        {
          id: `ann-${Date.now()}`,
          spanId: selectedSpanId,
          type,
          createdAt: Date.now(),
        },
      ]);
    }
  }

  async function handleAskAI() {
    if (!selectedSpanId) return;
    setShowQA(true);
    const span = findSpan(selectedSpanId);
    if (!span) return;

    const question =
      locale === "ko"
        ? `이 문장에 대해 설명해주세요: "${span.translated}"`
        : `Explain this sentence: "${span.original}"`;
    const userMsg: QAMessage = {
      id: `qa-${Date.now()}`,
      role: "user",
      content: question,
      supportSpanIds: [selectedSpanId],
    };

    const pendingMsg: QAMessage = {
      id: `qa-${Date.now() + 1}`,
      role: "assistant",
      content: locale === "ko" ? "백엔드에서 근거를 확인하는 중..." : "Checking backend evidence...",
      supportSpanIds: [selectedSpanId],
      isLoading: true,
    };

    setQaMessages((messages) => [...messages, userMsg, pendingMsg]);
    try {
      const answer = await askAboutSpan({
        paperId: paper.id,
        span,
        paperTitle: paper.title,
        sourceText,
        question,
        locale,
      });
      setQaMessages((messages) =>
        messages.map((message) => (message.id === pendingMsg.id ? answer : message)),
      );
    } catch {
      setQaMessages((messages) =>
        messages.map((message) =>
          message.id === pendingMsg.id
            ? {
                ...pendingMsg,
                content: generateMockAnswer(span, locale),
                isLoading: false,
              }
            : message,
        ),
      );
    }
  }

  function handleTryExperiment() {
    if (!selectedSpanId) return;
    const span = findSpan(selectedSpanId);
    if (span) setLabSpan(span);
  }

  function handleSectionClick(sectionId: string) {
    setActiveSectionId(sectionId);
    const el = document.getElementById(sectionId);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function handleLocaleChange(nextLocale: Locale) {
    setLocale(nextLocale);
    window.history.replaceState(null, "", `/reader?lang=${nextLocale}`);
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-surface">
      {/* Top Header */}
      <header className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <a
            href={`/?lang=${locale}`}
            className="flex items-center gap-1.5 text-sm font-semibold text-primary-700"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            PaperLens Lab
          </a>
          <span className="text-text-muted">|</span>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-medium text-text-primary">
              {viewMode === "translated" ? paper.titleKo : paper.title}
            </h1>
            <p className="text-xs text-text-muted">
              {paper.authors.join(", ")} &middot; {paper.source}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Language Toggle */}
          <div
            className="flex items-center gap-1 rounded-lg bg-surface-secondary p-1"
            aria-label={text.reader.languageLabel}
          >
            {(["en", "ko"] as const).map((option) => (
              <button
                key={option}
                onClick={() => handleLocaleChange(option)}
                className={`rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
                  locale === option
                    ? "bg-surface text-text-primary shadow-sm"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                {option.toUpperCase()}
              </button>
            ))}
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center gap-1 rounded-lg bg-surface-secondary p-1">
            {(
              [
                ["original", text.reader.viewModes.original],
                ["translated", text.reader.viewModes.translated],
                ["side-by-side", text.reader.viewModes["side-by-side"]],
              ] as const
            ).map(([mode, label]) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                  viewMode === mode
                    ? "bg-surface text-text-primary shadow-sm"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex min-h-0 flex-1">
        {/* Left Sidebar */}
        <SectionNav
          sections={paper.sections}
          activeSectionId={activeSectionId}
          annotations={annotations}
          allSpans={allSpans}
          locale={locale}
          onSectionClick={handleSectionClick}
          onSpanClick={handleSpanClick}
        />

        {/* Center Reading Area */}
        <ReadingPane
          paper={paper}
          viewMode={viewMode}
          selectedSpanId={selectedSpanId}
          activeSourceSpanId={activeSourceSpanId}
          annotations={annotations}
          onSpanClick={handleSpanClick}
        />

        {/* Right Panel */}
        <RightPanel
          paperId={paper.id}
          selectedSpanId={activeSourceSpanId}
          findSpan={findSpan}
          showQA={showQA}
          qaMessages={qaMessages}
          setQaMessages={setQaMessages}
          setShowQA={setShowQA}
          viewMode={viewMode}
          locale={locale}
          paperTitle={paper.title}
          sourceText={sourceText}
        />
      </div>

      {/* Floating Annotation Bar */}
      <AnnotationBar
        selectedSpanId={selectedSpanId}
        annotations={annotations}
        onAnnotate={handleAnnotate}
        onAskAI={handleAskAI}
        onTryExperiment={handleTryExperiment}
        locale={locale}
      />

      {/* Lab Modal */}
      {labSpan && (
        <LabModal
          span={labSpan}
          locale={locale}
          paperId={paper.id}
          paperTitle={paper.title}
          sourceText={sourceText}
          onClose={() => setLabSpan(null)}
        />
      )}
    </div>
  );
}

function isDraftTranslation(value: string): boolean {
  const trimmed = value.trim();
  return (
    trimmed.startsWith("[초안 번역]") ||
    trimmed.startsWith("[Korean draft pending]") ||
    trimmed === ""
  );
}

function generateMockAnswer(span: Span, locale: Locale): string {
  if (locale === "en") {
    const answers: Record<string, string> = {
      "P0.S3":
        "This sentence summarizes the paper's main reported result. A 34% reduction is substantial, but the exact measurement of 'irrelevant context injection' should still be checked in Section 3.",
      "P1.S2":
        "Context dilution is the trade-off the paper defines: retrieving more passages can improve recall, but it also adds distracting evidence that may mislead the generator.",
      "P6.S2":
        "This is an important result because evidential support contributes the most. [Beyond the paper] It suggests pure semantic similarity may be too weak for high-precision RAG.",
      "P7.S2":
        "The extra 40ms is per passage. Reranking top-100 passages would add about four seconds, so the limitation matters for latency-sensitive systems.",
    };

    return (
      answers[span.id] ||
      `This sentence, "${span.original.slice(0, 30)}...", is connected to one of the paper's central claims. I would check the surrounding section before treating it as a general conclusion.`
    );
  }

  const answers: Record<string, string> = {
    "P0.S3":
      "이 문장은 논문의 핵심 성과를 요약합니다. 34%의 감소는 상당한 수치이며, 이는 불필요한 passage가 생성기에 전달되는 비율이 크게 줄었음을 의미합니다. 다만 '관련 없는 컨텍스트 주입'의 정확한 측정 방식은 Section 3에서 확인할 필요가 있습니다.",
    "P1.S2":
      "컨텍스트 희석(context dilution)은 이 논문에서 정의한 용어입니다. 더 많은 passage를 검색하면 정답을 포함할 가능성(recall)은 높아지지만, 동시에 관련 없는 정보도 같이 들어와서 LLM이 혼란스러워할 수 있다는 의미입니다. 이는 RAG 시스템의 근본적인 트레이드오프입니다.",
    "P6.S2":
      "이 결과는 매우 흥미롭습니다. 증거 지원(evidential support) 구성 요소가 가장 큰 기여를 한다는 것은, passage가 단순히 관련있는 것을 넘어서 '검증 가능한 증거'를 제공하는지가 중요하다는 뜻입니다. [논문 밖 해석] 이는 단순 semantic similarity 기반 검색의 한계를 보여줍니다.",
    "P7.S2":
      "40ms의 추가 지연은 passage 하나당 소요되는 시간입니다. 만약 top-100 passage를 재순위화한다면 총 4초가 추가되므로, 실시간 시스템에서는 부담이 될 수 있습니다. 논문도 이를 한계로 인정하고 있습니다.",
  };

  return (
    answers[span.id] ||
    `이 문장은 "${span.translated.slice(0, 30)}..."에 대한 설명입니다. 원문을 기반으로 보면, 이는 논문의 핵심 주장 중 하나와 관련됩니다. 추가적인 맥락은 해당 섹션의 전후 문단을 참조하시면 좋습니다.`
  );
}
