"use client";

import { useState, useCallback, useEffect } from "react";
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
  PaperDocument,
  Span,
} from "@/lib/types";
import { getInitialLocale, Locale, UI_TEXT } from "@/lib/i18n";
import SectionNav from "@/components/SectionNav";
import ReadingPane from "@/components/ReadingPane";
import RightPanel from "@/components/RightPanel";
import AnnotationBar from "@/components/AnnotationBar";
import LabModal from "@/components/LabModal";

export default function ReaderPage() {
  const [paper, setPaper] = useState<PaperDocument | null>(null);
  const [hasCheckedSession, setHasCheckedSession] = useState(false);
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
  const [activeSectionId, setActiveSectionId] = useState<string>("");
  const text = UI_TEXT[locale];

  useEffect(() => {
    setLocale(getInitialLocale(new URLSearchParams(window.location.search).get("lang")));
    const sessionPaper = loadPaperFromSession();
    if (sessionPaper) {
      setPaper(sessionPaper);
      setActiveSectionId(sessionPaper.sections[0]?.id ?? "");
    }
    setHasCheckedSession(true);
  }, []);

  const allSpans = (paper?.sections ?? []).flatMap((s) =>
    s.paragraphs.flatMap((p) => p.spans),
  );

  const findSpan = useCallback(
    (id: string) => allSpans.find((s) => s.id === id) ?? null,
    [allSpans],
  );
  const sourceText = allSpans.map((span) => span.original).join(" ");

  if (!hasCheckedSession) {
    return <div className="min-h-screen bg-surface" />;
  }

  if (!paper) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-surface-secondary px-4">
        <section className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 text-center shadow-sm">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-primary-100 text-primary-700">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
          </div>
          <h1 className="text-lg font-semibold text-text-primary">{text.reader.noPaperTitle}</h1>
          <p className="mt-2 text-sm text-text-secondary">{text.reader.noPaperDescription}</p>
          <a
            href={`/?lang=${locale}`}
            className="mt-5 inline-flex rounded-xl bg-primary-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-primary-700"
          >
            {text.reader.backToStart}
          </a>
        </section>
      </div>
    );
  }

  function handleSpanClick(spanId: string) {
    setSelectedSpanId(spanId);
    setActiveSourceSpanId(spanId);
    void ensureSpanTranslation(spanId);
  }

  function updateSpanTranslation(spanId: string, translated: string) {
    setPaper((currentPaper) => {
      if (!currentPaper) return currentPaper;
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
    if (!paper) return;
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
    if (!paper || !selectedSpanId) return;
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
                content: text.rightPanel.backendErrorResponse,
                confidence: "low",
                error: "backend request failed",
                usedFallback: true,
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
