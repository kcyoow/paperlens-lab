"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  askAboutSpan,
  loadPaperFromSession,
  savePaperToSession,
  translateSpansBatch,
  translateSelectedSpan,
} from "@/lib/api";
import {
  ViewMode,
  Annotation,
  AnnotationType,
  QAMessage,
  PaperDocument,
  Span,
  TextSelection,
  TextSelectionRange,
} from "@/lib/types";
import { getInitialLocale, Locale, UI_TEXT } from "@/lib/i18n";
import {
  displayAuthors,
  displayPaperSource,
  displayPaperTitle,
  isDraftTranslation,
} from "@/lib/translation";
import SectionNav from "@/components/SectionNav";
import ReadingPane from "@/components/ReadingPane";
import RightPanel from "@/components/RightPanel";
import AnnotationBar from "@/components/AnnotationBar";
import LabModal from "@/components/LabModal";

export default function ReaderPage() {
  const [paper, setPaper] = useState<PaperDocument | null>(null);
  const [hasCheckedSession, setHasCheckedSession] = useState(false);
  const annotationsHydratedRef = useRef(false);
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
  const [freeSelectionSpan, setFreeSelectionSpan] = useState<Span | null>(null);
  const [activeTextSelection, setActiveTextSelection] = useState<TextSelection | null>(null);
  const [activeSectionId, setActiveSectionId] = useState<string>("");
  const [translatingSpanIds, setTranslatingSpanIds] = useState<Record<string, boolean>>({});
  const [backgroundTranslation, setBackgroundTranslation] = useState<{
    phase: "idle" | "running" | "done";
    failedSpans: number;
  }>({ phase: "idle", failedSpans: 0 });
  const backgroundTranslationStartedRef = useRef<string | null>(null);
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
  const translatedReadyCount = allSpans.filter(
    (span) => !isDraftTranslation(span.translated),
  ).length;
  const totalSpanCount = allSpans.length;
  const remainingDraftCount = totalSpanCount - translatedReadyCount;
  const hasSyntheticSections = (paper?.sections ?? []).every((section) =>
    isSyntheticSectionTitle(section.title, section.titleKo),
  );
  const translationStatus =
    remainingDraftCount === 0
      ? text.reader.translationStatusReady
      : backgroundTranslation.phase === "running"
        ? text.reader.translationStatusRunning
        : backgroundTranslation.failedSpans > 0
          ? locale === "ko"
            ? "자동 번역 미완료"
            : "Auto-translation incomplete"
          : locale === "ko"
            ? "번역 대기"
            : "Translation pending";
  const translationStatusDetail =
    remainingDraftCount === 0
      ? ""
      : backgroundTranslation.phase === "running"
        ? locale === "ko"
          ? `${remainingDraftCount}개 남음`
          : `${remainingDraftCount} remaining`
        : backgroundTranslation.failedSpans > 0
          ? locale === "ko"
            ? `${remainingDraftCount}개 남음 · 다시 번역 가능`
            : `${remainingDraftCount} remaining · retry available`
          : locale === "ko"
            ? `${remainingDraftCount}개 남음`
            : `${remainingDraftCount} remaining`;

  const findSpan = useCallback(
    (id: string) => allSpans.find((s) => s.id === id) ?? null,
    [allSpans],
  );
  const sourceText = allSpans.map((span) => span.original).join(" ");
  const annotationStorageKey = paper ? annotationStorageKeyForPaper(paper) : "";

  useEffect(() => {
    if (!paper || !annotationStorageKey) return;
    annotationsHydratedRef.current = false;
    setAnnotations(loadPersistedAnnotations(annotationStorageKey));
    annotationsHydratedRef.current = true;
  }, [paper, annotationStorageKey]);

  useEffect(() => {
    if (!annotationStorageKey || !annotationsHydratedRef.current) return;
    savePersistedAnnotations(annotationStorageKey, annotations);
  }, [annotationStorageKey, annotations]);

  useEffect(() => {
    if (!paper) return;
    backgroundTranslationStartedRef.current = null;
    setBackgroundTranslation({ phase: "idle", failedSpans: 0 });
  }, [paper?.id]);

  useEffect(() => {
    if (!paper || backgroundTranslationStartedRef.current === paper.id) return;
    const currentPaperId = paper.id;
    const currentPaperTitle = paper.title;
    const pendingDrafts = allSpans.filter((span) => isDraftTranslation(span.translated));
    backgroundTranslationStartedRef.current = currentPaperId;
    if (pendingDrafts.length === 0) {
      setBackgroundTranslation({ phase: "done", failedSpans: 0 });
      return;
    }

    let cancelled = false;

    async function runBackgroundTranslation() {
      let failedSpans = 0;
      setBackgroundTranslation({ phase: "running", failedSpans: 0 });
      for (let index = 0; index < pendingDrafts.length; index += 8) {
        if (cancelled) return;
        const batch = pendingDrafts.slice(index, index + 8);
        const batchIds = batch.map((span) => span.id);
        setSpanIdsTranslating(batchIds, true);
        try {
          const result = await translateSpansBatch({
            paperId: currentPaperId,
            paperTitle: currentPaperTitle,
            spans: batch,
            locale: "ko",
          });
          const resolved = result.translations
            .filter((item) => item.translation && !isDraftTranslation(item.translation))
            .map((item) => ({
              spanId: item.span_id,
              translated: item.translation,
              status: item.status,
            }));
          failedSpans += result.translations.filter((item) => item.status === "fallback").length;
          if (resolved.length > 0) {
            applySpanTranslations(currentPaperId, resolved);
          }
        } catch {
          failedSpans += batch.length;
        } finally {
          setSpanIdsTranslating(batchIds, false);
        }
        if (!cancelled) {
          setBackgroundTranslation({ phase: "running", failedSpans });
        }
      }
      if (!cancelled) {
        setBackgroundTranslation({ phase: "done", failedSpans });
      }
    }

    void runBackgroundTranslation();

    return () => {
      cancelled = true;
    };
  }, [paper?.id]);

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
    setFreeSelectionSpan(null);
    setActiveTextSelection(null);
    setSelectedSpanId(spanId);
    setActiveSourceSpanId(spanId);
    void ensureSpanTranslation(spanId);
  }

  function handleTextSelection(selection: TextSelection) {
    if (!paper) return;
    const anchor = findSpan(selection.spanId);
    if (!anchor) return;

    const resolved = resolveSelectedSourceText(anchor, selection, findSpan);
    if (!resolved.original) return;

    const selectionWithKey: TextSelection = {
      ...selection,
      selectionKey: selection.selectionKey ?? buildSelectionKey(paper.id, selection),
    };

    setFreeSelectionSpan({
      ...anchor,
      original: resolved.original,
      translated: resolved.translated,
      selectionKind: "free-text",
      selectionKey: selectionWithKey.selectionKey,
    });
    setActiveTextSelection(selectionWithKey);
    setSelectedSpanId(anchor.id);
    setActiveSourceSpanId(anchor.id);
  }

  function applySpanTranslations(
    paperId: string,
    items: Array<{
      spanId: string;
      translated: string;
      status?: Span["translationStatus"];
    }>,
  ) {
    if (items.length === 0) return;
    const translatedById = new Map(
      items
        .filter((item) => item.translated && !isDraftTranslation(item.translated))
        .map((item) => [item.spanId, item]),
    );
    if (translatedById.size === 0) return;
    setPaper((currentPaper) => {
      if (!currentPaper) return currentPaper;
      if (currentPaper.id !== paperId) return currentPaper;
      let translatedSpanCount = 0;
      const nextPaper = {
        ...currentPaper,
        sections: currentPaper.sections.map((section) => ({
          ...section,
          paragraphs: section.paragraphs.map((paragraph) => ({
            ...paragraph,
            spans: paragraph.spans.map((span) => {
              const item = translatedById.get(span.id);
              const translated = item?.translated ?? span.translated;
              if (!isDraftTranslation(translated)) {
                translatedSpanCount += 1;
              }
              if (!item) {
                return span;
              }
              const nextStatus = item.status ?? "ready";
              if (
                translated === span.translated &&
                span.translationStatus === nextStatus
              ) {
                return span;
              }
              return {
                ...span,
                translated,
                translationStatus: nextStatus,
              };
            }),
          })),
        })),
        metadata: {
          ...currentPaper.metadata,
          translatedSpanCount,
        },
      };
      savePaperToSession(nextPaper);
      return nextPaper;
    });
  }

  function setSpanIdsTranslating(spanIds: string[], isTranslating: boolean) {
    setTranslatingSpanIds((current) => {
      if (isTranslating) {
        const next = { ...current };
        for (const spanId of spanIds) {
          next[spanId] = true;
        }
        return next;
      }
      const next = { ...current };
      for (const spanId of spanIds) {
        delete next[spanId];
      }
      return next;
    });
  }

  async function ensureSpanTranslation(spanId: string, options?: { force?: boolean }) {
    if (!paper) return;
    const span = findSpan(spanId);
    const force = options?.force ?? false;
    if (!span || (!force && !isDraftTranslation(span.translated))) return;
    if (translatingSpanIds[spanId]) return;

    setSpanIdsTranslating([spanId], true);
    try {
      const result = await translateSelectedSpan({
        paperId: paper.id,
        paperTitle: paper.title,
        span,
        locale: "ko",
        forceRefresh: force,
      });
      if (result.translation) {
        applySpanTranslations(paper.id, [
          {
            spanId,
            translated: result.translation,
            status: result.status,
          },
        ]);
      }
    } catch {
      // Keep the previous draft/source text when the backend translation path fails.
    } finally {
      setSpanIdsTranslating([spanId], false);
    }
  }

  function handleAnnotate(type: AnnotationType) {
    if (!selectedSpanId) return;
    const exists = annotations.find(
      (a) =>
        a.spanId === selectedSpanId &&
        a.type === type &&
        (activeTextSelection ? a.selectionKey === activeTextSelection.selectionKey : !a.selectionKey),
    );
    if (exists) {
      setAnnotations(annotations.filter((a) => a.id !== exists.id));
    } else {
      const anchor = findSpan(selectedSpanId);
      const resolved = anchor && activeTextSelection
        ? resolveSelectedSourceText(anchor, activeTextSelection, findSpan)
        : null;
      setAnnotations([
        ...annotations,
        {
          id: `ann-${Date.now()}`,
          spanId: selectedSpanId,
          type,
          selectedText: activeTextSelection?.text,
          originalText: resolved?.original,
          translatedText: resolved?.translated,
          ranges: activeTextSelection?.ranges,
          selectionKey: activeTextSelection?.selectionKey,
          surface: activeTextSelection?.surface,
          startOffset: activeTextSelection?.startOffset,
          endOffset: activeTextSelection?.endOffset,
          createdAt: Date.now(),
        },
      ]);
    }
  }

  function handleAnnotationClick(annotation: Annotation) {
    const anchor = findSpan(annotation.spanId);
    if (!anchor) return;
    setSelectedSpanId(annotation.spanId);
    setActiveSourceSpanId(annotation.spanId);
    if (annotation.selectedText) {
      const selection: TextSelection = {
        spanId: annotation.spanId,
        text: annotation.selectedText,
        surface: annotation.surface ?? "original",
        ranges: annotation.ranges,
        selectionKey: annotation.selectionKey,
        startOffset: annotation.startOffset,
        endOffset: annotation.endOffset,
      };
      setActiveTextSelection(selection);
      setFreeSelectionSpan({
        ...anchor,
        original: annotation.originalText || anchor.original,
        translated: annotation.translatedText || "",
        selectionKind: "free-text",
        selectionKey: annotation.selectionKey,
      });
    } else {
      setActiveTextSelection(null);
      setFreeSelectionSpan(null);
    }
    void ensureSpanTranslation(annotation.spanId);
  }

  async function handleAskAI() {
    if (!paper || !selectedSpanId) return;
    setShowQA(true);
    const span = freeSelectionSpan?.id === selectedSpanId ? freeSelectionSpan : findSpan(selectedSpanId);
    if (!span) return;
    const selectedText = isDraftTranslation(span.translated)
      ? span.original
      : span.translated;

    const question =
      locale === "ko"
        ? `이 문장에 대해 설명해주세요: "${selectedText}"`
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
        selectedRanges: selectedEvidenceRangesForAsk(activeTextSelection, findSpan),
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
    const span = freeSelectionSpan?.id === selectedSpanId ? freeSelectionSpan : findSpan(selectedSpanId);
    if (span) {
      setLabSpan({
        ...span,
        translated: isDraftTranslation(span.translated) ? "" : span.translated,
      });
    }
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
              {displayPaperTitle(
                viewMode === "translated" ? paper.titleKo : paper.title,
                locale,
              )}
            </h1>
            <p className="text-xs text-text-muted">
              {displayAuthors(paper.authors, locale)} &middot;{" "}
              {displayPaperSource(paper.source, locale)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="hidden items-center gap-2 rounded-full border border-border bg-surface-secondary px-3 py-1.5 md:flex">
            <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-text-muted">
              {text.reader.translationStatusLabel}
            </span>
            <span className="text-xs font-semibold text-text-primary">
              {translatedReadyCount}/{totalSpanCount}
            </span>
            <span className="text-[11px] text-text-secondary">
              {translationStatus}
              {translationStatusDetail ? ` · ${translationStatusDetail}` : ""}
            </span>
          </div>

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
	          synthetic={hasSyntheticSections}
	          onSectionClick={handleSectionClick}
	          onAnnotationClick={handleAnnotationClick}
	        />

        {/* Center Reading Area */}
        <ReadingPane
          paper={paper}
          locale={locale}
          viewMode={viewMode}
          selectedSpanId={selectedSpanId}
          activeSourceSpanId={activeSourceSpanId}
          annotations={annotations}
          onSpanClick={handleSpanClick}
          onTextSelection={handleTextSelection}
        />

        {/* Right Panel */}
        <RightPanel
          paperId={paper.id}
          selectedSpanId={activeSourceSpanId}
          findSpan={(id) => freeSelectionSpan?.id === id ? freeSelectionSpan : findSpan(id)}
          showQA={showQA}
          qaMessages={qaMessages}
          setQaMessages={setQaMessages}
          setShowQA={setShowQA}
          viewMode={viewMode}
          locale={locale}
          paperTitle={paper.title}
          sourceText={sourceText}
          onRetranslate={
            activeSourceSpanId ? () => void ensureSpanTranslation(activeSourceSpanId, { force: true }) : null
          }
          isRetranslating={Boolean(activeSourceSpanId && translatingSpanIds[activeSourceSpanId])}
        />
      </div>

      {/* Floating Annotation Bar */}
	      <AnnotationBar
	        selectedSpanId={selectedSpanId}
	        activeSelectionKey={activeTextSelection?.selectionKey ?? null}
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

function resolveSelectedSourceText(
  anchor: Span,
  selection: Pick<TextSelection, "text" | "surface" | "ranges" | "startOffset" | "endOffset">,
  findSpan?: (id: string) => Span | null,
): { original: string; translated: string } {
  const normalized = selection.text.replace(/\s+/g, " ").trim();
  if (!normalized) return { original: anchor.original, translated: "" };
  const ranges = (selection.ranges ?? []).filter((range) => range.surface === selection.surface);
  if (ranges.length > 0) {
    if (selection.surface === "original") {
      const original = ranges
        .map((range) => {
          const span = findSpan?.(range.spanId) ?? (range.spanId === anchor.id ? anchor : null);
          const exact = span?.original.slice(range.startOffset, range.endOffset).trim();
          return exact || range.text;
        })
        .filter(Boolean)
        .join(" ")
        .replace(/\s+/g, " ")
        .trim();
      return { original, translated: "" };
    }
    const translated = ranges
      .map((range) => {
        const span = findSpan?.(range.spanId) ?? (range.spanId === anchor.id ? anchor : null);
        const exact = span?.translated.slice(range.startOffset, range.endOffset).trim();
        return exact || range.text;
      })
      .filter(Boolean)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    const original = ranges
      .map((range) => findSpan?.(range.spanId) ?? (range.spanId === anchor.id ? anchor : null))
      .filter((span): span is Span => Boolean(span))
      .map((span) => span.original)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    return { original, translated };
  }
  if (selection.surface === "original") {
    if (anchor.original.includes(normalized)) return { original: normalized, translated: "" };
    const start = typeof selection.startOffset === "number" ? selection.startOffset : -1;
    const end = typeof selection.endOffset === "number" ? selection.endOffset : -1;
    if (start >= 0 && end > start) {
      const exact = anchor.original.slice(start, end).trim();
      if (exact) return { original: exact, translated: "" };
    }
  }
  if (selection.surface === "translated" && anchor.translated.includes(normalized)) {
    return { original: anchor.original, translated: normalized };
  }
  return { original: "", translated: "" };
}

function selectedEvidenceRangesForAsk(
  selection: TextSelection | null,
  findSpan: (id: string) => Span | null,
): TextSelectionRange[] | undefined {
  if (!selection) return undefined;
  if (selection.surface === "original") {
    return selection.ranges?.length
      ? selection.ranges.filter((range) => range.surface === "original")
      : undefined;
  }

  const selectedSpanIds = selection.ranges?.length
    ? selection.ranges.map((range) => range.spanId)
    : [selection.spanId];
  const ranges: TextSelectionRange[] = [];
  const seen = new Set<string>();
  for (const spanId of selectedSpanIds) {
    if (seen.has(spanId)) continue;
    seen.add(spanId);
    const span = findSpan(spanId);
    if (!span?.original) continue;
    ranges.push({
      spanId,
      surface: "original",
      text: span.original,
      startOffset: 0,
      endOffset: span.original.length,
    });
  }
  return ranges.length > 0 ? ranges : undefined;
}

function annotationStorageKeyForPaper(paper: PaperDocument): string {
  const spans = paper.sections.flatMap((section) =>
    section.paragraphs.flatMap((paragraph) => paragraph.spans),
  );
  const signature = hashText(
    [
      paper.id,
      String(spans.length),
      spans[0]?.original ?? "",
      spans[spans.length - 1]?.original ?? "",
    ].join("|"),
  );
  return `paperlens-annotations:${paper.id}:${signature}`;
}

function loadPersistedAnnotations(key: string): Annotation[] {
  try {
    const raw = window.localStorage.getItem(key) ?? window.sessionStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isPersistedAnnotation);
  } catch {
    return [];
  }
}

function savePersistedAnnotations(key: string, annotations: Annotation[]) {
  const serialized = JSON.stringify(annotations);
  try {
    window.sessionStorage.setItem(key, serialized);
  } catch {
    // Keep in-memory marks working if session storage is unavailable.
  }
  try {
    window.localStorage.setItem(key, serialized);
  } catch {
    // Persistence is additive; annotation state remains in React if storage is blocked.
  }
}

function isPersistedAnnotation(value: unknown): value is Annotation {
  if (!value || typeof value !== "object") return false;
  const record = value as Partial<Annotation>;
  return (
    typeof record.id === "string" &&
    typeof record.spanId === "string" &&
    typeof record.type === "string" &&
    ["highlight", "underline", "question", "experiment", "limitation"].includes(record.type) &&
    typeof record.createdAt === "number"
  );
}

function buildSelectionKey(paperId: string, selection: Pick<TextSelection, "spanId" | "text" | "surface" | "ranges" | "startOffset" | "endOffset">) {
  if (selection.ranges?.length) {
    return [
      paperId,
      "ranges",
      selection.surface,
      hashText(
        selection.ranges
          .map((range) => [
            range.spanId,
            range.surface,
            range.startOffset,
            range.endOffset,
            hashText(range.text),
          ].join(":"))
          .join("|"),
      ),
      hashText(selection.text),
    ].join(":");
  }
  return [
    paperId,
    selection.spanId,
    selection.surface,
    selection.startOffset ?? "x",
    selection.endOffset ?? "x",
    hashText(selection.text),
  ].join(":");
}

function hashText(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function isSyntheticSectionTitle(title: string, titleKo: string): boolean {
  const english = title.trim();
  const korean = titleKo.trim();
  return (
    english === "Loaded Paper" ||
    korean === "불러온 논문" ||
    english.startsWith("Source Extract ") ||
    korean.startsWith("원문 추출 ")
  );
}
