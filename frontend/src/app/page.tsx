"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  loadPaper,
  loadValidationSummary,
  savePaperToSession,
  uploadPaper,
  ValidationSummary,
} from "@/lib/api";
import { getInitialLocale, LandingInputMode, Locale, UI_TEXT } from "@/lib/i18n";

export default function LandingPage() {
  const router = useRouter();
  const [locale, setLocale] = useState<Locale>("en");
  const [dragOver, setDragOver] = useState(false);
  const [inputMode, setInputMode] = useState<LandingInputMode>("upload");
  const [arxivUrl, setArxivUrl] = useState("");
  const [pastedText, setPastedText] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [validationSummary, setValidationSummary] = useState<ValidationSummary | null>(null);
  const [showEvidencePanel, setShowEvidencePanel] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const loadAbortRef = useRef<AbortController | null>(null);
  const text = UI_TEXT[locale].landing;

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setLocale(getInitialLocale(params.get("lang")));
    const shouldShowEvidence = params.get("evidence") === "1" || params.get("debug") === "1";
    setShowEvidencePanel(shouldShowEvidence);
    if (!shouldShowEvidence) {
      setValidationSummary(null);
      return;
    }
    let cancelled = false;
    loadValidationSummary()
      .then((summary) => {
        if (!cancelled) setValidationSummary(summary);
      })
      .catch(() => {
        if (!cancelled) setValidationSummary(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      loadAbortRef.current?.abort();
    };
  }, []);

  async function handleStart() {
    if (isLoading) return;
    setLoadError("");
    setIsLoading(true);
    const controller = new AbortController();
    loadAbortRef.current = controller;
    try {
      const paper =
        inputMode === "upload" && selectedFile
          ? await uploadPaper(selectedFile, 64, {
              signal: controller.signal,
              timeoutMessage: text.loadTimeout,
              abortMessage: text.loadCanceled,
            })
          : inputMode === "arxiv" && arxivUrl.trim()
            ? await loadPaper(
                {
                  arxiv_or_url: arxivUrl.trim(),
                  max_pdf_pages: 24,
                },
                {
                  signal: controller.signal,
                  timeoutMessage: text.loadTimeout,
                  abortMessage: text.loadCanceled,
                },
              )
            : inputMode === "paste" && pastedText.trim()
              ? await loadPaper(
                  { pasted_text: pastedText.trim() },
                  {
                    signal: controller.signal,
                    timeoutMessage: text.loadTimeout,
                    abortMessage: text.loadCanceled,
                  },
                )
              : null;

      if (!paper) {
        setLoadError(text.noInputError);
        return;
      }
      savePaperToSession(paper);
      router.push(`/reader?lang=${locale}`);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Could not load the paper.");
    } finally {
      if (loadAbortRef.current === controller) {
        loadAbortRef.current = null;
      }
      setIsLoading(false);
    }
  }

  function handleCancelLoading() {
    loadAbortRef.current?.abort();
    setLoadError(text.loadCanceled);
    setIsLoading(false);
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-16">
      <div className="w-full max-w-2xl">
        {/* Logo & Title */}
        <div className="mb-10 text-center">
          <div className="mb-4 inline-flex rounded-lg bg-surface p-1 shadow-sm">
            {(["en", "ko"] as const).map((option) => (
              <button
                key={option}
                onClick={() => setLocale(option)}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                  locale === option
                    ? "bg-primary-600 text-white"
                    : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
                }`}
              >
                {option === "en"
                  ? UI_TEXT[locale].common.english
                  : UI_TEXT[locale].common.korean}
              </button>
            ))}
          </div>
          <div className="mb-3 inline-flex items-center gap-2 rounded-full bg-primary-100 px-4 py-1.5 text-sm font-medium text-primary-700">
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
          </div>
          <h1 className="mb-3 text-3xl font-bold tracking-tight text-text-primary sm:text-4xl">
            {text.headline}
          </h1>
          <p className="mx-auto max-w-lg text-base text-text-secondary">
            {text.description}
          </p>
        </div>

        {/* Input Card */}
        <div className="rounded-2xl border border-border bg-surface p-6 shadow-sm">
          {/* Tab Selector */}
          <div className="mb-5 flex gap-1 rounded-lg bg-surface-secondary p-1">
            {(
              [
                ["upload", text.tabs.upload],
                ["arxiv", text.tabs.arxiv],
                ["paste", text.tabs.paste],
              ] as const
            ).map(([mode, label]) => (
              <button
                key={mode}
                onClick={() => setInputMode(mode)}
                className={`flex-1 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                  inputMode === mode
                    ? "bg-surface text-text-primary shadow-sm"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Upload Area */}
          {inputMode === "upload" && (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const file = e.dataTransfer.files.item(0);
                if (file) setSelectedFile(file);
              }}
              className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed py-16 transition-colors ${
                dragOver
                  ? "border-primary-400 bg-primary-50"
                  : "border-border bg-surface-secondary"
              }`}
            >
              <svg
                className="mb-3 text-text-muted"
                width="40"
                height="40"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="12" y1="18" x2="12" y2="12" />
                <line x1="9" y1="15" x2="12" y2="12" />
                <line x1="15" y1="15" x2="12" y2="12" />
              </svg>
              <p className="mb-1 text-sm font-medium text-text-secondary">
                {selectedFile ? selectedFile.name : text.uploadPrimary}
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                className="hidden"
                onChange={(event) => {
                  setSelectedFile(event.target.files?.item(0) ?? null);
                }}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="text-sm font-medium text-primary-600 hover:text-primary-700"
              >
                {text.uploadAction}
              </button>
            </div>
          )}

          {/* arXiv URL */}
          {inputMode === "arxiv" && (
            <div>
              <input
                type="text"
                value={arxivUrl}
                onChange={(e) => setArxivUrl(e.target.value)}
                placeholder={text.arxivPlaceholder}
                className="w-full rounded-xl border border-border bg-surface-secondary px-4 py-3 text-sm outline-none transition-colors placeholder:text-text-muted focus:border-primary-400 focus:ring-2 focus:ring-primary-100"
              />
              <p className="mt-2 text-xs text-text-muted">
                {text.arxivHint}
              </p>
            </div>
          )}

          {/* Paste */}
          {inputMode === "paste" && (
            <div>
              <textarea
                value={pastedText}
                onChange={(e) => setPastedText(e.target.value)}
                placeholder={text.pastePlaceholder}
                rows={8}
                className="w-full resize-none rounded-xl border border-border bg-surface-secondary px-4 py-3 text-sm outline-none transition-colors placeholder:text-text-muted focus:border-primary-400 focus:ring-2 focus:ring-primary-100"
              />
            </div>
          )}

          {/* Action */}
          <button
            onClick={handleStart}
            disabled={isLoading}
            className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-primary-600 px-6 py-3 text-sm font-semibold text-white transition-colors hover:bg-primary-700 active:bg-primary-800 disabled:cursor-wait disabled:bg-primary-500"
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
              <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
            </svg>
            {isLoading ? text.loadingPaper : text.startButton}
          </button>
          {isLoading && (
            <div className="mt-3 flex flex-col gap-2 rounded-lg border border-border bg-surface-secondary px-3 py-2 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-xs leading-5 text-text-secondary">{text.loadingHint}</p>
              <button
                type="button"
                onClick={handleCancelLoading}
                className="self-start rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-semibold text-text-secondary transition-colors hover:border-primary-300 hover:text-primary-700 sm:self-auto"
              >
                {text.cancelLoading}
              </button>
            </div>
          )}
          {loadError && (
            <p className="mt-3 rounded-lg border border-highlight-red bg-highlight-red/60 px-3 py-2 text-xs text-text-secondary">
              {loadError}
            </p>
          )}
        </div>

        {showEvidencePanel && validationSummary && (
          <ValidationEvidencePanel summary={validationSummary} locale={locale} />
        )}

        {/* Feature Highlights */}
        <div className="mt-8 grid grid-cols-3 gap-4">
          {[
            {
              icon: "M3 5h12M9 3v2m1.048 3.5A18.1 18.1 0 0 1 6.5 14.5M3 13c1.5-1 4-4 5-6.5M14 17l-4-4 4-4",
              title: text.features.sourceCompare.title,
              desc: text.features.sourceCompare.desc,
            },
            {
              icon: "M12 20h9M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z",
              title: text.features.annotation.title,
              desc: text.features.annotation.desc,
            },
            {
              icon: "M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 1 1 7.072 0l-.548.547A3.374 3.374 0 0 0 14 18.469V19a2 2 0 1 1-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z",
              title: text.features.experiment.title,
              desc: text.features.experiment.desc,
            },
          ].map((f) => (
            <div
              key={f.title}
              className="rounded-xl border border-border bg-surface p-4 text-center"
            >
              <div className="mx-auto mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-primary-50 text-primary-600">
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d={f.icon} />
                </svg>
              </div>
              <h3 className="mb-1 text-sm font-semibold">{f.title}</h3>
              <p className="text-xs text-text-secondary">{f.desc}</p>
            </div>
          ))}
        </div>

        {/* Tagline */}
        <p className="mt-8 text-center text-xs text-text-muted">
          {text.tagline}
        </p>
      </div>
    </div>
  );
}

function ValidationEvidencePanel({
  summary,
  locale,
}: {
  summary: ValidationSummary;
  locale: Locale;
}) {
  const realPaperRun = summary.realPaperRun;
  const traces = summary.modelTraces;
  const localDemo = summary.localDemo;
  const memory = summary.memory;
  const frontendExport = summary.frontendStaticExport;
  if (!realPaperRun && !traces && !localDemo && !frontendExport) return null;

  const title = locale === "ko" ? "실제 논문 검증 스냅샷" : "Real-paper validation snapshot";
  const subtitle =
    locale === "ko"
      ? "현재 로컬 검증 결과 파일 기준이며, 새 배포 환경에서는 다시 실행해야 합니다."
      : "Based on local validation artifacts; fresh deployments should run this again.";
  const fineTuning = realPaperRun?.fineTuningRecommendation ?? (locale === "ko" ? "알 수 없음" : "unknown");
  const fineTuningLabel =
    locale === "ko"
      ? fineTuning === "no"
        ? "현재 기준 불필요"
        : fineTuning === "maybe"
          ? "검토 필요"
          : fineTuning === "yes"
            ? "권장"
            : "알 수 없음"
      : fineTuning;
  const papers = realPaperRun?.papers?.map((paper) => paper.arxiv || paper.name).filter(Boolean) ?? [];
  const firstPaper = realPaperRun?.papers?.[0];
  const paperRuns = realPaperRun?.papers ?? [];
  const litm = firstPaper?.adversarialLitm;
  const growthEvidence = firstPaper?.growthIterationEvidence ?? [];
  const growthLoopOk =
    realPaperRun?.growthIterationPassed === true &&
    growthEvidence.some((item) => item.startsWith("growth_idea:")) &&
    growthEvidence.includes("run:r1");
  const starterOk = realPaperRun?.starterCodePassed === true;
  const aggregateOk =
    realPaperRun?.passed === true &&
    realPaperRun?.evaluationTotal === realPaperRun?.evaluationPassed &&
    traces?.traceIdsPassed !== false &&
    (traces?.fallbackCount ?? 0) === 0;
  const localProofOk =
    localDemo?.artifactBundleCoherent === true &&
    localDemo?.traceIdsPassed === true &&
    localDemo?.sourceIndexConsistent !== false &&
    localDemo?.quotesInSourceIndex !== false &&
    localDemo?.translationSourceConsistent !== false &&
    localDemo?.usedFallback !== true &&
    localDemo?.translationUsedFallback !== true;
  const scope = paperRuns.length
    ? locale === "ko"
      ? `${paperRuns.map((paper) => `${paper.pageMarkers}개 페이지 마커`).join(" / ")} · ${paperRuns
          .map((paper) => `${paper.readerSpans}개 리더 문장`)
          .join(" / ")}`
      : `${paperRuns.map((paper) => paper.pageMarkers).join("/")} parsed pages · ${paperRuns
          .map((paper) => paper.readerSpans)
          .join("/")} reader spans`
    : locale === "ko"
      ? "범위 없음"
      : "no scope";
  const litmRatio =
    typeof litm?.target_char_offset_ratio === "number"
      ? `${Math.round(litm.target_char_offset_ratio * 100)}%`
      : locale === "ko"
        ? "없음"
        : "n/a";
  const evidenceOk =
    realPaperRun?.evidenceConsistencyPassed !== false && localDemo?.quoteIdsWithinWindow !== false;
  const frontendExportOk = frontendExport?.ready === true;
  const frontendExportLabel = frontendExportOk
    ? locale === "ko"
      ? "정적 export 준비됨"
      : "static export ready"
    : locale === "ko"
      ? "정적 export 확인 필요"
      : "static export needs check";

  return (
    <section className="mt-5 rounded-2xl border border-border bg-surface p-5 shadow-sm">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-text-primary">{title}</p>
          <p className="mt-1 max-w-xl text-xs text-text-muted">{subtitle}</p>
          {realPaperRun?.artifactDate && (
            <p className="mt-1 text-[11px] text-text-muted">
              {realPaperRun.artifactDate} · {realPaperRun.runName || (locale === "ko" ? "검증 실행" : "validation run")}
            </p>
          )}
        </div>
        <span
          className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${
            summary.ok
              ? "bg-green-100 text-green-700"
              : "bg-yellow-100 text-yellow-700"
          }`}
        >
          {summary.ok
            ? locale === "ko"
              ? "현재 검증 묶음 사용 가능"
              : "validation bundle available"
            : locale === "ko"
              ? "재실행 필요"
              : "needs rerun"}
        </span>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <div className="rounded-2xl border border-border bg-surface-secondary/70 p-4">
          <div className="mb-3 flex items-center justify-between gap-2">
            <p className="text-sm font-semibold text-text-primary">
              {locale === "ko" ? "실제 논문 묶음" : "Real-paper bundle"}
            </p>
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${
                aggregateOk ? "bg-emerald-100 text-emerald-800" : "bg-yellow-100 text-yellow-800"
              }`}
            >
              {aggregateOk
                ? locale === "ko"
                  ? "3개 논문 회귀 통과"
                  : "3-paper regression passed"
                : locale === "ko"
                  ? "회귀 재실행 필요"
                  : "rerun needed"}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <EvidenceMetric
              label={locale === "ko" ? "실제 논문" : "real papers"}
              value={String(realPaperRun?.paperCount ?? 0)}
            />
            <EvidenceMetric
              label={locale === "ko" ? "평가 항목" : "eval passes"}
              value={`${realPaperRun?.evaluationPassed ?? 0}/${realPaperRun?.evaluationTotal ?? 0}`}
            />
            <EvidenceMetric
              label={locale === "ko" ? "모델 추적" : "model traces"}
              value={`${traces?.modelCount ?? 0}/${traces?.total ?? 0}`}
            />
            <EvidenceMetric
              label={locale === "ko" ? "폴백" : "fallbacks"}
              value={String(traces?.fallbackCount ?? 0)}
            />
            <EvidenceMetric
              label={locale === "ko" ? "프론트 파일" : "frontend files"}
              value={String(frontendExport?.fileCount ?? 0)}
            />
          </div>
          <div className="mt-4 space-y-2 text-[11px] leading-5 text-text-muted">
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "검증 논문" : "Papers"}:</span> {papers.length > 0 ? papers.join(" · ") : locale === "ko" ? "없음" : "n/a"}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "처리 범위" : "Scope"}:</span> {scope}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "중간 배치 근거 테스트" : "Long context"}:</span> {litm?.target_span_id || (locale === "ko" ? "없음" : "n/a")} · {litm?.context_chars ?? 0} {locale === "ko" ? "글자" : "chars"} · {litmRatio}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "근거 span ID" : "Evidence IDs"}:</span> {evidenceOk ? (locale === "ko" ? "검증됨" : "verified") : locale === "ko" ? "재실행 필요" : "rerun needed"}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "확장 아이디어 루프" : "Growth loop"}:</span> {growthLoopOk ? (locale === "ko" ? "선택 문장, 실행 기록, 이전 아이디어 메모 연결 확인" : "paper + run:r1 + prior idea") : locale === "ko" ? "재실행 필요" : "rerun needed"}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "미니 실험 코드" : "Mini-lab code"}:</span> {starterOk ? (locale === "ko" ? "논문 근거 행으로 실행 확인" : "source-evidence run verified") : locale === "ko" ? "재실행 필요" : "rerun needed"}</p>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-surface-secondary/70 p-4">
          <div className="mb-3 flex items-center justify-between gap-2">
            <p className="text-sm font-semibold text-text-primary">
              {locale === "ko" ? "현재 데모 선택 span 증거" : "Current selected-span proof"}
            </p>
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${
                localProofOk ? "bg-emerald-100 text-emerald-800" : "bg-yellow-100 text-yellow-800"
              }`}
            >
              {localProofOk
                ? locale === "ko"
                  ? "리더 증거 묶음 일치"
                  : "reader proof coherent"
                : locale === "ko"
                  ? "데모 proof 재생성 필요"
                  : "demo proof needs refresh"}
            </span>
          </div>
          <div className="space-y-2 text-[11px] leading-5 text-text-muted">
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "선택 문장" : "Selected span"}:</span> {localDemo?.selectedSpanId || (locale === "ko" ? "없음" : "n/a")}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "근거 구간" : "Evidence window"}:</span> {localDemo?.evidenceWindow || (locale === "ko" ? "없음" : "n/a")} · {localDemo?.quoteCount ?? 0} {locale === "ko" ? "개 인용" : "quotes"}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "답변 추적" : "Answer trace"}:</span> {localDemo?.traceIdsPassed ? (locale === "ko" ? "실모델 추적 확인" : "model trace verified") : locale === "ko" ? "추적 재확인 필요" : "trace check needed"}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "번역 연결" : "Translation binding"}:</span> {localDemo?.translationSourceConsistent ? (locale === "ko" ? "현재 소스 인덱스와 일치" : "source-index matched") : locale === "ko" ? "재번역 또는 재검증 필요" : "rerun or retranslate needed"}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "현재 소스 인덱스" : "Source index"}:</span> {basename(localDemo?.sourceIndexPath) || (locale === "ko" ? "없음" : "n/a")}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "증거 파일 묶음" : "Artifact bundle"}:</span> {localDemo?.artifactBundleCoherent ? (locale === "ko" ? "같은 문장 기준으로 정렬됨" : "same-span bundle") : locale === "ko" ? "섞인 artifact 가능성" : "possible mixed artifacts"}</p>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-surface-secondary/70 p-4">
          <div className="mb-3 flex items-center justify-between gap-2">
            <p className="text-sm font-semibold text-text-primary">
              {locale === "ko" ? "운영 판단" : "Operational readout"}
            </p>
            <span className="rounded-full bg-surface px-2.5 py-1 text-[10px] font-semibold text-text-secondary">
              {realPaperRun?.artifactDate || (locale === "ko" ? "날짜 없음" : "no date")}
            </span>
          </div>
          <div className="space-y-2 text-[11px] leading-5 text-text-muted">
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "메모리" : "Memory"}:</span> {locale === "ko" ? `${memory?.recordCount ?? 0}개 기록 / ${memory?.paperCount ?? 0}개 논문` : `${memory?.recordCount ?? 0} records across ${memory?.paperCount ?? 0} papers`}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "파인튜닝" : "Fine-tuning"}:</span> {fineTuningLabel}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "추적 계약" : "Trace contract"}:</span> {traces?.traceIdsPassed ? (locale === "ko" ? "필수 추적 ID 확인" : "required trace ids verified") : locale === "ko" ? "누락 추적 있음" : "trace gaps found"}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "프론트 배포 산출물" : "Frontend artifact"}:</span> {frontendExportLabel} · {frontendExport?.hasReader ? "reader" : locale === "ko" ? "reader 누락" : "reader missing"} · {frontendExport?.hasReaderChunk ? "chunk" : locale === "ko" ? "chunk 누락" : "chunk missing"}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "정적 export 크기" : "Static export size"}:</span> {formatBytes(frontendExport?.totalBytes ?? 0)}</p>
            <p><span className="font-semibold text-text-primary">{locale === "ko" ? "판단 기준" : "What this means"}:</span> {summary.ok ? (locale === "ko" ? "현재 저장된 검증 묶음은 데모 기준을 통과했습니다." : "The current stored validation bundle clears the demo bar.") : locale === "ko" ? "적어도 한 축에서 저장된 증거를 다시 생성해야 합니다." : "At least one proof axis needs to be regenerated."}</p>
          </div>
        </div>
      </div>

      {summary.warnings.length > 0 && (
        <p className="mt-3 text-[11px] text-text-muted">
          {summary.warnings.slice(0, 2).join(" · ")}
        </p>
      )}
    </section>
  );
}

function EvidenceMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface-secondary p-3">
      <p className="text-lg font-semibold text-text-primary">{value}</p>
      <p className="mt-0.5 text-[11px] text-text-muted">{label}</p>
    </div>
  );
}

function basename(value: string | undefined): string {
  if (!value) return "";
  const parts = value.split("/");
  return parts[parts.length - 1] || value;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
