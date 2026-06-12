"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { loadPaper, savePaperToSession, uploadPaper } from "@/lib/api";
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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const text = UI_TEXT[locale].landing;

  useEffect(() => {
    setLocale(getInitialLocale(new URLSearchParams(window.location.search).get("lang")));
  }, []);

  async function handleStart() {
    setLoadError("");
    setIsLoading(true);
    try {
      const paper =
        inputMode === "upload" && selectedFile
          ? await uploadPaper(selectedFile)
          : inputMode === "arxiv" && arxivUrl.trim()
            ? await loadPaper({ arxiv_or_url: arxivUrl.trim() })
            : inputMode === "paste" && pastedText.trim()
              ? await loadPaper({ pasted_text: pastedText.trim() })
              : null;

      if (paper) savePaperToSession(paper);
      router.push(`/reader?lang=${locale}`);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Could not load the paper.");
    } finally {
      setIsLoading(false);
    }
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
            className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-primary-600 px-6 py-3 text-sm font-semibold text-white transition-colors hover:bg-primary-700 active:bg-primary-800"
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
            {isLoading ? "Loading paper..." : text.startButton}
          </button>
          {loadError && (
            <p className="mt-3 rounded-lg border border-highlight-red bg-highlight-red/60 px-3 py-2 text-xs text-text-secondary">
              {loadError}
            </p>
          )}
        </div>

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
