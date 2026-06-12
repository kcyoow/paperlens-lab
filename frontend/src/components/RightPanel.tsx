"use client";

import { Dispatch, SetStateAction, useEffect, useState } from "react";
import { askAboutSpan } from "@/lib/api";
import { Span, QAMessage, ViewMode } from "@/lib/types";
import { Locale, UI_TEXT } from "@/lib/i18n";

interface Props {
  paperId: string;
  selectedSpanId: string | null;
  findSpan: (id: string) => Span | null;
  showQA: boolean;
  qaMessages: QAMessage[];
  setQaMessages: Dispatch<SetStateAction<QAMessage[]>>;
  setShowQA: (show: boolean) => void;
  viewMode: ViewMode;
  locale: Locale;
  paperTitle: string;
  sourceText: string;
}

export default function RightPanel({
  paperId,
  selectedSpanId,
  findSpan,
  showQA,
  qaMessages,
  setQaMessages,
  setShowQA,
  viewMode,
  locale,
  paperTitle,
  sourceText,
}: Props) {
  const [qaInput, setQaInput] = useState("");
  const span = selectedSpanId ? findSpan(selectedSpanId) : null;
  const text = UI_TEXT[locale].rightPanel;

  const [tab, setTab] = useState<"source" | "qa">("source");

  useEffect(() => {
    if (showQA) setTab("qa");
  }, [showQA]);

  async function handleSendQuestion() {
    if (!qaInput.trim() || !selectedSpanId || !span) return;

    const question = qaInput;
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
    setQaInput("");
    setShowQA(true);
    setTab("qa");

    try {
      const answer = await askAboutSpan({
        paperId,
        span,
        paperTitle,
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
                content: text.mockQuestionResponse(
                  locale === "ko" ? span.translated.slice(0, 40) : span.original.slice(0, 40),
                ),
                isLoading: false,
              }
            : message,
        ),
      );
    }
  }

  return (
    <aside className="flex w-72 shrink-0 flex-col border-l border-border bg-surface">
      {/* Tabs */}
      <div className="flex shrink-0 border-b border-border">
        <button
          onClick={() => setTab("source")}
          className={`flex-1 px-4 py-2.5 text-xs font-medium transition-colors ${
            tab === "source"
              ? "border-b-2 border-primary-500 text-primary-700"
              : "text-text-muted hover:text-text-secondary"
          }`}
        >
          {text.sourceTab}
        </button>
        <button
          onClick={() => setTab("qa")}
          className={`flex-1 px-4 py-2.5 text-xs font-medium transition-colors ${
            tab === "qa"
              ? "border-b-2 border-primary-500 text-primary-700"
              : "text-text-muted hover:text-text-secondary"
          }`}
        >
          {text.qaTab} {qaMessages.length > 0 && `(${qaMessages.length})`}
        </button>
      </div>

      {/* Source Tab */}
      {tab === "source" && (
        <div className="flex-1 overflow-y-auto p-4">
          {span ? (
            <div className="animate-fade-in">
              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
                {text.selectedSentence}
              </p>
              <span className="mb-3 inline-block rounded bg-primary-50 px-1.5 py-0.5 text-[10px] font-mono text-primary-700">
                {selectedSpanId}
              </span>

              {/* Original */}
              <div className="mb-4 rounded-lg border border-primary-200 bg-primary-50/50 p-3">
                <p className="mb-1 text-[10px] font-semibold text-primary-700">
                  {text.englishSource}
                </p>
                <p className="text-sm leading-relaxed text-text-primary">
                  {span.original}
                </p>
              </div>

              {/* Translation */}
              <div className="rounded-lg border border-border p-3">
                <p className="mb-1 text-[10px] font-semibold text-text-muted">
                  {text.koreanTranslation}
                </p>
                <p className="text-sm leading-relaxed text-text-primary">
                  {span.translated}
                </p>
              </div>

              {/* Actions */}
              <div className="mt-4 flex gap-2">
                <button className="rounded-lg border border-border px-3 py-1.5 text-[11px] font-medium text-text-secondary transition-colors hover:bg-surface-hover">
                  {text.reportTranslation}
                </button>
                <button className="rounded-lg border border-border px-3 py-1.5 text-[11px] font-medium text-text-secondary transition-colors hover:bg-surface-hover">
                  {text.retranslate}
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <svg
                className="mb-3 text-text-muted"
                width="32"
                height="32"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
              <p className="text-xs text-text-muted">
                {viewMode === "translated"
                  ? text.emptyTranslated
                  : text.emptyGeneric}
              </p>
            </div>
          )}
        </div>
      )}

      {/* QA Tab */}
      {tab === "qa" && (
        <>
          <div className="flex-1 overflow-y-auto p-4">
            {qaMessages.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <svg
                  className="mb-3 text-text-muted"
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <p className="mb-1 text-xs text-text-muted">
                  {text.qaEmptyTitle}
                </p>
                <p className="text-[11px] text-text-muted">
                  {text.qaEmptyHint}
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {qaMessages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`animate-fade-in rounded-lg p-3 ${
                      msg.role === "user"
                        ? "ml-4 bg-primary-50 text-primary-900"
                        : "mr-2 border border-border bg-surface"
                    }`}
                  >
                    <p className="mb-1 text-[10px] font-semibold text-text-muted">
                      {msg.role === "user" ? text.me : "AI"}
                    </p>
                    <p className="text-sm leading-relaxed">{msg.content}</p>
                    {msg.supportSpanIds && msg.supportSpanIds.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {msg.supportSpanIds.map((id) => (
                          <span
                            key={id}
                            className="rounded bg-primary-100 px-1.5 py-0.5 text-[10px] font-mono text-primary-700"
                          >
                            {id}
                          </span>
                        ))}
                      </div>
                    )}
                    {msg.role === "assistant" && (msg.provider || msg.usedFallback || msg.error) && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {msg.confidence && (
                          <span className="rounded bg-surface-secondary px-1.5 py-0.5 text-[10px] text-text-muted">
                            {msg.confidence}
                          </span>
                        )}
                        {msg.provider && (
                          <span className="rounded bg-surface-secondary px-1.5 py-0.5 text-[10px] text-text-muted">
                            {msg.provider}
                          </span>
                        )}
                        {msg.usedFallback && (
                          <span className="rounded bg-yellow-100 px-1.5 py-0.5 text-[10px] text-yellow-700">
                            fallback
                          </span>
                        )}
                        {msg.error && (
                          <span className="rounded bg-red-50 px-1.5 py-0.5 text-[10px] text-red-700">
                            check output
                          </span>
                        )}
                      </div>
                    )}
                    {msg.role === "assistant" && msg.evidenceWindow && (
                      <div className="mt-2 rounded border border-border bg-surface-secondary px-2 py-1.5 text-[10px] leading-relaxed text-text-muted">
                        <p>
                          {locale === "ko" ? "근거 범위" : "Evidence window"}:{" "}
                          <span className="font-mono">{msg.evidenceWindow.spanRange}</span>
                        </p>
                        <p>
                          {locale === "ko" ? "원문 해시" : "Source hash"}:{" "}
                          <span className="font-mono">{msg.evidenceWindow.sourceHash}</span>
                        </p>
                      </div>
                    )}
                    {msg.role === "assistant" && msg.evidence && msg.evidence.length > 0 && (
                      <div className="mt-2 space-y-1 border-l-2 border-primary-200 pl-2">
                        {msg.evidence.slice(0, 2).map((item, index) => (
                          <p key={`${msg.id}-evidence-${index}`} className="text-[10px] leading-relaxed text-text-muted">
                            {item.source_id ? `${item.source_id}: ` : ""}
                            {item.quote}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* QA Input */}
          <div className="shrink-0 border-t border-border p-3">
            <div className="flex gap-2">
              <input
                type="text"
                value={qaInput}
                onChange={(e) => setQaInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSendQuestion()}
                placeholder={
                  span
                    ? text.askPlaceholder
                    : text.selectFirstPlaceholder
                }
                disabled={!span}
                className="flex-1 rounded-lg border border-border bg-surface-secondary px-3 py-2 text-xs outline-none transition-colors placeholder:text-text-muted focus:border-primary-400 disabled:opacity-50"
              />
              <button
                onClick={handleSendQuestion}
                disabled={!span || !qaInput.trim()}
                className="rounded-lg bg-primary-600 px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-primary-700 disabled:opacity-50"
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
          </div>
        </>
      )}
    </aside>
  );
}
