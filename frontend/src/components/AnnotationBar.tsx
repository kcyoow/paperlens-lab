"use client";

import { Annotation, AnnotationType } from "@/lib/types";
import { Locale, UI_TEXT } from "@/lib/i18n";

interface Props {
  selectedSpanId: string | null;
  annotations: Annotation[];
  onAnnotate: (type: AnnotationType) => void;
  onAskAI: () => void;
  onTryExperiment: () => void;
  locale: Locale;
}

const TOOLS: {
  type: AnnotationType;
  icon: string;
  color: string;
  activeColor: string;
}[] = [
  {
    type: "highlight",
    icon: "M12 20h9M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z",
    color: "text-yellow-600 hover:bg-yellow-50",
    activeColor: "bg-yellow-100 text-yellow-700 ring-1 ring-yellow-300",
  },
  {
    type: "underline",
    icon: "M4 21h16M3 7v1a3 3 0 0 0 6 0V7m4 0v1a3 3 0 0 0 6 0V7",
    color: "text-blue-600 hover:bg-blue-50",
    activeColor: "bg-blue-100 text-blue-700 ring-1 ring-blue-300",
  },
  {
    type: "question",
    icon: "M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3m.08 4h.01M22 12c0 5.523-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2s10 4.477 10 10z",
    color: "text-orange-600 hover:bg-orange-50",
    activeColor: "bg-orange-100 text-orange-700 ring-1 ring-orange-300",
  },
  {
    type: "limitation",
    icon: "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4m0 4h.01",
    color: "text-red-600 hover:bg-red-50",
    activeColor: "bg-red-100 text-red-700 ring-1 ring-red-300",
  },
];

export default function AnnotationBar({
  selectedSpanId,
  annotations,
  onAnnotate,
  onAskAI,
  onTryExperiment,
  locale,
}: Props) {
  if (!selectedSpanId) return null;
  const text = UI_TEXT[locale].annotation;

  function isActive(type: AnnotationType) {
    return annotations.some(
      (a) => a.spanId === selectedSpanId && a.type === type,
    );
  }

  return (
    <div className="animate-slide-up fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 items-center gap-1 rounded-2xl border border-border bg-surface px-3 py-2 shadow-lg">
      {/* Annotation Tools */}
      {TOOLS.map((tool) => (
        <button
          key={tool.type}
          onClick={() => onAnnotate(tool.type)}
          title={text.tools[tool.type]}
          className={`rounded-xl px-3 py-2 text-xs font-medium transition-all ${
            isActive(tool.type) ? tool.activeColor : tool.color
          }`}
        >
          <div className="flex items-center gap-1.5">
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
              <path d={tool.icon} />
            </svg>
            <span className="hidden sm:inline">{text.tools[tool.type]}</span>
          </div>
        </button>
      ))}

      <div className="mx-1 h-6 w-px bg-border" />

      {/* AI Question */}
      <button
        onClick={onAskAI}
        className="flex items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-medium text-primary-700 transition-colors hover:bg-primary-50"
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
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        <span className="hidden sm:inline">{text.askAI}</span>
      </button>

      <div className="mx-1 h-6 w-px bg-border" />

      {/* Try Experiment */}
      <button
        onClick={onTryExperiment}
        className="flex items-center gap-1.5 rounded-xl bg-primary-600 px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-primary-700"
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
          <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 1 1 7.072 0l-.548.547A3.374 3.374 0 0 0 14 18.469V19a2 2 0 1 1-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <span className="hidden sm:inline">{text.tryExperiment}</span>
      </button>
    </div>
  );
}
