"use client";

import { Section, Annotation, Span, AnnotationType } from "@/lib/types";
import { Locale, UI_TEXT } from "@/lib/i18n";

const ANNOTATION_COLORS: Record<AnnotationType, string> = {
  highlight: "bg-highlight-yellow-strong",
  underline: "bg-highlight-blue-strong",
  question: "bg-highlight-orange-strong",
  experiment: "bg-highlight-purple-strong",
  limitation: "bg-highlight-red-strong",
};

interface Props {
  sections: Section[];
  activeSectionId: string;
  annotations: Annotation[];
  allSpans: Span[];
  locale: Locale;
  onSectionClick: (sectionId: string) => void;
  onSpanClick: (spanId: string) => void;
}

export default function SectionNav({
  sections,
  activeSectionId,
  annotations,
  allSpans,
  locale,
  onSectionClick,
  onSpanClick,
}: Props) {
  const navText = UI_TEXT[locale].sectionNav;
  const annotationText = UI_TEXT[locale].annotation.tools;

  return (
    <aside className="flex w-48 shrink-0 flex-col border-r border-border bg-surface-secondary">
      {/* Section Navigation */}
      <div className="flex-1 overflow-y-auto p-3">
        <p className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
          {navText.contents}
        </p>
        <nav className="space-y-0.5">
          {sections.map((sec) => (
            <button
              key={sec.id}
              onClick={() => onSectionClick(sec.id)}
              className={`w-full rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors ${
                activeSectionId === sec.id
                  ? "bg-primary-50 font-medium text-primary-700"
                  : "text-text-secondary hover:bg-surface-hover hover:text-text-primary"
              }`}
            >
              {locale === "ko" ? sec.titleKo : sec.title}
            </button>
          ))}
        </nav>
      </div>

      {/* Annotations */}
      <div className="shrink-0 border-t border-border p-3">
        <p className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
          {navText.marks} ({annotations.length})
        </p>
        {annotations.length === 0 ? (
          <p className="px-2 text-[11px] text-text-muted">
            {navText.emptyLine1}
            <br />
            {navText.emptyLine2}
          </p>
        ) : (
          <div className="max-h-48 space-y-1 overflow-y-auto">
            {annotations.map((ann) => {
              const span = allSpans.find((s) => s.id === ann.spanId);
              return (
                <button
                  key={ann.id}
                  onClick={() => onSpanClick(ann.spanId)}
                  className="flex w-full items-start gap-2 rounded-lg p-2 text-left transition-colors hover:bg-surface-hover"
                >
                  <span
                    className={`mt-0.5 inline-block h-2.5 w-2.5 shrink-0 rounded-full ${ANNOTATION_COLORS[ann.type]}`}
                    title={annotationText[ann.type]}
                  />
                  <span className="line-clamp-2 text-[11px] text-text-secondary">
                    {(locale === "ko" ? span?.translated : span?.original)?.slice(
                      0,
                      60,
                    )}
                    ...
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}
