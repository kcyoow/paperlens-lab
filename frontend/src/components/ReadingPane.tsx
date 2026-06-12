"use client";

import { PaperDocument, ViewMode, Annotation, AnnotationType } from "@/lib/types";

const ANNOTATION_STYLE: Record<AnnotationType, string> = {
  highlight: "span-highlight",
  underline: "span-underline",
  question: "span-question",
  experiment: "span-experiment",
  limitation: "span-limitation",
};

interface Props {
  paper: PaperDocument;
  viewMode: ViewMode;
  selectedSpanId: string | null;
  activeSourceSpanId: string | null;
  annotations: Annotation[];
  onSpanClick: (spanId: string) => void;
}

export default function ReadingPane({
  paper,
  viewMode,
  selectedSpanId,
  activeSourceSpanId,
  annotations,
  onSpanClick,
}: Props) {
  function getSpanClasses(spanId: string, side: "translated" | "original") {
    const classes: string[] = [
      "inline cursor-pointer rounded-sm border-0 bg-transparent px-0.5 text-left align-baseline [font:inherit] transition-all duration-150",
    ];

    const ann = annotations.find((a) => a.spanId === spanId);
    if (ann) {
      classes.push(ANNOTATION_STYLE[ann.type]);
    }

    if (selectedSpanId === spanId) {
      classes.push("ring-2 ring-primary-400 ring-offset-1");
    }

    if (activeSourceSpanId === spanId && side === "original") {
      classes.push("span-active-source");
    }

    return classes.join(" ");
  }

  return (
    <main className="flex-1 overflow-y-auto">
      <div
        className={`mx-auto px-6 py-6 ${viewMode === "side-by-side" ? "max-w-none" : "max-w-none"}`}
      >
        {/* Paper Title */}
        <div className="mb-8">
          <h1 className="mb-2 text-xl font-bold leading-tight text-text-primary [word-break:keep-all]">
            {viewMode === "translated" ? paper.titleKo : paper.title}
          </h1>
          {viewMode === "side-by-side" && (
            <p className="text-sm text-text-muted">{paper.titleKo}</p>
          )}
          <p className="mt-1 text-sm text-text-secondary">
            {paper.authors.join(", ")}
          </p>
        </div>

        {/* Sections */}
        {paper.sections.map((section) => (
          <div key={section.id} id={section.id} className="mb-8">
            <h2 className="mb-4 border-b border-border pb-2 text-lg font-semibold text-text-primary">
              {viewMode === "translated" ? section.titleKo : section.title}
              {viewMode === "side-by-side" && (
                <span className="ml-3 text-sm font-normal text-text-muted">
                  {section.titleKo}
                </span>
              )}
            </h2>

            {section.paragraphs.map((para) => (
              <div key={para.id} className="mb-5">
                {viewMode === "side-by-side" ? (
                  <div className="grid grid-cols-2 gap-6">
                    {/* English source */}
                    <div className="space-y-1 text-[15px] leading-relaxed text-text-primary">
                      {para.spans.map((span) => (
                        <button
                          type="button"
                          key={span.id}
                          data-span-id={span.id}
                          data-testid={`paper-span-${span.id}`}
                          aria-label={`Select source span ${span.id}`}
                          onClick={() => onSpanClick(span.id)}
                          className={getSpanClasses(span.id, "original")}
                        >
                          {span.original}{" "}
                        </button>
                      ))}
                    </div>
                    {/* Korean translation */}
                    <div className="space-y-1 border-l border-border pl-6 text-[14px] leading-relaxed text-text-secondary">
                      {para.spans.map((span) => (
                        <button
                          type="button"
                          key={span.id}
                          data-span-id={span.id}
                          data-testid={`paper-span-${span.id}`}
                          aria-label={`Select translated span ${span.id}`}
                          onClick={() => onSpanClick(span.id)}
                          className={getSpanClasses(span.id, "translated")}
                        >
                          {span.translated}{" "}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p className="text-[15px] leading-relaxed text-text-primary">
                    {para.spans.map((span) => (
                      <button
                        type="button"
                        key={span.id}
                        data-span-id={span.id}
                        data-testid={`paper-span-${span.id}`}
                        aria-label={`Select span ${span.id}`}
                        onClick={() => onSpanClick(span.id)}
                        className={getSpanClasses(
                          span.id,
                          viewMode === "original" ? "original" : "translated",
                        )}
                      >
                        {viewMode === "original"
                          ? span.original
                          : span.translated}{" "}
                      </button>
                    ))}
                  </p>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
    </main>
  );
}
