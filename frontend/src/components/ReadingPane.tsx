"use client";

import { useRef } from "react";
import type { KeyboardEvent, MouseEvent, PointerEvent, ReactNode } from "react";
import { PaperDocument, ViewMode, Annotation, AnnotationType, TextSelection } from "@/lib/types";
import { Locale } from "@/lib/i18n";
import {
  displayAuthors,
  displayPaperTitle,
  displayTranslationText,
} from "@/lib/translation";

const ANNOTATION_STYLE: Record<AnnotationType, string> = {
  highlight: "span-highlight",
  underline: "span-underline",
  question: "span-question",
  experiment: "span-experiment",
  limitation: "span-limitation",
};

interface PointerSelectionStart {
  spanId: string;
  surface: "translated" | "original";
  offset: number;
}

interface Props {
  paper: PaperDocument;
  locale: Locale;
  viewMode: ViewMode;
  selectedSpanId: string | null;
  activeSourceSpanId: string | null;
  annotations: Annotation[];
  onSpanClick: (spanId: string) => void;
  onTextSelection: (selection: TextSelection) => void;
}

export default function ReadingPane({
  paper,
  locale,
  viewMode,
  selectedSpanId,
  activeSourceSpanId,
  annotations,
  onSpanClick,
  onTextSelection,
}: Props) {
  const pointerSelectionStartRef = useRef<PointerSelectionStart | null>(null);
  const suppressNextClickRef = useRef(false);

  function handleMouseSelection() {
    window.requestAnimationFrame(() => {
      const selection = window.getSelection();
      const selectedText = normalizeSelectionText(selection?.toString() ?? "");
      if (!selection || selectedText.length < 3) return;

      const anchorElement = closestSpanElement(selection.anchorNode) ?? closestSpanElement(selection.focusNode);
      const surface = anchorElement?.dataset.spanSide === "translated" ? "translated" : "original";
      const ranges = selectionRangesForSurface(selection, surface);
      const firstRange = ranges[0];
      if (!anchorElement || !firstRange) return;
      onTextSelection({
        spanId: firstRange.spanId,
        text: normalizeSelectionText(ranges.map((range) => range.text).join(" ")),
        surface,
        ranges,
        startOffset: firstRange.startOffset,
        endOffset: firstRange.endOffset,
      });
    });
  }

  function handleSpanButtonClick(spanId: string) {
    if (suppressNextClickRef.current) {
      suppressNextClickRef.current = false;
      return;
    }
    if (normalizeSelectionText(window.getSelection()?.toString() ?? "").length >= 3) {
      return;
    }
    onSpanClick(spanId);
  }

  function handleSpanClick(event: MouseEvent<HTMLElement>, spanId: string) {
    event.stopPropagation();
    handleSpanButtonClick(spanId);
  }

  function handleParagraphClick(event: MouseEvent<HTMLElement>, surface: "translated" | "original") {
    if (normalizeSelectionText(window.getSelection()?.toString() ?? "").length >= 3) {
      return;
    }
    const spanElement = spanElementAtPoint(event.currentTarget, event.clientX, event.clientY, surface);
    if (!spanElement?.dataset.spanId) return;
    handleSpanButtonClick(spanElement.dataset.spanId);
  }

  function handleSpanKeyDown(event: KeyboardEvent<HTMLElement>, spanId: string) {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    handleSpanButtonClick(spanId);
  }

  function handleSpanPointerDown(
    event: PointerEvent<HTMLElement>,
    spanId: string,
    surface: "translated" | "original",
  ) {
    if (event.button !== 0) return;
    const offset = textOffsetFromPoint(event.currentTarget, event.clientX, event.clientY);
    if (offset === null) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    pointerSelectionStartRef.current = { spanId, surface, offset };
  }

  function handleSpanPointerUp(
    event: PointerEvent<HTMLElement>,
    spanId: string,
    surface: "translated" | "original",
  ) {
    const start = pointerSelectionStartRef.current;
    pointerSelectionStartRef.current = null;
    if (!start || start.spanId !== spanId || start.surface !== surface) return;
    const endOffset = textOffsetFromPoint(event.currentTarget, event.clientX, event.clientY);
    if (endOffset === null) return;
    const from = Math.min(start.offset, endOffset);
    const to = Math.max(start.offset, endOffset);
    if (to - from < 3) return;
    const selected = normalizeSelectionText((event.currentTarget.textContent ?? "").slice(from, to));
    if (selected.length < 3) return;
    suppressNextClickRef.current = true;
    onTextSelection({
      spanId,
      text: selected,
      surface,
      startOffset: from,
      endOffset: to,
    });
  }

  function getSpanClasses(spanId: string, side: "translated" | "original") {
    const classes: string[] = [
      "relative z-0 inline cursor-pointer select-text rounded-sm border-0 bg-transparent px-0.5 text-left align-baseline [font:inherit] transition-all duration-150",
    ];

    const ann = annotations.find((a) => a.spanId === spanId && !annotationTextForSide(a, side, spanId));
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

  function renderSpanText(spanId: string, side: "translated" | "original", text: string) {
    const sideAnnotations = annotations.filter(
      (a) => annotationBelongsToSpan(a, spanId) && annotationTextForSide(a, side, spanId),
    );
    if (sideAnnotations.length === 0) return text;
    return renderAnnotatedText(text, sideAnnotations, side, spanId);
  }

  return (
    <main className="flex-1 overflow-y-auto">
      <div
        data-reader-root="true"
        onMouseUp={handleMouseSelection}
        onKeyUp={handleMouseSelection}
        className={`mx-auto px-6 py-6 ${viewMode === "side-by-side" ? "max-w-none" : "max-w-none"}`}
      >
        {/* Paper Title */}
        <div className="mb-8">
          <h1 className="mb-2 text-xl font-bold leading-tight text-text-primary [word-break:keep-all]">
            {displayPaperTitle(
              viewMode === "translated" ? paper.titleKo : paper.title,
              locale,
            )}
          </h1>
          {viewMode === "side-by-side" && (
            <p className="text-sm text-text-muted">
              {displayPaperTitle(paper.titleKo, locale)}
            </p>
          )}
          <p className="mt-1 text-sm text-text-secondary">
            {displayAuthors(paper.authors, locale)}
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
                    <div
                      className="space-y-1 text-[15px] leading-relaxed text-text-primary"
                      onClick={(event) => handleParagraphClick(event, "original")}
                    >
                      {para.spans.map((span) => (
                        <button
                          type="button"
                          key={span.id}
                          data-span-id={span.id}
                          data-span-side="original"
                          data-testid={`paper-span-${span.id}`}
                          aria-label={`Select source span ${span.id}`}
                          onPointerDown={(event) => handleSpanPointerDown(event, span.id, "original")}
                          onPointerUp={(event) => handleSpanPointerUp(event, span.id, "original")}
                          onClick={(event) => handleSpanClick(event, span.id)}
                          onKeyDown={(event) => handleSpanKeyDown(event, span.id)}
                          className={getSpanClasses(span.id, "original")}
                        >
                          {renderSpanText(span.id, "original", span.original)}{" "}
                        </button>
                      ))}
                    </div>
                    {/* Korean translation */}
                    <div
                      className="space-y-1 border-l border-border pl-6 text-[14px] leading-relaxed text-text-secondary"
                      onClick={(event) => handleParagraphClick(event, "translated")}
                    >
                      {para.spans.map((span) => (
                        <button
                          type="button"
                          key={span.id}
                          data-span-id={span.id}
                          data-span-side="translated"
                          data-testid={`paper-span-${span.id}`}
                          aria-label={`Select translated span ${span.id}`}
                          onPointerDown={(event) => handleSpanPointerDown(event, span.id, "translated")}
                          onPointerUp={(event) => handleSpanPointerUp(event, span.id, "translated")}
                          onClick={(event) => handleSpanClick(event, span.id)}
                          onKeyDown={(event) => handleSpanKeyDown(event, span.id)}
                          className={getSpanClasses(span.id, "translated")}
                        >
                          {renderSpanText(span.id, "translated", displayTranslationText(
                            span.original,
                            span.translated,
                            locale,
                            span.translationStatus,
                          ))}{" "}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p
                    className="text-[15px] leading-relaxed text-text-primary"
                    onClick={(event) => handleParagraphClick(
                      event,
                      viewMode === "original" ? "original" : "translated",
                    )}
                  >
                    {para.spans.map((span) => (
                      <button
                        type="button"
                        key={span.id}
                        data-span-id={span.id}
                        data-span-side={viewMode === "original" ? "original" : "translated"}
                        data-testid={`paper-span-${span.id}`}
                        aria-label={`Select span ${span.id}`}
                        onPointerDown={(event) => handleSpanPointerDown(
                          event,
                          span.id,
                          viewMode === "original" ? "original" : "translated",
                        )}
                        onPointerUp={(event) => handleSpanPointerUp(
                          event,
                          span.id,
                          viewMode === "original" ? "original" : "translated",
                        )}
                        onClick={(event) => handleSpanClick(event, span.id)}
                        onKeyDown={(event) => handleSpanKeyDown(event, span.id)}
                        className={getSpanClasses(
                          span.id,
                          viewMode === "original" ? "original" : "translated",
                        )}
                      >
                        {renderSpanText(
                          span.id,
                          viewMode === "original" ? "original" : "translated",
                          viewMode === "original"
                            ? span.original
                            : displayTranslationText(
                              span.original,
                              span.translated,
                              locale,
                              span.translationStatus,
                            ),
                        )}{" "}
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

function normalizeSelectionText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function closestSpanElement(node: Node | null): HTMLElement | null {
  if (!node) return null;
  const element = node.nodeType === Node.ELEMENT_NODE
    ? (node as Element)
    : node.parentElement;
  return element?.closest<HTMLElement>("[data-span-id]") ?? null;
}

function selectionRangesForSurface(selection: Selection, surface: "original" | "translated") {
  if (selection.rangeCount === 0) return [];
  const range = selection.getRangeAt(0);
  const root = range.commonAncestorContainer.nodeType === Node.ELEMENT_NODE
    ? (range.commonAncestorContainer as Element)
    : range.commonAncestorContainer.parentElement;
  const owner = root?.closest("[data-reader-root]") ?? document;
  return Array.from(owner.querySelectorAll<HTMLElement>("[data-span-id][data-span-side]"))
    .filter((element) => element.dataset.spanSide === surface && rangeIntersectsElement(range, element))
    .map((element) => {
      const offsets = rangeOffsetsWithinElement(element, range);
      if (!offsets) return null;
      const selected = normalizeSelectionText((element.textContent ?? "").slice(offsets.start, offsets.end));
      if (selected.length < 1) return null;
      return {
        spanId: element.dataset.spanId ?? "",
        surface,
        text: selected,
        startOffset: offsets.start,
        endOffset: offsets.end,
      };
    })
    .filter((item): item is {
      spanId: string;
      surface: "original" | "translated";
      text: string;
      startOffset: number;
      endOffset: number;
    } => Boolean(item?.spanId));
}

function spanElementAtPoint(
  container: HTMLElement,
  x: number,
  y: number,
  surface: "original" | "translated",
): HTMLElement | null {
  const direct = container.ownerDocument.elementFromPoint(x, y)?.closest<HTMLElement>("[data-span-id][data-span-side]");
  if (direct?.dataset.spanSide === surface) return direct;
  const spans = Array.from(
    container.querySelectorAll<HTMLElement>(`[data-span-id][data-span-side="${surface}"]`),
  );
  return spans.find((span) =>
    Array.from(span.getClientRects()).some((rect) =>
      x >= rect.left &&
      x <= rect.right &&
      y >= rect.top &&
      y <= rect.bottom,
    ),
  ) ?? null;
}

function rangeIntersectsElement(range: Range, element: HTMLElement): boolean {
  try {
    return range.intersectsNode(element);
  } catch {
    return false;
  }
}

function rangeOffsetsWithinElement(element: HTMLElement, range: Range): { start: number; end: number } | null {
  const text = element.textContent ?? "";
  if (!text) return null;
  let start = 0;
  let end = text.length;
  if (element.contains(range.startContainer)) {
    start = offsetFromElementStart(element, range.startContainer, range.startOffset);
  }
  if (element.contains(range.endContainer)) {
    end = offsetFromElementStart(element, range.endContainer, range.endOffset);
  }
  start = clamp(start, 0, text.length);
  end = clamp(end, 0, text.length);
  if (end <= start) return null;
  return { start, end };
}

function offsetFromElementStart(element: HTMLElement, node: Node, offset: number): number {
  const before = element.ownerDocument.createRange();
  before.selectNodeContents(element);
  before.setEnd(node, offset);
  return before.toString().length;
}

function textOffsetFromPoint(element: HTMLElement, x: number, y: number): number | null {
  const doc = element.ownerDocument as Document & {
    caretPositionFromPoint?: (x: number, y: number) => { offsetNode: Node; offset: number } | null;
    caretRangeFromPoint?: (x: number, y: number) => Range | null;
  };
  const caretPosition = doc.caretPositionFromPoint?.(x, y);
  const caretRange = caretPosition ? null : doc.caretRangeFromPoint?.(x, y);
  const node = caretPosition?.offsetNode ?? caretRange?.startContainer;
  const offset = caretPosition?.offset ?? caretRange?.startOffset;
  if (node && typeof offset === "number" && element.contains(node)) {
    const range = doc.createRange();
    range.selectNodeContents(element);
    range.setEnd(node, offset);
    return range.toString().length;
  }

  const text = element.textContent ?? "";
  if (!text) return null;
  const rect = element.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  const style = window.getComputedStyle(element);
  const parsedLineHeight = Number.parseFloat(style.lineHeight);
  const lineHeight = Number.isFinite(parsedLineHeight)
    ? parsedLineHeight
    : Number.parseFloat(style.fontSize || "16") * 1.4;
  const lineCount = Math.max(1, Math.round(rect.height / Math.max(1, lineHeight)));
  const charsPerLine = Math.max(1, Math.ceil(text.length / lineCount));
  const lineIndex = clamp(Math.floor((y - rect.top) / Math.max(1, lineHeight)), 0, lineCount - 1);
  const columnRatio = clamp((x - rect.left) / rect.width, 0, 1);
  return clamp(lineIndex * charsPerLine + Math.round(columnRatio * charsPerLine), 0, text.length);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function annotationBelongsToSpan(annotation: Annotation, spanId: string): boolean {
  return annotation.spanId === spanId || Boolean(annotation.ranges?.some((range) => range.spanId === spanId));
}

function annotationTextForSide(annotation: Annotation, side: "original" | "translated", spanId?: string): string {
  const range = annotation.ranges?.find((item) => item.spanId === spanId && item.surface === side);
  if (range) return range.text;
  if (side === "original") return annotation.originalText || (annotation.surface === "original" ? annotation.selectedText || "" : "");
  return annotation.translatedText || (annotation.surface === "translated" ? annotation.selectedText || "" : "");
}

function renderAnnotatedText(text: string, annotations: Annotation[], side: "original" | "translated", spanId: string) {
  const ranges = annotationRanges(text, annotations, side, spanId);
  if (ranges.length === 0) return text;
  const parts: ReactNode[] = [];
  let cursor = 0;
  ranges.forEach((range, index) => {
    if (range.start > cursor) {
      parts.push(text.slice(cursor, range.start));
    }
    parts.push(
      <span key={`${range.start}-${range.end}-${index}`} className={range.className}>
        {text.slice(range.start, range.end)}
      </span>,
    );
    cursor = range.end;
  });
  if (cursor < text.length) {
    parts.push(text.slice(cursor));
  }
  return (
    <>
      {parts}
    </>
  );
}

function annotationRanges(text: string, annotations: Annotation[], side: "original" | "translated", spanId: string) {
  const ranges = annotations
    .map((annotation) => {
      const selected = annotationTextForSide(annotation, side, spanId);
      if (!selected) return null;
      const range = annotation.ranges?.find((item) => item.spanId === spanId && item.surface === side);
      const offsetStart =
        range
          ? range.startOffset
          : typeof annotation.startOffset === "number" &&
              typeof annotation.endOffset === "number" &&
              annotation.endOffset > annotation.startOffset &&
              text.slice(annotation.startOffset, annotation.endOffset).trim()
            ? annotation.startOffset
            : -1;
      const offsetEnd =
        range
          ? range.endOffset
          : typeof annotation.endOffset === "number"
            ? annotation.endOffset
          : -1;
      const start = offsetStart >= 0 ? offsetStart : text.indexOf(selected);
      if (start < 0) return null;
      const end =
        offsetStart >= 0 && offsetEnd > offsetStart
          ? Math.min(offsetEnd, text.length)
          : Math.min(start + selected.length, text.length);
      if (end <= start) return null;
      return {
        start,
        end,
        className: ANNOTATION_STYLE[annotation.type],
      };
    })
    .filter((range): range is { start: number; end: number; className: string } => Boolean(range))
    .sort((a, b) => a.start - b.start || a.end - b.end);

  const merged: Array<{ start: number; end: number; className: string }> = [];
  for (const range of ranges) {
    const previous = merged[merged.length - 1];
    if (previous && previous.start === range.start && previous.end === range.end) {
      previous.className = `${previous.className} ${range.className}`;
      continue;
    }
    if (previous && range.start < previous.end) {
      continue;
    }
    merged.push({ ...range });
  }
  return merged;
}
