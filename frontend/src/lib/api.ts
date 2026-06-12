import { EvidenceWindow, PaperDocument, QAMessage, Span } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";
const USE_MODEL = ["1", "true", "yes"].includes(
  (process.env.NEXT_PUBLIC_PAPERLENS_USE_MODEL ?? "").toLowerCase(),
);

export interface PaperLoadInput {
  arxiv_or_url?: string;
  pasted_text?: string;
  max_pdf_pages?: number;
  use_model?: boolean;
  max_translate_spans?: number;
  max_reader_spans?: number;
}

export interface ExperimentResult {
  card: string;
  starter: string;
  spec?: Record<string, unknown>;
  model?: string;
  provider?: string;
  traceId?: string;
  error?: string | null;
  usedFallback?: boolean;
}

export interface GrowthResult {
  ideas: Array<{
    idea: string;
    source_evidence?: string[];
    novelty_angle?: string;
    testable_next_step?: string;
    risk?: string;
  }>;
  fineTuningSignal: string;
  reason: string;
  paperId?: string;
  memoryCount?: number;
  model?: string;
  provider?: string;
  traceId?: string;
  error?: string | null;
  usedFallback?: boolean;
}

export interface ValidationSummary {
  ok: boolean;
  validationRoot?: string;
  warnings: string[];
  realPaperRun?: {
    summaryPath?: string;
    runName?: string;
    artifactDate?: string;
    passed: boolean;
    paperCount: number;
    evaluationPassed: number;
    evaluationTotal: number;
    fineTuningRecommendation: string;
    fineTuningReason: string;
    papers: Array<{
      title: string;
      arxiv: string;
      pageMarkers: number;
      sourceTextChars: number;
      wordCount?: number;
      totalSentenceCount?: number;
      readerSpanLimit?: number;
      translatedSpanCount?: number;
      readerSpans: number;
      evaluationsPassed: number;
      evaluationsTotal: number;
      evaluations?: Array<{ name: string; passed: boolean; reasons: string[] }>;
      memoryRecordsAfterGrowth: number;
    }>;
  } | null;
  modelTraces?: {
    tracePath?: string;
    total: number;
    modelCount: number;
    fallbackCount: number;
    errorCount: number;
    byTask: Record<string, number>;
    byProvider: Record<string, number>;
    byModel: Record<string, number>;
  } | null;
  localDemo?: {
    paperTitle: string;
    readerSpanCount: number;
    sourceTextChars: number;
    selectedSpanId: string;
    evidenceWindow: string;
    sourceHash?: string;
    sourceIndexHash?: string;
    sourceIndexConsistent?: boolean;
    neighborSpans?: Array<{ spanId: string; textHash: string; position?: number }>;
    quoteCount: number;
    confidence: string;
    needsMoreContext?: boolean;
    provider: string;
    model: string;
    traceId?: string;
    usedFallback: boolean;
    translationStatus: string;
    translationTraceId?: string;
    translationUsedFallback: boolean;
  } | null;
  memory?: {
    recordCount: number;
    paperCount: number;
    byKind: Record<string, number>;
  } | null;
}

export interface SpanTranslationResult {
  spanId: string;
  translation: string;
  status: "ready" | "cached" | "fallback";
  model?: string;
  provider?: string;
  traceId?: string;
  error?: string | null;
  usedFallback?: boolean;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Keep the HTTP status text when the server did not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function loadPaper(input: PaperLoadInput): Promise<PaperDocument> {
  const response = await fetch(`${API_BASE}/api/paper`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      arxiv_or_url: input.arxiv_or_url ?? "",
      pasted_text: input.pasted_text ?? "",
      max_pdf_pages: input.max_pdf_pages ?? 10,
      use_model: input.use_model ?? USE_MODEL,
      max_translate_spans: input.max_translate_spans ?? 24,
      max_reader_spans: input.max_reader_spans ?? 180,
    }),
  });
  return parseJson<PaperDocument>(response);
}

export async function loadValidationSummary(): Promise<ValidationSummary> {
  const response = await fetch(`${API_BASE}/api/validation`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  return parseJson<ValidationSummary>(response);
}

export async function uploadPaper(file: File, maxPdfPages = 10): Promise<PaperDocument> {
  const formData = new FormData();
  formData.set("pdf", file);
  formData.set("max_pdf_pages", String(maxPdfPages));
  formData.set("use_model", String(USE_MODEL));
  formData.set("max_translate_spans", "24");
  formData.set("max_reader_spans", "180");

  const response = await fetch(`${API_BASE}/api/paper/upload`, {
    method: "POST",
    body: formData,
  });
  return parseJson<PaperDocument>(response);
}

export async function askAboutSpan(params: {
  paperId: string;
  span: Span;
  paperTitle: string;
  sourceText: string;
  question: string;
  locale: "en" | "ko";
}): Promise<QAMessage> {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: params.paperId,
      span_id: params.span.id,
      question: params.question,
      original: params.span.original,
      translated: params.span.translated,
      paper_title: params.paperTitle,
      source_text: params.sourceText,
      locale: params.locale,
      use_model: USE_MODEL,
    }),
  });
  const body = await parseJson<{
    role: "assistant";
    content: string;
    supportSpanIds?: string[];
    evidence?: Array<{ source_id?: string; quote?: string }>;
    evidenceWindow?: EvidenceWindow | null;
    confidence?: "high" | "medium" | "low";
    needsMoreContext?: boolean;
    model?: string;
    provider?: string;
    traceId?: string;
    error?: string | null;
    usedFallback?: boolean;
  }>(response);
  return {
    id: `qa-${Date.now()}`,
    role: body.role,
    content: body.content,
    supportSpanIds: body.supportSpanIds,
    evidence: body.evidence,
    evidenceWindow: body.evidenceWindow,
    confidence: body.confidence,
    needsMoreContext: body.needsMoreContext,
    isBackendGenerated: true,
    model: body.model,
    provider: body.provider,
    traceId: body.traceId,
    error: body.error,
    usedFallback: body.usedFallback,
  };
}

export async function translateSelectedSpan(params: {
  paperId: string;
  paperTitle: string;
  span: Span;
  locale?: "ko";
}): Promise<SpanTranslationResult> {
  const response = await fetch(`${API_BASE}/api/translate-span`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: params.paperId,
      paper_title: params.paperTitle,
      span_id: params.span.id,
      source_text: params.span.original,
      locale: params.locale ?? "ko",
      use_model: USE_MODEL,
    }),
  });
  return parseJson<SpanTranslationResult>(response);
}

export async function buildExperiment(params: {
  span: Span;
  paperTitle: string;
  sourceText: string;
  locale: "en" | "ko";
}): Promise<ExperimentResult> {
  const response = await fetch(`${API_BASE}/api/experiment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_title: params.paperTitle,
      selected_span: params.span.original,
      translated_span: params.span.translated,
      source_text: params.sourceText,
      idea: params.span.original,
      locale: params.locale,
      use_model: USE_MODEL,
    }),
  });
  return parseJson<ExperimentResult>(response);
}

export async function buildGrowthIdeas(params: {
  paperId?: string;
  span: Span;
  paperTitle: string;
  paperMemory: Array<Record<string, unknown>>;
  miniLabResult: string;
  locale: "en" | "ko";
}): Promise<GrowthResult> {
  const response = await fetch(`${API_BASE}/api/growth`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: params.paperId ?? "",
      paper_title: params.paperTitle,
      selected_span: params.span.original,
      paper_memory: params.paperMemory,
      mini_lab_result: params.miniLabResult,
      locale: params.locale,
      use_model: USE_MODEL,
    }),
  });
  return parseJson<GrowthResult>(response);
}

export function savePaperToSession(paper: PaperDocument) {
  window.sessionStorage.setItem("paperlens-paper", JSON.stringify(paper));
}

export function loadPaperFromSession(): PaperDocument | null {
  const raw = window.sessionStorage.getItem("paperlens-paper");
  if (!raw) return null;
  try {
    return JSON.parse(raw) as PaperDocument;
  } catch {
    window.sessionStorage.removeItem("paperlens-paper");
    return null;
  }
}
