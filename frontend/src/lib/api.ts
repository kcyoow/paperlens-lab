import { PaperDocument, QAMessage, Span } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export interface PaperLoadInput {
  arxiv_or_url?: string;
  pasted_text?: string;
  max_pdf_pages?: number;
}

export interface ExperimentResult {
  card: string;
  starter: string;
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
    }),
  });
  return parseJson<PaperDocument>(response);
}

export async function uploadPaper(file: File, maxPdfPages = 10): Promise<PaperDocument> {
  const formData = new FormData();
  formData.set("pdf", file);
  formData.set("max_pdf_pages", String(maxPdfPages));

  const response = await fetch(`${API_BASE}/api/paper/upload`, {
    method: "POST",
    body: formData,
  });
  return parseJson<PaperDocument>(response);
}

export async function askAboutSpan(params: {
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
      span_id: params.span.id,
      question: params.question,
      original: params.span.original,
      translated: params.span.translated,
      paper_title: params.paperTitle,
      source_text: params.sourceText,
      locale: params.locale,
      use_model: false,
    }),
  });
  const body = await parseJson<{
    role: "assistant";
    content: string;
    supportSpanIds?: string[];
  }>(response);
  return {
    id: `qa-${Date.now()}`,
    role: body.role,
    content: body.content,
    supportSpanIds: body.supportSpanIds,
    isBackendGenerated: true,
  };
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
      use_model: false,
    }),
  });
  return parseJson<ExperimentResult>(response);
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
