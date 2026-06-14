import { EvidenceWindow, PaperDocument, QAMessage, Span, TextSelectionRange } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";
const MODEL_FLAG = process.env.NEXT_PUBLIC_PAPERLENS_USE_MODEL;
const USE_MODEL = MODEL_FLAG === undefined
  ? true
  : ["1", "true", "yes"].includes(MODEL_FLAG.toLowerCase());

export type ReproductionLevel = "probe" | "scaled" | "exact";

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
  implementationRepoManifests?: ImplementationRepoManifest[];
  experimentRunId?: string;
  experimentRun?: {
    id: string;
    paperId: string;
    paperTitle: string;
    spanId: string;
    selectedSpanHash: string;
    codeHash: string;
    experimentTraceId: string;
    starterTraceId: string;
    provider: string;
    model: string;
    starterProvider: string;
    starterModel: string;
    implementationRepoManifests?: ImplementationRepoManifest[];
    expiresAt: number;
  };
  spec?: Record<string, unknown>;
  specDisplay?: Record<string, unknown> | null;
  model?: string;
  provider?: string;
  traceId?: string;
  error?: string | null;
  usedFallback?: boolean;
  starterModel?: string;
  starterProvider?: string;
  starterTraceId?: string;
  starterError?: string | null;
  starterUsedFallback?: boolean;
}

export interface ExperimentCandidate {
  id: string;
  title: string;
  kind: string;
  reproduction_level?: ReproductionLevel;
  faithfulness?: {
    level?: ReproductionLevel;
    summary?: string;
    why_not_exact?: string;
    paper_targets?: string[];
    resource_note?: string;
  };
  is_recommended?: boolean;
  recommendation_reason?: string;
  hypothesis: string;
  paper_evidence_ids: string[];
  paper_evidence_quotes?: string[];
  dataset: {
    name?: string;
    source?: string;
    requires_download?: boolean;
  };
  implementation?: {
    type?: string;
    repo_url?: string;
    reason?: string;
  };
  run_plan?: {
    repo_url?: string;
    config_path?: string;
    command?: string;
    dataset?: string;
    expected_artifact?: string;
    faithfulness_note?: string;
  };
  why_not_exact?: string;
  gpu_required: boolean;
  estimated_runtime_minutes?: number;
  expected_metric: string;
  limitations: string[];
  approval_question?: string;
}

export interface ExperimentCandidatesResult {
  candidateSetId: string;
  candidateSet?: {
    id: string;
    paperId: string;
    paperTitle: string;
    spanId: string;
    selectedSpanHash: string;
    sourceHash: string;
    question: string;
    candidates: ExperimentCandidate[];
    recommendedCandidateId: string;
    candidateTraceId: string;
    provider: string;
    model: string;
    reproductionLevel?: ReproductionLevel;
    implementationLinks?: Array<Record<string, unknown>>;
    expiresAt: number;
  };
  candidates: ExperimentCandidate[];
  recommendedCandidateId: string;
  question: string;
  reproductionLevel?: ReproductionLevel;
  model?: string;
  provider?: string;
  traceId?: string;
  error?: string | null;
  usedFallback?: boolean;
}

export interface GpuScriptResult {
  gpuRunId: string;
  gpuRun?: {
    id: string;
    candidateSetId: string;
    candidateId: string;
    paperId: string;
    paperTitle: string;
    spanId: string;
    selectedSpanHash: string;
    sourceHash: string;
    codeHash: string;
    candidateTraceId: string;
    gpuTraceId: string;
    provider: string;
    model: string;
    reproductionLevel?: ReproductionLevel;
    requestedReproductionLevel?: ReproductionLevel;
    implementationRepoManifests?: ImplementationRepoManifest[];
    expiresAt: number;
  };
  candidate: ExperimentCandidate;
  reproductionLevel?: ReproductionLevel;
  requestedReproductionLevel?: ReproductionLevel;
  script: string;
  entrypoint: string;
  dependencies: string[];
  hardware: string;
  dataset: Record<string, unknown>;
  reproductionPlan?: Record<string, unknown>;
  expectedOutputs: string[];
  paperClaimComparisonPlan: string;
  limitations: string[];
  implementationRepoManifests?: ImplementationRepoManifest[];
  model?: string;
  provider?: string;
  traceId?: string;
  error?: string | null;
  usedFallback?: boolean;
}

export interface GpuProbeRunResult {
  passed: boolean;
  reasons: string[];
  provider: string;
  executionMode: string;
  runner: string;
  gpuRequested: boolean;
  hardware: Record<string, unknown>;
  paperId: string;
  paperTitle: string;
  spanId: string;
  candidateSetId: string;
  candidateId: string;
  sourceHash: string;
  codeHash: string;
  evidenceHash: string;
  evidenceRowCount: number;
  reproductionLevel?: ReproductionLevel;
  requestedReproductionLevel?: ReproductionLevel;
  validation: Record<string, boolean>;
  dataset: Record<string, unknown>;
  metrics: Record<string, unknown>;
  rows: Array<Record<string, unknown>>;
  logs: string[];
  claimComparison?: Record<string, unknown>;
  limitations: string[];
  durationMs: number;
}

export interface ImplementationRepoManifest {
  source_id?: string;
  url: string;
  source_url?: string;
  host?: string;
  usage?: string;
  execution?: "none" | string;
  status: "inspected" | "unavailable" | "rejected" | string;
  commit?: string;
  default_branch?: string;
  file_count?: number;
  total_bytes?: number;
  truncated?: boolean;
  files?: Array<{ path: string; bytes?: number; kind?: string }>;
  readme?: { path: string; excerpt?: string } | null;
  license?: { path: string; excerpt?: string } | null;
  error?: string;
}

export interface GrowthResult {
  ideas: Array<{
    idea: string;
    displayIdea?: string;
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

export interface StarterRunResult {
  passed: boolean;
  reasons: string[];
  rows: Array<Record<string, unknown>>;
  provider?: string;
  executionMode?: string;
  evidenceRowCount?: number;
}

export interface MiniLabRunResult {
  passed: boolean;
  reasons: string[];
  rows: Array<Record<string, unknown>>;
  logs: string[];
  provider: string;
  executionMode: string;
  runner: string;
  paperId: string;
  paperTitle: string;
  spanId: string;
  sourceHash: string;
  selectedSpanHash: string;
  codeHash: string;
  evidenceHash: string;
  evidenceRowCount: number;
  sourceIndexBound: boolean;
  durationMs: number;
  validation: Record<string, boolean>;
  claimComparison?: {
    verdict: string;
    improvedRows: number;
    failedRows: number;
    comparableRows: number;
    metrics: string[];
    limitations: string[];
  };
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
    evidenceConsistencyPassed?: boolean;
    evidenceConsistencyIssues?: string[];
    growthIterationPassed?: boolean;
    starterCodePassed?: boolean;
    fineTuningRecommendation: string;
    fineTuningReason: string;
    papers: Array<{
      name: string;
      title: string;
      arxiv: string;
      pageMarkers: number;
      sourceTextChars: number;
      wordCount?: number;
      totalSentenceCount?: number;
      readerSpanLimit?: number;
      translatedSpanCount?: number;
      readerSpans: number;
      adversarialLitm?: {
        context_span_count?: number;
        context_chars?: number;
        target_span_id?: string;
        target_char_offset_ratio?: number;
        distractor_count?: number;
      };
      evaluationsPassed: number;
      evaluationsTotal: number;
      evaluations?: Array<{ name: string; passed: boolean; reasons: string[] }>;
      memoryRecordsAfterGrowth: number;
      memoryRecordsBeforeGrowthIteration?: number;
      starterCodePassed?: boolean;
      growthIterationPassed?: boolean;
      growthIterationEvidence?: string[];
      growthIterationIdeaEvidence?: string[][];
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
    traceIdsPassed?: boolean;
    traceIdIssues?: string[];
    requiredTraceIdCount?: number;
  } | null;
  localDemo?: {
    paperTitle: string;
    readerSpanCount: number;
    sourceTextChars: number;
    bundleDir?: string;
    artifactBundleCoherent?: boolean;
    selectedSpanId: string;
    evidenceWindow: string;
    sourceIndexPath?: string;
    sourceHash?: string;
    sourceIndexHash?: string;
    sourceIndexConsistent?: boolean;
    evidenceIds?: string[];
    unknownEvidenceIds?: string[];
    badQuoteIds?: string[];
    missingQuoteTextIds?: string[];
    quoteIdsWithinWindow?: boolean;
    quotesInSourceIndex?: boolean;
    neighborSpans?: Array<{ spanId: string; textHash: string; position?: number }>;
    quoteCount: number;
    confidence: string;
    needsMoreContext?: boolean;
    provider: string;
    model: string;
    traceId?: string;
    tracePath?: string;
    traceIdsPassed?: boolean;
    traceIdIssues?: string[];
    usedFallback: boolean;
    translationStatus: string;
    translationTraceId?: string;
    translationUsedFallback: boolean;
    translationSourceHash?: string;
    translationExpectedSourceHash?: string;
    translationSourceIndexBound?: boolean;
    translationSourceConsistent?: boolean;
  } | null;
  frontendStaticExport?: {
    ready: boolean;
    outDir?: string;
    fileCount: number;
    totalBytes: number;
    requiredFiles?: Record<string, boolean>;
    hasIndex?: boolean;
    hasReader?: boolean;
    hasNextStatic?: boolean;
    hasReaderChunk?: boolean;
    readerChunkCount?: number;
    nextStaticFileCount?: number;
    issues?: string[];
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
  sourceHash?: string;
  sourceIndexBound?: boolean;
  model?: string;
  provider?: string;
  traceId?: string;
  error?: string | null;
  usedFallback?: boolean;
}

export interface BatchSpanTranslationItem {
  span_id: string;
  translation: string;
  status: "ready" | "cached" | "fallback";
  sourceHash?: string;
  sourceIndexBound?: boolean;
}

export interface BatchSpanTranslationResult {
  translations: BatchSpanTranslationItem[];
  notes?: string[];
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

function assertServiceBinding(fields: Record<string, string>) {
  for (const [name, value] of Object.entries(fields)) {
    if (!value.trim()) {
      throw new Error(`Missing service binding: ${name}`);
    }
  }
}

const PAPER_STORAGE_KEY = "paperlens-paper";

export async function loadPaper(input: PaperLoadInput): Promise<PaperDocument> {
  const response = await fetch(`${API_BASE}/api/paper`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      arxiv_or_url: input.arxiv_or_url ?? "",
      pasted_text: input.pasted_text ?? "",
      max_pdf_pages: input.max_pdf_pages ?? 64,
      use_model: input.use_model ?? USE_MODEL,
      max_translate_spans: input.max_translate_spans ?? 24,
      max_reader_spans: input.max_reader_spans ?? 800,
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

export async function uploadPaper(file: File, maxPdfPages = 64): Promise<PaperDocument> {
  const formData = new FormData();
  formData.set("pdf", file);
  formData.set("max_pdf_pages", String(maxPdfPages));
  formData.set("use_model", String(USE_MODEL));
  formData.set("max_translate_spans", "24");
  formData.set("max_reader_spans", "800");

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
  selectedRanges?: TextSelectionRange[];
}): Promise<QAMessage> {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: params.paperId,
      span_id: params.span.id,
      scope: "span",
      question: params.question,
      original: params.span.original,
      translated: params.span.translated,
      selected_spans: (params.selectedRanges ?? []).map((range) => ({
        span_id: range.spanId,
        text: range.text,
        surface: range.surface,
        start_offset: range.startOffset,
        end_offset: range.endOffset,
      })),
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

export async function askAboutPaper(params: {
  paperId: string;
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
      span_id: "",
      scope: "paper",
      question: params.question,
      original: "",
      translated: "",
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
  forceRefresh?: boolean;
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
      force_refresh: params.forceRefresh ?? false,
    }),
  });
  return parseJson<SpanTranslationResult>(response);
}

export async function translateSpansBatch(params: {
  paperId: string;
  paperTitle: string;
  spans: Span[];
  locale?: "ko";
}): Promise<BatchSpanTranslationResult> {
  const response = await fetch(`${API_BASE}/api/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: params.paperId,
      paper_title: params.paperTitle,
      spans: params.spans.map((span) => ({
        span_id: span.id,
        text: span.original,
      })),
      locale: params.locale ?? "ko",
      use_model: USE_MODEL,
    }),
  });
  return parseJson<BatchSpanTranslationResult>(response);
}

export async function translateTextBatch(params: {
  paperTitle: string;
  texts: string[];
  locale?: "ko";
}): Promise<string[]> {
  if (params.texts.length === 0) {
    return [];
  }
  const response = await fetch(`${API_BASE}/api/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: "",
      paper_title: params.paperTitle,
      spans: params.texts.map((text, index) => ({
        span_id: `ui-${index}`,
        text,
      })),
      locale: params.locale ?? "ko",
      use_model: USE_MODEL,
    }),
  });
  const body = await parseJson<BatchSpanTranslationResult>(response);
  const byId = new Map(
    body.translations.map((item) => [
      item.span_id,
      item.status === "ready" || item.status === "cached" ? item.translation : "",
    ]),
  );
  return params.texts.map((_, index) => byId.get(`ui-${index}`) ?? "");
}

export async function buildExperiment(params: {
  paperId: string;
  span: Span;
  paperTitle: string;
  sourceText: string;
  locale: "en" | "ko";
}): Promise<ExperimentResult> {
  assertServiceBinding({
    paperId: params.paperId,
    spanId: params.span.id,
    selectedSpan: params.span.original,
    sourceText: params.sourceText,
  });
  const response = await fetch(`${API_BASE}/api/experiment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: params.paperId,
      span_id: params.span.id,
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

export async function buildExperimentCandidates(params: {
  paperId: string;
  span: Span;
  paperTitle: string;
  sourceText: string;
  question: string;
  reproductionLevel: ReproductionLevel;
  locale: "en" | "ko";
}): Promise<ExperimentCandidatesResult> {
  assertServiceBinding({
    paperId: params.paperId,
    spanId: params.span.id,
    selectedSpan: params.span.original,
    sourceText: params.sourceText,
  });
  const response = await fetch(`${API_BASE}/api/experiment/candidates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: params.paperId,
      span_id: params.span.id,
      paper_title: params.paperTitle,
      selected_span: params.span.original,
      translated_span: params.span.translated,
      source_text: params.sourceText,
      question: params.question,
      idea: params.question,
      reproduction_level: params.reproductionLevel,
      locale: params.locale,
      use_model: USE_MODEL,
    }),
  });
  return parseJson<ExperimentCandidatesResult>(response);
}

export async function generateGpuScript(params: {
  candidateSetId: string;
  candidateId: string;
  paperId: string;
  span: Span;
  reproductionLevel: ReproductionLevel;
  locale: "en" | "ko";
}): Promise<GpuScriptResult> {
  assertServiceBinding({
    paperId: params.paperId,
    spanId: params.span.id,
    selectedSpan: params.span.original,
  });
  const response = await fetch(`${API_BASE}/api/experiment/gpu-script`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      candidate_set_id: params.candidateSetId,
      candidate_id: params.candidateId,
      paper_id: params.paperId,
      span_id: params.span.id,
      selected_span: params.span.original,
      reproduction_level: params.reproductionLevel,
      locale: params.locale,
      use_model: USE_MODEL,
    }),
  });
  return parseJson<GpuScriptResult>(response);
}

export async function runGpuProbe(params: {
  gpuRunId: string;
}): Promise<GpuProbeRunResult> {
  const response = await fetch(`${API_BASE}/api/gpu-lab/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      gpu_run_id: params.gpuRunId,
    }),
  });
  return parseJson<GpuProbeRunResult>(response);
}

export async function buildGrowthIdeas(params: {
  paperId: string;
  span: Span;
  paperTitle: string;
  paperMemory: Array<Record<string, unknown>>;
  miniLabResult: string;
  locale: "en" | "ko";
  persistMemory?: boolean;
}): Promise<GrowthResult> {
  assertServiceBinding({
    paperId: params.paperId,
    spanId: params.span.id,
    selectedSpan: params.span.original,
  });
  const response = await fetch(`${API_BASE}/api/growth`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: params.paperId,
      paper_title: params.paperTitle,
      selected_span: params.span.original,
      paper_memory: params.paperMemory,
      mini_lab_result: params.miniLabResult,
      locale: params.locale,
      use_model: USE_MODEL,
      persist_memory: params.persistMemory ?? true,
    }),
  });
  return parseJson<GrowthResult>(response);
}

export async function runMiniLab(params: {
  code: string;
  experimentRunId: string;
  paperId: string;
  paperTitle: string;
  span: Span;
}): Promise<MiniLabRunResult> {
  const response = await fetch(`${API_BASE}/api/mini-lab/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      code: params.code,
      experiment_run_id: params.experimentRunId,
      paper_id: params.paperId,
      paper_title: params.paperTitle,
      span_id: params.span.id,
      selected_span: params.span.original,
    }),
  });
  return parseJson<MiniLabRunResult>(response);
}

export function savePaperToSession(paper: PaperDocument) {
  const serialized = JSON.stringify(paper);
  try {
    window.sessionStorage.setItem(PAPER_STORAGE_KEY, serialized);
  } catch {
    // Keep local storage as a fallback if the session store is unavailable.
  }
  try {
    window.localStorage.setItem(PAPER_STORAGE_KEY, serialized);
  } catch {
    // Keep the in-memory app flow working even when persistent storage is blocked.
  }
}

export function loadPaperFromSession(): PaperDocument | null {
  const raw =
    safeGetPaperStorage(window.sessionStorage) ??
    safeGetPaperStorage(window.localStorage);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as PaperDocument;
    // Rehydrate both stores so refreshes and tab handoffs see the same last paper.
    savePaperToSession(parsed);
    return parsed;
  } catch {
    safeRemovePaperStorage(window.sessionStorage);
    safeRemovePaperStorage(window.localStorage);
    return null;
  }
}

function safeGetPaperStorage(storage: Storage): string | null {
  try {
    return storage.getItem(PAPER_STORAGE_KEY);
  } catch {
    return null;
  }
}

function safeRemovePaperStorage(storage: Storage) {
  try {
    storage.removeItem(PAPER_STORAGE_KEY);
  } catch {
    // Ignore storage cleanup failures and keep the UI usable.
  }
}
