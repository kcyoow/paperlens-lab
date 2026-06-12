export interface Span {
  id: string;
  original: string;
  translated: string;
}

export interface Paragraph {
  id: string;
  spans: Span[];
}

export interface Section {
  id: string;
  title: string;
  titleKo: string;
  paragraphs: Paragraph[];
}

export interface PaperDocument {
  id: string;
  title: string;
  titleKo: string;
  authors: string[];
  source: string;
  sections: Section[];
  metadata?: {
    pdfUrl?: string;
    warnings?: string[];
    totalSentenceCount?: number;
    readerSpanCount?: number;
    readerSpanLimit?: number;
    translatedSpanCount?: number;
    sourceTextChars?: number;
  };
}

export type AnnotationType =
  | "highlight"
  | "underline"
  | "question"
  | "experiment"
  | "limitation";

export interface Annotation {
  id: string;
  spanId: string;
  type: AnnotationType;
  note?: string;
  createdAt: number;
}

export type ViewMode = "translated" | "original" | "side-by-side";

export interface QAMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  supportSpanIds?: string[];
  evidence?: Array<{ source_id?: string; quote?: string }>;
  confidence?: "high" | "medium" | "low";
  needsMoreContext?: boolean;
  isExternalKnowledge?: boolean;
  isBackendGenerated?: boolean;
  isLoading?: boolean;
  model?: string;
  provider?: string;
  traceId?: string;
  error?: string | null;
  usedFallback?: boolean;
}
