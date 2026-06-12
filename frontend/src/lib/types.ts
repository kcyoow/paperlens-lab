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
  isExternalKnowledge?: boolean;
  isBackendGenerated?: boolean;
  isLoading?: boolean;
  model?: string;
  provider?: string;
  traceId?: string;
  error?: string | null;
}
