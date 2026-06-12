import type { AnnotationType, ViewMode } from "./types";

export type Locale = "en" | "ko";
export type LandingInputMode = "upload" | "arxiv" | "paste";

interface LandingText {
  headline: string;
  description: string;
  tabs: Record<LandingInputMode, string>;
  uploadPrimary: string;
  uploadAction: string;
  arxivPlaceholder: string;
  arxivHint: string;
  pastePlaceholder: string;
  startButton: string;
  features: {
    sourceCompare: { title: string; desc: string };
    annotation: { title: string; desc: string };
    experiment: { title: string; desc: string };
  };
  tagline: string;
}

interface ReaderText {
  viewModes: Record<ViewMode, string>;
  languageLabel: string;
}

interface AnnotationText {
  tools: Record<AnnotationType, string>;
  askAI: string;
  tryExperiment: string;
}

interface SectionNavText {
  contents: string;
  marks: string;
  emptyLine1: string;
  emptyLine2: string;
}

interface RightPanelText {
  sourceTab: string;
  qaTab: string;
  selectedSentence: string;
  koreanTranslation: string;
  englishSource: string;
  reportTranslation: string;
  retranslate: string;
  emptyTranslated: string;
  emptyGeneric: string;
  qaEmptyTitle: string;
  qaEmptyHint: string;
  me: string;
  askPlaceholder: string;
  selectFirstPlaceholder: string;
  mockQuestionResponse: (spanPreview: string) => string;
}

interface LabText {
  subtitle: string;
  selectedSource: string;
  paperSays: string;
  interpretation: string;
  unsupportedAssumption: string;
  hypothesis: string;
  baseline: string;
  metric: string;
  failureCondition: string;
  ablation: string;
  toyDataset: string;
  todos: string;
  smokeTestIncluded: string;
  copy: string;
  reviewWarning: string;
  close: string;
  download: string;
}

interface TextDictionary {
  common: {
    english: string;
    korean: string;
  };
  landing: LandingText;
  reader: ReaderText;
  annotation: AnnotationText;
  sectionNav: SectionNavText;
  rightPanel: RightPanelText;
  lab: LabText;
}

export const UI_TEXT: Record<Locale, TextDictionary> = {
  en: {
    common: {
      english: "English",
      korean: "Korean",
    },
    landing: {
      headline: "Read, verify, and experiment with papers",
      description:
        "Drop in a PDF or paper, read the English source with Korean translation on demand, highlight important ideas, ask an AI agent about specific lines, and turn promising claims into small experiments.",
      tabs: {
        upload: "PDF upload",
        arxiv: "arXiv URL",
        paste: "Paste text",
      },
      uploadPrimary: "Drag a PDF file here",
      uploadAction: "Choose file",
      arxivPlaceholder: "2505.09388 or https://arxiv.org/abs/2505.09388",
      arxivHint: "Enter an arXiv ID or full URL",
      pastePlaceholder: "Paste paper text here...",
      startButton: "Start reading",
      features: {
        sourceCompare: {
          title: "Source vs. translation",
          desc: "Click a translated line to inspect its English source",
        },
        annotation: {
          title: "Highlights & notes",
          desc: "Mark important lines and ask grounded questions",
        },
        experiment: {
          title: "Try this",
          desc: "Turn an interesting claim into a mini-test",
        },
      },
      tagline: "Read the paper. Trace the source. Try the idea.",
    },
    reader: {
      viewModes: {
        original: "English",
        translated: "Korean",
        "side-by-side": "Side by side",
      },
      languageLabel: "UI language",
    },
    annotation: {
      tools: {
        highlight: "Highlight",
        underline: "Underline",
        question: "Question",
        experiment: "Experiment candidate",
        limitation: "Limitation/Risk",
      },
      askAI: "Ask AI",
      tryExperiment: "Try this",
    },
    sectionNav: {
      contents: "Contents",
      marks: "My marks",
      emptyLine1: "Select a sentence and",
      emptyLine2: "add a mark from the toolbar",
    },
    rightPanel: {
      sourceTab: "Source check",
      qaTab: "Ask AI",
      selectedSentence: "Selected sentence",
      koreanTranslation: "Korean translation",
      englishSource: "English source",
      reportTranslation: "Report issue",
      retranslate: "Retranslate",
      emptyTranslated:
        "Click a Korean translation to inspect the English source here",
      emptyGeneric: "Click a sentence to compare source and translation",
      qaEmptyTitle: "Select a sentence and ask a question",
      qaEmptyHint: "Answers stay linked to the source line",
      me: "Me",
      askPlaceholder: "Ask about the selected sentence...",
      selectFirstPlaceholder: "Select a sentence first",
      mockQuestionResponse: (spanPreview) =>
        `"${spanPreview}..." is tied to one of the paper's core claims. I would answer from the cited source text first, then mark any interpretation that goes beyond the paper.`,
    },
    lab: {
      subtitle: "Turn the selected sentence into an experiment",
      selectedSource: "Selected source",
      paperSays: "Paper says",
      interpretation: "Our interpretation",
      unsupportedAssumption: "Unsupported assumption",
      hypothesis: "Hypothesis",
      baseline: "Baseline",
      metric: "Metric",
      failureCondition: "Failure condition",
      ablation: "Ablation",
      toyDataset: "Toy dataset",
      todos: "2 TODOs",
      smokeTestIncluded: "smoke test included",
      copy: "Copy",
      reviewWarning:
        "Review generated code before running it. TODO-marked parts still need implementation.",
      close: "Close",
      download: "Download",
    },
  },
  ko: {
    common: {
      english: "영어",
      korean: "한국어",
    },
    landing: {
      headline: "논문을 읽고, 확인하고, 실험하다",
      description:
        "PDF나 논문을 넣으면 영어 원문을 기본으로 읽고, 필요할 때 한국어 번역과 대조하며, 밑줄 치고 질문하고, 흥미로운 아이디어를 실제 실험으로 바꿀 수 있습니다.",
      tabs: {
        upload: "PDF 업로드",
        arxiv: "arXiv URL",
        paste: "텍스트 붙여넣기",
      },
      uploadPrimary: "PDF 파일을 여기에 드래그하거나",
      uploadAction: "파일 선택",
      arxivPlaceholder: "2505.09388 또는 https://arxiv.org/abs/2505.09388",
      arxivHint: "arXiv ID 또는 전체 URL을 입력하세요",
      pastePlaceholder: "논문 텍스트를 붙여넣으세요...",
      startButton: "논문 읽기 시작",
      features: {
        sourceCompare: {
          title: "원문-번역 대조",
          desc: "번역문을 클릭하면 영어 원문이 바로 표시됩니다",
        },
        annotation: {
          title: "밑줄 & 메모",
          desc: "중요한 부분에 표시하고 질문할 수 있습니다",
        },
        experiment: {
          title: "실험해보기",
          desc: "흥미로운 claim을 mini-test로 바꿔봅니다",
        },
      },
      tagline: "Read the paper. Trace the source. Try the idea.",
    },
    reader: {
      viewModes: {
        original: "영어",
        translated: "한국어",
        "side-by-side": "나란히",
      },
      languageLabel: "UI 언어",
    },
    annotation: {
      tools: {
        highlight: "형광펜",
        underline: "밑줄",
        question: "물음표",
        experiment: "실험 후보",
        limitation: "한계/위험",
      },
      askAI: "AI에게 묻기",
      tryExperiment: "실험해보기",
    },
    sectionNav: {
      contents: "목차",
      marks: "내 표시",
      emptyLine1: "문장을 선택하고",
      emptyLine2: "도구모음에서 표시를 추가하세요",
    },
    rightPanel: {
      sourceTab: "원문 대조",
      qaTab: "AI 질문",
      selectedSentence: "선택된 문장",
      koreanTranslation: "한국어 번역",
      englishSource: "영어 원문",
      reportTranslation: "번역 이상함",
      retranslate: "다시 번역",
      emptyTranslated: "번역문을 클릭하면 영어 원문이 여기에 표시됩니다",
      emptyGeneric: "문장을 클릭하면 대조 정보가 표시됩니다",
      qaEmptyTitle: "문장을 선택하고 질문해보세요",
      qaEmptyHint: "답변은 원문 근거와 함께 표시됩니다",
      me: "나",
      askPlaceholder: "선택한 문장에 대해 질문...",
      selectFirstPlaceholder: "먼저 문장을 선택하세요",
      mockQuestionResponse: (spanPreview) =>
        `"${spanPreview}..."에 대한 질문이시군요. 이 부분은 논문의 핵심 주장 중 하나와 관련이 있습니다. 원문에서 명시적으로 언급된 내용을 기반으로 답변드리겠습니다.`,
    },
    lab: {
      subtitle: "선택한 문장을 실험으로 바꿉니다",
      selectedSource: "선택한 원문",
      paperSays: "Paper says",
      interpretation: "우리의 해석",
      unsupportedAssumption: "논문 밖 가정",
      hypothesis: "Hypothesis",
      baseline: "Baseline",
      metric: "Metric",
      failureCondition: "Failure condition",
      ablation: "Ablation",
      toyDataset: "Toy dataset",
      todos: "TODO 2개",
      smokeTestIncluded: "smoke-test 포함",
      copy: "복사",
      reviewWarning:
        "코드는 실행 전에 반드시 검토하세요. TODO 표시된 부분은 구현이 필요합니다.",
      close: "닫기",
      download: "다운로드",
    },
  },
};

export function getInitialLocale(value: string | null | undefined): Locale {
  return value === "ko" ? "ko" : "en";
}
