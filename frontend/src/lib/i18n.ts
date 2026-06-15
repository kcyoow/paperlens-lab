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
  loadingPaper: string;
  loadingHint: string;
  cancelLoading: string;
  loadCanceled: string;
  loadTimeout: string;
  noInputError: string;
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
  noPaperTitle: string;
  noPaperDescription: string;
  backToStart: string;
  translationStatusLabel: string;
  translationStatusRunning: string;
  translationStatusReady: string;
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
  retranslate: string;
  retranslating: string;
  emptyTranslated: string;
  emptyGeneric: string;
  qaReadyTitle: string;
  qaEmptyTitle: string;
  qaEmptyHint: string;
  me: string;
  qaScopeSelected: string;
  qaScopePaper: string;
  paperScopeTitle: string;
  paperScopeHint: string;
  askPlaceholder: string;
  askPaperPlaceholder: string;
  selectFirstPlaceholder: string;
  backendErrorResponse: string;
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
  evidenceSource: string;
  todos: string;
  sourceRunIncluded: string;
  copy: string;
  copied: string;
  runWithSource: string;
  runningWithSource: string;
  sourceRunPassed: string;
  sourceRunFailed: string;
  runMiniLab: string;
  runningMiniLab: string;
  miniLabPassed: string;
  miniLabFailed: string;
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
      loadingPaper: "Loading paper...",
      loadingHint: "Large PDFs can take a little while. The reader opens first; translation continues after loading.",
      cancelLoading: "Cancel loading",
      loadCanceled: "Paper loading was canceled.",
      loadTimeout:
        "The paper is taking too long to load. Try again, paste the paper text, or upload the PDF directly.",
      noInputError: "Add a PDF, arXiv URL, or pasted paper text before opening the reader.",
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
      noPaperTitle: "No paper loaded",
      noPaperDescription: "Start from a PDF, arXiv URL, or pasted paper text so the reader is grounded in a real source.",
      backToStart: "Back to start",
      translationStatusLabel: "KO translation",
      translationStatusRunning: "background sync",
      translationStatusReady: "ready",
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
      retranslate: "Retranslate",
      retranslating: "Retranslating...",
      emptyTranslated:
        "Click a Korean translation to inspect the English source here",
      emptyGeneric: "Click a sentence to compare source and translation",
      qaReadyTitle: "Ask about this selected sentence",
      qaEmptyTitle: "Select a sentence and ask a question",
      qaEmptyHint: "Answers stay linked to the source line",
      me: "Me",
      qaScopeSelected: "Selected",
      qaScopePaper: "Paper",
      paperScopeTitle: "Ask about the paper",
      paperScopeHint: "Answers cite indexed paper evidence",
      askPlaceholder: "Ask about the selected sentence...",
      askPaperPlaceholder: "Ask about the whole paper...",
      selectFirstPlaceholder: "Ask about the whole paper, or select text first",
      backendErrorResponse:
        "The backend could not return a source-grounded answer. Try again after the model/API connection is healthy.",
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
      evidenceSource: "Evidence source",
      todos: "runnable",
      sourceRunIncluded: "source-bound run",
      copy: "Copy",
      copied: "Copied",
      runWithSource: "Run with source",
      runningWithSource: "Running with source...",
      sourceRunPassed: "Source run passed",
      sourceRunFailed: "Source run failed",
      runMiniLab: "Run mini-lab",
      runningMiniLab: "Running mini-lab...",
      miniLabPassed: "Mini-lab passed",
      miniLabFailed: "Mini-lab failed",
      reviewWarning:
        "Review generated code against the indexed paper evidence before expanding it.",
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
        arxiv: "arXiv 주소",
        paste: "텍스트 붙여넣기",
      },
      uploadPrimary: "PDF 파일을 여기에 드래그하거나",
      uploadAction: "파일 선택",
      arxivPlaceholder: "2505.09388 또는 https://arxiv.org/abs/2505.09388",
      arxivHint: "arXiv ID 또는 전체 URL을 입력하세요",
      pastePlaceholder: "논문 텍스트를 붙여넣으세요...",
      startButton: "논문 읽기 시작",
      loadingPaper: "논문 불러오는 중...",
      loadingHint: "큰 PDF는 시간이 조금 걸릴 수 있습니다. 먼저 리더를 열고 번역은 이후에 이어집니다.",
      cancelLoading: "불러오기 취소",
      loadCanceled: "논문 불러오기를 취소했습니다.",
      loadTimeout:
        "논문을 불러오는 데 시간이 너무 오래 걸리고 있습니다. 다시 시도하거나, 논문 텍스트를 붙여넣거나, PDF를 직접 올려주세요.",
      noInputError: "리더를 열기 전에 PDF, arXiv 주소, 또는 논문 텍스트를 넣어주세요.",
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
          desc: "흥미로운 주장이나 결과를 작은 검증 실험으로 바꿔봅니다",
        },
      },
      tagline: "논문을 읽고, 근거를 추적하고, 아이디어를 실험하세요.",
    },
    reader: {
      viewModes: {
        original: "영어",
        translated: "한국어",
        "side-by-side": "나란히",
      },
      languageLabel: "UI 언어",
      noPaperTitle: "불러온 논문이 없습니다",
      noPaperDescription: "실제 원문에 근거한 리더를 열려면 PDF, arXiv URL, 또는 논문 텍스트부터 넣어주세요.",
      backToStart: "처음으로",
      translationStatusLabel: "한국어 번역",
      translationStatusRunning: "백그라운드 진행 중",
      translationStatusReady: "준비 완료",
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
      retranslate: "다시 번역",
      retranslating: "다시 번역 중...",
      emptyTranslated: "번역문을 클릭하면 영어 원문이 여기에 표시됩니다",
      emptyGeneric: "문장을 클릭하면 대조 정보가 표시됩니다",
      qaReadyTitle: "선택한 문장에 대해 질문해보세요",
      qaEmptyTitle: "문장을 선택하고 질문해보세요",
      qaEmptyHint: "답변은 원문 근거와 함께 표시됩니다",
      me: "나",
      qaScopeSelected: "선택",
      qaScopePaper: "논문",
      paperScopeTitle: "논문 전체에 질문",
      paperScopeHint: "답변은 색인된 논문 근거를 인용합니다",
      askPlaceholder: "선택한 문장에 대해 질문...",
      askPaperPlaceholder: "논문 전체에 대해 질문...",
      selectFirstPlaceholder: "논문 전체에 대해 묻거나, 먼저 텍스트를 선택하세요",
      backendErrorResponse:
        "백엔드가 원문 근거가 묶인 답변을 반환하지 못했습니다. 모델/API 연결이 정상인지 확인한 뒤 다시 시도해주세요.",
    },
    lab: {
      subtitle: "선택한 문장을 실험으로 바꿉니다",
      selectedSource: "선택한 원문",
      paperSays: "논문 근거",
      interpretation: "우리의 해석",
      unsupportedAssumption: "논문 밖 가정",
      hypothesis: "가설",
      baseline: "기준선",
      metric: "평가 지표",
      failureCondition: "실패 조건",
      ablation: "제거 실험",
      evidenceSource: "원문 근거",
      todos: "실행 가능",
      sourceRunIncluded: "원문 근거 실행",
      copy: "복사",
      copied: "복사됨",
      runWithSource: "원문 근거로 실행",
      runningWithSource: "원문 근거로 실행 중...",
      sourceRunPassed: "원문 실행 통과",
      sourceRunFailed: "원문 실행 실패",
      runMiniLab: "미니 실험 실행",
      runningMiniLab: "미니 실험 실행 중...",
      miniLabPassed: "미니 실험 통과",
      miniLabFailed: "미니 실험 실패",
      reviewWarning:
        "확장하기 전에 생성 코드가 인덱싱된 논문 근거에서 어떻게 동작하는지 검토하세요.",
      close: "닫기",
      download: "다운로드",
    },
  },
};

export function getInitialLocale(value: string | null | undefined): Locale {
  return value === "ko" ? "ko" : "en";
}
