"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  buildExperimentCandidates,
  buildGrowthIdeas,
  ExperimentCandidate,
  ExperimentCandidatesResult,
  generateGpuScript,
  GpuProbeRunResult,
  GpuScriptResult,
  ReproductionLevel,
  SandboxWorkspaceFile,
  runGpuProbe,
} from "@/lib/api";
import { Span } from "@/lib/types";
import { Locale } from "@/lib/i18n";

interface Props {
  span: Span;
  locale: Locale;
  paperId: string;
  paperTitle: string;
  sourceText: string;
  onClose: () => void;
}

type FlowStep = "idle" | "candidates" | "script" | "running" | "result";
type LabLoading = "candidates" | "script" | "run" | "growth" | null;
type SandboxView = "files" | "run" | "logs" | "report" | "result" | "growth";
type SandboxActivityEvent = {
  label: string;
  detail: string;
  state: "done" | "active" | "waiting" | "error";
};

type LevelSlotState = {
  step: FlowStep;
  status: string;
  candidateResult: ExperimentCandidatesResult | null;
  selectedCandidateId: string;
  questionAtGeneration: string;
  scriptResult: GpuScriptResult | null;
  runResult: GpuProbeRunResult | null;
  growthIdeas: string[];
  growthStatus: string;
};

const REPRODUCTION_LEVELS: ReproductionLevel[] = ["probe", "exact"];

export default function LabModal({ span, locale, paperId, paperTitle, sourceText, onClose }: Props) {
  const copy = modalCopy(locale);
  const [question, setQuestion] = useState(copy.defaultQuestion);
  const [reproductionLevel, setReproductionLevel] = useState<ReproductionLevel>("probe");
  const [levelStates, setLevelStates] = useState<Record<ReproductionLevel, LevelSlotState>>(() =>
    createLevelStates(copy),
  );
  const [loading, setLoading] = useState<LabLoading>(null);
  const [activeFilePath, setActiveFilePath] = useState("experiment.py");
  const [sandboxView, setSandboxView] = useState<SandboxView>("files");
  const autoStartedRef = useRef(false);

  const activeState = levelStates[reproductionLevel] ?? createEmptyLevelState(copy);
  const {
    step,
    status,
    candidateResult,
    selectedCandidateId,
    scriptResult,
    runResult,
    growthIdeas,
    growthStatus,
  } = activeState;

  const candidates = candidateResult?.candidates ?? [];
  const recommendedCandidateId = candidateResult?.recommendedCandidateId ?? "";
  const selectedCandidate = useMemo(
    () => candidates.find((candidate) => candidate.id === selectedCandidateId) ?? null,
    [candidates, selectedCandidateId],
  );
  const canGenerateScript = Boolean(candidateResult?.candidateSetId && selectedCandidate && !loading);
  const canRunGpu = Boolean(scriptResult?.gpuRunId && !loading);
  const workspaceFiles = useMemo(
    () => buildWorkspaceFiles(scriptResult, runResult, locale),
    [scriptResult, runResult, locale],
  );
  const activeFile =
    workspaceFiles.find((file) => file.path === activeFilePath) ?? workspaceFiles[0] ?? null;
  const activityEvents = buildSandboxActivity({
    loading,
    step,
    candidateResult,
    scriptResult,
    runResult,
    locale,
  });

  useEffect(() => {
    if (autoStartedRef.current) return;
    autoStartedRef.current = true;
    void handleGenerateCandidates();
    // Lab Mode should open by asking the model for paper-level research directions first.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function updateLevelState(
    level: ReproductionLevel,
    update: Partial<LevelSlotState> | ((state: LevelSlotState) => LevelSlotState),
  ) {
    setLevelStates((previous) => {
      const current = previous[level] ?? createEmptyLevelState(copy);
      const next = typeof update === "function" ? update(current) : { ...current, ...update };
      return { ...previous, [level]: next };
    });
  }

  function updateActiveLevelState(update: Partial<LevelSlotState> | ((state: LevelSlotState) => LevelSlotState)) {
    updateLevelState(reproductionLevel, update);
  }

  function handleLevelChange(level: ReproductionLevel) {
    if (level === reproductionLevel) return;
    setReproductionLevel(level);
  }

  async function handleGenerateCandidates() {
    if (!question.trim()) return;
    const level = reproductionLevel;
    setLoading("candidates");
    setSandboxView("files");
    updateLevelState(level, {
      step: "candidates",
      status: copy.generatingCandidates,
      candidateResult: null,
      selectedCandidateId: "",
      questionAtGeneration: question.trim(),
      scriptResult: null,
      runResult: null,
      growthIdeas: [],
      growthStatus: copy.growthWaiting,
    });
    try {
      const result = await buildExperimentCandidates({
        paperId,
        span,
        paperTitle,
        sourceText,
        question,
        reproductionLevel: level,
        locale,
      });
      updateLevelState(level, {
        candidateResult: result,
        selectedCandidateId: result.recommendedCandidateId || result.candidates[0]?.id || "",
        status: copy.candidatesReady,
      });
    } catch (error) {
      updateLevelState(level, { status: error instanceof Error ? error.message : copy.candidatesFailed });
    } finally {
      setLoading(null);
    }
  }

  async function handleApproveCandidate() {
    if (!candidateResult?.candidateSetId || !selectedCandidate) return;
    const level = reproductionLevel;
    const approvedCandidate = selectedCandidate;
    const approvedCandidateSetId = candidateResult.candidateSetId;
    setLoading("script");
    setSandboxView("files");
    updateLevelState(level, {
      step: "script",
      status: copy.generatingScript,
      scriptResult: null,
      runResult: null,
      growthIdeas: [],
      growthStatus: copy.growthWaiting,
    });
    try {
      const result = await generateGpuScript({
        candidateSetId: approvedCandidateSetId,
        candidateId: approvedCandidate.id,
        paperId,
        span,
        reproductionLevel: level,
        locale,
      });
      updateLevelState(level, { scriptResult: result, status: copy.scriptReady });
    } catch (error) {
      updateLevelState(level, { status: error instanceof Error ? error.message : copy.scriptFailed });
    } finally {
      setLoading(null);
    }
  }

  async function handleRunGpu() {
    if (!scriptResult?.gpuRunId) return;
    const level = reproductionLevel;
    const gpuRunId = scriptResult.gpuRunId;
    setLoading("run");
    setSandboxView("logs");
    updateLevelState(level, {
      step: "running",
      status: copy.runningGpu,
      runResult: null,
    });
    try {
      const result = await runGpuProbe({ gpuRunId });
      updateLevelState(level, {
        runResult: result,
        step: "result",
        status: result.passed ? copy.gpuPassed : `${copy.gpuFailed}: ${result.reasons.join("; ")}`,
        growthStatus: copy.growthReady,
      });
      setSandboxView(result.artifacts?.reportHtml ? "report" : "result");
    } catch (error) {
      updateLevelState(level, { status: error instanceof Error ? error.message : copy.gpuFailed });
    } finally {
      setLoading(null);
    }
  }

  async function handleGrowth() {
    if (!runResult || !selectedCandidate) return;
    const level = reproductionLevel;
    const currentRunResult = runResult;
    const currentCandidate = selectedCandidate;
    setLoading("growth");
    updateLevelState(level, { growthStatus: copy.growthLoading });
    try {
      const growth = await buildGrowthIdeas({
        paperId,
        span,
        paperTitle,
        paperMemory: [
          {
            id: "paper:selected-span",
            summary: span.original,
            translation: span.translated,
          },
          {
            id: `candidate:${currentCandidate.id}`,
            summary: currentCandidate.hypothesis,
            source_evidence: currentCandidate.paper_evidence_ids,
          },
        ],
        miniLabResult: summarizeGpuRunForGrowth(currentRunResult, currentCandidate),
        locale,
        persistMemory: false,
      });
      updateLevelState(level, {
        growthIdeas: growth.ideas.map((idea) => idea.displayIdea || idea.idea),
        growthStatus: growth.error ? copy.growthNeedsReview : copy.growthDone,
      });
    } catch {
      updateLevelState(level, { growthIdeas: [], growthStatus: copy.growthFailed });
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-black/55 p-2 backdrop-blur-sm sm:p-4">
      <div className="mx-auto flex h-[calc(100dvh-1rem)] min-h-0 w-full max-w-[1540px] flex-col overflow-hidden rounded-lg border border-neutral-800 bg-[#090c10] shadow-2xl sm:h-[92dvh]">
        <header className="flex shrink-0 items-center justify-between gap-4 border-b border-neutral-800 bg-[#10151b] px-4 py-3 text-slate-100 sm:px-5">
          <div className="min-w-0">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <h2 className="truncate text-lg font-semibold text-slate-50">{copy.title}</h2>
              <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2.5 py-1 text-[10px] font-semibold text-cyan-100">
                {stepLabel(step, locale)}
              </span>
            </div>
            <p className="mt-1 truncate text-xs text-slate-400">{paperTitle}</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-50"
            aria-label={copy.close}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </header>

        <div className="min-h-0 flex-1 bg-[#090c10]">
            <SandboxShell
              locale={locale}
              copy={copy}
              paperTitle={paperTitle}
              span={span}
              reproductionLevel={reproductionLevel}
              levelStates={levelStates}
              question={question}
              status={status}
              activeState={activeState}
              candidates={candidates}
              recommendedCandidateId={recommendedCandidateId}
              selectedCandidateId={selectedCandidateId}
              canGenerateScript={canGenerateScript}
              view={sandboxView}
              onViewChange={setSandboxView}
              activityEvents={activityEvents}
              scriptResult={scriptResult}
              runResult={runResult}
              selectedCandidate={scriptResult?.candidate || selectedCandidate}
              workspaceFiles={workspaceFiles}
              activeFile={activeFile}
              onSelectFile={setActiveFilePath}
              canRunGpu={canRunGpu}
              loading={loading}
              step={step}
              growthStatus={growthStatus}
              growthIdeas={growthIdeas}
              onQuestionChange={setQuestion}
              onLevelChange={handleLevelChange}
              onGenerateCandidates={handleGenerateCandidates}
              onSelectCandidate={(candidateId) => updateActiveLevelState({ selectedCandidateId: candidateId })}
              onApproveCandidate={handleApproveCandidate}
              onRunGpu={handleRunGpu}
              onGrowth={handleGrowth}
            />
        </div>
      </div>
    </div>
  );
}

function createEmptyLevelState(copy: ReturnType<typeof modalCopy>): LevelSlotState {
  return {
    step: "idle",
    status: copy.readyStatus,
    candidateResult: null,
    selectedCandidateId: "",
    questionAtGeneration: "",
    scriptResult: null,
    runResult: null,
    growthIdeas: [],
    growthStatus: copy.growthWaiting,
  };
}

function createLevelStates(copy: ReturnType<typeof modalCopy>): Record<ReproductionLevel, LevelSlotState> {
  return {
    probe: createEmptyLevelState(copy),
    exact: createEmptyLevelState(copy),
  };
}

function hasLevelWork(state: LevelSlotState) {
  return Boolean(state.candidateResult || state.scriptResult || state.runResult || state.growthIdeas.length > 0);
}

function ModeToggle({
  locale,
  activeLevel,
  levelStates,
  loading,
  onChange,
  savedLabel,
}: {
  locale: Locale;
  activeLevel: ReproductionLevel;
  levelStates: Record<ReproductionLevel, LevelSlotState>;
  loading: LabLoading;
  onChange: (level: ReproductionLevel) => void;
  savedLabel: string;
}) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-[#090c10] p-1">
      <div className="grid grid-cols-2 gap-1">
        {REPRODUCTION_LEVELS.map((level) => {
          const levelMeta = levelCopy(level, locale);
          const hasSavedWork = hasLevelWork(levelStates[level] ?? createEmptyLevelState(modalCopy(locale)));
          return (
            <button
              key={level}
              type="button"
              onClick={() => onChange(level)}
              disabled={Boolean(loading)}
              className={`rounded-md px-3 py-2 text-xs font-semibold transition-colors disabled:opacity-50 ${
                activeLevel === level
                  ? "bg-cyan-300 text-slate-950 shadow-sm"
                  : "text-slate-400 hover:bg-slate-900 hover:text-slate-100"
              }`}
              title={levelMeta.description}
            >
              <span className="inline-flex items-center justify-center gap-1">
                {levelMeta.label}
                {hasSavedWork && (
                  <span
                    className="inline-block h-1.5 w-1.5 rounded-full bg-primary-500"
                    aria-label={savedLabel}
                  />
                )}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function SandboxShell({
  locale,
  copy,
  paperTitle,
  span,
  reproductionLevel,
  levelStates,
  question,
  status,
  activeState,
  candidates,
  recommendedCandidateId,
  selectedCandidateId,
  canGenerateScript,
  view,
  onViewChange,
  activityEvents,
  scriptResult,
  runResult,
  selectedCandidate,
  workspaceFiles,
  activeFile,
  onSelectFile,
  canRunGpu,
  loading,
  step,
  growthStatus,
  growthIdeas,
  onQuestionChange,
  onLevelChange,
  onGenerateCandidates,
  onSelectCandidate,
  onApproveCandidate,
  onRunGpu,
  onGrowth,
}: {
  locale: Locale;
  copy: ReturnType<typeof modalCopy>;
  paperTitle: string;
  span: Span;
  reproductionLevel: ReproductionLevel;
  levelStates: Record<ReproductionLevel, LevelSlotState>;
  question: string;
  status: string;
  activeState: LevelSlotState;
  candidates: ExperimentCandidate[];
  recommendedCandidateId: string;
  selectedCandidateId: string;
  canGenerateScript: boolean;
  view: SandboxView;
  onViewChange: (view: SandboxView) => void;
  activityEvents: SandboxActivityEvent[];
  scriptResult: GpuScriptResult | null;
  runResult: GpuProbeRunResult | null;
  selectedCandidate: ExperimentCandidate | null;
  workspaceFiles: SandboxWorkspaceFile[];
  activeFile: SandboxWorkspaceFile | null;
  onSelectFile: (path: string) => void;
  canRunGpu: boolean;
  loading: LabLoading;
  step: FlowStep;
  growthStatus: string;
  growthIdeas: string[];
  onQuestionChange: (value: string) => void;
  onLevelChange: (level: ReproductionLevel) => void;
  onGenerateCandidates: () => void;
  onSelectCandidate: (candidateId: string) => void;
  onApproveCandidate: () => void;
  onRunGpu: () => void;
  onGrowth: () => void;
}) {
  const tabs: Array<{ id: SandboxView; label: string; disabled?: boolean }> = [
    { id: "files", label: copy.sandboxFiles },
    { id: "run", label: copy.sandboxRun, disabled: !scriptResult },
    { id: "logs", label: copy.sandboxLogs, disabled: !scriptResult },
    { id: "result", label: copy.sandboxResult, disabled: !runResult && step !== "running" },
    { id: "report", label: copy.sandboxReport, disabled: !runResult?.artifacts?.reportHtml },
    { id: "growth", label: copy.sandboxGrowth, disabled: !runResult },
  ];
  const workspaceLabel = scriptResult?.workspaceId || scriptResult?.gpuRunId || copy.waiting;
  return (
    <section className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-[#090c10] text-slate-100">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-neutral-800 bg-[#10151b] px-3 py-2.5 sm:px-4">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-slate-50 sm:text-base">{copy.sandboxTitle}</p>
          <p className="mt-0.5 text-xs text-slate-500">{copy.subtitle}</p>
        </div>
        <div className="flex min-w-0 flex-wrap items-center justify-end gap-2">
          <span className="max-w-[44vw] truncate rounded-md border border-neutral-700 bg-[#171d24] px-2.5 py-1 text-[10px] font-semibold text-slate-300">
            {workspaceLabel}
          </span>
          <LevelBadge level={scriptResult?.reproductionLevel || selectedCandidate?.reproduction_level || reproductionLevel} locale={locale} />
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-rows-[minmax(230px,38vh)_minmax(0,1fr)] lg:grid-cols-[minmax(300px,350px)_minmax(0,1fr)] lg:grid-rows-1 xl:grid-cols-[minmax(320px,380px)_minmax(0,1fr)]">
        <aside className="min-h-0 overflow-y-auto border-b border-neutral-800 bg-[#0f141a] p-3 lg:border-b-0 lg:border-r">
          <SandboxAgentRail
            locale={locale}
            copy={copy}
            paperTitle={paperTitle}
            span={span}
            reproductionLevel={reproductionLevel}
            levelStates={levelStates}
            question={question}
            status={status}
            activeState={activeState}
            candidates={candidates}
            recommendedCandidateId={recommendedCandidateId}
            selectedCandidateId={selectedCandidateId}
            canGenerateScript={canGenerateScript}
            loading={loading}
            activityEvents={activityEvents}
            selectedCandidate={selectedCandidate}
            scriptResult={scriptResult}
            onQuestionChange={onQuestionChange}
            onLevelChange={onLevelChange}
            onGenerateCandidates={onGenerateCandidates}
            onSelectCandidate={onSelectCandidate}
            onApproveCandidate={onApproveCandidate}
          />
        </aside>

        <main className="flex min-h-0 min-w-0 flex-col bg-[#090c10]">
          <div className="shrink-0 border-b border-neutral-800 bg-[#10151b] px-2 py-2">
            <div className="flex min-w-0 gap-1 overflow-x-auto">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => !tab.disabled && onViewChange(tab.id)}
                  disabled={tab.disabled}
                  className={`shrink-0 rounded-md px-3 py-2 text-xs font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                    view === tab.id
                      ? "bg-cyan-300 text-slate-950"
                      : "text-slate-300 hover:bg-slate-800 hover:text-white"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-auto p-3 sm:p-4">
            {view === "files" && (
              scriptResult ? (
                <WorkspaceFilesPanel
                  files={workspaceFiles}
                  activeFile={activeFile}
                  onSelectFile={onSelectFile}
                  locale={locale}
                />
              ) : (
                <SandboxEmptyState copy={copy} events={activityEvents} locale={locale} />
              )
            )}
            {view === "run" && (
              <SandboxRunPane
                copy={copy}
                scriptResult={scriptResult}
                runResult={runResult}
                canRunGpu={canRunGpu}
                loading={loading}
                step={step}
                onRunGpu={onRunGpu}
              />
            )}
            {view === "logs" && (
              <SandboxLogsPane
                copy={copy}
                events={activityEvents}
                runResult={runResult}
                step={step}
                locale={locale}
              />
            )}
            {view === "result" && (
              runResult ? (
                <GpuResultView result={runResult} locale={locale} />
              ) : (
                <SandboxEmptyState copy={copy} events={activityEvents} locale={locale} />
              )
            )}
            {view === "report" && (
              runResult?.artifacts?.reportHtml ? (
                <ArtifactReportView result={runResult} locale={locale} />
              ) : (
                <SandboxEmptyState copy={copy} events={activityEvents} locale={locale} />
              )
            )}
            {view === "growth" && (
              <SandboxGrowthPane
                copy={copy}
                loading={loading}
                runResult={runResult}
                growthStatus={growthStatus}
                growthIdeas={growthIdeas}
                onGrowth={onGrowth}
              />
            )}
          </div>
        </main>
      </div>
    </section>
  );
}

function SandboxAgentRail({
  locale,
  copy,
  paperTitle,
  span,
  reproductionLevel,
  levelStates,
  question,
  status,
  activeState,
  candidates,
  recommendedCandidateId,
  selectedCandidateId,
  canGenerateScript,
  loading,
  activityEvents,
  selectedCandidate,
  scriptResult,
  onQuestionChange,
  onLevelChange,
  onGenerateCandidates,
  onSelectCandidate,
  onApproveCandidate,
}: {
  locale: Locale;
  copy: ReturnType<typeof modalCopy>;
  paperTitle: string;
  span: Span;
  reproductionLevel: ReproductionLevel;
  levelStates: Record<ReproductionLevel, LevelSlotState>;
  question: string;
  status: string;
  activeState: LevelSlotState;
  candidates: ExperimentCandidate[];
  recommendedCandidateId: string;
  selectedCandidateId: string;
  canGenerateScript: boolean;
  loading: LabLoading;
  activityEvents: SandboxActivityEvent[];
  selectedCandidate: ExperimentCandidate | null;
  scriptResult: GpuScriptResult | null;
  onQuestionChange: (value: string) => void;
  onLevelChange: (level: ReproductionLevel) => void;
  onGenerateCandidates: () => void;
  onSelectCandidate: (candidateId: string) => void;
  onApproveCandidate: () => void;
}) {
  return (
    <div className="flex min-h-full flex-col gap-3">
      <div className="rounded-lg border border-neutral-800 bg-[#141a22] p-3 shadow-sm">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-cyan-200/70">
              {copy.researchKicker}
            </p>
            <h3 className="mt-1 text-sm font-semibold leading-5 text-slate-50">{copy.researchTitle}</h3>
          </div>
          <span className="shrink-0 rounded-md border border-neutral-700 bg-[#090c10] px-2 py-1 text-[10px] font-semibold text-slate-300">
            {candidates.length > 0 ? `${candidates.length}` : copy.waiting}
          </span>
        </div>
        <ModeToggle
          locale={locale}
          activeLevel={reproductionLevel}
          levelStates={levelStates}
          loading={loading}
          onChange={onLevelChange}
          savedLabel={copy.savedLevelWork}
        />
        <label className="mt-3 block text-xs font-semibold text-slate-300" htmlFor="lab-question">
          {copy.askLabel}
        </label>
        <textarea
          id="lab-question"
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          className="mt-2 min-h-20 w-full resize-none rounded-md border border-neutral-700 bg-[#090c10] px-3 py-2 text-sm leading-6 text-slate-100 outline-none placeholder:text-slate-500 focus:border-cyan-300"
        />
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
          <p className="min-w-0 flex-1 text-xs leading-5 text-slate-400">{status}</p>
          <button
            onClick={onGenerateCandidates}
            disabled={loading !== null || !question.trim()}
            className="rounded-md bg-cyan-300 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-200 disabled:opacity-50"
          >
            {loading === "candidates" ? copy.generating : copy.generateCandidates}
          </button>
        </div>
        {activeState.questionAtGeneration && (
          <p className="mt-2 line-clamp-2 text-[11px] leading-5 text-slate-500">
            <span className="font-semibold text-slate-300">{copy.savedQuestion}</span>{" "}
            {activeState.questionAtGeneration}
          </p>
        )}
      </div>

      {candidates.length > 0 && (
        <div className="min-h-0 rounded-lg border border-neutral-800 bg-[#111820] p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="min-w-0">
              <p className="truncate text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                {copy.candidatesTitle}
              </p>
              <p className="mt-0.5 text-[11px] font-medium text-slate-500">{levelCopy(reproductionLevel, locale).label}</p>
            </div>
            <button
              onClick={onApproveCandidate}
              disabled={!canGenerateScript}
              className="shrink-0 rounded-md bg-emerald-400 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-emerald-300 disabled:opacity-50"
            >
              {loading === "script" ? copy.generatingScript : copy.openSandbox}
            </button>
          </div>
          <div className="max-h-[42vh] space-y-2 overflow-y-auto pr-1">
            {candidates.map((candidate) => (
              <SandboxDirectionCard
                key={candidate.id}
                candidate={candidate}
                selected={candidate.id === selectedCandidateId}
                recommended={candidate.id === recommendedCandidateId || candidate.is_recommended === true}
                locale={locale}
                requestedLevel={reproductionLevel}
                onSelect={() => onSelectCandidate(candidate.id)}
              />
            ))}
          </div>
        </div>
      )}

      {scriptResult && (
        <details className="rounded-lg border border-neutral-800 bg-[#111820] p-3">
          <summary className="cursor-pointer text-xs font-semibold text-slate-300">
            {copy.sandboxPlan}
          </summary>
          <div className="mt-3 space-y-3">
            <ReproductionPlan
              candidate={scriptResult.candidate || selectedCandidate}
              scriptResult={scriptResult}
              locale={locale}
            />
            <WorkspaceProvenance scriptResult={scriptResult} locale={locale} />
          </div>
        </details>
      )}

      <details className="rounded-lg border border-neutral-800 bg-[#111820] px-3 py-2">
        <summary className="cursor-pointer text-xs font-semibold text-slate-300">
          {copy.anchorDetails}
        </summary>
        <p className="mt-3 rounded-lg bg-slate-900 px-3 py-2 text-xs leading-6 text-slate-300">{span.original}</p>
        {span.translated && (
          <p className="mt-2 rounded-lg bg-slate-900/70 px-3 py-2 text-xs leading-6 text-slate-500">{span.translated}</p>
        )}
        <p className="mt-2 truncate text-[11px] text-slate-500">
          {paperTitle} · {copy.anchorLabel} {span.id}
        </p>
      </details>
    </div>
  );
}

function SandboxDirectionCard({
  candidate,
  selected,
  recommended,
  locale,
  requestedLevel,
  onSelect,
}: {
  candidate: ExperimentCandidate;
  selected: boolean;
  recommended: boolean;
  locale: Locale;
  requestedLevel: ReproductionLevel;
  onSelect: () => void;
}) {
  const candidateLevel = candidate.reproduction_level || requestedLevel;
  const whyNotExact = candidate.why_not_exact || candidate.faithfulness?.why_not_exact || "";
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full min-w-0 rounded-lg border p-3 text-left transition-colors ${
        selected
          ? "border-cyan-300 bg-cyan-300/10"
          : "border-neutral-800 bg-[#151b23] hover:border-neutral-600 hover:bg-[#19212a]"
      }`}
    >
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <h4 className="min-w-0 flex-1 text-sm font-semibold leading-5 text-slate-50">{candidate.title}</h4>
        {recommended && (
          <span className="rounded-md bg-emerald-400/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-100">
            {locale === "ko" ? "추천" : "recommended"}
          </span>
        )}
        <LevelBadge level={candidateLevel} locale={locale} />
      </div>
      <p className="mt-2 line-clamp-3 text-xs leading-5 text-slate-300">{candidate.hypothesis}</p>
      <div className="mt-3 grid gap-2 text-[11px] text-slate-400">
        <DarkMeta label={locale === "ko" ? "data" : "data"} value={candidate.dataset?.name || ""} />
        <DarkMeta label={locale === "ko" ? "metric" : "metric"} value={candidate.expected_metric} />
        <DarkMeta label={locale === "ko" ? "evidence" : "evidence"} value={candidate.paper_evidence_ids.join(", ")} />
      </div>
      {(candidate.recommendation_reason || whyNotExact) && (
        <p className="mt-3 rounded-md bg-[#090c10] px-3 py-2 text-[11px] leading-5 text-slate-400">
          {candidate.recommendation_reason || whyNotExact}
        </p>
      )}
    </button>
  );
}

function SandboxRunPane({
  copy,
  scriptResult,
  runResult,
  canRunGpu,
  loading,
  step,
  onRunGpu,
}: {
  copy: ReturnType<typeof modalCopy>;
  scriptResult: GpuScriptResult | null;
  runResult: GpuProbeRunResult | null;
  canRunGpu: boolean;
  loading: LabLoading;
  step: FlowStep;
  onRunGpu: () => void;
}) {
  if (!scriptResult) {
    return <p className="rounded-lg bg-[#151b23] px-4 py-4 text-sm leading-6 text-slate-300">{copy.noScriptYet}</p>;
  }
  return (
    <div className="space-y-4">
      <div className="grid gap-2 text-xs text-slate-300 sm:grid-cols-2">
        <DarkMeta label={copy.entrypoint} value={scriptResult.entrypoint} />
        <DarkMeta label={copy.model} value={scriptResult.model || ""} />
        <DarkMeta label={copy.trace} value={scriptResult.traceId || ""} />
        <DarkMeta label="workspace" value={scriptResult.workspaceId || scriptResult.gpuRunId} />
      </div>
      <button
        onClick={onRunGpu}
        disabled={!canRunGpu}
        className="w-full rounded-xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-slate-950 hover:bg-emerald-400 disabled:opacity-50"
      >
        {loading === "run" ? copy.runningGpu : runResult ? copy.rerunGpu : copy.runGpu}
      </button>
      {step === "running" && (
        <p className="rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-3 text-sm leading-6 text-amber-100">
          {copy.runningGpuLong}
        </p>
      )}
      {scriptResult.paperClaimComparisonPlan && (
        <div className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
            {copy.comparisonPlan}
          </p>
          <p className="mt-2 text-sm leading-6 text-slate-300">{scriptResult.paperClaimComparisonPlan}</p>
        </div>
      )}
    </div>
  );
}

function SandboxLogsPane({
  copy,
  events,
  runResult,
  step,
  locale,
}: {
  copy: ReturnType<typeof modalCopy>;
  events: SandboxActivityEvent[];
  runResult: GpuProbeRunResult | null;
  step: FlowStep;
  locale: Locale;
}) {
  const logs = runResult?.logs || [];
  return (
    <div className="space-y-4">
      <SandboxActivityFeed events={events} locale={locale} />
      <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-4 py-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
            {copy.executionLog}
          </p>
          <span className="rounded-full bg-slate-800 px-2.5 py-1 text-[10px] font-semibold text-slate-300">
            {step === "running" ? copy.runningGpu : `${logs.length} lines`}
          </span>
        </div>
        <pre className="max-h-[460px] min-h-[260px] overflow-auto px-4 py-3 text-[12px] leading-6 text-slate-200">
          {logs.length > 0 ? logs.join("\n") : copy.liveLogsWaiting}
        </pre>
      </div>
    </div>
  );
}

function SandboxGrowthPane({
  copy,
  loading,
  runResult,
  growthStatus,
  growthIdeas,
  onGrowth,
}: {
  copy: ReturnType<typeof modalCopy>;
  loading: LabLoading;
  runResult: GpuProbeRunResult | null;
  growthStatus: string;
  growthIdeas: string[];
  onGrowth: () => void;
}) {
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-4">
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
          {copy.growthTitle} · {growthStatus}
        </p>
        <p className="mt-2 text-sm leading-6 text-slate-300">{copy.growthDescription}</p>
        {runResult && (
          <button
            onClick={onGrowth}
            disabled={loading !== null}
            className="mt-3 rounded-xl border border-slate-700 px-4 py-2 text-xs font-semibold text-slate-200 hover:bg-slate-800 disabled:opacity-50"
          >
            {loading === "growth" ? copy.growthLoading : copy.generateGrowth}
          </button>
        )}
      </div>
      {growthIdeas.length > 0 && (
        <ul className="space-y-2">
          {growthIdeas.map((idea, index) => (
            <li key={`${idea}-${index}`} className="rounded-xl bg-slate-900 px-4 py-3 text-sm leading-6 text-slate-100">
              {idea}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SandboxEmptyState({
  copy,
  events,
  locale,
}: {
  copy: ReturnType<typeof modalCopy>;
  events: SandboxActivityEvent[];
  locale: Locale;
}) {
  return (
    <div className="rounded-xl border border-dashed border-slate-700 bg-slate-900/60 p-5">
      <p className="text-sm font-semibold text-slate-100">{copy.sandboxWaitingTitle}</p>
      <p className="mt-2 text-sm leading-6 text-slate-400">{copy.noScriptYet}</p>
      <div className="mt-4">
        <SandboxActivityFeed events={events} locale={locale} compact />
      </div>
    </div>
  );
}

function SandboxActivityFeed({
  events,
  locale,
  compact = false,
}: {
  events: SandboxActivityEvent[];
  locale: Locale;
  compact?: boolean;
}) {
  const title = locale === "ko" ? "작업 상태" : "Workspace Activity";
  return (
    <div className={compact ? "" : "rounded-xl border border-slate-800 bg-slate-950/70 p-3"}>
      {!compact && (
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">{title}</p>
      )}
      <ol className={compact ? "space-y-2" : "mt-3 space-y-3"}>
        {events.map((event, index) => (
          <li key={`${event.label}-${index}`} className="flex gap-3">
            <span
              className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${
                event.state === "done"
                  ? "bg-emerald-400"
                  : event.state === "active"
                    ? "bg-amber-300"
                    : event.state === "error"
                      ? "bg-rose-400"
                      : "bg-slate-600"
              }`}
            />
            <span className="min-w-0">
              <span className="block text-xs font-semibold text-slate-100">{event.label}</span>
              <span className="block text-xs leading-5 text-slate-400">{event.detail}</span>
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function ReproductionPlan({
  candidate,
  scriptResult,
  locale,
}: {
  candidate: ExperimentCandidate | null;
  scriptResult: GpuScriptResult;
  locale: Locale;
}) {
  const plan = {
    ...(candidate?.run_plan || {}),
    ...(scriptResult.reproductionPlan || {}),
  };
  const level = scriptResult.reproductionLevel || candidate?.reproduction_level || "probe";
  const implementation = candidate?.implementation || {};
  const repoUrl = stringValue(plan.repo_url) || stringValue(implementation.repo_url);
  const dataset = stringValue(plan.dataset) || stringValue(candidate?.dataset?.name) || stringValue(scriptResult.dataset?.name);
  const command = stringValue(plan.command);
  const configPath = stringValue(plan.config_path);
  const expectedArtifact = stringValue(plan.expected_artifact) || scriptResult.expectedOutputs.join(", ");
  const faithfulnessNote =
    stringValue(plan.faithfulness_note) ||
    candidate?.faithfulness?.summary ||
    candidate?.why_not_exact ||
    candidate?.faithfulness?.why_not_exact ||
    "";
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
          {locale === "ko" ? "재현 계획" : "Reproduction Plan"}
        </p>
        <LevelBadge level={level} locale={locale} />
      </div>
      <div className="mt-3 grid gap-2 text-[11px] text-slate-400 sm:grid-cols-2">
        <DarkMeta label={locale === "ko" ? "Repo" : "repo"} value={repoUrl || implementation.type || "-"} />
        <DarkMeta label={locale === "ko" ? "Dataset" : "dataset"} value={dataset} />
        <DarkMeta label={locale === "ko" ? "Config" : "config"} value={configPath} />
        <DarkMeta label={locale === "ko" ? "Artifact" : "artifact"} value={expectedArtifact} />
      </div>
      {command && (
        <p className="mt-3 rounded-lg bg-slate-950 px-3 py-2 font-mono text-[11px] leading-5 text-slate-300">
          {command}
        </p>
      )}
      {faithfulnessNote && (
        <p className="mt-3 text-xs leading-5 text-slate-400">{faithfulnessNote}</p>
      )}
    </div>
  );
}

function WorkspaceProvenance({ scriptResult, locale }: { scriptResult: GpuScriptResult; locale: Locale }) {
  const workspace = scriptResult.workspace;
  const provenance = workspace?.provenance || {};
  const repoManifests = Array.isArray(provenance.implementationRepoManifests)
    ? provenance.implementationRepoManifests
    : scriptResult.implementationRepoManifests || [];
  const inspectedRepoCount = repoManifests.filter(
    (manifest) => String(manifest?.status || "").toLowerCase() === "inspected",
  ).length;
  return (
    <div className="mt-3 grid gap-2 text-[11px] text-slate-400">
      <DarkMeta label="workspace" value={scriptResult.workspaceId || workspace?.id || scriptResult.gpuRunId} />
      <DarkMeta label={locale === "ko" ? "검사된 repo" : "inspected repos"} value={String(inspectedRepoCount)} />
      <DarkMeta label="code hash" value={scriptResult.gpuRun?.codeHash || ""} />
      <DarkMeta label={locale === "ko" ? "trace" : "trace"} value={scriptResult.traceId || ""} />
    </div>
  );
}

function WorkspaceFilesPanel({
  files,
  activeFile,
  onSelectFile,
  locale,
}: {
  files: SandboxWorkspaceFile[];
  activeFile: SandboxWorkspaceFile | null;
  onSelectFile: (path: string) => void;
  locale: Locale;
}) {
  if (!activeFile) {
    return <p className="text-sm leading-6 text-text-muted">{locale === "ko" ? "표시할 sandbox 파일이 없습니다." : "No sandbox files to show."}</p>;
  }
  return (
    <div className="min-w-0 overflow-hidden rounded-lg border border-neutral-800 bg-[#090c10] text-slate-100">
      <div className="flex min-w-0 flex-wrap gap-1 border-b border-neutral-800 bg-[#111820] p-2">
        {files.map((file) => (
          <button
            key={file.path}
            type="button"
            onClick={() => onSelectFile(file.path)}
            className={`max-w-full break-all rounded-md px-2.5 py-1.5 text-[11px] font-semibold transition-colors sm:max-w-48 ${
              file.path === activeFile.path
                ? "bg-cyan-300 text-slate-950"
                : "text-slate-300 hover:bg-[#1f2933] hover:text-white"
            }`}
            title={file.role || file.language || file.path}
          >
            {file.path}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-neutral-800 px-4 py-3">
        <div>
          <p className="text-sm font-semibold text-slate-100">{activeFile.path}</p>
          <p className="text-[11px] text-slate-400">{activeFile.role || activeFile.language || "workspace file"}</p>
        </div>
        <span className="rounded-md bg-[#1f2933] px-2.5 py-1 text-[10px] font-semibold text-slate-300">
          {activeFile.language || "text"}
        </span>
      </div>
      <pre className="max-h-[58vh] min-h-[420px] overflow-auto p-4 text-[12px] leading-6 sm:min-h-[520px] xl:min-h-[600px]">
        <code>{activeFile.content}</code>
      </pre>
    </div>
  );
}

function ArtifactReportView({ result, locale }: { result: GpuProbeRunResult; locale: Locale }) {
  const artifacts = result.artifacts || {};
  const reportHtml = artifacts.reportHtml || "";
  const sandbox = artifacts.sandbox || {};
  return (
    <div className="space-y-3">
      <div className="grid gap-2 text-[11px] text-slate-300 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
        <DarkMeta label={locale === "ko" ? "생성 주체" : "generated by"} value={String(artifacts.generatedBy || "model")} />
        <DarkMeta label={locale === "ko" ? "리포트 상태" : "report status"} value={String(artifacts.reportStatus || "")} />
        <DarkMeta label={locale === "ko" ? "스크립트 허용" : "scripts"} value={String(sandbox.scriptsAllowed ?? false)} />
        <DarkMeta label={locale === "ko" ? "외부 네트워크" : "external network"} value={String(sandbox.externalNetworkAllowed ?? false)} />
      </div>
      <div className="overflow-hidden rounded-xl border border-slate-800 bg-white shadow-lg shadow-black/20">
        <iframe
          title={locale === "ko" ? "샌드박스 HTML 결과 리포트" : "Sandbox HTML result report"}
          sandbox=""
          srcDoc={reportHtml}
          className="h-[460px] w-full bg-white"
        />
      </div>
    </div>
  );
}

function buildSandboxActivity({
  loading,
  step,
  candidateResult,
  scriptResult,
  runResult,
  locale,
}: {
  loading: LabLoading;
  step: FlowStep;
  candidateResult: ExperimentCandidatesResult | null;
  scriptResult: GpuScriptResult | null;
  runResult: GpuProbeRunResult | null;
  locale: Locale;
}): SandboxActivityEvent[] {
  const ko = locale === "ko";
  return [
    {
      label: ko ? "논문 근거 읽기" : "Read paper evidence",
      detail: ko
        ? "모델이 논문 전체 맥락과 현재 anchor 근거를 함께 봅니다."
        : "The model uses the paper context with the current anchor evidence.",
      state: loading === "candidates" ? "active" : candidateResult || scriptResult || runResult ? "done" : "waiting",
    },
    {
      label: ko ? "연구 방향 제안" : "Propose research directions",
      detail: ko
        ? "논문 안에서 실제로 해볼 만한 Probe/Exact 방향 2-3개를 고릅니다."
        : "The model chooses 2-3 Probe/Exact directions grounded in the paper.",
      state: loading === "candidates" ? "active" : candidateResult ? "done" : "waiting",
    },
    {
      label: ko ? "샌드박스 파일 작성" : "Write sandbox files",
      detail: ko
        ? "승인된 방향을 코드, config, 실행 command, manifest로 분리합니다."
        : "The approved direction becomes code, config, run command, and manifest files.",
      state: loading === "script" ? "active" : scriptResult ? "done" : "waiting",
    },
    {
      label: ko ? "실행 계약 검증" : "Validate execution contract",
      detail: ko
        ? "서비스 검사를 통과한 코드만 Modal GPU 실행 버튼을 엽니다."
        : "Only service-checked code enables the Modal GPU run button.",
      state: loading === "script" ? "active" : scriptResult ? "done" : "waiting",
    },
    {
      label: ko ? "Modal GPU 실행" : "Run on Modal GPU",
      detail: ko
        ? "실행 중에는 이 샌드박스 안에서 상태와 로그를 계속 보여줍니다."
        : "During execution, this sandbox keeps status and logs visible.",
      state: loading === "run" || step === "running" ? "active" : runResult ? "done" : "waiting",
    },
    {
      label: ko ? "HTML 리포트 생성" : "Generate HTML report",
      detail: ko
        ? "완료되면 모델이 작성한 HTML 리포트가 Report 탭에 바로 나타납니다."
        : "When finished, the model-authored HTML report appears in the Report tab.",
      state: runResult?.artifacts?.reportHtml ? "done" : runResult ? "active" : "waiting",
    },
  ];
}

function buildWorkspaceFiles(
  scriptResult: GpuScriptResult | null,
  runResult: GpuProbeRunResult | null,
  locale: Locale,
): SandboxWorkspaceFile[] {
  if (!scriptResult) return [];
  const baseFiles = scriptResult.workspace?.files?.length
    ? scriptResult.workspace.files
    : [
        {
          path: "experiment.py",
          language: "python",
          role: "entrypoint",
          content: scriptResult.script,
        },
        {
          path: "config.json",
          language: "json",
          role: "configuration",
          content: JSON.stringify(
            {
              reproductionLevel: scriptResult.reproductionLevel,
              requestedReproductionLevel: scriptResult.requestedReproductionLevel,
              dataset: scriptResult.dataset,
              reproductionPlan: scriptResult.reproductionPlan,
              expectedOutputs: scriptResult.expectedOutputs,
              limitations: scriptResult.limitations,
            },
            null,
            2,
          ),
        },
      ];
  const resultFiles: SandboxWorkspaceFile[] = [];
  if (runResult?.artifacts?.manifest) {
    resultFiles.push({
      path: "manifest.json",
      language: "json",
      role: "run provenance",
      content: JSON.stringify(runResult.artifacts.manifest, null, 2),
    });
  }
  if (runResult?.artifacts?.metrics || runResult?.metrics) {
    resultFiles.push({
      path: "metrics.json",
      language: "json",
      role: "structured result",
      content: JSON.stringify(runResult.artifacts?.metrics || runResult.metrics, null, 2),
    });
  }
  if (runResult?.artifacts?.reportHtml) {
    resultFiles.push({
      path: "report.html",
      language: "html",
      role: locale === "ko" ? "샌드박스 리포트" : "sandbox report",
      content: runResult.artifacts.reportHtml,
    });
  }
  const merged = [...baseFiles];
  for (const file of resultFiles) {
    const index = merged.findIndex((item) => item.path === file.path);
    if (index >= 0) merged[index] = file;
    else merged.push(file);
  }
  return merged;
}

function GpuResultView({ result, locale }: { result: GpuProbeRunResult; locale: Locale }) {
  const hardware = result.hardware || {};
  const claim = result.claimComparison || {};
  const metrics = Object.entries(result.metrics || {});
  const resultLevel = result.reproductionLevel || "probe";
  const claimSummary = stringValue(claim.summary);
  const claimVerdict = stringValue(claim.verdict) || (result.passed ? "completed" : "failed");
  const generatedPassed = claim.generatedPassed;
  const statusTone = !result.passed
    ? {
        box: "border-rose-400/35 bg-rose-400/10",
        eyebrow: "text-rose-200",
        title: "text-rose-50",
        pill: "border-rose-400/35 bg-rose-400/15 text-rose-100",
        label: locale === "ko" ? "실행 실패" : "Execution Failed",
      }
    : generatedPassed === false || claimVerdict.includes("not_supported") || claimVerdict.includes("inconclusive")
      ? {
          box: "border-amber-400/35 bg-amber-400/10",
          eyebrow: "text-amber-200",
          title: "text-amber-50",
          pill: "border-amber-400/35 bg-amber-400/15 text-amber-100",
          label: locale === "ko" ? "실행 완료 · 검토 필요" : "Completed · Review Needed",
        }
      : {
          box: "border-emerald-400/35 bg-emerald-400/10",
          eyebrow: "text-emerald-200",
          title: "text-emerald-50",
          pill: "border-emerald-400/35 bg-emerald-400/15 text-emerald-100",
          label: locale === "ko" ? "실행 완료" : "Execution Completed",
        };
  const limitations = listValue(claim.limitations).length > 0 ? listValue(claim.limitations) : result.limitations;
  const topRows = result.rows.slice(0, 4);
  return (
    <div className="space-y-4">
      <div className={`rounded-xl border px-4 py-4 ${statusTone.box}`}>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className={`text-[11px] font-semibold uppercase tracking-[0.14em] ${statusTone.eyebrow}`}>
              {statusTone.label}
            </p>
            <p className={`mt-1 text-sm font-semibold ${statusTone.title}`}>
              {result.provider} · {String(hardware.gpuName || "GPU")} · {formatDuration(result.durationMs)}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <LevelBadge level={resultLevel} locale={locale} />
            <span className={`rounded-full border px-3 py-1 text-[11px] font-semibold ${statusTone.pill}`}>
              {claimVerdict.replaceAll("_", " ")}
            </span>
          </div>
        </div>
      </div>

      <div className="grid gap-2 text-[11px] text-slate-300 sm:grid-cols-2">
        <DarkMeta label="provider" value={result.provider} />
        <DarkMeta label="runner" value={publicRunnerLabel(result.runner, locale)} />
        <DarkMeta label={locale === "ko" ? "레벨" : "level"} value={levelCopy(resultLevel, locale).label} />
        <DarkMeta label="GPU" value={String(hardware.gpuName || hardware.cudaAvailable || "")} />
        <DarkMeta label={locale === "ko" ? "시간" : "duration"} value={formatDuration(result.durationMs)} />
      </div>

      {result.reasons.length > 0 && (
        <div className="rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-3 text-xs leading-6 text-amber-100">
          {result.reasons.join("; ")}
        </div>
      )}

      {metrics.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
            {locale === "ko" ? "측정값" : "Metrics"}
          </p>
          <div className="grid gap-2 sm:grid-cols-3">
            {metrics.map(([key, value]) => (
              <MetricTile key={key} label={metricLabel(key)} value={formatMetricValue(value, locale)} />
            ))}
          </div>
        </div>
      )}

      {Object.keys(claim).length > 0 && (
        <div className="rounded-xl border border-cyan-400/25 bg-cyan-400/10 px-4 py-3 text-sm leading-6 text-cyan-50">
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-cyan-200">
            {locale === "ko" ? "논문 주장 비교" : "Claim Comparison"}
          </p>
          <p className="mt-2 font-medium">{claimSummary || claimVerdict.replaceAll("_", " ")}</p>
          {limitations.length > 0 && (
            <ul className="mt-3 space-y-1 text-xs leading-5 text-cyan-100/80">
              {limitations.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {topRows.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
            {locale === "ko" ? "결과 행" : "Result Rows"}
          </p>
          <div className="overflow-hidden rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs">
              <tbody>
                {topRows.map((row, index) => (
                  <tr key={`${index}-${rowLabel(row, index)}`} className="border-t border-slate-800 first:border-t-0">
                    <th className="w-36 bg-slate-900 px-3 py-2 font-semibold text-slate-100">
                      {rowLabel(row, index)}
                    </th>
                    <td className="bg-slate-950 px-3 py-2 font-mono text-slate-300">
                      {formatRowValue(row, locale)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {result.logs.length > 0 && (
        <details className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950 text-slate-100">
          <summary className="cursor-pointer px-4 py-3 text-xs font-semibold text-slate-200">
            {locale === "ko" ? "실행 로그" : "Execution Log"}
          </summary>
          <pre className="max-h-36 overflow-auto border-t border-slate-800 px-4 py-3 text-[11px] leading-6">
            {result.logs.slice(-12).join("\n")}
          </pre>
        </details>
      )}
    </div>
  );
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-xl border border-slate-800 bg-slate-900 px-3 py-3">
      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">{label}</p>
      <p className="mt-1 break-words font-mono text-sm font-semibold text-slate-100">{value}</p>
    </div>
  );
}

function LevelBadge({ level, locale }: { level: string; locale: Locale }) {
  const meta = levelCopy(level, locale);
  const tone =
    level === "exact"
      ? "border-emerald-400/35 bg-emerald-400/15 text-emerald-100"
      : "border-slate-600 bg-slate-800 text-slate-200";
  return (
    <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${tone}`}>
      {meta.label}
    </span>
  );
}

function publicRunnerLabel(runner: string, locale: Locale) {
  const normalized = runner.toLowerCase();
  if (normalized.includes("paperlens-modal") || normalized.includes("modal")) {
    return locale === "ko" ? "Modal GPU 실행" : "Modal GPU";
  }
  return runner;
}

function DarkMeta({ label, value }: { label: string; value: string }) {
  return (
    <p className="min-w-0 rounded-lg bg-slate-900 px-3 py-2">
      <span className="font-semibold text-slate-100">{label}</span>{" "}
      <span className="break-words font-mono text-slate-400">{value || "-"}</span>
    </p>
  );
}

function stringValue(value: unknown) {
  return typeof value === "string" ? value : "";
}

function listValue(value: unknown) {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.length > 0) : [];
}

function metricLabel(key: string) {
  return key.replaceAll("_", " ");
}

function formatMetricValue(value: unknown, locale: Locale): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return new Intl.NumberFormat(locale === "ko" ? "ko-KR" : "en-US", {
      maximumFractionDigits: value >= 100 ? 0 : 3,
    }).format(value);
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const parts = Object.entries(value as Record<string, unknown>)
      .filter(([, item]) => item !== undefined && item !== null && item !== "")
      .slice(0, 4)
      .map(([key, item]) => `${metricLabel(key)} ${formatRowCell(item, locale)}`);
    return parts.length > 0 ? parts.join(" · ") : "-";
  }
  if (Array.isArray(value)) {
    return value.map((item) => formatRowCell(item, locale)).join(", ");
  }
  return String(value ?? "-");
}

function formatDuration(ms: number) {
  if (!Number.isFinite(ms) || ms <= 0) return "-";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatRowValue(row: Record<string, unknown>, locale: Locale) {
  if (typeof row.throughput === "number") return `${formatMetricValue(row.throughput, locale)} tokens/sec`;
  if (typeof row.value === "number") return formatMetricValue(row.value, locale);
  const priorityKeys = ["epoch", "acc", "accuracy", "loss", "train_loss", "val_loss", "score", "bleu", "split"];
  const parts = priorityKeys
    .filter((key) => row[key] !== undefined && row[key] !== null && row[key] !== "")
    .map((key) => `${metricLabel(key)} ${formatRowCell(row[key], locale)}`);
  if (parts.length > 0) return parts.join(" · ");
  const fallbackParts = Object.entries(row)
    .filter(([key]) => !["model", "metric", "type", "name", "label"].includes(key))
    .slice(0, 4)
    .map(([key, value]) => `${metricLabel(key)} ${formatRowCell(value, locale)}`);
  return fallbackParts.length > 0 ? fallbackParts.join(" · ") : "-";
}

function rowLabel(row: Record<string, unknown>, index: number) {
  return String(row.model || row.type || row.name || row.label || row.metric || `row ${index + 1}`);
}

function formatRowCell(value: unknown, locale: Locale): string {
  if (typeof value === "number") return formatMetricValue(value, locale);
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "true" : "false";
  return JSON.stringify(value);
}

function summarizeGpuRunForGrowth(result: GpuProbeRunResult, candidate: ExperimentCandidate): string {
  return `run:r1 actual Modal GPU replication probe\n${JSON.stringify(
    {
      evidence_id: "run:r1",
      candidate_id: candidate.id,
      candidate_title: candidate.title,
      candidate_evidence: candidate.paper_evidence_ids,
      reproduction_level: result.reproductionLevel || candidate.reproduction_level || "probe",
      requested_reproduction_level: result.requestedReproductionLevel || "",
      provider: result.provider,
      execution_mode: result.executionMode,
      runner: result.runner,
      gpu_requested: result.gpuRequested,
      hardware: result.hardware,
      metrics: result.metrics,
      claim_comparison: result.claimComparison,
      limitations: result.limitations,
      passed: result.passed,
      reasons: result.reasons,
    },
    null,
    2,
  )}`;
}

function levelCopy(level: string, locale: Locale) {
  const normalized: "probe" | "exact" = level === "exact" ? "exact" : "probe";
  const copy: Record<"probe" | "exact", { label: string; description: string }> =
    locale === "ko"
      ? {
          probe: {
            label: "Probe",
            description: "빠르지만 실제 코드, 실제 데이터 경로, 실제 GPU로 논문 주장의 가능성을 확인합니다.",
          },
          exact: {
            label: "Exact",
            description: "논문에 나온 repo, config, dataset 경로가 확인될 때만 정확 재현으로 진행합니다.",
          },
        }
      : {
          probe: {
            label: "Probe",
            description: "A fast but real experiment using actual code, data paths, and GPU execution.",
          },
          exact: {
            label: "Exact",
            description: "Use only when the paper repo, config, and dataset path can be verified.",
          },
        };
  return copy[normalized];
}

function stepLabel(step: FlowStep, locale: Locale) {
  const labels: Record<FlowStep, { en: string; ko: string }> = {
    idle: { en: "Ask", ko: "질문" },
    candidates: { en: "Candidates", ko: "후보" },
    script: { en: "Script", ko: "스크립트" },
    running: { en: "Running", ko: "실행" },
    result: { en: "Result", ko: "결과" },
  };
  return labels[step][locale];
}

function modalCopy(locale: Locale) {
  if (locale === "ko") {
    return {
      title: "Paper Research Sandbox",
      subtitle: "논문 전체에서 실험 방향을 먼저 찾고, Probe 또는 Exact로 승인한 실험만 하나의 샌드박스에서 실행합니다.",
      researchKicker: "Paper Research Mode",
      researchTitle: "이 논문에서 실제로 해볼 연구 방향을 먼저 찾습니다.",
      researchDescription:
        "모델이 논문 맥락과 현재 anchor 근거를 함께 보고 2-3개의 실험 방향을 제안합니다. 질문을 바꾸면 그 방향으로 다시 논문 안 근거를 확인합니다.",
      anchorLabel: "anchor",
      anchorDetails: "현재 reader 위치 확인",
      selectedEvidence: "Anchor Evidence",
      translation: "한국어 번역",
      askTitle: "연구 질문",
      reproductionLevel: "실험 모드",
      askLabel: "연구 방향을 좁히거나 새로 요청하기",
      defaultQuestion: "이 논문에서 실제로 해볼 만한 연구 실험 방향 2-3개를 찾아줘.",
      userDriven: "user-driven",
      readyStatus: "모델이 논문에서 연구 방향을 찾을 준비가 되었습니다.",
      generatingCandidates: "모델이 논문 근거에서 연구 방향을 찾고 있습니다.",
      candidatesReady: "연구 방향이 준비되었습니다. 하나를 골라 샌드박스로 여세요.",
      candidatesFailed: "연구 방향 생성 실패",
      savedQuestion: "저장된 질문:",
      generating: "방향 찾는 중",
      generateCandidates: "연구 방향 찾기",
      candidatesTitle: "모델이 제안한 연구 방향",
      directionHint: "방향을 고르면 모델이 샌드박스 파일과 실행 계약을 만듭니다.",
      openSandbox: "샌드박스 열기",
      approveAndGenerate: "선택 방향을 샌드박스로 열기",
      generatingScript: "승인된 방향에서 샌드박스 파일을 작성 중입니다.",
      scriptFailed: "GPU script 생성 실패",
      scriptReady: "샌드박스 파일이 준비되었습니다.",
      scriptTitle: "연구 계획과 GPU script",
      scriptReadyBadge: "ready",
      waiting: "waiting",
      generatedScript: "생성된 GPU script",
      viewGeneratedScript: "생성된 GPU script 보기",
      noScriptYet: "연구 방향을 승인하면 모델이 이 샌드박스 안에 코드, config, 실행 command, manifest를 만듭니다.",
      sandboxKicker: "Isolated Code Sandbox",
      sandboxTitle: "Paper experiment workspace",
      sandboxFiles: "Files",
      sandboxRun: "Run",
      sandboxLogs: "Logs",
      sandboxReport: "Report",
      sandboxResult: "Result",
      sandboxGrowth: "Growth",
      sandboxPlan: "Model plan",
      sandboxWaitingTitle: "샌드박스가 아직 비어 있습니다.",
      comparisonPlan: "비교 계획",
      executionLog: "실행 로그",
      liveLogsWaiting: "실행을 시작하면 Modal GPU 로그가 여기에 표시됩니다.",
      hardware: "하드웨어",
      entrypoint: "엔트리포인트",
      model: "모델",
      trace: "trace",
      runGpu: "Modal GPU 실행",
      rerunGpu: "Modal GPU 다시 실행",
      runningGpu: "Modal GPU 실행 중",
      runningGpuLong: "Modal GPU 컨테이너에서 실행 결과를 기다리는 중입니다.",
      gpuPassed: "Modal GPU 실행 완료",
      gpuFailed: "Modal GPU 실행 실패",
      resultTitle: "실행 결과",
      growthTitle: "Growth 연결",
      growthWaiting: "대기",
      growthReady: "실행 결과 준비됨",
      growthDescription: "GPU 결과 trace를 기반으로 다음 실험이나 학습 에피소드 추천을 붙일 수 있습니다.",
      generateGrowth: "Growth 아이디어 생성",
      growthLoading: "Growth 정리 중",
      growthDone: "Growth 아이디어 준비됨",
      growthNeedsReview: "Growth 확인 필요",
      growthFailed: "Growth 생성 실패",
      savedLevelWork: "이 레벨에 진행 중이거나 완료된 실험이 있습니다.",
      footer: "제품 증거는 이 UI에서 시작된 연구 방향, Probe/Exact 선택, 샌드박스 생성, Modal GPU 실행만 인정합니다.",
      close: "닫기",
    };
  }
  return {
    title: "Paper Research Sandbox",
    subtitle: "Find paper-level research directions first, then run only approved Probe or Exact experiments inside one sandbox.",
    researchKicker: "Paper Research Mode",
    researchTitle: "Start by finding experiments worth trying from this paper.",
    researchDescription:
      "The model reads the paper context with the current anchor evidence and proposes 2-3 experiment directions. Edit the question to steer the search back into the paper.",
    anchorLabel: "anchor",
    anchorDetails: "Current reader anchor",
    selectedEvidence: "Anchor Evidence",
    translation: "Korean Translation",
    askTitle: "Research Question",
    reproductionLevel: "Experiment Mode",
    askLabel: "Steer or refine the research direction",
    defaultQuestion: "Find 2-3 experiments worth trying from this paper.",
    userDriven: "user-driven",
    readyStatus: "Ready to search the paper for research directions.",
    generatingCandidates: "The model is finding research directions from paper evidence.",
    candidatesReady: "Research directions are ready. Choose one to open in the sandbox.",
    candidatesFailed: "Research direction generation failed",
    savedQuestion: "Saved question:",
    generating: "Finding directions",
    generateCandidates: "Find directions",
    candidatesTitle: "Model-proposed research directions",
    directionHint: "Choose a direction and the model will prepare sandbox files plus the execution contract.",
    openSandbox: "Open sandbox",
    approveAndGenerate: "Open selected direction in sandbox",
    generatingScript: "Writing sandbox files from the approved direction.",
    scriptFailed: "GPU script generation failed",
    scriptReady: "Sandbox files are ready.",
    scriptTitle: "Research plan and GPU script",
    scriptReadyBadge: "ready",
    waiting: "waiting",
    generatedScript: "Generated GPU script",
    viewGeneratedScript: "View generated GPU script",
    noScriptYet: "Approve a research direction and the model will create code, config, run command, and manifest files here.",
    sandboxKicker: "Isolated Code Sandbox",
    sandboxTitle: "Paper experiment workspace",
    sandboxFiles: "Files",
    sandboxRun: "Run",
    sandboxLogs: "Logs",
    sandboxReport: "Report",
    sandboxResult: "Result",
    sandboxGrowth: "Growth",
    sandboxPlan: "Model plan",
    sandboxWaitingTitle: "The sandbox is empty for now.",
    comparisonPlan: "Comparison plan",
    executionLog: "Execution log",
    liveLogsWaiting: "Start the run and Modal GPU logs will appear here.",
    hardware: "hardware",
    entrypoint: "entrypoint",
    model: "model",
    trace: "trace",
    runGpu: "Run on Modal GPU",
    rerunGpu: "Run again on Modal GPU",
    runningGpu: "Running on Modal GPU",
    runningGpuLong: "Waiting for the Modal GPU container to return results.",
    gpuPassed: "Modal GPU run completed",
    gpuFailed: "Modal GPU run failed",
    resultTitle: "Run result",
    growthTitle: "Growth slot",
    growthWaiting: "waiting",
    growthReady: "run ready",
    growthDescription: "The GPU result trace can feed the next experiment or learning-episode recommendation.",
    generateGrowth: "Generate Growth ideas",
    growthLoading: "Preparing Growth",
    growthDone: "Growth ideas ready",
    growthNeedsReview: "Growth needs review",
    growthFailed: "Growth unavailable",
    savedLevelWork: "This level has saved experiment work.",
    footer: "Product proof must originate from this UI: research direction, Probe/Exact choice, sandbox files, and Modal GPU execution.",
    close: "Close",
  };
}
