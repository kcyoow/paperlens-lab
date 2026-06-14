"use client";

import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  buildExperimentCandidates,
  buildGrowthIdeas,
  ExperimentCandidate,
  ExperimentCandidatesResult,
  generateGpuScript,
  GpuProbeRunResult,
  GpuScriptResult,
  ReproductionLevel,
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

const REPRODUCTION_LEVELS: ReproductionLevel[] = ["probe", "scaled", "exact"];

export default function LabModal({ span, locale, paperId, paperTitle, sourceText, onClose }: Props) {
  const copy = modalCopy(locale);
  const [question, setQuestion] = useState(copy.defaultQuestion);
  const [reproductionLevel, setReproductionLevel] = useState<ReproductionLevel>("scaled");
  const [levelStates, setLevelStates] = useState<Record<ReproductionLevel, LevelSlotState>>(() =>
    createLevelStates(copy),
  );
  const [loading, setLoading] = useState<LabLoading>(null);

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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4">
      <div className="flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-border bg-surface shadow-xl">
        <header className="flex shrink-0 items-start justify-between gap-4 border-b border-border px-6 py-5">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-xl font-semibold text-text-primary">{copy.title}</h2>
              <span className="rounded-full bg-surface-secondary px-2.5 py-1 text-[10px] font-semibold text-text-secondary">
                {span.id}
              </span>
              <span className="rounded-full bg-primary-50 px-2.5 py-1 text-[10px] font-semibold text-primary-700">
                {stepLabel(step, locale)}
              </span>
              <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800">
                {levelCopy(reproductionLevel, locale).label}
              </span>
            </div>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-text-muted">{copy.subtitle}</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-text-muted transition-colors hover:bg-surface-hover hover:text-text-primary"
            aria-label={copy.close}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto bg-surface-secondary/30 p-6">
          <div className="grid min-w-0 gap-5 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
            <section className="min-w-0 space-y-5">
              <Panel title={copy.selectedEvidence} badge={span.id}>
                <p className="text-sm leading-7 text-text-primary">{span.original}</p>
                {span.translated && (
                  <div className="mt-4 rounded-lg bg-surface-secondary px-4 py-3">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted">
                      {copy.translation}
                    </p>
                    <p className="mt-2 text-sm leading-7 text-text-secondary">{span.translated}</p>
                  </div>
                )}
              </Panel>

              <Panel title={copy.askTitle} badge={copy.userDriven}>
                <div className="mb-4">
                  <p className="mb-2 text-xs font-semibold text-text-secondary">{copy.reproductionLevel}</p>
                  <div className="grid grid-cols-3 overflow-hidden rounded-xl border border-border bg-surface-secondary p-1">
                    {REPRODUCTION_LEVELS.map((level) => {
                      const levelMeta = levelCopy(level, locale);
                      const hasSavedWork = hasLevelWork(levelStates[level] ?? createEmptyLevelState(copy));
                      return (
                        <button
                          key={level}
                          type="button"
                          onClick={() => handleLevelChange(level)}
                          disabled={Boolean(loading)}
                          className={`rounded-lg px-2 py-2 text-xs font-semibold transition-colors disabled:opacity-50 ${
                            reproductionLevel === level
                              ? "bg-surface text-text-primary shadow-sm"
                              : "text-text-muted hover:bg-surface-hover hover:text-text-secondary"
                          }`}
                          title={levelMeta.description}
                        >
                          <span className="inline-flex items-center justify-center gap-1">
                            {levelMeta.label}
                            {hasSavedWork && (
                              <span
                                className="inline-block h-1.5 w-1.5 rounded-full bg-primary-500"
                                aria-label={copy.savedLevelWork}
                              />
                            )}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                  <p className="mt-2 text-xs leading-5 text-text-muted">
                    {levelCopy(reproductionLevel, locale).description}
                  </p>
                </div>
                <label className="block text-xs font-semibold text-text-secondary" htmlFor="lab-question">
                  {copy.askLabel}
                </label>
                <textarea
                  id="lab-question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  className="mt-2 min-h-24 w-full resize-y rounded-xl border border-border bg-surface px-3 py-2 text-sm leading-6 text-text-primary outline-none focus:border-primary-400"
                />
                <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                  <p className="text-xs leading-5 text-text-muted">{status}</p>
                  <button
                    onClick={handleGenerateCandidates}
                    disabled={loading !== null || !question.trim()}
                    className="rounded-xl bg-primary-600 px-4 py-2 text-xs font-semibold text-white hover:bg-primary-700 disabled:opacity-50"
                  >
                    {loading === "candidates" ? copy.generating : copy.generateCandidates}
                  </button>
                </div>
              </Panel>

              {candidates.length > 0 && (
                <Panel title={copy.candidatesTitle} badge={`${candidates.length}`}>
                  {activeState.questionAtGeneration && (
                    <p className="mb-3 rounded-lg bg-surface-secondary px-3 py-2 text-xs leading-5 text-text-secondary">
                      <span className="font-semibold text-text-primary">{copy.savedQuestion}</span>{" "}
                      {activeState.questionAtGeneration}
                    </p>
                  )}
                  <div className="space-y-3">
                    {candidates.map((candidate) => (
                      <CandidateCard
                        key={candidate.id}
                        candidate={candidate}
                        selected={candidate.id === selectedCandidateId}
                        recommended={candidate.id === recommendedCandidateId || candidate.is_recommended === true}
                        locale={locale}
                        requestedLevel={reproductionLevel}
                        onSelect={() => updateActiveLevelState({ selectedCandidateId: candidate.id })}
                      />
                    ))}
                  </div>
                  <button
                    onClick={handleApproveCandidate}
                    disabled={!canGenerateScript}
                    className="mt-4 w-full rounded-xl border border-primary-200 bg-primary-50 px-4 py-2 text-sm font-semibold text-primary-700 hover:bg-primary-100 disabled:opacity-50"
                  >
                    {loading === "script" ? copy.generatingScript : copy.approveAndGenerate}
                  </button>
                </Panel>
              )}
            </section>

            <section className="min-w-0 space-y-5">
              <Panel title={copy.scriptTitle} badge={scriptResult ? levelCopy(scriptResult.reproductionLevel || selectedCandidate?.reproduction_level || reproductionLevel, locale).label : copy.waiting}>
                {scriptResult ? (
                  <div className="space-y-4">
                    <ReproductionPlan
                      candidate={scriptResult.candidate || selectedCandidate}
                      scriptResult={scriptResult}
                      locale={locale}
                    />
                    <div className="grid gap-2 text-xs text-text-secondary sm:grid-cols-2">
                      <Meta label={copy.hardware} value={String(scriptResult.hardware || "T4")} />
                      <Meta label={copy.entrypoint} value={scriptResult.entrypoint} />
                      <Meta label={copy.model} value={scriptResult.model || ""} />
                      <Meta label={copy.trace} value={scriptResult.traceId || ""} />
                    </div>
                    {scriptResult.paperClaimComparisonPlan && (
                      <p className="rounded-xl bg-surface-secondary px-4 py-3 text-xs leading-6 text-text-secondary">
                        {scriptResult.paperClaimComparisonPlan}
                      </p>
                    )}
                    <details
                      open={!runResult}
                      className="min-w-0 overflow-hidden rounded-xl border border-slate-800 bg-slate-950 text-slate-100"
                    >
                      <summary className="cursor-pointer px-4 py-3 text-xs font-semibold text-slate-200">
                        {runResult ? copy.viewGeneratedScript : copy.generatedScript}
                      </summary>
                      <pre className="max-h-[360px] w-full max-w-full overflow-auto border-t border-slate-800 p-4 text-[12px] leading-6">
                        <code>{scriptResult.script}</code>
                      </pre>
                    </details>
                    <button
                      onClick={handleRunGpu}
                      disabled={!canRunGpu}
                      className="w-full rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
                    >
                      {loading === "run" ? copy.runningGpu : runResult ? copy.rerunGpu : copy.runGpu}
                    </button>
                  </div>
                ) : (
                  <p className="text-sm leading-6 text-text-muted">{copy.noScriptYet}</p>
                )}
              </Panel>

              {(runResult || step === "running") && (
                <Panel title={copy.resultTitle} badge={runResult?.provider || "Modal"}>
                  {runResult ? (
                    <GpuResultView result={runResult} locale={locale} />
                  ) : (
                    <p className="text-sm leading-6 text-text-muted">{copy.runningGpuLong}</p>
                  )}
                </Panel>
              )}

              <Panel title={copy.growthTitle} badge={growthStatus}>
                <p className="text-sm leading-6 text-text-muted">{copy.growthDescription}</p>
                {runResult && (
                  <button
                    onClick={handleGrowth}
                    disabled={loading !== null}
                    className="mt-3 rounded-xl border border-border px-4 py-2 text-xs font-semibold text-text-secondary hover:bg-surface-hover disabled:opacity-50"
                  >
                    {loading === "growth" ? copy.growthLoading : copy.generateGrowth}
                  </button>
                )}
                {growthIdeas.length > 0 && (
                  <ul className="mt-3 space-y-2">
                    {growthIdeas.map((idea, index) => (
                      <li key={`${idea}-${index}`} className="rounded-xl bg-surface-secondary px-4 py-3 text-sm leading-6 text-text-primary">
                        {idea}
                      </li>
                    ))}
                  </ul>
                )}
              </Panel>
            </section>
          </div>
        </div>

        <footer className="flex shrink-0 items-center justify-between gap-3 border-t border-border px-6 py-3">
          <p className="text-[11px] leading-5 text-text-muted">{copy.footer}</p>
          <button
            onClick={onClose}
            className="rounded-lg border border-border px-4 py-2 text-xs font-medium text-text-secondary hover:bg-surface-hover"
          >
            {copy.close}
          </button>
        </footer>
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
    scaled: createEmptyLevelState(copy),
    exact: createEmptyLevelState(copy),
  };
}

function hasLevelWork(state: LevelSlotState) {
  return Boolean(state.candidateResult || state.scriptResult || state.runResult || state.growthIdeas.length > 0);
}

function CandidateCard({
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
  const levelMeta = levelCopy(candidateLevel, locale);
  const whyNotExact = candidate.why_not_exact || candidate.faithfulness?.why_not_exact || "";
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`min-w-0 w-full rounded-xl border p-4 text-left transition-colors ${
        selected
          ? "border-primary-300 bg-primary-50"
          : "border-border bg-surface hover:border-primary-200 hover:bg-surface-hover"
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold text-text-primary">{candidate.title}</h3>
        {recommended && (
          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-800">
            {locale === "ko" ? "추천" : "recommended"}
          </span>
        )}
        {candidate.gpu_required && (
          <span className="rounded-full bg-slate-900 px-2 py-0.5 text-[10px] font-semibold text-white">
            GPU
          </span>
        )}
        <LevelBadge level={candidateLevel} locale={locale} />
      </div>
      <p className="mt-2 text-sm leading-6 text-text-secondary">{candidate.hypothesis}</p>
      <div className="mt-3 grid gap-2 text-[11px] text-text-muted sm:grid-cols-2">
        <Meta label={locale === "ko" ? "레벨" : "level"} value={levelMeta.label} />
        <Meta label={locale === "ko" ? "데이터" : "dataset"} value={candidate.dataset?.name || ""} />
        <Meta label={locale === "ko" ? "지표" : "metric"} value={candidate.expected_metric} />
        <Meta label={locale === "ko" ? "예상 시간" : "runtime"} value={`${candidate.estimated_runtime_minutes ?? "?"} min`} />
        <Meta label={locale === "ko" ? "근거" : "evidence"} value={candidate.paper_evidence_ids.join(", ")} />
      </div>
      {(candidate.faithfulness?.summary || whyNotExact) && (
        <div className="mt-3 space-y-2">
          {candidate.faithfulness?.summary && (
            <p className="rounded-lg bg-white/70 px-3 py-2 text-[11px] leading-5 text-text-secondary">
              {candidate.faithfulness.summary}
            </p>
          )}
          {whyNotExact && (
            <p className="rounded-lg bg-amber-50 px-3 py-2 text-[11px] leading-5 text-amber-900">
              {whyNotExact}
            </p>
          )}
        </div>
      )}
      {candidate.recommendation_reason && (
        <p className="mt-3 rounded-lg bg-white/70 px-3 py-2 text-[11px] leading-5 text-text-secondary">
          {candidate.recommendation_reason}
        </p>
      )}
    </button>
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
  const level = scriptResult.reproductionLevel || candidate?.reproduction_level || "scaled";
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
    <div className="rounded-xl border border-border bg-surface-secondary px-4 py-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-text-muted">
          {locale === "ko" ? "재현 계획" : "Reproduction Plan"}
        </p>
        <LevelBadge level={level} locale={locale} />
      </div>
      <div className="mt-3 grid gap-2 text-[11px] text-text-secondary sm:grid-cols-2">
        <Meta label={locale === "ko" ? "Repo" : "repo"} value={repoUrl || implementation.type || "-"} />
        <Meta label={locale === "ko" ? "Dataset" : "dataset"} value={dataset} />
        <Meta label={locale === "ko" ? "Config" : "config"} value={configPath} />
        <Meta label={locale === "ko" ? "Artifact" : "artifact"} value={expectedArtifact} />
      </div>
      {command && (
        <p className="mt-3 rounded-lg bg-surface px-3 py-2 font-mono text-[11px] leading-5 text-text-secondary">
          {command}
        </p>
      )}
      {faithfulnessNote && (
        <p className="mt-3 text-xs leading-5 text-text-muted">{faithfulnessNote}</p>
      )}
    </div>
  );
}

function GpuResultView({ result, locale }: { result: GpuProbeRunResult; locale: Locale }) {
  const hardware = result.hardware || {};
  const claim = result.claimComparison || {};
  const metrics = Object.entries(result.metrics || {});
  const resultLevel = result.reproductionLevel || "scaled";
  const claimSummary = stringValue(claim.summary);
  const claimVerdict = stringValue(claim.verdict) || (result.passed ? "completed" : "failed");
  const limitations = listValue(claim.limitations).length > 0 ? listValue(claim.limitations) : result.limitations;
  const topRows = result.rows.slice(0, 4);
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-emerald-800">
              {locale === "ko" ? "실행 완료" : "Execution Completed"}
            </p>
            <p className="mt-1 text-sm font-semibold text-emerald-950">
              {result.provider} · {String(hardware.gpuName || "GPU")} · {formatDuration(result.durationMs)}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <LevelBadge level={resultLevel} locale={locale} />
            <span className="rounded-full bg-white px-3 py-1 text-[11px] font-semibold text-emerald-800">
              {claimVerdict.replaceAll("_", " ")}
            </span>
          </div>
        </div>
      </div>

      <div className="grid gap-2 text-[11px] text-text-secondary sm:grid-cols-2">
        <Meta label="provider" value={result.provider} />
        <Meta label="runner" value={result.runner} />
        <Meta label={locale === "ko" ? "레벨" : "level"} value={levelCopy(resultLevel, locale).label} />
        <Meta label="GPU" value={String(hardware.gpuName || hardware.cudaAvailable || "")} />
        <Meta label={locale === "ko" ? "시간" : "duration"} value={formatDuration(result.durationMs)} />
      </div>

      {result.reasons.length > 0 && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-6 text-amber-900">
          {result.reasons.join("; ")}
        </div>
      )}

      {metrics.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-text-muted">
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
        <div className="rounded-xl bg-primary-50 px-4 py-3 text-sm leading-6 text-primary-900">
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-primary-700">
            {locale === "ko" ? "논문 주장 비교" : "Claim Comparison"}
          </p>
          <p className="mt-2 font-medium">{claimSummary || claimVerdict.replaceAll("_", " ")}</p>
          {limitations.length > 0 && (
            <ul className="mt-3 space-y-1 text-xs leading-5 text-primary-800">
              {limitations.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {topRows.length > 0 && (
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-text-muted">
            {locale === "ko" ? "결과 행" : "Result Rows"}
          </p>
          <div className="overflow-hidden rounded-xl border border-border">
            <table className="w-full text-left text-xs">
              <tbody>
                {topRows.map((row, index) => (
                  <tr key={`${index}-${rowLabel(row, index)}`} className="border-t border-border first:border-t-0">
                    <th className="w-36 bg-surface-secondary px-3 py-2 font-semibold text-text-primary">
                      {rowLabel(row, index)}
                    </th>
                    <td className="px-3 py-2 font-mono text-text-secondary">
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
    <div className="min-w-0 rounded-xl bg-surface-secondary px-3 py-3">
      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-muted">{label}</p>
      <p className="mt-1 break-words font-mono text-sm font-semibold text-text-primary">{value}</p>
    </div>
  );
}

function LevelBadge({ level, locale }: { level: string; locale: Locale }) {
  const meta = levelCopy(level, locale);
  const tone =
    level === "exact"
      ? "border-emerald-200 bg-emerald-50 text-emerald-800"
      : level === "scaled"
        ? "border-primary-200 bg-primary-50 text-primary-700"
        : "border-slate-200 bg-slate-100 text-slate-700";
  return (
    <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${tone}`}>
      {meta.label}
    </span>
  );
}

function Panel({ title, badge, children }: { title: string; badge?: string; children: ReactNode }) {
  return (
    <section className="min-w-0 overflow-hidden rounded-xl border border-border bg-surface p-5 shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-text-muted">{title}</p>
        {badge && (
          <span className="rounded-full bg-surface-secondary px-2.5 py-1 text-[10px] font-semibold text-text-secondary">
            {badge}
          </span>
        )}
      </div>
      {children}
    </section>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <p className="min-w-0 rounded-lg bg-surface-secondary px-3 py-2">
      <span className="font-semibold text-text-primary">{label}</span>{" "}
      <span className="break-words font-mono">{value || "-"}</span>
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
      reproduction_level: result.reproductionLevel || candidate.reproduction_level || "scaled",
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
  const normalized: ReproductionLevel =
    level === "exact" || level === "probe" || level === "scaled" ? level : "scaled";
  const copy: Record<ReproductionLevel, { label: string; description: string }> =
    locale === "ko"
      ? {
          probe: {
            label: "Probe",
            description: "논문 주장의 방향을 빠르게 확인하는 실제 GPU 실험입니다.",
          },
          scaled: {
            label: "Scaled",
            description: "논문 방식과 실제 데이터 흐름을 유지하되 시간과 규모를 줄인 재현입니다.",
          },
          exact: {
            label: "Exact",
            description: "논문에 나온 repo, config, dataset 경로가 확인될 때만 정확 재현으로 진행합니다.",
          },
        }
      : {
          probe: {
            label: "Probe",
            description: "A real GPU experiment that checks the direction of the paper claim.",
          },
          scaled: {
            label: "Scaled",
            description: "A bounded reproduction that keeps the method and data path close to the paper.",
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
      title: "Lab Mode",
      subtitle: "선택한 논문 문장에서 Probe, Scaled, Exact 레벨을 고르고 승인한 실험만 Modal GPU로 실행합니다.",
      selectedEvidence: "선택한 논문 근거",
      translation: "한국어 번역",
      askTitle: "실험 질문",
      reproductionLevel: "재현 레벨",
      askLabel: "제품 안에서 직접 물어볼 질문",
      defaultQuestion: "어떤 실험을 해볼까? 이 부분과 관련해서 논문에 나온 실험을 직접 진행해볼까?",
      userDriven: "user-driven",
      readyStatus: "질문을 입력하고 실험 후보를 생성하세요.",
      generatingCandidates: "모델이 논문 근거에서 실험 후보를 만들고 있습니다.",
      candidatesReady: "실험 후보가 준비되었습니다. 하나를 승인하세요.",
      candidatesFailed: "실험 후보 생성 실패",
      savedQuestion: "저장된 질문:",
      generating: "후보 생성 중",
      generateCandidates: "실험 후보 생성",
      candidatesTitle: "모델 제안 실험 후보",
      approveAndGenerate: "선택 후보 승인하고 GPU script 생성",
      generatingScript: "승인된 후보에서 GPU script를 생성 중입니다.",
      scriptFailed: "GPU script 생성 실패",
      scriptReady: "GPU script가 준비되었습니다.",
      scriptTitle: "재현 계획과 GPU script",
      scriptReadyBadge: "ready",
      waiting: "waiting",
      generatedScript: "생성된 GPU script",
      viewGeneratedScript: "생성된 GPU script 보기",
      noScriptYet: "후보를 승인하면 모델이 GPU 실행 스크립트를 생성합니다.",
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
      footer: "제품 증거는 이 UI에서 시작된 레벨 선택, 후보 생성, 승인, script, Modal GPU 실행만 인정합니다.",
      close: "닫기",
    };
  }
  return {
    title: "Lab Mode",
    subtitle: "Choose a Probe, Scaled, or Exact reproduction level from the selected paper span, approve one candidate, then run it on Modal GPU.",
    selectedEvidence: "Selected Paper Evidence",
    translation: "Korean Translation",
    askTitle: "Experiment Question",
    reproductionLevel: "Reproduction Level",
    askLabel: "Ask inside the product",
    defaultQuestion: "What experiment should we run? Should we try a paper-related experiment for this selected span?",
    userDriven: "user-driven",
    readyStatus: "Ask a question and generate experiment candidates.",
    generatingCandidates: "The model is proposing experiments from paper evidence.",
    candidatesReady: "Experiment candidates are ready. Approve one to continue.",
    candidatesFailed: "Experiment candidate generation failed",
    savedQuestion: "Saved question:",
    generating: "Generating",
    generateCandidates: "Generate candidates",
    candidatesTitle: "Model-proposed candidates",
    approveAndGenerate: "Approve selected candidate and generate GPU script",
    generatingScript: "Generating a GPU script from the approved candidate.",
    scriptFailed: "GPU script generation failed",
    scriptReady: "GPU script is ready.",
    scriptTitle: "Reproduction plan and GPU script",
    scriptReadyBadge: "ready",
    waiting: "waiting",
    generatedScript: "Generated GPU script",
    viewGeneratedScript: "View generated GPU script",
    noScriptYet: "Approve a candidate and the model will generate the GPU execution script.",
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
    footer: "Product proof must originate from this UI: level choice, candidates, approval, script, and Modal GPU execution.",
    close: "Close",
  };
}
