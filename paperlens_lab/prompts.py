from __future__ import annotations

import json
from typing import Any


def translation_prompt(title: str, spans: list[dict[str, str]], locale: str = "ko") -> str:
    return f"""You are PaperLens Lab, helping a non-native undergraduate read an English research paper.
Translate each source span into polished Korean while preserving technical terms, variables, dataset names, numbers, citations, uncertainty, and result direction.
Return only valid JSON with this shape:
{{
  "translations": [
    {{"span_id": "P0.S1", "translation": "...", "preserved_terms": ["..."], "uncertain_phrases": []}}
  ],
  "notes": ["..."]
}}

Paper title: {title}
Target locale: {locale}
Source spans:
{json.dumps(spans, ensure_ascii=False, indent=2)}

Rules:
- For target locale ko, use natural Korean sentences, not romanized or word-by-word Korean.
- Keep established ML terms readable: leave model names, dataset names, metric names, equations, and citations as-is.
- Translate ordinary English descriptions into Korean; do not return the English source unchanged.
- Preserve negation, caveats, probability, and result direction exactly.
- If a term is genuinely ambiguous, keep the term and add a short item to `uncertain_phrases`; do not invent a confident claim.
"""


def translation_repair_prompt(
    previous_output: str,
    error_text: str,
    spans: list[dict[str, str]],
    locale: str,
) -> str:
    return f"""You are repairing a PaperLens Lab translation response that failed JSON/schema validation.
Return only valid JSON with this shape:
{{
  "translations": [
    {{"span_id": "P0.S1", "translation": "...", "preserved_terms": ["..."], "uncertain_phrases": []}}
  ],
  "notes": ["..."]
}}

Errors to fix:
{error_text}

Expected source spans:
{json.dumps(spans, ensure_ascii=False, indent=2)}

Previous output:
{previous_output}

Rules:
- Output one JSON object only. No markdown fences or commentary.
- Preserve every input span_id exactly once.
- Do not add quality warnings, review labels, or UI-facing error text.
- For locale ko, use natural Korean while preserving technical terms, numbers, citations, equations, and result direction.
"""


def qa_prompt(
    paper_title: str,
    span_id: str,
    selected_span: str,
    translated_span: str,
    question: str,
    evidence: list[dict[str, str]],
    locale: str,
) -> str:
    return f"""You are PaperLens Lab. Answer the student's question using only the selected span and supplied neighboring evidence.
Use Korean when locale is ko. Cite source IDs for every substantive claim. If the evidence is insufficient, say what is missing.
Return only valid JSON:
{{
  "answer": "...",
  "evidence": [{{"source_id": "{span_id}", "quote": "..."}}],
  "confidence": "high|medium|low",
  "needs_more_context": false,
  "unsupported_assumptions": []
}}

Paper: {paper_title}
Locale: {locale}
Selected span id: {span_id}
Selected span: {selected_span}
Available translation: {translated_span}
Question: {question}
Neighbor evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Rules:
- Avoid strong words such as proves, guarantees, SOTA, 증명, 입증, 최고 unless the evidence literally says that.
- Prefer "suggests", "shows within this evidence", or Korean equivalents such as "보여준다" / "시사한다".
- If the student's question asks for a broader claim than the evidence supports, set needs_more_context to true and confidence to low or medium.
- Every evidence quote must be copied verbatim from one supplied `text` field. Do not paraphrase quotes.
- Prefer quoting the shortest exact phrase that directly supports the answer.
"""


def evidence_probe_prompt(
    paper_title: str,
    question: str,
    target_phrase: str,
    evidence: list[dict[str, str]],
    locale: str,
) -> str:
    target_phrase_json = json.dumps(target_phrase, ensure_ascii=False)
    return f"""You are PaperLens Lab running an adversarial long-context evidence check.
Answer using only the long evidence packet. The target evidence item is not named for you; find it by matching the exact phrase in the question.
Use Korean when locale is ko. Cite source IDs for every substantive claim.

Critical validation rules for this probe:
- Find the single evidence item that contains the exact phrase.
- Set the quote to exactly the exact phrase, and nothing longer, when the phrase appears in that item.
- This probe always asks whether one evidence item proves a broader full-paper conclusion or fine-tuning need.
- Because one evidence item cannot prove that broader conclusion here, set `needs_more_context` to true and `confidence` to "medium".
- Put the unsupported broader conclusion in `unsupported_assumptions`.

Return only valid JSON:
{{
  "answer": "...",
  "evidence": [{{"source_id": "P0.S1", "quote": {target_phrase_json}}}],
  "confidence": "medium",
  "needs_more_context": true,
  "unsupported_assumptions": ["full-paper superiority or fine-tuning need is not proven by one evidence item"]
}}

Paper: {paper_title}
Locale: {locale}
Question: {question}
Exact phrase to locate: {target_phrase}
Long evidence packet:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

The packet intentionally contains front and end distractors. Do not cite a nearby item unless it contains the exact phrase or directly supports the answer.
"""


def experiment_prompt(
    paper_title: str,
    selected_span: str,
    translated_span: str,
    source_evidence: list[dict[str, str]],
    idea: str,
    locale: str,
    implementation_links: list[dict[str, str]] | None = None,
) -> str:
    implementation_links = implementation_links or []
    return f"""You are PaperLens Lab. Convert a highlighted paper span into a 30-60 minute undergraduate mini-lab.
Keep it faithful, runnable, small, and honest about limitations. Do not claim full paper reproduction.
Return only valid JSON:
{{
  "research_question": "...",
  "mini_lab_goal": "...",
  "dataset": {{"name": "...", "source": "..."}},
  "baseline": "...",
  "metric": "...",
  "steps": ["..."],
  "ablation": "...",
  "failure_condition": "...",
  "expected_result": "...",
  "faithfulness_notes": ["..."],
  "implementation_repositories": [],
  "starter_code_plan": ["..."],
  "support_span_ids": ["..."]
}}

Paper: {paper_title}
Locale: {locale}
Student idea: {idea}
Selected span: {selected_span}
Available translation: {translated_span}
Evidence:
{json.dumps(source_evidence, ensure_ascii=False, indent=2)}
Implementation links found in the paper source:
{json.dumps(implementation_links, ensure_ascii=False, indent=2)}

Rules:
- The first dataset must be the actual PaperLens evidence window around the selected span, passed to starter.py as `evidence_rows`.
- If Implementation links are present, include only those source-listed repositories in `implementation_repositories`, normalized to HTTPS repo root, and treat them as an optional inspected implementation path. Do not invent repositories.
- If Implementation links are absent, set `implementation_repositories` to [] and make the mini-lab a transparent source-bound reproduction/probe.
- Do not invent an unrelated demo dataset. If an optional public benchmark is useful, list it only as a later upgrade after the source-bound evidence run.
- Do not invent synthetic sequences, simulated examples, pseudo datasets, or generated toy inputs. The mini-lab must operate on indexed paper evidence rows around the selected span.
- Starter code should be dependency-light and executable against indexed paper evidence rows before any optional library-specific upgrade.
- The ablation must start with "Remove only" or "Disable only" and keep everything else fixed.
- The failure_condition must explicitly name the metric and say what metric outcome would falsify the mini-lab.
- The expected_result must be modest; do not promise that the paper's original delta will reproduce.
- The ablation should isolate one variable.
- Return one JSON object only. Do not add markdown fences, headings, or commentary before/after the JSON.
"""


def experiment_candidates_prompt(
    paper_title: str,
    selected_span: str,
    translated_span: str,
    source_evidence: list[dict[str, str]],
    question: str,
    reproduction_level: str,
    locale: str,
    implementation_links: list[dict[str, str]] | None = None,
) -> str:
    implementation_links = implementation_links or []
    return f"""You are PaperLens Lab planning real service research directions from a paper.
The user is asking inside the product UI. Read the supplied paper context and current anchor evidence, then propose choices they can approve.
Return only valid JSON:
{{
  "candidates": [
    {{
      "id": "gpu-replication-probe",
      "title": "...",
      "kind": "gpu_replication_probe",
      "reproduction_level": "probe|exact",
      "faithfulness": {{"level": "probe|exact", "summary": "...", "why_not_exact": "", "paper_targets": ["..."], "resource_note": "..."}},
      "is_recommended": true,
      "recommendation_reason": "...",
      "hypothesis": "...",
      "paper_evidence_ids": ["..."],
      "paper_evidence_quotes": ["..."],
      "dataset": {{"name": "...", "source": "...", "requires_download": true}},
      "implementation": {{"type": "paper_repo|public_dataset|source_bound_probe", "repo_url": "", "reason": "..."}},
      "run_plan": {{"repo_url": "", "config_path": "", "command": "", "dataset": "", "expected_artifact": ""}},
      "why_not_exact": "",
      "gpu_required": true,
      "estimated_runtime_minutes": 10,
      "expected_metric": "...",
      "limitations": ["..."],
      "approval_question": "..."
    }}
  ],
  "recommended_candidate_id": "gpu-replication-probe"
}}

Paper: {paper_title}
Locale: {locale}
Requested reproduction level: {reproduction_level}
User question: {question}
Current anchor evidence:
{selected_span}
Available translation:
{translated_span}
Allowed paper evidence:
{json.dumps(source_evidence, ensure_ascii=False, indent=2)}
Implementation links found in the paper source:
{json.dumps(implementation_links, ensure_ascii=False, indent=2)}

Rules:
- Return 2 or 3 research directions.
- Treat the requested reproduction level as the user's target, but classify every candidate honestly as `exact` or `probe` only.
- `exact` means source-listed paper implementation/config/data path. Use `exact` only when the needed repo URL appears in Implementation links, the candidate `implementation.type` is `paper_repo`, and `run_plan.repo_url`, `config_path`, `command`, and `dataset` are all filled from paper/source-listed implementation evidence.
- If the user requests `exact` and no source-listed implementation path is available, return probe candidates with `why_not_exact` explaining the missing repo/config/data path. Do not label them exact.
- `probe` means a bounded but real experiment that uses actual code/data paths where possible and helps understand the paper claim without claiming exact reproduction.
- At least one candidate should be a GPU replication probe when a small public dataset or implementation path can test the selected claim directionally.
- Mark exactly one candidate as recommended and set `recommended_candidate_id` to that id.
- Every candidate must cite only `source_id` values from Allowed paper evidence in `paper_evidence_ids`.
- Include short source quotes copied from Allowed paper evidence in `paper_evidence_quotes`.
- Prefer source-listed GitHub repos only when they appear in Implementation links; never invent repo URLs.
- For `probe` candidates, leave `implementation.repo_url` and `run_plan.repo_url` empty unless the repo appears in Implementation links. Public libraries or framework repos should be described in `dataset.source`, `run_plan.command`, or `implementation.reason`, not as a paper repo.
- If no repo is listed, use a transparent public-dataset or source-bound replication probe and say that it is not an exact reproduction.
- Do not propose toy, fake, random-vector, synthetic, simulated, placeholder, or template-only experiments unless the selected paper explicitly studies synthetic data; if so, state that limitation.
- Keep each candidate runnable for a hackathon demo. Avoid multi-day training, multi-GPU clusters, proprietary datasets, and full benchmark reproduction.
- The recommended candidate should be the best balance of paper faithfulness, visible result, runtime, and GPU usefulness.
- If the user's question is not supported by the supplied paper evidence, do not invent a direction. Return only the nearest supported paper-grounded alternatives.
- Return one JSON object only. Do not add markdown fences, headings, or commentary before/after the JSON.
"""


def experiment_candidates_repair_prompt(
    previous_output: str,
    error_text: str,
    source_evidence: list[dict[str, str]],
    reproduction_level: str,
    locale: str,
    implementation_links: list[dict[str, str]] | None = None,
) -> str:
    implementation_links = implementation_links or []
    return f"""Repair PaperLens Lab research-direction JSON so it passes the service contract.
Return only valid JSON with this shape:
{{
  "candidates": [
    {{
      "id": "...",
      "title": "...",
      "kind": "gpu_replication_probe",
      "reproduction_level": "probe|exact",
      "faithfulness": {{"level": "probe|exact", "summary": "...", "why_not_exact": "", "paper_targets": ["..."], "resource_note": "..."}},
      "is_recommended": true,
      "recommendation_reason": "...",
      "hypothesis": "...",
      "paper_evidence_ids": ["..."],
      "paper_evidence_quotes": ["..."],
      "dataset": {{"name": "...", "source": "...", "requires_download": true}},
      "implementation": {{"type": "paper_repo|public_dataset|source_bound_probe", "repo_url": "", "reason": "..."}},
      "run_plan": {{"repo_url": "", "config_path": "", "command": "", "dataset": "", "expected_artifact": ""}},
      "why_not_exact": "",
      "gpu_required": true,
      "estimated_runtime_minutes": 10,
      "expected_metric": "...",
      "limitations": ["..."],
      "approval_question": "..."
    }}
  ],
  "recommended_candidate_id": "..."
}}

Locale: {locale}
Requested reproduction level: {reproduction_level}
Errors to fix:
{error_text}

Allowed paper evidence:
{json.dumps(source_evidence, ensure_ascii=False, indent=2)}
Implementation links found in the paper source:
{json.dumps(implementation_links, ensure_ascii=False, indent=2)}

Previous output:
{previous_output}

Rules:
- Return 2 or 3 candidates and exactly one recommendation.
- Cite only allowed `source_id` values.
- Never invent GitHub/repo URLs.
- `exact` is allowed only with an Implementation links repo and filled repo/config/command/dataset.
- For `probe`, leave `implementation.repo_url` and `run_plan.repo_url` empty unless the repo appears in Implementation links.
- Do not use toy, fake, random, synthetic, simulated, placeholder, or template-only experiments.
- Return one JSON object only. No markdown fences, headings, or commentary.
"""


def gpu_script_prompt(
    paper_title: str,
    selected_span: str,
    source_evidence: list[dict[str, str]],
    candidate: dict[str, Any],
    reproduction_level: str,
    locale: str,
    implementation_repo_manifests: list[dict[str, Any]] | None = None,
) -> str:
    implementation_repo_manifests = implementation_repo_manifests or []
    return f"""You are PaperLens Lab generating an approved Paper Research Sandbox GPU script.
The script will be shown to the user in Lab Modal and then executed in a Modal GPU container.
Return only valid JSON:
{{
  "script": "...",
  "entrypoint": "run_paperlens_gpu_probe",
  "dependencies": ["torch", "torchvision", "numpy"],
  "hardware": "T4",
  "dataset": {{"name": "...", "source": "..."}},
  "reproduction_level": "probe|exact",
  "reproduction_plan": {{"level": "probe|exact", "repo_url": "", "config_path": "", "command": "", "dataset": "", "expected_artifact": "", "faithfulness_note": ""}},
  "expected_outputs": ["..."],
  "paper_claim_comparison_plan": "...",
  "limitations": ["..."]
}}

Paper: {paper_title}
Locale: {locale}
Approved reproduction level: {reproduction_level}
Current anchor evidence:
{selected_span}
Approved candidate:
{json.dumps(candidate, ensure_ascii=False, indent=2)}
Allowed paper evidence:
{json.dumps(source_evidence, ensure_ascii=False, indent=2)}
Implementation repo inspection:
{json.dumps(implementation_repo_manifests, ensure_ascii=False, indent=2)}

Script contract:
- The Python script must define `def run_paperlens_gpu_probe(config=None):`.
- The function must return a JSON-serializable dict with keys: `passed`, `metrics`, `rows`, `logs`, `hardware`, `dataset`, `limitations`, `claim_comparison`, and `artifacts`.
- Keep the returned JSON and Python script compact. Prefer one readable file under about 180 lines, short comments, and simple helper functions. Do not include long prose blocks outside `reportHtml`.
- Prefer the simplest real public dataset path that tests the approved direction. For image probes, use a bounded `torchvision.datasets.CIFAR10` subset under `/tmp` when it is faithful to the candidate.
- Use PyTorch and GPU when CUDA is available. Record `torch.cuda.is_available()` and the GPU name in `hardware`.
- Keep runtime bounded: default to a small subset, at most 2 epochs or an equivalent short probe.
- Preserve the approved reproduction level in `reproduction_level` and `reproduction_plan.level`.
- If the approved level is `exact`, use only a source-listed inspected implementation path from Implementation repo inspection and fill `reproduction_plan.repo_url`, `config_path`, `dataset`, and `command`; otherwise return a script that fails clearly instead of pretending a public-dataset probe is exact.
- If the approved level is `probe`, leave `reproduction_plan.repo_url` empty unless it is a source-listed inspected paper implementation repo. Put public libraries such as torchvision in `dataset`, `command`, or `faithfulness_note`, not in `repo_url`.
- If the approved level is `probe`, label the comparison as a bounded real probe, not exact reproduction.
- Use a real public dataset or the source-listed implementation path when available. If using a public dataset as a probe, label it as a directional probe rather than a full reproduction.
- Do not use mock classes, random tensors, random labels, random-vector, fake, placeholder, toy, simulated, or synthetic data.
- Do not create a Dataset from `torch.randint`, `torch.randn`, `np.random`, or hardcoded made-up examples. If a download fails, return a failed result with logs instead of substituting generated data.
- For translation claims, prefer a real public translation dataset through the `datasets` library and report BLEU or loss on a bounded subset.
- If using Multi30k, call `load_dataset("bentrevett/multi30k", split="train[:N]")`; do not call `load_dataset("multi30k", ...)`.
- `bentrevett/multi30k` rows have `en` and `de` fields. Do not read a `translation` field.
- For Multi30k text probes, do not use `DataLoader` or custom `Dataset`. Build one fixed-shape tensor directly by truncating/padding every row to exactly `seq_len`.
- If benchmarking Transformer/LSTM throughput, create one embedding layer and pass the same embedded float tensor with shape `[batch, seq_len, d_model]` into both model families. Do not pass raw token id tensors directly to `TransformerEncoder`.
- If using `Counter(...).most_common(N)`, iterate directly over the returned `(item, count)` pairs. Do not call `.items()` on the list returned by `most_common`.
- Returned `hardware`, `dataset`, `metrics`, `rows`, `logs`, `limitations`, and `claim_comparison` values must be JSON-serializable; convert devices, tensors, numpy scalars, and paths to strings/floats/ints/lists/dicts.
- The `script` value must be plain syntactically valid Python source inside the JSON string. Do not wrap it in markdown fences. Avoid nested triple-quoted f-strings; when building `reportHtml`, prefer short escaped string fragments in a list and `"".join(parts)` so quotes cannot break the Python parser.
- Return `artifacts` as a JSON-serializable object with:
  - `reportHtml`: a self-contained HTML report for the user to inspect in the sandbox result panel.
  - `manifest`: structured provenance including reproduction level, dataset, metric names, and the comparison target.
  - `metrics`: the same structured metrics used by the system.
- `reportHtml` must be authored by the generated script/model, not by PaperLens post-processing. PaperLens will sanitize and display it, but will not append, synthesize, or prettify semantic/visual explanations after execution.
- `reportHtml` is the user's main sandbox artifact. It must not be a generic metric dashboard. It must explain the approved paper-specific experiment in the user's language when possible.
- Build `reportHtml` from the approved candidate, allowed paper evidence, measured metrics/rows, `claim_comparison`, dataset/provenance, and limitations. Do not invent paper results that were not in the evidence.
- The visible report must include compact sections or labels for: paper claim, paper evidence/source span, experiment setup/code path, measured metrics/result, comparison to the paper claim, and limitations/next step.
- `reportHtml` may include inline CSS, inline SVG plots, inline tables, and safe self-contained previews generated by the script. When numeric metrics, rows, images, or dataset examples are available, include at least one meaningful self-contained visual artifact chosen by the script/model, using inline `<svg>` or `<figure>`. It must not include scripts, iframes, forms, external URLs, remote images, or hidden network calls. Keep it useful for a user asking "what did this experiment show, how is it tied to this paper, and what should I look at next?".
- Do not let the HTML report overclaim. It must clearly say whether the run is `exact` or `probe`, compare only against the approved paper claim/evidence, and list the same limitations as the structured result.
- Do not require secrets, shell commands, notebook state, user files, or manual setup.
- Do not open arbitrary local files. Do not call subprocess, requests, sockets, external APIs, Python built-in `eval(...)`, `exec`, `compile`, `open`, `input`, or `__import__`. PyTorch lifecycle calls such as `model.eval()` are allowed.
- Use only these import roots: `torch`, `torchvision`, `numpy`, `datasets`, `sacrebleu`, `json`, `math`, `time`, `collections`, `itertools`.
- Do not import `torchtext`, `spacy`, `transformers`, or any package that is not in the allowed import roots.
- If downloading a dataset through torchvision, use a cache directory under `/tmp`.
- Keep the code understandable for a user reading it in Lab Modal.
- Return one JSON object only. Do not add markdown fences, headings, or commentary before/after the JSON.
"""


def gpu_script_repair_prompt(
    previous_output: str,
    error_text: str,
    candidate: dict[str, Any],
    source_evidence: list[dict[str, str]],
    reproduction_level: str,
    locale: str,
) -> str:
    return f"""You are repairing a PaperLens Lab GPU replication probe JSON that failed service validation.
Return only valid JSON:
{{
  "script": "...",
  "entrypoint": "run_paperlens_gpu_probe",
  "dependencies": ["torch", "numpy"],
  "hardware": "T4",
  "dataset": {{"name": "...", "source": "..."}},
  "reproduction_level": "probe|exact",
  "reproduction_plan": {{"level": "probe|exact", "repo_url": "", "config_path": "", "command": "", "dataset": "", "expected_artifact": "", "faithfulness_note": ""}},
  "expected_outputs": ["..."],
  "paper_claim_comparison_plan": "...",
  "limitations": ["..."]
}}

Locale: {locale}
Approved reproduction level: {reproduction_level}
Errors to fix:
{error_text}

Approved candidate:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

Allowed paper evidence:
{json.dumps(source_evidence, ensure_ascii=False, indent=2)}

Previous output:
{previous_output}

Rules:
- The script must define `def run_paperlens_gpu_probe(config=None):`.
- Keep the repaired JSON and Python script compact. Prefer one readable file under about 180 lines, short comments, and simple helper functions.
- Use PyTorch and record CUDA availability and GPU name in the returned `hardware`.
- Use a real public dataset or source-listed implementation context. Do not use mock classes, random tensors, random labels, fake, dummy, placeholder, toy, simulated, or synthetic data.
- Preserve the approved reproduction level in `reproduction_level` and `reproduction_plan.level`; never repair a public-dataset probe into `exact`.
- If the approved level is `exact`, `reproduction_plan.repo_url`, `config_path`, `dataset`, and `command` must all be filled from source-listed implementation evidence.
- If the approved level is `probe`, `reproduction_plan.repo_url` must be empty unless it is a source-listed inspected paper implementation repo.
- Do not create a Dataset from `torch.randint`, `torch.randn`, `np.random`, or hardcoded made-up examples. If a dataset download fails, return `passed: false` with logs and limitations.
- For translation claims, prefer a real public translation dataset through the `datasets` library and use a bounded subset.
- If using Multi30k, call `load_dataset("bentrevett/multi30k", split="train[:N]")`; do not call `load_dataset("multi30k", ...)`.
- `bentrevett/multi30k` rows have `en` and `de` fields. Do not read a `translation` field.
- For Multi30k text probes, do not use `DataLoader` or custom `Dataset`. Build one fixed-shape tensor directly by truncating/padding every row to exactly `seq_len`.
- If benchmarking Transformer/LSTM throughput, create one embedding layer and pass the same embedded float tensor with shape `[batch, seq_len, d_model]` into both model families. Do not pass raw token id tensors directly to `TransformerEncoder`.
- If using `Counter(...).most_common(N)`, iterate directly over the returned `(item, count)` pairs. Do not call `.items()` on the list returned by `most_common`.
- Returned `hardware`, `dataset`, `metrics`, `rows`, `logs`, `limitations`, `claim_comparison`, and `artifacts` values must be JSON-serializable; convert devices, tensors, numpy scalars, and paths to strings/floats/ints/lists/dicts.
- The repaired `script` value must be plain syntactically valid Python source inside the JSON string. Do not wrap it in markdown fences. Avoid nested triple-quoted f-strings; when building `reportHtml`, prefer short escaped string fragments in a list and `"".join(parts)` so quotes cannot break the Python parser.
- Return `artifacts.reportHtml` as a concise self-contained HTML report authored by the generated script/model, with no scripts, iframes, forms, external URLs, remote images, or hidden network calls. The report must not be a generic metric dashboard.
- Build `artifacts.reportHtml` from the approved candidate, allowed paper evidence, measured metrics/rows, `claim_comparison`, dataset/provenance, and limitations. It must include compact visible sections or labels for paper claim, paper evidence/source span, experiment setup/code path, measured metrics/result, comparison to the paper claim, and limitations/next step.
- When numeric metrics, rows, images, or dataset examples are available, `artifacts.reportHtml` must include at least one meaningful self-contained visual artifact chosen by the script/model, using inline `<svg>` or `<figure>`. PaperLens will sanitize and display the report, but will not append, synthesize, or prettify semantic/visual explanations after execution.
- Use only these import roots: `torch`, `torchvision`, `numpy`, `datasets`, `sacrebleu`, `json`, `math`, `time`, `collections`, `itertools`.
- Do not import `torchtext`, `spacy`, `transformers`, or any package that is not in the allowed import roots.
- Do not call Python built-in `eval(...)`, `exec`, `compile`, `open`, `input`, or `__import__`. PyTorch lifecycle calls such as `model.eval()` are allowed.
- Return one JSON object only. No markdown fences, headings, or commentary.
"""


def starter_code_prompt(
    paper_title: str,
    selected_span: str,
    source_evidence: list[dict[str, str]],
    experiment_spec: dict[str, Any],
    locale: str,
    implementation_repo_manifests: list[dict[str, Any]] | None = None,
) -> str:
    attention_mode_rule = ""
    lowered_span = selected_span.lower()
    if "attention" in lowered_span and ("recurr" in lowered_span or "convol" in lowered_span):
        attention_mode_rule = (
            "- Because this span contrasts attention against removed recurrence/convolutions, each structured example "
            "must include an explicit `mode` label such as `central`, `removed`, `global`, or `control`, and the "
            "paper-inspired scorer should behave differently across those modes.\n"
        )
    starter_shape = _starter_prompt_shape(
        selected_span,
        str(experiment_spec.get("metric") or "source-bound evidence score"),
    )
    repo_manifest_context = _starter_repo_manifest_context(implementation_repo_manifests)
    return f"""You are PaperLens Lab's code generator.
Write a dependency-free Python mini-lab starter that is genuinely grounded in the paper evidence, not a generic keyword placeholder.
Return only valid JSON:
{{
  "code": "...",
  "why_this_matches_span": "...",
  "limitations": ["..."]
}}

Paper: {paper_title}
Locale: {locale}
Selected span:
{selected_span}

Experiment spec:
{json.dumps(experiment_spec, ensure_ascii=False, indent=2)}

Grounding evidence:
{json.dumps(source_evidence, ensure_ascii=False, indent=2)}

Implementation repo inspection (read-only context):
{json.dumps(repo_manifest_context, ensure_ascii=False, indent=2)}

Passing starter shape to adapt. Keep this row logic if unsure:
```python
{starter_shape}
```

Rules:
- Output Python code only inside the `code` string.
- The code must define `baseline`, `paper_inspired`, `score`, and `run`.
- `run` must accept `evidence_rows=None`; when evidence rows are supplied, it must build rows from those paper evidence records rather than from internal examples.
- Use no imports unless absolutely necessary. If needed, only `json`, `re`, or `math` are allowed. No network, file I/O, package installs, torch, tensorflow, transformers, benchmark downloads, or training loops.
- `run()` must return dict rows built from the supplied paper evidence rows.
- Each row must include `source_id`, `text_hash`, numeric `baseline_score`, numeric `prototype_score`, string `metric`, boolean `failure_condition`, and string `failure_rule`.
- `failure_condition` is the observed row outcome: true only when the prototype fails to beat the baseline or violates the metric. Put the natural-language criterion in `failure_rule`.
- Use only supplied paper evidence rows or the selected span text; do not add unrelated internal examples.
- Do not use unrelated generic facts such as capitals, weather, presidents, or programming tutorials unless those facts are in the selected span.
- The baseline may be simple, but it must still inspect the example `context` or `query`. Do not implement the baseline as an unconditional `return candidates[0]` or another fixed first-option selector.
- Use structured examples such as `query`, `context`, `candidates`, and `gold` when the task is a mechanism or claim proxy.
- For supplied evidence rows, use `source_id`, `text_hash`, `text`, `label`, `gold`, and `query`; echo the exact `source_id` and `text_hash` in every returned row.
- Prefer zero imports. Do not import typing, dataclasses, statistics, numpy, pandas, sklearn, pathlib, os, sys, subprocess, requests, torch, tensorflow, or transformers.
- Treat implementation repo inspection as metadata only. Do not clone, install, import from, execute, read local files from, fetch network data from, or depend on repository code in the starter.
- If inspected implementation metadata is useful, mention URL/commit/README/license context only in `why_this_matches_span` or `limitations`; the generated Python code must still run only on supplied `evidence_rows` and the selected span.
- Do not print from `run()`; return rows instead.
- Make `paper_inspired` do something more specific than checking raw keyword presence alone. Use a source-bound scoring rule, alignment step, attention-style weighting, or paper-specific proxy that matches the selected span's mechanism.
- If the selected span contrasts one mechanism against removed alternatives, the code should make that contrast explicit in the examples and scoring logic.
{attention_mode_rule}- Keep the source-bound run small enough to inspect by hand.
- Keep the code runnable in under a few seconds.
- Be honest about limits: the code should test a source-bound proxy of the paper idea, not pretend to reproduce the full paper.
- Return one JSON object only. Do not add markdown fences, headings, or commentary before/after the JSON.
"""


def _starter_repo_manifest_context(manifests: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for manifest in manifests or []:
        if not isinstance(manifest, dict):
            continue
        files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
        readme = manifest.get("readme") if isinstance(manifest.get("readme"), dict) else None
        license_info = manifest.get("license") if isinstance(manifest.get("license"), dict) else None
        summarized.append(
            {
                "source_id": manifest.get("source_id", ""),
                "url": manifest.get("url", ""),
                "source_url": manifest.get("source_url", manifest.get("url", "")),
                "status": manifest.get("status", ""),
                "execution": manifest.get("execution", "none"),
                "commit": manifest.get("commit", ""),
                "default_branch": manifest.get("default_branch", ""),
                "file_count": manifest.get("file_count", 0),
                "truncated": bool(manifest.get("truncated", False)),
                "files": [
                    {
                        "path": item.get("path", ""),
                        "kind": item.get("kind", ""),
                    }
                    for item in files[:25]
                    if isinstance(item, dict)
                ],
                "readme": _starter_text_excerpt(readme),
                "license": _starter_text_excerpt(license_info),
                "error": manifest.get("error", ""),
            }
        )
    return summarized


def _starter_text_excerpt(value: dict[str, Any] | None) -> dict[str, str] | None:
    if not value:
        return None
    return {
        "path": str(value.get("path", "")),
        "excerpt": str(value.get("excerpt", ""))[:700],
    }


def experiment_repair_prompt(previous_output: str, error_text: str) -> str:
    return f"""You are repairing a PaperLens Lab experiment spec that failed contract validation.
Return only valid JSON with this shape:
{{
  "research_question": "...",
  "mini_lab_goal": "...",
  "dataset": {{"name": "...", "source": "..."}},
  "baseline": "...",
  "metric": "...",
  "steps": ["..."],
  "ablation": "...",
  "failure_condition": "...",
  "expected_result": "...",
  "faithfulness_notes": ["..."],
  "starter_code_plan": ["..."],
  "support_span_ids": ["..."]
}}

Errors to fix:
{error_text}

Previous output:
{previous_output}

Rules:
- Output one JSON object only. No markdown fences or commentary.
- Keep the plan dependency-free and suitable for a 30-60 minute undergraduate mini-lab.
- The first dataset must be the actual PaperLens evidence window around the selected span, not an unrelated invented dataset.
- Do not use legacy `fallback` dataset fields. Use `dataset.source` for the indexed paper evidence source.
- Do not invent synthetic sequences, simulated examples, pseudo datasets, random vector inputs, controlled sequences, or generated toy inputs.
- `ablation` must start with "Remove only" or "Disable only".
- `failure_condition` must explicitly repeat the metric name or the word `metric`.
- Do not add training loops, large downloads, GPUs, or full benchmark runs.
"""


def starter_code_repair_prompt(
    previous_output: str,
    error_text: str,
    selected_span: str,
    locale: str,
) -> str:
    attention_mode_rule = ""
    lowered_span = selected_span.lower()
    lowered_error = error_text.lower()
    if "attention" in lowered_span and ("recurr" in lowered_span or "convol" in lowered_span):
        attention_mode_rule = (
            "- For this attention-vs-removed-alternatives span, every example must include a `mode` field and at least "
            "one global/contrast example where the paper-inspired scorer behaves differently from the local baseline.\n"
        )
    targeted_repairs: list[str] = []
    if "trivial first-candidate selector" in lowered_error:
        targeted_repairs.append(
            "- The previous baseline was too trivial. Replace any unconditional first-candidate return with a local rule that reads `query` or `context`, loops over `candidates`, and makes a context-conditioned choice."
        )
    if "failure_condition must be a boolean" in lowered_error:
        targeted_repairs.append(
            "- In every `run()` row, set `failure_condition` to a boolean expression such as `prototype_score <= baseline_score`. Put the natural-language rule in a separate `failure_rule` string."
        )
    if "failure_condition must match prototype_score <= baseline_score" in lowered_error:
        targeted_repairs.append(
            "- In every row, compute `failure_condition` exactly as `prototype_score <= baseline_score`; do not hard-code it."
        )
    if "at least two rows" in lowered_error:
        targeted_repairs.append(
            "- Return at least two rows from `run()`: one row derived from the selected span and one contrast/control row."
        )
    if "takes 2 positional arguments but 3 were given" in lowered_error or "takes 3 positional arguments but 2 were given" in lowered_error or "takes 1 positional argument but 2 were given" in lowered_error:
        targeted_repairs.append(
            "- Keep exact callable signatures: `def baseline(example):`, `def paper_inspired(example):`, `def score(output, gold):`, and `def run():`. The `run()` function must call those helpers using exactly those argument shapes."
        )
    if "may only import" in lowered_error:
        targeted_repairs.append(
            "- Delete every import line. Use plain lists, dicts, strings, numbers, loops, and helper functions only."
        )
    if "name 'hash' is not defined" in lowered_error:
        targeted_repairs.append(
            "- Do not rely on extra helpers just to make the heuristic work. Prefer direct string comparison or small dictionary scoring over custom hashing tricks."
        )
    if "name 'json' is not defined" in lowered_error:
        targeted_repairs.append(
            "- If the code calls `json.dumps` or `json.loads`, add `import json` at the top. Otherwise remove every `json.` call."
        )
    if "never show a prototype improvement" in lowered_error:
        targeted_repairs.append(
            "- Include one selected-span row where `prototype_score > baseline_score` and `failure_condition` is false."
        )
    if "never exercise a failure or contrast case" in lowered_error:
        targeted_repairs.append(
            "- Include one contrast/control row where `prototype_score <= baseline_score` and `failure_condition` is true."
        )
    if "omits the selected span mechanism terms" in lowered_error:
        targeted_repairs.append(
            "- Copy the selected span's core mechanism words into a `SELECTED_SPAN` or `MECHANISM_TERMS` constant and use them in at least one example context plus the paper-inspired scoring rule."
        )
    if "unrelated generic examples" in lowered_error:
        targeted_repairs.append(
            "- Replace every unrelated fact example with examples derived from the selected span, its metric, its baseline, and a near-miss contrast/control case."
        )
    targeted_rules = "\n".join(targeted_repairs)
    if targeted_rules:
        targeted_rules += "\n"
    starter_shape = _starter_prompt_shape(selected_span, "source-bound evidence score")
    return f"""You are repairing a PaperLens Lab starter-code JSON response that failed runtime or grounding checks.
Return only valid JSON:
{{
  "code": "...",
  "why_this_matches_span": "...",
  "limitations": ["..."]
}}

Locale: {locale}
Selected span:
{selected_span}

Errors to fix:
{error_text}

Previous output:
{previous_output}

Known passing starter shape to adapt:
```python
{starter_shape}
```

Rules:
- Output one JSON object only. No markdown fences or commentary.
- The `code` string must be valid Python and use only the standard library.
- The code must define `baseline`, `paper_inspired`, `score`, and `run`.
- `run()` must accept `evidence_rows=None` and return dict rows with `source_id`, `text_hash`, `baseline_score`, `prototype_score`, `metric`, `failure_condition`, and `failure_rule`.
- When `evidence_rows` are supplied, build rows from those paper evidence records and echo every row's exact `source_id` and `text_hash`.
- `failure_condition` is a boolean row outcome, not a sentence. Put the sentence in `failure_rule`.
- Compute `failure_condition` from the row scores as `prototype_score <= baseline_score`.
- Include a `SELECTED_SPAN` or `MECHANISM_TERMS` constant using words from the selected span and use it in examples or scoring.
- Do not use unrelated generic facts such as capitals, weather, presidents, or programming tutorials unless those facts are in the selected span.
- The baseline may be simple, but it must still inspect the example `context` or `query`. Replace any unconditional first-candidate return with a context-conditioned rule.
- Use structured examples such as `query`, `context`, `candidates`, and `gold` when testing a paper mechanism or claim proxy.
- Prefer zero imports. If the previous code imported anything other than `json`, `re`, or `math`, delete all import lines.
- Do not print from `run()`; return rows instead.
- Do not rely on raw keyword hits alone; use a query-conditioned scoring rule, global comparison, or another small paper-grounded heuristic.
- If the selected span contrasts one mechanism against removed alternatives, include explicit contrast modes and make the scorer use them.
{attention_mode_rule}- Do not fall back to a generic placeholder.
{targeted_rules}"""


def _starter_prompt_shape(selected_span: str, metric: str) -> str:
    selected = str(selected_span or "selected paper claim").strip()
    metric_value = str(metric or "source-bound evidence score").strip()
    terms = _starter_prompt_terms(selected)
    return f'''SELECTED_SPAN = {json.dumps(selected, ensure_ascii=False)}
MECHANISM_TERMS = {json.dumps(terms or ["paper", "evidence"], ensure_ascii=False)}
METRIC = {json.dumps(metric_value, ensure_ascii=False)}

def examples_from_evidence(evidence_rows):
    if not evidence_rows:
        raise ValueError("paper evidence rows are required")
    examples = []
    for item in evidence_rows:
        examples.append({{
            "id": item.get("source_id", "paper-evidence"),
            "source_id": item.get("source_id", "paper-evidence"),
            "text_hash": item.get("text_hash", ""),
            "query": item.get("query", SELECTED_SPAN),
            "context": item.get("text", ""),
            "is_selected": bool(item.get("gold")),
            "label": item.get("label", ""),
        }})
    return examples

def baseline(example):
    query = example["query"].lower()
    context = example["context"].lower()
    return {{"score": 1.0 if query == context else 0.0}}

def paper_inspired(example):
    query = example["query"].lower()
    context = example["context"].lower()
    term_hits = sum(1 for term in MECHANISM_TERMS if term.lower() in context)
    direct_match = query in context or context in query
    score_value = max(float(direct_match), term_hits / max(1, len(MECHANISM_TERMS)))
    return {{"score": score_value}}

def score(output, gold):
    return float(output.get("score", 0.0))

def run(evidence_rows=None):
    rows = []
    for example in examples_from_evidence(evidence_rows):
        baseline_score = score(baseline(example), None)
        prototype_score = score(paper_inspired(example), None)
        rows.append({{
            "id": example["id"],
            "source_id": example["source_id"],
            "text_hash": example["text_hash"],
            "baseline_score": baseline_score,
            "prototype_score": prototype_score,
            "metric": METRIC,
            "failure_condition": prototype_score <= baseline_score,
            "failure_rule": "prototype_score <= baseline_score",
        }})
    return rows'''


def _starter_prompt_terms(text: str) -> list[str]:
    stopwords = {
        "the",
        "and",
        "that",
        "with",
        "from",
        "this",
        "into",
        "over",
        "under",
        "than",
        "small",
        "validation",
        "set",
        "says",
        "improves",
        "improve",
    }
    terms: list[str] = []
    for raw in text.replace("-", " ").replace("/", " ").split():
        cleaned = "".join(ch for ch in raw.lower() if ch.isalnum())
        if len(cleaned) < 4 or cleaned in stopwords:
            continue
        if cleaned not in terms:
            terms.append(cleaned)
        if len(terms) >= 6:
            break
    return terms


def growth_prompt(
    paper_title: str,
    paper_memory: list[dict[str, Any]],
    mini_lab_result: str,
    selected_span: str,
    locale: str,
) -> str:
    available_evidence_ids = _growth_evidence_ids(paper_memory)
    previous_growth_ids = [item for item in available_evidence_ids if item.startswith("growth_idea:")]
    return f"""You are PaperLens Lab Research Growth Mode.
Given paper memories and a mini-lab result, generate next testable research ideas for a student.
Each idea must cite paper/result evidence, state novelty angle, and propose a low-cost next step.
Return only valid JSON:
{{
  "ideas": [
    {{
      "idea": "...",
      "source_evidence": ["paper:s1", "run:r1"],
      "novelty_angle": "...",
      "testable_next_step": "...",
      "risk": "..."
    }}
  ],
  "fine_tuning_signal": "none|maybe|recommended",
  "reason": "..."
}}

Paper: {paper_title}
Locale: {locale}
Selected span: {selected_span}
Mini-lab result evidence id: run:r1
Mini-lab result:
{mini_lab_result}
Available source_evidence ids, use only these exact ids:
{json.dumps(available_evidence_ids, ensure_ascii=False, indent=2)}
Previous growth_idea ids present in memory:
{json.dumps(previous_growth_ids, ensure_ascii=False, indent=2)}
Paper memories:
{json.dumps(paper_memory, ensure_ascii=False, indent=2)}

Rules:
- Use only ids from the Available source_evidence ids list. Do not invent ids.
- Each idea must include at least one paper memory id and `run:r1` in source_evidence.
- If the Previous growth_idea ids list is empty, do not cite any `growth_idea:*` id.
- If a previous `growth_idea:*` memory is present in the Previous growth_idea ids list, at least one idea must cite one of those exact ids together with `run:r1` and a paper memory id. Use the previous idea as a stepping stone, not as a final answer.
- Do not call fine-tuning recommended just because an idea is promising; use recommended only for repeated observed model-output failures.
- Keep each next step low-cost and directly testable.
- Use "source-evidence mini-lab", "indexed evidence rows", or "source-bound probe" for the experiment surface. Do not call it a toy, toy setup, toy dataset, or toy scale.
- Do not invent synthetic sequences, simulated examples, pseudo datasets, or unrelated generated inputs. The mini-lab must operate on indexed paper evidence rows around the selected span.
"""


def growth_repair_prompt(
    previous_output: str,
    error_text: str,
    available_evidence_ids: list[str],
    locale: str,
) -> str:
    previous_growth_ids = [item for item in available_evidence_ids if item.startswith("growth_idea:")]
    return f"""You are repairing PaperLens Lab Research Growth JSON that failed evidence validation.
Return only valid JSON with this shape:
{{
  "ideas": [
    {{
      "idea": "...",
      "source_evidence": ["paper:selected-span", "run:r1"],
      "novelty_angle": "...",
      "testable_next_step": "...",
      "risk": "..."
    }}
  ],
  "fine_tuning_signal": "none|maybe|recommended",
  "reason": "..."
}}

Locale: {locale}
Errors to fix:
{error_text}

Available source_evidence ids, use only these exact ids:
{json.dumps(available_evidence_ids, ensure_ascii=False, indent=2)}
Previous growth_idea ids present in memory:
{json.dumps(previous_growth_ids, ensure_ascii=False, indent=2)}

Previous output:
{previous_output}

Rules:
- Use only ids from the Available source_evidence ids list. Do not invent ids.
- Every idea must cite `run:r1` and at least one paper memory id.
- If the Previous growth_idea ids list is empty, do not cite any `growth_idea:*` id.
- Do not invent synthetic sequences, simulated examples, pseudo datasets, or unrelated generated inputs.
- Return one JSON object only. No markdown fences or commentary.
"""


def _growth_evidence_ids(paper_memory: list[dict[str, Any]]) -> list[str]:
    ids = []
    for memory in paper_memory:
        if not isinstance(memory, dict):
            continue
        for key in ("id", "span_id", "source_id"):
            value = str(memory.get(key) or "").strip()
            if value and value not in ids:
                ids.append(value)
    if "run:r1" not in ids:
        ids.append("run:r1")
    return ids
