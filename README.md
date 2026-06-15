---
title: PaperLens Lab
emoji: "📄"
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: 3.12
app_file: app.py
base_path: /
pinned: false
license: apache-2.0
---

# PaperLens Lab

Read, verify, and prototype ideas from research papers.

PaperLens Lab is moving toward a reader-first paper workspace: read the source text, compare translation, mark important lines, ask grounded questions, and turn promising ideas into small experiments.

## Hybrid Hackathon Runtime

The active Hugging Face Space stays on the Gradio SDK for Build Small Hackathon compatibility, but the product surface is a React/Next reader. `app.py` runs a FastAPI shell that serves the exported frontend at `/` and exposes Python model endpoints under `/api/*`. The public demo path is the service reader, not a separate demo route.

- Default UI language: English.
- Optional UI language: Korean.
- Main runtime: `app.py`.
- Product frontend: `frontend/`, exported to `frontend/out` and kept deployable for the Hugging Face Space.
- Current boundary: PDF/arXiv ingestion, source extraction, on-demand Korean span translation, selected-span grounded Q&A, Paper Research Sandbox directions, model-authored experiment files, sanitized model-authored HTML reports, Research Growth memory loops, and Modal-backed GPU execution are wired through Python. Exact paper reproduction is available only when a paper-linked implementation/config/data path is verified; otherwise Lab Mode stays honest as a Probe.

## Current Preview Flow

- Shows a paper input landing screen.
- Opens a reader workspace with English source text by default.
- Switches UI language between English and Korean.
- Switches paper view between English, Korean translation, and side-by-side.
- Lets the user select lines, inspect source/translation, add marks, ask AI, and open Lab Mode.

## Backend Prototype

The Python backend keeps the original hooks for source-grounded analysis and experiment cards, and the React UI now calls it through:

- `POST /api/paper` for arXiv/text source loading.
- `POST /api/paper/upload` for PDF extraction.
- `POST /api/ask` for selected-span grounded answers.
- `POST /api/experiment/candidates` for model-proposed Probe/Exact research directions grounded in the loaded paper.
- `POST /api/experiment/gpu-script` for approved, model-authored sandbox files and execution metadata.
- `POST /api/gpu-lab/run` for Modal GPU execution of the approved sandbox script.
- Legacy `POST /api/experiment`, `POST /api/starter/run`, and `POST /api/mini-lab/run` remain for earlier source-bound mini-lab paths.

The reader-first flow remains in React: original text, Korean translation draft, side-by-side checking, marks, source inspection, AI question affordance, and Lab Mode affordance.

## Model Use

The app works in fallback mode without external credentials. For model-backed outputs, set:

```bash
HF_TOKEN=...
PAPERLENS_PROVIDER=hf
PAPERLENS_MODEL=google/gemma-4-26B-A4B-it
PAPERLENS_TRANSLATION_MODEL=google/gemma-4-26B-A4B-it
```

The default runtime model is now a current small multilingual model rather than
the older Qwen3-era default. Translation has its own `PAPERLENS_TRANSLATION_MODEL`
so Korean reading quality can move independently from general Q&A/Lab Mode.
The current default, `google/gemma-4-26B-A4B-it`, is the latest <=32B Gemma 4
candidate verified on this app's HF chat path. Gemma 4 12B, DiffusionGemma 26B
A4B, Qwen3 30B A3B, and other <=32B current multilingual models remain
configurable candidates, but a failed HF chat route should be treated as a
serving-path mismatch until Transformers, vLLM, SGLang, or Modal has also been
checked.

Useful runtime switches:

- `PAPERLENS_PROVIDER=fallback|hf|modal`
- `PAPERLENS_FORCE_MODEL=1` to make the backend use the configured provider even when the static frontend was built with model mode off
- `PAPERLENS_MODEL=google/gemma-4-26B-A4B-it`
- `PAPERLENS_TRANSLATION_MODEL=google/gemma-4-26B-A4B-it`
- `PAPERLENS_QUALITY_MODEL=...` for an optional Modal/vLLM quality model; defaults to `PAPERLENS_MODEL`
- `PAPERLENS_TRACE_PATH=outputs/agent_traces.jsonl`
- `PAPERLENS_TRACE_CONTENT=1` to opt into logging prompt/output text for evaluation or fine-tuning data; default traces store metadata only
- `PAPERLENS_MINILAB_PROVIDER=local|modal` for the Lab Mode `Run mini-lab` path; `modal` uses a bounded Modal CPU function and validates returned paper/span/source/code hashes
- `PAPERLENS_MODAL_BIN=/opt/anaconda3/bin/modal` when the Modal CLI is installed outside the active shell `PATH`
- `PAPERLENS_MODAL_MINILAB_TIMEOUT=180` to bound the local wait for a Modal mini-lab job
- `NEXT_PUBLIC_PAPERLENS_USE_MODEL=1` for the exported React frontend
- `NEXT_PUBLIC_PAPERLENS_PAPER_LOAD_TIMEOUT_MS=75000` to keep initial PDF/arXiv loading recoverable from the product UI
- `NEXT_PUBLIC_PAPERLENS_LAB_MODEL_TIMEOUT_MS=300000` to bound visible Lab Mode model waits for candidate/script generation

## Paper Research Sandbox Contract

Lab Mode is intentionally product-facing rather than a hidden backend demo:

- visible modes are only `Probe` and `Exact`;
- the model proposes paper-grounded research directions before code is generated;
- the user approves a direction before sandbox files are created;
- `experiment.py`, `config.json`, and `manifest.json` are inspectable before GPU execution;
- returned `reportHtml` is authored by the generated model script, sanitized by PaperLens, and rendered in a sandboxed iframe;
- internal validator, provider, Modal CLI, stack trace, and fallback details are not shown in the Lab Modal;
- raw Python without the required JSON envelope is repaired by the model or fails closed, not wrapped by PaperLens into a successful experiment;
- `Exact` requires source-listed implementation/config/data evidence, while public-dataset runs stay labeled as `Probe`.

## Backend Scenario Checks

Run deterministic fallback checks:

```bash
PAPERLENS_TRACE_ENABLED=0 .venv/bin/python -m paperlens_lab.scenario_runner --compact
```

Run the same translation -> Q&A -> ExperimentSpec -> Growth path through the
configured small model:

```bash
PAPERLENS_PROVIDER=hf \
PAPERLENS_MODEL=google/gemma-4-26B-A4B-it \
PAPERLENS_TRANSLATION_MODEL=google/gemma-4-26B-A4B-it \
PAPERLENS_QUALITY_MODEL=google/gemma-4-26B-A4B-it \
.venv/bin/python -m paperlens_lab.scenario_runner --use-model --compact
```

## Real-Paper Validation Snapshot

The current local validation evidence is intentionally stored as ignored runtime
artifacts under `outputs/service_demo_validation/2026-06-14/`. The hybrid app
summarizes the latest available evidence at `GET /api/validation`. The landing
page keeps that developer/judging evidence hidden by default; open `/?evidence=1`
or `/?debug=1` to show the compact validation snapshot.

Latest saved snapshot under the current Gemma 4 26B A4B default:

- Real arXiv/PDF papers: 3
- Named papers: `1706.03762`, `2005.11401`, `2106.09685`
- Real-paper evaluations: 36/36 passed in `hf_three_papers_starter_trace_v14`
- Model traces: `/api/validation` verifies the current Gemma 4 contract, summary trace binding, and zero fallback/error trace status for the referenced run
- Trace binding: 27 summary-referenced trace IDs are present in the matching trace JSONL with `status=model` and no errors
- Starter-code trace proof: all 3 saved real-paper runs include non-empty `starter_code` trace IDs with `used_fallback=false`
- Scope: parsed 15/19/26 PDF pages for `1706.03762` / `2005.11401` / `2106.09685`; reader spans are 548/1000/1000, with larger papers capped by the 1000-span validation limit
- Adversarial long-context proof: each paper includes an 8k+ character ordered evidence packet with the target evidence buried near the middle and at least 111 distractor spans; the model must cite the exact target source ID and quote without fallback
- Runnable mini-lab proof: every paper generates starter code and passes `starter_code_source_run` by parsing, importing, and running `run(evidence_rows)` over indexed paper evidence
- ExperimentSpec proof: saved paper-to-lab specs are revalidated by the current evaluator so old heavy benchmark/GPU plans cannot stay green
- Research Growth iteration proof: every paper runs a second Growth pass after the first ideas are written to memory; the second pass must cite `paper:selected-middle`, `run:r1`, and a prior `growth_idea:*` memory
- Persistent memory proof: 20 JSONL memory records across 3 papers, including `paper_span`, `mini_lab_result`, `growth_idea`, and `growth_iteration_idea`
- Local selected-span API proof: `1706.03762`, span `P5.S8`, evidence window `P5.S5-P5.S11`, quote id `P5.S8`, source hash `824e98d76dcb7231`
- Multi-span selection proof: `1706.03762`, spans `P0.S4` and `P0.S5`, returned `usedFallback=false`, support IDs `P0.S4/P0.S5`, and separate source quotes for each selected fragment
- Implementation repository proof: `2106.09685`, span `P0.S9`, extracts `https://github.com/microsoft/LoRA`, inspects commit `c4593f060e6a368d7bb5af5273b8e42810cdef90` with `execution=none`, and keeps the generated starter source-bound to indexed evidence rows
- Modal mini-lab proof: LoRA source-bound mini-lab returned `provider=modal`, `executionMode=modal-source-bound-remote-function`, `runner=paperlens-modal-minilab`, hash validation true, and `claimComparison=mixed_or_not_supported`
- Evidence consistency: saved QA citations are checked against stored source-evidence maps, adversarial quotes are validated against the long-context evidence packet, and local quote ids must stay inside the source-index window
- Fine-tuning decision from real failures: `no`; no repeated trainable failure cluster has been observed yet

Useful validation commands:

```bash
RUN_ROOT="outputs/service_demo_validation/$(date +%F)"
RUN_NAME="hf_three_papers_starter_trace_v14"
PAPERLENS_PROVIDER=hf \
PAPERLENS_MODEL=google/gemma-4-26B-A4B-it \
PAPERLENS_TRANSLATION_MODEL=google/gemma-4-26B-A4B-it \
PAPERLENS_QUALITY_MODEL=google/gemma-4-26B-A4B-it \
PAPERLENS_TRACE_PATH="$RUN_ROOT/${RUN_NAME}_traces.jsonl" \
PAPERLENS_MEMORY_PATH="$RUN_ROOT/${RUN_NAME}_memory.jsonl" \
PAPERLENS_SOURCE_INDEX_DIR="$RUN_ROOT/source_index" \
PAPERLENS_TRANSLATION_CACHE_DIR="$RUN_ROOT/translation_cache" \
.venv/bin/python -m paperlens_lab.real_paper_runner \
  --use-model \
  --paper 1706.03762 \
  --paper 2005.11401 \
  --paper 2106.09685 \
  --max-pdf-pages 64 \
  --max-reader-spans 1000 \
  --max-translate-spans 3 \
  --output-dir "$RUN_ROOT/$RUN_NAME" \
  --compact
```

```bash
curl -sS http://127.0.0.1:7860/api/validation | jq .
```

## Local Run

Frontend product surface:

```bash
cd frontend
npm install
npm run build
```

The exported `frontend/out/` directory is intentionally deployable. The
Hugging Face Space runs `app.py` as a Python app, so public Space sync must
include the latest static export unless the Space build pipeline is changed to
run the frontend build itself.

Hybrid Space app:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860/` for the React reader.

## Hackathon Entry Status

- GitHub repository: https://github.com/kcyoow/paperlens-lab
- Hugging Face Space: https://huggingface.co/spaces/build-small-hackathon/paperlens-lab
- Codex-attributed commits: this project is scaffolded and iterated with Codex as the coding agent.
- Space README repo link: this README contains the public GitHub repository link above.
- Local service status: current local code serves the React reader and `/api/*` from FastAPI. The public Space must be synced with the current repository state before public `/api/health` parity can be claimed.
- Space frontend artifact: include `frontend/out/` in the synced Space files so the Python app can serve the React reader without relying on a Node build during Space startup.

## Built With Codex

This Space was scaffolded and prepared with Codex as the coding agent for the Build Small Hackathon.

## Safety And Grounding

PaperLens Lab keeps source evidence visible and treats generated explanation as interpretation unless it is tied back to a source span. It is intended as a research reading assistant, not a substitute for reading the original paper.
