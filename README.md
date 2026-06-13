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

The active Hugging Face Space stays on the Gradio SDK for Build Small Hackathon compatibility, but the product surface is a React/Next reader. `app.py` runs a FastAPI shell that serves the exported frontend at `/`, exposes Python model endpoints under `/api/*`, and keeps the Gradio demo available at `/gradio`.

- Default UI language: English.
- Optional UI language: Korean.
- Main runtime: `app.py`.
- Product frontend: `frontend/`, exported to `frontend/out`.
- Gradio fallback/demo: `/gradio`.
- Current boundary: PDF/arXiv ingestion, source extraction, on-demand Korean span translation, selected-span grounded Q&A, experiment cards, Research Growth memory loops, dependency-free starter smoke execution, and Modal-backed mini-lab execution are wired through Python; full-document batch translation and full notebook execution remain staged Lab Mode extensions.

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
- `POST /api/experiment` for paper-to-experiment cards and starter code.
- `POST /api/starter/run` for executing generated dependency-free starter smoke tests.
- `POST /api/mini-lab/run` for running the selected starter as a source/code-bound mini-lab job, locally or through Modal.

The reader-first flow remains in React: original text, Korean translation draft, side-by-side checking, marks, source inspection, AI question affordance, and Lab Mode affordance.

## Model Use

The app works in fallback mode without external credentials. For model-backed outputs, set:

```bash
HF_TOKEN=...
PAPERLENS_MODEL=Qwen/Qwen3-4B-Instruct-2507
PAPERLENS_PROVIDER=hf
```

The current Hugging Face inference smoke path uses `Qwen/Qwen3-4B-Instruct-2507`
because it is small, under the Build Small Hackathon's 32B limit, and currently
responds through the authenticated HF Inference path. Larger <=32B candidates
such as Mistral Small, Magistral Small, Granite, or Gemma 4 remain Modal/vLLM
quality-booster candidates.

Useful runtime switches:

- `PAPERLENS_PROVIDER=fallback|hf|modal`
- `PAPERLENS_FORCE_MODEL=1` to make the backend use the configured provider even when the static frontend was built with model mode off
- `PAPERLENS_MODEL=Qwen/Qwen3-4B-Instruct-2507`
- `PAPERLENS_QUALITY_MODEL=...` for an optional Modal/vLLM quality model; defaults to `PAPERLENS_MODEL`
- `PAPERLENS_TRACE_PATH=outputs/agent_traces.jsonl`
- `PAPERLENS_TRACE_CONTENT=1` to opt into logging prompt/output text for evaluation or fine-tuning data; default traces store metadata only
- `PAPERLENS_MINILAB_PROVIDER=local|modal` for the Lab Mode `Run mini-lab` path; `modal` uses a bounded Modal CPU function and validates returned paper/span/source/code hashes
- `PAPERLENS_MODAL_BIN=/opt/anaconda3/bin/modal` when the Modal CLI is installed outside the active shell `PATH`
- `PAPERLENS_MODAL_MINILAB_TIMEOUT=180` to bound the local wait for a Modal mini-lab job
- `NEXT_PUBLIC_PAPERLENS_USE_MODEL=1` for the exported React frontend

## Backend Scenario Checks

Run deterministic fallback checks:

```bash
PAPERLENS_TRACE_ENABLED=0 .venv/bin/python -m paperlens_lab.scenario_runner --compact
```

Run the same translation -> Q&A -> ExperimentSpec -> Growth path through the
configured small model:

```bash
PAPERLENS_PROVIDER=hf \
PAPERLENS_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
PAPERLENS_QUALITY_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
.venv/bin/python -m paperlens_lab.scenario_runner --use-model --compact
```

## Real-Paper Validation Snapshot

The current local validation evidence is intentionally stored as ignored runtime
artifacts under `outputs/service_demo_validation/2026-06-13/`. The hybrid app
summarizes the latest available evidence at `GET /api/validation` and displays a
compact snapshot on the landing page when those artifacts exist.

Current verified snapshot:

- Real arXiv/PDF papers: 3
- Named papers: `1706.03762`, `2005.11401`, `2106.09685`
- Real-paper evaluations: 36/36 passed in `hf_three_papers_starter_smoke_v13`
- Model traces: 27/27 stored latest real-paper traces are `status=model`, with 0 fallbacks and 0 trace errors
- Trace binding: 24 summary-referenced trace IDs are present in the matching trace JSONL with `status=model` and no errors
- Scope: parsed 15/19/26 PDF pages for `1706.03762` / `2005.11401` / `2106.09685`; reader spans are 548/1000/1000, with larger papers capped by the 1000-span validation limit
- Adversarial long-context proof: each paper includes an 8k+ character ordered evidence packet with the target evidence buried near the middle and at least 111 distractor spans; the model must cite the exact target source ID and quote without fallback
- Runnable mini-lab proof: every paper generates dependency-free starter code and passes `starter_code_smoke` by parsing, importing, and running `run()`
- ExperimentSpec proof: saved paper-to-lab specs are revalidated by the current evaluator so old heavy benchmark/GPU plans cannot stay green
- Research Growth iteration proof: every paper runs a second Growth pass after the first ideas are written to memory; the second pass must cite `paper:selected-middle`, `run:r1`, and a prior `growth_idea:*` memory
- Persistent memory proof: 14 JSONL memory records across 3 papers, including `paper_span`, `mini_lab_result`, `growth_idea`, and `growth_iteration_idea`
- Local selected-span API proof: `1706.03762`, span `P5.S8`, evidence window `P5.S5-P5.S11`, quote id `P5.S8`, source hash `824e98d76dcb7231`
- Evidence consistency: saved QA citations are checked against stored source-evidence maps, adversarial quotes are validated against the long-context evidence packet, and local quote ids must stay inside the source-index window
- Fine-tuning decision from real failures: `no`; no repeated trainable failure cluster has been observed yet

Useful validation commands:

```bash
RUN_ROOT="outputs/service_demo_validation/$(date +%F)"
RUN_NAME="hf_three_papers_starter_smoke_v13"
PAPERLENS_PROVIDER=hf \
PAPERLENS_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
PAPERLENS_QUALITY_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
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

Hybrid Space app:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860/` for the React reader and `http://127.0.0.1:7860/gradio` for the Gradio fallback.

## Hackathon Entry Status

- GitHub repository: https://github.com/kcyoow/paperlens-lab
- Hugging Face Space: https://huggingface.co/spaces/build-small-hackathon/paperlens-lab
- Codex-attributed commits: this project is scaffolded and iterated with Codex as the coding agent.
- Space README repo link: this README contains the public GitHub repository link above.

## Built With Codex

This Space was scaffolded and prepared with Codex as the coding agent for the Build Small Hackathon.

## Safety And Grounding

PaperLens Lab keeps source evidence visible and treats generated explanation as interpretation unless it is tied back to a source span. It is intended as a research reading assistant, not a substitute for reading the original paper.
