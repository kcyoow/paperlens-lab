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
- Current boundary: PDF parsing and source extraction work locally; full translation generation, AI answers, and experiment execution are still staged backend work.

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
- `PAPERLENS_MODEL=Qwen/Qwen3-4B-Instruct-2507`
- `PAPERLENS_QUALITY_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- `PAPERLENS_TRACE_PATH=outputs/agent_traces.jsonl`
- `NEXT_PUBLIC_PAPERLENS_USE_MODEL=1` for the exported React frontend

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
