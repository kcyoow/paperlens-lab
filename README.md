---
title: PaperLens Lab
emoji: "📄"
colorFrom: green
colorTo: gray
sdk: static
app_build_command: cd frontend && npm ci && npm run build && cp -R out/* ..
app_file: index.html
pinned: false
license: apache-2.0
---

# PaperLens Lab

Read, verify, and prototype ideas from research papers.

PaperLens Lab is moving toward a reader-first paper workspace: read the source text, compare translation, mark important lines, ask grounded questions, and turn promising ideas into small experiments.

## Frontend Preview Baseline

The current Hugging Face Space serves the Next.js frontend baseline from `frontend/` as a static preview. This is the product frame we will refine before connecting the backend.

- Default UI language: English.
- Optional UI language: Korean.
- Main route: `/reader`.
- Build output: `frontend/out/`, copied to the Space root during the Static Space build.
- Current boundary: PDF parsing, translation generation, AI answers, and experiment generation are still mocked.

## Current Preview Flow

- Shows a paper input landing screen.
- Opens a reader workspace with English source text by default.
- Switches UI language between English and Korean.
- Switches paper view between English, Korean translation, and side-by-side.
- Lets the user select lines, inspect source/translation, add marks, ask AI, and open Lab Mode.

## Backend Prototype

The earlier Gradio prototype code remains in this repository as the backend direction, but it is not the active Space runtime while the Space is configured as a static frontend preview.

Later, the single-Space path should move to a Docker Space so the same URL can serve the frontend and backend together.

## Model Use

The app works in fallback mode without external credentials. For model-backed outputs, set:

```bash
HF_TOKEN=...
PAPERLENS_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

The default model target is under the Build Small Hackathon's 32B limit.

## Local Run

Frontend preview:

```bash
cd frontend
npm install
npm run dev
```

Backend prototype:

```bash
python -m pip install -r requirements.txt
python app.py
```

## Hackathon Entry Status

- GitHub repository: https://github.com/kcyoow/paperlens-lab
- Hugging Face Space: https://huggingface.co/spaces/build-small-hackathon/paperlens-lab
- Codex-attributed commits: this project is scaffolded and iterated with Codex as the coding agent.
- Space README repo link: this README contains the public GitHub repository link above.

## Built With Codex

This Space was scaffolded and prepared with Codex as the coding agent for the Build Small Hackathon.

## Safety And Grounding

PaperLens Lab keeps source evidence visible and treats generated explanation as interpretation unless it is tied back to a source span. It is intended as a research reading assistant, not a substitute for reading the original paper.
