---
title: PaperLens Lab
emoji: "📄"
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: 3.12
app_file: app.py
pinned: false
license: apache-2.0
---

# PaperLens Lab

Read, verify, and prototype ideas from research papers.

PaperLens Lab is moving toward a reader-first paper workspace: read the source text, compare translation, mark important lines, ask grounded questions, and turn promising ideas into small experiments.

## Gradio Hackathon Runtime

The active Hugging Face Space is a Gradio app, matching the Build Small Hackathon requirement. The Gradio runtime now renders a custom reader workspace that preserves the current Next.js mockup's visual direction as closely as practical inside a Space-friendly Gradio app.

- Default UI language: English.
- Optional UI language: Korean.
- Main runtime: `app.py`.
- Visual source-of-truth: `frontend/`.
- Current boundary: PDF parsing and source extraction work locally; full translation generation, AI answers, and experiment execution are still staged backend work.

## Current Preview Flow

- Shows a paper input landing screen.
- Opens a reader workspace with English source text by default.
- Switches UI language between English and Korean.
- Switches paper view between English, Korean translation, and side-by-side.
- Lets the user select lines, inspect source/translation, add marks, ask AI, and open Lab Mode.

## Backend Prototype

The Gradio app keeps the original backend hooks for source-grounded analysis and experiment cards. It also keeps the reader-first flow visible: original text, Korean translation draft, side-by-side checking, marks, source inspection, AI question affordance, and Lab Mode affordance.

Later, the single-Space path can move to `gradio.Server` or Docker if the custom frontend needs full React-level interaction while staying compatible with the Gradio-centered hackathon rules.

## Model Use

The app works in fallback mode without external credentials. For model-backed outputs, set:

```bash
HF_TOKEN=...
PAPERLENS_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

The default model target is under the Build Small Hackathon's 32B limit.

## Local Run

Next design reference:

```bash
cd frontend
npm install
npm run dev
```

Active Gradio Space app:

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
