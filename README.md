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

Translate, explain, and prototype ideas from research papers with small models.

PaperLens Lab is a Hugging Face Space for turning a paper into a grounded reading guide and a small experiment card. It accepts a PDF, arXiv ID/URL, or pasted paper text, then separates direct paper claims from interpretation and generates a starter experiment scaffold.

## What It Does

- Extracts paper text from PDF, arXiv metadata, or pasted text.
- Builds a Korean-friendly reading guide with evidence IDs.
- Lists source-backed claims and key terms.
- Creates an experiment card for testing a paper idea.
- Generates a small `starter.py` scaffold for the experiment.

## Model Use

The app works in fallback mode without external credentials. For model-backed outputs, set:

```bash
HF_TOKEN=...
PAPERLENS_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

The default model target is under the Build Small Hackathon's 32B limit.

## Local Run

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
