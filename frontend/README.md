---
title: PaperLens Lab Frontend
sdk: static
app_build_command: npm ci && npm run build
app_file: out/index.html
pinned: false
license: apache-2.0
---

# PaperLens Lab Frontend

This folder is the current frontend baseline for PaperLens Lab.

## Product Baseline

- Default audience: international capstone/demo reviewers.
- Default UI language: English.
- Optional UI language: Korean.
- Default paper view: English source text.
- Optional paper views: Korean translation and side-by-side source/translation.
- Core flow: upload or provide a paper, read it with source-linked translation checks, mark important lines, ask AI about a selected line, then open Lab Mode for a small experiment sketch.

## Routes

- `/` - paper input landing screen.
- `/reader` - mock paper reader and Lab Mode workspace.

## Commands

```bash
npm install
npm run dev
npm run build
```

`npm run build` produces a static export in `out/`.

To preview the static export locally:

```bash
npm run preview:static
```

## Current Boundaries

- The current reader uses mock paper data from `src/lib/mock-data.ts`.
- PDF parsing, arXiv fetching, translation generation, AI answers, and experiment generation are not connected to a backend yet.
- UI text is handled by the local `src/lib/i18n.ts` dictionary.
- Generated folders such as `.next/`, `out/`, and `node_modules/` are intentionally ignored.

## Deployment Direction

The frontend is prepared for a static-first deployment path, such as a Hugging Face Static Space. A backend can be connected later through API calls without changing this baseline reader flow.
