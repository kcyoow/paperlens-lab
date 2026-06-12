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
- `/reader` - paper reader and Lab Mode workspace.

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

## Backend Connection

- The static export is served by the Python app at `/`.
- API calls go to the same origin under `/api/*`.
- `src/lib/mock-data.ts` remains the offline fallback when no paper is loaded.
- PDF parsing, arXiv fetching, selected-span Q&A, and experiment card generation are connected to the Python backend.
- On-demand span translation is connected through `/api/translate-span`; full-document batch translation remains a staged model feature.
- UI text is handled by the local `src/lib/i18n.ts` dictionary.
- Generated folders such as `.next/`, `out/`, and `node_modules/` are intentionally ignored.

## Deployment Direction

The product frontend stays in React/Next, while the Hugging Face Space remains `sdk: gradio`. `app.py` serves the exported `out/` files and mounts the Gradio fallback under `/gradio`.
