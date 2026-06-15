# PaperLens Lab Frontend

This folder is the current frontend baseline for PaperLens Lab.

## Product Baseline

- Default audience: international capstone/demo reviewers.
- Default UI language: English.
- Optional UI language: Korean.
- Default paper view: English source text.
- Optional paper views: Korean translation and side-by-side source/translation.
- Core flow: upload or provide a real paper, read it with source-linked translation, mark exact selected text, ask AI about selected text or the whole paper, then open Lab Mode for a source-bound experiment run.

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
That `out/` directory is intentionally deployable for the Hugging Face Space,
because the Python `app.py` serves static files directly and should not depend
on a Node build step during Space startup.

To preview the static export locally:

```bash
npm run preview:static
```

## Backend Connection

- The static export is served by the Python app at `/`.
- API calls go to the same origin under `/api/*`.
- Empty paper states stay empty until the user loads a PDF, arXiv URL/ID, or paper text.
- PDF parsing, arXiv fetching, selected-span/whole-paper Q&A, and experiment card generation are connected to the Python backend.
- On-demand span translation is connected through `/api/translate-span`; background batch translation uses `/api/translate` for pending reader spans.
- UI text is handled by the local `src/lib/i18n.ts` dictionary.
- Generated folders such as `.next/` and `node_modules/` are intentionally ignored.
- `out/` is generated, but it is intentionally kept as a deploy artifact for the Space sync.

## Deployment Direction

The product frontend stays in React/Next, while the Hugging Face Space remains `sdk: gradio` for Build Small compatibility. `app.py` serves the exported `out/` files and exposes the service API directly; there is no separate demo route.
