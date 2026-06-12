from __future__ import annotations

import gradio as gr

from .analysis import analyze_paper, experiment_card
from .ingest import build_source


CSS = """
.paperlens-shell { max-width: 1280px; margin: 0 auto; }
.paperlens-title h1 { font-size: 30px; line-height: 1.15; margin-bottom: 2px; }
.paperlens-title p { margin-top: 0; color: #586174; }
.paperlens-output { min-height: 420px; }
"""


EXAMPLE_TEXT = """Abstract: We propose a retrieval-augmented reading assistant for scientific papers. The system grounds every explanation in cited source spans, separates paper claims from interpretation, and produces small experiment cards that help readers test an idea before investing in a full implementation."""


def _source_from_inputs(uploaded_pdf, arxiv_or_url, pasted_text, max_pdf_pages):
    return build_source(uploaded_pdf, arxiv_or_url or "", pasted_text or "", int(max_pdf_pages))


def run_analysis(uploaded_pdf, arxiv_or_url, pasted_text, audience, focus, max_pdf_pages, use_model):
    try:
        source = _source_from_inputs(uploaded_pdf, arxiv_or_url, pasted_text, max_pdf_pages)
        return analyze_paper(source, audience, focus, use_model)
    except Exception as exc:
        message = f"## Could not analyze paper\n\n{exc}"
        return message, "", "", ""


def run_experiment(uploaded_pdf, arxiv_or_url, pasted_text, idea, audience, max_pdf_pages, use_model):
    try:
        source = _source_from_inputs(uploaded_pdf, arxiv_or_url, pasted_text, max_pdf_pages)
        return experiment_card(source, idea, audience, use_model)
    except Exception as exc:
        return f"## Could not build experiment card\n\n{exc}", ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="PaperLens Lab",
        css=CSS,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    ) as demo:
        with gr.Column(elem_classes=["paperlens-shell"]):
            gr.Markdown(
                "# PaperLens Lab\n"
                "Translate, explain, and prototype ideas from research papers with small models.",
                elem_classes=["paperlens-title"],
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=4):
                    uploaded_pdf = gr.File(
                        label="PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    arxiv_or_url = gr.Textbox(
                        label="arXiv ID or URL",
                        placeholder="2505.09388 or https://arxiv.org/abs/2505.09388",
                    )
                    pasted_text = gr.Textbox(
                        label="Paper text",
                        value=EXAMPLE_TEXT,
                        lines=9,
                        max_lines=16,
                    )
                with gr.Column(scale=2):
                    audience = gr.Radio(
                        ["Undergraduate", "Practitioner", "Researcher"],
                        value="Practitioner",
                        label="Audience",
                    )
                    focus = gr.Dropdown(
                        [
                            "Core idea",
                            "Math and notation",
                            "Implementation details",
                            "Limitations",
                            "Experiment design",
                        ],
                        value="Core idea",
                        label="Focus",
                    )
                    max_pdf_pages = gr.Slider(
                        minimum=2,
                        maximum=24,
                        value=10,
                        step=1,
                        label="PDF pages",
                    )
                    use_model = gr.Checkbox(
                        value=False,
                        label="Use HF model adapter",
                    )
                    analyze_btn = gr.Button("Analyze Paper", variant="primary")

            with gr.Tabs():
                with gr.Tab("Reader"):
                    reader_output = gr.Markdown(elem_classes=["paperlens-output"])
                with gr.Tab("Evidence"):
                    evidence_output = gr.Markdown(elem_classes=["paperlens-output"])
                with gr.Tab("Claims"):
                    claims_output = gr.Markdown(elem_classes=["paperlens-output"])
                with gr.Tab("Raw Extract"):
                    raw_output = gr.Textbox(lines=18, label="Top source spans")

            gr.Markdown("## Idea Lab")
            with gr.Row(equal_height=False):
                idea = gr.Textbox(
                    label="Paper idea to test",
                    placeholder="Example: test whether evidence-linked explanations reduce hallucinated claims",
                    lines=4,
                )
                experiment_btn = gr.Button("Build Experiment Card", variant="secondary")

            with gr.Row(equal_height=False):
                experiment_output = gr.Markdown(label="Experiment card", elem_classes=["paperlens-output"])
                starter_output = gr.Code(label="starter.py", language="python", lines=22)

        analyze_btn.click(
            fn=run_analysis,
            inputs=[uploaded_pdf, arxiv_or_url, pasted_text, audience, focus, max_pdf_pages, use_model],
            outputs=[reader_output, evidence_output, claims_output, raw_output],
        )
        experiment_btn.click(
            fn=run_experiment,
            inputs=[uploaded_pdf, arxiv_or_url, pasted_text, idea, audience, max_pdf_pages, use_model],
            outputs=[experiment_output, starter_output],
        )

    return demo
