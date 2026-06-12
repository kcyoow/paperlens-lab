from __future__ import annotations

import os
from typing import Optional


DEFAULT_MODEL = os.getenv("PAPERLENS_MODEL", "Qwen/Qwen3-4B-Instruct-2507")


def generate_with_hf_inference(
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    max_new_tokens: int = 420,
) -> Optional[str]:
    """Best-effort optional model adapter.

    The app is useful without this adapter, but a Space can set HF_TOKEN and
    PAPERLENS_MODEL to use a small model for richer translations/explanations.
    """

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return None

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return None

    try:
        client = InferenceClient(model=model_id, token=token)
        return client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=False,
        ).strip()
    except Exception:
        return None
