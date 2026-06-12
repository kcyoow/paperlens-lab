from __future__ import annotations

import os

import uvicorn

from paperlens_lab.server import create_app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or "7860")
    uvicorn.run(app, host=host, port=port)
