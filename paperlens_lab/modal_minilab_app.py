from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "paperlens-modal-minilab"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("paperlens_lab")


@app.function(image=image, timeout=30, memory=256)
def execute_minilab(payload: dict[str, Any]) -> dict[str, Any]:
    from paperlens_lab.scenario_eval import run_starter_code

    started = time.time()
    execution = run_starter_code(
        str(payload.get("code", "")),
        evidence_rows=list(payload.get("evidenceRows") or []),
        require_evidence_rows=True,
    )
    rows = execution.get("rows") if isinstance(execution.get("rows"), list) else []
    return {
        "provider": "modal",
        "executionMode": "modal-source-bound-remote-function",
        "runner": APP_NAME,
        "paperId": payload.get("paperId", ""),
        "paperTitle": payload.get("paperTitle", ""),
        "spanId": payload.get("spanId", ""),
        "sourceHash": payload.get("sourceHash", ""),
        "selectedSpanHash": payload.get("selectedSpanHash", ""),
        "codeHash": payload.get("codeHash", ""),
        "sourceIndexBound": bool(payload.get("sourceIndexBound")),
        "evidenceRowCount": int(payload.get("evidenceRowCount") or 0),
        "evidenceHash": payload.get("evidenceHash", ""),
        "passed": bool(execution.get("passed")),
        "reasons": list(execution.get("reasons") or []),
        "rows": rows,
        "logs": [
            f"modal app={APP_NAME}",
            f"remote code_hash={payload.get('codeHash', '')}",
            f"remote source_hash={payload.get('sourceHash', '')}",
            f"remote evidence_hash={payload.get('evidenceHash', '')}",
            f"remote evidence_rows={len(payload.get('evidenceRows') or [])}",
            f"remote rows={len(rows)}",
        ],
        "durationMs": int((time.time() - started) * 1000),
    }


@app.local_entrypoint()
def main(payload_path: str) -> str:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    result = execute_minilab.remote(payload)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)
