from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "paperlens-modal-gpu-probe"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "numpy", "datasets", "sacrebleu")
    .add_local_python_source("paperlens_lab")
)


@app.function(image=image, gpu="T4", timeout=900, memory=8192)
def execute_gpu_probe(job: dict[str, Any]) -> str:
    import time

    import torch

    from paperlens_lab.gpu_lab import _validate_gpu_script_contract

    started = time.time()
    code = str(job.get("code") or "")
    validation_errors = _validate_gpu_script_contract(code)
    hardware = {
        "cudaAvailable": bool(torch.cuda.is_available()),
        "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "torchVersion": torch.__version__,
    }
    if validation_errors:
        return _result_json(
            job,
            passed=False,
            reasons=validation_errors,
            hardware=hardware,
            logs=["GPU script contract validation failed inside Modal."],
            duration_ms=int((time.time() - started) * 1000),
        )

    namespace: dict[str, Any] = {"__name__": "paperlens_generated_gpu_probe"}
    try:
        exec(compile(code, "paperlens_generated_gpu_probe.py", "exec"), namespace)
        fn = namespace.get("run_paperlens_gpu_probe")
        if not callable(fn):
            raise RuntimeError("run_paperlens_gpu_probe is not callable")
        raw = fn(
            {
                "candidate": job.get("candidate", {}),
                "evidence_rows": job.get("evidenceRows", []),
                "selected_span": job.get("selectedSpan", ""),
                "paper_title": job.get("paperTitle", ""),
                "reproduction_level": job.get("reproductionLevel", "probe"),
                "requested_reproduction_level": job.get("requestedReproductionLevel", job.get("reproductionLevel", "probe")),
                "max_train_samples": 12000,
                "max_test_samples": 2000,
                "max_epochs": 2,
            }
        )
    except Exception as exc:
        return _result_json(
            job,
            passed=False,
            reasons=[f"{type(exc).__name__}: {exc}"],
            hardware=hardware,
            logs=["Generated GPU probe raised an exception inside Modal."],
            duration_ms=int((time.time() - started) * 1000),
        )

    if not isinstance(raw, dict):
        return _result_json(
            job,
            passed=False,
            reasons=["GPU probe did not return an object."],
            hardware=hardware,
            logs=["Generated GPU probe returned a non-dict payload."],
            duration_ms=int((time.time() - started) * 1000),
        )
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
    limitations = raw.get("limitations") if isinstance(raw.get("limitations"), list) else []
    generated_reasons = list(raw.get("reasons") or [])
    generated_passed = bool(raw.get("passed", bool(metrics or rows)))
    execution_completed = bool(metrics or rows)
    execution_reasons = [] if execution_completed else (generated_reasons or ["GPU probe returned no rows or metrics."])
    claim_comparison = _claim_comparison_from_raw(
        raw,
        generated_passed=generated_passed,
        generated_reasons=generated_reasons,
        limitations=limitations,
    )
    artifacts = _artifacts_from_raw(raw)
    return _result_json(
        job,
        passed=execution_completed,
        reasons=execution_reasons,
        hardware={**hardware, **(raw.get("hardware") if isinstance(raw.get("hardware"), dict) else {})},
        dataset=raw.get("dataset") if isinstance(raw.get("dataset"), dict) else {},
        metrics=metrics,
        rows=rows,
        logs=list(raw.get("logs") or []),
        claim_comparison=claim_comparison,
        artifacts=artifacts,
        limitations=limitations,
        duration_ms=int((time.time() - started) * 1000),
    )


def _result_json(job: dict[str, Any], **kwargs: Any) -> str:
    return json.dumps(_result(job, **kwargs), ensure_ascii=False, sort_keys=True)


def _claim_comparison_from_raw(
    raw: dict[str, Any],
    *,
    generated_passed: bool,
    generated_reasons: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    raw_claim = raw.get("claim_comparison")
    if isinstance(raw_claim, dict):
        claim = dict(raw_claim)
    elif raw_claim:
        claim = {"summary": str(raw_claim)}
    else:
        claim = {}
    claim.setdefault("verdict", "supported" if generated_passed else "not_supported")
    claim.setdefault("generatedPassed", generated_passed)
    if generated_reasons:
        claim.setdefault("reasons", generated_reasons)
    if limitations:
        claim.setdefault("limitations", limitations)
    return claim


def _artifacts_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    artifacts = dict(raw.get("artifacts")) if isinstance(raw.get("artifacts"), dict) else {}
    report_html = raw.get("report_html") or raw.get("reportHtml")
    if report_html and not (artifacts.get("reportHtml") or artifacts.get("report_html")):
        artifacts["reportHtml"] = str(report_html)
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    if metrics and not isinstance(artifacts.get("metrics"), dict):
        artifacts["metrics"] = metrics
    return artifacts


def _result(
    job: dict[str, Any],
    *,
    passed: bool,
    reasons: list[str],
    hardware: dict[str, Any],
    dataset: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    rows: list[Any] | None = None,
    logs: list[str] | None = None,
    claim_comparison: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
    duration_ms: int,
) -> dict[str, Any]:
    hardware = _json_safe(hardware)
    dataset = _json_safe(dataset or {})
    metrics = _json_safe(metrics or {})
    rows = _json_safe(rows or [])
    claim_comparison = _json_safe(claim_comparison or {"verdict": "inconclusive", "limitations": limitations or []})
    artifacts = _json_safe(artifacts or {})
    limitations = _json_safe(limitations or [])
    return {
        "passed": passed,
        "reasons": reasons,
        "provider": "modal",
        "executionMode": "modal-gpu-replication-probe",
        "runner": APP_NAME,
        "gpuRequested": True,
        "hardware": hardware,
        "paperId": job.get("paperId", ""),
        "paperTitle": job.get("paperTitle", ""),
        "spanId": job.get("spanId", ""),
        "candidateSetId": job.get("candidateSetId", ""),
        "candidateId": job.get("candidateId", ""),
        "reproductionLevel": job.get("reproductionLevel", "probe"),
        "requestedReproductionLevel": job.get("requestedReproductionLevel", job.get("reproductionLevel", "probe")),
        "sourceHash": job.get("sourceHash", ""),
        "codeHash": job.get("codeHash", ""),
        "evidenceHash": job.get("evidenceHash", ""),
        "dataset": dataset,
        "metrics": metrics,
        "rows": rows,
        "logs": [
            f"modal app={APP_NAME}",
            f"cuda={hardware.get('cudaAvailable')}",
            f"gpu={hardware.get('gpuName')}",
            *(logs or []),
        ],
        "claimComparison": claim_comparison,
        "artifacts": artifacts,
        "limitations": limitations,
        "durationMs": duration_ms,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


@app.local_entrypoint()
def main(payload_path: str) -> str:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    result = execute_gpu_probe.remote(payload)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, sort_keys=True)
