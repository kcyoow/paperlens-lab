from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .ingest import clean_text
from .mini_lab import build_evidence_rows
from .source_index import get_span_text, text_hash


MODAL_GPU_SCRIPT = Path(__file__).with_name("modal_gpu_lab_app.py")


class GpuLabError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def gpu_code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def run_gpu_probe_job(binding: dict[str, Any]) -> dict[str, Any]:
    job = _build_gpu_job_payload(binding)
    local_errors = _validate_gpu_script_contract(str(job.get("code") or ""))
    if local_errors:
        return _failed_gpu_result(job, "; ".join(local_errors))
    raw_result = _run_modal_gpu_probe_with_cli(job)
    return _validated_gpu_result(raw_result, job)


def _build_gpu_job_payload(binding: dict[str, Any]) -> dict[str, Any]:
    code = str(binding.get("code") or "")
    if not code.strip():
        raise GpuLabError("GPU script is required.")
    paper_id = str(binding.get("paperId") or "")
    span_id = str(binding.get("spanId") or "")
    selected_span = clean_text(str(binding.get("selectedSpan") or ""))
    indexed_span = get_span_text(paper_id, span_id) if paper_id and span_id else ""
    source_span = clean_text(indexed_span or selected_span)
    if not paper_id or not span_id or not source_span:
        raise GpuLabError("GPU probe requires indexed paper evidence.")
    evidence_rows = build_evidence_rows(paper_id=paper_id, span_id=span_id, selected_span=source_span)
    if not evidence_rows:
        raise GpuLabError("GPU probe requires indexed evidence rows.", status_code=404)
    evidence_hash = text_hash(json.dumps(evidence_rows, ensure_ascii=False, sort_keys=True))
    return {
        "gpuRunId": str(binding.get("id") or ""),
        "candidateSetId": str(binding.get("candidateSetId") or ""),
        "candidateId": str(binding.get("candidateId") or ""),
        "candidate": binding.get("candidate") if isinstance(binding.get("candidate"), dict) else {},
        "reproductionLevel": str(binding.get("reproductionLevel") or "scaled"),
        "requestedReproductionLevel": str(binding.get("requestedReproductionLevel") or binding.get("reproductionLevel") or "scaled"),
        "code": code,
        "codeHash": gpu_code_hash(code),
        "paperId": paper_id,
        "paperTitle": str(binding.get("paperTitle") or "Untitled paper"),
        "spanId": span_id,
        "selectedSpan": source_span,
        "sourceHash": text_hash(source_span),
        "evidenceRows": evidence_rows,
        "evidenceRowCount": len(evidence_rows),
        "evidenceHash": evidence_hash,
        "createdAt": time.time(),
    }


def _validate_gpu_script_contract(code: str) -> list[str]:
    if not code.strip():
        return ["GPU script is empty"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"GPU script syntax error: {exc.msg}"]
    reasons: list[str] = []
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "run_paperlens_gpu_probe" not in functions:
        reasons.append("GPU script must define run_paperlens_gpu_probe(config=None)")
    allowed_import_roots = {
        "collections",
        "datasets",
        "itertools",
        "json",
        "math",
        "numpy",
        "sacrebleu",
        "time",
        "torch",
        "torchvision",
    }
    blocked_names = {"subprocess", "socket", "requests", "httpx", "urllib", "pathlib", "shutil"}
    blocked_calls = {"eval", "exec", "compile", "open", "__import__", "input"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_import_roots:
                    reasons.append(f"GPU script imports blocked module {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in allowed_import_roots:
                reasons.append(f"GPU script imports blocked module {node.module}")
        elif isinstance(node, ast.Name) and node.id in blocked_names:
            reasons.append(f"GPU script uses blocked name {node.id}")
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in blocked_calls:
                    reasons.append(f"GPU script uses blocked call {name}")
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name in {"__import__"}:
                    reasons.append(f"GPU script uses blocked call {name}")
    lowered = code.lower()
    if "torch" not in lowered:
        reasons.append("GPU script must use torch")
    if "cuda" not in lowered:
        reasons.append("GPU script must record CUDA/GPU availability")
    if 'load_dataset("multi30k"' in lowered or "load_dataset('multi30k'" in lowered:
        reasons.append("GPU script must use the exact Hugging Face dataset id bentrevett/multi30k for Multi30k")
    if "bentrevett/multi30k" in lowered and ("['translation']" in lowered or '["translation"]' in lowered):
        reasons.append("GPU script must read bentrevett/multi30k rows from en/de fields, not a translation field")
    if "bentrevett/multi30k" in lowered and ("dataloader" in lowered or "torch.utils.data" in lowered):
        reasons.append("GPU script must build fixed-shape Multi30k tensors directly instead of using DataLoader/custom Dataset")
    if "transformerencoder" in lowered and (
        "transformer_model(src_tensor" in lowered
        or "transformer_model(tgt_tensor" in lowered
        or "transformer(src_tensor" in lowered
        or "transformer(tgt_tensor" in lowered
    ):
        reasons.append("GPU script must pass embedded float tensors with shape [batch, seq_len, d_model] to TransformerEncoder")
    reasons.extend(_counter_most_common_items_errors(tree))
    blocked_data_terms = (
        "mock",
        "dummy",
        "fake",
        "placeholder",
        "synthetic",
        "simulated",
        "toy",
        "random-vector",
        "random vector",
        "randomly generated",
        "torch.randint",
        "torch.randn",
        "torch.rand(",
        "np.random",
        "numpy.random",
        "random.random",
        "random.randint",
        "random.choice",
    )
    for term in blocked_data_terms:
        if term in lowered:
            reasons.append(f"GPU script uses blocked generated/mock data term {term}")
    return list(dict.fromkeys(reasons))


def _counter_most_common_items_errors(tree: ast.AST) -> list[str]:
    most_common_vars: set[str] = set()
    for node in ast.walk(tree):
        assigned_value = None
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            assigned_value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            assigned_value = node.value
            targets = [node.target]
        if assigned_value is None or not _is_most_common_call(assigned_value):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                most_common_vars.add(target.id)

    reasons: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "items":
            continue
        value = node.func.value
        if _is_most_common_call(value) or (isinstance(value, ast.Name) and value.id in most_common_vars):
            reasons.append("Counter.most_common returns a list; GPU script must iterate over its returned pairs directly without .items()")
    return reasons


def _is_most_common_call(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "most_common"
    )


def _run_modal_gpu_probe_with_cli(job: dict[str, Any]) -> dict[str, Any]:
    modal_bin = _modal_bin()
    if not modal_bin:
        return _failed_gpu_result(job, "Modal CLI was not found. Set PAPERLENS_MODAL_BIN or install modal.")

    timeout = _modal_timeout_seconds()
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="paperlens-modal-gpu-probe-") as tmp:
        tmp_path = Path(tmp)
        payload_path = tmp_path / "payload.json"
        result_path = tmp_path / "result.json"
        payload_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        command = [
            modal_bin,
            "run",
            "-q",
            "--write-result",
            str(result_path),
            str(MODAL_GPU_SCRIPT),
            "--payload-path",
            str(payload_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return _failed_gpu_result(job, f"Modal GPU probe timed out after {timeout}s.")

        logs = _trim_logs(completed.stdout, completed.stderr)
        if completed.returncode != 0:
            detail = logs[-1] if logs else "no Modal output"
            return _failed_gpu_result(job, f"Modal CLI exited with {completed.returncode}: {detail}", logs=logs)
        if not result_path.exists():
            return _failed_gpu_result(job, "Modal GPU probe did not write a result payload.", logs=logs)
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _failed_gpu_result(job, f"Modal GPU result payload was not JSON: {exc}", logs=logs)
        if not isinstance(result, dict):
            return _failed_gpu_result(job, "Modal GPU result payload was not an object.", logs=logs)
        result.setdefault("logs", [])
        result["logs"] = [*logs, *list(result.get("logs") or [])]
        result["durationMs"] = int((time.time() - started) * 1000)
        return result


def _validated_gpu_result(result: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    reasons = list(result.get("reasons") or [])
    validation = {
        "paperIdMatches": result.get("paperId") == job.get("paperId"),
        "spanIdMatches": result.get("spanId") == job.get("spanId"),
        "candidateIdMatches": result.get("candidateId") == job.get("candidateId"),
        "codeHashMatches": result.get("codeHash") == job.get("codeHash"),
        "evidenceHashMatches": result.get("evidenceHash") == job.get("evidenceHash"),
        "providerIsModal": result.get("provider") == "modal",
        "gpuRequested": bool(result.get("gpuRequested")),
    }
    hardware = result.get("hardware") if isinstance(result.get("hardware"), dict) else {}
    if not hardware.get("cudaAvailable"):
        reasons.append("Modal GPU probe did not report CUDA availability")
    for key, passed in validation.items():
        if not passed:
            reasons.append(f"GPU probe validation failed: {key}")
    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    if not rows and not metrics:
        reasons.append("GPU probe returned no rows or metrics")
    claim_comparison = _normalized_gpu_claim_comparison(result, rows=rows, metrics=metrics)
    return {
        "passed": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "provider": result.get("provider") or "modal",
        "executionMode": result.get("executionMode") or "modal-gpu-replication-probe",
        "runner": result.get("runner") or "paperlens-modal-gpu-probe",
        "gpuRequested": bool(result.get("gpuRequested")),
        "hardware": hardware,
        "paperId": job.get("paperId", ""),
        "paperTitle": job.get("paperTitle", ""),
        "spanId": job.get("spanId", ""),
        "candidateSetId": job.get("candidateSetId", ""),
        "candidateId": job.get("candidateId", ""),
        "sourceHash": job.get("sourceHash", ""),
        "codeHash": job.get("codeHash", ""),
        "evidenceHash": job.get("evidenceHash", ""),
        "evidenceRowCount": int(job.get("evidenceRowCount") or 0),
        "reproductionLevel": job.get("reproductionLevel", "scaled"),
        "requestedReproductionLevel": job.get("requestedReproductionLevel", job.get("reproductionLevel", "scaled")),
        "validation": validation,
        "dataset": result.get("dataset") if isinstance(result.get("dataset"), dict) else {},
        "metrics": metrics,
        "rows": rows,
        "logs": list(result.get("logs") or []),
        "claimComparison": claim_comparison,
        "limitations": list(result.get("limitations") or []),
        "durationMs": int(result.get("durationMs") or 0),
    }


def _normalized_gpu_claim_comparison(
    result: dict[str, Any],
    *,
    rows: list[Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    raw_claim = result.get("claimComparison")
    claim = dict(raw_claim) if isinstance(raw_claim, dict) else {}
    if "generatedPassed" in claim:
        generated_passed = bool(claim.get("generatedPassed"))
    elif (rows or metrics) and result.get("passed") is False:
        generated_passed = False
    else:
        generated_passed = bool(result.get("passed"))
    claim.setdefault("generatedPassed", generated_passed)
    claim.setdefault("verdict", "supported" if generated_passed else "not_supported")
    return claim


def _failed_gpu_result(job: dict[str, Any], reason: str, *, logs: list[str] | None = None) -> dict[str, Any]:
    return {
        "passed": False,
        "reasons": [reason],
        "provider": "modal",
        "executionMode": "modal-gpu-replication-probe",
        "runner": "paperlens-modal-gpu-probe",
        "gpuRequested": True,
        "hardware": {},
        "paperId": job.get("paperId", ""),
        "paperTitle": job.get("paperTitle", ""),
        "spanId": job.get("spanId", ""),
        "candidateSetId": job.get("candidateSetId", ""),
        "candidateId": job.get("candidateId", ""),
        "sourceHash": job.get("sourceHash", ""),
        "codeHash": job.get("codeHash", ""),
        "evidenceHash": job.get("evidenceHash", ""),
        "evidenceRowCount": int(job.get("evidenceRowCount") or 0),
        "reproductionLevel": job.get("reproductionLevel", "scaled"),
        "requestedReproductionLevel": job.get("requestedReproductionLevel", job.get("reproductionLevel", "scaled")),
        "validation": {},
        "dataset": {},
        "metrics": {},
        "rows": [],
        "logs": logs or [],
        "claimComparison": {"verdict": "failed_to_execute", "limitations": [reason]},
        "limitations": [reason],
        "durationMs": 0,
    }


def _modal_bin() -> str:
    configured = os.getenv("PAPERLENS_MODAL_BIN", "").strip()
    if configured:
        return configured if Path(configured).exists() or shutil.which(configured) else ""
    found = shutil.which("modal")
    if found:
        return found
    anaconda_modal = Path("/opt/anaconda3/bin/modal")
    return str(anaconda_modal) if anaconda_modal.exists() else ""


def _modal_timeout_seconds() -> float:
    raw = os.getenv("PAPERLENS_MODAL_GPU_TIMEOUT", "900")
    try:
        value = float(raw)
    except ValueError:
        value = 900.0
    return max(60.0, min(value, 1800.0))


def _trim_logs(stdout: str, stderr: str) -> list[str]:
    lines = []
    for stream in (stdout, stderr):
        for line in stream.splitlines():
            cleaned = line.strip()
            if cleaned:
                lines.append(cleaned)
    return lines[-40:]
