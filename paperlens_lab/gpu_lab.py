from __future__ import annotations

import ast
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .ingest import clean_text
from .mini_lab import build_evidence_rows
from .source_index import get_span_text, text_hash


MODAL_GPU_SCRIPT = Path(__file__).with_name("modal_gpu_lab_app.py")
PUBLIC_GPU_EXECUTION_FAILURE = (
    "The GPU run did not complete. Regenerate the sandbox files or choose a different paper-grounded direction."
)
PUBLIC_GPU_VALIDATION_FAILURE = "The GPU result could not be verified against the approved paper experiment contract."
PUBLIC_GPU_EMPTY_RESULT = "The GPU run completed but did not return measurable rows or metrics."
_INTERNAL_GPU_DETAIL_RE = re.compile(
    r"(?i)(traceback|runtimeerror|typeerror|valueerror|exception|modal cli|validation failed|service validation|schema|"
    r"internal server error|jsondecodeerror|syntaxerror|file \"|/tmp/|exited with|not json|"
    r"did not write|not callable|modal app=|paperlens-modal-gpu-probe|^cuda=|^gpu=|"
    r"token missing|authenticate client|api_key|modal\.com/docs|modal token|register an account|"
    r"token credentials|deserializationerror|modulenotfounderror|deserialize|_serialization\.py|"
    r"site-packages|module is not available|╭─ error|╰|│|❱)"
)


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
        "reproductionLevel": str(binding.get("reproductionLevel") or "probe"),
        "requestedReproductionLevel": str(binding.get("requestedReproductionLevel") or binding.get("reproductionLevel") or "probe"),
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
        location = f"line {exc.lineno}" if exc.lineno else "unknown line"
        if exc.offset:
            location += f", column {exc.offset}"
        text = (exc.text or "").strip()
        snippet = f": {text[:160]}" if text else ""
        return [f"GPU script syntax error: {exc.msg} at {location}{snippet}"]
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
    reasons.extend(_script_report_artifact_errors(tree))
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


def _script_report_artifact_errors(tree: ast.AST) -> list[str]:
    has_artifacts_key = False
    has_report_key = False
    has_visual_literal = False
    literal_text_parts = _report_html_literal_parts(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [_literal_string(key) for key in node.keys]
            if "artifacts" in keys:
                has_artifacts_key = True
            if "reportHtml" in keys or "report_html" in keys:
                has_report_key = True
        if _node_contains_visual_html_literal(node):
            has_visual_literal = True
    errors: list[str] = []
    if not has_artifacts_key or not has_report_key:
        errors.append("GPU script must return a model-authored artifacts.reportHtml report")
    if not has_visual_literal:
        errors.append("GPU script must author a self-contained visual report artifact such as inline SVG or figure")
    missing_categories = _missing_report_grounding_categories(" ".join(literal_text_parts))
    if missing_categories:
        errors.append(
            "GPU script reportHtml must include paper-grounded sections: "
            + ", ".join(missing_categories)
        )
    return errors


def _literal_string(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _node_contains_visual_html_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _contains_visual_html(node.value)
    if isinstance(node, ast.JoinedStr):
        return any(
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and _contains_visual_html(value.value)
            for value in node.values
        )
    return False


def _node_report_literal_parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [str(value.value) for value in node.values if isinstance(value, ast.Constant) and isinstance(value.value, str)]
    parts: list[str] = []
    for child in ast.iter_child_nodes(node):
        parts.extend(_node_report_literal_parts(child))
    return parts


def _report_html_literal_parts(tree: ast.AST) -> list[str]:
    report_names = {"report", "report_html", "reportHtml"}
    container_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and "report" in target.id.lower() for target in node.targets):
                report_names.update(target.id for target in node.targets if isinstance(target, ast.Name))
                container_names.update(_join_argument_names(node.value))
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if _literal_string(key) in {"reportHtml", "report_html"}:
                    container_names.update(_join_argument_names(value))

    parts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target_names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if target_names & (report_names | container_names):
                parts.extend(_node_report_literal_parts(node.value))
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if _literal_string(key) in {"reportHtml", "report_html"}:
                    parts.extend(_node_report_literal_parts(value))
    return parts


def _join_argument_names(node: ast.AST) -> set[str]:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and node.args
        and isinstance(node.args[0], ast.Name)
    ):
        return {node.args[0].id}
    return []


_REPORT_GROUNDING_CATEGORIES = {
    "paper claim": ("paper claim", "claim", "hypothesis", "paper-specific", "논문 주장", "주장", "가설"),
    "paper evidence": ("paper evidence", "source span", "evidence", "paper span", "근거", "증거", "인용"),
    "experiment setup": ("experiment setup", "setup", "code path", "dataset", "method", "실험 설계", "데이터셋", "방법"),
    "measured metrics": (
        "measured metric",
        "metrics",
        "metric",
        "accuracy",
        "loss",
        "error",
        "bleu",
        "latency",
        "throughput",
        "지표",
        "정확도",
        "손실",
        "오차",
    ),
    "claim comparison": (
        "claim comparison",
        "compared to the paper",
        "comparison",
        "verdict",
        "paper result",
        "주장 비교",
        "논문과 비교",
        "비교",
        "판정",
    ),
    "limitations": ("limitations", "limitation", "bounded", "not exact", "next step", "한계", "제한", "다음 단계"),
}


def _missing_report_grounding_categories(report_html: str) -> list[str]:
    text = clean_text(re.sub(r"<[^>]+>", " ", html.unescape(report_html))).lower()
    if not text:
        return list(_REPORT_GROUNDING_CATEGORIES)
    missing = []
    for category, markers in _REPORT_GROUNDING_CATEGORIES.items():
        if not any(marker.lower() in text for marker in markers):
            missing.append(category)
    return missing


def _contains_visual_html(value: str) -> bool:
    lowered = value.lower()
    return "<svg" in lowered or "<figure" in lowered


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
    claim_comparison = _public_gpu_claim_comparison(
        _normalized_gpu_claim_comparison(result, rows=rows, metrics=metrics)
    )
    artifacts = _normalized_gpu_artifacts(
        result,
        job=job,
        rows=rows,
        metrics=metrics,
        claim_comparison=claim_comparison,
        limitations=list(result.get("limitations") or []),
    )
    reasons.extend(_gpu_artifact_contract_errors(artifacts))
    return {
        "passed": not reasons,
        "reasons": _public_gpu_reasons(reasons),
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
        "reproductionLevel": job.get("reproductionLevel", "probe"),
        "requestedReproductionLevel": job.get("requestedReproductionLevel", job.get("reproductionLevel", "probe")),
        "validation": validation,
        "dataset": result.get("dataset") if isinstance(result.get("dataset"), dict) else {},
        "metrics": metrics,
        "rows": rows,
        "logs": _public_gpu_logs(list(result.get("logs") or []), passed=not reasons),
        "claimComparison": claim_comparison,
        "artifacts": artifacts,
        "limitations": _public_gpu_limitations(list(result.get("limitations") or []), passed=not reasons),
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


def _normalized_gpu_artifacts(
    result: dict[str, Any],
    *,
    job: dict[str, Any],
    rows: list[Any],
    metrics: dict[str, Any],
    claim_comparison: dict[str, Any],
    limitations: list[Any],
) -> dict[str, Any]:
    raw_artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    raw_report = (
        raw_artifacts.get("reportHtml")
        or raw_artifacts.get("report_html")
        or result.get("reportHtml")
        or result.get("report_html")
        or ""
    )
    raw_report_text = str(raw_report).strip() if raw_report else ""
    report_html = _sanitize_report_html(raw_report_text) if raw_report_text else ""
    missing_model_report = not bool(raw_report_text) or not bool(report_html)
    manifest = raw_artifacts.get("manifest") if isinstance(raw_artifacts.get("manifest"), dict) else {}
    if not manifest:
        manifest = {
            "paperId": job.get("paperId", ""),
            "spanId": job.get("spanId", ""),
            "candidateId": job.get("candidateId", ""),
            "codeHash": job.get("codeHash", ""),
            "evidenceHash": job.get("evidenceHash", ""),
            "reproductionLevel": job.get("reproductionLevel", "probe"),
            "requestedReproductionLevel": job.get("requestedReproductionLevel", job.get("reproductionLevel", "probe")),
        }
    artifact_metrics = raw_artifacts.get("metrics") if isinstance(raw_artifacts.get("metrics"), dict) else metrics
    files = raw_artifacts.get("files") if isinstance(raw_artifacts.get("files"), list) else []
    safe_files = []
    for item in files[:12]:
        if not isinstance(item, dict):
            continue
        safe_files.append(
            {
                "path": clean_text(str(item.get("path") or ""))[:180],
                "kind": clean_text(str(item.get("kind") or ""))[:80],
                "summary": clean_text(str(item.get("summary") or ""))[:500],
            }
        )
    return {
        "reportHtml": report_html,
        "reportTitle": clean_text(str(raw_artifacts.get("reportTitle") or raw_artifacts.get("report_title") or "GPU Experiment Report"))[:120],
        "generatedBy": "model",
        "reportStatus": "model_html" if report_html else "missing_model_html",
        "missingModelReport": missing_model_report,
        "manifest": manifest,
        "metrics": artifact_metrics,
        "files": safe_files,
        "sandbox": {
            "scriptsAllowed": False,
            "externalNetworkAllowed": False,
            "source": "sanitized-model-html" if report_html else "missing-model-html",
        },
    }


def _gpu_artifact_contract_errors(artifacts: dict[str, Any]) -> list[str]:
    report_html = str(artifacts.get("reportHtml") or "").strip()
    if not report_html:
        return ["Generated GPU script did not return a model-authored artifacts.reportHtml report"]
    if not _contains_visual_html(report_html):
        return ["Generated GPU script reportHtml did not include a model-authored inline SVG or figure"]
    missing_categories = _missing_report_grounding_categories(report_html)
    if missing_categories:
        return [
            "Generated GPU script reportHtml did not connect the experiment to the paper: "
            + ", ".join(missing_categories)
        ]
    return []


def _short_value(value: Any, max_chars: int = 160) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return clean_text(text)[:max_chars]


class _ReportSanitizer(HTMLParser):
    allowed_tags = {
        "html",
        "head",
        "body",
        "main",
        "section",
        "article",
        "header",
        "footer",
        "div",
        "span",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "ul",
        "ol",
        "li",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "code",
        "pre",
        "strong",
        "em",
        "small",
        "figure",
        "figcaption",
        "svg",
        "g",
        "path",
        "rect",
        "circle",
        "ellipse",
        "line",
        "polyline",
        "polygon",
        "text",
        "br",
        "hr",
        "style",
    }
    allowed_attrs = {
        "aria-label",
        "class",
        "colspan",
        "cx",
        "cy",
        "d",
        "dominant-baseline",
        "fill",
        "font-size",
        "font-weight",
        "height",
        "points",
        "r",
        "role",
        "rowspan",
        "rx",
        "ry",
        "stroke",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-width",
        "style",
        "text-anchor",
        "title",
        "viewbox",
        "width",
        "x",
        "x1",
        "x2",
        "y",
        "y1",
        "y2",
    }
    void_tags = {"br", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag not in self.allowed_tags:
            return
        safe_attrs = []
        for key, value in attrs:
            key = key.lower()
            if key.startswith("on") or key not in self.allowed_attrs or value is None:
                continue
            cleaned = _sanitize_attr_value(value)
            if cleaned:
                safe_attrs.append(f'{key}="{html.escape(cleaned, quote=True)}"')
        suffix = (" " + " ".join(safe_attrs)) if safe_attrs else ""
        self.parts.append(f"<{tag}{suffix}>")
        if tag not in self.void_tags:
            self.tag_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.allowed_tags and tag not in self.void_tags:
            self.parts.append(f"</{tag}>")
            if tag in self.tag_stack:
                self.tag_stack = self.tag_stack[: len(self.tag_stack) - 1 - self.tag_stack[::-1].index(tag)]

    def handle_data(self, data: str) -> None:
        self.parts.append(html.escape(_sanitize_css(data) if self.tag_stack and self.tag_stack[-1] == "style" else data))

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")


def _sanitize_report_html(raw_html: str) -> str:
    raw_html = raw_html[:25_000]
    raw_html = re.sub(r"(?is)<(script|iframe|object|embed|link|meta|base|form|input|button|textarea|select)[^>]*>.*?</\1>", "", raw_html)
    raw_html = re.sub(r"(?is)<(script|iframe|object|embed|link|meta|base|form|input|button|textarea|select)[^>]*?/?>", "", raw_html)
    sanitizer = _ReportSanitizer()
    sanitizer.feed(raw_html)
    cleaned = "".join(sanitizer.parts).strip()
    return cleaned


def _sanitize_attr_value(value: str) -> str:
    lowered = value.lower()
    if "javascript:" in lowered or "data:" in lowered or "http:" in lowered or "https:" in lowered:
        return ""
    return _sanitize_css(value)


def _sanitize_css(value: str) -> str:
    value = re.sub(r"(?is)@import[^;]+;?", "", value)
    value = re.sub(r"(?is)url\s*\([^)]*\)", "", value)
    value = value.replace("<", "").replace(">", "")
    return value[:1200]


def _failed_gpu_result(job: dict[str, Any], reason: str, *, logs: list[str] | None = None) -> dict[str, Any]:
    public_reason = _public_gpu_reason(reason)
    claim_comparison = {"verdict": "failed_to_execute", "limitations": [public_reason]}
    return {
        "passed": False,
        "reasons": [public_reason],
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
        "reproductionLevel": job.get("reproductionLevel", "probe"),
        "requestedReproductionLevel": job.get("requestedReproductionLevel", job.get("reproductionLevel", "probe")),
        "validation": {},
        "dataset": {},
        "metrics": {},
        "rows": [],
        "logs": _public_gpu_logs(logs or [], passed=False),
        "claimComparison": claim_comparison,
        "artifacts": _normalized_gpu_artifacts(
            {},
            job=job,
            rows=[],
            metrics={},
            claim_comparison=claim_comparison,
            limitations=[public_reason],
        ),
        "limitations": [public_reason],
        "durationMs": 0,
    }


def _public_gpu_reasons(reasons: list[Any]) -> list[str]:
    public = [_public_gpu_reason(str(reason)) for reason in reasons if str(reason).strip()]
    return list(dict.fromkeys(public))


def _public_gpu_reason(reason: str) -> str:
    cleaned = clean_text(reason)
    if not cleaned:
        return PUBLIC_GPU_EXECUTION_FAILURE
    lower = cleaned.lower()
    if "no rows or metrics" in lower:
        return PUBLIC_GPU_EMPTY_RESULT
    if cleaned.startswith("GPU probe validation failed") or cleaned.startswith("Generated GPU script"):
        return PUBLIC_GPU_VALIDATION_FAILURE
    if _INTERNAL_GPU_DETAIL_RE.search(cleaned):
        return PUBLIC_GPU_EXECUTION_FAILURE
    return cleaned[:220]


def _public_gpu_logs(logs: list[Any], *, passed: bool) -> list[str]:
    public = []
    for item in logs:
        cleaned = clean_text(str(item))
        if not cleaned or _INTERNAL_GPU_DETAIL_RE.search(cleaned):
            continue
        public.append(cleaned[:240])
    if not passed and not public:
        public.append(PUBLIC_GPU_EXECUTION_FAILURE)
    return public[-20:]


def _public_gpu_limitations(limitations: list[Any], *, passed: bool) -> list[str]:
    public = []
    for item in limitations:
        cleaned = clean_text(str(item))
        if not cleaned:
            continue
        public.append(_public_gpu_reason(cleaned))
    if not passed and not public:
        public.append(PUBLIC_GPU_EXECUTION_FAILURE)
    return list(dict.fromkeys(public))


def _public_gpu_claim_comparison(claim: dict[str, Any]) -> dict[str, Any]:
    public = dict(claim)
    if str(public.get("verdict") or "") == "failed_to_execute":
        public["limitations"] = [PUBLIC_GPU_EXECUTION_FAILURE]
        public.pop("reasons", None)
        return public
    if isinstance(public.get("limitations"), list):
        public["limitations"] = _public_gpu_limitations(public["limitations"], passed=True)
    if isinstance(public.get("reasons"), list):
        public["reasons"] = _public_gpu_reasons(public["reasons"])
    return public


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
