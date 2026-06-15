from __future__ import annotations

import ast
import base64
from collections import Counter
from dataclasses import dataclass
import json
import re
import subprocess
import sys
import textwrap
from typing import Any


HEAVY_EXPERIMENT_TERMS = (
    "wmt14",
    "full benchmark",
    "train for",
    "epochs",
    "epoch",
    "pytorch",
    "tensorflow",
    "sacrebleu",
    "faiss",
    "download and load",
    "fine-tune",
    "large dataset",
    "cuda",
    "p100",
    "v100",
    "a100",
    "h100",
    "tpu",
    "multi-gpu",
    "multi gpu",
    "multi-day",
    "multi day",
    "full training run",
    "end-to-end training",
    "distributed training",
    "large-scale",
    "large scale",
)

HEAVY_EXPERIMENT_BLOCKERS = (
    "wmt14",
    "full benchmark",
    "train for",
    "epochs",
    "epoch",
    "fine-tune",
    "cuda",
    "p100",
    "v100",
    "a100",
    "h100",
    "tpu",
    "multi-gpu",
    "multi gpu",
    "multi-day",
    "multi day",
    "full training run",
    "end-to-end training",
    "distributed training",
)

SAFE_STARTER_IMPORTS = {"json", "re", "math"}
SAFE_STARTER_IMPORTS_TEXT = "json, re, or math"


@dataclass
class EvalResult:
    name: str
    passed: bool
    reasons: list[str]


@dataclass(frozen=True)
class FailureRecord:
    task: str
    label: str
    scenario_id: str
    model: str = "unknown"
    severity: str = "medium"
    root_cause: str = "unknown"
    fix_attempted: bool = False


def evaluate_translation(
    source: str,
    translation_data: dict[str, Any],
    expected_span_ids: list[str] | None = None,
) -> EvalResult:
    reasons: list[str] = []
    translations = translation_data.get("translations", [])
    if not translations:
        reasons.append("missing translations")
    if expected_span_ids is not None:
        ids = [item.get("span_id") for item in translations if isinstance(item, dict)]
        for span_id in expected_span_ids:
            if ids.count(span_id) != 1:
                reasons.append(f"span {span_id} does not map to exactly one translation")
    joined = " ".join(str(item.get("translation", "")) for item in translations if isinstance(item, dict))
    for number in _numbers(source):
        if number not in joined:
            reasons.append(f"changed or dropped number {number}")
    for marker in _citation_markers(source):
        if not _marker_preserved(marker, joined):
            reasons.append(f"changed or dropped citation/table marker {marker}")
    for term in _technical_terms(source):
        if not _term_preserved(term, joined):
            reasons.append(f"changed or dropped technical term {term}")
    if _has_negation_or_limit(source) and not _has_negation_or_limit(joined):
        reasons.append("lost negation, limitation, or result qualifier")
    if any("translation" not in item or not item.get("translation") for item in translations):
        reasons.append("translation item missing translation field")
    if _adds_unsupported_strength(source, joined):
        reasons.append("translation adds unsupported strong claim")
    return EvalResult("translation_fidelity", not reasons, reasons)


def evaluate_grounded_qa(
    answer_data: dict[str, Any],
    expected_span_id: str,
    source_evidence: dict[str, str] | None = None,
    require_needs_more_context: bool = False,
) -> EvalResult:
    reasons: list[str] = []
    answer = str(answer_data.get("answer", ""))
    evidence = answer_data.get("evidence", [])
    ids = [item.get("source_id") for item in evidence if isinstance(item, dict)]
    if not answer:
        reasons.append("missing answer")
    if expected_span_id not in ids:
        reasons.append("selected span is not cited")
    if answer_data.get("confidence") not in {"high", "medium", "low"}:
        reasons.append("missing confidence label")
    if require_needs_more_context:
        if answer_data.get("needs_more_context") is not True:
            reasons.append("should ask for more context")
        if answer_data.get("confidence") == "high":
            reasons.append("unsupported question should not have high confidence")
    if source_evidence:
        known_ids = {str(source_id) for source_id in source_evidence}
        for source_id in ids:
            if str(source_id) not in known_ids:
                reasons.append(f"answer cites unknown evidence {source_id}")
        for item in evidence:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id")
            quote = str(item.get("quote", "")).strip()
            source_text = source_evidence.get(str(source_id), "")
            if quote and source_text and not source_contains_quote(source_text, quote):
                reasons.append(f"quote for {source_id} is not in source evidence")
    joined_evidence = " ".join(source_evidence.values()) if source_evidence else ""
    if _adds_unsupported_strength(joined_evidence, answer):
        unsupported = _flatten_text(answer_data.get("unsupported_assumptions", ""))
        if not (answer_data.get("needs_more_context") and _mentions_strong_marker(unsupported)):
            reasons.append("answer adds unsupported strong claim")
    return EvalResult("grounded_qa", not reasons, reasons)


def evaluate_experiment_spec(spec: dict[str, Any]) -> EvalResult:
    reasons: list[str] = []
    required = ["research_question", "mini_lab_goal", "dataset", "baseline", "metric", "steps"]
    for key in required:
        if not spec.get(key):
            reasons.append(f"missing {key}")
    if len(spec.get("steps", [])) < 3:
        reasons.append("mini-lab needs at least three steps")
    if not spec.get("failure_condition"):
        reasons.append("missing failure condition")
    dataset_text = _flatten_text(spec.get("dataset", ""))
    spec_text = _flatten_text(spec)
    dataset = spec.get("dataset")
    forbidden_dataset_terms = (
        "toy",
        "hand-built",
        "built-in example",
        "sample dataset",
        "synthetic",
        "simulated",
        "pseudo",
        "randomly initialized",
        "random initialization",
        "random vector",
        "random vectors",
        "random-vector",
        "random-vectors",
        "controlled sequence",
        "controlled-sequence",
        "small sequence",
        "small-sequence",
        "generated sequence",
        "generated-sequence",
        "generated inputs",
        "generated-inputs",
    )
    if isinstance(dataset, dict) and "fallback" in dataset:
        reasons.append("mini-lab dataset must not define a fallback input source")
    if any(term in dataset_text.lower() for term in forbidden_dataset_terms):
        reasons.append("mini-lab dataset must use indexed paper evidence, not synthetic/internal examples")
    if re.search(r"(?<![a-z0-9])toy(?![a-z0-9])", spec_text.lower()):
        reasons.append("mini-lab spec must not describe the service experiment as toy")
    if re.search(r"(?<![a-z0-9])(synthetic|simulated|pseudo)(?![a-z0-9])", spec_text.lower()):
        reasons.append("mini-lab spec must use indexed paper evidence rows, not synthetic or simulated examples")
    if re.search(
        r"(?<![a-z0-9])(randomly initialized|random[- ]vectors?|controlled[- ]sequence|small[- ]sequence|generated[- ](?:sequence|inputs?))(?![a-z0-9])",
        spec_text.lower(),
    ):
        reasons.append("mini-lab spec must use indexed paper evidence rows, not generated vector or sequence inputs")
    if not any(term in dataset_text.lower() for term in ("indexed", "source", "evidence", "paperlens", "paper")):
        reasons.append("mini-lab dataset must name the indexed paper evidence source")
    if any(term in spec_text.lower() for term in ("8xa100", "a100", "gpu cluster", "proprietary dataset")):
        if not any(term in dataset_text.lower() for term in ("indexed", "source", "evidence", "paperlens")):
            reasons.append("large or proprietary setup must be reduced to indexed paper evidence")
    heavy_terms = experiment_heavy_terms(spec_text)
    if heavy_terms:
        if not any(term in dataset_text.lower() for term in ("indexed", "source", "evidence", "paperlens")):
            reasons.append("mini-lab should use indexed paper evidence instead of heavy training or full benchmarks")
        if any(term in HEAVY_EXPERIMENT_BLOCKERS for term in heavy_terms):
            reasons.append("mini-lab is too heavy for a service demo source-bound run")
    if "metric" in spec and spec.get("metric") and spec.get("failure_condition"):
        metric_head = _normalize_metric_token(str(spec["metric"]).split(",")[0].split()[0])
        failure_text = _normalize_metric_token(str(spec["failure_condition"]))
        if metric_head and metric_head not in failure_text and "metric" not in failure_text:
            reasons.append("failure condition should reference the metric")
    if spec.get("ablation"):
        ablation = str(spec["ablation"]).lower()
        if not any(term in ablation for term in ("remove", "disable", "only", "one", "without", "isolate")):
            reasons.append("ablation should isolate one variable")
    return EvalResult("experiment_spec", not reasons, reasons)


def experiment_heavy_terms(text: str) -> list[str]:
    lowered = text.lower()
    return [
        term
        for term in HEAVY_EXPERIMENT_TERMS
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered)
    ]


def evaluate_starter_code(
    code: str,
    *,
    evidence_rows: list[dict[str, Any]] | None = None,
    require_evidence_rows: bool = False,
) -> EvalResult:
    reasons: list[str] = []
    execution = run_starter_code(
        code,
        evidence_rows=evidence_rows,
        require_evidence_rows=require_evidence_rows,
    )
    if not execution["passed"]:
        return EvalResult("starter_code_source_run", False, list(execution["reasons"]))
    return EvalResult("starter_code_source_run", True, reasons)


def evaluate_starter_grounding(code: str, selected_span: str) -> EvalResult:
    span = str(selected_span or "").strip().lower()
    if not span:
        return EvalResult("starter_code_grounding", True, [])

    lowered = code.lower()
    reasons: list[str] = []
    mechanism_terms = [
        term
        for term in (
            "attention",
            "transformer",
            "recurrence",
            "convolution",
            "retrieval",
            "rerank",
            "adapter",
            "low-rank",
            "lora",
        )
        if term in span
    ]
    if mechanism_terms and not any(term in lowered for term in mechanism_terms):
        reasons.append("starter code omits the selected span mechanism terms")

    generic_placeholder_terms = (
        "capital of france",
        "who wrote hamlet",
        "weather like today",
        "president of the united states",
        "best way to learn python",
        "meaning of life",
        "joe biden",
    )
    if any(term in lowered for term in generic_placeholder_terms) and not any(
        term in span for term in generic_placeholder_terms
    ):
        reasons.append("starter code uses unrelated generic examples")

    keyword_only_markers = (
        "paper_span.split()",
        "example.split()",
        '"paper-related" if',
        "len(hits) >= 1",
        "most frequent word",
    )
    if any(marker in lowered for marker in keyword_only_markers) and "candidates" not in lowered:
        reasons.append("starter code still relies on keyword-only matching")

    source_bound_run = "evidence_rows" in lowered and "source_id" in lowered

    if "def paper_inspired" in lowered and "candidates" not in lowered and "query" not in lowered and not source_bound_run:
        reasons.append("starter code lacks structured query/context examples")

    baseline_match = re.search(r"def\s+baseline\s*\([^)]*\):(?P<body>.*?)(?:\ndef\s+|\Z)", lowered, re.DOTALL)
    baseline_body = baseline_match.group("body") if baseline_match else lowered
    has_first_candidate_return = re.search(r"return\s+candidates\s*\[\s*0\s*\]", baseline_body) or re.search(
        r"""return\s+\w+\s*\[\s*['"]candidates['"]\s*\]\s*\[\s*0\s*\]""",
        baseline_body,
    )
    baseline_reads_input = any(
        marker in baseline_body
        for marker in (
            '["context"]',
            "['context']",
            '["query"]',
            "['query']",
            "for candidate in",
            ".find(",
        )
    )
    if has_first_candidate_return and not baseline_reads_input:
        reasons.append("starter baseline is a trivial first-candidate selector")

    if re.search(r"return\s+scores\s*(?:$|#)", lowered) or re.search(r"return\s+score_map\s*(?:$|#)", lowered):
        reasons.append("paper_inspired returns raw scores instead of a concrete paper-grounded prediction")

    if "attention" in span and not source_bound_run:
        example_count = lowered.count('"gold":') + lowered.count("'gold':")
        if example_count < 3:
            reasons.append("attention-style starter uses too few structured examples")
        has_explicit_mode = any(
            marker in lowered
            for marker in ('"mode":', "'mode':", 'example.get("mode"', "example.get('mode'")
        )
        if not has_explicit_mode:
            reasons.append("attention-style starter lacks explicit contrast modes")

    return EvalResult("starter_code_grounding", not reasons, reasons)


def run_starter_code(
    code: str,
    *,
    evidence_rows: list[dict[str, Any]] | None = None,
    require_evidence_rows: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not code.strip():
        return {"passed": False, "reasons": ["missing starter code"], "rows": []}
    if require_evidence_rows and not evidence_rows:
        return {"passed": False, "reasons": ["mini-lab requires paper evidence rows"], "rows": []}
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"passed": False, "reasons": [f"starter code syntax error: {exc.msg}"], "rows": []}
    safety_reasons = _starter_code_safety_reasons(tree)
    if safety_reasons:
        return {"passed": False, "reasons": safety_reasons, "rows": []}

    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for required in ("baseline", "paper_inspired", "score", "run"):
        if required not in function_names:
            reasons.append(f"starter code missing {required}()")

    execution = _run_starter_code_subprocess(
        code,
        evidence_rows=evidence_rows,
        require_evidence_rows=require_evidence_rows,
    )
    reasons.extend(execution["reasons"])
    rows = execution["rows"]
    reasons.extend(_starter_row_contract_reasons(rows))
    reasons.extend(_starter_evidence_binding_reasons(rows, evidence_rows or []))
    return {"passed": not reasons, "reasons": reasons, "rows": rows if isinstance(rows, list) else []}


def _starter_row_contract_reasons(rows: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(rows, list) or not rows:
        return ["starter run() did not return non-empty rows"]
    if len(rows) < 2:
        reasons.append("starter run() should return at least two rows so the mini-lab has a contrast case")
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            reasons.append(f"starter row {idx} is not a dict")
            continue
        for key in ("baseline_score", "prototype_score", "metric", "failure_condition"):
            if key not in row:
                reasons.append(f"starter row {idx} missing {key}")
        baseline = row.get("baseline_score")
        prototype = row.get("prototype_score")
        if not _is_finite_number(baseline):
            reasons.append(f"starter row {idx} baseline_score must be numeric")
        if not _is_finite_number(prototype):
            reasons.append(f"starter row {idx} prototype_score must be numeric")
        if not str(row.get("metric", "")).strip():
            reasons.append(f"starter row {idx} metric must be non-empty")
        failure_condition = row.get("failure_condition")
        if not isinstance(failure_condition, bool):
            reasons.append(f"starter row {idx} failure_condition must be a boolean")
        if _is_finite_number(baseline) and _is_finite_number(prototype):
            expected_failure = prototype <= baseline
            if isinstance(failure_condition, bool) and failure_condition != expected_failure:
                reasons.append(
                    f"starter row {idx} failure_condition must match prototype_score <= baseline_score"
                )
    return reasons


def _starter_evidence_binding_reasons(rows: Any, evidence_rows: list[dict[str, Any]]) -> list[str]:
    if not evidence_rows:
        return []
    allowed = {}
    selected_source_ids = {
        str(row.get("source_id") or "")
        for row in evidence_rows
        if isinstance(row, dict) and (str(row.get("label") or "") == "selected" or row.get("gold") is True)
    }
    selected_seen = False
    for row in evidence_rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or "")
        if not source_id:
            continue
        allowed[source_id] = str(row.get("text_hash") or "")
    reasons: list[str] = []
    if not isinstance(rows, list):
        return reasons
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or "")
        if not source_id:
            reasons.append(f"starter row {idx} missing source_id")
        elif source_id not in allowed:
            reasons.append(f"starter row {idx} source_id is outside supplied paper evidence")
        else:
            if source_id in selected_source_ids:
                selected_seen = True
            expected_hash = allowed.get(source_id, "")
            row_hash = str(row.get("text_hash") or "")
            if not row_hash:
                reasons.append(f"starter row {idx} missing text_hash")
            elif expected_hash and row_hash != expected_hash:
                reasons.append(f"starter row {idx} text_hash does not match supplied paper evidence")
    if selected_source_ids and not selected_seen:
        reasons.append("starter rows must include the selected paper evidence row")
    return reasons


def _is_finite_number(value: Any) -> bool:
    return type(value) in {int, float} and value == value and value not in {float("inf"), float("-inf")}


def _run_starter_code_subprocess(
    code: str,
    *,
    evidence_rows: list[dict[str, Any]] | None = None,
    require_evidence_rows: bool = False,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    encoded_evidence = base64.b64encode(
        json.dumps(evidence_rows or [], ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    runner = textwrap.dedent(
        """
        import ast
        import base64
        import json
        import math
        import re
        import sys

        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        except Exception:
            pass

        def safe_import(name, globals_=None, locals_=None, fromlist=(), level=0):
            if level == 0 and name in {"json", "re", "math"}:
                return {"json": json, "re": re, "math": math}[name]
            raise ImportError(f"starter code may only import {SAFE_STARTER_IMPORTS_TEXT}, not {name}")

        safe_builtins = {
            "all": all,
            "any": any,
            "__import__": safe_import,
            "abs": abs,
            "bool": bool,
            "chr": chr,
            "dict": dict,
            "enumerate": enumerate,
            "Exception": Exception,
            "float": float,
            "hash": hash,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "ord": ord,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "ValueError": ValueError,
            "zip": zip,
        }

        code = base64.b64decode(sys.argv[1]).decode("utf-8")
        evidence_rows = json.loads(base64.b64decode(sys.argv[2]).decode("utf-8"))
        require_evidence_rows = sys.argv[3] == "1"
        namespace = {
            "__name__": "paperlens_starter_source_run",
            "__builtins__": safe_builtins,
            "PAPERLENS_EVIDENCE_ROWS": evidence_rows,
        }
        result = {"passed": False, "reasons": [], "rows": []}
        try:
            tree = ast.parse(code)
            exec(compile(tree, "<paperlens_starter>", "exec"), namespace)
            run = namespace.get("run")
            if not callable(run):
                result["reasons"].append("starter code missing runnable run()")
            else:
                if evidence_rows:
                    try:
                        rows = run(evidence_rows)
                    except TypeError as exc:
                        if require_evidence_rows:
                            result["reasons"].append(f"starter run() must accept paper evidence rows: {exc}")
                            rows = []
                        else:
                            rows = run()
                elif require_evidence_rows:
                    result["reasons"].append("mini-lab requires paper evidence rows")
                    rows = []
                else:
                    rows = run()
                result["rows"] = rows if isinstance(rows, list) else []
                result["passed"] = isinstance(rows, list)
        except Exception as exc:
            result["reasons"].append(f"starter subprocess failed: {type(exc).__name__}: {exc}")
        print(json.dumps(result, ensure_ascii=False))
        """
    ).strip()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                runner,
                encoded,
                encoded_evidence,
                "1" if require_evidence_rows else "0",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "reasons": ["starter subprocess timed out"], "rows": []}
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        return {
            "passed": False,
            "reasons": [f"starter subprocess exited with {completed.returncode}: {detail[-1] if detail else 'no output'}"],
            "rows": [],
        }
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"passed": False, "reasons": ["starter subprocess returned invalid JSON"], "rows": []}
    return {
        "passed": bool(result.get("passed")),
        "reasons": list(result.get("reasons") or []),
        "rows": result.get("rows") if isinstance(result.get("rows"), list) else [],
    }


def _starter_code_safety_reasons(tree: ast.AST) -> list[str]:
    reasons: list[str] = []
    forbidden_names = {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names]
            if any(name not in SAFE_STARTER_IMPORTS for name in names):
                reasons.append(f"starter code may only import {SAFE_STARTER_IMPORTS_TEXT}")
        elif isinstance(node, ast.Name) and (
            node.id in forbidden_names or (node.id.startswith("__") and node.id != "__name__")
        ):
            reasons.append(f"starter code uses unsafe name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            reasons.append(f"starter code uses unsafe attribute {node.attr}")
    return sorted(set(reasons))


def evaluate_growth_ideas(
    data: dict[str, Any],
    known_evidence_ids: set[str] | None = None,
    require_multiple_sources: bool = False,
) -> EvalResult:
    reasons: list[str] = []
    ideas = data.get("ideas", [])
    if not ideas:
        reasons.append("missing ideas")
    has_multi_source_idea = False
    for idx, idea in enumerate(ideas, start=1):
        idea_text = _flatten_text(idea)
        if any(term in idea_text.lower() for term in ("toy", "toy setup", "toy dataset", "toy scale")):
            reasons.append(f"idea {idx} describes the source-evidence mini-lab as toy")
        if re.search(r"(?<![a-z0-9])(synthetic|simulated|pseudo)(?![a-z0-9])", idea_text.lower()):
            reasons.append(f"idea {idx} invents synthetic or simulated follow-up inputs")
        if re.search(
            r"(?<![a-z0-9])(randomly initialized|random[- ]vectors?|controlled[- ]sequence|small[- ]sequence|generated[- ](?:sequence|inputs?))(?![a-z0-9])",
            idea_text.lower(),
        ):
            reasons.append(f"idea {idx} invents generated vector or sequence follow-up inputs")
        evidence_ids = idea.get("source_evidence") or []
        if not evidence_ids:
            reasons.append(f"idea {idx} missing evidence")
        if known_evidence_ids:
            missing = [source_id for source_id in evidence_ids if source_id not in known_evidence_ids]
            if missing:
                reasons.append(f"idea {idx} cites unknown evidence {', '.join(missing)}")
        if len(set(evidence_ids)) >= 2:
            has_multi_source_idea = True
        if not idea.get("testable_next_step"):
            reasons.append(f"idea {idx} missing testable next step")
        if not idea.get("risk"):
            reasons.append(f"idea {idx} missing risk")
        if _looks_like_restatement(str(idea.get("idea", "")), str(idea.get("testable_next_step", ""))):
            reasons.append(f"idea {idx} restates the test instead of adding a direction")
    if require_multiple_sources and not has_multi_source_idea:
        reasons.append("growth mode should combine at least two evidence sources")
    return EvalResult("growth_ideas", not reasons, reasons)


def fine_tuning_gate(failures: list[str | FailureRecord]) -> dict[str, Any]:
    if failures and all(isinstance(failure, FailureRecord) for failure in failures):
        return _fine_tuning_gate_records([failure for failure in failures if isinstance(failure, FailureRecord)])
    labels = [failure if isinstance(failure, str) else failure.label for failure in failures]
    counts = {failure: labels.count(failure) for failure in set(labels)}
    repeated = sorted(name for name, count in counts.items() if count >= 3)
    if not repeated:
        return {
            "recommendation": "no",
            "reason": "Fix prompt, retrieval, parsing, or scenario coverage before fine-tuning.",
            "repeated_failures": [],
        }
    return {
        "recommendation": "maybe",
        "reason": "Repeated model-output failures survived the same task boundary and may justify a bounded task-specific fine-tune.",
        "repeated_failures": repeated,
    }


def _fine_tuning_gate_records(failures: list[FailureRecord]) -> dict[str, Any]:
    trainable_causes = {"schema", "style", "terminology", "model_capability"}
    eligible = [
        failure
        for failure in failures
        if failure.fix_attempted and failure.root_cause in trainable_causes and failure.severity in {"medium", "high"}
    ]
    grouped = Counter((failure.task, failure.label, failure.model) for failure in eligible)
    repeated = [
        {"task": task, "label": label, "model": model, "count": count}
        for (task, label, model), count in sorted(grouped.items())
        if count >= 3
    ]
    if repeated:
        return {
            "recommendation": "maybe",
            "reason": "Repeated task-specific failures remain after prompt/RAG/parser fixes; prepare a bounded SFT/LoRA probe.",
            "repeated_failures": repeated,
        }
    untried = [
        failure
        for failure in failures
        if failure.root_cause in trainable_causes and not failure.fix_attempted
    ]
    return {
        "recommendation": "no",
        "reason": (
            "Try prompt, retrieval, parser, or rubric fixes before fine-tuning."
            if untried
            else "Failures do not yet point to a trainable model-style or terminology gap."
        ),
        "repeated_failures": [],
    }


def _numbers(text: str) -> list[str]:
    import re

    return re.findall(r"\b\d+(?:\.\d+)?%?\b", text)


def _citation_markers(text: str) -> list[str]:
    import re

    markers = re.findall(r"\[[0-9,\s-]+\]", text)
    markers.extend(re.findall(r"\b(?:Table|Figure|Fig\.)\s+\d+\b", text))
    return markers


def _marker_preserved(marker: str, output: str) -> bool:
    if marker in output:
        return True
    match = re.fullmatch(r"(Table|Figure|Fig\.)\s+(\d+)", marker)
    if not match:
        return False
    label, number = match.groups()
    if label == "Table":
        return bool(re.search(rf"(?:Table|표)\s*{re.escape(number)}(?!\d)", output, re.IGNORECASE))
    return bool(re.search(rf"(?:Figure|Fig\.|그림)\s*{re.escape(number)}(?!\d)", output, re.IGNORECASE))


def source_contains_quote(source_text: str, quote: str) -> bool:
    if quote in source_text:
        return True
    return _quote_match_text(quote) in _quote_match_text(source_text)


def _quote_match_text(text: str) -> str:
    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r",(?=[A-Za-z0-9])", ", ", text)
    return text.strip().casefold()


def _technical_terms(text: str) -> list[str]:
    import re

    patterns = [
        r"\b[A-Za-z]+-\d+(?:\.\d+)?-[A-Za-z0-9]+\b",
        r"\b[A-Z]{2,}[A-Za-z0-9-]*\b",
        r"\bp\s*<\s*0\.\d+\b",
    ]
    terms: list[str] = []
    for pattern in patterns:
        terms.extend(re.findall(pattern, text))
    return list(dict.fromkeys(terms))


def _has_negation_or_limit(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "does not",
            "do not",
            "no ",
            "not ",
            "fails",
            "failed",
            "weak",
            "limitation",
            "however",
            "only",
            "disappears",
            "않",
            "없",
            "아니",
            "만",
            "실패",
            "제한",
            "그러나",
            "하지만",
            "약함",
            "사라",
        )
    )


def _adds_unsupported_strength(source: str, output: str) -> bool:
    source_lower = source.lower()
    output_lower = output.lower()
    return any(
        marker in output_lower and not _strong_marker_supported(source_lower, marker)
        for marker in _strong_markers()
    )


def _mentions_strong_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _strong_markers())


def _strong_markers() -> tuple[str, ...]:
    return ("state-of-the-art", "sota", "proves", "proven", "guarantees", "입증", "증명", "최고", "완벽")


def _strong_marker_supported(source_lower: str, marker: str) -> bool:
    if marker in source_lower:
        return True
    equivalents = {
        "sota": ("state-of-the-art",),
        "state-of-the-art": ("sota",),
        "입증": ("prove", "proves", "proven", "demonstrate", "demonstrates", "demonstrated"),
        "증명": ("prove", "proves", "proven", "demonstrate", "demonstrates", "demonstrated"),
        "최고": ("best", "state-of-the-art", "sota", "top-performing", "top performing"),
        "완벽": ("perfect", "perfectly", "guarantee", "guarantees"),
    }
    return any(_contains_phrase(source_lower, term) for term in equivalents.get(marker, ()))


def _contains_phrase(text: str, phrase: str) -> bool:
    if phrase.isascii() and phrase.replace("-", "").replace(" ", "").isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None
    return phrase in text


def _term_preserved(term: str, output: str) -> bool:
    if term in output:
        return True
    term_lower = term.lower()
    output_lower = output.lower()
    if term_lower in output_lower:
        return True
    equivalents = {
        "qa": ("question answering", "질문 응답", "질의응답", "질문응답", "문답"),
    }
    if any(alias in output_lower for alias in equivalents.get(term_lower, ())):
        return True
    if term_lower.endswith("s") and len(term_lower) > 3 and term_lower[:-1] in output_lower:
        return True
    return False


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def _looks_like_restatement(idea: str, next_step: str) -> bool:
    idea_words = {word for word in idea.lower().split() if len(word) > 4}
    next_words = {word for word in next_step.lower().split() if len(word) > 4}
    if not idea_words or not next_words:
        return False
    overlap = len(idea_words & next_words) / max(1, len(idea_words))
    return overlap > 0.85


def _normalize_metric_token(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")
