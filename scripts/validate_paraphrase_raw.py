"""Audit the frozen paraphrase records without API calls or code execution.

This validator is specific to the v1.1.0-rnr frozen dataset. It parses generated
Python with ast.parse, but never executes generated programs. Known schedule
exceptions are reported, not rewritten or treated as duplicate sample keys.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from parse_ast import build_metrics

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ("ast_depth", "branch_count", "loop_count", "function_count", "control_flow_ratio")
KEYS = ("task_id", "model_name", "prompt_name", "repeat_idx")
ARCHIVE_SHA256 = "cfd4590dc353511f8b2a239efebcfdafaa19f189caeac43dd729ed9f39761341"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "outputs/paraphrase/raw")
    parser.add_argument("--archive", type=Path, help="Also verify the release archive SHA-256")
    parser.add_argument("--report", type=Path, help="Optional JSON audit report")
    args = parser.parse_args()
    if args.archive:
        digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
        require(digest == ARCHIVE_SHA256, "Release archive SHA-256 mismatch")

    problems = json.loads((ROOT / "datasets/problems.json").read_text(encoding="utf-8"))
    problem_map = {p["task_id"]: p for p in problems}
    models = yaml.safe_load((ROOT / "configs/paraphrase_models.yaml").read_text(encoding="utf-8"))["models"]
    model_map = {m["name"]: m for m in models}
    prompts = yaml.safe_load((ROOT / "configs/paraphrase_prompts.yaml").read_text(encoding="utf-8"))
    schedule = [(p["task_id"], m["name"], prompt, rep)
                for p in problems for m in models for prompt in prompts for rep in range(1, 6)]
    random.Random(42).shuffle(schedule)
    expected_indices = {key: i for i, key in enumerate(schedule, 1)}
    with (ROOT / "results/paraphrase_structural/metrics_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    saved = {r["file"]: r for r in rows}
    require(len(rows) == len(saved) == 12300, "Saved metric rows or filenames are not unique")
    files = sorted(args.raw_dir.glob("*.json"))
    require(len(files) == 12300, f"Expected 12300 raw JSON files, found {len(files)}")
    require({p.name for p in files} == set(saved), "Raw and metric filenames differ")

    keys = Counter()
    metadata = defaultdict(lambda: {"dates": [], "identifiers": Counter(), "finish_reasons": Counter()})
    schedule_indices = Counter()
    exceptions = []
    main_order = []
    max_errors = dict.fromkeys(FEATURES, 0.0)
    ast_ok = 0
    for path in files:
        d = json.loads(path.read_text(encoding="utf-8"))
        key = tuple(d[k] for k in KEYS)
        keys[key] += 1
        require(key in expected_indices, f"Unexpected sample key: {key}")
        require(d["status"] == "success" and d["schedule_seed"] == 42, f"Status or seed mismatch: {key}")
        model = model_map[d["model_name"]]
        problem = problem_map[d["task_id"]]
        require(d["official_prompt"] == problem["prompt"], f"Official prompt mismatch: {key}")
        require(d["generation_prompt"] == prompts[d["prompt_name"]].format(problem=problem["prompt"]), f"Wrapper mismatch: {key}")
        require(d["requested_model_id"] == model["model"], f"Requested model mismatch: {key}")
        expected_returned = "gpt-5-mini-2025-08-07" if model["provider"] == "openai" else model["model"]
        require(d["response_model_id"] == expected_returned, f"Returned model mismatch: {key}")
        settings = d["request_settings"]
        cap_key = "max_output_tokens" if model["provider"] == "openai" else "max_tokens"
        require(settings[cap_key] == model["max_tokens"], f"Output cap mismatch: {key}")
        require(settings.get("temperature") == model.get("temperature"), f"Temperature mismatch: {key}")
        require(settings.get("reasoning_effort") == model.get("reasoning_effort"), f"Reasoning setting mismatch: {key}")
        require(settings.get("thinking_mode") == model.get("thinking_mode"), f"Thinking setting mismatch: {key}")
        require(d["started_at"].startswith("2026-09-15T") and d["generated_at"].startswith("2026-09-15T"), f"Collection date mismatch: {key}")
        require(d["started_at"].endswith("+00:00") and d["generated_at"].endswith("+00:00"), f"Non-UTC timestamp: {key}")
        md = metadata[d["model_name"]]
        md["dates"].extend([d["started_at"], d["generated_at"]])
        md["identifiers"][(d["requested_model_id"], d["response_model_id"])] += 1
        md["finish_reasons"][str(d["response_metadata"].get("finish_reason"))] += 1
        schedule_indices[d["schedule_index"]] += 1
        if d["schedule_index"] != expected_indices[key]:
            exceptions.append({"key": key, "saved_index": d["schedule_index"], "full_schedule_index": expected_indices[key], "started_at": d["started_at"]})
        else:
            main_order.append((d["schedule_index"], d["started_at"]))
        row = saved[path.name]
        try:
            extracted = build_metrics(ast.parse(d["generated_code"]))
        except SyntaxError:
            require(row["ast_success"].lower() == "false", f"AST status mismatch: {key}")
            continue
        ast_ok += 1
        require(row["ast_success"].lower() == "true", f"AST status mismatch: {key}")
        for feature in FEATURES:
            error = abs(extracted[feature] - float(row[feature]))
            max_errors[feature] = max(max_errors[feature], error)
            require(math.isfinite(error) and error <= 1e-12, f"Metric mismatch for {feature}: {key}")

    require(len(keys) == 12300 and max(keys.values()) == 1, "Duplicate or missing sample keys")
    require(ast_ok == 12291, f"AST count mismatch: {ast_ok}")
    expected_early = {("HumanEval/0", m["name"], "paraphrase_a", 1) for m in models}
    require({tuple(x["key"]) for x in exceptions} == expected_early, "Unexpected schedule exceptions")
    main_order.sort()
    require(all(a[1] <= b[1] for a, b in zip(main_order, main_order[1:])), "Main-loop timestamps violate schedule order")
    require(all(x["started_at"] < main_order[0][1] for x in exceptions), "Expected earlier records do not predate main loop")
    report = {
        "status": "passed", "records": len(files), "unique_sample_keys": len(keys),
        "ast_successes": ast_ok, "ast_failures": len(files) - ast_ok,
        "maximum_absolute_descriptor_errors": max_errors,
        "matched_full_schedule_records": len(main_order),
        "unique_stored_schedule_indices": len(schedule_indices),
        "known_earlier_records": exceptions,
        "models": {m: {"utc_range": [min(v["dates"]), max(v["dates"])],
                       "identifiers": [{"requested": k[0], "returned": k[1], "count": c} for k, c in v["identifiers"].items()],
                       "finish_reasons": dict(v["finish_reasons"])} for m, v in metadata.items()},
        "scope": "Static raw-record and AST audit only; no generated code or functional tests executed.",
    }
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
