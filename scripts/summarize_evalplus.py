"""Merge EvalPlus results with experiment metadata and export summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES_PATH = (
    ROOT_DIR / "outputs" / "evalplus" / "humaneval_plus_samples.jsonl"
)
DEFAULT_EVAL_PATH = (
    ROOT_DIR
    / "outputs"
    / "evalplus"
    / "humaneval_plus_samples_eval_results.json"
)
DEFAULT_ORIGINAL_RESULTS_PATH = ROOT_DIR / "functional_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results"
PASS = "pass"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--eval-results", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument(
        "--original-results",
        type=Path,
        default=DEFAULT_ORIGINAL_RESULTS_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-samples", type=int, default=12_300)
    parser.add_argument("--expected-tasks", type=int, default=164)
    parser.add_argument(
        "--allow-base-disagreement",
        action="store_true",
        help="Write summaries without failing when EvalPlus base results differ.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_samples(path: Path) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict) or "task_id" not in sample:
                raise ValueError(f"Invalid sample at line {line_number}")
            samples[str(sample["task_id"])].append(sample)
    return dict(samples)


def merge_eval_results(
    samples_by_task: dict[str, list[dict[str, Any]]],
    eval_payload: dict[str, Any],
) -> pd.DataFrame:
    evaluations = eval_payload.get("eval")
    if not isinstance(evaluations, dict):
        raise ValueError("EvalPlus result does not contain an 'eval' mapping")

    if set(samples_by_task) != set(evaluations):
        missing = sorted(set(samples_by_task) - set(evaluations))
        extra = sorted(set(evaluations) - set(samples_by_task))
        raise AssertionError(
            f"Task mismatch between samples and results; missing={missing}, extra={extra}"
        )

    rows: list[dict[str, Any]] = []
    for task_id, samples in samples_by_task.items():
        task_results = evaluations[task_id]
        if len(samples) != len(task_results):
            raise AssertionError(
                f"{task_id}: {len(samples)} samples but {len(task_results)} results"
            )

        for sample, result in zip(samples, task_results, strict=True):
            base_status = result.get("base_status")
            plus_status = result.get("plus_status")
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "task_id": task_id,
                    "problem_id": sample["problem_id"],
                    "model_name": sample["model_name"],
                    "model_id": sample["model_id"],
                    "prompt_name": sample["prompt_name"],
                    "repeat_idx": int(sample["repeat_idx"]),
                    "base_status": base_status,
                    "plus_status": plus_status,
                    "base_passed": base_status == PASS,
                    "plus_passed": base_status == PASS and plus_status == PASS,
                }
            )

    return pd.DataFrame(rows)


def add_original_results(
    results: pd.DataFrame,
    original_results_path: Path,
) -> pd.DataFrame:
    original = pd.read_csv(original_results_path, usecols=["file", "passed"])
    original = original.rename(
        columns={"file": "sample_id", "passed": "original_humaneval_passed"}
    )
    original["original_humaneval_passed"] = (
        original["original_humaneval_passed"].fillna(False).astype(bool)
    )

    merged = results.merge(original, on="sample_id", how="left", validate="one_to_one")
    if merged["original_humaneval_passed"].isna().any():
        raise AssertionError("Some EvalPlus samples have no original HumanEval result")
    merged["base_agrees_with_original"] = (
        merged["base_passed"] == merged["original_humaneval_passed"]
    )
    return merged


def summarize(dataframe: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    working = dataframe.copy()
    working["plus_failure_after_base_pass"] = (
        working["base_passed"] & ~working["plus_passed"]
    )
    working["base_disagreement"] = ~working["base_agrees_with_original"]
    working["original_pass_evalplus_base_fail"] = (
        working["original_humaneval_passed"] & ~working["base_passed"]
    )
    working["original_fail_evalplus_base_pass"] = (
        ~working["original_humaneval_passed"] & working["base_passed"]
    )

    if group_columns:
        grouped = working.groupby(group_columns, dropna=False)
        summary = grouped.agg(
            num_samples=("sample_id", "size"),
            original_humaneval_passes=("original_humaneval_passed", "sum"),
            evalplus_base_passes=("base_passed", "sum"),
            humaneval_plus_passes=("plus_passed", "sum"),
            base_disagreements=("base_disagreement", "sum"),
            original_pass_evalplus_base_fail=(
                "original_pass_evalplus_base_fail",
                "sum",
            ),
            original_fail_evalplus_base_pass=(
                "original_fail_evalplus_base_pass",
                "sum",
            ),
            plus_failures_after_base_pass=("plus_failure_after_base_pass", "sum"),
        ).reset_index()
    else:
        summary = pd.DataFrame(
            [
                {
                    "scope": "overall",
                    "num_samples": len(working),
                    "original_humaneval_passes": int(
                        working["original_humaneval_passed"].sum()
                    ),
                    "evalplus_base_passes": int(working["base_passed"].sum()),
                    "humaneval_plus_passes": int(working["plus_passed"].sum()),
                    "base_disagreements": int(working["base_disagreement"].sum()),
                    "original_pass_evalplus_base_fail": int(
                        working["original_pass_evalplus_base_fail"].sum()
                    ),
                    "original_fail_evalplus_base_pass": int(
                        working["original_fail_evalplus_base_pass"].sum()
                    ),
                    "plus_failures_after_base_pass": int(
                        working["plus_failure_after_base_pass"].sum()
                    ),
                }
            ]
        )

    summary["original_humaneval_pass_rate"] = (
        summary["original_humaneval_passes"] / summary["num_samples"]
    )
    summary["evalplus_base_pass_rate"] = (
        summary["evalplus_base_passes"] / summary["num_samples"]
    )
    summary["humaneval_plus_pass_rate"] = (
        summary["humaneval_plus_passes"] / summary["num_samples"]
    )
    return summary


def main() -> None:
    args = parse_args()
    samples_by_task = load_samples(args.samples)
    eval_payload = load_json(args.eval_results)
    sample_results = merge_eval_results(samples_by_task, eval_payload)

    if len(sample_results) != args.expected_samples:
        raise AssertionError(
            f"Expected {args.expected_samples} results, found {len(sample_results)}"
        )
    if len(samples_by_task) != args.expected_tasks:
        raise AssertionError(
            f"Expected {args.expected_tasks} tasks, found {len(samples_by_task)}"
        )

    sample_results = add_original_results(
        sample_results,
        args.original_results,
    )
    overall = summarize(sample_results, [])
    by_model = summarize(sample_results, ["model_name", "model_id"])
    by_prompt = summarize(sample_results, ["prompt_name"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "humaneval_plus_sample_results.csv": sample_results,
        "humaneval_plus_overall_summary.csv": overall,
        "humaneval_plus_model_summary.csv": by_model,
        "humaneval_plus_prompt_summary.csv": by_prompt,
    }
    for filename, dataframe in outputs.items():
        path = args.output_dir / filename
        dataframe.to_csv(path, index=False)
        print(f"Saved: {path}")

    disagreement_count = int((~sample_results["base_agrees_with_original"]).sum())
    print(overall.to_string(index=False))
    if disagreement_count and not args.allow_base_disagreement:
        raise AssertionError(
            f"EvalPlus base tests disagree with original HumanEval for "
            f"{disagreement_count} samples. Inspect the sample-level CSV before "
            "using HumanEval+ results in the paper."
        )


if __name__ == "__main__":
    main()
