"""Export sample-level disagreements between the original and EvalPlus base tests."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLES_PATH = ROOT_DIR / "outputs" / "evalplus" / "humaneval_plus_samples.jsonl"
EVAL_PATH = (
    ROOT_DIR
    / "outputs"
    / "evalplus"
    / "humaneval_plus_samples_eval_results.json"
)
ORIGINAL_PATH = ROOT_DIR / "functional_summary.csv"
OUTPUT_PATH = ROOT_DIR / "results" / "humaneval_plus_base_disagreements.csv"


def load_samples() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with SAMPLES_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                sample = json.loads(line)
                grouped[str(sample["task_id"])].append(sample)
    return dict(grouped)


def main() -> None:
    samples = load_samples()
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        evaluations = json.load(file)["eval"]

    original = pd.read_csv(ORIGINAL_PATH).set_index("file")
    rows: list[dict[str, Any]] = []

    for task_id, task_samples in samples.items():
        task_results = evaluations[task_id]
        if len(task_samples) != len(task_results):
            raise AssertionError(f"{task_id}: sample/result count mismatch")

        for sample, result in zip(task_samples, task_results, strict=True):
            if result.get("solution") != sample.get("solution"):
                raise AssertionError(f"{task_id}: sample/result ordering mismatch")

            sample_id = str(sample["sample_id"])
            original_row = original.loc[sample_id]
            original_passed = bool(original_row["passed"])
            evalplus_base_passed = result.get("base_status") == "pass"
            if original_passed == evalplus_base_passed:
                continue

            rows.append(
                {
                    "sample_id": sample_id,
                    "task_id": task_id,
                    "problem_id": sample["problem_id"],
                    "model_name": sample["model_name"],
                    "model_id": sample["model_id"],
                    "prompt_name": sample["prompt_name"],
                    "repeat_idx": int(sample["repeat_idx"]),
                    "original_humaneval_passed": original_passed,
                    "original_execution_success": bool(
                        original_row["execution_success"]
                    ),
                    "original_error_type": original_row.get("error_type"),
                    "original_error_message": original_row.get("error_message"),
                    "evalplus_base_status": result.get("base_status"),
                    "evalplus_base_fail_tests": json.dumps(
                        result.get("base_fail_tests", []), ensure_ascii=False
                    ),
                    "evalplus_plus_status": result.get("plus_status"),
                    "evalplus_plus_fail_tests": json.dumps(
                        result.get("plus_fail_tests", []), ensure_ascii=False
                    ),
                }
            )

    disagreements = pd.DataFrame(rows)
    disagreements.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Disagreements: {len(disagreements)}")
    print("\nDirection:")
    print(
        disagreements.groupby(
            ["original_humaneval_passed", "evalplus_base_status"]
        ).size()
    )
    print("\nTop tasks:")
    print(disagreements.groupby("task_id").size().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()
