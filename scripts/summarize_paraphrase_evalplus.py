"""Summarize EvalPlus results for the semantic-paraphrase experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from summarize_evalplus import load_json, load_samples, merge_eval_results


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_PATH = ROOT_DIR / "outputs" / "evalplus" / "paraphrase_samples.jsonl"
DEFAULT_EVAL_PATH = (
    ROOT_DIR
    / "outputs"
    / "evalplus"
    / "paraphrase_samples_eval_results.json"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--eval-results", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-samples", type=int, default=12_300)
    parser.add_argument("--expected-tasks", type=int, default=164)
    return parser.parse_args()


def summarize(dataframe: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    working = dataframe.copy()
    working["plus_failure_after_base_pass"] = (
        working["base_passed"] & ~working["plus_passed"]
    )

    if group_columns:
        summary = (
            working.groupby(group_columns, dropna=False)
            .agg(
                num_samples=("sample_id", "size"),
                evalplus_base_passes=("base_passed", "sum"),
                humaneval_plus_passes=("plus_passed", "sum"),
                plus_failures_after_base_pass=(
                    "plus_failure_after_base_pass",
                    "sum",
                ),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "scope": "overall",
                    "num_samples": len(working),
                    "evalplus_base_passes": int(working["base_passed"].sum()),
                    "humaneval_plus_passes": int(working["plus_passed"].sum()),
                    "plus_failures_after_base_pass": int(
                        working["plus_failure_after_base_pass"].sum()
                    ),
                }
            ]
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

    overall = summarize(sample_results, [])
    by_model = summarize(sample_results, ["model_name", "model_id"])
    by_prompt = summarize(sample_results, ["prompt_name"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "paraphrase_evalplus_sample_results.csv": sample_results,
        "paraphrase_evalplus_overall_summary.csv": overall,
        "paraphrase_evalplus_model_summary.csv": by_model,
        "paraphrase_evalplus_prompt_summary.csv": by_prompt,
    }
    for filename, dataframe in outputs.items():
        path = args.output_dir / filename
        dataframe.to_csv(path, index=False)
        print(f"Saved: {path}")

    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
