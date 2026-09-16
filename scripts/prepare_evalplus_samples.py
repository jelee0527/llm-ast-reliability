"""Convert the frozen generations to EvalPlus JSONL input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT_DIR / "outputs" / "raw"
DEFAULT_OUTPUT_PATH = (
    ROOT_DIR / "outputs" / "evalplus" / "humaneval_plus_samples.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--expected-samples", type=int, default=12_300)
    parser.add_argument("--expected-tasks", type=int, default=164)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def make_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    required = [
        "task_id",
        "problem_id",
        "model_name",
        "model_id",
        "prompt_name",
        "repeat_idx",
        "generated_code",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path.name}: missing fields {missing}")

    solution = payload["generated_code"]
    if not isinstance(solution, str):
        raise TypeError(f"{path.name}: generated_code must be a string")

    return {
        "task_id": payload["task_id"],
        "solution": solution,
        "sample_id": path.name,
        "problem_id": payload["problem_id"],
        "model_name": payload["model_name"],
        "model_id": payload["model_id"],
        "prompt_name": payload["prompt_name"],
        "repeat_idx": int(payload["repeat_idx"]),
    }


def main() -> None:
    args = parse_args()
    raw_files = sorted(args.raw_dir.glob("*.json"))
    records = [make_record(path, load_json(path)) for path in raw_files]

    if len(records) != args.expected_samples:
        raise AssertionError(
            f"Expected {args.expected_samples} samples, found {len(records)}"
        )

    task_ids = {record["task_id"] for record in records}
    if len(task_ids) != args.expected_tasks:
        raise AssertionError(
            f"Expected {args.expected_tasks} tasks, found {len(task_ids)}"
        )

    experiment_keys = {
        (
            record["problem_id"],
            record["model_name"],
            record["prompt_name"],
            record["repeat_idx"],
        )
        for record in records
    }
    if len(experiment_keys) != len(records):
        raise AssertionError("Duplicate experiment keys found in frozen generations")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(args.output)

    print(f"Saved {len(records)} samples: {args.output}")
    print(f"HumanEval tasks: {len(task_ids)}")


if __name__ == "__main__":
    main()
