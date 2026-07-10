import gzip
import json
import re
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent

SOURCE_PATH = (
    ROOT_DIR
    / "external"
    / "human-eval"
    / "data"
    / "HumanEval.jsonl.gz"
)

OUTPUT_PATH = ROOT_DIR / "datasets" / "problems.json"


def load_humaneval(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"HumanEval dataset not found: {path}"
        )

    problems: list[dict[str, Any]] = []

    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            problems.append(json.loads(line))

    return problems


def make_problem_id(task_id: str) -> str:
    match = re.search(r"(\d+)$", task_id)

    if not match:
        safe_id = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            task_id,
        ).strip("_")

        return safe_id.lower()

    task_number = int(match.group(1))

    return f"humaneval_{task_number:03d}"


def make_problem_name(entry_point: str) -> str:
    return entry_point.replace("_", " ").title()


def convert_problem(
    problem: dict[str, Any],
) -> dict[str, Any]:
    task_id = problem["task_id"]
    entry_point = problem["entry_point"]

    return {
        "id": make_problem_id(task_id),
        "task_id": task_id,
        "name": make_problem_name(entry_point),
        "entry_point": entry_point,
        "prompt": problem["prompt"],
        "test": problem["test"],
        "canonical_solution": problem.get(
            "canonical_solution",
            "",
        ),
        "evaluation_type": "humaneval",
    }


def validate_problem(problem: dict[str, Any]) -> None:
    required_fields = [
        "id",
        "task_id",
        "name",
        "entry_point",
        "prompt",
        "test",
        "evaluation_type",
    ]

    missing_fields = [
        field
        for field in required_fields
        if not problem.get(field)
    ]

    if missing_fields:
        raise ValueError(
            f"Missing fields in {problem.get('task_id')}: "
            f"{missing_fields}"
        )


def main() -> None:
    raw_problems = load_humaneval(SOURCE_PATH)

    converted_problems = [
        convert_problem(problem)
        for problem in raw_problems
    ]

    for problem in converted_problems:
        validate_problem(problem)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            converted_problems,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 60)
    print("HumanEval preparation completed")
    print("=" * 60)
    print(f"Source file: {SOURCE_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Number of problems: {len(converted_problems)}")

    if len(converted_problems) != 164:
        print(
            "WARNING: HumanEval normally contains "
            "164 problems."
        )

    print("\nFirst problem:")
    print(
        json.dumps(
            converted_problems[0],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()