import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "outputs" / "raw"
EVAL_DIR = ROOT_DIR / "outputs" / "eval"
PROBLEMS_PATH = ROOT_DIR / "datasets" / "problems.json"
SUMMARY_PATH = ROOT_DIR / "functional_summary.csv"

DOCKER_IMAGE = "humaneval-evaluator"

# 컨테이너 내부 실행 제한은 worker에서 15초,
# Docker 명령 자체는 바깥에서 30초 제한
TIMEOUT_SECONDS = 30


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_problem_map() -> dict[str, dict[str, Any]]:
    problems = load_json(PROBLEMS_PATH)

    return {
        problem["task_id"]: problem
        for problem in problems
    }


def remove_container(container_name: str) -> None:
    try:
        subprocess.run(
            [
                "docker",
                "rm",
                "-f",
                container_name,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        pass


def parse_docker_output(stdout: str) -> dict[str, Any]:
    """
    생성 코드가 stdout에 로그를 출력하더라도
    마지막 유효 JSON 결과를 찾아 반환한다.
    """
    output_lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.strip()
    ]

    for line in reversed(output_lines):
        try:
            parsed = json.loads(line)

            if (
                isinstance(parsed, dict)
                and "passed" in parsed
            ):
                return parsed

        except json.JSONDecodeError:
            continue

    return {
        "passed": False,
        "error_type": "InvalidDockerOutput",
        "error_message": stdout,
        "traceback": None,
    }


def run_in_docker(
    generated_code: str,
    test_code: str,
    entry_point: str,
) -> dict[str, Any]:
    payload = {
        "generated_code": generated_code,
        "test": test_code,
        "entry_point": entry_point,
    }

    container_name = (
        f"humaneval-{uuid.uuid4().hex[:12]}"
    )

    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "-i",
        "--network",
        "none",
        "--memory",
        "512m",
        "--cpus",
        "1",
        "--pids-limit",
        "64",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        DOCKER_IMAGE,
    ]

    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )

    except subprocess.TimeoutExpired:
        remove_container(container_name)

        return {
            "passed": False,
            "error_type": "DockerTimeoutExpired",
            "error_message": (
                f"Docker evaluation exceeded "
                f"{TIMEOUT_SECONDS} seconds"
            ),
            "traceback": None,
        }

    except KeyboardInterrupt:
        remove_container(container_name)
        raise

    except Exception as error:
        remove_container(container_name)

        return {
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": None,
        }

    if completed.returncode != 0:
        return {
            "passed": False,
            "error_type": "DockerExecutionError",
            "error_message": (
                completed.stderr.strip()
                or f"Docker exited with code "
                f"{completed.returncode}"
            ),
            "traceback": None,
        }

    stdout = completed.stdout.strip()

    if not stdout:
        return {
            "passed": False,
            "error_type": "EmptyDockerOutput",
            "error_message": "Docker returned no output",
            "traceback": None,
        }

    return parse_docker_output(stdout)


def make_base_result(
    raw_path: Path,
    raw: dict[str, Any],
) -> dict[str, Any]:
    return {
        "file": raw_path.name,
        "problem_id": raw.get("problem_id"),
        "task_id": raw.get("task_id"),
        "model_name": raw.get("model_name"),
        "model_id": raw.get("model_id"),
        "provider": raw.get("provider"),
        "prompt_name": raw.get("prompt_name"),
        "repeat_idx": raw.get("repeat_idx"),
    }


def evaluate_file(
    raw_path: Path,
    problem_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw = load_json(raw_path)

    base_result = make_base_result(
        raw_path=raw_path,
        raw=raw,
    )

    if raw.get("status") != "success":
        return {
            **base_result,
            "execution_success": False,
            "passed": False,
            "error_type": "GenerationError",
            "error_message": raw.get(
                "error_message",
                "Generation failed",
            ),
            "traceback": None,
        }

    task_id = raw.get("task_id")
    problem = problem_map.get(task_id)

    if not problem:
        return {
            **base_result,
            "execution_success": False,
            "passed": False,
            "error_type": "ProblemNotFound",
            "error_message": (
                f"Problem not found for "
                f"task_id={task_id}"
            ),
            "traceback": None,
        }

    docker_result = run_in_docker(
        generated_code=raw.get(
            "generated_code",
            "",
        ),
        test_code=problem["test"],
        entry_point=problem["entry_point"],
    )

    passed = bool(
        docker_result.get("passed", False)
    )

    error_type = docker_result.get(
        "error_type"
    )

    return {
        **base_result,
        "execution_success": (
            passed or error_type is None
        ),
        "passed": passed,
        "error_type": error_type,
        "error_message": docker_result.get(
            "error_message"
        ),
        "traceback": docker_result.get(
            "traceback"
        ),
    }


def build_unexpected_error_result(
    raw_path: Path,
    error: Exception,
) -> dict[str, Any]:
    try:
        raw = load_json(raw_path)

        base_result = make_base_result(
            raw_path=raw_path,
            raw=raw,
        )

    except Exception:
        base_result = {
            "file": raw_path.name,
            "problem_id": None,
            "task_id": None,
            "model_name": None,
            "model_id": None,
            "provider": None,
            "prompt_name": None,
            "repeat_idx": None,
        }

    return {
        **base_result,
        "execution_success": False,
        "passed": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": None,
    }


def main() -> None:
    EVAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_files = sorted(
        RAW_DIR.glob("*.json")
    )

    if not raw_files:
        print(
            "No files found in outputs/raw"
        )
        return

    problem_map = load_problem_map()
    records: list[dict[str, Any]] = []

    total_files = len(raw_files)
    existing_count = 0
    evaluated_count = 0

    print(
        f"Raw files found: {total_files}"
    )

    for index, raw_path in enumerate(
        raw_files,
        start=1,
    ):
        output_path = (
            EVAL_DIR / raw_path.name
        )

        if output_path.exists():
            try:
                existing_result = load_json(
                    output_path
                )

                records.append(
                    existing_result
                )

                existing_count += 1

                print(
                    f"[{index}/{total_files}] "
                    f"Skip existing: "
                    f"{raw_path.name}"
                )

                continue

            except Exception as error:
                print(
                    f"[{index}/{total_files}] "
                    f"Broken eval result; "
                    f"re-evaluating: "
                    f"{raw_path.name}"
                )

                print(
                    f"  "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

        print(
            f"[{index}/{total_files}] "
            f"Evaluating: "
            f"{raw_path.name}"
        )

        try:
            result = evaluate_file(
                raw_path=raw_path,
                problem_map=problem_map,
            )

        except KeyboardInterrupt:
            print(
                "\nEvaluation interrupted by user."
            )

            print(
                "Saved results will be skipped "
                "when restarted."
            )

            raise

        except Exception as error:
            result = build_unexpected_error_result(
                raw_path=raw_path,
                error=error,
            )

        records.append(result)
        evaluated_count += 1

        save_json(
            output_path,
            result,
        )

        status_text = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"{status_text}: "
            f"{raw_path.name}"
        )

        if not result["passed"]:
            print(
                f"  "
                f"{result.get('error_type')}: "
                f"{result.get('error_message')}"
            )

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(
        "\nFunctional evaluation completed."
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )

    print(
        f"Existing results skipped: "
        f"{existing_count}"
    )

    print(
        f"Newly evaluated results: "
        f"{evaluated_count}"
    )

    print(
        f"Total summary records: "
        f"{len(dataframe)}"
    )

    if dataframe.empty:
        return

    print(
        "\n===== Overall Pass Rate ====="
    )

    print(
        dataframe["passed"].mean()
    )

    print(
        "\n===== Model-wise Pass Rate ====="
    )

    print(
        dataframe.groupby(
            "model_name"
        )["passed"].mean()
    )

    print(
        "\n===== Prompt-wise Pass Rate ====="
    )

    print(
        dataframe.groupby(
            "prompt_name"
        )["passed"].mean()
    )

    failed = dataframe[
        dataframe["passed"] == False
    ]

    print(
        "\n===== Failure Count ====="
    )

    print(len(failed))

    if not failed.empty:
        print(
            "\n===== Failure Types ====="
        )

        print(
            failed["error_type"]
            .value_counts(
                dropna=False
            )
        )


if __name__ == "__main__":
    main()