"""Compute structural metrics for the semantic-paraphrase experiment.

This wrapper reuses the existing AST extraction and SSI/PSSI/SDS
implementations while keeping all paraphrase-derived files separate from the
original experiment outputs.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pandas as pd

import compute_metrics
import parse_ast


ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT_DIR / "outputs" / "paraphrase" / "raw"
AST_DIR = ROOT_DIR / "outputs" / "paraphrase" / "ast"
METRICS_DIR = ROOT_DIR / "outputs" / "paraphrase" / "metrics"

EVALPLUS_RESULTS_PATH = (
    ROOT_DIR / "results" / "paraphrase_evalplus_sample_results.csv"
)
FUNCTIONAL_SUMMARY_PATH = (
    ROOT_DIR / "outputs" / "paraphrase" / "evalplus_functional_summary.csv"
)
RESULTS_DIR = ROOT_DIR / "results" / "paraphrase_structural"

EXPECTED_SAMPLES = 12_300
EXPECTED_TASKS = 164


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def validate_raw_files() -> list[Path]:
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Raw directory not found: {RAW_DIR}")

    raw_files = sorted(RAW_DIR.glob("*.json"))
    if len(raw_files) != EXPECTED_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_SAMPLES} raw files, found {len(raw_files)}"
        )
    return raw_files


def build_functional_summary(raw_files: list[Path]) -> None:
    require_file(EVALPLUS_RESULTS_PATH)
    dataframe = pd.read_csv(EVALPLUS_RESULTS_PATH)

    required_columns = {
        "sample_id",
        "task_id",
        "plus_status",
        "plus_passed",
    }
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(
            "Missing EvalPlus result columns: " f"{sorted(missing)}"
        )

    if len(dataframe) != EXPECTED_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_SAMPLES} EvalPlus rows, found {len(dataframe)}"
        )
    if dataframe["sample_id"].duplicated().any():
        raise AssertionError("Duplicate sample_id values in EvalPlus results")
    if dataframe["task_id"].nunique() != EXPECTED_TASKS:
        raise AssertionError(
            f"Expected {EXPECTED_TASKS} tasks, found "
            f"{dataframe['task_id'].nunique()}"
        )

    raw_names = {path.name for path in raw_files}
    result_names = set(dataframe["sample_id"].astype(str))
    if raw_names != result_names:
        missing_results = sorted(raw_names - result_names)[:5]
        unknown_results = sorted(result_names - raw_names)[:5]
        raise AssertionError(
            "Raw/EvalPlus sample IDs do not match. "
            f"Missing results: {missing_results}; "
            f"unknown results: {unknown_results}"
        )

    plus_passed = dataframe["plus_passed"]
    if plus_passed.dtype != bool:
        normalized = plus_passed.astype(str).str.strip().str.lower()
        allowed = {"true", "false"}
        observed = set(normalized.dropna().unique())
        if not observed <= allowed:
            raise ValueError(
                "plus_passed contains non-boolean values: " f"{sorted(observed)}"
            )
        plus_passed = normalized.eq("true")

    status = dataframe["plus_status"].fillna("missing").astype(str)
    functional = pd.DataFrame(
        {
            "file": dataframe["sample_id"].astype(str),
            "execution_success": status.ne("missing"),
            "passed": plus_passed.astype(bool),
            "error_type": status.where(~plus_passed.astype(bool), ""),
            "error_message": "",
        }
    )

    FUNCTIONAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    functional.to_csv(FUNCTIONAL_SUMMARY_PATH, index=False)
    print(f"Functional summary: {FUNCTIONAL_SUMMARY_PATH}")
    print(f"HumanEval+ passes: {int(functional['passed'].sum())}")


def parse_paraphrase_asts(raw_files: list[Path]) -> None:
    parse_ast.RAW_DIR = RAW_DIR
    parse_ast.AST_DIR = AST_DIR
    parse_ast.METRICS_DIR = METRICS_DIR
    parse_ast.ensure_dirs()

    print(f"Parsing ASTs: {len(raw_files)} samples")
    for index, raw_path in enumerate(raw_files, start=1):
        # process_file prints one line per sample. Suppress those lines and
        # provide compact progress without changing the shared implementation.
        with contextlib.redirect_stdout(io.StringIO()):
            parse_ast.process_file(raw_path)
        if index % 1_000 == 0 or index == len(raw_files):
            print(f"  parsed {index}/{len(raw_files)}")

    metric_files = sorted(METRICS_DIR.glob("*.json"))
    expected_names = {path.name for path in raw_files}
    metric_names = {path.name for path in metric_files}
    if metric_names != expected_names:
        raise AssertionError(
            "Metric files do not exactly match raw files. "
            "Remove stale files from outputs/paraphrase/metrics and rerun."
        )


def configure_metric_outputs() -> None:
    compute_metrics.METRICS_DIR = METRICS_DIR
    compute_metrics.FUNCTIONAL_SUMMARY_PATH = FUNCTIONAL_SUMMARY_PATH
    compute_metrics.RESULTS_DIR = RESULTS_DIR
    compute_metrics.METRICS_SUMMARY_PATH = RESULTS_DIR / "metrics_summary.csv"
    compute_metrics.REPEAT_STABILITY_PATH = RESULTS_DIR / "repeat_stability.csv"
    compute_metrics.PROMPT_SENSITIVITY_PATH = (
        RESULTS_DIR / "prompt_sensitivity.csv"
    )
    compute_metrics.STRUCTURAL_DIVERSITY_PATH = (
        RESULTS_DIR / "structural_diversity.csv"
    )
    compute_metrics.MODEL_SUMMARY_PATH = RESULTS_DIR / "model_summary.csv"
    compute_metrics.PROMPT_SUMMARY_PATH = RESULTS_DIR / "prompt_summary.csv"


def validate_outputs() -> None:
    metrics = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")
    stability = pd.read_csv(RESULTS_DIR / "repeat_stability.csv")

    if len(metrics) != EXPECTED_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_SAMPLES} metric rows, found {len(metrics)}"
        )
    if metrics["task_id"].nunique() != EXPECTED_TASKS:
        raise AssertionError(
            f"Expected {EXPECTED_TASKS} metric tasks, found "
            f"{metrics['task_id'].nunique()}"
        )
    if int(metrics["functional_passed"].sum()) != 11_299:
        raise AssertionError(
            "HumanEval+ pass count changed; expected 11299"
        )

    print("\nParaphrase structural validation passed.")
    print(f"Samples: {len(metrics)}")
    print(f"AST successes: {int(metrics['ast_success'].sum())}")
    print(f"HumanEval+ passes: {int(metrics['functional_passed'].sum())}")
    print(f"Stability conditions: {len(stability)}")
    print(f"Results: {RESULTS_DIR}")


def main() -> None:
    raw_files = validate_raw_files()
    build_functional_summary(raw_files)
    parse_paraphrase_asts(raw_files)
    configure_metric_outputs()
    compute_metrics.main()
    validate_outputs()


if __name__ == "__main__":
    main()
