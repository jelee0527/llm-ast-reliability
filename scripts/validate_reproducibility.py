"""Validate the frozen experiment package and its reported sample counts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

EXPECTED_TOTAL_SAMPLES = 12_300
EXPECTED_AST_SUCCESSES = 12_287
EXPECTED_FUNCTIONAL_PASSES = 11_766
EXPECTED_CONDITIONS = 2_460
EXPECTED_VALID_SSI_CONDITIONS = 2_459
EXPECTED_PROBLEM_MODEL_GROUPS = 492
EXPECTED_TREE_EDIT_PAIRS = 24_558
MINIMUM_TREE_EDIT_VALIDATION_RHO = 0.90


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_csv(path: Path) -> pd.DataFrame:
    require(path.exists(), f"Missing required file: {path.relative_to(ROOT_DIR)}")
    return pd.read_csv(path)


def validate_file_counts() -> None:
    expected_counts = {
        ROOT_DIR / "outputs" / "raw": EXPECTED_TOTAL_SAMPLES,
        ROOT_DIR / "outputs" / "eval": EXPECTED_TOTAL_SAMPLES,
        ROOT_DIR / "outputs" / "ast": EXPECTED_TOTAL_SAMPLES,
        ROOT_DIR / "outputs" / "metrics": EXPECTED_TOTAL_SAMPLES,
    }

    for directory, expected in expected_counts.items():
        actual = len(list(directory.glob("*.json")))
        require(
            actual == expected,
            f"{directory.relative_to(ROOT_DIR)}: expected {expected} JSON files, "
            f"found {actual}",
        )


def validate_sample_summaries() -> None:
    functional = load_csv(ROOT_DIR / "functional_summary.csv")
    metrics = load_csv(RESULTS_DIR / "metrics_summary.csv")

    require(
        len(functional) == EXPECTED_TOTAL_SAMPLES,
        f"functional_summary.csv: expected {EXPECTED_TOTAL_SAMPLES} rows, "
        f"found {len(functional)}",
    )
    require(
        int(functional["passed"].fillna(False).astype(bool).sum())
        == EXPECTED_FUNCTIONAL_PASSES,
        "functional_summary.csv: functional pass count does not match the paper",
    )
    require(
        len(metrics) == EXPECTED_TOTAL_SAMPLES,
        f"metrics_summary.csv: expected {EXPECTED_TOTAL_SAMPLES} rows, "
        f"found {len(metrics)}",
    )
    require(
        int(metrics["ast_success"].fillna(False).astype(bool).sum())
        == EXPECTED_AST_SUCCESSES,
        "metrics_summary.csv: AST success count does not match the paper",
    )

    key_columns = ["problem_id", "model_name", "prompt_name", "repeat_idx"]
    require(
        not metrics.duplicated(key_columns).any(),
        f"metrics_summary.csv contains duplicate experiment keys: {key_columns}",
    )


def validate_group_summaries() -> None:
    repeat = load_csv(RESULTS_DIR / "repeat_stability.csv")
    prompt = load_csv(RESULTS_DIR / "prompt_sensitivity.csv")
    diversity = load_csv(RESULTS_DIR / "structural_diversity.csv")

    require(
        len(repeat) == EXPECTED_CONDITIONS,
        f"repeat_stability.csv: expected {EXPECTED_CONDITIONS} conditions, "
        f"found {len(repeat)}",
    )
    valid_ssi = int(repeat["ssi"].notna().sum())
    require(
        valid_ssi == EXPECTED_VALID_SSI_CONDITIONS,
        f"repeat_stability.csv: expected {EXPECTED_VALID_SSI_CONDITIONS} valid "
        f"SSI conditions, found {valid_ssi}",
    )
    require(
        len(prompt) == EXPECTED_PROBLEM_MODEL_GROUPS,
        f"prompt_sensitivity.csv: expected {EXPECTED_PROBLEM_MODEL_GROUPS} rows, "
        f"found {len(prompt)}",
    )
    require(
        len(diversity) == EXPECTED_PROBLEM_MODEL_GROUPS,
        f"structural_diversity.csv: expected {EXPECTED_PROBLEM_MODEL_GROUPS} rows, "
        f"found {len(diversity)}",
    )


def validate_legacy_correlations() -> None:
    legacy_samples_path = RESULTS_DIR / "legacy_sample_metrics.csv"
    if legacy_samples_path.exists():
        legacy_samples = load_csv(legacy_samples_path)
        require(
            len(legacy_samples) == EXPECTED_TOTAL_SAMPLES,
            "legacy_sample_metrics.csv must contain all frozen generations",
        )
        require(
            legacy_samples["repeat"].notna().all(),
            "legacy_sample_metrics.csv contains missing repetition indices",
        )
        require(
            legacy_samples["functional_pass"].notna().all(),
            "legacy_sample_metrics.csv contains missing functional results",
        )
        require(
            int(legacy_samples["functional_pass"].astype(bool).sum())
            == EXPECTED_FUNCTIONAL_PASSES,
            "legacy_sample_metrics.csv: functional pass count does not match",
        )

    path = RESULTS_DIR / "legacy_vs_ssi_correlation.csv"
    if not path.exists():
        return

    correlations = load_csv(path)
    observed = set(correlations["n"].dropna().astype(int))
    require(
        observed == {EXPECTED_VALID_SSI_CONDITIONS},
        "legacy_vs_ssi_correlation.csv must exclude conditions where SSI is "
        f"undefined; expected N={EXPECTED_VALID_SSI_CONDITIONS}, found "
        f"{sorted(observed)}",
    )


def validate_tree_edit_results() -> None:
    repeat_path = RESULTS_DIR / "tree_edit_repeat_stability.csv"
    validation_path = RESULTS_DIR / "tree_edit_validation_summary.csv"
    if not repeat_path.exists() and not validation_path.exists():
        return

    require(
        repeat_path.exists() and validation_path.exists(),
        "Tree-edit result files are incomplete; rerun "
        "scripts/compute_tree_edit_validation.py",
    )
    repeat = load_csv(repeat_path)
    validation = load_csv(validation_path)

    require(
        len(repeat) == EXPECTED_CONDITIONS,
        f"tree_edit_repeat_stability.csv: expected {EXPECTED_CONDITIONS} "
        f"conditions, found {len(repeat)}",
    )
    valid = int(repeat["tree_edit_stability"].notna().sum())
    require(
        valid == EXPECTED_VALID_SSI_CONDITIONS,
        f"tree_edit_repeat_stability.csv: expected "
        f"{EXPECTED_VALID_SSI_CONDITIONS} valid conditions, found {valid}",
    )
    require(
        int(repeat["num_pairs"].sum()) == EXPECTED_TREE_EDIT_PAIRS,
        "tree_edit_repeat_stability.csv: repeat-pair count does not match",
    )
    require(
        len(validation) == 1
        and int(validation.loc[0, "n"]) == EXPECTED_VALID_SSI_CONDITIONS,
        "tree_edit_validation_summary.csv: validation N does not match SSI",
    )
    require(
        float(validation.loc[0, "spearman_rho"])
        >= MINIMUM_TREE_EDIT_VALIDATION_RHO,
        "Tree-edit validation correlation is below the documented threshold",
    )


def main() -> None:
    validate_file_counts()
    validate_sample_summaries()
    validate_group_summaries()
    validate_legacy_correlations()
    validate_tree_edit_results()

    print("Reproducibility validation passed.")
    print(f"Total samples: {EXPECTED_TOTAL_SAMPLES}")
    print(f"AST success samples: {EXPECTED_AST_SUCCESSES}")
    print(f"Functional passes: {EXPECTED_FUNCTIONAL_PASSES}")
    print(f"Valid SSI conditions: {EXPECTED_VALID_SSI_CONDITIONS}")
    tree_edit_path = RESULTS_DIR / "tree_edit_validation_summary.csv"
    if tree_edit_path.exists():
        validation = pd.read_csv(tree_edit_path)
        print(
            "SSI vs. tree-edit stability Spearman rho: "
            f"{float(validation.loc[0, 'spearman_rho']):.6f}"
        )


if __name__ == "__main__":
    main()
