"""Statistical and truncation-sensitivity analyses for the paraphrase study."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

import compute_metrics


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
PARAPHRASE_DIR = RESULTS_DIR / "paraphrase_structural"
OUTPUT_DIR = RESULTS_DIR / "paraphrase_robustness"
NO_TRUNCATION_DIR = OUTPUT_DIR / "no_truncation"
RAW_DIR = ROOT_DIR / "outputs" / "paraphrase" / "raw"

ORIGINAL_PSSI_PATH = RESULTS_DIR / "prompt_sensitivity.csv"
PARAPHRASE_PSSI_PATH = PARAPHRASE_DIR / "prompt_sensitivity.csv"
PARAPHRASE_SSI_PATH = PARAPHRASE_DIR / "repeat_stability.csv"
PARAPHRASE_SDS_PATH = PARAPHRASE_DIR / "structural_diversity.csv"
PARAPHRASE_METRICS_PATH = PARAPHRASE_DIR / "metrics_summary.csv"
PARAPHRASE_MODEL_PATH = PARAPHRASE_DIR / "model_summary.csv"
EVALPLUS_SAMPLE_PATH = RESULTS_DIR / "paraphrase_evalplus_sample_results.csv"

EXPECTED_SAMPLES = 12_300
EXPECTED_TASKS = 164
EXPECTED_TRUNCATIONS = 6
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required files not found: {missing}")


def mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.choice(clean, size=clean.size, replace=True)
        estimates[index] = sample.mean()

    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(clean.mean()), float(low), float(high)


def paired_difference_ci(
    differences: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    return mean_ci(differences, seed)


def paired_rank_biserial(differences: np.ndarray) -> float:
    clean = np.asarray(differences, dtype=float)
    clean = clean[np.isfinite(clean) & (clean != 0)]
    if clean.size == 0:
        return 0.0

    ranks = rankdata(np.abs(clean), method="average")
    positive = float(ranks[clean > 0].sum())
    negative = float(ranks[clean < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def wilcoxon_test(differences: np.ndarray) -> tuple[float, float]:
    clean = np.asarray(differences, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0 or np.allclose(clean, 0.0):
        return 0.0, 1.0
    result = wilcoxon(
        clean,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.astype(float).to_numpy()
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    total = len(values)

    for rank_index, original_index in enumerate(order):
        candidate = (total - rank_index) * values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(running_max, 1.0)

    return pd.Series(adjusted, index=p_values.index)


def prepare_pssi_pairs() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_PSSI_PATH)
    paraphrase = pd.read_csv(PARAPHRASE_PSSI_PATH)

    keys = ["problem_id", "task_id", "model_name"]
    original = original[keys + ["pssi"]].rename(
        columns={"pssi": "instruction_pssi"}
    )
    paraphrase = paraphrase[keys + ["pssi"]].rename(
        columns={"pssi": "paraphrase_pssi"}
    )
    paired = original.merge(paraphrase, on=keys, how="inner", validate="one_to_one")

    expected = EXPECTED_TASKS * 3
    if len(paired) != expected:
        raise AssertionError(f"Expected {expected} paired PSSI rows, found {len(paired)}")

    paired["difference_paraphrase_minus_instruction"] = (
        paired["paraphrase_pssi"] - paired["instruction_pssi"]
    )
    return paired


def summarize_pssi_comparison(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    groups: list[tuple[str, pd.DataFrame]] = []

    overall = (
        paired.groupby(["problem_id", "task_id"], as_index=False)[
            ["instruction_pssi", "paraphrase_pssi"]
        ]
        .mean()
    )
    groups.append(("overall_problem_mean", overall))

    for model_name, group in paired.groupby("model_name", sort=True):
        groups.append((str(model_name), group.copy()))

    for index, (scope, group) in enumerate(groups):
        differences = (
            group["paraphrase_pssi"] - group["instruction_pssi"]
        ).to_numpy(dtype=float)
        mean_difference, ci_low, ci_high = paired_difference_ci(
            differences,
            BOOTSTRAP_SEED + index,
        )
        statistic, p_value = wilcoxon_test(differences)
        instruction_mean = float(group["instruction_pssi"].mean())
        paraphrase_mean = float(group["paraphrase_pssi"].mean())

        rows.append(
            {
                "scope": scope,
                "n_paired_problems": len(group),
                "instruction_pssi_mean": instruction_mean,
                "paraphrase_pssi_mean": paraphrase_mean,
                "reduction_percent": (
                    100.0 * (instruction_mean - paraphrase_mean) / instruction_mean
                    if instruction_mean != 0
                    else np.nan
                ),
                "mean_difference_paraphrase_minus_instruction": mean_difference,
                "mean_difference_ci_low": ci_low,
                "mean_difference_ci_high": ci_high,
                "median_difference": float(np.median(differences)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "paired_rank_biserial": paired_rank_biserial(differences),
            }
        )

    summary = pd.DataFrame(rows)
    summary["holm_adjusted_p"] = holm_adjust(summary["p_value"])
    return summary


def model_confidence_intervals() -> pd.DataFrame:
    ssi = pd.read_csv(PARAPHRASE_SSI_PATH)
    pssi = pd.read_csv(PARAPHRASE_PSSI_PATH)
    sds = pd.read_csv(PARAPHRASE_SDS_PATH)
    functional = pd.read_csv(EVALPLUS_SAMPLE_PATH)

    # Use one value per independent HumanEval problem before bootstrapping.
    ssi_problem = (
        ssi.groupby(["problem_id", "model_name"], as_index=False)["ssi"].mean()
    )
    pssi_problem = pssi[["problem_id", "model_name", "pssi"]].copy()
    sds_problem = sds[["problem_id", "model_name", "sds"]].copy()
    functional_problem = (
        functional.groupby(["problem_id", "model_name"], as_index=False)[
            "plus_passed"
        ]
        .mean()
        .rename(columns={"plus_passed": "human_eval_plus_pass_rate"})
    )

    sources = [
        ("ssi", ssi_problem, "ssi"),
        ("pssi", pssi_problem, "pssi"),
        ("sds", sds_problem, "sds"),
        (
            "human_eval_plus_pass_rate",
            functional_problem,
            "human_eval_plus_pass_rate",
        ),
    ]
    rows: list[dict[str, float | int | str]] = []
    seed_offset = 0

    for metric_name, dataframe, value_column in sources:
        for model_name, group in dataframe.groupby("model_name", sort=True):
            if group["problem_id"].nunique() != EXPECTED_TASKS:
                raise AssertionError(
                    f"{metric_name}/{model_name}: expected {EXPECTED_TASKS} problems"
                )
            estimate, ci_low, ci_high = mean_ci(
                group[value_column].to_numpy(dtype=float),
                BOOTSTRAP_SEED + 100 + seed_offset,
            )
            rows.append(
                {
                    "group_type": "model",
                    "group_name": model_name,
                    "metric": metric_name,
                    "n_problems": EXPECTED_TASKS,
                    "mean": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
            seed_offset += 1

    return pd.DataFrame(rows)


def prompt_confidence_intervals() -> pd.DataFrame:
    ssi = pd.read_csv(PARAPHRASE_SSI_PATH)
    functional = pd.read_csv(EVALPLUS_SAMPLE_PATH)

    # Average across models within each problem so the bootstrap unit remains
    # the HumanEval problem, not an individual correlated generation.
    ssi_problem = (
        ssi.groupby(["problem_id", "prompt_name"], as_index=False)["ssi"].mean()
    )
    functional_problem = (
        functional.groupby(["problem_id", "prompt_name"], as_index=False)[
            "plus_passed"
        ]
        .mean()
        .rename(columns={"plus_passed": "human_eval_plus_pass_rate"})
    )

    rows: list[dict[str, float | int | str]] = []
    seed_offset = 0
    for metric_name, dataframe, value_column in [
        ("ssi", ssi_problem, "ssi"),
        (
            "human_eval_plus_pass_rate",
            functional_problem,
            "human_eval_plus_pass_rate",
        ),
    ]:
        for prompt_name, group in dataframe.groupby("prompt_name", sort=True):
            if group["problem_id"].nunique() != EXPECTED_TASKS:
                raise AssertionError(
                    f"{metric_name}/{prompt_name}: expected {EXPECTED_TASKS} problems"
                )
            estimate, ci_low, ci_high = mean_ci(
                group[value_column].to_numpy(dtype=float),
                BOOTSTRAP_SEED + 200 + seed_offset,
            )
            rows.append(
                {
                    "group_type": "prompt",
                    "group_name": prompt_name,
                    "metric": metric_name,
                    "n_problems": EXPECTED_TASKS,
                    "mean": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
            seed_offset += 1

    return pd.DataFrame(rows)


def find_truncated_samples() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        metadata = payload.get("response_metadata") or {}
        if metadata.get("finish_reason") == "length":
            rows.append(
                {
                    "file": path.name,
                    "task_id": payload.get("task_id"),
                    "problem_id": payload.get("problem_id"),
                    "model_name": payload.get("model_name"),
                    "prompt_name": payload.get("prompt_name"),
                    "repeat_idx": payload.get("repeat_idx"),
                    "max_tokens": payload.get("max_tokens"),
                    "output_tokens": metadata.get("output_tokens"),
                    "finish_reason": metadata.get("finish_reason"),
                }
            )

    truncated = pd.DataFrame(rows)
    if len(truncated) != EXPECTED_TRUNCATIONS:
        raise AssertionError(
            f"Expected {EXPECTED_TRUNCATIONS} length-truncated samples, "
            f"found {len(truncated)}"
        )
    return truncated


def recompute_without_truncation(
    truncated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(PARAPHRASE_METRICS_PATH)
    if len(metrics) != EXPECTED_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_SAMPLES} metric rows, found {len(metrics)}"
        )

    excluded = set(truncated["file"].astype(str))
    filtered = metrics.loc[~metrics["file"].isin(excluded)].copy()
    if len(filtered) != EXPECTED_SAMPLES - EXPECTED_TRUNCATIONS:
        raise AssertionError("Not all truncated samples matched metrics_summary.csv")

    # Re-standardize after exclusion, then reuse the published implementations.
    filtered = compute_metrics.add_standardized_features(filtered)
    stability = compute_metrics.compute_repeat_stability(filtered)
    sensitivity = compute_metrics.compute_prompt_sensitivity(filtered)
    diversity = compute_metrics.compute_structural_diversity(filtered)
    model_summary = compute_metrics.compute_model_summary(
        metrics_df=filtered,
        stability_df=stability,
        sensitivity_df=sensitivity,
        diversity_df=diversity,
    )
    prompt_summary = compute_metrics.compute_prompt_summary(
        metrics_df=filtered,
        stability_df=stability,
    )

    NO_TRUNCATION_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics_summary.csv": filtered,
        "repeat_stability.csv": stability,
        "prompt_sensitivity.csv": sensitivity,
        "structural_diversity.csv": diversity,
        "model_summary.csv": model_summary,
        "prompt_summary.csv": prompt_summary,
    }
    for filename, dataframe in outputs.items():
        dataframe.to_csv(NO_TRUNCATION_DIR / filename, index=False)

    full_model = pd.read_csv(PARAPHRASE_MODEL_PATH)
    comparison_columns = [
        "ast_success_rate",
        "functional_pass_rate",
        "ssi",
        "pssi",
        "sds",
    ]
    comparison = full_model[["model_name"] + comparison_columns].merge(
        model_summary[["model_name"] + comparison_columns],
        on="model_name",
        suffixes=("_full", "_no_truncation"),
        validate="one_to_one",
    )
    for column in comparison_columns:
        comparison[f"{column}_difference"] = (
            comparison[f"{column}_no_truncation"]
            - comparison[f"{column}_full"]
        )

    return model_summary, comparison


def main() -> None:
    require_files(
        [
            ORIGINAL_PSSI_PATH,
            PARAPHRASE_PSSI_PATH,
            PARAPHRASE_SSI_PATH,
            PARAPHRASE_SDS_PATH,
            PARAPHRASE_METRICS_PATH,
            PARAPHRASE_MODEL_PATH,
            EVALPLUS_SAMPLE_PATH,
        ]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paired = prepare_pssi_pairs()
    comparison = summarize_pssi_comparison(paired)
    model_ci = model_confidence_intervals()
    prompt_ci = prompt_confidence_intervals()
    truncated = find_truncated_samples()
    no_truncation_model, truncation_comparison = recompute_without_truncation(
        truncated
    )

    paired.to_csv(OUTPUT_DIR / "pssi_paired_values.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "pssi_comparison.csv", index=False)
    model_ci.to_csv(OUTPUT_DIR / "model_confidence_intervals.csv", index=False)
    prompt_ci.to_csv(OUTPUT_DIR / "prompt_confidence_intervals.csv", index=False)
    truncated.to_csv(OUTPUT_DIR / "truncated_samples.csv", index=False)
    truncation_comparison.to_csv(
        OUTPUT_DIR / "truncation_sensitivity_comparison.csv",
        index=False,
    )

    print("\n===== Paired PSSI comparison =====")
    print(comparison.to_string(index=False))
    print("\n===== No-truncation model summary =====")
    print(no_truncation_model.to_string(index=False))
    print("\n===== Truncation sensitivity deltas =====")
    delta_columns = [
        "model_name",
        "functional_pass_rate_difference",
        "ssi_difference",
        "pssi_difference",
        "sds_difference",
    ]
    print(truncation_comparison[delta_columns].to_string(index=False))
    print(f"\nSaved robustness analyses: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
