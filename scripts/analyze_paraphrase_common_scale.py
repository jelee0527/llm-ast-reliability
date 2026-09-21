"""Common-scale statistical and truncation analyses for the paraphrase study."""

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
GPT_1200_RESTRICTED_DIR = OUTPUT_DIR / "gpt_1200_restricted"
RAW_DIR = ROOT_DIR / "outputs" / "paraphrase" / "raw"

ORIGINAL_PSSI_PATH = RESULTS_DIR / "prompt_sensitivity.csv"
ORIGINAL_METRICS_PATH = RESULTS_DIR / "metrics_summary.csv"
PARAPHRASE_PSSI_PATH = PARAPHRASE_DIR / "prompt_sensitivity.csv"
PARAPHRASE_SSI_PATH = PARAPHRASE_DIR / "repeat_stability.csv"
PARAPHRASE_SDS_PATH = PARAPHRASE_DIR / "structural_diversity.csv"
PARAPHRASE_METRICS_PATH = PARAPHRASE_DIR / "metrics_summary.csv"
PARAPHRASE_MODEL_PATH = PARAPHRASE_DIR / "model_summary.csv"
EVALPLUS_SAMPLE_PATH = RESULTS_DIR / "paraphrase_evalplus_sample_results.csv"

EXPECTED_SAMPLES = 12_300
EXPECTED_TASKS = 164
EXPECTED_TRUNCATIONS = 6
GPT_TOKEN_THRESHOLD = 1_200
EXPECTED_GPT_OVER_THRESHOLD = 19
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


def prepare_common_scale_pssi_pairs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute both experiments with one pooled structural-feature scale.

    Dataset-native PSSI remains the primary within-experiment result. This
    pooled scaling is a cross-experiment sensitivity analysis that prevents
    separate z-score denominators from driving the observed reduction.
    """
    original_metrics = pd.read_csv(ORIGINAL_METRICS_PATH)
    paraphrase_metrics = pd.read_csv(PARAPHRASE_METRICS_PATH)

    if len(original_metrics) != EXPECTED_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_SAMPLES} original metric rows, "
            f"found {len(original_metrics)}"
        )
    if len(paraphrase_metrics) != EXPECTED_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_SAMPLES} paraphrase metric rows, "
            f"found {len(paraphrase_metrics)}"
        )

    original_metrics = original_metrics.copy()
    paraphrase_metrics = paraphrase_metrics.copy()
    original_metrics["analysis_experiment"] = "instruction"
    paraphrase_metrics["analysis_experiment"] = "paraphrase"

    combined = pd.concat(
        [original_metrics, paraphrase_metrics],
        ignore_index=True,
        sort=False,
    )
    combined = compute_metrics.add_standardized_features(combined)

    instruction_scaled = combined.loc[
        combined["analysis_experiment"] == "instruction"
    ].copy()
    paraphrase_scaled = combined.loc[
        combined["analysis_experiment"] == "paraphrase"
    ].copy()

    instruction_pssi = compute_metrics.compute_prompt_sensitivity(
        instruction_scaled
    )
    paraphrase_pssi = compute_metrics.compute_prompt_sensitivity(
        paraphrase_scaled
    )

    keys = ["problem_id", "task_id", "model_name"]
    paired = instruction_pssi[keys + ["pssi"]].rename(
        columns={"pssi": "instruction_pssi"}
    ).merge(
        paraphrase_pssi[keys + ["pssi"]].rename(
            columns={"pssi": "paraphrase_pssi"}
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    expected = EXPECTED_TASKS * 3
    if len(paired) != expected:
        raise AssertionError(
            f"Expected {expected} common-scale PSSI pairs, found {len(paired)}"
        )
    paired["difference_paraphrase_minus_instruction"] = (
        paired["paraphrase_pssi"] - paired["instruction_pssi"]
    )
    return instruction_pssi, paraphrase_pssi, paired


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

    # Fit once on all valid paraphrase outputs; preserve this scale after exclusion.
    metrics = compute_metrics.add_standardized_features(metrics)
    excluded = set(truncated["file"].astype(str))
    filtered = metrics.loc[~metrics["file"].isin(excluded)].copy()
    if len(filtered) != EXPECTED_SAMPLES - EXPECTED_TRUNCATIONS:
        raise AssertionError("Not all truncated samples matched metrics_summary.csv")

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


def find_gpt_over_token_threshold() -> pd.DataFrame:
    """Identify GPT-5 mini outputs that used more than 1,200 output tokens."""
    rows: list[dict[str, object]] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("model_name") != "gpt5_model":
            continue

        metadata = payload.get("response_metadata") or {}
        output_tokens = metadata.get("output_tokens")
        if output_tokens is not None and output_tokens > GPT_TOKEN_THRESHOLD:
            rows.append(
                {
                    "file": path.name,
                    "task_id": payload.get("task_id"),
                    "problem_id": payload.get("problem_id"),
                    "model_name": payload.get("model_name"),
                    "prompt_name": payload.get("prompt_name"),
                    "repeat_idx": payload.get("repeat_idx"),
                    "max_tokens": payload.get("max_tokens"),
                    "output_tokens": output_tokens,
                }
            )

    excluded = pd.DataFrame(rows)
    if len(excluded) != EXPECTED_GPT_OVER_THRESHOLD:
        raise AssertionError(
            f"Expected {EXPECTED_GPT_OVER_THRESHOLD} GPT samples above "
            f"{GPT_TOKEN_THRESHOLD} tokens, found {len(excluded)}"
        )
    return excluded


def recompute_without_gpt_over_threshold(
    excluded_samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-hoc restriction analysis excluding GPT outputs above 1,200 tokens."""
    metrics = pd.read_csv(PARAPHRASE_METRICS_PATH)
    if len(metrics) != EXPECTED_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_SAMPLES} metric rows, found {len(metrics)}"
        )

    # Fit once on all valid paraphrase outputs; preserve this scale after exclusion.
    metrics = compute_metrics.add_standardized_features(metrics)
    excluded = set(excluded_samples["file"].astype(str))
    filtered = metrics.loc[~metrics["file"].astype(str).isin(excluded)].copy()
    expected_rows = EXPECTED_SAMPLES - EXPECTED_GPT_OVER_THRESHOLD
    if len(filtered) != expected_rows:
        raise AssertionError("Not all GPT >1,200-token samples matched metrics data")

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

    GPT_1200_RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics_summary.csv": filtered,
        "repeat_stability.csv": stability,
        "prompt_sensitivity.csv": sensitivity,
        "structural_diversity.csv": diversity,
        "model_summary.csv": model_summary,
        "prompt_summary.csv": prompt_summary,
    }
    for filename, dataframe in outputs.items():
        dataframe.to_csv(GPT_1200_RESTRICTED_DIR / filename, index=False)

    full_model = pd.read_csv(PARAPHRASE_MODEL_PATH)
    comparison_columns = [
        "num_samples",
        "functional_pass_rate",
        "branch_count",
        "ssi",
        "pssi",
        "sds",
    ]
    comparison = full_model[["model_name"] + comparison_columns].merge(
        model_summary[["model_name"] + comparison_columns],
        on="model_name",
        suffixes=("_full", "_restricted"),
        validate="one_to_one",
    )
    for column in comparison_columns:
        comparison[f"{column}_difference"] = (
            comparison[f"{column}_restricted"] - comparison[f"{column}_full"]
        )
    comparison.to_csv(GPT_1200_RESTRICTED_DIR / "comparison.csv", index=False)
    return model_summary, comparison


def main() -> None:
    require_files(
        [
            ORIGINAL_PSSI_PATH,
            ORIGINAL_METRICS_PATH,
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
    (
        common_instruction_pssi,
        common_paraphrase_pssi,
        common_paired,
    ) = prepare_common_scale_pssi_pairs()
    common_comparison = summarize_pssi_comparison(common_paired)
    model_ci = model_confidence_intervals()
    prompt_ci = prompt_confidence_intervals()
    truncated = find_truncated_samples()
    no_truncation_model, truncation_comparison = recompute_without_truncation(
        truncated
    )
    gpt_over_threshold = find_gpt_over_token_threshold()
    _, gpt_restriction_comparison = recompute_without_gpt_over_threshold(
        gpt_over_threshold
    )

    paired.to_csv(OUTPUT_DIR / "pssi_paired_values.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "pssi_comparison.csv", index=False)
    common_instruction_pssi.to_csv(
        OUTPUT_DIR / "pssi_common_scale_instruction.csv", index=False
    )
    common_paraphrase_pssi.to_csv(
        OUTPUT_DIR / "pssi_common_scale_paraphrase.csv", index=False
    )
    common_paired.to_csv(
        OUTPUT_DIR / "pssi_common_scale_paired_values.csv", index=False
    )
    common_comparison.to_csv(
        OUTPUT_DIR / "pssi_common_scale_comparison.csv", index=False
    )
    model_ci.to_csv(OUTPUT_DIR / "model_confidence_intervals.csv", index=False)
    prompt_ci.to_csv(OUTPUT_DIR / "prompt_confidence_intervals.csv", index=False)
    truncated.to_csv(OUTPUT_DIR / "truncated_samples.csv", index=False)
    truncation_comparison.to_csv(
        OUTPUT_DIR / "truncation_sensitivity_comparison.csv",
        index=False,
    )
    gpt_over_threshold.to_csv(
        OUTPUT_DIR / "gpt_over_1200_token_samples.csv", index=False
    )

    print("\n===== Paired PSSI comparison =====")
    print(comparison.to_string(index=False))
    print("\n===== Paired PSSI comparison (common pooled scale) =====")
    print(common_comparison.to_string(index=False))
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
    print("\n===== GPT post-hoc 1,200-token restriction =====")
    gpt_row = gpt_restriction_comparison.loc[
        gpt_restriction_comparison["model_name"] == "gpt5_model"
    ]
    gpt_columns = [
        "model_name",
        "num_samples_full",
        "num_samples_restricted",
        "functional_pass_rate_difference",
        "branch_count_difference",
        "ssi_difference",
        "pssi_difference",
        "sds_difference",
    ]
    print(gpt_row[gpt_columns].to_string(index=False))
    print(f"\nSaved robustness analyses: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
