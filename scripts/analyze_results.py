from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

METRICS_SUMMARY_PATH = RESULTS_DIR / "metrics_summary.csv"
REPEAT_STABILITY_PATH = RESULTS_DIR / "repeat_stability.csv"
PROMPT_SENSITIVITY_PATH = RESULTS_DIR / "prompt_sensitivity.csv"
STRUCTURAL_DIVERSITY_PATH = RESULTS_DIR / "structural_diversity.csv"
MODEL_SUMMARY_PATH = RESULTS_DIR / "model_summary.csv"
PROMPT_SUMMARY_PATH = RESULTS_DIR / "prompt_summary.csv"

STATISTICAL_TESTS_PATH = RESULTS_DIR / "statistical_tests.csv"
PAIRWISE_TESTS_PATH = RESULTS_DIR / "pairwise_tests.csv"
CORRELATION_ANALYSIS_PATH = RESULTS_DIR / "correlation_analysis.csv"
MODEL_METRIC_RANKING_PATH = RESULTS_DIR / "model_metric_ranking.csv"
PASSED_ONLY_SUMMARY_PATH = RESULTS_DIR / "passed_only_model_summary.csv"
PASSED_ONLY_STABILITY_PATH = (
    RESULTS_DIR / "passed_only_repeat_stability_summary.csv"
)

MODEL_ORDER = [
    "claude_model",
    "gpt5_model",
    "deepseek_model",
]

PROMPT_ORDER = [
    "basic",
    "concise",
    "constraint",
    "optimized",
    "readable",
]

PRIMARY_METRICS = [
    "functional_pass_rate",
    "ast_success_rate",
    "ssi",
    "pssi",
    "sds",
    "structural_variance",
]

STRUCTURAL_FEATURES = [
    "ast_depth",
    "branch_count",
    "loop_count",
    "function_count",
    "control_flow_ratio",
]


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return pd.read_csv(path)


def holm_adjust_pvalues(pvalues: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""
    n = len(pvalues)

    if n == 0:
        return []

    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * n
    previous_adjusted = 0.0

    for rank, (original_index, pvalue) in enumerate(indexed):
        corrected = min(pvalue * (n - rank), 1.0)
        corrected = max(corrected, previous_adjusted)
        adjusted[original_index] = corrected
        previous_adjusted = corrected

    return adjusted


def build_complete_block_matrix(
    dataframe: pd.DataFrame,
    metric: str,
    block_columns: list[str],
    condition_column: str,
    condition_order: list[str],
) -> pd.DataFrame:
    """Create a complete repeated-measures matrix for blocked tests."""
    required_columns = [
        *block_columns,
        condition_column,
        metric,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns for blocked analysis: {missing_columns}"
        )

    wide = dataframe.pivot_table(
        index=block_columns,
        columns=condition_column,
        values=metric,
        aggfunc="mean",
    )

    available_conditions = [
        condition
        for condition in condition_order
        if condition in wide.columns
    ]

    if len(available_conditions) != len(condition_order):
        missing_conditions = sorted(
            set(condition_order) - set(available_conditions)
        )
        raise ValueError(
            f"Missing repeated conditions for {metric}: "
            f"{missing_conditions}"
        )

    return wide.dropna(subset=condition_order)[condition_order]


def run_friedman_test(
    dataframe: pd.DataFrame,
    metric: str,
    block_columns: list[str],
    condition_column: str,
    condition_order: list[str],
    source_name: str,
    comparison_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run a Friedman test and return both the result and its block matrix."""
    wide = build_complete_block_matrix(
        dataframe=dataframe,
        metric=metric,
        block_columns=block_columns,
        condition_column=condition_column,
        condition_order=condition_order,
    )

    num_blocks = len(wide)
    num_conditions = len(condition_order)

    if num_blocks < 2 or num_conditions < 2:
        result = {
            "source": source_name,
            "comparison": comparison_name,
            "test": "Friedman",
            "metric": metric,
            "block_columns": "+".join(block_columns),
            "condition_column": condition_column,
            "num_blocks": num_blocks,
            "num_conditions": num_conditions,
            "statistic": np.nan,
            "p_value": np.nan,
            "kendalls_w": np.nan,
            "significant_0_05": False,
            "note": "Not enough complete repeated-measures blocks",
        }
        return result, wide

    statistic, p_value = friedmanchisquare(
        *[
            wide[condition].to_numpy(dtype=float)
            for condition in condition_order
        ]
    )

    kendalls_w = float(
        statistic / (num_blocks * (num_conditions - 1))
    )

    result = {
        "source": source_name,
        "comparison": comparison_name,
        "test": "Friedman",
        "metric": metric,
        "block_columns": "+".join(block_columns),
        "condition_column": condition_column,
        "num_blocks": num_blocks,
        "num_conditions": num_conditions,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "kendalls_w": kendalls_w,
        "significant_0_05": bool(p_value < 0.05),
        "note": "",
    }

    return result, wide


def run_pairwise_wilcoxon(
    wide: pd.DataFrame,
    metric: str,
    condition_order: list[str],
    source_name: str,
    comparison_name: str,
) -> list[dict[str, Any]]:
    """Run paired Wilcoxon tests with Holm correction."""
    rows: list[dict[str, Any]] = []
    raw_pvalues: list[float] = []

    for left_condition, right_condition in combinations(
        condition_order,
        2,
    ):
        paired = wide[
            [left_condition, right_condition]
        ].dropna()

        left_values = paired[left_condition].to_numpy(dtype=float)
        right_values = paired[right_condition].to_numpy(dtype=float)
        differences = left_values - right_values

        if len(paired) == 0:
            statistic = np.nan
            p_value = np.nan
        elif np.allclose(differences, 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            statistic, p_value = wilcoxon(
                left_values,
                right_values,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

        row = {
            "source": source_name,
            "comparison": comparison_name,
            "test": "Wilcoxon signed-rank",
            "metric": metric,
            "group_a": left_condition,
            "group_b": right_condition,
            "n_pairs": len(paired),
            "median_a": (
                float(np.median(left_values))
                if len(left_values) > 0
                else np.nan
            ),
            "median_b": (
                float(np.median(right_values))
                if len(right_values) > 0
                else np.nan
            ),
            "median_difference_a_minus_b": (
                float(np.median(differences))
                if len(differences) > 0
                else np.nan
            ),
            "statistic": float(statistic),
            "p_value": float(p_value),
        }

        rows.append(row)
        raw_pvalues.append(float(p_value))

    adjusted_pvalues = holm_adjust_pvalues(raw_pvalues)

    for row, adjusted_pvalue in zip(rows, adjusted_pvalues):
        row["p_value_holm"] = adjusted_pvalue
        row["significant_0_05_holm"] = bool(
            adjusted_pvalue < 0.05
        )

    return rows


def analyze_distribution_tests(
    repeat_stability_df: pd.DataFrame,
    prompt_sensitivity_df: pd.DataFrame,
    structural_diversity_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run blocked repeated-measures tests at each metric's native level."""
    statistical_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    prompt_result, prompt_wide = run_friedman_test(
        dataframe=repeat_stability_df,
        metric="ssi",
        block_columns=["problem_id", "model_name"],
        condition_column="prompt_name",
        condition_order=PROMPT_ORDER,
        source_name="repeat_stability",
        comparison_name="prompt_effect_on_ssi",
    )
    statistical_rows.append(prompt_result)
    pairwise_rows.extend(
        run_pairwise_wilcoxon(
            wide=prompt_wide,
            metric="ssi",
            condition_order=PROMPT_ORDER,
            source_name="repeat_stability",
            comparison_name="prompt_effect_on_ssi",
        )
    )

    model_ssi_df = (
        repeat_stability_df.groupby(
            ["problem_id", "model_name"],
            as_index=False,
            dropna=False,
        )["ssi"]
        .mean()
    )

    model_specs = [
        (
            model_ssi_df,
            "ssi",
            "repeat_stability",
            "model_effect_on_ssi",
        ),
        (
            prompt_sensitivity_df,
            "pssi",
            "prompt_sensitivity",
            "model_effect_on_pssi",
        ),
        (
            structural_diversity_df,
            "sds",
            "structural_diversity",
            "model_effect_on_sds",
        ),
    ]

    for dataframe, metric, source_name, comparison_name in model_specs:
        result, wide = run_friedman_test(
            dataframe=dataframe,
            metric=metric,
            block_columns=["problem_id"],
            condition_column="model_name",
            condition_order=MODEL_ORDER,
            source_name=source_name,
            comparison_name=comparison_name,
        )
        statistical_rows.append(result)
        pairwise_rows.extend(
            run_pairwise_wilcoxon(
                wide=wide,
                metric=metric,
                condition_order=MODEL_ORDER,
                source_name=source_name,
                comparison_name=comparison_name,
            )
        )

    return pd.DataFrame(statistical_rows), pd.DataFrame(pairwise_rows)


def run_spearman_correlation(
    dataframe: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    source_name: str,
) -> dict[str, Any]:
    subset = dataframe[[x_metric, y_metric]].dropna()

    if len(subset) < 3:
        return {
            "source": source_name,
            "x_metric": x_metric,
            "y_metric": y_metric,
            "n": len(subset),
            "spearman_rho": np.nan,
            "p_value": np.nan,
            "significant_0_05": False,
            "note": "Not enough observations",
        }

    rho, p_value = spearmanr(
        subset[x_metric],
        subset[y_metric],
    )

    return {
        "source": source_name,
        "x_metric": x_metric,
        "y_metric": y_metric,
        "n": len(subset),
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "significant_0_05": bool(p_value < 0.05),
        "note": "",
    }


def analyze_correlations(
    repeat_stability_df: pd.DataFrame,
    prompt_sensitivity_df: pd.DataFrame,
    structural_diversity_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        run_spearman_correlation(
            repeat_stability_df,
            "success_rate",
            "ssi",
            "repeat_stability",
        ),
        run_spearman_correlation(
            repeat_stability_df,
            "success_rate",
            "avg_repeat_distance",
            "repeat_stability",
        ),
        run_spearman_correlation(
            prompt_sensitivity_df,
            "success_rate",
            "pssi",
            "prompt_sensitivity",
        ),
        run_spearman_correlation(
            structural_diversity_df,
            "success_rate",
            "sds",
            "structural_diversity",
        ),
        run_spearman_correlation(
            model_summary_df,
            "functional_pass_rate",
            "ssi",
            "model_summary",
        ),
        run_spearman_correlation(
            model_summary_df,
            "functional_pass_rate",
            "pssi",
            "model_summary",
        ),
        run_spearman_correlation(
            model_summary_df,
            "functional_pass_rate",
            "sds",
            "model_summary",
        ),
    ]

    return pd.DataFrame(rows)


def rank_model_metrics(model_summary_df: pd.DataFrame) -> pd.DataFrame:
    dataframe = model_summary_df.copy()

    ranking_rules = {
        "functional_pass_rate": False,
        "ast_success_rate": False,
        "ssi": False,
        "pssi": True,
        "sds": True,
        "structural_variance": True,
    }

    for metric, ascending in ranking_rules.items():
        if metric not in dataframe.columns:
            continue

        dataframe[f"{metric}_rank"] = dataframe[metric].rank(
            ascending=ascending,
            method="min",
        )

    return dataframe


def summarize_passed_only_by_model(
    metrics_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize AST features using only functionally correct programs."""
    passed_df = metrics_summary_df[
        (metrics_summary_df["functional_passed"] == True)
        & (metrics_summary_df["ast_success"] == True)
    ].copy()

    return (
        passed_df.groupby(
            ["model_name", "model_id", "provider"],
            as_index=False,
        )
        .agg(
            num_passed_ast_samples=("file", "count"),
            ast_depth=("ast_depth", "mean"),
            branch_count=("branch_count", "mean"),
            loop_count=("loop_count", "mean"),
            function_count=("function_count", "mean"),
            control_flow_ratio=("control_flow_ratio", "mean"),
        )
    )


def summarize_passed_only_repeat_stability(
    metrics_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize within-condition feature variation among passed programs."""
    passed_df = metrics_summary_df[
        (metrics_summary_df["functional_passed"] == True)
        & (metrics_summary_df["ast_success"] == True)
    ].copy()

    group_columns = [
        "problem_id",
        "task_id",
        "model_name",
        "model_id",
        "provider",
        "prompt_name",
    ]

    group_rows: list[dict[str, Any]] = []

    for group_keys, group in passed_df.groupby(
        group_columns,
        dropna=False,
    ):
        normalized_keys = (
            group_keys
            if isinstance(group_keys, tuple)
            else (group_keys,)
        )
        row = dict(zip(group_columns, normalized_keys))
        row["num_passed_samples"] = len(group)

        for feature in STRUCTURAL_FEATURES:
            row[f"{feature}_std"] = float(
                group[feature]
                .dropna()
                .astype(float)
                .std(ddof=0)
            )

        group_rows.append(row)

    group_df = pd.DataFrame(group_rows)
    std_columns = [
        f"{feature}_std"
        for feature in STRUCTURAL_FEATURES
    ]

    return (
        group_df.groupby(
            ["model_name", "model_id", "provider"],
            as_index=False,
        )
        .agg(
            groups=("problem_id", "count"),
            mean_passed_samples_per_group=(
                "num_passed_samples",
                "mean",
            ),
            **{
                column: (column, "mean")
                for column in std_columns
            },
        )
    )


def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    dataframe.to_csv(path, index=False)
    print(f"Saved: {path}")


def main() -> None:
    ensure_results_dir()

    metrics_summary_df = load_csv(METRICS_SUMMARY_PATH)
    repeat_stability_df = load_csv(REPEAT_STABILITY_PATH)
    prompt_sensitivity_df = load_csv(PROMPT_SENSITIVITY_PATH)
    structural_diversity_df = load_csv(STRUCTURAL_DIVERSITY_PATH)
    model_summary_df = load_csv(MODEL_SUMMARY_PATH)
    load_csv(PROMPT_SUMMARY_PATH)

    statistical_tests_df, pairwise_tests_df = analyze_distribution_tests(
        repeat_stability_df=repeat_stability_df,
        prompt_sensitivity_df=prompt_sensitivity_df,
        structural_diversity_df=structural_diversity_df,
    )

    correlation_analysis_df = analyze_correlations(
        repeat_stability_df=repeat_stability_df,
        prompt_sensitivity_df=prompt_sensitivity_df,
        structural_diversity_df=structural_diversity_df,
        model_summary_df=model_summary_df,
    )

    model_metric_ranking_df = rank_model_metrics(model_summary_df)
    passed_only_model_summary_df = summarize_passed_only_by_model(
        metrics_summary_df
    )
    passed_only_repeat_summary_df = (
        summarize_passed_only_repeat_stability(metrics_summary_df)
    )

    save_dataframe(statistical_tests_df, STATISTICAL_TESTS_PATH)
    save_dataframe(pairwise_tests_df, PAIRWISE_TESTS_PATH)
    save_dataframe(correlation_analysis_df, CORRELATION_ANALYSIS_PATH)
    save_dataframe(model_metric_ranking_df, MODEL_METRIC_RANKING_PATH)
    save_dataframe(passed_only_model_summary_df, PASSED_ONLY_SUMMARY_PATH)
    save_dataframe(
        passed_only_repeat_summary_df,
        PASSED_ONLY_STABILITY_PATH,
    )

    print("\n===== Blocked Statistical Tests =====")
    print(statistical_tests_df.to_string(index=False))

    print("\n===== Holm-adjusted Pairwise Tests =====")
    print(pairwise_tests_df.to_string(index=False))

    print("\n===== Correlation Analysis =====")
    print(correlation_analysis_df.to_string(index=False))

    print("\n===== Model Metric Ranking =====")
    print(model_metric_ranking_df.to_string(index=False))

    print("\n===== Passed-only Model Summary =====")
    print(passed_only_model_summary_df.to_string(index=False))

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
