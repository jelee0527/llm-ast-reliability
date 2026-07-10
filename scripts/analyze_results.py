from pathlib import Path
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr


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
PASSED_ONLY_STABILITY_PATH = RESULTS_DIR / "passed_only_repeat_stability_summary.csv"


MODEL_COLUMN = "model_name"

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
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(path)


def holm_adjust_pvalues(
    pvalues: list[float],
) -> list[float]:
    """
    Holm-Bonferroni correction.
    Returns adjusted p-values in the original order.
    """
    n = len(pvalues)

    if n == 0:
        return []

    indexed = sorted(
        enumerate(pvalues),
        key=lambda item: item[1],
    )

    adjusted = [0.0] * n
    previous_adjusted = 0.0

    for rank, (original_index, pvalue) in enumerate(indexed):
        factor = n - rank
        corrected = min(
            pvalue * factor,
            1.0,
        )

        corrected = max(
            corrected,
            previous_adjusted,
        )

        adjusted[original_index] = corrected
        previous_adjusted = corrected

    return adjusted


def run_kruskal_by_model(
    dataframe: pd.DataFrame,
    metric: str,
    source_name: str,
) -> dict[str, Any]:
    groups = []

    for model_name, group in dataframe.groupby(MODEL_COLUMN):
        values = (
            group[metric]
            .dropna()
            .astype(float)
            .to_numpy()
        )

        if len(values) > 0:
            groups.append(
                (
                    model_name,
                    values,
                )
            )

    if len(groups) < 2:
        return {
            "source": source_name,
            "test": "Kruskal-Wallis",
            "metric": metric,
            "group_column": MODEL_COLUMN,
            "num_groups": len(groups),
            "statistic": np.nan,
            "p_value": np.nan,
            "significant_0_05": False,
            "note": "Not enough groups",
        }

    statistic, p_value = kruskal(
        *[
            values
            for _, values in groups
        ]
    )

    return {
        "source": source_name,
        "test": "Kruskal-Wallis",
        "metric": metric,
        "group_column": MODEL_COLUMN,
        "num_groups": len(groups),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_0_05": bool(p_value < 0.05),
        "note": "",
    }


def run_pairwise_mannwhitney_by_model(
    dataframe: pd.DataFrame,
    metric: str,
    source_name: str,
) -> list[dict[str, Any]]:
    model_values: dict[str, np.ndarray] = {}

    for model_name, group in dataframe.groupby(MODEL_COLUMN):
        values = (
            group[metric]
            .dropna()
            .astype(float)
            .to_numpy()
        )

        if len(values) > 0:
            model_values[model_name] = values

    raw_rows: list[dict[str, Any]] = []
    raw_pvalues: list[float] = []

    for left_model, right_model in combinations(
        sorted(model_values.keys()),
        2,
    ):
        left_values = model_values[left_model]
        right_values = model_values[right_model]

        if (
            len(left_values) == 0
            or len(right_values) == 0
        ):
            continue

        statistic, p_value = mannwhitneyu(
            left_values,
            right_values,
            alternative="two-sided",
        )

        left_median = float(
            np.median(left_values)
        )

        right_median = float(
            np.median(right_values)
        )

        row = {
            "source": source_name,
            "test": "Mann-Whitney U",
            "metric": metric,
            "group_column": MODEL_COLUMN,
            "group_a": left_model,
            "group_b": right_model,
            "n_a": len(left_values),
            "n_b": len(right_values),
            "median_a": left_median,
            "median_b": right_median,
            "median_difference_a_minus_b": (
                left_median - right_median
            ),
            "statistic": float(statistic),
            "p_value": float(p_value),
        }

        raw_rows.append(row)
        raw_pvalues.append(float(p_value))

    adjusted_pvalues = holm_adjust_pvalues(
        raw_pvalues
    )

    for row, adjusted_pvalue in zip(
        raw_rows,
        adjusted_pvalues,
    ):
        row["p_value_holm"] = adjusted_pvalue
        row["significant_0_05_holm"] = bool(
            adjusted_pvalue < 0.05
        )

    return raw_rows


def analyze_distribution_tests(
    repeat_stability_df: pd.DataFrame,
    prompt_sensitivity_df: pd.DataFrame,
    structural_diversity_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_specs = [
        (
            repeat_stability_df,
            "repeat_stability",
            [
                "ssi",
                "avg_repeat_distance",
                "success_rate",
            ],
        ),
        (
            prompt_sensitivity_df,
            "prompt_sensitivity",
            [
                "pssi",
                "avg_prompt_distance",
                "success_rate",
            ],
        ),
        (
            structural_diversity_df,
            "structural_diversity",
            [
                "sds",
                "structural_variance",
                "success_rate",
            ],
        ),
    ]

    kruskal_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for dataframe, source_name, metrics in test_specs:
        for metric in metrics:
            if metric not in dataframe.columns:
                continue

            kruskal_rows.append(
                run_kruskal_by_model(
                    dataframe=dataframe,
                    metric=metric,
                    source_name=source_name,
                )
            )

            pairwise_rows.extend(
                run_pairwise_mannwhitney_by_model(
                    dataframe=dataframe,
                    metric=metric,
                    source_name=source_name,
                )
            )

    return (
        pd.DataFrame(kruskal_rows),
        pd.DataFrame(pairwise_rows),
    )


def run_spearman_correlation(
    dataframe: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    source_name: str,
) -> dict[str, Any]:
    subset = dataframe[
        [
            x_metric,
            y_metric,
        ]
    ].dropna()

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
    rows: list[dict[str, Any]] = []

    rows.append(
        run_spearman_correlation(
            dataframe=repeat_stability_df,
            x_metric="success_rate",
            y_metric="ssi",
            source_name="repeat_stability",
        )
    )

    rows.append(
        run_spearman_correlation(
            dataframe=repeat_stability_df,
            x_metric="success_rate",
            y_metric="avg_repeat_distance",
            source_name="repeat_stability",
        )
    )

    rows.append(
        run_spearman_correlation(
            dataframe=prompt_sensitivity_df,
            x_metric="success_rate",
            y_metric="pssi",
            source_name="prompt_sensitivity",
        )
    )

    rows.append(
        run_spearman_correlation(
            dataframe=structural_diversity_df,
            x_metric="success_rate",
            y_metric="sds",
            source_name="structural_diversity",
        )
    )

    rows.append(
        run_spearman_correlation(
            dataframe=model_summary_df,
            x_metric="functional_pass_rate",
            y_metric="ssi",
            source_name="model_summary",
        )
    )

    rows.append(
        run_spearman_correlation(
            dataframe=model_summary_df,
            x_metric="functional_pass_rate",
            y_metric="pssi",
            source_name="model_summary",
        )
    )

    rows.append(
        run_spearman_correlation(
            dataframe=model_summary_df,
            x_metric="functional_pass_rate",
            y_metric="sds",
            source_name="model_summary",
        )
    )

    return pd.DataFrame(rows)


def rank_model_metrics(
    model_summary_df: pd.DataFrame,
) -> pd.DataFrame:
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

        dataframe[
            f"{metric}_rank"
        ] = dataframe[metric].rank(
            ascending=ascending,
            method="min",
        )

    return dataframe


def summarize_passed_only_by_model(
    metrics_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    기능 테스트를 통과한 코드만 대상으로
    구조적 특징의 평균을 요약한다.

    이 분석은 '실패 코드가 많아서 구조가 불안정해 보인 것'
    이라는 반론을 완화하기 위한 보조 분석이다.
    """
    passed_df = metrics_summary_df[
        (
            metrics_summary_df[
                "functional_passed"
            ]
            == True
        )
        & (
            metrics_summary_df[
                "ast_success"
            ]
            == True
        )
    ].copy()

    summary = (
        passed_df.groupby(
            [
                "model_name",
                "model_id",
                "provider",
            ],
            as_index=False,
        )
        .agg(
            num_passed_ast_samples=(
                "file",
                "count",
            ),
            ast_depth=(
                "ast_depth",
                "mean",
            ),
            branch_count=(
                "branch_count",
                "mean",
            ),
            loop_count=(
                "loop_count",
                "mean",
            ),
            function_count=(
                "function_count",
                "mean",
            ),
            control_flow_ratio=(
                "control_flow_ratio",
                "mean",
            ),
        )
    )

    return summary


def summarize_passed_only_repeat_stability(
    metrics_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    기능 통과 코드만 대상으로
    같은 문제·모델·프롬프트 조건에서
    구조 지표의 표준편차를 간단히 요약한다.

    이미 compute_metrics.py의 SSI 재계산은 아니지만,
    통과 코드 내부에서도 구조 변동이 남는지 보는 보조 지표다.
    """
    passed_df = metrics_summary_df[
        (
            metrics_summary_df[
                "functional_passed"
            ]
            == True
        )
        & (
            metrics_summary_df[
                "ast_success"
            ]
            == True
        )
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
        row = dict(
            zip(
                group_columns,
                group_keys,
            )
        )

        row["num_passed_samples"] = len(group)

        for feature in STRUCTURAL_FEATURES:
            row[
                f"{feature}_std"
            ] = float(
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

    summary = (
        group_df.groupby(
            [
                "model_name",
                "model_id",
                "provider",
            ],
            as_index=False,
        )
        .agg(
            groups=(
                "problem_id",
                "count",
            ),
            mean_passed_samples_per_group=(
                "num_passed_samples",
                "mean",
            ),
            **{
                column: (
                    column,
                    "mean",
                )
                for column in std_columns
            },
        )
    )

    return summary


def save_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    dataframe.to_csv(
        path,
        index=False,
    )

    print(f"Saved: {path}")


def main() -> None:
    ensure_results_dir()

    metrics_summary_df = load_csv(
        METRICS_SUMMARY_PATH
    )

    repeat_stability_df = load_csv(
        REPEAT_STABILITY_PATH
    )

    prompt_sensitivity_df = load_csv(
        PROMPT_SENSITIVITY_PATH
    )

    structural_diversity_df = load_csv(
        STRUCTURAL_DIVERSITY_PATH
    )

    model_summary_df = load_csv(
        MODEL_SUMMARY_PATH
    )

    prompt_summary_df = load_csv(
        PROMPT_SUMMARY_PATH
    )

    statistical_tests_df, pairwise_tests_df = (
        analyze_distribution_tests(
            repeat_stability_df=repeat_stability_df,
            prompt_sensitivity_df=prompt_sensitivity_df,
            structural_diversity_df=structural_diversity_df,
        )
    )

    correlation_analysis_df = (
        analyze_correlations(
            repeat_stability_df=repeat_stability_df,
            prompt_sensitivity_df=prompt_sensitivity_df,
            structural_diversity_df=structural_diversity_df,
            model_summary_df=model_summary_df,
        )
    )

    model_metric_ranking_df = (
        rank_model_metrics(
            model_summary_df=model_summary_df
        )
    )

    passed_only_model_summary_df = (
        summarize_passed_only_by_model(
            metrics_summary_df=metrics_summary_df
        )
    )

    passed_only_repeat_summary_df = (
        summarize_passed_only_repeat_stability(
            metrics_summary_df=metrics_summary_df
        )
    )

    save_dataframe(
        statistical_tests_df,
        STATISTICAL_TESTS_PATH,
    )

    save_dataframe(
        pairwise_tests_df,
        PAIRWISE_TESTS_PATH,
    )

    save_dataframe(
        correlation_analysis_df,
        CORRELATION_ANALYSIS_PATH,
    )

    save_dataframe(
        model_metric_ranking_df,
        MODEL_METRIC_RANKING_PATH,
    )

    save_dataframe(
        passed_only_model_summary_df,
        PASSED_ONLY_SUMMARY_PATH,
    )

    save_dataframe(
        passed_only_repeat_summary_df,
        PASSED_ONLY_STABILITY_PATH,
    )

    print("\n===== Model Metric Ranking =====")
    print(
        model_metric_ranking_df.to_string(
            index=False
        )
    )

    print("\n===== Statistical Tests =====")
    print(
        statistical_tests_df.to_string(
            index=False
        )
    )

    print("\n===== Correlation Analysis =====")
    print(
        correlation_analysis_df.to_string(
            index=False
        )
    )

    print("\n===== Passed-only Structural Summary =====")
    print(
        passed_only_model_summary_df.to_string(
            index=False
        )
    )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()