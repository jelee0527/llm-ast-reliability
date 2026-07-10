from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

METRICS_SUMMARY_PATH = RESULTS_DIR / "metrics_summary.csv"
REPEAT_STABILITY_PATH = RESULTS_DIR / "repeat_stability.csv"
PROMPT_SENSITIVITY_PATH = RESULTS_DIR / "prompt_sensitivity.csv"
STRUCTURAL_DIVERSITY_PATH = RESULTS_DIR / "structural_diversity.csv"

TOP_PROMPT_SENSITIVITY_PATH = RESULTS_DIR / "top_prompt_sensitivity_problems.csv"
TOP_STRUCTURAL_DIVERSITY_PATH = RESULTS_DIR / "top_structural_diversity_problems.csv"
PROBLEM_COMPLEXITY_CORRELATION_PATH = RESULTS_DIR / "problem_complexity_correlation.csv"
PROBLEM_COMPLEXITY_SUMMARY_PATH = RESULTS_DIR / "problem_complexity_summary.csv"
DISTANCE_FUNCTION_SENSITIVITY_PATH = RESULTS_DIR / "distance_function_sensitivity.csv"
STRUCTURE_AWARE_RERANKING_PATH = RESULTS_DIR / "structure_aware_reranking.csv"
STRUCTURE_AWARE_RERANKING_GROUPS_PATH = RESULTS_DIR / "structure_aware_reranking_groups.csv"


STRUCTURAL_FEATURES = [
    "ast_depth",
    "branch_count",
    "loop_count",
    "function_count",
    "control_flow_ratio",
]

STANDARDIZED_FEATURES = [
    f"z_{feature}"
    for feature in STRUCTURAL_FEATURES
]

PROMPT_ORDER = [
    "basic",
    "concise",
    "constraint",
    "optimized",
    "readable",
]


def ensure_dirs() -> None:
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


def save_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    dataframe.to_csv(
        path,
        index=False,
    )

    print(f"Saved: {path}")


def get_problem_label(row: pd.Series) -> str:
    task_id = row.get("task_id")

    if isinstance(task_id, str):
        return task_id.replace(
            "HumanEval/",
            "HE-",
        )

    problem_id = row.get("problem_id")

    if isinstance(problem_id, str):
        return problem_id

    return "unknown"


def pairwise_distances(
    matrix: np.ndarray,
    metric: str,
) -> list[float]:
    distances: list[float] = []

    if len(matrix) < 2:
        return distances

    for left_index, right_index in combinations(
        range(len(matrix)),
        2,
    ):
        left = matrix[left_index]
        right = matrix[right_index]

        if metric == "euclidean":
            distance = float(
                np.linalg.norm(left - right)
            )

        elif metric == "manhattan":
            distance = float(
                np.sum(np.abs(left - right))
            )

        elif metric == "cosine":
            left_norm = float(
                np.linalg.norm(left)
            )
            right_norm = float(
                np.linalg.norm(right)
            )

            if (
                left_norm == 0.0
                or right_norm == 0.0
            ):
                distance = 0.0
            else:
                similarity = float(
                    np.dot(left, right)
                    / (left_norm * right_norm)
                )

                distance = 1.0 - similarity

        else:
            raise ValueError(
                f"Unsupported distance metric: {metric}"
            )

        distances.append(distance)

    return distances


def distance_to_ssi(
    average_distance: float,
) -> float:
    return float(
        1.0 / (1.0 + average_distance)
    )


def build_top_prompt_sensitivity(
    prompt_sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = prompt_sensitivity_df.copy()

    dataframe["problem_label"] = dataframe.apply(
        get_problem_label,
        axis=1,
    )

    top_df = (
        dataframe.groupby(
            [
                "problem_id",
                "task_id",
                "problem_label",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            pssi=(
                "pssi",
                "mean",
            ),
            avg_prompt_distance=(
                "avg_prompt_distance",
                "mean",
            ),
            success_rate=(
                "success_rate",
                "mean",
            ),
        )
        .sort_values(
            "pssi",
            ascending=False,
        )
        .head(10)
    )

    return top_df


def build_top_structural_diversity(
    structural_diversity_df: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = structural_diversity_df.copy()

    dataframe["problem_label"] = dataframe.apply(
        get_problem_label,
        axis=1,
    )

    top_df = (
        dataframe.groupby(
            [
                "problem_id",
                "task_id",
                "problem_label",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            sds=(
                "sds",
                "mean",
            ),
            structural_variance=(
                "structural_variance",
                "mean",
            ),
            success_rate=(
                "success_rate",
                "mean",
            ),
        )
        .sort_values(
            "sds",
            ascending=False,
        )
        .head(10)
    )

    return top_df


def compute_problem_complexity_correlation(
    metrics_summary_df: pd.DataFrame,
    repeat_stability_df: pd.DataFrame,
    prompt_sensitivity_df: pd.DataFrame,
    structural_diversity_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_df = metrics_summary_df[
        metrics_summary_df["ast_success"] == True
    ].copy()

    valid_df["problem_complexity"] = (
        valid_df["ast_depth"]
        + valid_df["branch_count"]
        + valid_df["loop_count"]
        + valid_df["function_count"]
        + valid_df["control_flow_ratio"]
    )

    valid_df["problem_label"] = valid_df.apply(
        get_problem_label,
        axis=1,
    )

    complexity_df = (
        valid_df.groupby(
            [
                "problem_id",
                "task_id",
                "problem_label",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            problem_complexity=(
                "problem_complexity",
                "mean",
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
            functional_pass_rate=(
                "functional_passed",
                "mean",
            ),
        )
    )

    ssi_df = (
        repeat_stability_df.groupby(
            [
                "problem_id",
                "task_id",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            ssi=(
                "ssi",
                "mean",
            )
        )
    )

    pssi_df = (
        prompt_sensitivity_df.groupby(
            [
                "problem_id",
                "task_id",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            pssi=(
                "pssi",
                "mean",
            )
        )
    )

    sds_df = (
        structural_diversity_df.groupby(
            [
                "problem_id",
                "task_id",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            sds=(
                "sds",
                "mean",
            ),
            structural_variance=(
                "structural_variance",
                "mean",
            ),
        )
    )

    merged = complexity_df.merge(
        ssi_df,
        on=[
            "problem_id",
            "task_id",
        ],
        how="left",
    )

    merged = merged.merge(
        pssi_df,
        on=[
            "problem_id",
            "task_id",
        ],
        how="left",
    )

    merged = merged.merge(
        sds_df,
        on=[
            "problem_id",
            "task_id",
        ],
        how="left",
    )

    correlation_pairs = [
        (
            "problem_complexity",
            "pssi",
        ),
        (
            "problem_complexity",
            "sds",
        ),
        (
            "problem_complexity",
            "ssi",
        ),
        (
            "problem_complexity",
            "functional_pass_rate",
        ),
        (
            "branch_count",
            "pssi",
        ),
        (
            "loop_count",
            "pssi",
        ),
        (
            "branch_count",
            "sds",
        ),
        (
            "loop_count",
            "sds",
        ),
    ]

    rows: list[dict[str, Any]] = []

    for x_metric, y_metric in correlation_pairs:
        subset = merged[
            [
                x_metric,
                y_metric,
            ]
        ].dropna()

        if len(subset) < 3:
            rho = np.nan
            p_value = np.nan
        else:
            rho, p_value = spearmanr(
                subset[x_metric],
                subset[y_metric],
            )

        rows.append(
            {
                "x_metric": x_metric,
                "y_metric": y_metric,
                "n": len(subset),
                "spearman_rho": rho,
                "p_value": p_value,
                "significant_0_05": (
                    bool(p_value < 0.05)
                    if not pd.isna(p_value)
                    else False
                ),
            }
        )

    correlation_df = pd.DataFrame(rows)

    return merged, correlation_df


def compute_distance_function_sensitivity(
    metrics_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    valid_df = metrics_summary_df[
        (
            metrics_summary_df["ast_success"]
            == True
        )
        & (
            metrics_summary_df[
                STANDARDIZED_FEATURES
            ]
            .notna()
            .all(axis=1)
        )
    ].copy()

    group_columns = [
        "problem_id",
        "task_id",
        "model_name",
        "prompt_name",
    ]

    distance_metrics = [
        "euclidean",
        "manhattan",
        "cosine",
    ]

    rows: list[dict[str, Any]] = []

    for group_keys, group in valid_df.groupby(
        group_columns,
        dropna=False,
    ):
        matrix = (
            group[STANDARDIZED_FEATURES]
            .to_numpy(dtype=float)
        )

        base_row = dict(
            zip(
                group_columns,
                group_keys,
            )
        )

        base_row["num_samples"] = len(group)

        for metric in distance_metrics:
            distances = pairwise_distances(
                matrix,
                metric=metric,
            )

            avg_distance = (
                float(np.mean(distances))
                if distances
                else 0.0
            )

            base_row[
                f"{metric}_avg_distance"
            ] = avg_distance

            base_row[
                f"{metric}_ssi"
            ] = distance_to_ssi(
                avg_distance
            )

        rows.append(base_row)

    condition_df = pd.DataFrame(rows)

    summary_df = (
        condition_df.groupby(
            "prompt_name",
            as_index=False,
        )
        .agg(
            cosine_ssi=(
                "cosine_ssi",
                "mean",
            ),
            euclidean_ssi=(
                "euclidean_ssi",
                "mean",
            ),
            manhattan_ssi=(
                "manhattan_ssi",
                "mean",
            ),
            cosine_avg_distance=(
                "cosine_avg_distance",
                "mean",
            ),
            euclidean_avg_distance=(
                "euclidean_avg_distance",
                "mean",
            ),
            manhattan_avg_distance=(
                "manhattan_avg_distance",
                "mean",
            ),
        )
    )

    summary_df["prompt_name"] = pd.Categorical(
        summary_df["prompt_name"],
        categories=PROMPT_ORDER,
        ordered=True,
    )

    summary_df = (
        summary_df.sort_values(
            "prompt_name"
        )
        .reset_index(drop=True)
    )

    return summary_df


def compute_structure_aware_reranking(
    metrics_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_df = metrics_summary_df[
        (
            metrics_summary_df["functional_passed"]
            == True
        )
        & (
            metrics_summary_df["ast_success"]
            == True
        )
    ].copy()

    valid_df["structural_complexity"] = (
        valid_df["ast_depth"]
        + valid_df["branch_count"]
        + valid_df["loop_count"]
        + valid_df["function_count"]
        + valid_df["control_flow_ratio"]
    )

    group_columns = [
        "problem_id",
        "task_id",
        "model_name",
        "prompt_name",
    ]

    rows: list[dict[str, Any]] = []

    for group_keys, group in valid_df.groupby(
        group_columns,
        dropna=False,
    ):
        if len(group) == 0:
            continue

        group = group.sort_values(
            "repeat_idx"
        )

        baseline_row = group.iloc[0]
        reranked_row = group.sort_values(
            "structural_complexity",
            ascending=True,
        ).iloc[0]

        baseline_complexity = float(
            baseline_row["structural_complexity"]
        )

        reranked_complexity = float(
            reranked_row["structural_complexity"]
        )

        improvement = (
            baseline_complexity
            - reranked_complexity
        )

        row = dict(
            zip(
                group_columns,
                group_keys,
            )
        )

        row.update(
            {
                "num_successful_candidates":
                    len(group),
                "baseline_file":
                    baseline_row["file"],
                "reranked_file":
                    reranked_row["file"],
                "baseline_repeat_idx":
                    baseline_row["repeat_idx"],
                "reranked_repeat_idx":
                    reranked_row["repeat_idx"],
                "baseline_complexity":
                    baseline_complexity,
                "reranked_complexity":
                    reranked_complexity,
                "improvement":
                    improvement,
                "improved":
                    improvement > 0,
            }
        )

        rows.append(row)

    group_df = pd.DataFrame(rows)

    if group_df.empty:
        summary_df = pd.DataFrame(
            [
                {
                    "num_groups": 0,
                    "baseline_complexity": np.nan,
                    "reranked_complexity": np.nan,
                    "average_improvement": np.nan,
                    "improved_groups": 0,
                    "improved_ratio": np.nan,
                }
            ]
        )

        return group_df, summary_df

    summary_df = pd.DataFrame(
        [
            {
                "num_groups":
                    len(group_df),
                "baseline_complexity":
                    group_df[
                        "baseline_complexity"
                    ].mean(),
                "reranked_complexity":
                    group_df[
                        "reranked_complexity"
                    ].mean(),
                "average_improvement":
                    group_df[
                        "improvement"
                    ].mean(),
                "improved_groups":
                    int(
                        group_df[
                            "improved"
                        ].sum()
                    ),
                "improved_ratio":
                    float(
                        group_df[
                            "improved"
                        ].mean()
                    ),
            }
        ]
    )

    return group_df, summary_df


def main() -> None:
    ensure_dirs()

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

    top_prompt_sensitivity_df = (
        build_top_prompt_sensitivity(
            prompt_sensitivity_df
        )
    )

    top_structural_diversity_df = (
        build_top_structural_diversity(
            structural_diversity_df
        )
    )

    problem_complexity_summary_df, problem_complexity_correlation_df = (
        compute_problem_complexity_correlation(
            metrics_summary_df=metrics_summary_df,
            repeat_stability_df=repeat_stability_df,
            prompt_sensitivity_df=prompt_sensitivity_df,
            structural_diversity_df=structural_diversity_df,
        )
    )

    distance_function_sensitivity_df = (
        compute_distance_function_sensitivity(
            metrics_summary_df
        )
    )

    reranking_groups_df, reranking_summary_df = (
        compute_structure_aware_reranking(
            metrics_summary_df
        )
    )

    save_dataframe(
        top_prompt_sensitivity_df,
        TOP_PROMPT_SENSITIVITY_PATH,
    )

    save_dataframe(
        top_structural_diversity_df,
        TOP_STRUCTURAL_DIVERSITY_PATH,
    )

    save_dataframe(
        problem_complexity_summary_df,
        PROBLEM_COMPLEXITY_SUMMARY_PATH,
    )

    save_dataframe(
        problem_complexity_correlation_df,
        PROBLEM_COMPLEXITY_CORRELATION_PATH,
    )

    save_dataframe(
        distance_function_sensitivity_df,
        DISTANCE_FUNCTION_SENSITIVITY_PATH,
    )

    save_dataframe(
        reranking_summary_df,
        STRUCTURE_AWARE_RERANKING_PATH,
    )

    save_dataframe(
        reranking_groups_df,
        STRUCTURE_AWARE_RERANKING_GROUPS_PATH,
    )

    print("\n===== Top-10 Problems by Prompt Sensitivity =====")
    print(
        top_prompt_sensitivity_df.to_string(
            index=False
        )
    )

    print("\n===== Top-10 Problems by Structural Diversity =====")
    print(
        top_structural_diversity_df.to_string(
            index=False
        )
    )

    print("\n===== Problem Complexity Correlation =====")
    print(
        problem_complexity_correlation_df.to_string(
            index=False
        )
    )

    print("\n===== Distance Function Sensitivity =====")
    print(
        distance_function_sensitivity_df.to_string(
            index=False
        )
    )

    print("\n===== Structure-aware Reranking =====")
    print(
        reranking_summary_df.to_string(
            index=False
        )
    )

    print("\nExtra analysis completed.")


if __name__ == "__main__":
    main()