import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent

METRICS_DIR = ROOT_DIR / "outputs" / "metrics"
FUNCTIONAL_SUMMARY_PATH = ROOT_DIR / "functional_summary.csv"
RESULTS_DIR = ROOT_DIR / "results"

METRICS_SUMMARY_PATH = RESULTS_DIR / "metrics_summary.csv"
REPEAT_STABILITY_PATH = RESULTS_DIR / "repeat_stability.csv"
PROMPT_SENSITIVITY_PATH = RESULTS_DIR / "prompt_sensitivity.csv"
STRUCTURAL_DIVERSITY_PATH = RESULTS_DIR / "structural_diversity.csv"
MODEL_SUMMARY_PATH = RESULTS_DIR / "model_summary.csv"
PROMPT_SUMMARY_PATH = RESULTS_DIR / "prompt_summary.csv"

STRUCTURAL_FEATURES = [
    "ast_depth",
    "branch_count",
    "loop_count",
    "function_count",
    "control_flow_ratio",
]

EPSILON = 1e-12


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_metric_records() -> pd.DataFrame:
    metric_files = sorted(
        METRICS_DIR.glob("*.json")
    )

    if not metric_files:
        raise FileNotFoundError(
            f"No metric JSON files found in: "
            f"{METRICS_DIR}"
        )

    print(
        f"Metric JSON files found: "
        f"{len(metric_files)}"
    )

    records: list[dict[str, Any]] = []

    for metric_path in metric_files:
        payload = load_json(metric_path)

        base_record = {
            "file": payload.get(
                "source_file",
                metric_path.name,
            ),
            "problem_id": payload.get(
                "problem_id"
            ),
            "task_id": payload.get(
                "task_id"
            ),
            "problem_name": payload.get(
                "problem_name"
            ),
            "entry_point": payload.get(
                "entry_point"
            ),
            "model_name": payload.get(
                "model_name"
            ),
            "model_id": payload.get(
                "model_id"
            ),
            "provider": payload.get(
                "provider"
            ),
            "prompt_name": payload.get(
                "prompt_name"
            ),
            "repeat_idx": payload.get(
                "repeat_idx"
            ),
            "ast_status": payload.get(
                "status"
            ),
            "ast_error_type": payload.get(
                "error_type"
            ),
            "ast_error_message": payload.get(
                "error_message"
            ),
        }

        metrics = payload.get(
            "metrics",
            {},
        )

        if (
            payload.get("status") == "success"
            and isinstance(metrics, dict)
        ):
            record = {
                **base_record,
                "total_nodes": metrics.get(
                    "total_nodes"
                ),
                "ast_depth": metrics.get(
                    "ast_depth"
                ),
                "branch_count": metrics.get(
                    "branch_count"
                ),
                "loop_count": metrics.get(
                    "loop_count"
                ),
                "function_count": metrics.get(
                    "function_count"
                ),
                "control_flow_count": metrics.get(
                    "control_flow_count"
                ),
                "control_flow_ratio": metrics.get(
                    "control_flow_ratio"
                ),
                "function_names": json.dumps(
                    metrics.get(
                        "function_names",
                        [],
                    ),
                    ensure_ascii=False,
                ),
            }

        else:
            record = {
                **base_record,
                "total_nodes": np.nan,
                "ast_depth": np.nan,
                "branch_count": np.nan,
                "loop_count": np.nan,
                "function_count": np.nan,
                "control_flow_count": np.nan,
                "control_flow_ratio": np.nan,
                "function_names": None,
            }

        records.append(record)

    dataframe = pd.DataFrame(records)

    print(
        "AST success:",
        (
            dataframe["ast_status"]
            == "success"
        ).sum(),
    )

    print(
        "AST errors:",
        (
            dataframe["ast_status"]
            != "success"
        ).sum(),
    )

    return dataframe


def load_functional_summary() -> pd.DataFrame:
    if not FUNCTIONAL_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Functional summary not found: "
            f"{FUNCTIONAL_SUMMARY_PATH}"
        )

    dataframe = pd.read_csv(
        FUNCTIONAL_SUMMARY_PATH
    )

    required_columns = {
        "file",
        "execution_success",
        "passed",
        "error_type",
        "error_message",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing functional summary columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.rename(
        columns={
            "execution_success":
                "functional_execution_success",
            "passed":
                "functional_passed",
            "error_type":
                "functional_error_type",
            "error_message":
                "functional_error_message",
            "traceback":
                "functional_traceback",
        }
    )

    return dataframe


def merge_metric_and_functional_data(
    metric_df: pd.DataFrame,
    functional_df: pd.DataFrame,
) -> pd.DataFrame:
    functional_columns = [
        "file",
        "functional_execution_success",
        "functional_passed",
        "functional_error_type",
        "functional_error_message",
    ]

    if (
        "functional_traceback"
        in functional_df.columns
    ):
        functional_columns.append(
            "functional_traceback"
        )

    merged = metric_df.merge(
        functional_df[functional_columns],
        on="file",
        how="left",
        validate="one_to_one",
    )

    merged[
        "functional_passed"
    ] = (
        merged["functional_passed"]
        .fillna(False)
        .astype(bool)
    )

    merged[
        "ast_success"
    ] = (
        merged["ast_status"]
        == "success"
    )

    return merged


def add_standardized_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    valid_mask = (
        result["ast_success"]
        & result[STRUCTURAL_FEATURES]
        .notna()
        .all(axis=1)
    )

    for feature in STRUCTURAL_FEATURES:
        standardized_column = (
            f"z_{feature}"
        )

        result[
            standardized_column
        ] = np.nan

        valid_values = result.loc[
            valid_mask,
            feature,
        ].astype(float)

        mean_value = valid_values.mean()
        std_value = valid_values.std(ddof=0)

        if (
            pd.isna(std_value)
            or std_value < EPSILON
        ):
            result.loc[
                valid_mask,
                standardized_column,
            ] = 0.0
        else:
            result.loc[
                valid_mask,
                standardized_column,
            ] = (
                valid_values - mean_value
            ) / std_value

    return result


def pairwise_euclidean_distances(
    matrix: np.ndarray,
) -> list[float]:
    distances: list[float] = []

    if len(matrix) < 2:
        return distances

    for left_index, right_index in combinations(
        range(len(matrix)),
        2,
    ):
        distance = float(
            np.linalg.norm(
                matrix[left_index]
                - matrix[right_index]
            )
        )

        distances.append(distance)

    return distances


def distance_to_stability(
    average_distance: float,
) -> float:
    if pd.isna(average_distance):
        return np.nan

    return float(
        1.0 / (1.0 + average_distance)
    )


def compute_repeat_stability(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    SSI:
    같은 문제·모델·프롬프트 조건에서
    반복 생성된 코드들 사이의 평균 구조 거리를 계산한다.

    Average Distance가 작을수록 안정적이며,
    SSI = 1 / (1 + Average Distance)로 변환한다.
    """

    valid_df = dataframe[
        dataframe["ast_success"]
    ].copy()

    standardized_features = [
        f"z_{feature}"
        for feature in STRUCTURAL_FEATURES
    ]

    group_columns = [
        "problem_id",
        "task_id",
        "model_name",
        "model_id",
        "provider",
        "prompt_name",
    ]

    rows: list[dict[str, Any]] = []

    for group_keys, group in valid_df.groupby(
        group_columns,
        dropna=False,
    ):
        matrix = (
            group[standardized_features]
            .dropna()
            .to_numpy(dtype=float)
        )

        distances = (
            pairwise_euclidean_distances(
                matrix
            )
        )

        average_distance = (
            float(np.mean(distances))
            if distances
            else 0.0
        )

        stability_score = (
            distance_to_stability(
                average_distance
            )
        )

        row = dict(
            zip(
                group_columns,
                group_keys,
            )
        )

        row.update(
            {
                "num_samples":
                    len(group),
                "num_valid_vectors":
                    len(matrix),
                "num_pairs":
                    len(distances),
                "avg_repeat_distance":
                    average_distance,
                "stability_score":
                    stability_score,
                "ssi":
                    stability_score,
                "success_rate":
                    float(
                        group[
                            "functional_passed"
                        ].mean()
                    ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def compute_prompt_centroids(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    valid_df = dataframe[
        dataframe["ast_success"]
    ].copy()

    standardized_features = [
        f"z_{feature}"
        for feature in STRUCTURAL_FEATURES
    ]

    group_columns = [
        "problem_id",
        "task_id",
        "model_name",
        "model_id",
        "provider",
        "prompt_name",
    ]

    aggregations = {
        column: "mean"
        for column in standardized_features
    }

    aggregations[
        "functional_passed"
    ] = "mean"

    centroids = (
        valid_df.groupby(
            group_columns,
            dropna=False,
        )
        .agg(aggregations)
        .reset_index()
        .rename(
            columns={
                "functional_passed":
                    "prompt_success_rate"
            }
        )
    )

    return centroids


def compute_prompt_sensitivity(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    PSSI:
    동일 문제·동일 모델에서
    프롬프트별 구조 중심점 사이의 평균 거리이다.

    값이 클수록 프롬프트 변화에 민감하다.
    """

    centroids = compute_prompt_centroids(
        dataframe
    )

    standardized_features = [
        f"z_{feature}"
        for feature in STRUCTURAL_FEATURES
    ]

    group_columns = [
        "problem_id",
        "task_id",
        "model_name",
        "model_id",
        "provider",
    ]

    rows: list[dict[str, Any]] = []

    for group_keys, group in centroids.groupby(
        group_columns,
        dropna=False,
    ):
        matrix = (
            group[standardized_features]
            .dropna()
            .to_numpy(dtype=float)
        )

        distances = (
            pairwise_euclidean_distances(
                matrix
            )
        )

        average_prompt_distance = (
            float(np.mean(distances))
            if distances
            else 0.0
        )

        row = dict(
            zip(
                group_columns,
                group_keys,
            )
        )

        row.update(
            {
                "num_prompt_types":
                    len(group),
                "num_valid_prompt_vectors":
                    len(matrix),
                "num_prompt_pairs":
                    len(distances),
                "avg_prompt_distance":
                    average_prompt_distance,
                "pssi":
                    average_prompt_distance,
                "success_rate":
                    float(
                        group[
                            "prompt_success_rate"
                        ].mean()
                    ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def compute_structural_diversity(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    SDS:
    동일 문제·동일 모델에서 생성된 전체 코드의
    구조 벡터가 중심점에서 얼마나 퍼져 있는지 측정한다.

    각 표준화 구조 벡터와 그룹 중심점 사이의
    평균 Euclidean distance를 SDS로 사용한다.
    """

    valid_df = dataframe[
        dataframe["ast_success"]
    ].copy()

    standardized_features = [
        f"z_{feature}"
        for feature in STRUCTURAL_FEATURES
    ]

    group_columns = [
        "problem_id",
        "task_id",
        "model_name",
        "model_id",
        "provider",
    ]

    rows: list[dict[str, Any]] = []

    for group_keys, group in valid_df.groupby(
        group_columns,
        dropna=False,
    ):
        matrix = (
            group[standardized_features]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(matrix) == 0:
            structural_variance = np.nan
            sds = np.nan
        else:
            centroid = matrix.mean(axis=0)

            centroid_distances = (
                np.linalg.norm(
                    matrix - centroid,
                    axis=1,
                )
            )

            sds = float(
                np.mean(
                    centroid_distances
                )
            )

            structural_variance = float(
                np.mean(
                    np.var(
                        matrix,
                        axis=0,
                        ddof=0,
                    )
                )
            )

        row = dict(
            zip(
                group_columns,
                group_keys,
            )
        )

        row.update(
            {
                "num_samples":
                    len(group),
                "num_valid_vectors":
                    len(matrix),
                "structural_variance":
                    structural_variance,
                "sds":
                    sds,
                "success_rate":
                    float(
                        group[
                            "functional_passed"
                        ].mean()
                    ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def compute_model_summary(
    metrics_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
) -> pd.DataFrame:
    structural_summary = (
        metrics_df[
            metrics_df["ast_success"]
        ]
        .groupby(
            [
                "model_name",
                "model_id",
                "provider",
            ],
            as_index=False,
        )[STRUCTURAL_FEATURES]
        .mean()
    )

    functional_summary = (
        metrics_df.groupby(
            [
                "model_name",
                "model_id",
                "provider",
            ],
            as_index=False,
        )
        .agg(
            num_samples=(
                "file",
                "count",
            ),
            ast_success_rate=(
                "ast_success",
                "mean",
            ),
            functional_pass_rate=(
                "functional_passed",
                "mean",
            ),
        )
    )

    stability_summary = (
        stability_df.groupby(
            [
                "model_name",
                "model_id",
                "provider",
            ],
            as_index=False,
        )
        .agg(
            ssi=(
                "ssi",
                "mean",
            ),
            avg_repeat_distance=(
                "avg_repeat_distance",
                "mean",
            ),
        )
    )

    sensitivity_summary = (
        sensitivity_df.groupby(
            [
                "model_name",
                "model_id",
                "provider",
            ],
            as_index=False,
        )
        .agg(
            pssi=(
                "pssi",
                "mean",
            ),
        )
    )

    diversity_summary = (
        diversity_df.groupby(
            [
                "model_name",
                "model_id",
                "provider",
            ],
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

    summary = functional_summary.merge(
        structural_summary,
        on=[
            "model_name",
            "model_id",
            "provider",
        ],
        how="left",
    )

    summary = summary.merge(
        stability_summary,
        on=[
            "model_name",
            "model_id",
            "provider",
        ],
        how="left",
    )

    summary = summary.merge(
        sensitivity_summary,
        on=[
            "model_name",
            "model_id",
            "provider",
        ],
        how="left",
    )

    summary = summary.merge(
        diversity_summary,
        on=[
            "model_name",
            "model_id",
            "provider",
        ],
        how="left",
    )

    return summary


def compute_prompt_summary(
    metrics_df: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> pd.DataFrame:
    structural_summary = (
        metrics_df[
            metrics_df["ast_success"]
        ]
        .groupby(
            "prompt_name",
            as_index=False,
        )[STRUCTURAL_FEATURES]
        .mean()
    )

    functional_summary = (
        metrics_df.groupby(
            "prompt_name",
            as_index=False,
        )
        .agg(
            num_samples=(
                "file",
                "count",
            ),
            ast_success_rate=(
                "ast_success",
                "mean",
            ),
            success_rate=(
                "functional_passed",
                "mean",
            ),
        )
    )

    stability_summary = (
        stability_df.groupby(
            "prompt_name",
            as_index=False,
        )
        .agg(
            avg_distance=(
                "avg_repeat_distance",
                "mean",
            ),
            ssi=(
                "ssi",
                "mean",
            ),
        )
    )

    summary = functional_summary.merge(
        structural_summary,
        on="prompt_name",
        how="left",
    )

    summary = summary.merge(
        stability_summary,
        on="prompt_name",
        how="left",
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
    ensure_dirs()

    metric_df = load_metric_records()
    functional_df = (
        load_functional_summary()
    )

    merged_df = (
        merge_metric_and_functional_data(
            metric_df=metric_df,
            functional_df=functional_df,
        )
    )

    merged_df = (
        add_standardized_features(
            merged_df
        )
    )

    stability_df = (
        compute_repeat_stability(
            merged_df
        )
    )

    sensitivity_df = (
        compute_prompt_sensitivity(
            merged_df
        )
    )

    diversity_df = (
        compute_structural_diversity(
            merged_df
        )
    )

    model_summary_df = (
        compute_model_summary(
            metrics_df=merged_df,
            stability_df=stability_df,
            sensitivity_df=sensitivity_df,
            diversity_df=diversity_df,
        )
    )

    prompt_summary_df = (
        compute_prompt_summary(
            metrics_df=merged_df,
            stability_df=stability_df,
        )
    )

    save_dataframe(
        merged_df,
        METRICS_SUMMARY_PATH,
    )

    save_dataframe(
        stability_df,
        REPEAT_STABILITY_PATH,
    )

    save_dataframe(
        sensitivity_df,
        PROMPT_SENSITIVITY_PATH,
    )

    save_dataframe(
        diversity_df,
        STRUCTURAL_DIVERSITY_PATH,
    )

    save_dataframe(
        model_summary_df,
        MODEL_SUMMARY_PATH,
    )

    save_dataframe(
        prompt_summary_df,
        PROMPT_SUMMARY_PATH,
    )

    print(
        "\n===== Dataset Summary ====="
    )
    print(
        f"Total samples: "
        f"{len(merged_df)}"
    )
    print(
        f"AST success samples: "
        f"{merged_df['ast_success'].sum()}"
    )
    print(
        f"Functional passes: "
        f"{merged_df['functional_passed'].sum()}"
    )
    print(
        f"Functional pass rate: "
        f"{merged_df['functional_passed'].mean():.6f}"
    )

    print(
        "\n===== Model Summary ====="
    )
    print(
        model_summary_df.to_string(
            index=False
        )
    )

    print(
        "\n===== Prompt Summary ====="
    )
    print(
        prompt_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nMetric computation completed."
    )


if __name__ == "__main__":
    main()