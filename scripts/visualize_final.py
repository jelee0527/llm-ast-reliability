from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"

MODEL_SUMMARY_PATH = RESULTS_DIR / "model_summary.csv"
PROMPT_SUMMARY_PATH = RESULTS_DIR / "prompt_summary.csv"
REPEAT_STABILITY_PATH = RESULTS_DIR / "repeat_stability.csv"
PROMPT_SENSITIVITY_PATH = RESULTS_DIR / "prompt_sensitivity.csv"
STRUCTURAL_DIVERSITY_PATH = RESULTS_DIR / "structural_diversity.csv"

TOP_PROMPT_SENSITIVITY_PATH = RESULTS_DIR / "top_prompt_sensitivity_problems.csv"
TOP_STRUCTURAL_DIVERSITY_PATH = RESULTS_DIR / "top_structural_diversity_problems.csv"


MODEL_LABELS = {
    "claude_model": "Claude Sonnet 4.6",
    "gpt5_model": "GPT-5 mini",
    "deepseek_model": "DeepSeek Chat",
}

PROMPT_ORDER = [
    "basic",
    "concise",
    "constraint",
    "optimized",
    "readable",
]


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(path)


def apply_model_labels(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    if "model_name" in result.columns:
        result["model_label"] = (
            result["model_name"]
            .map(MODEL_LABELS)
            .fillna(result["model_name"])
        )

    return result


def save_figure(
    filename: str,
) -> None:
    output_path = FIGURE_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path}")


def plot_functional_vs_structural_by_model(
    model_summary_df: pd.DataFrame,
) -> None:
    dataframe = apply_model_labels(
        model_summary_df
    )

    dataframe = dataframe.sort_values(
        "functional_pass_rate",
        ascending=False,
    )

    plot_df = dataframe[
        [
            "model_label",
            "functional_pass_rate",
            "ssi",
        ]
    ].set_index("model_label")

    ax = plot_df.plot(
        kind="bar",
        figsize=(8, 5),
        rot=0,
    )

    ax.set_title(
        "Functional correctness and structural stability by generation condition"
    )
    ax.set_xlabel("Generation condition")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(
        [
            "Functional pass rate",
            "Structural Stability Index (SSI)",
        ],
        loc="lower right",
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f",
            fontsize=8,
            padding=2,
        )

    save_figure(
        "fig1_functional_vs_structural_by_model.png"
    )


def plot_model_structural_metrics(
    model_summary_df: pd.DataFrame,
) -> None:
    dataframe = apply_model_labels(
        model_summary_df
    )

    dataframe = dataframe.sort_values(
        "ssi",
        ascending=False,
    )

    plot_df = dataframe[
        [
            "model_label",
            "ssi",
            "pssi",
            "sds",
        ]
    ].set_index("model_label")

    ax = plot_df.plot(
        kind="bar",
        figsize=(8, 5),
        rot=0,
    )

    ax.set_title(
        "AST-based structural reliability metrics by generation condition"
    )
    ax.set_xlabel("Generation condition")
    ax.set_ylabel("Metric value")
    ax.legend(
        [
            "SSI",
            "PSSI",
            "SDS",
        ],
        loc="upper left",
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f",
            fontsize=8,
            padding=2,
        )

    save_figure(
        "fig2_model_structural_metrics.png"
    )


def plot_prompt_success_and_ssi(
    prompt_summary_df: pd.DataFrame,
) -> None:
    dataframe = prompt_summary_df.copy()

    dataframe["prompt_name"] = pd.Categorical(
        dataframe["prompt_name"],
        categories=PROMPT_ORDER,
        ordered=True,
    )

    dataframe = dataframe.sort_values(
        "prompt_name"
    )

    plot_df = dataframe[
        [
            "prompt_name",
            "success_rate",
            "ssi",
        ]
    ].set_index("prompt_name")

    ax = plot_df.plot(
        kind="bar",
        figsize=(8, 5),
        rot=0,
    )

    ax.set_title(
        "Functional pass rate and structural stability by prompt type"
    )
    ax.set_xlabel("Prompt type")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(
        [
            "Functional pass rate",
            "SSI",
        ],
        loc="lower right",
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f",
            fontsize=8,
            padding=2,
        )

    save_figure(
        "fig3_prompt_success_and_ssi.png"
    )


def plot_success_rate_vs_ssi_scatter(
    repeat_stability_df: pd.DataFrame,
) -> None:
    dataframe = apply_model_labels(
        repeat_stability_df
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for model_label, group in dataframe.groupby(
        "model_label"
    ):
        ax.scatter(
            group["success_rate"],
            group["ssi"],
            label=model_label,
            alpha=0.55,
            s=20,
        )

    ax.set_title(
        "Relationship between functional success rate and structural stability"
    )
    ax.set_xlabel("Functional success rate")
    ax.set_ylabel("Structural Stability Index (SSI)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(
        loc="lower left",
        fontsize=8,
    )
    ax.grid(
        True,
        alpha=0.3,
    )

    save_figure(
        "fig4_success_rate_vs_ssi_scatter.png"
    )


def get_problem_label(
    row: pd.Series,
) -> str:
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


def build_top_prompt_sensitivity(
    prompt_sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = prompt_sensitivity_df.copy()

    dataframe["problem_label"] = dataframe.apply(
        get_problem_label,
        axis=1,
    )

    grouped = (
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
    )

    top_df = grouped.head(10)

    top_df.to_csv(
        TOP_PROMPT_SENSITIVITY_PATH,
        index=False,
    )

    print(
        f"Saved: {TOP_PROMPT_SENSITIVITY_PATH}"
    )

    return top_df


def plot_top_prompt_sensitivity(
    prompt_sensitivity_df: pd.DataFrame,
) -> None:
    top_df = build_top_prompt_sensitivity(
        prompt_sensitivity_df
    )

    plot_df = top_df.sort_values(
        "pssi",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        plot_df["problem_label"],
        plot_df["pssi"],
    )

    ax.set_title(
        "Top-10 problems by prompt sensitivity"
    )
    ax.set_xlabel(
        "Prompt Sensitivity Index (PSSI)"
    )
    ax.set_ylabel("Problem")

    for index, value in enumerate(
        plot_df["pssi"]
    ):
        ax.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
            fontsize=8,
        )

    save_figure(
        "fig5_top_prompt_sensitivity.png"
    )


def build_top_structural_diversity(
    structural_diversity_df: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = structural_diversity_df.copy()

    dataframe["problem_label"] = dataframe.apply(
        get_problem_label,
        axis=1,
    )

    grouped = (
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
    )

    top_df = grouped.head(10)

    top_df.to_csv(
        TOP_STRUCTURAL_DIVERSITY_PATH,
        index=False,
    )

    print(
        f"Saved: {TOP_STRUCTURAL_DIVERSITY_PATH}"
    )

    return top_df


def plot_top_structural_diversity(
    structural_diversity_df: pd.DataFrame,
) -> None:
    top_df = build_top_structural_diversity(
        structural_diversity_df
    )

    plot_df = top_df.sort_values(
        "sds",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        plot_df["problem_label"],
        plot_df["sds"],
    )

    ax.set_title(
        "Top-10 problems by structural diversity"
    )
    ax.set_xlabel(
        "Structural Diversity Score (SDS)"
    )
    ax.set_ylabel("Problem")

    for index, value in enumerate(
        plot_df["sds"]
    ):
        ax.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
            fontsize=8,
        )

    save_figure(
        "fig6_top_structural_diversity.png"
    )


def print_summary(
    model_summary_df: pd.DataFrame,
    prompt_summary_df: pd.DataFrame,
) -> None:
    print("\n===== Model Summary Used for Figures =====")
    print(
        model_summary_df[
            [
                "model_name",
                "functional_pass_rate",
                "ast_success_rate",
                "ssi",
                "pssi",
                "sds",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n===== Prompt Summary Used for Figures =====")
    print(
        prompt_summary_df[
            [
                "prompt_name",
                "success_rate",
                "ast_success_rate",
                "ssi",
            ]
        ].to_string(
            index=False
        )
    )


def main() -> None:
    ensure_dirs()

    model_summary_df = load_csv(
        MODEL_SUMMARY_PATH
    )

    prompt_summary_df = load_csv(
        PROMPT_SUMMARY_PATH
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

    plot_functional_vs_structural_by_model(
        model_summary_df
    )

    plot_model_structural_metrics(
        model_summary_df
    )

    plot_prompt_success_and_ssi(
        prompt_summary_df
    )

    plot_success_rate_vs_ssi_scatter(
        repeat_stability_df
    )

    plot_top_prompt_sensitivity(
        prompt_sensitivity_df
    )

    plot_top_structural_diversity(
        structural_diversity_df
    )

    print_summary(
        model_summary_df,
        prompt_summary_df,
    )

    print(
        "\nVisualization completed."
    )


if __name__ == "__main__":
    main()