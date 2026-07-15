from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
TABLE_DIR = ROOT_DIR / "reports" / "tables"

MODEL_SUMMARY_PATH = RESULTS_DIR / "model_summary.csv"
PROMPT_SUMMARY_PATH = RESULTS_DIR / "prompt_summary.csv"
STATISTICAL_TESTS_PATH = RESULTS_DIR / "statistical_tests.csv"
PAIRWISE_TESTS_PATH = RESULTS_DIR / "pairwise_tests.csv"
CORRELATION_ANALYSIS_PATH = RESULTS_DIR / "correlation_analysis.csv"
PASSED_ONLY_MODEL_SUMMARY_PATH = RESULTS_DIR / "passed_only_model_summary.csv"

TOP_PROMPT_SENSITIVITY_PATH = RESULTS_DIR / "top_prompt_sensitivity_problems.csv"
TOP_STRUCTURAL_DIVERSITY_PATH = RESULTS_DIR / "top_structural_diversity_problems.csv"
PROBLEM_COMPLEXITY_CORRELATION_PATH = RESULTS_DIR / "problem_complexity_correlation.csv"
DISTANCE_FUNCTION_SENSITIVITY_PATH = RESULTS_DIR / "distance_function_sensitivity.csv"
STRUCTURE_AWARE_RERANKING_PATH = RESULTS_DIR / "structure_aware_reranking.csv"
FEATURE_ABLATION_SUMMARY_PATH = RESULTS_DIR / "feature_ablation_summary.csv"
MODEL_BOOTSTRAP_CI_PATH = RESULTS_DIR / "model_metric_bootstrap_ci.csv"


MODEL_LABELS = {
    "claude_model": "Claude Sonnet 4.6",
    "gpt5_model": "GPT-5 mini",
    "deepseek_model": "DeepSeek Chat",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(path)


def escape_latex(value: Any) -> str:
    text = str(value)

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def format_number(
    value: Any,
    digits: int = 3,
) -> str:
    if pd.isna(value):
        return "-"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    try:
        number = float(value)
    except Exception:
        return escape_latex(value)

    return f"{number:.{digits}f}"


def format_p_value(value: Any) -> str:
    if pd.isna(value):
        return "-"

    try:
        number = float(value)
    except Exception:
        return escape_latex(value)

    if number < 0.001:
        return f"{number:.2e}"

    return f"{number:.4f}"


def write_latex_table(
    filename: str,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
) -> None:
    output_path = TABLE_DIR / filename

    column_format = "l" + "c" * (len(headers) - 1)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{column_format}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for row in rows:
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved: {output_path}")


def export_experimental_setup_table() -> None:
    headers = [
        "Item",
        "Setting",
    ]

    rows = [
        [
            "Benchmark",
            "Official HumanEval",
        ],
        [
            "Problem type",
            "Python function-level programming problems",
        ],
        [
            "Number of problems",
            "164",
        ],
        [
            "Number of models",
            "3",
        ],
        [
            "Models",
            "Claude Sonnet 4.6, GPT-5 mini, DeepSeek Chat",
        ],
        [
            "Prompt types",
            "basic, concise, readable, optimized, constraint",
        ],
        [
            "Number of repetitions",
            "5 per condition",
        ],
        [
            "Total generated codes",
            "12,300",
        ],
        [
            "AST-parsable samples",
            "12,287 / 12,300",
        ],
        [
            "Functional passes",
            "11,766 / 12,300",
        ],
        [
            "Main metrics",
            "SSI, PSSI, SDS",
        ],
    ]

    write_latex_table(
        filename="table3_experimental_setup.tex",
        caption="Experimental setup",
        label="tab:experimental_setup",
        headers=headers,
        rows=rows,
    )


def export_model_summary_table() -> None:
    dataframe = load_csv(MODEL_SUMMARY_PATH)

    dataframe["Model"] = (
        dataframe["model_name"]
        .map(MODEL_LABELS)
        .fillna(dataframe["model_name"])
    )

    dataframe = dataframe.sort_values(
        "functional_pass_rate",
        ascending=False,
    )

    headers = [
        "Model",
        "Pass Rate",
        "AST Rate",
        "Depth",
        "Branch",
        "Loop",
        "Func.",
        "CFR",
        "SSI",
        "PSSI",
        "SDS",
    ]

    rows = []

    for _, row in dataframe.iterrows():
        rows.append(
            [
                escape_latex(row["Model"]),
                format_number(row["functional_pass_rate"]),
                format_number(row["ast_success_rate"]),
                format_number(row["ast_depth"]),
                format_number(row["branch_count"]),
                format_number(row["loop_count"]),
                format_number(row["function_count"]),
                format_number(row["control_flow_ratio"]),
                format_number(row["ssi"]),
                format_number(row["pssi"]),
                format_number(row["sds"]),
            ]
        )

    write_latex_table(
        filename="table5_model_structural_reliability.tex",
        caption="Model-wise structural reliability comparison",
        label="tab:model_structural_reliability",
        headers=headers,
        rows=rows,
    )


def export_prompt_summary_table() -> None:
    dataframe = load_csv(PROMPT_SUMMARY_PATH)

    prompt_order = [
        "basic",
        "concise",
        "constraint",
        "optimized",
        "readable",
    ]

    dataframe["prompt_name"] = pd.Categorical(
        dataframe["prompt_name"],
        categories=prompt_order,
        ordered=True,
    )

    dataframe = dataframe.sort_values(
        "prompt_name"
    )

    headers = [
        "Prompt",
        "Pass Rate",
        "AST Rate",
        "Avg. Distance",
        "SSI",
        "Depth",
        "Branch",
        "Loop",
        "CFR",
    ]

    rows = []

    for _, row in dataframe.iterrows():
        rows.append(
            [
                escape_latex(row["prompt_name"]),
                format_number(row["success_rate"]),
                format_number(row["ast_success_rate"]),
                format_number(row["avg_distance"]),
                format_number(row["ssi"]),
                format_number(row["ast_depth"]),
                format_number(row["branch_count"]),
                format_number(row["loop_count"]),
                format_number(row["control_flow_ratio"]),
            ]
        )

    write_latex_table(
        filename="table6_prompt_structural_stability.tex",
        caption="Structural stability by prompt type",
        label="tab:prompt_structural_stability",
        headers=headers,
        rows=rows,
    )


def export_top_prompt_sensitivity_table() -> None:
    dataframe = load_csv(TOP_PROMPT_SENSITIVITY_PATH)

    headers = [
        "Problem",
        "PSSI",
        "Success Rate",
    ]

    rows = []

    for _, row in dataframe.head(10).iterrows():
        rows.append(
            [
                escape_latex(row["problem_label"]),
                format_number(row["pssi"]),
                format_number(row["success_rate"]),
            ]
        )

    write_latex_table(
        filename="table7_top_prompt_sensitivity.tex",
        caption="Top-10 problems by prompt sensitivity",
        label="tab:top_prompt_sensitivity",
        headers=headers,
        rows=rows,
    )


def export_top_structural_diversity_table() -> None:
    dataframe = load_csv(TOP_STRUCTURAL_DIVERSITY_PATH)

    headers = [
        "Problem",
        "SDS",
        "Structural Variance",
        "Success Rate",
    ]

    rows = []

    for _, row in dataframe.head(10).iterrows():
        rows.append(
            [
                escape_latex(row["problem_label"]),
                format_number(row["sds"]),
                format_number(row["structural_variance"]),
                format_number(row["success_rate"]),
            ]
        )

    write_latex_table(
        filename="table8_top_structural_diversity.tex",
        caption="Top-10 problems by structural diversity",
        label="tab:top_structural_diversity",
        headers=headers,
        rows=rows,
    )


def export_statistical_tests_table() -> None:
    dataframe = load_csv(STATISTICAL_TESTS_PATH)

    headers = [
        "Comparison",
        "Metric",
        "Blocks",
        "Friedman $\\chi^2$",
        "p-value",
        "Kendall's $W$",
    ]

    rows = []

    for _, row in dataframe.iterrows():
        rows.append(
            [
                escape_latex(row["comparison"]),
                escape_latex(row["metric"]),
                str(int(row["num_blocks"])),
                format_number(row["statistic"]),
                format_p_value(row["p_value"]),
                format_number(row["kendalls_w"]),
            ]
        )

    write_latex_table(
        filename="table9_statistical_tests.tex",
        caption=(
            "Blocked repeated-measures analysis of structural "
            "reliability measures"
        ),
        label="tab:statistical_tests",
        headers=headers,
        rows=rows,
    )


def export_pairwise_tests_table() -> None:
    dataframe = load_csv(PAIRWISE_TESTS_PATH)
    dataframe = dataframe[
        dataframe["significant_0_05_holm"] == True
    ].copy()

    headers = [
        "Comparison",
        "Metric",
        "Group A",
        "Group B",
        "Pairs",
        "Holm $p$",
    ]

    rows = []

    for _, row in dataframe.iterrows():
        rows.append(
            [
                escape_latex(row["comparison"]),
                escape_latex(row["metric"]),
                escape_latex(row["group_a"]),
                escape_latex(row["group_b"]),
                str(int(row["n_pairs"])),
                format_p_value(row["p_value_holm"]),
            ]
        )

    write_latex_table(
        filename="table9b_pairwise_tests.tex",
        caption=(
            "Significant Holm-adjusted Wilcoxon signed-rank comparisons"
        ),
        label="tab:pairwise_tests",
        headers=headers,
        rows=rows,
    )


def export_correlation_table() -> None:
    dataframe = load_csv(CORRELATION_ANALYSIS_PATH)

    headers = [
        "Source",
        "Relationship",
        "N",
        "Spearman $\\rho$",
        "p-value",
    ]

    rows = []

    for _, row in dataframe.iterrows():
        relationship = (
            f"{row['x_metric']} vs. "
            f"{row['y_metric']}"
        )

        rows.append(
            [
                escape_latex(row["source"]),
                escape_latex(relationship),
                str(int(row["n"])),
                format_number(row["spearman_rho"]),
                format_p_value(row["p_value"]),
            ]
        )

    write_latex_table(
        filename="table10_correlation_analysis.tex",
        caption="Correlation analysis between functional and structural metrics",
        label="tab:correlation_analysis",
        headers=headers,
        rows=rows,
    )


def export_passed_only_table() -> None:
    dataframe = load_csv(PASSED_ONLY_MODEL_SUMMARY_PATH)

    dataframe["Model"] = (
        dataframe["model_name"]
        .map(MODEL_LABELS)
        .fillna(dataframe["model_name"])
    )

    dataframe = dataframe.sort_values(
        "num_passed_ast_samples",
        ascending=False,
    )

    headers = [
        "Model",
        "Passed AST Samples",
        "Depth",
        "Branch",
        "Loop",
        "Func.",
        "CFR",
    ]

    rows = []

    for _, row in dataframe.iterrows():
        rows.append(
            [
                escape_latex(row["Model"]),
                str(int(row["num_passed_ast_samples"])),
                format_number(row["ast_depth"]),
                format_number(row["branch_count"]),
                format_number(row["loop_count"]),
                format_number(row["function_count"]),
                format_number(row["control_flow_ratio"]),
            ]
        )

    write_latex_table(
        filename="table11_passed_only_structural_characteristics.tex",
        caption="Structural characteristics among functionally correct codes",
        label="tab:passed_only_structural_characteristics",
        headers=headers,
        rows=rows,
    )


def export_problem_complexity_correlation_table() -> None:
    dataframe = load_csv(PROBLEM_COMPLEXITY_CORRELATION_PATH)

    headers = [
        "Relationship",
        "N",
        "Spearman $\\rho$",
        "p-value",
        "Significant",
    ]

    rows = []

    for _, row in dataframe.iterrows():
        relationship = (
            f"{row['x_metric']} vs. "
            f"{row['y_metric']}"
        )

        rows.append(
            [
                escape_latex(relationship),
                str(int(row["n"])),
                format_number(row["spearman_rho"]),
                format_p_value(row["p_value"]),
                "Yes" if row["significant_0_05"] else "No",
            ]
        )

    write_latex_table(
        filename="table12_problem_complexity_correlation.tex",
        caption="Correlation between problem complexity and structural sensitivity",
        label="tab:problem_complexity_correlation",
        headers=headers,
        rows=rows,
    )


def export_distance_function_sensitivity_table() -> None:
    dataframe = load_csv(DISTANCE_FUNCTION_SENSITIVITY_PATH)

    headers = [
        "Prompt",
        "Cosine SSI",
        "Euclidean SSI",
        "Manhattan SSI",
    ]

    rows = []

    for _, row in dataframe.iterrows():
        rows.append(
            [
                escape_latex(row["prompt_name"]),
                format_number(row["cosine_ssi"]),
                format_number(row["euclidean_ssi"]),
                format_number(row["manhattan_ssi"]),
            ]
        )

    write_latex_table(
        filename="table13_distance_function_sensitivity.tex",
        caption="SSI comparison under different distance functions",
        label="tab:distance_function_sensitivity",
        headers=headers,
        rows=rows,
    )


def export_structure_aware_reranking_table() -> None:
    dataframe = load_csv(STRUCTURE_AWARE_RERANKING_PATH)

    row = dataframe.iloc[0]

    headers = [
        "Metric",
        "Value",
    ]

    rows = [
        [
            "Number of groups",
            str(int(row["num_groups"])),
        ],
        [
            "Baseline complexity",
            format_number(row["baseline_complexity"]),
        ],
        [
            "Reranked complexity",
            format_number(row["reranked_complexity"]),
        ],
        [
            "Average improvement",
            format_number(row["average_improvement"]),
        ],
        [
            "Improved groups",
            str(int(row["improved_groups"])),
        ],
        [
            "Improved ratio",
            format_number(row["improved_ratio"]),
        ],
    ]

    write_latex_table(
        filename="table14_structure_aware_reranking.tex",
        caption="Effect of structure-aware code reranking",
        label="tab:structure_aware_reranking",
        headers=headers,
        rows=rows,
    )



def format_estimate_ci(
    mean_value: Any,
    lower_value: Any,
    upper_value: Any,
) -> str:
    if any(
        pd.isna(value)
        for value in [mean_value, lower_value, upper_value]
    ):
        return "-"

    return (
        f"{float(mean_value):.3f} "
        f"[{float(lower_value):.3f}, {float(upper_value):.3f}]"
    )


def export_feature_ablation_table() -> None:
    dataframe = load_csv(FEATURE_ABLATION_SUMMARY_PATH)
    dataframe = dataframe[
        dataframe["excluded_feature"] != "none"
    ].copy()

    headers = [
        "Excluded Feature",
        "SSI $\\rho$",
        "PSSI $\\rho$",
        "SDS $\\rho$",
    ]

    rows = []

    for _, row in dataframe.iterrows():
        rows.append(
            [
                escape_latex(row["excluded_feature"]),
                format_number(row["ssi_rank_correlation"]),
                format_number(row["pssi_rank_correlation"]),
                format_number(row["sds_rank_correlation"]),
            ]
        )

    write_latex_table(
        filename="table_feature_ablation.tex",
        caption="Leave-one-feature-out robustness analysis",
        label="tab:feature_ablation",
        headers=headers,
        rows=rows,
    )


def export_model_bootstrap_ci_table() -> None:
    dataframe = load_csv(MODEL_BOOTSTRAP_CI_PATH)

    pivoted = dataframe.pivot_table(
        index="model_name",
        columns="metric",
        values=["mean", "ci_lower", "ci_upper"],
        aggfunc="first",
    )

    headers = [
        "Model",
        "SSI [95\\% CI]",
        "PSSI [95\\% CI]",
        "SDS [95\\% CI]",
    ]

    rows = []

    for model_name in MODEL_LABELS:
        if model_name not in pivoted.index:
            continue

        row = pivoted.loc[model_name]
        rows.append(
            [
                escape_latex(MODEL_LABELS[model_name]),
                format_estimate_ci(
                    row[("mean", "ssi")],
                    row[("ci_lower", "ssi")],
                    row[("ci_upper", "ssi")],
                ),
                format_estimate_ci(
                    row[("mean", "pssi")],
                    row[("ci_lower", "pssi")],
                    row[("ci_upper", "pssi")],
                ),
                format_estimate_ci(
                    row[("mean", "sds")],
                    row[("ci_lower", "sds")],
                    row[("ci_upper", "sds")],
                ),
            ]
        )

    write_latex_table(
        filename="table_model_bootstrap_ci.tex",
        caption=(
            "Problem-cluster bootstrap confidence intervals for "
            "model-level structural measures"
        ),
        label="tab:model_bootstrap_ci",
        headers=headers,
        rows=rows,
    )


def main() -> None:
    ensure_dirs()

    export_experimental_setup_table()
    export_model_summary_table()
    export_prompt_summary_table()
    export_top_prompt_sensitivity_table()
    export_top_structural_diversity_table()
    export_statistical_tests_table()
    export_pairwise_tests_table()
    export_correlation_table()
    export_passed_only_table()
    export_problem_complexity_correlation_table()
    export_distance_function_sensitivity_table()
    export_structure_aware_reranking_table()
    export_feature_ablation_table()
    export_model_bootstrap_ci_table()

    print("\nLaTeX table export completed.")
    print(f"Tables saved to: {TABLE_DIR}")


if __name__ == "__main__":
    main()