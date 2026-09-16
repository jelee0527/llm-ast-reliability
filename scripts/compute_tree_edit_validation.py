"""Validate descriptor-based SSI with normalized reduced-AST edit distance."""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from apted import APTED
from apted.helpers import Tree
from scipy.stats import spearmanr


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT_DIR / "outputs" / "raw"
DEFAULT_REPEAT_STABILITY_PATH = ROOT_DIR / "results" / "repeat_stability.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results"

# Identifiers, literal values, contexts, and operator tokens are deliberately
# normalized away. The retained nodes still preserve ordered tree structure,
# statement nesting, control flow, calls, and expression/comprehension shape.
RETAINED_NODE_TYPES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Return,
    ast.Delete,
    ast.Assign,
    ast.TypeAlias,
    ast.AugAssign,
    ast.AnnAssign,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.Match,
    ast.Raise,
    ast.Try,
    ast.TryStar,
    ast.Assert,
    ast.Import,
    ast.ImportFrom,
    ast.Expr,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.BoolOp,
    ast.NamedExpr,
    ast.BinOp,
    ast.UnaryOp,
    ast.Lambda,
    ast.IfExp,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.Compare,
    ast.Call,
    ast.FormattedValue,
    ast.JoinedStr,
    ast.Attribute,
    ast.Subscript,
    ast.Starred,
    ast.List,
    ast.Tuple,
    ast.Slice,
    ast.comprehension,
    ast.ExceptHandler,
    ast.match_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--repeat-stability",
        type=Path,
        default=DEFAULT_REPEAT_STABILITY_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Limit groups for a smoke test; zero processes all groups.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 2) - 1)),
        help="Worker processes used for condition-level tree comparisons.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def reduce_ast_node(node: ast.AST) -> list[Tree]:
    children: list[Tree] = []
    for child in ast.iter_child_nodes(node):
        children.extend(reduce_ast_node(child))

    if isinstance(node, RETAINED_NODE_TYPES):
        return [Tree(type(node).__name__, *children)]
    return children


def make_reduced_tree(code: str) -> Tree:
    parsed = ast.parse(code)
    reduced = reduce_ast_node(parsed)
    if len(reduced) != 1:
        raise ValueError(f"Expected one reduced AST root, found {len(reduced)}")
    return reduced[0]


def count_tree_nodes(tree: Tree) -> int:
    return 1 + sum(count_tree_nodes(child) for child in tree.children)


def normalized_tree_edit_distance(
    first: tuple[Tree, int],
    second: tuple[Tree, int],
) -> float:
    first_tree, first_size = first
    second_tree, second_size = second
    denominator = max(first_size, second_size, 1)
    return float(APTED(first_tree, second_tree).compute_edit_distance()) / denominator


def load_grouped_trees(
    raw_dir: Path,
) -> dict[tuple[str, str, str], list[tuple[Tree, int]]]:
    groups: dict[tuple[str, str, str], list[tuple[Tree, int]]] = defaultdict(list)

    for index, path in enumerate(sorted(raw_dir.glob("*.json")), start=1):
        payload = load_json(path)
        code = payload.get("generated_code")
        if not isinstance(code, str):
            continue

        try:
            tree = make_reduced_tree(code)
        except (SyntaxError, ValueError):
            continue

        key = (
            str(payload["problem_id"]),
            str(payload["model_name"]),
            str(payload["prompt_name"]),
        )
        groups[key].append((tree, count_tree_nodes(tree)))

        if index % 1_000 == 0:
            print(f"Loaded reduced ASTs: {index}")

    return dict(groups)


def compute_group_row(
    item: tuple[tuple[str, str, str], list[tuple[Tree, int]]],
) -> dict[str, Any]:
    key, trees = item
    distances = [
        normalized_tree_edit_distance(first, second)
        for first, second in combinations(trees, 2)
    ]
    average_distance = float(sum(distances) / len(distances)) if distances else None
    problem_id, model_name, prompt_name = key
    return {
        "problem_id": problem_id,
        "model_name": model_name,
        "prompt_name": prompt_name,
        "num_valid_trees": len(trees),
        "num_pairs": len(distances),
        "avg_normalized_tree_edit_distance": average_distance,
        "tree_edit_stability": (
            1.0 - average_distance if average_distance is not None else None
        ),
    }


def compute_group_distances(
    grouped_trees: dict[tuple[str, str, str], list[tuple[Tree, int]]],
    max_groups: int,
    workers: int,
) -> pd.DataFrame:
    items = sorted(grouped_trees.items())
    if max_groups > 0:
        items = items[:max_groups]

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        iterator = executor.map(compute_group_row, items, chunksize=8)
        for index, row in enumerate(iterator, start=1):
            rows.append(row)

            if index % 250 == 0 or index == len(items):
                print(f"Computed tree-edit groups: {index}/{len(items)}", flush=True)

    return pd.DataFrame(rows)


def make_validation_summary(
    tree_edit: pd.DataFrame,
    repeat_stability_path: Path,
) -> pd.DataFrame:
    descriptor = pd.read_csv(repeat_stability_path)
    merge_keys = ["problem_id", "model_name", "prompt_name"]
    merged = descriptor[merge_keys + ["ssi"]].merge(
        tree_edit,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    valid = merged[["ssi", "tree_edit_stability"]].dropna()
    rho, p_value = spearmanr(valid["ssi"], valid["tree_edit_stability"])
    return pd.DataFrame(
        [
            {
                "comparison": "descriptor_ssi_vs_tree_edit_stability",
                "n": len(valid),
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            }
        ]
    )


def make_group_summary(tree_edit: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        tree_edit.groupby(column, dropna=False)
        .agg(
            num_conditions=("problem_id", "size"),
            valid_conditions=("tree_edit_stability", "count"),
            mean_normalized_tree_edit_distance=(
                "avg_normalized_tree_edit_distance",
                "mean",
            ),
            mean_tree_edit_stability=("tree_edit_stability", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    grouped_trees = load_grouped_trees(args.raw_dir)
    tree_edit = compute_group_distances(
        grouped_trees,
        args.max_groups,
        args.workers,
    )
    validation = make_validation_summary(tree_edit, args.repeat_stability)
    model_summary = make_group_summary(tree_edit, "model_name")
    prompt_summary = make_group_summary(tree_edit, "prompt_name")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tree_edit_repeat_stability.csv": tree_edit,
        "tree_edit_validation_summary.csv": validation,
        "tree_edit_model_summary.csv": model_summary,
        "tree_edit_prompt_summary.csv": prompt_summary,
    }
    for filename, dataframe in outputs.items():
        path = args.output_dir / filename
        dataframe.to_csv(path, index=False)
        print(f"Saved: {path}")

    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
