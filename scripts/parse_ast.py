import ast
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "outputs" / "raw"
AST_DIR = ROOT_DIR / "outputs" / "ast"
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"


def ensure_dirs() -> None:
    AST_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def get_ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))

    if not children:
        return 1

    return 1 + max(
        get_ast_depth(child)
        for child in children
    )


def count_nodes(tree: ast.AST) -> int:
    return sum(1 for _ in ast.walk(tree))


def count_branch_nodes(tree: ast.AST) -> int:
    branch_types = (
        ast.If,
        ast.IfExp,
        ast.Match,
        ast.Try,
    )

    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, branch_types)
    )


def count_loop_nodes(tree: ast.AST) -> int:
    loop_types = (
        ast.For,
        ast.AsyncFor,
        ast.While,
    )

    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, loop_types)
    )


def count_function_nodes(tree: ast.AST) -> int:
    function_types = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
    )

    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, function_types)
    )


def count_control_flow_nodes(tree: ast.AST) -> int:
    control_flow_types = (
        ast.If,
        ast.IfExp,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.Break,
        ast.Continue,
        ast.Return,
        ast.Raise,
        ast.Match,
    )

    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, control_flow_types)
    )


def extract_function_names(
    tree: ast.AST,
) -> list[str]:
    names: list[str] = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            names.append(node.name)

    return names


def build_metrics(
    tree: ast.AST,
) -> dict[str, Any]:
    total_nodes = count_nodes(tree)
    control_flow_count = count_control_flow_nodes(tree)

    control_flow_ratio = (
        control_flow_count / total_nodes
        if total_nodes > 0
        else 0.0
    )

    return {
        "total_nodes": total_nodes,
        "ast_depth": get_ast_depth(tree),
        "branch_count": count_branch_nodes(tree),
        "loop_count": count_loop_nodes(tree),
        "function_count": count_function_nodes(tree),
        "control_flow_count": control_flow_count,
        "control_flow_ratio": round(
            control_flow_ratio,
            6,
        ),
        "function_names": extract_function_names(tree),
    }


def process_file(raw_path: Path) -> None:
    payload = load_json(raw_path)
    output_stem = raw_path.stem

    base_info = {
        "source_file": raw_path.name,
        "problem_id": payload.get("problem_id"),
        "task_id": payload.get("task_id"),
        "problem_name": payload.get("problem_name"),
        "entry_point": payload.get("entry_point"),
        "model_name": payload.get("model_name"),
        "model_id": payload.get("model_id"),
        "provider": payload.get("provider"),
        "prompt_name": payload.get("prompt_name"),
        "repeat_idx": payload.get("repeat_idx"),
    }

    if payload.get("status") != "success":
        error_payload = {
            **base_info,
            "status": "error",
            "error_type": "generation_error",
            "error_message": payload.get(
                "error_message",
                "Generation failed",
            ),
        }

        save_json(
            AST_DIR / f"{output_stem}.json",
            error_payload,
        )
        save_json(
            METRICS_DIR / f"{output_stem}.json",
            error_payload,
        )
        return

    generated_code = payload.get(
        "generated_code",
        "",
    ).strip()

    if not generated_code:
        error_payload = {
            **base_info,
            "status": "error",
            "error_type": "empty_code",
            "error_message": "generated_code is empty",
        }

        save_json(
            AST_DIR / f"{output_stem}.json",
            error_payload,
        )
        save_json(
            METRICS_DIR / f"{output_stem}.json",
            error_payload,
        )
        return

    try:
        tree = ast.parse(generated_code)

        ast_payload = {
            **base_info,
            "status": "success",
            "generated_code": generated_code,
            "ast_dump": ast.dump(
                tree,
                indent=2,
            ),
        }

        metrics_payload = {
            **base_info,
            "status": "success",
            "generated_code": generated_code,
            "metrics": build_metrics(tree),
        }

        save_json(
            AST_DIR / f"{output_stem}.json",
            ast_payload,
        )
        save_json(
            METRICS_DIR / f"{output_stem}.json",
            metrics_payload,
        )

        print(f"SUCCESS: {raw_path.name}")

    except SyntaxError as error:
        error_payload = {
            **base_info,
            "status": "error",
            "error_type": "syntax_error",
            "error_message": str(error),
            "generated_code": generated_code,
        }

        save_json(
            AST_DIR / f"{output_stem}.json",
            error_payload,
        )
        save_json(
            METRICS_DIR / f"{output_stem}.json",
            error_payload,
        )

        print(
            f"SYNTAX ERROR: {raw_path.name}: "
            f"{error}"
        )

    except Exception as error:
        error_payload = {
            **base_info,
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "generated_code": generated_code,
        }

        save_json(
            AST_DIR / f"{output_stem}.json",
            error_payload,
        )
        save_json(
            METRICS_DIR / f"{output_stem}.json",
            error_payload,
        )

        print(
            f"ERROR: {raw_path.name}: "
            f"{error}"
        )


def main() -> None:
    ensure_dirs()

    raw_files = sorted(RAW_DIR.glob("*.json"))

    if not raw_files:
        print("No JSON files found in outputs/raw")
        return

    print(f"Raw files found: {len(raw_files)}")

    for raw_path in raw_files:
        process_file(raw_path)

    print("AST parsing completed.")
    print(f"AST output: {AST_DIR}")
    print(f"Metrics output: {METRICS_DIR}")


if __name__ == "__main__":
    main()