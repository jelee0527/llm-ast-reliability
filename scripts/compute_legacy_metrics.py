"""
compute_legacy_metrics.py

Official HumanEval 12,300-sample experiment용 기존 코드 품질 지표 계산 스크립트.

계산 지표:
- LOC / SLOC / blank / comments
- Cyclomatic Complexity
- Halstead Volume / Difficulty / Effort / Vocabulary / Length / Bugs
- Maintainability Index
- Token length

입력:
- outputs/raw/**/*.json
- outputs/eval/*.csv 또는 outputs/eval/**/*.csv
- results/model_summary.csv
- results/prompt_summary.csv
- results/repeat_stability_summary.csv 또는 유사 파일

출력:
- results/legacy_sample_metrics.csv
- results/legacy_model_summary.csv
- results/legacy_prompt_summary.csv
- results/legacy_passed_only_model_summary.csv
- results/legacy_vs_structural_model_comparison.csv
- results/legacy_vs_structural_prompt_comparison.csv
- results/legacy_vs_ssi_correlation.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
    from radon.raw import analyze
except ImportError as e:
    raise ImportError(
        "radon이 설치되어 있지 않습니다.\n"
        "먼저 아래 명령어를 실행하세요:\n"
        "python -m pip install radon"
    ) from e


RAW_DIR = Path("outputs/raw")
EVAL_DIR = Path("outputs/eval")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


CODE_KEYS = [
    "generated_code",
    "code",
    "solution",
    "output",
    "response",
    "completion",
    "content",
]

PROBLEM_KEYS = [
    "problem_id",
    "task_id",
    "problem",
    "id",
]

MODEL_KEYS = [
    "model_name",
    "model",
    "llm",
]

PROMPT_KEYS = [
    "prompt_name",
    "prompt_type",
    "prompt",
    "condition",
]

REPEAT_KEYS = [
    "repeat",
    "repetition",
    "rep",
    "trial",
    "run",
]


def first_existing_key(d: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        if key in d:
            return key
    return None


def normalize_problem_id(value: Any) -> str | None:
    """
    HumanEval/81, humaneval_081, 81 등을 humaneval_081 형태로 정규화.
    """
    if value is None:
        return None

    s = str(value)

    match = re.search(r"HumanEval/(\d+)", s)
    if match:
        return f"humaneval_{int(match.group(1)):03d}"

    match = re.search(r"humaneval[_-]?(\d+)", s, re.IGNORECASE)
    if match:
        return f"humaneval_{int(match.group(1)):03d}"

    if s.isdigit():
        return f"humaneval_{int(s):03d}"

    return s


def normalize_model_name(value: Any) -> str | None:
    if value is None:
        return None

    s = str(value).strip()

    mapping = {
        "claude-sonnet-4-6": "claude_model",
        "claude_sonnet_4_6": "claude_model",
        "claude sonnet 4.6": "claude_model",
        "claude": "claude_model",
        "gpt-5-mini": "gpt5_model",
        "gpt5": "gpt5_model",
        "gpt-5 mini": "gpt5_model",
        "deepseek-chat": "deepseek_model",
        "deepseek": "deepseek_model",
    }

    lower = s.lower()
    for k, v in mapping.items():
        if k in lower:
            return v

    return s


def normalize_prompt_name(value: Any) -> str | None:
    if value is None:
        return None

    s = str(value).strip().lower()

    for name in ["basic", "concise", "readable", "optimized", "constraint"]:
        if name in s:
            return name

    return s


def normalize_repeat(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        match = re.search(r"\d+", str(value))
        if match:
            return int(match.group(0))
        return None


def extract_code_from_sample(sample: dict[str, Any]) -> str:
    """
    JSON 구조가 조금 달라도 코드 텍스트를 최대한 찾아냄.
    """
    code_key = first_existing_key(sample, CODE_KEYS)
    if code_key:
        value = sample.get(code_key)
        if isinstance(value, str):
            return value

    # OpenAI/Anthropic 응답처럼 중첩되어 있을 가능성 방어
    for key in ["result", "data", "message"]:
        nested = sample.get(key)
        if isinstance(nested, dict):
            nested_key = first_existing_key(nested, CODE_KEYS)
            if nested_key and isinstance(nested.get(nested_key), str):
                return nested[nested_key]

    return ""


def load_raw_samples() -> list[dict[str, Any]]:
    json_files = sorted(RAW_DIR.glob("**/*.json"))
    if not json_files:
        raise FileNotFoundError(f"{RAW_DIR} 아래에서 JSON 파일을 찾지 못했습니다.")

    rows: list[dict[str, Any]] = []

    for fp in json_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] JSON 로딩 실패: {fp} / {e}")
            continue

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            problem_key = first_existing_key(item, PROBLEM_KEYS)
            model_key = first_existing_key(item, MODEL_KEYS)
            prompt_key = first_existing_key(item, PROMPT_KEYS)
            repeat_key = first_existing_key(item, REPEAT_KEYS)

            code = extract_code_from_sample(item)

            row = {
                "source_file": str(fp),
                "problem_id": normalize_problem_id(item.get(problem_key) if problem_key else None),
                "model_name": normalize_model_name(item.get(model_key) if model_key else None),
                "prompt_name": normalize_prompt_name(item.get(prompt_key) if prompt_key else None),
                "repeat": normalize_repeat(item.get(repeat_key) if repeat_key else None),
                "code": code,
            }

            # 파일 경로에서 빠진 정보 보완
            path_text = str(fp).replace("\\", "/").lower()

            if row["model_name"] is None:
                row["model_name"] = normalize_model_name(path_text)

            if row["prompt_name"] is None:
                row["prompt_name"] = normalize_prompt_name(path_text)

            if row["problem_id"] is None:
                row["problem_id"] = normalize_problem_id(path_text)

            if row["repeat"] is None:
                match = re.search(r"repeat[_-]?(\d+)|rep[_-]?(\d+)", path_text)
                if match:
                    row["repeat"] = int(next(g for g in match.groups() if g is not None))

            rows.append(row)

    print(f"[INFO] raw JSON files: {len(json_files)}")
    print(f"[INFO] raw samples loaded: {len(rows)}")
    return rows


def compute_radon_metrics(code: str) -> dict[str, Any]:
    empty_result = {
        "legacy_parse_success": False,
        "loc": None,
        "sloc": None,
        "comments": None,
        "blank": None,
        "cc_avg": None,
        "cc_max": None,
        "cc_sum": None,
        "halstead_volume": None,
        "halstead_difficulty": None,
        "halstead_effort": None,
        "halstead_vocabulary": None,
        "halstead_length": None,
        "halstead_bugs": None,
        "maintainability_index": None,
        "token_length": None,
    }

    if not isinstance(code, str) or not code.strip():
        return empty_result

    result = empty_result.copy()
    result["token_length"] = len(code.split())

    try:
        raw = analyze(code)
        result["loc"] = raw.loc
        result["sloc"] = raw.sloc
        result["comments"] = raw.comments
        result["blank"] = raw.blank
    except Exception:
        pass

    try:
        blocks = cc_visit(code)
        complexities = [b.complexity for b in blocks]
        if complexities:
            result["cc_avg"] = sum(complexities) / len(complexities)
            result["cc_max"] = max(complexities)
            result["cc_sum"] = sum(complexities)
        else:
            result["cc_avg"] = 1.0
            result["cc_max"] = 1.0
            result["cc_sum"] = 1.0
    except Exception:
        return result

    try:
        h = h_visit(code)
        total = h.total
        result["halstead_volume"] = total.volume
        result["halstead_difficulty"] = total.difficulty
        result["halstead_effort"] = total.effort
        result["halstead_vocabulary"] = total.vocabulary
        result["halstead_length"] = total.length
        result["halstead_bugs"] = total.bugs
    except Exception:
        pass

    try:
        result["maintainability_index"] = mi_visit(code, multi=True)
    except Exception:
        pass

    result["legacy_parse_success"] = True
    return result


def load_functional_eval() -> pd.DataFrame | None:
    """
    outputs/eval 아래 평가 CSV를 찾아서 pass 여부를 병합.
    파일명이 달라도 pass 관련 컬럼을 최대한 탐색.
    """
    csv_files = sorted(EVAL_DIR.glob("**/*.csv"))
    if not csv_files:
        print("[WARN] outputs/eval 아래 CSV가 없습니다. pass 여부 병합은 생략합니다.")
        return None

    frames = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        cols = set(df.columns)
        useful = {"problem_id", "task_id", "model_name", "model", "prompt_name", "prompt_type", "repeat", "repetition"}
        pass_like = [c for c in df.columns if c.lower() in ["passed", "pass", "success", "is_passed", "functional_pass"]]

        if cols.intersection(useful) and pass_like:
            df = df.copy()
            df["eval_source_file"] = str(fp)
            frames.append(df)

    if not frames:
        print("[WARN] 병합 가능한 평가 CSV를 찾지 못했습니다.")
        return None

    eval_df = pd.concat(frames, ignore_index=True)

    rename_map = {}
    if "task_id" in eval_df.columns and "problem_id" not in eval_df.columns:
        rename_map["task_id"] = "problem_id"
    if "model" in eval_df.columns and "model_name" not in eval_df.columns:
        rename_map["model"] = "model_name"
    if "prompt_type" in eval_df.columns and "prompt_name" not in eval_df.columns:
        rename_map["prompt_type"] = "prompt_name"
    if "repetition" in eval_df.columns and "repeat" not in eval_df.columns:
        rename_map["repetition"] = "repeat"

    eval_df = eval_df.rename(columns=rename_map)

    if "problem_id" in eval_df.columns:
        eval_df["problem_id"] = eval_df["problem_id"].apply(normalize_problem_id)
    if "model_name" in eval_df.columns:
        eval_df["model_name"] = eval_df["model_name"].apply(normalize_model_name)
    if "prompt_name" in eval_df.columns:
        eval_df["prompt_name"] = eval_df["prompt_name"].apply(normalize_prompt_name)
    if "repeat" in eval_df.columns:
        eval_df["repeat"] = eval_df["repeat"].apply(normalize_repeat)

    pass_col = None
    for c in eval_df.columns:
        if c.lower() in ["passed", "pass", "success", "is_passed", "functional_pass"]:
            pass_col = c
            break

    if pass_col is None:
        return None

    eval_df["functional_pass"] = eval_df[pass_col].astype(str).str.lower().isin(["true", "1", "yes", "passed", "pass"])

    keep_cols = ["problem_id", "model_name", "prompt_name", "repeat", "functional_pass"]
    keep_cols = [c for c in keep_cols if c in eval_df.columns]

    eval_df = eval_df[keep_cols].drop_duplicates()
    print(f"[INFO] functional eval rows loaded: {len(eval_df)}")
    return eval_df


def safe_group_mean(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    return out


def find_results_csv(required_cols: list[str]) -> Path | None:
    """
    results/ 아래에서 required_cols를 모두 가진 CSV를 찾음.
    """
    for fp in sorted(RESULTS_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(fp, nrows=5)
        except Exception:
            continue

        cols = set(df.columns)
        if all(c in cols for c in required_cols):
            return fp

    return None


def merge_with_model_summary(legacy_model: pd.DataFrame) -> None:
    fp = find_results_csv(["model_name", "functional_pass_rate", "ssi", "pssi", "sds"])
    if fp is None:
        print("[WARN] model_summary CSV를 찾지 못했습니다. model comparison 병합 생략.")
        return

    structural = pd.read_csv(fp)
    merged = pd.merge(structural, legacy_model, on="model_name", how="left")
    out = RESULTS_DIR / "legacy_vs_structural_model_comparison.csv"
    merged.to_csv(out, index=False)
    print(f"[SAVE] {out}")


def merge_with_prompt_summary(legacy_prompt: pd.DataFrame) -> None:
    fp = find_results_csv(["prompt_name", "success_rate", "ssi"])
    if fp is None:
        print("[WARN] prompt_summary CSV를 찾지 못했습니다. prompt comparison 병합 생략.")
        return

    structural = pd.read_csv(fp)
    merged = pd.merge(structural, legacy_prompt, on="prompt_name", how="left")
    out = RESULTS_DIR / "legacy_vs_structural_prompt_comparison.csv"
    merged.to_csv(out, index=False)
    print(f"[SAVE] {out}")


def correlation_with_repeat_stability(legacy_df: pd.DataFrame) -> None:
    """
    repeat stability 단위:
    problem_id + model_name + prompt_name 별 기존 지표 평균과 SSI를 병합해서 상관분석.
    """
    fp = find_results_csv(["problem_id", "model_name", "prompt_name", "ssi"])
    if fp is None:
        print("[WARN] repeat_stability_summary CSV를 찾지 못했습니다. SSI correlation 생략.")
        return

    structural = pd.read_csv(fp)

    for col in ["problem_id", "model_name", "prompt_name"]:
        if col not in structural.columns:
            print("[WARN] repeat stability CSV 컬럼 부족. correlation 생략.")
            return

    structural["problem_id"] = structural["problem_id"].apply(normalize_problem_id)
    structural["model_name"] = structural["model_name"].apply(normalize_model_name)
    structural["prompt_name"] = structural["prompt_name"].apply(normalize_prompt_name)

    metric_cols = [
        "loc",
        "sloc",
        "cc_avg",
        "cc_max",
        "cc_sum",
        "halstead_volume",
        "halstead_difficulty",
        "halstead_effort",
        "maintainability_index",
        "token_length",
    ]

    group_cols = ["problem_id", "model_name", "prompt_name"]

    legacy_group = safe_group_mean(
        legacy_df[legacy_df["legacy_parse_success"] == True],
        group_cols,
        metric_cols,
    )

    merged = pd.merge(structural, legacy_group, on=group_cols, how="inner")

    corr_rows = []
    for legacy_col in metric_cols:
        if legacy_col not in merged.columns:
            continue
        for structural_col in ["ssi"]:
            if structural_col not in merged.columns:
                continue

            temp = merged[[legacy_col, structural_col]].dropna()
            if len(temp) < 3:
                continue

            rho = temp[legacy_col].corr(temp[structural_col], method="spearman")
            corr_rows.append({
                "analysis_unit": "problem_model_prompt",
                "legacy_metric": legacy_col,
                "structural_metric": structural_col,
                "n": len(temp),
                "spearman_rho": rho,
            })

    corr_df = pd.DataFrame(corr_rows)
    out = RESULTS_DIR / "legacy_vs_ssi_correlation.csv"
    corr_df.to_csv(out, index=False)
    print(f"[SAVE] {out}")


def main() -> None:
    samples = load_raw_samples()

    rows = []
    for idx, sample in enumerate(samples, start=1):
        if idx % 500 == 0:
            print(f"[INFO] processing {idx}/{len(samples)}")

        code = sample.pop("code")
        metric = compute_radon_metrics(code)
        rows.append({**sample, **metric})

    legacy_df = pd.DataFrame(rows)

    # 평가 결과 pass 여부 병합
    eval_df = load_functional_eval()
    if eval_df is not None:
        merge_keys = ["problem_id", "model_name", "prompt_name", "repeat"]
        legacy_df = pd.merge(legacy_df, eval_df, on=merge_keys, how="left")
    else:
        legacy_df["functional_pass"] = pd.NA

    sample_out = RESULTS_DIR / "legacy_sample_metrics.csv"
    legacy_df.to_csv(sample_out, index=False)
    print(f"[SAVE] {sample_out} ({len(legacy_df)} rows)")

    metric_cols = [
        "loc",
        "sloc",
        "comments",
        "blank",
        "cc_avg",
        "cc_max",
        "cc_sum",
        "halstead_volume",
        "halstead_difficulty",
        "halstead_effort",
        "halstead_vocabulary",
        "halstead_length",
        "halstead_bugs",
        "maintainability_index",
        "token_length",
    ]

    # radon 계산 성공 샘플만 기존 지표 집계
    valid_df = legacy_df[legacy_df["legacy_parse_success"] == True].copy()

    # 모델별
    model_summary = safe_group_mean(valid_df, ["model_name"], metric_cols)
    model_summary.insert(1, "legacy_metric_samples", valid_df.groupby("model_name").size().values)
    model_out = RESULTS_DIR / "legacy_model_summary.csv"
    model_summary.to_csv(model_out, index=False)
    print(f"[SAVE] {model_out}")

    # 프롬프트별
    prompt_summary = safe_group_mean(valid_df, ["prompt_name"], metric_cols)
    prompt_summary.insert(1, "legacy_metric_samples", valid_df.groupby("prompt_name").size().values)
    prompt_out = RESULTS_DIR / "legacy_prompt_summary.csv"
    prompt_summary.to_csv(prompt_out, index=False)
    print(f"[SAVE] {prompt_out}")

    # functionally correct only
    if "functional_pass" in legacy_df.columns and legacy_df["functional_pass"].notna().any():
        passed_df = legacy_df[
            (legacy_df["legacy_parse_success"] == True) &
            (legacy_df["functional_pass"] == True)
        ].copy()

        passed_model_summary = safe_group_mean(passed_df, ["model_name"], metric_cols)
        passed_model_summary.insert(
            1,
            "passed_legacy_metric_samples",
            passed_df.groupby("model_name").size().values
        )

        passed_out = RESULTS_DIR / "legacy_passed_only_model_summary.csv"
        passed_model_summary.to_csv(passed_out, index=False)
        print(f"[SAVE] {passed_out}")

    merge_with_model_summary(model_summary)
    merge_with_prompt_summary(prompt_summary)
    correlation_with_repeat_stability(legacy_df)

    print("\n[DONE] Legacy metric analysis completed.")
    print("논문에 우선 넣을 파일:")
    print("- results/legacy_vs_structural_model_comparison.csv")
    print("- results/legacy_vs_structural_prompt_comparison.csv")
    print("- results/legacy_passed_only_model_summary.csv")
    print("- results/legacy_vs_ssi_correlation.csv")


if __name__ == "__main__":
    main()