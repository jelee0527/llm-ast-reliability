"""Generate the semantic-preserving paraphrase robustness experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_PATH = ROOT_DIR / "configs" / "paraphrase_models.yaml"
PROMPT_PATHS = {
    "paraphrase": ROOT_DIR / "configs" / "paraphrase_prompts.yaml",
}
PROBLEMS_PATH = ROOT_DIR / "datasets" / "problems.json"
OUTPUT_ROOT = ROOT_DIR / "outputs" / "paraphrase"
RAW_DIR = OUTPUT_ROOT / "raw"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"

DEFAULT_REPEATS = 5
DEFAULT_SEED = 42
DEFAULT_MAX_ATTEMPTS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument(
        "--prompt-family",
        action="append",
        choices=sorted(PROMPT_PATHS),
        default=[],
    )
    parser.add_argument("--max-problems", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Do not retry an existing output whose status is error.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "working_tree_dirty": bool(run("status", "--porcelain")),
    }


def strip_markdown_fence(text: str) -> str:
    value = (text or "").strip()
    block = re.search(
        r"```(?:python|py)?\s*\n([\s\S]*?)```",
        value,
        re.IGNORECASE,
    )
    if block:
        return block.group(1).strip()
    value = re.sub(r"^```(?:python|py)?\s*", "", value, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", value).strip()


def select_named(
    records: list[dict[str, Any]],
    selected: list[str],
    label: str,
) -> list[dict[str, Any]]:
    if not selected:
        return records
    known = {str(record["name"]) for record in records}
    missing = sorted(set(selected) - known)
    if missing:
        raise ValueError(f"Unknown {label}: {missing}")
    wanted = set(selected)
    return [record for record in records if str(record["name"]) in wanted]


def load_prompt_records(
    selected_families: list[str],
    selected_prompts: list[str],
) -> list[dict[str, str]]:
    families = selected_families or list(PROMPT_PATHS)
    records: list[dict[str, str]] = []
    for family in families:
        prompts = load_yaml(PROMPT_PATHS[family])
        records.extend(
            {
                "family": family,
                "name": str(name),
                "template": str(template),
            }
            for name, template in prompts.items()
        )

    if selected_prompts:
        known = {record["name"] for record in records}
        missing = sorted(set(selected_prompts) - known)
        if missing:
            raise ValueError(f"Unknown prompts: {missing}")
        wanted = set(selected_prompts)
        records = [record for record in records if record["name"] in wanted]

    names = [record["name"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Prompt names must be unique across prompt families")
    return records


def get_client(provider: str) -> Any:
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set")
        return OpenAI(api_key=key)
    if provider == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        return OpenAI(api_key=key, base_url="https://api.deepseek.com")
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return Anthropic(api_key=key)
    raise ValueError(f"Unsupported provider: {provider}")


def generate_once(
    client: Any,
    model: dict[str, Any],
    prompt: str,
) -> tuple[
    str,
    str | None,
    str | None,
    dict[str, Any],
    dict[str, Any],
]:
    provider = str(model["provider"])
    model_id = str(model["model"])
    max_tokens = int(model["max_tokens"])
    temperature = model.get("temperature")
    reasoning_effort = model.get("reasoning_effort")

    if provider == "openai":
        request: dict[str, Any] = {
            "model": model_id,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        if model_id.startswith("gpt-5"):
            request["reasoning"] = {"effort": reasoning_effort or "minimal"}
        elif temperature is not None:
            request["temperature"] = float(temperature)
        response = client.responses.create(**request)
        settings = {
            "temperature": request.get("temperature"),
            "max_output_tokens": max_tokens,
            "reasoning_effort": (
                request.get("reasoning", {}).get("effort")
                if isinstance(request.get("reasoning"), dict)
                else None
            ),
        }
        return (
            response.output_text.strip(),
            getattr(response, "model", None),
            getattr(response, "id", None),
            settings,
            {
                "response_status": getattr(response, "status", None),
                "finish_reason": getattr(
                    getattr(response, "incomplete_details", None),
                    "reason",
                    None,
                ),
                "input_tokens": getattr(
                    getattr(response, "usage", None), "input_tokens", None
                ),
                "output_tokens": getattr(
                    getattr(response, "usage", None), "output_tokens", None
                ),
                "total_tokens": getattr(
                    getattr(response, "usage", None), "total_tokens", None
                ),
            },
        )

    if provider == "deepseek":
        thinking_mode = str(model.get("thinking_mode", "disabled"))
        if thinking_mode not in {"enabled", "disabled"}:
            raise ValueError(
                "DeepSeek thinking_mode must be 'enabled' or 'disabled'"
            )
        request: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
            "extra_body": {"thinking": {"type": thinking_mode}},
        }
        if thinking_mode == "disabled" and temperature is not None:
            request["temperature"] = float(temperature)
        response = client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""
        return (
            content.strip(),
            getattr(response, "model", None),
            getattr(response, "id", None),
            {
                "temperature": request.get("temperature"),
                "max_tokens": max_tokens,
                "reasoning_effort": None,
                "thinking_mode": thinking_mode,
            },
            {
                "response_status": None,
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "input_tokens": getattr(
                    getattr(response, "usage", None), "prompt_tokens", None
                ),
                "output_tokens": getattr(
                    getattr(response, "usage", None), "completion_tokens", None
                ),
                "total_tokens": getattr(
                    getattr(response, "usage", None), "total_tokens", None
                ),
            },
        )

    if provider == "anthropic":
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=float(temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        return (
            "\n".join(text_parts).strip(),
            getattr(response, "model", None),
            getattr(response, "id", None),
            {
                "temperature": float(temperature),
                "max_tokens": max_tokens,
                "reasoning_effort": None,
            },
            {
                "response_status": None,
                "finish_reason": getattr(response, "stop_reason", None),
                "input_tokens": getattr(
                    getattr(response, "usage", None), "input_tokens", None
                ),
                "output_tokens": getattr(
                    getattr(response, "usage", None), "output_tokens", None
                ),
                "total_tokens": (
                    (getattr(getattr(response, "usage", None), "input_tokens", 0) or 0)
                    + (getattr(getattr(response, "usage", None), "output_tokens", 0) or 0)
                ),
            },
        )

    raise ValueError(f"Unsupported provider: {provider}")


def build_schedule(
    problems: list[dict[str, Any]],
    models: list[dict[str, Any]],
    prompts: list[dict[str, str]],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    schedule = [
        {
            "problem": problem,
            "model": model,
            "prompt_family": prompt["family"],
            "prompt_name": prompt["name"],
            "prompt_template": prompt["template"],
            "repeat_idx": repeat_idx,
        }
        for problem in problems
        for model in models
        for prompt in prompts
        for repeat_idx in range(1, repeats + 1)
    ]
    random.Random(seed).shuffle(schedule)
    return schedule


def should_skip(
    path: Path,
    skip_errors: bool,
    model: dict[str, Any],
    generation_prompt: str,
) -> bool:
    if not path.exists():
        return False
    try:
        existing = load_json(path)
    except Exception:
        return False
    if existing.get("status") != "success":
        return skip_errors

    provider = str(model["provider"])
    expected_temperature = (
        None
        if provider == "openai" and str(model["model"]).startswith("gpt-5")
        else model.get("temperature")
    )
    expected_reasoning = (
        model.get("reasoning_effort")
        if provider == "openai" and str(model["model"]).startswith("gpt-5")
        else None
    )
    expected_thinking = (
        str(model.get("thinking_mode", "disabled"))
        if provider == "deepseek"
        else None
    )
    return all(
        [
            existing.get("requested_model_id") == model["model"],
            existing.get("generation_prompt") == generation_prompt,
            existing.get("max_tokens") == int(model["max_tokens"]),
            existing.get("temperature") == expected_temperature,
            existing.get("reasoning_effort") == expected_reasoning,
            existing.get("thinking_mode") == expected_thinking,
        ]
    )


def write_manifest(
    args: argparse.Namespace,
    problems: list[dict[str, Any]],
    models: list[dict[str, Any]],
    prompts: list[dict[str, str]],
    schedule_size: int,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = MANIFEST_DIR / f"run_{stamp}.json"
    safe_models = [
        {
            key: model.get(key)
            for key in [
                "name",
                "provider",
                "model",
                "temperature",
                "max_tokens",
                "reasoning_effort",
                "thinking_mode",
            ]
        }
        for model in models
    ]
    save_json_atomic(
        path,
        {
            "experiment": "semantic_paraphrase_reliability",
            "started_at": utc_now(),
            "problems": len(problems),
            "models": safe_models,
            "prompt_families": sorted({prompt["family"] for prompt in prompts}),
            "prompts": prompts,
            "repeats": args.repeats,
            "randomization_seed": args.seed,
            "planned_samples": schedule_size,
            "max_attempts": args.max_attempts,
            "delay_seconds": args.delay_seconds,
            "dataset_sha256": sha256_file(PROBLEMS_PATH),
            "prompt_files_sha256": {
                family: sha256_file(path)
                for family, path in PROMPT_PATHS.items()
                if family in {prompt["family"] for prompt in prompts}
            },
            "git": git_state(),
        },
    )
    return path


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.max_attempts < 1 or args.max_problems < 0:
        raise ValueError("Repeats/attempts must be positive; max-problems cannot be negative")

    model_config = load_yaml(MODELS_PATH)
    all_models = model_config.get("models", [])
    if not isinstance(all_models, list):
        raise ValueError("configs/paraphrase_models.yaml must contain a models list")
    models = select_named(all_models, args.model, "models")

    prompts = load_prompt_records(args.prompt_family, args.prompt)

    problems = load_json(PROBLEMS_PATH)
    if args.max_problems:
        problems = problems[: args.max_problems]

    schedule = build_schedule(problems, models, prompts, args.repeats, args.seed)
    print("=" * 72)
    print("Semantic-preserving paraphrase robustness experiment")
    print(f"Problems: {len(problems)}")
    print(f"Models: {len(models)} {[model['name'] for model in models]}")
    print(
        "Prompt families:",
        sorted({prompt["family"] for prompt in prompts}),
    )
    print(f"Prompts: {len(prompts)} {[prompt['name'] for prompt in prompts]}")
    print(f"Repeats: {args.repeats}")
    print(f"Planned samples: {len(schedule)}")
    print(f"Randomization seed: {args.seed}")
    print("=" * 72)

    if args.dry_run:
        print("Dry run: no API clients were created and no files were written.")
        for index, item in enumerate(schedule[:10], start=1):
            print(
                index,
                item["problem"]["task_id"],
                item["model"]["name"],
                item["prompt_family"],
                item["prompt_name"],
                f"r{item['repeat_idx']}",
            )
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(
        args,
        problems,
        models,
        prompts,
        len(schedule),
    )
    print(f"Manifest: {manifest_path}")

    clients: dict[str, Any] = {}
    completed = 0
    skipped = 0
    failed = 0

    for sequence_index, item in enumerate(schedule, start=1):
        problem = item["problem"]
        model = item["model"]
        prompt_family = item["prompt_family"]
        prompt_name = item["prompt_name"]
        repeat_idx = int(item["repeat_idx"])
        filename = (
            f"{problem['id']}__{model['name']}__{prompt_family}"
            f"__{prompt_name}__r{repeat_idx}.json"
        )
        output_path = RAW_DIR / filename

        generation_prompt = item["prompt_template"].format(
            problem=problem["prompt"]
        )
        if should_skip(
            output_path,
            args.skip_errors,
            model,
            generation_prompt,
        ):
            skipped += 1
            continue

        provider = str(model["provider"])
        if provider not in clients:
            clients[provider] = get_client(provider)

        print(
            f"[{sequence_index}/{len(schedule)}] {problem['task_id']} | "
            f"{model['name']} | {prompt_family}/{prompt_name} | r{repeat_idx}",
            flush=True,
        )
        started_at = utc_now()
        started_clock = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(1, args.max_attempts + 1):
            try:
                (
                    raw_response,
                    response_model,
                    request_id,
                    settings,
                    response_metadata,
                ) = generate_once(clients[provider], model, generation_prompt)
                payload = {
                    "status": "success",
                    "experiment": "semantic_paraphrase_reliability",
                    "problem_id": problem["id"],
                    "task_id": problem["task_id"],
                    "problem_name": problem["name"],
                    "entry_point": problem["entry_point"],
                    "evaluation_type": "humaneval",
                    "model_name": model["name"],
                    "provider": provider,
                    "model_id": model["model"],
                    "requested_model_id": model["model"],
                    "response_model_id": response_model,
                    "provider_request_id": request_id,
                    "prompt_family": prompt_family,
                    "prompt_name": prompt_name,
                    "repeat_idx": repeat_idx,
                    "request_settings": settings,
                    "response_metadata": response_metadata,
                    "temperature": settings["temperature"],
                    "max_tokens": int(model["max_tokens"]),
                    "reasoning_effort": settings["reasoning_effort"],
                    "thinking_mode": settings.get("thinking_mode"),
                    "schedule_seed": args.seed,
                    "schedule_index": sequence_index,
                    "attempts": attempt,
                    "started_at": started_at,
                    "generated_at": utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started_clock, 3),
                    "official_prompt": problem["prompt"],
                    "generation_prompt": generation_prompt,
                    "raw_response": raw_response,
                    "generated_code": strip_markdown_fence(raw_response),
                }
                save_json_atomic(output_path, payload)
                completed += 1
                last_error = None
                break
            except Exception as error:
                last_error = error
                print(
                    f"  attempt {attempt}/{args.max_attempts}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                if attempt < args.max_attempts:
                    time.sleep(min(60.0, 5.0 * (2 ** (attempt - 1))))

        if last_error is not None:
            failed += 1
            save_json_atomic(
                output_path,
                {
                    "status": "error",
                    "experiment": "semantic_paraphrase_reliability",
                    "problem_id": problem["id"],
                    "task_id": problem["task_id"],
                    "problem_name": problem["name"],
                    "entry_point": problem["entry_point"],
                    "evaluation_type": "humaneval",
                    "model_name": model["name"],
                    "provider": provider,
                    "model_id": model["model"],
                    "requested_model_id": model["model"],
                    "prompt_family": prompt_family,
                    "prompt_name": prompt_name,
                    "repeat_idx": repeat_idx,
                    "schedule_seed": args.seed,
                    "schedule_index": sequence_index,
                    "attempts": args.max_attempts,
                    "started_at": started_at,
                    "generated_at": utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started_clock, 3),
                    "error_type": type(last_error).__name__,
                    "error_message": str(last_error),
                },
            )

        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    print("=" * 72)
    print(f"Completed now: {completed}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed after retries: {failed}")
    print(f"Files currently present: {len(list(RAW_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
