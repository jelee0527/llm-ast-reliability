import json
import os
import re
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
CONFIG_DIR = ROOT_DIR / "configs"
DATASET_DIR = ROOT_DIR / "datasets"
OUTPUT_DIR = ROOT_DIR / "outputs" / "raw"

MODELS_PATH = CONFIG_DIR / "models.yaml"
PROMPTS_PATH = CONFIG_DIR / "prompts.yaml"
PROBLEMS_PATH = DATASET_DIR / "problems.json"

DEFAULT_REPEATS = 1
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1200


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure: {path}")

    return data


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def strip_markdown_fence(text: str) -> str:
    if not text:
        return ""

    value = text.strip()

    block_match = re.search(
        r"```(?:python|py)?\s*\n([\s\S]*?)```",
        value,
        re.IGNORECASE,
    )

    if block_match:
        return block_match.group(1).strip()

    value = re.sub(
        r"^```(?:python|py)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s*```$", "", value)

    return value.strip()


def get_client(provider: str) -> Any:
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        return OpenAI(api_key=api_key)

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")

        return OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        return Anthropic(api_key=api_key)

    raise ValueError(f"Unsupported provider: {provider}")

def generate_openai(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    request_args = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
    }

    if model.startswith("gpt-5"):
        request_args["reasoning"] = {
            "effort": "minimal"
        }
    else:
        request_args["temperature"] = temperature

    response = client.responses.create(
        **request_args
    )

    return response.output_text.strip()


def generate_deepseek(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    content = response.choices[0].message.content

    return content.strip() if content else ""


def generate_anthropic(
    client: Anthropic,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]

    return "\n".join(text_parts).strip()


def generate_code(
    client: Any,
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    if provider == "openai":
        return generate_openai(
            client=client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "deepseek":
        return generate_deepseek(
            client=client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        return generate_anthropic(
            client=client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def get_enabled_models(
    model_config: dict[str, Any],
) -> list[dict[str, Any]]:
    models = model_config.get("models", [])

    enabled_models = [
        model
        for model in models
        if model.get("enabled", True)
    ]

    model_filter = os.getenv(
        "EXPERIMENT_MODEL_FILTER",
        "",
    ).strip()

    if model_filter:
        enabled_models = [
            model
            for model in enabled_models
            if model["name"] == model_filter
        ]

    return enabled_models


def get_selected_prompts(
    prompts: dict[str, str],
) -> dict[str, str]:
    prompt_filter = os.getenv(
        "EXPERIMENT_PROMPT_FILTER",
        "",
    ).strip()

    if not prompt_filter:
        return prompts

    if prompt_filter not in prompts:
        raise ValueError(
            f"Prompt not found: {prompt_filter}"
        )

    return {
        prompt_filter: prompts[prompt_filter]
    }


def main() -> None:
    ensure_dirs()

    problems = load_json(PROBLEMS_PATH)
    prompts = get_selected_prompts(
        load_yaml(PROMPTS_PATH)
    )
    model_config = load_yaml(MODELS_PATH)
    models = get_enabled_models(model_config)

    repeats = int(
        os.getenv(
            "EXPERIMENT_REPEATS",
            DEFAULT_REPEATS,
        )
    )

    temperature = float(
        os.getenv(
            "EXPERIMENT_TEMPERATURE",
            DEFAULT_TEMPERATURE,
        )
    )

    max_tokens = int(
        os.getenv(
            "EXPERIMENT_MAX_TOKENS",
            DEFAULT_MAX_TOKENS,
        )
    )

    max_problems = int(
        os.getenv(
            "EXPERIMENT_MAX_PROBLEMS",
            "0",
        )
    )

    if max_problems > 0:
        problems = problems[:max_problems]

    if not models:
        raise ValueError("No enabled models found")

    print("=" * 70)
    print("HumanEval generation experiment")
    print("=" * 70)
    print(f"Problems: {len(problems)}")
    print(f"Models: {len(models)}")
    print(f"Prompts: {len(prompts)}")
    print(f"Repeats: {repeats}")
    print(
        "Expected samples:",
        len(problems)
        * len(models)
        * len(prompts)
        * repeats,
    )
    print("=" * 70)

    for model_info in models:
        model_name = model_info["name"]
        provider = model_info["provider"]
        model_id = model_info["model"]

        print(
            f"\n[Model] {model_name} "
            f"({provider}: {model_id})"
        )

        client = get_client(provider)

        for problem in problems:
            problem_id = problem["id"]
            task_id = problem["task_id"]
            official_prompt = problem["prompt"]

            for prompt_name, prompt_template in prompts.items():
                full_prompt = prompt_template.format(
                    problem=official_prompt
                )

                for repeat_idx in range(1, repeats + 1):
                    filename = (
                        f"{problem_id}"
                        f"__{model_name}"
                        f"__{prompt_name}"
                        f"__r{repeat_idx}.json"
                    )

                    output_path = OUTPUT_DIR / filename

                    if output_path.exists():
                        print(f"Skip existing: {filename}")
                        continue

                    print(
                        f"{model_name} | "
                        f"{task_id} | "
                        f"{prompt_name} | "
                        f"r{repeat_idx}"
                    )

                    started_at = time.time()

                    try:
                        raw_response = generate_code(
                            client=client,
                            provider=provider,
                            model=model_id,
                            prompt=full_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )

                        cleaned_code = strip_markdown_fence(
                            raw_response
                        )

                        payload = {
                            "status": "success",
                            "problem_id": problem_id,
                            "task_id": task_id,
                            "problem_name": problem["name"],
                            "entry_point": problem["entry_point"],
                            "evaluation_type": "humaneval",
                            "model_name": model_name,
                            "model_id": model_id,
                            "provider": provider,
                            "prompt_name": prompt_name,
                            "repeat_idx": repeat_idx,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "generated_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "elapsed_seconds": round(
                                time.time() - started_at,
                                3,
                            ),
                            "official_prompt": official_prompt,
                            "generation_prompt": full_prompt,
                            "raw_response": raw_response,
                            "generated_code": cleaned_code,
                        }

                    except Exception as error:
                        payload = {
                            "status": "error",
                            "problem_id": problem_id,
                            "task_id": task_id,
                            "problem_name": problem["name"],
                            "entry_point": problem["entry_point"],
                            "evaluation_type": "humaneval",
                            "model_name": model_name,
                            "model_id": model_id,
                            "provider": provider,
                            "prompt_name": prompt_name,
                            "repeat_idx": repeat_idx,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "generated_at": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "elapsed_seconds": round(
                                time.time() - started_at,
                                3,
                            ),
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }

                        print(
                            f"ERROR: {type(error).__name__}: "
                            f"{error}"
                        )

                    save_json(output_path, payload)

                    time.sleep(0.5)

    print("\nGeneration completed.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()