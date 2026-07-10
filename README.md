# LLM AST Reliability

Replication package for the paper:

> **A Model-Agnostic AST-Based Framework for Evaluating Structural Reliability in LLM Code Generation**

This repository contains the code, prompt templates, processed results, and analysis scripts used in the study.

## Overview

The project evaluates the structural reliability of LLM-generated Python code using Abstract Syntax Tree (AST) features.

The main metrics are:

- **SSI**: Structural Stability Index
- **PSSI**: Prompt Sensitivity Index
- **SDS**: Structural Diversity Score

The experiment uses:

- 164 HumanEval problems
- 3 LLM conditions
- 5 prompt types
- 5 repeated generations
- 12,300 generated code samples

## Models

- Claude Sonnet 4.6
- GPT-5 mini
- DeepSeek Chat

## Setup

```bash
git clone https://github.com/jelee0527/llm-ast-reliability.git
cd llm-ast-reliability

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env`.

```env
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
ANTHROPIC_API_KEY=

EXPERIMENT_REPEATS=5
EXPERIMENT_TEMPERATURE=0.2
EXPERIMENT_MAX_TOKENS=3000

EXPERIMENT_MAX_PROBLEMS=0
EXPERIMENT_MODEL_FILTER=
EXPERIMENT_PROMPT_FILTER=
```

Do not commit the actual `.env` file or API keys.

## Repository Contents

This repository includes:

- Code generation scripts
- HumanEval functional evaluation
- AST metric extraction
- SSI, PSSI, and SDS computation
- Statistical analysis
- Processed experimental results
- Figure and table reproduction scripts

## Main Results

| Item                 | Result |
| -------------------- | -----: |
| Generated samples    | 12,300 |
| AST-parsable samples | 12,287 |
| Functional passes    | 11,766 |

The results indicate that functional correctness and structural reliability are complementary evaluation dimensions.

## Citation

```text
Jieun Lee,
“A Model-Agnostic AST-Based Framework for Evaluating
Structural Reliability in LLM Code Generation,”
manuscript, 2026.
```

Citation information and a Zenodo DOI will be updated after publication.

## Author

Jieun Lee  
Hanyang University
