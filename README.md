# LLM AST Reliability

Replication package for the manuscript:

> **A Model-Agnostic AST-Based Framework for Evaluating Structural Reliability in LLM Code Generation**  
> Jieun Lee and Inwhee Joe, 2026.

This repository contains the frozen LLM generations, HumanEval functional-evaluation outputs, AST-derived metrics, statistical analyses, prompt templates, tables, figures, and source code used in the study.

## Study Overview

The study evaluates whether functionally correct LLM-generated Python programs remain structurally consistent across repeated generations and prompt variations.

The structural measures are:

- **Structural Stability Index (SSI):** within-condition consistency across repeated generations.
- **Prompt Sensitivity Index (PSSI):** structural shifts between prompt-level centroids for the same problem and model.
- **Structural Diversity Score (SDS):** overall dispersion of generated samples in the standardized AST feature space.

The AST feature vector contains:

- AST depth
- Branch count
- Loop count
- Function count
- Control-flow ratio

## Experimental Design

| Item                  | Setting                              |
| --------------------- | ------------------------------------ |
| Benchmark             | Official HumanEval                   |
| Problems              | 164                                  |
| Models                | 3                                    |
| Prompt conditions     | 5                                    |
| Repetitions           | 5 per problem-model-prompt condition |
| Total generations     | 12,300                               |
| API collection dates  | June 9–10, 2026                      |
| Temperature           | 0.2                                  |
| Maximum output length | 3,000 tokens                         |

### Model Conditions

| Display name      | Provider  | Model identifier    |
| ----------------- | --------- | ------------------- |
| Claude Sonnet 4.6 | Anthropic | `claude-sonnet-4-6` |
| GPT-5 mini        | OpenAI    | `gpt-5-mini`        |
| DeepSeek Chat     | DeepSeek  | `deepseek-chat`     |

The model conditions are used to evaluate the proposed framework rather than to construct a general-purpose model leaderboard.

## Prompt Templates

The exact templates used in the experiment are stored in [`configs/prompts.yaml`](configs/prompts.yaml).

The five prompt conditions are:

1. `basic`
2. `concise`
3. `readable`
4. `optimized`
5. `constraint`

Each template is combined with the original HumanEval problem prompt through the `{problem}` placeholder. The stored templates, rather than shortened descriptions in the manuscript, are the authoritative prompts used for generation.

## Main Results

| Item                         | Result |
| ---------------------------- | -----: |
| Generated samples            | 12,300 |
| AST-parsable samples         | 12,287 |
| Functional passes            | 11,766 |
| Overall functional pass rate | 0.9566 |

### Model-Level Summary

| Model             | Pass rate | AST rate |   SSI |  PSSI |   SDS |
| ----------------- | --------: | -------: | ----: | ----: | ----: |
| Claude Sonnet 4.6 |     0.983 |    1.000 | 0.953 | 0.875 | 0.641 |
| GPT-5 mini        |     0.966 |    0.997 | 0.541 | 1.139 | 1.095 |
| DeepSeek Chat     |     0.921 |    1.000 | 0.873 | 0.718 | 0.576 |

The results show that functional correctness and structural reliability are complementary evaluation dimensions.

## Repository Structure

```text
.
├── configs/
│   ├── models.yaml              # Provider and model identifiers
│   └── prompts.yaml             # Exact experimental prompt templates
├── datasets/
│   └── problems.json            # Prepared HumanEval problems
├── external/human-eval/         # HumanEval source package and license
├── outputs/
│   ├── raw/                     # Frozen LLM responses and extracted code
│   ├── eval/                    # Per-sample functional-evaluation results
│   ├── ast/                     # Parsed AST representations
│   └── metrics/                 # Per-sample AST metrics
├── results/                     # Aggregated metrics and statistical results
├── reports/
│   ├── figures/                 # Reproduced manuscript figures
│   └── tables/                  # Generated LaTeX tables
├── scripts/                     # Generation, evaluation, analysis, and export code
├── Dockerfile.eval              # Isolated HumanEval execution environment
├── functional_summary.csv       # Functional result summary for all samples
├── requirements.txt
└── README.md
```

## Reproducing the Published Analyses

The included outputs are sufficient to reproduce the reported metrics, statistical tests, tables, and figures. **No API keys or new LLM calls are required for this workflow.**

### 1. Create the analysis environment

Python 3.13 is recommended for the pinned dependencies. The isolated functional evaluator uses Python 3.11 through Docker.

```bash
git clone https://github.com/jelee0527/llm-ast-reliability.git
cd llm-ast-reliability

python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Recompute metrics and analyses

Run the following commands from the repository root:

```bash
python scripts/compute_metrics.py
python scripts/analyze_results.py
python scripts/extra_analysis.py
python scripts/export_latex_tables.py
python scripts/visualize_final.py
```

Expected key checks:

```text
Total samples: 12300
AST success samples: 12287
Functional passes: 11766
```

Outputs are written to:

- `results/` for CSV summaries and statistical results
- `reports/tables/` for LaTeX tables
- `reports/figures/` for 300-dpi PNG figures

The analysis workflow was verified from the included outputs and reproduced the manuscript-level counts, rankings, blocked Friedman tests, Holm-adjusted Wilcoxon comparisons, bootstrap confidence intervals, and feature-ablation results.

## Re-running the Functional Evaluation

Functional execution uses an isolated Docker container with network access disabled, a read-only filesystem, resource limits, and a 15-second worker timeout.

Build the evaluator image:

```bash
docker build -f Dockerfile.eval -t humaneval-evaluator .
```

Run functional evaluation:

```bash
python scripts/evaluate_functional.py
```

Existing files in `outputs/eval/` are reused. The consolidated output is written to `functional_summary.csv`.

To regenerate AST and per-sample metric files from the frozen raw generations:

```bash
python scripts/parse_ast.py
python scripts/compute_metrics.py
```

## Generating New LLM Outputs

This step is optional, invokes paid external APIs, and is **not required** to reproduce the reported analyses.

1. Copy `.env.example` to `.env`.
2. Add only the API keys for the providers you intend to run.
3. Review `configs/models.yaml` and explicitly enable the desired model conditions.
4. Use filters for a small validation run before starting a full experiment.

Example `.env` configuration:

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

Small validation run example:

```env
EXPERIMENT_REPEATS=1
EXPERIMENT_MAX_PROBLEMS=2
EXPERIMENT_MODEL_FILTER=gpt5_model
EXPERIMENT_PROMPT_FILTER=basic
```

Then run:

```bash
python scripts/generate.py
```

API-hosted models may change over time. Therefore, newly generated responses are not expected to be byte-identical to the frozen responses collected on June 9–10, 2026. The files in `outputs/raw/` are the authoritative generation artifacts for the reported study.

## Data and Reproducibility Notes

- All 12,300 raw generation records include the provider, model identifier, prompt condition, repetition index, decoding settings, generation timestamp, original HumanEval prompt, complete generation prompt, raw response, and extracted Python code.
- Non-parsable outputs are retained for functional summaries but excluded from AST-based distance calculations.
- SSI is undefined when fewer than two AST-parsable repetitions are available for a condition.
- PSSI is computed from distances between valid prompt-level centroids.
- Standardization is recomputed from the AST-parsable sample set before structural distances are calculated.
- Problem-cluster bootstrap confidence intervals use 10,000 resamples with seed 42.

## Security

Never commit `.env`, API keys, local virtual environments, or editor-specific files. The repository `.gitignore` excludes these paths.

Before sharing or archiving a local project folder, remove at least:

```text
.env
.git/
.venv/
venv/
```

Only `.env.example` should be distributed.

## Citation

Until a final publication record and DOI are available, cite the manuscript as:

```text
J. Lee and I. Joe,
“A Model-Agnostic AST-Based Framework for Evaluating Structural
Reliability in LLM Code Generation,” manuscript, 2026.
```

The citation and archival DOI will be updated after publication.

## Authors

**Jieun Lee**  
Department of Information Systems, Graduate School, Hanyang University, Seoul, Republic of Korea

**Inwhee Joe** — Corresponding Author  
Department of Computer Science, Hanyang University, Seoul, Republic of Korea  
Email: iwjoe@hanyang.ac.kr

## Third-Party Material

The files under `external/human-eval/` retain their original HumanEval license and attribution. No ownership of third-party benchmark material is claimed by this repository.
