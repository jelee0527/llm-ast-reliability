# LLM AST Reliability

Replication package for the manuscript:

> **A Model-Agnostic AST-Based Framework for Evaluating Structural Reliability in LLM Code Generation**  
> Jieun Lee and Inwhee Joe, 2026.

This repository contains the frozen LLM generations, HumanEval and HumanEval+ functional-evaluation outputs, AST-derived metrics, statistical analyses, semantic-preserving paraphrase experiments, prompt templates, tables, figures, and source code used in the study and its reviewer-requested robustness analyses.

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

| Item                 | Setting                              |
| -------------------- | ------------------------------------ |
| Benchmark            | Official HumanEval                   |
| Problems             | 164                                  |
| Models               | 3                                    |
| Prompt conditions    | 5                                    |
| Repetitions          | 5 per problem-model-prompt condition |
| Total generations    | 12,300                               |
| API collection dates | June 9–11, 2026                      |

The supplementary semantic-preserving paraphrase experiment uses the same 164 problems and three model conditions with five equivalent phrasings of one neutral instruction and five repetitions, producing another 12,300 samples.

### Provider-Specific Generation Settings

| Model             | Temperature sent to API | Maximum output length | Additional setting          |
| ----------------- | ----------------------: | --------------------: | --------------------------- |
| Claude Sonnet 4.6 |                     0.2 |          1,200 tokens | None                        |
| GPT-5 mini        |           Not specified |          3,000 tokens | Reasoning effort: `minimal` |
| DeepSeek Chat     |                     0.2 |          1,200 tokens | None                        |

### Model Conditions

| Display name      | Provider  | Model identifier    |
| ----------------- | --------- | ------------------- |
| Claude Sonnet 4.6 | Anthropic | `claude-sonnet-4-6` |
| GPT-5 mini        | OpenAI    | `gpt-5-mini`        |
| DeepSeek Chat     | DeepSeek  | `deepseek-chat`     |

For the supplementary paraphrase collection, the DeepSeek endpoint returned `deepseek-flash`; requested and returned identifiers are preserved separately in the per-sample metadata.

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

### Reviewer-Requested Robustness Results

- Original samples: 11,701 EvalPlus base passes (95.13%) and 11,110 HumanEval+ passes (90.33%); 79 evaluator disagreements are reported separately.
- Paraphrase samples: 11,904 EvalPlus base passes (96.78%) and 11,299 HumanEval+ passes (91.86%).
- On a common pooled feature scale, mean PSSI decreased from 0.914 to 0.392 for semantic-preserving paraphrases (57.1%; Holm-adjusted Wilcoxon p < 0.001).
- Descriptor-based SSI agreed strongly with reduced-AST edit stability over 2,459 valid conditions (Spearman's rho = 0.969).
- Six DeepSeek outputs reached the 1,200-token limit. In a separate post-hoc restriction analysis, 19 of 4,100 GPT-5 mini outputs exceeded 1,200 tokens. Excluding them changed branch count by -0.0918, SSI by -0.0038, PSSI by +0.0198, SDS by +0.0152, and HumanEval+ pass rate by +0.0037. This is not equivalent to regeneration under a common cap.

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
python scripts/compute_legacy_metrics.py
python scripts/compute_tree_edit_validation.py
python scripts/export_latex_tables.py
python scripts/visualize_final.py
python scripts/validate_reproducibility.py
```

Expected key checks:

```text
Total samples: 12300
AST success samples: 12287
Functional passes: 11766
Valid SSI conditions: 2459
SSI vs. tree-edit stability Spearman rho: 0.968974
```

The valid SSI count is one lower than the full set of 2,460
problem-model-prompt conditions because `HumanEval/41` × `gpt5_model` ×
`constraint` contains only one AST-parsable repetition. SSI requires at least
two valid repetitions, so this condition is excluded consistently from every
SSI-based correlation analysis.

Outputs are written to:

- `results/` for CSV summaries and statistical results
- `reports/tables/` for LaTeX tables
- `reports/figures/` for 300-dpi PNG figures

The analysis workflow was verified from the included outputs and reproduced the manuscript-level counts, rankings, blocked Friedman tests, Holm-adjusted Wilcoxon comparisons, bootstrap confidence intervals, and feature-ablation results.

## Validating SSI with Tree Edit Distance

`scripts/compute_tree_edit_validation.py` provides an independent structural
validation of the descriptor-based SSI. It converts each parsable generation
to an ordered, reduced AST that removes identifier names, literal values,
contexts, and operator-token leaves while retaining statement nesting,
control-flow, call, expression, and comprehension structure. For each
problem-model-prompt condition, it computes the normalized APTED distance for
all pairs of valid repetitions and defines tree-edit stability as one minus
the mean normalized distance.

Across the 2,459 conditions for which SSI is defined, descriptor-based SSI and
tree-edit stability have Spearman's rho = 0.968974 (p < 0.001). The calculation
uses 24,558 repeat-pair comparisons. Detailed condition-, model-, and
prompt-level results are stored in `results/tree_edit_*.csv`.

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

## Running the HumanEval+ Evaluation

HumanEval+ is evaluated against the same 12,300 frozen generations; no new LLM
calls are required. Because this step executes generated code, run it only in
the isolated Docker container provided here.

Prepare EvalPlus input and build the pinned evaluator image:

```bash
python scripts/prepare_evalplus_samples.py
docker build -f Dockerfile.evalplus -t evalplus-evaluator .
```

Run the evaluation with network access disabled and container privileges
restricted:

```bash
docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --cpus 4 \
  --memory 6g \
  --pids-limit 1024 \
  --shm-size 1g \
  --tmpfs /tmp:rw,exec,nosuid,size=2g \
  -v "${PWD}/outputs/evalplus:/work" \
  evalplus-evaluator \
  humaneval \
  --samples /work/humaneval_plus_samples.jsonl \
  --parallel 4
```

Then merge the detailed EvalPlus output with the model, prompt, and repetition
metadata from the frozen experiment:

```bash
python scripts/summarize_evalplus.py
```

The large derived JSONL input and detailed EvalPlus output remain under
`outputs/evalplus/` and are not committed. Reusable CSV summaries are written
to `results/`. The summarizer compares EvalPlus base-test decisions with the
original HumanEval decisions; the 79 observed disagreements are exported and
reported explicitly rather than silently reconciled.

## Reproducing the Semantic-Paraphrase Robustness Study

The exact supplementary settings and prompts are stored in
`configs/paraphrase_models.yaml` and `configs/paraphrase_prompts.yaml`.
Generating new responses is optional and requires provider API keys.

```bash
python scripts/generate_paraphrase.py
python scripts/prepare_evalplus_samples.py \
  --raw-dir outputs/paraphrase/raw \
  --output outputs/evalplus/paraphrase_samples.jsonl
```

After evaluating that JSONL with the isolated EvalPlus container, run:

```bash
python scripts/summarize_paraphrase_evalplus.py
python scripts/run_paraphrase_structural.py
python scripts/analyze_paraphrase_common_scale.py
python scripts/validate_reproducibility.py
```

The common-scale script reproduces the paired PSSI comparison, confidence
intervals, DeepSeek truncation analysis, and post-hoc GPT 1,200-token
restriction analysis. Primary outputs are stored under
`results/paraphrase_structural/` and `results/paraphrase_robustness/`.

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

# Optional global overrides. Leave commented to use the exact
# provider-specific settings in configs/models.yaml.
# EXPERIMENT_TEMPERATURE=0.2
# EXPERIMENT_MAX_TOKENS=1200

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

API-hosted models may change over time. Therefore, newly generated responses are not expected to be byte-identical to the frozen responses collected on June 9–11, 2026. The files in `outputs/raw/` are the authoritative generation artifacts for the reported study.

## Data and Reproducibility Notes

- All 12,300 raw generation records include the provider, model identifier, prompt condition, repetition index, stored generation metadata, generation timestamp, original HumanEval prompt, complete generation prompt, raw response, and extracted Python code.
- **Metadata note for GPT-5 mini:** the frozen GPT-5 mini JSON records contain `"temperature": 0.2` because the original logger stored the global environment value for every provider. The OpenAI request path did not transmit a temperature parameter for `gpt-5-mini`; it used `reasoning={"effort": "minimal"}` and `max_output_tokens=3000`. The frozen records are preserved unchanged for provenance, while the current generation script records the actual provider-specific request settings.
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
