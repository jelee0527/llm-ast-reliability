# Additional structural validation for Access-2026-35219

This directory reproduces the post hoc prompt-label calibration in Section VI-E
(Table 27) and the PSSI/SDS ordered-subtree validation in Section VI-F (Table 28).
It uses existing generated programs. It makes no model API calls and never
executes generated code. Functional test results are not recomputed here.

## Inputs and provenance

- Repository: https://github.com/jelee0527/llm-ast-reliability
- Frozen repository commit: `cdff479ea5134979c3f3d65e203688cac27bc6de`
- Primary raw programs: `outputs/raw/` in that repository snapshot, 12,300 JSONs.
- Paraphrase raw archive: release `v1.1.0-rnr`,
  `paraphrase_raw_responses_12300.zip`, 12,300 JSONs under `raw/`.
- Archive URL:
  https://github.com/jelee0527/llm-ast-reliability/releases/download/v1.1.0-rnr/paraphrase_raw_responses_12300.zip
- Archive SHA-256:
  `cfd4590dc353511f8b2a239efebcfdafaa19f189caeac43dd729ed9f39761341`
- `results/primary_input_sha256.csv` and
  `results/paraphrase_input_sha256.csv` identify every analyzed input by filename
  and SHA-256. The raw archive and repository remain the authoritative sources.

The primary outputs were collected June 9–11, 2026, and the paraphrase outputs
September 15, 2026. The prompt families, collection times and provider identifiers
are not interchangeable experimental treatments. The main manuscript documents
the deployment settings and their limitations. The three early supplementary
schedule exceptions are HE-0 / paraphrase A / repetition 1, one per model.

## Reproduction

The executed environment was Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and SciPy
1.17.0. Python 3.12 is required for consistency of the AST types and definitions.
Use an isolated environment and install `requirements-validation.txt`.

Run from the **repository root**, using a separate environment for these
scripts. The root `requirements.txt` belongs to the original analyses.

Windows PowerShell:

```powershell
py -3.12 -m venv supplementary_validation/.venv
supplementary_validation\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3.12 -m venv supplementary_validation/.venv
source supplementary_validation/.venv/bin/activate
```

Then install the additional analysis dependencies:

```bash
python -m pip install -r supplementary_validation/requirements-validation.txt
```

Download the named raw archive from the URL above into the repository root.
Confirm its SHA-256 against the value above and extract it (the archive already
has a `raw/` top level):

```bash
python -c "import hashlib; from pathlib import Path; p=Path('paraphrase_raw_responses_12300.zip'); actual=hashlib.sha256(p.read_bytes()).hexdigest(); assert actual == 'cfd4590dc353511f8b2a239efebcfdafaa19f189caeac43dd729ed9f39761341', actual; print('SHA-256 verified')"
python -m zipfile -e paraphrase_raw_responses_12300.zip outputs/paraphrase
python supplementary_validation/validate_structure.py --repo . --paraphrase-raw outputs/paraphrase/raw --output supplementary_validation/reproduced
python supplementary_validation/calibrate_pssi.py --input supplementary_validation/reproduced/paraphrase_features.csv --output supplementary_validation/reproduced
```

The permutation analysis can also be reproduced from the included minimal
descriptor CSV without downloading the raw programs:

```bash
python supplementary_validation/calibrate_pssi.py --input supplementary_validation/results/paraphrase_features.csv --output supplementary_validation/reproduced_permutation
```

The recorded repository commit identifies the **frozen input data**; it predates
the addition of these scripts. To reproduce from that exact input snapshot,
keep this working tree and create a separate checkout:

```bash
git clone https://github.com/jelee0527/llm-ast-reliability ../llm-ast-reliability-frozen
git -C ../llm-ast-reliability-frozen checkout cdff479ea5134979c3f3d65e203688cac27bc6de
python supplementary_validation/validate_structure.py --repo ../llm-ast-reliability-frozen --paraphrase-raw outputs/paraphrase/raw --output supplementary_validation/reproduced
```

The paraphrase records still come from the independently checksummed release.
After this exact-snapshot command, run `calibrate_pssi.py` on
`supplementary_validation/reproduced/paraphrase_features.csv` as above.
Reproduced outputs are written separately from the supplied `results/` files.

`validate_structure.py` checks all sample keys, AST success flags, five extracted
features and recomputed PSSI/SDS against the frozen repository CSVs. It also
checks that a change from addition to multiplication is visible to the richer
representation despite identical five-descriptor vectors, and checks its
sparse-Gram calculations against direct Euclidean aggregation. Raw input files
are opened only as JSON/text and parsed using `ast.parse`.

## Table 27: prompt-label permutation calibration

The five original descriptors are standardized by their population standard
deviations (`ddof=0`) pooled over all 12,291 valid paraphrase ASTs. This is the
experiment-specific scale of Table 24, not the common primary/paraphrase scale
of Table 23. Labels are randomly reassigned separately within each task–model
group, preserving all five valid prompt-group sizes. Each shuffled statistic is
the equal-weight mean of task–model PSSI values, either over one model's 164
tasks or all 492 groups. We use 2,000 permutations and NumPy generator seed 4201.

The one-sided Monte Carlo p-value is `(1 + exceedances)/(2000 + 1)`, counting
permuted means greater than or equal to the observed mean. The three model
tests receive Holm adjustment within one family. The overall aggregate is
reported separately and is not counted as a fourth independent confirmation.
Central 95% ranges of permuted means describe the null reference; they are not
confidence intervals for the observed effect.

The sensitivity result removes every HE-0 model group, retaining the fitted
scale and the already generated null draws. It removes all three groups with
the early schedule exceptions, not merely the three individual observations.

| Scope | Observed PSSI | Mean under relabeling | Difference |
| --- | ---: | ---: | ---: |
| Claude | 0.220958 | 0.116051 | 0.104907 |
| DeepSeek | 0.258078 | 0.173158 | 0.084919 |
| GPT-5 mini | 0.702483 | 0.577248 | 0.125235 |
| Overall | 0.393840 | 0.288819 | 0.105021 |

Every raw p-value is 1/2001; each model's adjusted p-value is 3/2001. Omitting
HE-0 retains the result. This is a post hoc, conditional diagnostic under label
exchangeability. It does not identify a causal cross-run effect or remove the
sampling component from an individual PSSI, and it cannot rule out temporal
dependence or generalize beyond the fixed prompt set and parsable subset.

## Table 28: richer representation for PSSI and SDS

For every node in the complete parsed AST, we collect exact ordered rooted
neighborhood signatures at depths 0 through H. At depth 0 the signature is the
node type. At depth h+1 it is the node type paired with the ordered sequence
of child-field names and their depth-h signatures. Lists of children keep their
order. All AST node types, including operators and contexts, are retained.
Scalar values (identifiers, literals and positions) are excluded. Counts use
exact tuple keys; there is no feature hashing or dimension truncation.

For signature t at depth h, the vector component is:

`u[h,t] = sqrt(count[h,t] / (number_of_AST_nodes * (H + 1)))`.

Thus each depth contributes equally to a concatenated unit-length vector.
Square-root relative-frequency coordinates compare structural composition
without letting raw program size alone set the scale. Unseen signatures have
zero coordinates. Vocabularies are fitted separately by experiment; this only
indexes observed signatures and does not impose an experiment-specific learned
weight. This is an ordered downward AST adaptation inspired by subtree-feature
methods, not the standard undirected Weisfeiler–Lehman kernel. A finite set of
local signatures is not a lossless representation of the full tree.

The subtree PSSI counterpart is the mean Euclidean distance over unordered
pairs of prompt mean vectors. The subtree SDS counterpart is the mean
Euclidean distance of all valid vectors from their task–model centroid. We
hold these aggregations fixed to isolate the representation change. No
five-descriptor z scores enter the subtree representation.

H=2 was the main depth limit, and H=1 and H=3 were specified as sensitivity
checks before examining their correlations. All three are reported in
`subtree_correlations.csv`. At H=2 the primary and paraphrase vocabularies
contain 18,722 and 14,223 features. There are 492 task–model groups in each
experiment, based on 12,287 and 12,291 valid ASTs, respectively.

Spearman correlations are reported overall and separately for each model. The
main H=2 comparisons use 3,000 percentile bootstrap resamples (seed 20260920).
The sampling unit is one of the 164 HumanEval tasks. In overall comparisons,
all three models of a drawn task are retained together. Ranks are recomputed
within each bootstrap sample, including ties and repeated sampled tasks.
Intervals are pointwise 95% intervals, not simultaneous intervals. We do not
use nominal independent-observation correlation p-values as additional claims.

| Experiment | Comparison | Overall rho | Task-cluster 95% CI |
| --- | --- | ---: | --- |
| Primary | PSSI | 0.585734 | [0.501268, 0.658712] |
| Primary | SDS | 0.699917 | [0.637685, 0.754191] |
| Paraphrase | PSSI | 0.850506 | [0.812010, 0.881287] |
| Paraphrase | SDS | 0.896908 | [0.869854, 0.917367] |

Within-model correlations are positive. Their magnitudes vary, and primary
PSSI in particular shows material disagreement across representations. This
is convergent structural evidence, not independent external validation of
semantics, defect risk, maintainability or developer benefit. Local profiles
still omit long-range relations, data flow and semantic behavior. The common
aggregation formulas and samples also limit the independence of the check.

Methodological reference: N. Shervashidze et al., “Weisfeiler–Lehman graph
kernels,” Journal of Machine Learning Research, vol. 12, pp. 2539–2561, 2011.
https://jmlr.org/papers/v12/shervashidze11a.html

## File map

| File | Contents |
| --- | --- |
| `validate_structure.py` | Raw AST extraction, validation and subtree comparisons |
| `calibrate_pssi.py` | Prompt-label permutation calibration and HE-0 sensitivity |
| `requirements-validation.txt` | Versions used for these new scripts |
| `configs/` | Copies of both prompt-template YAMLs from the frozen commit |
| `results/*_features.csv` | Minimal sample keys, parsing flags and five descriptors |
| `results/*_input_sha256.csv` | SHA-256 manifest of each raw input file |
| `results/subtree_group_scores.csv` | All 984 groups at H=1,2,3 (2,952 rows) |
| `results/subtree_correlations.csv` | All scopes, metrics and depth limits |
| `results/subtree_metadata.json` | Actual runtime versions, counts, depths and seed |
| `results/permutation_group_scores.csv` | Observed PSSI and null mean per group |
| `results/permutation_summary.csv` | Full and HE-0-excluded scope results |
| `results/permutation_null_means.npz` | Group by permutation array; row order matches group CSV |
| `results/permutation_metadata.json` | Permutation definitions, seed and interpretation |

The scripts, configuration copies, minimal inputs, and results match the
accompanying `Supplementary_Validation_Access-2026-35219.zip`. This README adapts
the reproduction commands to the repository directory layout.
