"""Within-paraphrase, within-task/model prompt-label permutation calibration.

Uses the same experiment-specific population z scores as manuscript Table 24.
Preserves the observed number of valid ASTs in each prompt group. The statistic
is the equally weighted mean PSSI over task/model groups, tested one-sided.
The HE-0 sensitivity analysis omits all three groups containing early records;
it retains the original scale and permutation draws, without re-standardizing.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

FEATURES = ['ast_depth', 'branch_count', 'loop_count', 'function_count', 'control_flow_ratio']


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/paraphrase_features.csv'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--permutations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=4201)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    valid = data[data.ast_success].copy()
    valid[FEATURES] = (valid[FEATURES] - valid[FEATURES].mean()) / valid[FEATURES].std(ddof=0)
    rng = np.random.default_rng(args.seed)
    rows, nulls = [], []
    for (problem, model), group in valid.groupby(['problem_id', 'model_name']):
        group = group.sort_values('prompt_name')
        x = group[FEATURES].to_numpy()
        sizes = group.groupby('prompt_name', sort=True).size().to_numpy()
        assert len(sizes) == 5 and sizes.min() >= 2
        cuts = np.r_[0, sizes.cumsum()]
        centers = np.stack([x[cuts[j]:cuts[j+1]].mean(0) for j in range(5)])
        obs = pdist(centers).mean()
        orders = np.argsort(rng.random((args.permutations, len(x))), axis=1)
        shuffled = x[orders]
        centers = np.stack([shuffled[:, cuts[j]:cuts[j+1]].mean(1) for j in range(5)], axis=1)
        null = np.zeros(args.permutations)
        for j in range(5):
            for k in range(j + 1, 5):
                null += np.linalg.norm(centers[:, j] - centers[:, k], axis=1) / 10
        rows.append({'problem_id': problem, 'model_name': model, 'n_valid': len(x),
                     'observed_pssi': obs, 'null_mean_pssi': null.mean()})
        nulls.append(null)
    frame = pd.DataFrame(rows)
    nulls = np.stack(nulls)
    summaries = []
    for subset in ['all_tasks', 'exclude_he0']:
        for scope in [*sorted(frame.model_name.unique()), 'overall']:
            mask = np.ones(len(frame), dtype=bool)
            if subset == 'exclude_he0':
                mask &= frame.problem_id.to_numpy() != 'humaneval_000'
            if scope != 'overall':
                mask &= frame.model_name.to_numpy() == scope
            obs = frame.loc[mask, 'observed_pssi'].mean()
            null = nulls[mask].mean(0)
            row = {'subset': subset, 'scope': scope, 'n_groups': int(mask.sum()),
                   'observed_mean_pssi': obs, 'permuted_mean_pssi': null.mean(),
                   'observed_minus_null_mean': obs - null.mean(),
                   'null_central95_low': np.quantile(null, 0.025),
                   'null_central95_high': np.quantile(null, 0.975),
                   'exceedances': int(np.sum(null >= obs)),
                   'one_sided_p': (1 + np.sum(null >= obs)) / (args.permutations + 1)}
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary['holm_p_three_models'] = np.nan
    for subset in ['all_tasks', 'exclude_he0']:
        mask = (summary.subset == subset) & (summary.scope != 'overall')
        summary.loc[mask, 'holm_p_three_models'] = holm(summary.loc[mask, 'one_sided_p'])
    frame.to_csv(args.output / 'permutation_group_scores.csv', index=False)
    summary.to_csv(args.output / 'permutation_summary.csv', index=False)
    np.savez_compressed(args.output / 'permutation_null_means.npz', group_null_pssi=nulls)
    (args.output / 'permutation_metadata.json').write_text(json.dumps({
        'permutations': args.permutations, 'seed': args.seed,
        'scope': 'conditional on valid ASTs and fixed prompt set; post hoc calibration',
        'scale': 'within-paraphrase population z scores, ddof=0',
        'null': 'exchangeable prompt labels within each problem/model; valid counts preserved',
        'p_value': '(1 + null statistics >= observed)/(B + 1), one-sided',
        'multiplicity': 'Holm adjustment of three model tests; overall aggregate reported separately',
        'interval': 'central 95% of null means, NOT effect confidence intervals',
        'he0': 'exclude all three HE-0 task/model groups; retain fitted scale and same null draws'
    }, indent=2) + '\n')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
