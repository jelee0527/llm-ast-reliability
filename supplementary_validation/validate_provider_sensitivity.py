"""Reproduce Section VI-D exclusions on the fixed paraphrase feature scale.

Reads the included descriptors and saved EvalPlus outcomes. No API calls,
raw-program execution, or functional retesting is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

FEATURES = ['ast_depth', 'branch_count', 'loop_count', 'function_count',
            'control_flow_ratio']
ZFEATURES = ['z_' + name for name in FEATURES]
METRICS = ['num_samples', 'num_valid_asts', 'functional_pass_rate',
           'branch_count', 'ssi', 'pssi', 'sds']


def model_scores(frame):
    rows = []
    for model, all_rows in frame.groupby('model_name', sort=True):
        valid = all_rows.loc[all_rows['ast_success'] &
                             all_rows[ZFEATURES].notna().all(axis=1)]
        stability, sensitivity, diversity = [], [], []
        for _, task in valid.groupby('problem_id', sort=True):
            centroids = []
            for _, prompt in task.groupby('prompt_name', sort=True):
                x = prompt[ZFEATURES].to_numpy(float)
                centroids.append(x.mean(axis=0))
                if len(x) >= 2:
                    stability.append(1.0 / (1.0 + pdist(x).mean()))
            if len(centroids) >= 2:
                sensitivity.append(pdist(np.asarray(centroids)).mean())
            x = task[ZFEATURES].to_numpy(float)
            diversity.append(np.linalg.norm(x - x.mean(axis=0), axis=1).mean())
        rows.append(dict(model_name=model, num_samples=len(all_rows),
                         num_valid_asts=len(valid),
                         functional_pass_rate=all_rows['functional_passed'].mean(),
                         branch_count=valid['branch_count'].mean(),
                         ssi=np.mean(stability), pssi=np.mean(sensitivity),
                         sds=np.mean(diversity)))
    return pd.DataFrame(rows).set_index('model_name')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--output', type=Path, default=Path('reproduced_provider_sensitivity'))
    args = parser.parse_args()
    files = ['paraphrase_features.csv', 'paraphrase_functional_status.csv',
             'truncated_samples.csv', 'gpt_over_1200_token_samples.csv']
    frame = pd.read_csv(args.input_dir / files[0])
    statuses = pd.read_csv(args.input_dir / files[1])
    assert len(frame) == len(statuses) == 12300
    assert set(frame['file']) == set(statuses['file'])
    assert frame['ast_success'].isin([True, False]).all()
    assert statuses['functional_passed'].isin([True, False]).all()
    frame = frame.merge(statuses, on='file', validate='one_to_one')
    assert not frame.duplicated(['problem_id', 'model_name', 'prompt_name', 'repeat_idx']).any()
    valid = frame['ast_success'] & frame[FEATURES].notna().all(axis=1)
    assert int(valid.sum()) == 12291
    reference = frame.loc[valid, FEATURES]
    means, stds = reference.mean(), reference.std(ddof=0)
    for name in FEATURES:
        frame['z_' + name] = np.nan
        frame.loc[valid, 'z_' + name] = (
            (reference[name] - means[name]) / stds[name]
            if stds[name] >= 1e-12 else 0.0
        )
    baseline = model_scores(frame)
    rows = []
    exclusions = [('deepseek_no_truncation', files[2], 6, 'deepseek_model'),
                  ('gpt_1200_restricted', files[3], 19, 'gpt5_model')]
    for analysis, filename, count, affected in exclusions:
        excluded = pd.read_csv(args.input_dir / filename)
        assert len(excluded) == excluded['file'].nunique() == count
        assert set(excluded['file']).issubset(set(frame['file']))
        assert set(excluded['model_name']) == {affected}
        # Keep z coordinates fitted above: no scale refit follows this filter.
        retained = frame.loc[~frame['file'].isin(excluded['file'])].copy()
        assert len(retained) == 12300 - count
        scores = model_scores(retained)
        for model in baseline.index:
            row = {'analysis': analysis, 'model_name': model,
                   'excluded_samples': count if model == affected else 0}
            for metric in METRICS:
                row[metric + '_full'] = baseline.loc[model, metric]
                row[metric + '_retained'] = scores.loc[model, metric]
                row[metric + '_difference'] = scores.loc[model, metric] - baseline.loc[model, metric]
            if model != affected:
                np.testing.assert_allclose(scores.loc[model], baseline.loc[model], rtol=0, atol=1e-12)
            rows.append(row)
    output = pd.DataFrame(rows)
    metadata = {
        'section': 'VI-D', 'reference_samples': 12300, 'reference_valid_asts': 12291,
        'scale': 'All valid paraphrase ASTs, all models pooled; held fixed after exclusion',
        'ddof': 0, 'feature_mean': means.to_dict(), 'feature_std': stds.to_dict(),
        'functional_status': 'Saved HumanEval+ outcomes; not re-executed',
        'input_repository_commit': 'cdff479ea5134979c3f3d65e203688cac27bc6de',
        'input_sha256': {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                         for name in files},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'pandas': pd.__version__},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / 'provider_sensitivity_summary.csv', index=False)
    (args.output / 'provider_sensitivity_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(output[['analysis', 'model_name', 'ssi_difference', 'pssi_difference',
                  'sds_difference', 'functional_pass_rate_difference']].to_string(index=False))


if __name__ == '__main__':
    main()
