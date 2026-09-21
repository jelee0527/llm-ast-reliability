"""Reproduce additional PSSI/SDS validation without executing generated code.

Python 3.12; numpy, pandas and scipy. See README.md for definitions and provenance.
The main representation uses ordered rooted AST neighborhoods of depths 0, 1, 2.
Depth limits 1 and 3 are reported as sensitivity checks, not selected by outcomes.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

FEATURES = ['ast_depth', 'branch_count', 'loop_count', 'function_count', 'control_flow_ratio']
KEYS = ['problem_id', 'model_name', 'prompt_name', 'repeat_idx']


def descriptors(tree):
    nodes = list(ast.walk(tree))
    def depth(n):
        return 1 + max((depth(c) for c in ast.iter_child_nodes(n)), default=0)
    return [depth(tree),
            sum(isinstance(n, (ast.If, ast.IfExp, ast.Match, ast.Try)) for n in nodes),
            sum(isinstance(n, (ast.For, ast.AsyncFor, ast.While)) for n in nodes),
            sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes),
            round(sum(isinstance(n, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While,
                                    ast.Try, ast.Break, ast.Continue, ast.Return,
                                    ast.Raise, ast.Match)) for n in nodes) / len(nodes), 6)]


def rooted_profiles(tree, max_depth=3):
    """Exact tuple signatures: AST types, child field names and ordered children.

    Scalar fields (names, literal values, type comments, locations) are excluded.
    Operator/context AST nodes ARE retained. No lossy feature hashing is used.
    """
    nodes = list(ast.walk(tree))
    labels = {id(n): type(n).__name__ for n in nodes}
    profiles = [Counter(labels[id(n)] for n in nodes)]
    for _ in range(max_depth):
        nxt = {}
        for n in nodes:
            edges = []
            for field, value in ast.iter_fields(n):
                if isinstance(value, ast.AST):
                    edges.append((field, labels[id(value)]))
                elif isinstance(value, list):
                    edges.extend((field, labels[id(c)]) for c in value if isinstance(c, ast.AST))
            nxt[id(n)] = (type(n).__name__, tuple(edges))
        labels = nxt
        profiles.append(Counter(labels[id(n)] for n in nodes))
    return profiles


def profile_matrix(profiles, height):
    # For each level h: sqrt(count / node_count), with weight 1/sqrt(H+1).
    # Concatenation has L2 norm 1 and prevents raw program size dominating.
    vocab, data, indices, indptr = {}, [], [], [0]
    for levels in profiles:
        n = sum(levels[0].values())
        for h in range(height + 1):
            for sig, count in levels[h].items():
                key = (h, sig)
                j = vocab.setdefault(key, len(vocab))
                indices.append(j)
                data.append(np.sqrt(count / (n * (height + 1))))
        indptr.append(len(data))
    matrix = csr_matrix((data, indices, indptr), shape=(len(profiles), len(vocab)))
    assert np.allclose(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel(), 1)
    return matrix, len(vocab)


def distances_from_gram(k):
    sq = np.diag(k)[:, None] + np.diag(k)[None, :] - 2 * k
    sq[np.abs(sq) < 1e-14] = 0
    assert sq.min() >= -1e-12
    return np.sqrt(np.maximum(sq, 0))


def rich_scores(k, prompt_names):
    unique = sorted(set(prompt_names))
    a = np.array([(prompt_names == p).astype(float) / np.sum(prompt_names == p) for p in unique])
    centroids = a @ k @ a.T
    between = distances_from_gram(centroids)
    pssi = between[np.triu_indices(len(unique), 1)].mean()
    sq = np.diag(k) - 2 * k.mean(axis=1) + k.mean()
    sq[np.abs(sq) < 1e-14] = 0
    sds = np.sqrt(np.maximum(sq, 0)).mean()
    return pssi, sds


def sanity_checks():
    a, b = ast.parse('def f(a,b):\n return a+b'), ast.parse('def f(a,b):\n return a*b')
    assert descriptors(a) == descriptors(b)
    matrix, _ = profile_matrix([rooted_profiles(a), rooted_profiles(b)], 2)
    k = (matrix @ matrix.T).toarray()
    assert distances_from_gram(k)[0, 1] > 0  # operators invisible to five descriptors
    # Names and literal values deliberately remain outside this comparison.
    assert rooted_profiles(ast.parse('x = 1')) == rooted_profiles(ast.parse('y = 2'))
    # Verify sparse-Gram calculations against direct Euclidean calculations.
    rng = np.random.default_rng(12)
    x = rng.normal(size=(25, 9))
    labels = np.repeat(np.arange(5), 5)
    p, s = rich_scores(x @ x.T, labels)
    assert np.isclose(p, pdist(x.reshape(5, 5, 9).mean(1)).mean())
    assert np.isclose(s, np.linalg.norm(x - x.mean(0), axis=1).mean())


def analyze_experiment(name, raw_dir, reference_dir, out):
    records, profiles, manifest = [], [], []
    for path in sorted(raw_dir.glob('*.json')):
        raw = path.read_bytes()
        item = json.loads(raw)
        row = {k: item[k] for k in KEYS}
        row['file'] = path.name
        code = item.get('generated_code', '')
        try:
            if item.get('status') != 'success' or not code.strip():
                raise ValueError('generation_error_or_empty')
            tree = ast.parse(code)
            row.update(zip(FEATURES, descriptors(tree)))
            row['ast_success'] = True
            row['profile_index'] = len(profiles)
            profiles.append(rooted_profiles(tree))
        except (SyntaxError, ValueError):
            row['ast_success'] = False
            row['profile_index'] = -1
        records.append(row)
        manifest.append({'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest()})
    data = pd.DataFrame(records)
    assert len(data) == 12300 and not data.duplicated(KEYS).any()
    ref = pd.read_csv(reference_dir / 'metrics_summary.csv')
    checked = data.merge(ref[KEYS + ['ast_success'] + FEATURES], on=KEYS,
                         suffixes=('', '_ref'), validate='one_to_one')
    assert len(checked) == 12300
    assert (checked.ast_success == checked.ast_success_ref).all()
    for f in FEATURES:
        assert np.allclose(checked[f], checked[f + '_ref'], equal_nan=True, atol=1e-12, rtol=0), f
    data.drop(columns='profile_index').to_csv(out / f'{name}_features.csv', index=False)
    pd.DataFrame(manifest).to_csv(out / f'{name}_input_sha256.csv', index=False)
    valid = data[data.ast_success].copy()
    valid[FEATURES] = (valid[FEATURES] - valid[FEATURES].mean()) / valid[FEATURES].std(ddof=0)
    rows, vocabs = [], {}
    for height in (1, 2, 3):
        matrix, dimension = profile_matrix(profiles, height)
        vocabs[height] = dimension
        for (problem, model), group in valid.groupby(['problem_id', 'model_name']):
            x = group[FEATURES].to_numpy()
            centroids = group.groupby('prompt_name')[FEATURES].mean().to_numpy()
            pssi = pdist(centroids).mean()
            sds = np.linalg.norm(x - x.mean(axis=0), axis=1).mean()
            sub = matrix[group.profile_index.to_numpy()]
            rp, rs = rich_scores((sub @ sub.T).toarray(), group.prompt_name.to_numpy())
            rows.append({'experiment': name, 'height': height, 'problem_id': problem, 'model_name': model,
                         'n_valid': len(group), 'n_prompts': group.prompt_name.nunique(),
                         'pssi': pssi, 'sds': sds, 'subtree_pssi': rp, 'subtree_sds': rs})
        print(f'{name}: height={height}, valid={len(valid)}, vocabulary={dimension}', flush=True)
    result = pd.DataFrame(rows)
    for metric, filename in [('pssi', 'prompt_sensitivity.csv'), ('sds', 'structural_diversity.csv')]:
        saved = pd.read_csv(reference_dir / filename)
        merged = result[result.height == 2].merge(saved[['problem_id', 'model_name', metric]],
                    on=['problem_id', 'model_name'], suffixes=('', '_ref'), validate='one_to_one')
        assert len(merged) == 492
        assert np.allclose(merged[metric], merged[metric+'_ref'], atol=2e-12, rtol=0), metric
    return result, {'n_generated': len(data), 'n_valid': len(valid), 'vocabulary_sizes': vocabs}


def correlations(frame, bootstrap_count, seed):
    rows = []
    for (experiment, height), part in frame.groupby(['experiment', 'height']):
        for scope in ['overall', *sorted(part.model_name.unique())]:
            group = part if scope == 'overall' else part[part.model_name == scope]
            group = group.sort_values(['problem_id', 'model_name'])
            tasks = sorted(group.problem_id.unique())
            task_rows = [np.flatnonzero(group.problem_id.to_numpy() == task) for task in tasks]
            assert len({len(a) for a in task_rows}) == 1
            task_rows = np.stack(task_rows)
            for metric in ('pssi', 'sds'):
                x, y = group[metric].to_numpy(), group['subtree_' + metric].to_numpy()
                rho = float(spearmanr(x, y).statistic)
                row = {'experiment': experiment, 'height': height, 'scope': scope,
                       'metric': metric, 'n_groups': len(group), 'n_tasks': len(tasks), 'spearman_rho': rho}
                if height == 2:
                    # Same draws for both metrics/scopes; all models of a sampled task stay together.
                    rng = np.random.default_rng(seed)
                    values = []
                    for draw in rng.integers(len(tasks), size=(bootstrap_count, len(tasks))):
                        ix = task_rows[draw].ravel()
                        values.append(spearmanr(x[ix], y[ix]).statistic)
                    assert np.isfinite(values).all()
                    row['ci_low'], row['ci_high'] = np.quantile(values, [0.025, 0.975])
                    row['bootstrap_replicates'] = bootstrap_count
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--paraphrase-raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--bootstrap', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=20260920)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sanity_checks()
    metadata = {'python': platform.python_version(), 'numpy': np.__version__,
                'pandas': pd.__version__, 'scipy': scipy.__version__,
                'bootstrap': args.bootstrap, 'seed': args.seed, 'primary_height': 2,
                'sensitivity_heights': [1, 3], 'experiments': {}}
    frames = []
    for name, raw, ref in [('primary', args.repo / 'outputs/raw', args.repo / 'results'),
                           ('paraphrase', args.paraphrase_raw, args.repo / 'results/paraphrase_structural')]:
        frame, info = analyze_experiment(name, raw, ref, args.output)
        metadata['experiments'][name] = info
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.output / 'subtree_group_scores.csv', index=False)
    summary = correlations(frame, args.bootstrap, args.seed)
    summary.to_csv(args.output / 'subtree_correlations.csv', index=False)
    (args.output / 'subtree_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(summary[summary.height == 2].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
