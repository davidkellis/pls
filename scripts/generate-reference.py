#!/usr/bin/env python3
"""Generate independent PLS/OLS reference fixtures; requires NumPy and scikit-learn."""
from pathlib import Path
import json
import warnings
import numpy as np
import scipy
import sklearn
from sklearn.cross_decomposition import PLSRegression

ROOT = Path(__file__).resolve().parents[1]
cases = []
for seed in range(8):
    rng = np.random.default_rng(seed)
    for name, n, p, targets in [('single', 24, 5, 1), ('multi', 27, 6, 3), ('wide', 14, 23, 2)]:
        x = rng.normal(size=(n, p))
        x[:, 1] = 0.97 * x[:, 0] + 0.12 * x[:, 1]
        beta = rng.normal(size=(p, targets))
        y = x @ beta + rng.normal(scale=0.15, size=(n, targets)) + np.arange(targets) + 4
        evaluation = rng.normal(size=(5, p))
        for scale in (True, False):
            components = 1 + seed % 4
            cases.append((f'{name}-{seed}-scale-{scale}', x, y, evaluation, components, scale))

rng = np.random.default_rng(314)
x = rng.normal(size=(20, 4))
y = x @ rng.normal(size=(4, 2)) + 3
cases += [('constant-column', np.column_stack([x, np.full(20, 7), np.zeros(20)]), y,
           np.column_stack([x[:4], np.full(4, 7), np.zeros(4)]), 3, True),
          ('duplicates', np.column_stack([x, x[:, 0], 2 * x[:, 1]]), y,
           np.column_stack([x[:4], x[:4, 0], 2 * x[:4, 1]]), 3, True),
          ('constant-target', x, np.full((20, 1), 9.5), x[:4], 2, True),
          ('mixed-constant-target', x, np.column_stack([y, np.full(20, -2)]), x[:4], 3, True),
          ('many-targets', x[:, :2], np.column_stack([y, 2*y, -y]), x[:4, :2], 2, True),
          ('different-units', x * [0.001, 1000, 3, 1] + [100, -50, 1000, 0], y * [20, .02] + [20, 80],
           x[:4] * [0.001, 1000, 3, 1] + [100, -50, 1000, 0], 3, True)]

records = []
for name, x, y, evaluation, components, scale in cases:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        pls = PLSRegression(n_components=components, scale=scale, tol=1e-24, max_iter=10000).fit(x, y)
    means = x.mean(axis=0)
    scales = x.std(axis=0, ddof=1) if scale else np.ones(x.shape[1])
    scales[scales == 0] = 1
    z = (x - means) / scales
    centered_y = y - y.mean(axis=0)
    b, _, rank, _ = np.linalg.lstsq(z, centered_y, rcond=np.finfo(float).eps * max(z.shape))
    coefficients = b / scales[:, None]
    intercept = y.mean(axis=0) - means @ coefficients
    records.append(dict(name=name, x=x.tolist(), y=y.tolist(), evaluation=evaluation.tolist(), components=components,
                        scale=scale, plsPredictions=pls.predict(evaluation).tolist(),
                        plsCoefficients=pls.coef_.T.tolist(),
                        olsPredictions=(evaluation @ coefficients + intercept).tolist(),
                        olsCoefficients=coefficients.tolist(), olsRank=int(rank)))

target = ROOT / 'src/test/resources/pls-reference.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps({'reference': {'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__,
    'pls': 'PLSRegression (NIPALS), tol=1e-24, max_iter=10000',
    'ols': 'numpy.linalg.lstsq, rcond=eps*max(shape), centered and optionally sample-SD-scaled X'}, 'cases': records}, separators=(',', ':')) + '\n')
print(f'Generated {len(records)} independent reference cases at {target}.')
