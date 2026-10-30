# TypeScript PLS

This is the numerical implementation used by
[Conquer the Lawn's PLS comparison](https://conquerthelawn.com/demos/pls/).
The three files in `src/` are copied byte for byte from the website:

- `regression.ts`: Dayal–MacGregor algorithm 1 for PLS1/PLS2, SVD-based OLS,
  original-unit coefficients and intercepts, prediction, and RMSE.
- `experiment.ts`: seeded synthetic datasets, the three website scenarios,
  and five-fold component selection with preprocessing fitted within each fold.
- `experiment.worker.ts`: the website's worker entry point.

`provenance.json` records their website paths and SHA-256 hashes. The source
comment referring to `ops/pls-audit.md` refers to the website's audit document.
The MIT license is preserved in `LICENSE.txt`. The Astro page and presentation
code remain in the website project.

## Use

Requires Node 22.23.4 or later for direct execution of these TypeScript files.
Browser applications can bundle the modules with their existing build tools;
the only runtime dependency is pinned to `ml-matrix` 6.15.1.

```sh
npm ci
npm run check
npm test
```

```ts
import { fitPLS, predict } from './src/regression.ts';

const model = fitPLS(
  [[1, 1], [2, 2], [3, 3]],
  [[1], [2], [3]],
  { components: 2 },
);
console.log(model.components);          // 1: duplicate columns exhaust the rank
console.log(predict(model, [[4, 4]]));  // approximately [[4]]
```

The API always centers X and Y. Scaling by training sample standard deviations
is enabled by default and can be disabled with `scale: false`. Prediction
reuses the stored preprocessing through original-unit coefficients and
intercepts. Multiple responses use one column per response in Y.

The TypeScript API rejects component requests outside
`1..min(observations - 1, predictors)`; Scala's raw API allows larger positive
requests and caps fitting internally. The comparison uses the common valid
range. For unscaled comparisons, the harness centers data before calling the
raw Scala API and restores the training response means afterward.

## Compare with Scala

From the repository root, run:

```sh
python3 scripts/verify.py
```

This installs the locked npm dependencies, checks TypeScript, runs the 60
TypeScript tests and all Scala tests, and executes both actual Scala
algorithms against the TypeScript implementation on identical JSON datasets.
It requires Python 3 and Coursier, and provisions JDK 8 as described in the
root README. Optional arguments include `--coursier`, `--java-home`, and
`--website ../conquerthelawn.com`; the last checks that the copied modules and
independent fixtures still exactly match the website checkout.

The check covers 640 datasets: 54 independent NIPALS reference cases, 244
additional component counts for those datasets, 64 website scenario fits,
160 website validation-fold fits, 108 seeded single/multiple-response fits,
and 10 edge cases. Every fitted component count must match exactly. Every
coefficient, intercept, training prediction, and held-out prediction must be
finite and satisfy `abs(actual - expected) <= 2e-7 * max(1, abs(expected))`.
The three pairs of engines are compared directly, and each engine is also
checked against the 54 independent reference cases.

Five-fold RMSE curves are reconstructed from each engine's held-out
predictions and checked against the actual website selection procedure.
Negative controls ensure that changing the last response column, introducing
NaN, omitting a result, or duplicating a result fails verification.

The generated inputs, Scala outputs and report are under `target/parity/`.
Use `--report docs/parity-results.json` to refresh the saved report, including
source and dataset hashes. Latent-vector signs and rotations can differ
between decomposition methods, so the check compares fitted regression
results rather than requiring identical latent bases. It establishes
agreement on these cases within the stated tolerance, rather than a proof
for every possible input. VIP and the TypeScript-only OLS implementation are
outside the Scala–TypeScript parity check.

To update the copied website code in future, copy all three modules, refresh
their hashes in `provenance.json`, and rerun verification against the website
checkout. Changing the numerical code without updating the recorded snapshot
fails the verification run.
