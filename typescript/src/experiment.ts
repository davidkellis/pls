import { fitOLS, fitPLS, predict, rmse } from './regression.ts';
import type { Data } from './regression.ts';

export interface ExperimentOptions { observations: number; predictors: number; correlation: number; noise: number; seed: number }
export const presets = {
  correlated: { observations: 40, predictors: 36, correlation: 0.98, noise: 0.6, seed: 42, components: 2 },
  wide: { observations: 24, predictors: 64, correlation: 0.95, noise: 0.6, seed: 42, components: 2 },
  abundant: { observations: 240, predictors: 8, correlation: 0.25, noise: 0.15, seed: 42, components: 1 },
} satisfies Record<string, ExperimentOptions & { components: number }>;

export function random(seed: number) {
  let state = (seed >>> 0) || 0x9e3779b9;
  return () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return ((state >>> 0) + 0.5) / 4294967296; };
}
function gaussian(seed: number) {
  const uniform = random(seed);
  let spare: number | undefined;
  return () => {
    if (spare !== undefined) { const value = spare; spare = undefined; return value; }
    const radius = Math.sqrt(-2 * Math.log(uniform())), angle = 2 * Math.PI * uniform();
    spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };
}

function validate(options: ExperimentOptions) {
  const { observations: n, predictors: p, correlation, noise, seed } = options;
  if (!Number.isInteger(n) || n < 20 || n > 300 || !Number.isInteger(p) || p < 4 || p > 80
    || !Number.isFinite(correlation) || correlation < 0 || correlation > 1
    || !Number.isFinite(noise) || noise < 0 || noise > 2 || !Number.isInteger(seed) || seed < 0 || seed > 1_000_000) {
    throw new Error('Use 20–300 observations, 4–80 predictors, correlation 0–1, noise 0–2, and seed 0–1,000,000.');
  }
}

export function generateData(options: ExperimentOptions) {
  validate(options);
  const { observations, predictors: p, correlation: rho, noise, seed } = options;
  const counts = [Math.ceil(p / 2), Math.floor(p / 2)];
  // Keep the signal variance constant when correlation and p change. These
  // known coefficients define y as a LINEAR function of the observed X.
  const truth = Array.from({ length: p }, (_, j) => {
    const group = j % 2, count = counts[group];
    return [2, -1.2][group] / (count * Math.sqrt(rho + (1 - rho) / count));
  });
  function sample(n: number, sampleSeed: number) {
    const normal = gaussian(sampleSeed);
    const x: Data = [], y: Data = [];
    for (let i = 0; i < n; i++) {
      const latent = [normal(), normal()];
      const row = truth.map((_, j) => Math.sqrt(rho) * latent[j % 2] + Math.sqrt(1 - rho) * normal());
      const error = normal(); // Always consume this draw, even at noise = 0.
      x.push(row); y.push([5 + row.reduce((sum, value, j) => sum + value * truth[j], 0) + noise * error]);
    }
    return { x, y };
  }
  return { training: sample(observations, seed ^ 0x13579bdf), test: sample(200, seed ^ 0x2468ace0), truth, trueIntercept: 5 };
}

/** Train-only, out-of-fold RMSE. Each fold learns its OWN centering/scaling. */
export function crossValidate(x: Data, y: Data, maxComponents: number, seed: number) {
  const indices = Array.from({ length: x.length }, (_, i) => i), uniform = random(seed ^ 0x6c8e9cf5);
  for (let i = indices.length - 1; i > 0; i--) { const j = Math.floor(uniform() * (i + 1)); [indices[i], indices[j]] = [indices[j], indices[i]]; }
  const folds = Array.from({ length: 5 }, (_, fold) => indices.filter((_, i) => i % 5 === fold));
  const squaredErrors = Array<number>(maxComponents + 1).fill(0);
  let available = maxComponents, olsSquaredError = 0;
  for (const heldOut of folds) {
    const heldSet = new Set(heldOut), train = indices.filter((i) => !heldSet.has(i));
    const tx = train.map((i) => x[i]), ty = train.map((i) => y[i]);
    const vx = heldOut.map((i) => x[i]), vy = heldOut.map((i) => y[i]);
    const fit = fitPLS(tx, ty, { components: Math.min(maxComponents, tx.length - 1, tx[0].length) });
    available = Math.min(available, fit.components);
    for (let k = 0; k <= fit.components; k++) squaredErrors[k] += rmse(vy, predict(fit.path[k], vx)) ** 2 * vy.length;
    olsSquaredError += rmse(vy, predict(fitOLS(tx, ty), vx)) ** 2 * vy.length;
  }
  const errors = squaredErrors.slice(0, available + 1).map((sum) => Math.sqrt(sum / x.length));
  let suggested = available ? 1 : 0;
  for (let k = suggested + 1; k <= available; k++) if (errors[k] < errors[suggested]) suggested = k;
  return { errors, suggested, available, olsError: Math.sqrt(olsSquaredError / x.length), folds };
}

export function runExperiment(options: ExperimentOptions) {
  const { training, test, truth, trueIntercept } = generateData(options);
  const maxComponents = Math.min(12, options.predictors, options.observations - Math.ceil(options.observations / 5) - 1);
  // Selection is completely separate from test-set evaluation.
  const cv = crossValidate(training.x, training.y, maxComponents, options.seed);
  const pls = fitPLS(training.x, training.y, { components: maxComponents });
  const ols = fitOLS(training.x, training.y);
  const olsPredictions = predict(ols, test.x).map((row) => row[0]);
  const path = pls.path.map((model, k) => ({
    components: k, trainingError: rmse(training.y, predict(model, training.x)),
    testPredictions: predict(model, test.x).map((row) => row[0]),
    testError: rmse(test.y, predict(model, test.x)), coefficients: model.coefficients.map((row) => row[0]), intercept: model.intercept[0],
  }));
  return { options, rank: ols.rank, maxComponents: Math.min(pls.components, cv.available), cv,
    path, ols: { trainingError: rmse(training.y, predict(ols, training.x)), testError: rmse(test.y, olsPredictions.map((value) => [value])),
      testPredictions: olsPredictions, coefficients: ols.coefficients.map((row) => row[0]), intercept: ols.intercept[0] },
    actual: test.y.map((row) => row[0]), training, test, truth, trueIntercept };
}
export type ExperimentResult = ReturnType<typeof runExperiment>;
