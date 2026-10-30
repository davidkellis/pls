import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fitPLS, fitOLS, predict, rmse } from '../src/regression.ts';
import { crossValidate, generateData, runExperiment, presets } from '../src/experiment.ts';

const reference = JSON.parse(readFileSync(new URL('../../src/test/resources/pls-reference.json', import.meta.url), 'utf8'));
let comparisons = 0, largestPredictionError = 0;
function close(actual, expected, message = '') {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, message);
    expected.forEach((value, i) => close(actual[i], value, `${message}[${i}]`));
  } else {
    comparisons++;
    assert(Number.isFinite(actual), message);
    assert(Math.abs(actual - expected) <= 2e-7 * Math.max(1, Math.abs(expected)), `${message}: ${actual} != ${expected}`);
  }
}

for (const fixture of reference.cases) test(`independent reference: ${fixture.name}`, () => {
  const { x, y, evaluation, components, scale } = fixture;
  const snapshot = JSON.stringify([x, y, evaluation]);
  const pls = fitPLS(x, y, { components, scale }), ols = fitOLS(x, y, { scale });
  const predictions = predict(pls, evaluation);
  close(predictions, fixture.plsPredictions, 'PLS prediction');
  close(pls.coefficients, fixture.plsCoefficients, 'PLS coefficient');
  close(predict(ols, evaluation), fixture.olsPredictions, 'OLS prediction');
  close(ols.coefficients, fixture.olsCoefficients, 'OLS coefficient');
  assert.equal(ols.rank, fixture.olsRank);
  assert.equal(JSON.stringify([x, y, evaluation]), snapshot, 'Do not mutate caller data');
  for (let i = 0; i < predictions.length; i++) for (let j = 0; j < y[0].length; j++) largestPredictionError = Math.max(largestPredictionError, Math.abs(predictions[i][j] - fixture.plsPredictions[i][j]));
  for (let a = 0; a < pls.components; a++) for (let b = 0; b < a; b++) {
    const inner = pls.scores.reduce((sum, row) => sum + row[a] * row[b], 0);
    const norms = Math.hypot(...pls.scores.map((row) => row[a])) * Math.hypot(...pls.scores.map((row) => row[b]));
    assert(Math.abs(inner) <= 1e-8 * Math.max(1, norms), 'PLS scores should be orthogonal');
  }
});

test('full-rank PLS converges to OLS, including the intercept and final response column', () => {
  const data = reference.cases.find((fixture) => fixture.name === 'multi-3-scale-True');
  const pls = fitPLS(data.x, data.y, { components: data.x[0].length });
  close(predict(pls, data.evaluation), predict(fitOLS(data.x, data.y), data.evaluation));
});

test('rank exhaustion and constant data produce finite predictions', () => {
  const duplicates = fitPLS([[1, 1], [2, 2], [3, 3]], [[1], [2], [3]], { components: 2 });
  assert.equal(duplicates.components, 1);
  close(predict(duplicates, [[4, 4]]), [[4]]);
  const constant = fitPLS([[7, 0], [7, 0], [7, 0]], [[2], [3], [4]], { components: 2 });
  assert.equal(constant.components, 0);
  close(predict(constant, [[7, 0]]), [[3]]);
});

test('invalid dimensions, nonfinite values and component requests are rejected', () => {
  const x = [[1, 2], [2, 4], [3, 6]], y = [[1], [2], [3]];
  for (const components of [0, -1, 1.5, 3]) assert.throws(() => fitPLS(x, y, { components }));
  assert.throws(() => fitPLS([[1], [NaN]], [[1], [2]], { components: 1 }));
  assert.throws(() => fitPLS([[1], [Infinity]], [[1], [2]], { components: 1 }));
  assert.throws(() => fitPLS([[1, 2], [3]], [[1], [2]], { components: 1 }));
  assert.throws(() => fitPLS(x, y.slice(1), { components: 1 }));
  assert.throws(() => predict(fitOLS(x, y), [[1]]));
  assert.throws(() => rmse([[1]], [[1, 2]]));
});

test('five-fold CV agrees with independent out-of-fold reconstruction using only training data', () => {
  const { training } = generateData(presets.correlated);
  const cv = crossValidate(training.x, training.y, 5, 42);
  const errors = [];
  for (let k = 1; k <= cv.available; k++) {
    let squared = 0;
    for (const held of cv.folds) {
      const trainingIndices = training.x.map((_, i) => i).filter((i) => !held.includes(i));
      const fit = fitPLS(trainingIndices.map((i) => training.x[i]), trainingIndices.map((i) => training.y[i]), { components: k });
      squared += rmse(held.map((i) => training.y[i]), predict(fit, held.map((i) => training.x[i]))) ** 2 * held.length;
    }
    errors.push(Math.sqrt(squared / training.x.length));
  }
  close(cv.errors.slice(1), errors);
  const changedTest = generateData(presets.correlated);
  changedTest.test.y = changedTest.test.y.map(() => [1e12]);
  close(crossValidate(changedTest.training.x, changedTest.training.y, 5, 42).errors, cv.errors);
});

test('presets reproduce both PLS advantages and a valid OLS counterexample', () => {
  const correlated = runExperiment(presets.correlated), wide = runExperiment(presets.wide), abundant = runExperiment(presets.abundant);
  assert(correlated.path[2].testError < correlated.ols.testError);
  assert(wide.path[2].testError < wide.ols.testError);
  assert(abundant.path[1].testError > abundant.ols.testError);
  close(abundant.path[8].testPredictions, abundant.ols.testPredictions);
  const duplicateData = runExperiment({ ...presets.correlated, correlation: 1 });
  assert.equal(duplicateData.rank, 2);
  assert.equal(duplicateData.maxComponents, 2);
  const noisy = generateData(presets.correlated), noiseless = generateData({ ...presets.correlated, noise: 0 });
  assert.deepEqual(noisy.training.x, noiseless.training.x, 'Noise slider should not regenerate X');
  const exact = fitOLS(noiseless.training.x, noiseless.training.y);
  close(exact.coefficients.map((row) => row[0]), noiseless.truth);
  close(exact.intercept, [5]);
});

test('reference summary', () => console.log(JSON.stringify({ cases: reference.cases.length, numericalComparisons: comparisons, largestPredictionError, reference: reference.reference })));
