import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fitPLS, predict } from '../src/regression.ts';
import { crossValidate, generateData, presets, random } from '../src/experiment.ts';

const root = fileURLToPath(new URL('../../', import.meta.url));
const tolerance = 2e-7;
const read = path => JSON.parse(readFileSync(path, 'utf8'));
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const save = (path, value) => writeFileSync(path, JSON.stringify(value, null, 2) + '\n');

function verifySources(website) {
  const provenance = read(resolve(root, 'typescript/provenance.json'));
  for (const [path, record] of Object.entries(provenance.files)) {
    assert.equal(hash(resolve(root, 'typescript', path)), record.sha256, `Copied source changed: ${path}`);
    if (website) assert.equal(hash(resolve(website, record.website_path)), record.sha256, `Website source differs: ${path}`);
  }
  if (website) assert.equal(hash(resolve(website, 'tests/fixtures/pls-reference.json')),
    hash(resolve(root, 'src/test/resources/pls-reference.json')), 'Independent fixtures differ');
  return { website_checkout_checked: Boolean(website), files: provenance.files };
}

function prepare(output, website) {
  const sources = verifySources(website);
  const reference = read(resolve(root, 'src/test/resources/pls-reference.json'));
  const cases = [], expectedCV = {};
  const add = (name, group, data) => cases.push({ name, group, ...data });
  for (const fixture of reference.cases) {
    const { name, x, y, evaluation, components, scale, plsPredictions, plsCoefficients } = fixture;
    add(`reference:${name}`, 'independent-reference', { x, y, evaluation, components, scale,
      reference: { predictions: plsPredictions, coefficients: plsCoefficients } });
    for (let k = 1; k <= Math.min(6, x.length - 1, x[0].length); k++) if (k !== components) {
      add(`path:${name}:k${k}`, 'reference-component-paths', { x, y, evaluation, components: k, scale });
    }
  }

  for (const [name, preset] of Object.entries(presets)) for (const seed of [42, 101]) {
    const options = { ...preset, seed };
    const { training, test } = generateData(options);
    const maximum = Math.min(12, options.predictors, options.observations - Math.ceil(options.observations / 5) - 1);
    for (let k = 1; k <= maximum; k++) add(`website:${name}:seed${seed}:k${k}`, 'website-scenarios', {
      ...training, evaluation: test.x, components: k, scale: true });
    if (seed !== 42) continue;
    const cv = crossValidate(training.x, training.y, maximum, seed);
    expectedCV[name] = { errors: cv.errors.slice(1), suggested: cv.suggested, available: cv.available };
    for (const [fold, held] of cv.folds.entries()) {
      const trainingIndices = training.x.map((_, i) => i).filter(i => !held.includes(i));
      for (let k = 1; k <= cv.available; k++) add(`cv:${name}:fold${fold}:k${k}`, 'website-validation-folds', {
        x: trainingIndices.map(i => training.x[i]), y: trainingIndices.map(i => training.y[i]),
        evaluation: held.map(i => training.x[i]), evaluationY: held.map(i => training.y[i]),
        components: k, scale: true, cv: { preset: name, fold } });
    }
  }

  for (let seed = 0; seed < 6; seed++) for (const [n, p, targets] of [[9, 4, 1], [8, 15, 3], [16, 6, 2]]) {
    const uniform = random(seed * 100 + n * 10 + targets);
    const coefficients = Array.from({ length: p }, () => Array.from({ length: targets }, () => 4 * uniform() - 2));
    const x = Array.from({ length: n }, () => Array.from({ length: p }, () => 2 * uniform() - 1));
    for (const row of x) row[1] = 0.95 * row[0] + 0.1 * row[1];
    const y = x.map(row => Array.from({ length: targets }, (_, k) =>
      3 + 2 * k + row.reduce((sum, v, j) => sum + v * coefficients[j][k], 0) + (uniform() - 0.5) * 0.1));
    const evaluation = Array.from({ length: 7 }, () => Array.from({ length: p }, () => 2 * uniform() - 1));
    for (const scale of [true, false]) for (const k of new Set([1, 2, Math.min(6, p, n - 1)])) {
      add(`seeded:${seed}:${n}x${p}x${targets}:scale${scale}:k${k}`, 'seeded-single-and-multiple-responses',
        { x, y, evaluation, components: k, scale });
    }
  }

  for (const constant of [9.5, -2, 0]) add(`edge:constant-target:${constant}`, 'edge-cases', {
    x: [[1, 0], [2, 1], [3, 0], [4, 1]], y: Array.from({ length: 4 }, () => [constant]),
    evaluation: [[10, 2], [-4, 1]], components: 2, scale: true });
  add('edge:constant-predictors', 'edge-cases', { x: [[7, 0], [7, 0], [7, 0]], y: [[2], [3], [4]],
    evaluation: [[7, 0]], components: 2, scale: true });
  for (const scale of [true, false]) {
    add(`edge:duplicates:scale${scale}`, 'edge-cases', { x: [[1, 1], [2, 2], [3, 3]], y: [[1], [2], [3]],
      evaluation: [[4, 4], [-1, -1]], components: 2, scale });
    add(`edge:zero-covariance:scale${scale}`, 'edge-cases', { x: [[-1], [0], [1]], y: [[1], [-2], [1]],
      evaluation: [[4], [-4]], components: 1, scale });
    add(`edge:two-observations:scale${scale}`, 'edge-cases', { x: [[1, 2, 0, 7], [3, 4, 1, 7]], y: [[2, -3], [6, 1]],
      evaluation: [[5, 6, 2, 7]], components: 1, scale });
  }
  const names = new Set(cases.map(c => c.name));
  assert.equal(names.size, cases.length, 'Dataset names must be unique');
  writeFileSync(output, JSON.stringify({ sources, reference: reference.reference, cases, expectedCV }) + '\n');
  console.log(`Prepared ${cases.length} shared datasets; TypeScript source hashes match the website copy${website ? ' and live checkout' : ''}.`);
}

function statistics() {
  return { fits: 0, numeric_comparisons: 0, maximum_absolute_difference: 0,
    maximum_scaled_difference: 0, largest_scaled_difference_at: null };
}

function close(actual, expected, stats, path) {
  if (Array.isArray(expected)) {
    assert(Array.isArray(actual), `${path}: expected an array`);
    assert.equal(actual.length, expected.length, `${path}: dimensions differ`);
    expected.forEach((value, i) => close(actual[i], value, stats, `${path}[${i}]`));
    return;
  }
  assert.equal(typeof expected, 'number', `${path}: expected numeric reference`);
  assert.equal(typeof actual, 'number', `${path}: expected numeric result`);
  assert(Number.isFinite(actual) && Number.isFinite(expected), `${path}: nonfinite result`);
  const difference = Math.abs(actual - expected), scaled = difference / Math.max(1, Math.abs(expected));
  stats.numeric_comparisons++;
  stats.maximum_absolute_difference = Math.max(stats.maximum_absolute_difference, difference);
  if (scaled > stats.maximum_scaled_difference) {
    stats.maximum_scaled_difference = scaled;
    stats.largest_scaled_difference_at = path;
  }
  assert(scaled <= tolerance, `${path}: ${actual} != ${expected} (scaled difference ${scaled})`);
}

function compareModel(actual, expected, stats, path) {
  assert.equal(actual.components, expected.components, `${path}: fitted component count differs`);
  for (const field of ['coefficients', 'intercept', 'predictions', 'trainingPredictions']) {
    close(actual[field], expected[field], stats, `${path}.${field}`);
  }
  stats.fits++;
}

function indexResults(records, cases) {
  assert.equal(records.length, cases.length * 2, 'Missing or extra Scala results');
  const indexed = new Map(records.map(record => [`${record.case}/${record.algorithm}`, record]));
  assert.equal(indexed.size, records.length, 'Duplicate Scala results');
  for (const dataset of cases) for (const algorithm of ['Algorithm1', 'Algorithm2']) {
    assert(indexed.has(`${dataset.name}/${algorithm}`), `Missing ${dataset.name}/${algorithm}`);
  }
  return indexed;
}

function compare(inputPath, scalaPath, reportPath) {
  const input = read(inputPath), scala = read(scalaPath);
  assert.deepEqual(verifySources().files, input.sources.files, 'Copied sources changed during the run');
  const indexed = indexResults(scala.results, input.cases);
  const pairs = { typescript_vs_scala_algorithm1: statistics(), typescript_vs_scala_algorithm2: statistics(),
    scala_algorithm1_vs_algorithm2: statistics() };
  const independent = Object.fromEntries(['TypeScript', 'Algorithm1', 'Algorithm2'].map(name => [name, statistics()]));
  const cvSums = {};
  const coverage = {};
  let multiResponse;
  for (const dataset of input.cases) {
    coverage[dataset.group] = (coverage[dataset.group] || 0) + 1;
    const fit = fitPLS(dataset.x, dataset.y, { components: dataset.components, scale: dataset.scale });
    const ts = { components: fit.components, coefficients: fit.coefficients, intercept: fit.intercept,
      predictions: predict(fit, dataset.evaluation), trainingPredictions: predict(fit, dataset.x) };
    const first = indexed.get(`${dataset.name}/Algorithm1`), second = indexed.get(`${dataset.name}/Algorithm2`);
    compareModel(ts, first, pairs.typescript_vs_scala_algorithm1, dataset.name);
    compareModel(ts, second, pairs.typescript_vs_scala_algorithm2, dataset.name);
    compareModel(first, second, pairs.scala_algorithm1_vs_algorithm2, dataset.name);
    if (!multiResponse && dataset.y[0].length > 1) multiResponse = ts;
    for (const [engine, model] of [['TypeScript', ts], ['Algorithm1', first], ['Algorithm2', second]]) {
      if (dataset.reference) {
        for (const field of ['predictions', 'coefficients']) close(model[field], dataset.reference[field], independent[engine], `${dataset.name}.${field}`);
        independent[engine].fits++;
      }
      if (dataset.cv) {
        const key = `${dataset.cv.preset}/${engine}/${dataset.components}`;
        const sums = cvSums[key] ||= { squared: 0, count: 0 };
        for (let i = 0; i < dataset.evaluationY.length; i++) for (let j = 0; j < dataset.evaluationY[i].length; j++) {
          sums.squared += (model.predictions[i][j] - dataset.evaluationY[i][j]) ** 2;
          sums.count++;
        }
      }
    }
  }
  const crossValidation = {};
  for (const [name, expected] of Object.entries(input.expectedCV)) {
    crossValidation[name] = { website_suggested: expected.suggested, available: expected.available };
    for (const engine of ['TypeScript', 'Algorithm1', 'Algorithm2']) {
      const errors = Array.from({ length: expected.available }, (_, k) => {
        const sums = cvSums[`${name}/${engine}/${k + 1}`];
        return Math.sqrt(sums.squared / sums.count);
      });
      const suggested = 1 + errors.reduce((best, error, k) => error < errors[best] ? k : best, 0);
      close(errors, expected.errors, statistics(), `${name}.${engine}.crossValidation`);
      assert.equal(suggested, expected.suggested, `${name}.${engine}: component selection differs`);
      crossValidation[name][engine] = { suggested, errors };
    }
  }

  // Negative controls prove that a mismatched final response, nonfinite value,
  // omitted case or duplicate export cannot silently produce a passing report.
  const bad = structuredClone(multiResponse);
  bad.predictions[0][bad.predictions[0].length - 1] += 1;
  assert.throws(() => compareModel(bad, multiResponse, statistics(), 'negative-control'));
  bad.predictions[0][0] = NaN;
  assert.throws(() => compareModel(bad, multiResponse, statistics(), 'negative-control'));
  assert.throws(() => indexResults(scala.results.slice(1), input.cases));
  const duplicate = scala.results.slice(); duplicate[0] = duplicate[1];
  assert.throws(() => indexResults(duplicate, input.cases));

  const scalaSources = ['Standardize.scala', 'pls.scala', 'Utils.scala'];
  const report = { status: 'passed', datasets: input.cases.length, coverage,
    tolerance: 'abs(actual-expected) <= 2e-7 * max(1, abs(expected))',
    scope: 'Centered PLS1/PLS2 with optional sample-SD scaling; original-unit coefficients, intercepts, training and held-out predictions, fitted component counts, and website five-fold selection. Latent-vector signs and unsupported API requests are not compared.',
    versions: { node: process.version, scala: scala.scala_version, java: scala.java_version,
      breeze: '0.11.2', matrix_library: 'ml-matrix 6.15.1' },
    source_identity: input.sources,
    source_sha256: Object.fromEntries(scalaSources.map(name => [`src/main/scala/com/github/davidkellis/pls/${name}`,
      hash(resolve(root, 'src/main/scala/com/github/davidkellis/pls', name))])),
    fixture_sha256: hash(resolve(root, 'src/test/resources/pls-reference.json')),
    dataset_sha256: createHash('sha256').update(JSON.stringify(input.cases)).digest('hex'),
    harness_sha256: Object.fromEntries(['scripts/verify.py', 'scripts/test.py', 'typescript/test/parity.mjs',
      'src/test/scala/com/github/davidkellis/pls/CrossLanguageExporter.scala'].map(path => [path, hash(resolve(root, path))])),
    pairs, independent_reference: { versions: input.reference, engines: independent },
    cross_validation: crossValidation,
    negative_controls: { changed_final_response_rejected: true, nonfinite_result_rejected: true,
      missing_result_rejected: true, duplicate_result_rejected: true } };
  save(reportPath, report);
  console.log(`PASS: ${report.datasets} datasets, ${report.datasets * 2} Scala fits, ${report.datasets} TypeScript fits.`);
  for (const [name, stats] of Object.entries(pairs)) console.log(`${name}: ${stats.numeric_comparisons} numeric comparisons; max absolute difference ${stats.maximum_absolute_difference}; max scaled difference ${stats.maximum_scaled_difference}`);
  console.log('All three website component selections agree; all four negative controls rejected bad results.');
  console.log(`Report: ${reportPath}`);
}

const [mode, ...args] = process.argv.slice(2);
if (mode === 'prepare' && args.length >= 1 && args.length <= 2) prepare(...args);
else if (mode === 'compare' && args.length === 3) compare(...args);
else throw new Error('Usage: parity.mjs prepare input.json [website] | compare input.json scala.json report.json');
