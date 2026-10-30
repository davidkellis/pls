/**
 * Kernel PLS regression: Dayal & MacGregor (1997), modified algorithm 1.
 * TypeScript adaptation of David Ellis's MIT-licensed Scala implementation.
 * This is linear kernel PLS, not a nonlinear kernel-method estimator.
 * See ops/pls-audit.md for provenance, numerical safeguards and validation.
 */
import { Matrix, SingularValueDecomposition } from 'ml-matrix';

export type Data = number[][];
export interface LinearModel {
  coefficients: Data; // predictors × responses, in original units
  intercept: number[];
  rank: number; // numerical rank of centered/scaled X
}
export interface PlsModel extends LinearModel {
  components: number;
  requestedComponents: number;
  path: LinearModel[]; // path[0] is the training-mean baseline
  scores: Data;
  weights: Data;
  loadings: Data;
  responseLoadings: Data;
  rotations: Data;
  xMean: number[];
  xScale: number[];
  yMean: number[];
  yScale: number[];
}

function shape(data: Data, label: string, minRows = 1): [number, number] {
  if (!Array.isArray(data) || data.length < minRows || !Array.isArray(data[0]) || data[0].length < 1) {
    throw new Error(`${label} needs at least ${minRows} rows and one column.`);
  }
  const width = data[0].length;
  if (data.some((row) => !Array.isArray(row) || row.length !== width || row.some((v) => !Number.isFinite(v)))) {
    throw new Error(`${label} must be rectangular and contain only finite numbers.`);
  }
  return [data.length, width];
}

const dot = (a: number[], b: number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const norm = (values: number[]) => values.reduce((sum, value) => Math.hypot(sum, value), 0);
const finite = (values: number[]) => {
  if (!values.every(Number.isFinite)) throw new Error('Numerical range is too large. Rescale the input values.');
};

function standardize(data: Data, scale: boolean) {
  const rows = data.length, columns = data[0].length;
  const means = Array<number>(columns).fill(0);
  for (let i = 0; i < rows; i++) for (let j = 0; j < columns; j++) means[j] += (data[i][j] - means[j]) / (i + 1);
  finite(means);
  const centered = data.map((row) => row.map((value, j) => value - means[j]));
  const scales = means.map((_, j) => scale ? norm(centered.map((row) => row[j])) / Math.sqrt(rows - 1) || 1 : 1);
  finite(scales);
  const values = centered.map((row) => row.map((value, j) => value / scales[j]));
  values.forEach(finite);
  return { means, scales, matrix: new Matrix(values) };
}

function prepare(x: Data, y: Data, scale: boolean) {
  const [rows, columns] = shape(x, 'X', 2);
  shape(y, 'Y', 2);
  if (y.length !== rows) throw new Error('X and Y must have the same number of observations.');
  const xs = standardize(x, scale), ys = standardize(y, scale);
  const svd = new SingularValueDecomposition(xs.matrix, { autoTranspose: true });
  const singular = svd.diagonal;
  const cutoff = Number.EPSILON * Math.max(rows, columns) * (singular[0] || 0);
  const rank = Math.min(rows - 1, singular.filter((value) => value > cutoff).length);
  finite(singular);
  return { xs, ys, svd, singular, cutoff, rank, rows, columns };
}

function originalUnits(beta: Matrix, prepared: ReturnType<typeof prepare>): LinearModel {
  const { xs, ys, rank } = prepared;
  const coefficients = beta.to2DArray().map((row, j) => row.map((value, k) => value * ys.scales[k] / xs.scales[j]));
  const intercept = ys.means.map((mean, k) => mean - coefficients.reduce((sum, row, j) => sum + row[k] * xs.means[j], 0));
  coefficients.forEach(finite);
  finite(intercept);
  return { coefficients, intercept, rank };
}

/** Centered, optionally autoscaled PLS1/PLS2. Training statistics are retained. */
export function fitPLS(x: Data, y: Data, options: { components: number; scale?: boolean }): PlsModel {
  const prepared = prepare(x, y, options.scale ?? true);
  const { rows, columns, rank, xs, ys } = prepared;
  const requested = options.components;
  if (!Number.isInteger(requested) || requested < 1 || requested > Math.min(rows - 1, columns)) {
    throw new Error('Components must be an integer from 1 to min(observations − 1, predictors).');
  }
  const responses = y[0].length;
  let covariance = xs.matrix.transpose().mmul(ys.matrix);
  const initialCovarianceNorm = norm(covariance.to1DArray());
  finite([initialCovarianceNorm]);
  const relativeTolerance = 64 * Number.EPSILON * Math.max(rows, columns, responses);
  const xNorm = norm(xs.matrix.to1DArray());
  const weights: number[][] = [], loadings: number[][] = [], responseLoadings: number[][] = [], rotations: number[][] = [], scores: number[][] = [];
  const beta = Matrix.zeros(columns, responses);
  const path = [originalUnits(beta, prepared)];

  for (let component = 0; component < Math.min(requested, rank); component++) {
    // Exhausted response covariance or rank ends the path without dividing by 0.
    const covarianceNorm = norm(covariance.to1DArray());
    if (covarianceNorm <= initialCovarianceNorm * relativeTolerance || covarianceNorm === 0) break;
    const direction = responses === 1 ? covariance.getColumn(0)
      : new SingularValueDecomposition(covariance, { autoTranspose: true }).leftSingularVectors.getColumn(0);
    const weightNorm = norm(direction);
    if (weightNorm === 0) break;
    const w = direction.map((value) => value / weightNorm);
    const r = w.slice();
    // Orthogonalize against PREVIOUS components, using the original w each time.
    for (let j = 0; j < component; j++) {
      const overlap = dot(loadings[j], w);
      for (let feature = 0; feature < columns; feature++) r[feature] -= overlap * rotations[j][feature];
    }
    const rNorm = norm(r);
    const t = xs.matrix.mmul(Matrix.columnVector(r)).getColumn(0);
    const tNorm = norm(t);
    if (rNorm <= relativeTolerance || tNorm <= relativeTolerance * xNorm * rNorm) break;
    const tt = dot(t, t);
    const p = xs.matrix.transpose().mmul(Matrix.columnVector(t)).getColumn(0).map((value) => value / tt);
    const q = covariance.transpose().mmul(Matrix.columnVector(r)).getColumn(0).map((value) => value / tt);
    finite([...p, ...q]);
    for (let feature = 0; feature < columns; feature++) for (let response = 0; response < responses; response++) {
      beta.set(feature, response, beta.get(feature, response) + r[feature] * q[response]);
    }
    covariance = Matrix.sub(covariance, Matrix.columnVector(p).mmul(Matrix.rowVector(q)).mul(tt));
    weights.push(w); loadings.push(p); responseLoadings.push(q); rotations.push(r); scores.push(t);
    path.push(originalUnits(beta, prepared));
  }
  const byRows = (vectors: number[][], count: number) => Array.from({ length: count }, (_, i) => vectors.map((vector) => vector[i]));
  return { ...path.at(-1)!, path, components: path.length - 1, requestedComponents: requested,
    scores: byRows(scores, rows), weights: byRows(weights, columns), loadings: byRows(loadings, columns),
    responseLoadings: byRows(responseLoadings, responses), rotations: byRows(rotations, columns),
    xMean: xs.means, xScale: xs.scales, yMean: ys.means, yScale: ys.scales };
}

/** SVD least squares; no normal-equation inverse and no ridge penalty. */
export function fitOLS(x: Data, y: Data, options: { scale?: boolean } = {}): LinearModel {
  const prepared = prepare(x, y, options.scale ?? true);
  const { svd, singular, cutoff, columns, ys } = prepared;
  const u = svd.leftSingularVectors, v = svd.rightSingularVectors;
  const beta = Matrix.zeros(columns, y[0].length);
  const projected = u.transpose().mmul(ys.matrix);
  // Explicit cutoff matches NumPy lstsq's rcond = eps * max(X.shape).
  for (let k = 0; k < singular.length; k++) if (singular[k] > cutoff) {
    for (let j = 0; j < columns; j++) for (let target = 0; target < y[0].length; target++) {
      beta.set(j, target, beta.get(j, target) + v.get(j, k) * projected.get(k, target) / singular[k]);
    }
  }
  return originalUnits(beta, prepared);
}

export function predict(model: LinearModel, x: Data): Data {
  const [, columns] = shape(x, 'Prediction X');
  if (columns !== model.coefficients.length) throw new Error('Prediction X has the wrong number of predictors.');
  return x.map((row) => {
    const values = model.intercept.map((intercept, k) => intercept + row.reduce((sum, value, j) => sum + value * model.coefficients[j][k], 0));
    finite(values);
    return values;
  });
}

export function rmse(actual: Data, predicted: Data): number {
  const [rows, columns] = shape(actual, 'Actual values');
  const [otherRows, otherColumns] = shape(predicted, 'Predictions');
  if (rows !== otherRows || columns !== otherColumns) throw new Error('Actual values and predictions must have the same dimensions.');
  const residuals = actual.flatMap((row, i) => row.map((value, j) => value - predicted[i][j]));
  return norm(residuals) / Math.sqrt(rows * columns);
}
