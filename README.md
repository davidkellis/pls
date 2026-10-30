# pls
A Scala implementation of the PLS algorithms described in Dayal and MacGregor's "Improved PLS Algorithms" paper,
published in Journal of Chemometrics, Volume 11, Issue 1, pages 73–85, January 1997.

The website's TypeScript implementation is included in
[`typescript/`](typescript/README.md), with its original source hashes,
experiment generator, worker, locked dependencies and tests. Run the complete
Scala/TypeScript verification from this directory with:

```sh
python3 scripts/verify.py
```

The runner requires Node 22.23.4+, npm, Python 3 and Coursier; it provisions
JDK 8 and uses the existing Scala/Breeze versions. Optional arguments include
`--coursier /path/to/cs`, `--java-home /path/to/jdk8`, and
`--website ../conquerthelawn.com` to confirm byte identity with that checkout.
Results are written to `target/parity/report.json`; a saved verification run
is in [`docs/parity-results.json`](docs/parity-results.json). This compares
actual coefficients, intercepts, predictions and component selection across
640 shared datasets, with explicit tolerance and negative controls.

## Fitting and prediction

Use `DayalMcGregor.Algorithm1.standardizeAndTrain(X, Y, components)` for a
centered, sample-standard-deviation-scaled model with an intercept. Pass the
returned model to `standardizeAndPredict(model, newX)`; prediction reuses the
training means and scales. `Algorithm2` exposes the same API, using the
cross-product formulation. Both support one or multiple responses.

All columns are centered, including constants; zero-variance columns use unit
scale. Constant responses therefore predict their training mean. An explicit
column of ones is unnecessary with the standardized API. The lower-level
`train` and `predict` methods operate directly on the supplied matrices, without
adding an intercept or performing preprocessing.

The requested component count must be positive. Both algorithms stop before
dividing by exhausted covariance or score directions, and cap work at the
matrix dimensions. `model.R.cols` (also `W.cols`, `P.cols`, and `Q.cols`) reports
the **actual fitted component count**, which may be smaller than requested or
zero. No NaN padding is returned. A standardized zero-component model predicts
the training response mean.

VIP currently supports a single response and expects X and Y in the same
coordinates used to train the underlying model. For a standardized model,
apply its stored factors to X and Y first. Zero-component models return zero
VIP scores because no response variation was explained. The previously
unfinished `computeVIP2` entry point delegates to `computeVIP`.

## Verification

The tests cover both kernel algorithms, constant predictors/responses, mixed
constant and variable responses, redundant columns, rank exhaustion, input
validation, reuse of training statistics, and assertions that check every
response column. Historical tests that expected NaNs now check retained finite
models instead.

`src/test/resources/pls-reference.json` contains 54 reproducible synthetic cases
generated independently with NumPy 2.5.4 and scikit-learn 1.9.1
`PLSRegression` (NIPALS, `tol=1e-24`, `max_iter=10000`). For each algorithm, tests
compare all prediction and coefficient entries at combined absolute/relative
tolerance `2e-7 * max(1, abs(expected))`. These include tall/wide matrices,
different units, scaled/unscaled inputs and multiple responses. The fixture
can be regenerated independently with `scripts/generate-reference.py` in a
separate Python environment containing `scripts/reference-requirements.txt`:

```sh
python3 -m venv target/reference-env
target/reference-env/bin/pip install -r scripts/reference-requirements.txt
target/reference-env/bin/python scripts/generate-reference.py
```

Ordinary test runs use the saved fixture and do not require NumPy or
scikit-learn. The reference generator is copied from the companion website
and writes the shared Scala/TypeScript fixture in this repository.

The project retains Scala 2.11.7 and Breeze 0.11.2. Its original sbt 0.13.9
configuration includes a retired Bintray publishing plugin. To run the full
suite without resolving that plugin, install [Coursier](https://get-coursier.io/)
and run:

```sh
python3 scripts/test.py
```

This downloads the pinned dependencies and Temurin JDK 8 to Coursier's cache,
compiles into `target/manual-tests`, and runs all ScalaTest suites. Optional
`--coursier /path/to/cs` and `--java-home /path/to/jdk8` arguments reuse existing
tools. No system Java or Scala configuration is changed.

## Sample data sets

Some sample data sets are in ./data:

Given this ruby data set generating function:
  ```
  def write_artificial_data(coefficients, row_count)
    open("artificial#{coefficients.count}.csv", 'w') do |f|
      row_count.times.map do
        observation_vector = coefficients.count.times.map{ rand(100000).to_f }
        y = coefficients.zip(observation_vector).map{|pair| coeff, observation = *pair; coeff * observation }.reduce(:+)
        f.puts ([y] + observation_vector).join(",")
      end
    end
  end
  ```

1. artificial13.csv was produced with the following ruby snippet:
  ```
  coefficients = -5..7   # 13 coefficients between [-5, -4, ..., 6, 7]
  write_artificial_data(coefficients, 100)
  ```

  So the data conforms to the equation:

  <code>
  y = -5*a + -4*b + -3*c + -2*d + -1*e + 0*f + 1*g + 2*h + 3*i + 4*j + 5*k + 6*l + 7*m
  </code>

2. artificial100.csv was produced with the following ruby snippet:
  ```
  coefficients = -50...50   # 100 coefficients between [-50, -49, -48, ..., 48, 49]
  write_artificial_data(coefficients, 200)
  ```

  So the data conforms to the equation:

  <code>
  y = -50*x_1 + -49*x_2 + ... + 48*x_99 + 49*x_100
  </code>

3. artificial1000.csv was produced with the following ruby snippet:
  ```
  coefficients = -500...500   # 1,000 coefficients between [-500, 499]
  write_artificial_data(coefficients, 1000)
  ```

  So the data conforms to the equation:

  <code>
  y = -500*x_1 + -499*x_2 + ... + 499*x_1000
  </code>

4. artificial10000.csv was produced with the following ruby snippet:
  ```
  coefficients = -5000...5000   # 10,000 coefficients between [-5000, 4999]
  write_artificial_data(coefficients, 1000)
  ```

  So the data conforms to the equation:

  <code>
  y = -5000*x_1 + -4999*x_2 + ... + 4999*x_10000
  </code>

5. gld_gdx_uso.csv created with:
  ```
  gld=CSV.read("gld.csv", headers: true)
  gdx=CSV.read("gdx.csv", headers: true)
  uso=CSV.read("uso.csv", headers: true)
  File.open("gld_gdx_uso.csv", "w") do |f|
    gld.map.with_index do |row,i|
      f.puts "#{gld[i]['Close']},#{gld[i]['Volume']},#{gdx[i]['Close']},#{gdx[i]['Volume']},#{uso[i]['Close']},#{uso[i]['Volume']}"
    end
  end

  ```

### Timings

  On a MacBook Pro (Retina, 13-inch, Early 2015), 2.7 GHz Intel Core i5, 16 GB 1867 MHz DDR3

  ```
  Algorithm 1:
  K,      N,      A,      memory (GB),  runtime (seconds)
  100,    200,    10,     ?,            ?
  100,    200,    100,    ?,            ?
  100,    200,    1000,   ?,            ?
  100,    200,    10000,  ?,            ?
  1000,   1000,   10,     ?,            ?
  1000,   1000,   100,    ?,            ?
  1000,   1000,   1000,   ?,            ?
  1000,   1000,   10000,  ?,            ?
  10000,  1000,   10,     ?,            ?
  10000,  1000,   100,    ?,            14
  10000,  1000,   500,    ?,            ?
  10000,  1000,   1000,   ?,            52
  10000,  2000,   10,     ?,            ?
  10000,  2000,   100,    ?,            ?
  10000,  2000,   1000,   ?,            ?
  20000,  500,    10,     ?,            ?
  20000,  500,    100,    ?,            ?
  20000,  1000,   10,     ?,            9
  20000,  1000,   100,    ?,            17
  20000,  1000,   500,    ?,            46
  ```

  ```
  Algorithm 2:
  K,      N,      A,      memory (GB),  runtime (seconds)
  100,    200,    10,     0.5,          1
  100,    200,    100,    0.5,          1
  100,    200,    1000,   0.56,         2
  100,    200,    10000,  3.24,         49
  1000,   1000,   10,     0.55,         2
  1000,   1000,   100,    0.57,         2
  1000,   1000,   1000,   1.67,         7
  1000,   1000,   10000,  4.39,         235
  10000,  1000,   10,     2.41,         12
  10000,  1000,   100,    2.44,         52
  10000,  1000,   500,    5.34,         227
  10000,  1000,   1000,   6.26,         446
  10000,  2000,   10,     3.62,         23
  10000,  2000,   100,    3.72,         62
  10000,  2000,   1000,   5.30,         462
  20000,  500,    10,     4.53,         31
  20000,  500,    100,    4.87,         184
  ```
