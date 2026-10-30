package com.github.davidkellis.pls

import breeze.linalg._
import breeze.linalg.eigSym.EigSym
import breeze.numerics._

case class StandardizedModel[ModelT](
  model: ModelT,
  standardizationFactorsX: Standardize.StandardizationFactors,
  standardizationFactorsY: Standardize.StandardizationFactors
)

trait PlsModel[ModelT] {
  // X - predictor variables matrix (N x K)
  // Y - response variables matrix (N x M)
  // A - is the number of latent variables (a.k.a. components) to use
  def train(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): ModelT

  // trains a model on some training data;
  // also returns the Variable Importance in the Projection (VIP) vector indicating the relative importance of each X variable
  def trainAndComputeVIP(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): (ModelT, DenseVector[Double])
  def predict(model: ModelT, X: DenseMatrix[Double]): DenseMatrix[Double]

  def standardizeAndTrain(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): StandardizedModel[ModelT]
  def standardizeAndPredict(standardizedModel: StandardizedModel[ModelT], X: DenseMatrix[Double]): DenseMatrix[Double]
}

object DayalMcGregor {
  // Beta - PLS regression coefficients matrix (K x M)
  // W - PLS weights matrix for X (K x A)
  // P - PLS loadings matrix for X (K x A)
  // Q - PLS loadings matrix for Y (M x A)
  // R - PLS weights matrix to compute scores T directly from original X (K x A)
  case class Model(
    Beta: DenseMatrix[Double],
    W: DenseMatrix[Double],
    P: DenseMatrix[Double],
    Q: DenseMatrix[Double],
    R: DenseMatrix[Double]
  )

  private def finite(value: Double): Boolean = !value.isNaN && !value.isInfinity

  private def matrixNorm(matrix: DenseMatrix[Double]): Double =
    matrix.valuesIterator.foldLeft(0.0)((acc, value) => math.hypot(acc, value))

  private def validateTraining(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): Unit = {
    require(X.rows > 0 && X.cols > 0 && Y.cols > 0, "Training matrices must be nonempty")
    require(X.rows == Y.rows, "X and Y must have the same number of observations")
    require(A > 0, "The requested component count must be positive")
    require(X.valuesIterator.forall(finite) && Y.valuesIterator.forall(finite), "Training values must be finite")
  }

  // Store only fitted components. An exhausted model may contain zero of them;
  // its zero prediction becomes the training response mean after unscaling.
  private def finishModel(W: DenseMatrix[Double], P: DenseMatrix[Double], Q: DenseMatrix[Double], R: DenseMatrix[Double], used: Int): Model = {
    val w = W(::, 0 until used).copy
    val p = P(::, 0 until used).copy
    val q = Q(::, 0 until used).copy
    val r = R(::, 0 until used).copy
    val beta = if (used == 0) DenseMatrix.zeros[Double](R.rows, Q.rows) else r * q.t
    require(beta.valuesIterator.forall(finite), "Numerical range exceeded; rescale the training data")
    Model(beta, w, p, q, r)
  }

  object VIP {
    // X - predictor variables matrix (N x K); K - number of X-variables
    // Y - response variables matrix (N x M); M - number of Y-variables; *** This method assumes M = 1 ***
    // model.Beta - PLS regression coefficients matrix (K x M)
    // model.W - PLS weights matrix for X (K x A)
    // model.P - PLS loadings matrix for X (K x A)
    // model.Q - PLS loadings matrix for Y (M x A)
    // model.R - PLS weights matrix to compute scores T directly from original X (K x A)
    // References:
    // 1. Interpretation of variable importance in Partial Least Squares with Significance Multivariate Correlation (SMC)
    // 2. http://math.arizona.edu/~hzhang/waeso/vsTutorial.pdf
    def computeVIP(model: Model, X: DenseMatrix[Double], Y: DenseMatrix[Double]): DenseVector[Double] = {
      require(Y.cols == 1, "computeVIP currently supports a single response")
      if (model.R.cols == 0) return DenseVector.zeros[Double](X.cols)
      val K = model.R.rows                      // K is the number of predictor variables; R is a (K x A) matrix
      val A = model.R.cols                      // A is the number of components (latent variables); R is a (K x A) matrix
      val T = X * model.R                       // (N x K) * (K x A) = (N x A)
      val N = X.rows                            // N is the number of rows in T (as well as X)
      val W = model.W                           // W is the PLS weights matrix for X (K x A)
      val y: DenseVector[Double] = Y(::, 0)     // y is the column vector representation of Y - a (N x 1)
      val vip = DenseVector.zeros[Double](K)    // (K x 1)
      val p: Double = K.toDouble

      var t_a = DenseVector.zeros[Double](N)    // (N x 1) column vector of T
      var c_a: Double = 0.0
      var v_a: Double = 0.0
      var tTt: Double = 0.0
      var numerator: Double = 0.0
      var denominator: Double = 0.0

      for (k <- 0 to (K - 1)) {
        numerator = 0.0
        denominator = 0.0
        for (a <- 0 to (A - 1)) {
          t_a = T(::, a)              // (N x 1)
          tTt = t_a.t * t_a           // (1 x N) * (N x 1) === (1 x 1) === scalar
          c_a = (t_a.t * y) / tTt     // (1 x N) * (N x 1) / scalar === (1 x 1) / scalar === scalar / scalar === scalar
          v_a = pow(c_a, 2) * tTt
          numerator += v_a * pow(W(k, a), 2)
          denominator += v_a
        }
        vip(k to k) := sqrt(p * numerator / denominator)
      }

      vip
    }

    // Retained for source compatibility; use the verified single-response
    // calculation instead of the unfinished zero/zero implementation.
    def computeVIP2(model: Model, X: DenseMatrix[Double], Y: DenseMatrix[Double]): DenseVector[Double] =
      computeVIP(model, X, Y)

  }

  // Direct-X kernel formulation. Stops at exhausted response covariance or scores.
  object Algorithm1 extends PlsModel[Model]{
    // X - predictor variables matrix (N x K)
    // Y - response variables matrix (N x M)
    // B_PLS/Beta - PLS regression coefficients matrix (K x M)
    // W - PLS weights matrix for X (K x A)
    // P - PLS loadings matrix for X (K x A)
    // Q - PLS loadings matrix for Y (M x A)
    // R - PLS weights matrix to compute scores T directly from original X (K x A)
    // T - PLS scores matrix of X (N x A)
    // w_a - a column vector of W
    // p_a - a column vector of P
    // q_a - a column vector of Q
    // r_a - a column vector of R
    // t_a - a column vector of T
    // K - number of X-variables
    // M - number of Y-variables
    // N - number of objects
    // A - number of components in PLS model
    // a - integer counter for latent variable dimension.

    // converts a (1 x 1) DenseMatrix[T] or DenseVector[T] to a T
    def scalar[T](x: DenseMatrix[T]): T = x(0, 0)
    def scalar[T](x: DenseVector[T]): T = x(0)

    /**
     * Implements "Modified kernel algorithm #1" as described in Dayal and MacGregor's "Improved PLS Algorithms" paper,
     * published in Journal of Chemometrics, Volume 11, Issue 1, pages 73–85, January 1997.
     *
     * X - predictor variables matrix (N x K); K - number of X-variables
     * Y - response variables matrix (N x M); M - number of Y-variables
     * A - number of components in PLS model
     */
    def train(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): Model = {
      validateTraining(X, Y, A)
      val limit = math.min(A, math.min(X.rows, X.cols))
      val K = X.cols // number of columns in X - number of predictor variables
      val M = Y.cols // number of columns in Y - number of response variables
      val N = X.rows // number of rows in X === number of rows in Y - number of observations of predictor and response variables
      val W: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, limit)        // (K x A)
      val P: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, limit)        // (K x A)
      val Q: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, limit)        // (M x A)
      val R: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, limit)        // (K x A)
      var YtXXtY: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, M)   // (M x M) matrix
      var w_a: DenseVector[Double] = DenseVector.zeros[Double](K)         // (K x 1) - column vector of W
      var p_a: DenseVector[Double] = DenseVector.zeros[Double](K)         // (K x 1) - column vector of P
      var q_a: DenseVector[Double] = DenseVector.zeros[Double](M)         // (M x 1) - column vector of Q
      var r_a: DenseVector[Double] = DenseVector.zeros[Double](K)         // (K x 1) - column vector of R
      var t_a: DenseVector[Double] = DenseVector.zeros[Double](N)         // (N x 1) - column vector of T [which is N x A]
      var tt: Double = 0.0
      var indexOfLargestEigenvalue: Int = 0


      var XY: DenseMatrix[Double] = X.t * Y                               // compute the covariance matrices; (K x M) matrix

      val initialCovarianceNorm = matrixNorm(XY)
      val xNorm = matrixNorm(X)
      require(finite(initialCovarianceNorm) && finite(xNorm), "Numerical range exceeded; rescale the training data")
      val tolerance = 64 * math.ulp(1.0) * math.max(N, math.max(K, M))
      var a = 0
      var exhausted = false
      while (a < limit && !exhausted) {
        val covarianceNorm = matrixNorm(XY)
        require(finite(covarianceNorm), "Numerical range exceeded; rescale the training data")
        if (covarianceNorm <= initialCovarianceNorm * tolerance || covarianceNorm == 0.0) {
          exhausted = true
        } else {
          if (M == 1) {
            // if there is a single response variable, compute the X-weights as:
            w_a = XY.toDenseVector                                          // in this case, XY (K x M) is a column vector (K x 1) since M == 1
          } else {
            // otherwise there are multiple response variables, so compute the X-weights as:
            // The source code of pls/R/kernelpls.fit.R in R's "pls" package (see https://cran.r-project.org/web/packages/pls/index.html
            // or http://mevik.net/work/software/pls.html) states that YtXXtY is a symmetric matrix.
            // The fact that YtXXtY is **wonderful** because all eigenvalues of a real-valued symmetric matrix are real values.
            // Otherwise, we'd have to potentially handle Complex-valued eigenvalues.
            YtXXtY = XY.t * XY                                              // XY.t * XY is an (M x M) matrix
            val EigSym(eigenvalues, eigenvectors) = eigSym(YtXXtY)          // eigenvalues is a DenseVector[Double] and eigenvectors is a DenseMatrix[Double]
            indexOfLargestEigenvalue = argmax(eigenvalues)                  // find index of largest eigenvalue
            q_a = eigenvectors(::, indexOfLargestEigenvalue)                // find the eigenvector corresponding to the largest eigenvalue; eigenvector is (M x 1)
            w_a = XY * q_a                                                  // compute X-weights; w_a is (K x 1); expression yields a (K x M) * (M x 1) === (K x 1)
          }

          val weightNorm = norm(w_a)
          require(finite(weightNorm) && weightNorm > 0.0, "Numerical range exceeded; rescale the training data")
          w_a = w_a / weightNorm                                     // normalize w_a to unity - the denominator is a scalar
          r_a = w_a                                                         // loop to compute r_a
          for (j <- 0 to (a - 1)) {
            r_a = r_a - ( (P(::, j).t * w_a) * R(::, j) )                   // (K x 1) - ( (1 x K) * (K x 1) ) * (K x 1) === (K x 1) - scalar * (K x 1)
          }
          t_a = X * r_a                                                     // compute score vector
          tt = t_a.t * t_a                                                  // compute t't - (1 x 1) that is auto-converted to a scalar
          val rotationNorm = norm(r_a)
          require(finite(tt) && finite(rotationNorm), "Numerical range exceeded; rescale the training data")
          if (rotationNorm <= tolerance || tt <= math.pow(tolerance * xNorm * rotationNorm, 2)) {
            exhausted = true
          } else {
            p_a = (X.t * t_a) / tt                                            // X-loadings - ((K x 1)' * (K x K))' / tt === (K x 1) / tt
            q_a = (r_a.t * XY).t / tt                                         // Y-loadings - ((K x 1)' * (K x M))' / tt === (M x 1) / tt
            require(p_a.valuesIterator.forall(finite) && q_a.valuesIterator.forall(finite),
              "Numerical range exceeded; rescale the training data")
            XY = XY - ((p_a * q_a.t) * tt)                                    // XtY deflation

            // update loadings and weights
            W(::, a) := w_a
            P(::, a) := p_a
            Q(::, a) := q_a
            R(::, a) := r_a
            a += 1
          }
        }
      }
      finishModel(W, P, Q, R, a)
    }


    def trainAndComputeVIP(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): (Model, DenseVector[Double]) = {
      val model = train(X, Y, A)
      val vipVector = VIP.computeVIP(model, X, Y)
      (model, vipVector)
    }

    // Y = X * B + e
    // predict just ignores the error term, since it only consists of the residuals
    def predict(model: Model, X: DenseMatrix[Double]): DenseMatrix[Double] = {
      require(X.cols == model.Beta.rows, "Prediction X has the wrong number of predictors")
      require(X.valuesIterator.forall(finite), "Prediction values must be finite")
      val prediction = X * model.Beta
      require(prediction.valuesIterator.forall(finite), "Numerical range exceeded; rescale the prediction data")
      prediction
    }

    def standardizeAndTrain(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): StandardizedModel[Model] = {
      validateTraining(X, Y, A)
      require(X.rows >= 2, "Standardization needs at least two training observations")
      val (standardizedX, factorsX) = Standardize.centerAndScaleColumnsReturningFactors(X)
      val (standardizedY, factorsY) = Standardize.centerAndScaleColumnsReturningFactors(Y)
      val model = train(standardizedX, standardizedY, A)
      StandardizedModel(model, factorsX, factorsY)
    }

    def standardizeAndPredict(standardizedModel: StandardizedModel[Model], X: DenseMatrix[Double]): DenseMatrix[Double] = {
      val factorsX = standardizedModel.standardizationFactorsX
      val factorsY = standardizedModel.standardizationFactorsY
      val standardizedX = Standardize.centerAndScaleColumnsWithFactors(X, factorsX.meanOfColumns, factorsX.stdDevOfColumns)
      val standardizedEstimateY = predict(standardizedModel.model, standardizedX)
      Standardize.denormalizeCenteredAndScaledColumns(standardizedEstimateY, factorsY.meanOfColumns, factorsY.stdDevOfColumns)
    }

  }

  // Cross-product kernel formulation, with the same stopping rules as Algorithm1.
  object Algorithm2 extends PlsModel[Model]{
    // X - predictor variables matrix (N x K)
    // Y - response variables matrix (N x M)
    // B_PLS/Beta - PLS regression coefficients matrix (K x M)
    // W - PLS weights matrix for X (K x A)
    // P - PLS loadings matrix for X (K x A)
    // Q - PLS loadings matrix for Y (M x A)
    // R - PLS weights matrix to compute scores T directly from original X (K x A)
    // T - PLS scores matrix of X (N x A)
    // w_a - a column vector of W
    // p_a - a column vector of P
    // q_a - a column vector of Q
    // r_a - a column vector of R
    // t_a - a column vector of T
    // K - number of X-variables
    // M - number of Y-variables
    // N - number of objects
    // A - number of components in PLS model
    // a - integer counter for latent variable dimension.

    // converts a (1 x 1) DenseMatrix[T] or DenseVector[T] to a T
    def scalar[T](x: DenseMatrix[T]): T = x(0, 0)
    def scalar[T](x: DenseVector[T]): T = x(0)

    /**
     * Implements "Modified kernel algorithm #2" as described in Dayal and MacGregor's "Improved PLS Algorithms" paper,
     * published in Journal of Chemometrics, Volume 11, Issue 1, pages 73–85, January 1997.
     *
     * X - predictor variables matrix (N x K); K - number of X-variables
     * Y - response variables matrix (N x M); M - number of Y-variables
     * A - number of components in PLS model
     */
    def train(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): Model = {
      validateTraining(X, Y, A)
      val limit = math.min(A, math.min(X.rows, X.cols))
      val K = X.cols // number of columns in X - number of predictor variables
      val M = Y.cols // number of columns in Y - number of response variables
      val N = X.rows // number of rows in X === number of rows in Y - number of observations of predictor and response variables
      val W: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, limit)        // (K x A)
      val P: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, limit)        // (K x A)
      val Q: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, limit)        // (M x A)
      val R: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, limit)        // (K x A)
      var YtXXtY: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, M)   // (M x M) matrix
      var w_a: DenseVector[Double] = DenseVector.zeros[Double](K)         // (K x 1) - column vector of W
      var p_a: DenseVector[Double] = DenseVector.zeros[Double](K)         // (K x 1) - column vector of P
      var q_a: DenseVector[Double] = DenseVector.zeros[Double](M)         // (M x 1) - column vector of Q
      var r_a: DenseVector[Double] = DenseVector.zeros[Double](K)         // (K x 1) - column vector of R
      var tt: Double = 0.0
      var indexOfLargestEigenvalue: Int = 0


      var XY: DenseMatrix[Double] = X.t * Y                               // compute the covariance matrices; (K x M) matrix
      val XX: DenseMatrix[Double] = X.t * X                               // (K x K) matrix


      val initialCovarianceNorm = matrixNorm(XY)
      val xNorm = matrixNorm(X)
      require(finite(initialCovarianceNorm) && finite(xNorm), "Numerical range exceeded; rescale the training data")
      val tolerance = 64 * math.ulp(1.0) * math.max(N, math.max(K, M))
      var a = 0
      var exhausted = false
      while (a < limit && !exhausted) {
        val covarianceNorm = matrixNorm(XY)
        require(finite(covarianceNorm), "Numerical range exceeded; rescale the training data")
        if (covarianceNorm <= initialCovarianceNorm * tolerance || covarianceNorm == 0.0) {
          exhausted = true
        } else {
          if (M == 1) {
            // if there is a single response variable, compute the X-weights as:
            w_a = XY.toDenseVector                                          // in this case, XY (K x M) is a column vector (K x 1) since M == 1
          } else {
            // otherwise there are multiple response variables, so compute the X-weights as:
            // The source code of pls/R/kernelpls.fit.R in R's "pls" package (see https://cran.r-project.org/web/packages/pls/index.html
            // or http://mevik.net/work/software/pls.html) states that YtXXtY is a symmetric matrix.
            // The fact that YtXXtY is **wonderful** because all eigenvalues of a real-valued symmetric matrix are real values.
            // Otherwise, we'd have to potentially handle Complex-valued eigenvalues.
            YtXXtY = XY.t * XY                                              // XY.t * XY is an (M x M) matrix
            val EigSym(eigenvalues, eigenvectors) = eigSym(YtXXtY)          // eigenvalues is a DenseVector[Double] and eigenvectors is a DenseMatrix[Double]
            indexOfLargestEigenvalue = argmax(eigenvalues)                  // find index of largest eigenvalue
            q_a = eigenvectors(::, indexOfLargestEigenvalue)                // find the eigenvector corresponding to the largest eigenvalue; eigenvector is (M x 1)
            w_a = XY * q_a                                                  // compute X-weights; w_a is (K x 1); expression yields a (K x M) * (M x 1) === (K x 1)
          }


          val weightNorm = norm(w_a)
          require(finite(weightNorm) && weightNorm > 0.0, "Numerical range exceeded; rescale the training data")
          w_a = w_a / weightNorm                                     // normalize w_a to unity - the denominator is a scalar
          r_a = w_a                                                         // loop to compute r_a
          for (j <- 0 to (a - 1)) {
            r_a = r_a - ( (P(::, j).t * w_a) * R(::, j) )                   // (K x 1) - ( (1 x K) * (K x 1) ) * (K x 1) === (K x 1) - scalar * (K x 1)
          }
          tt = r_a.t * XX * r_a                                             // compute t't - (1 x 1) that is auto-converted to a scalar
          val rotationNorm = norm(r_a)
          require(finite(tt) && finite(rotationNorm), "Numerical range exceeded; rescale the training data")
          if (rotationNorm <= tolerance || tt <= math.pow(tolerance * xNorm * rotationNorm, 2)) {
            exhausted = true
          } else {
            p_a = (r_a.t * XX).t / tt                                         // X-loadings - ((K x 1)' * (K x K))' / tt === (K x 1) / tt
            q_a = (r_a.t * XY).t / tt                                         // Y-loadings - ((K x 1)' * (K x M))' / tt === (M x 1) / tt
            require(p_a.valuesIterator.forall(finite) && q_a.valuesIterator.forall(finite),
              "Numerical range exceeded; rescale the training data")
            XY = XY - ((p_a * q_a.t) * tt)                                    // XtY deflation


            // update loadings and weights
            W(::, a) := w_a
            P(::, a) := p_a
            Q(::, a) := q_a
            R(::, a) := r_a
            a += 1
          }
        }
      }
      finishModel(W, P, Q, R, a)
    }

    def trainAndComputeVIP(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): (Model, DenseVector[Double]) = {
      val model = train(X, Y, A)
      val vipVector = VIP.computeVIP(model, X, Y)
      (model, vipVector)
    }

    // Y = X * B + e
    // predict just ignores the error term, since it only consists of the residuals
    def predict(model: Model, X: DenseMatrix[Double]): DenseMatrix[Double] = {
      require(X.cols == model.Beta.rows, "Prediction X has the wrong number of predictors")
      require(X.valuesIterator.forall(finite), "Prediction values must be finite")
      val prediction = X * model.Beta
      require(prediction.valuesIterator.forall(finite), "Numerical range exceeded; rescale the prediction data")
      prediction
    }

    def standardizeAndTrain(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): StandardizedModel[Model] = {
      validateTraining(X, Y, A)
      require(X.rows >= 2, "Standardization needs at least two training observations")
      val (standardizedX, factorsX) = Standardize.centerAndScaleColumnsReturningFactors(X)
      val (standardizedY, factorsY) = Standardize.centerAndScaleColumnsReturningFactors(Y)
      val model = train(standardizedX, standardizedY, A)
      StandardizedModel(model, factorsX, factorsY)
    }

    def standardizeAndPredict(standardizedModel: StandardizedModel[Model], X: DenseMatrix[Double]): DenseMatrix[Double] = {
      val factorsX = standardizedModel.standardizationFactorsX
      val factorsY = standardizedModel.standardizationFactorsY
      val standardizedX = Standardize.centerAndScaleColumnsWithFactors(X, factorsX.meanOfColumns, factorsX.stdDevOfColumns)
      val standardizedEstimateY = predict(standardizedModel.model, standardizedX)
      Standardize.denormalizeCenteredAndScaledColumns(standardizedEstimateY, factorsY.meanOfColumns, factorsY.stdDevOfColumns)
    }

  }
}
