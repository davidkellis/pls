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

  // Algorithm2 doesn't work if any column in X or Y is all zeros.
  // Algorithm2 may be unable to train a model if X has duplicate columns, which will be the case if X has any column that is a scalar-multiple
  //   of another column in X, and is subsequently mean-centered and scaled before being trained.
  object Algorithm2 extends PlsModel[Model]{
    // X - predictor variables matrix (N x K)
    // Y - response variables matrix (N x M)
    // B_PLS - PLS regression coefficients matrix (K x M)
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
      val K = X.cols // number of columns in X - number of predictor variables
      val M = Y.cols // number of columns in Y - number of response variables
      val N = X.rows // number of rows in X === number of rows in Y - number of observations of predictor and response variables
      val W: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, A)        // (K x A)
      val P: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, A)        // (K x A)
      val Q: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, A)        // (M x A)
      val R: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, A)        // (K x A)
      var YtXXtY: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, M)   // (M x M) matrix
      var w_a: DenseVector[Double] = DenseVector.zeros[Double](K)         // (K x 1) - column vector of W
      var p_a: DenseVector[Double] = DenseVector.zeros[Double](M)         // (K x 1) - column vector of P
      var q_a: DenseVector[Double] = DenseVector.zeros[Double](M)         // (M x 1) - column vector of Q
      var r_a: DenseVector[Double] = DenseVector.zeros[Double](K)         // (K x 1) - column vector of R
      var tt: Double = 0.0
      var indexOfLargestEigenvalue: Int = 0
      var j: Int = 0

//      println("***")
//      println(X.t)
//      println(Y)

      var XY: DenseMatrix[Double] = X.t * Y                               // compute the covariance matrices; (K x M) matrix
      val XX: DenseMatrix[Double] = X.t * X                               // (K x K) matrix

//      println("XX")
//      println(XX)
//      println("XY")
//      println(XY)

      // A = number of PLS components to compute
      for (a <- 0 to (A - 1)) {
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

//        println("0000000000000000000000000000000000000000000000000000000000")
//        println("w_a")
//        println(w_a)
//        println("p_a")
//        println(p_a)
//        println("q_a")
//        println(q_a)
//        println("r_a")
//        println(r_a)

        w_a = w_a / sqrt(w_a.t * w_a)                                     // normalize w_a to unity - the denominator is a scalar
//        println("11111111111111111111111111111 - w_a")
//        println(w_a)
        r_a = w_a                                                         // loop to compute r_a
//        println("22222222222222222222222222222 - r_a")
//        println(r_a)
        for (j <- 0 to (a - 1)) {
          r_a = r_a - ( (P(::, j).t * w_a) * R(::, j) )                   // (K x 1) - ( (1 x K) * (K x 1) ) * (K x 1) === (K x 1) - scalar * (K x 1)
//          println(r_a)
        }
        tt = r_a.t * XX * r_a                                             // compute t't - (1 x 1) that is auto-converted to a scalar
//        println("33333333333333333333333333333 - tt")
//        println(tt)
        p_a = (r_a.t * XX).t / tt                                         // X-loadings - ((K x 1)' * (K x K))' / tt === (K x 1) / tt
//        println("44444444444444444444444444444 - p_a")
//        println(p_a)
        q_a = (r_a.t * XY).t / tt                                         // Y-loadings - ((K x 1)' * (K x M))' / tt === (M x 1) / tt
//        println("55555555555555555555555555555 - q_a")
//        println(q_a)
        XY = XY - ((p_a * q_a.t) * tt)                                    // XtY deflation
//        println("66666666666666666666666666666 - XY")
//        println(XY)

//        println("777777777777777777777777777777777777777777777777777777777777")
//        println("tt", tt)
//        println("w_a")
//        println(w_a)
//        println("p_a")
//        println(p_a)
//        println("q_a")
//        println(q_a)
//        println("r_a")
//        println(r_a)

        // update loadings and weights
        W(::, a) := w_a
        P(::, a) := p_a
        Q(::, a) := q_a
        R(::, a) := r_a
      }
      val beta = R * Q.t // compute the regression coefficients; (K x M)

      Model(beta, W, P, Q, R)
    }

    // Y = X * B + e
    // predict just ignores the error term, since it only consists of the residuals
    def predict(model: Model, X: DenseMatrix[Double]): DenseMatrix[Double] = {
      X * model.Beta
    }

    def standardizeAndTrain(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): StandardizedModel[Model] = {
      val (standardizedX, factorsX) = Standardize.centerAndScaleColumnsReturningFactors(X)
      val (standardizedY, factorsY) = Standardize.centerAndScaleColumnsReturningFactors(Y)
//      println("standardizeAndTrain")
//      println("standardizedX")
//      println(standardizedX)
//      println("factorsX")
//      println(factorsX)
//      println("standardizedY")
//      println(standardizedY)
//      println("factorsY")
//      println(factorsY)
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