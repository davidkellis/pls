package com.github.davidkellis.pls

import breeze.linalg._
import breeze.linalg.eigSym.EigSym
import breeze.numerics._
import breeze.stats._

import scala.annotation.tailrec

object Utils {
  // perform column-wise sum; given M (RxC), yields a (1xC) matrix of per-column sums
  def sumColumns(M: DenseMatrix[Double]): DenseMatrix[Double] = sum(M(::, *))

  // perform column-wise mean; given M (RxC), yields a (1xC) matrix of per-column means
  def meanColumns(M: DenseMatrix[Double]): DenseMatrix[Double] = mean(M(::, *))

  // perform column-wise standard deviation; given M (RxC), yields a (1xC) matrix of per-column standard deviations
  def stddevColumns(M: DenseMatrix[Double]): DenseMatrix[Double] = stddev(M(::, *))

  // vertically concatenate matrix M to itself n-1 times.
  // returns a matrix with n vertically stacked copies of M.
  def vertcat(M: DenseMatrix[Double], n: Int): DenseMatrix[Double] = {
    @tailrec
    def vertcatR(A: DenseMatrix[Double], i: Int): DenseMatrix[Double] = {
      if (i <= 1)
        A
      else
        vertcatR(DenseMatrix.vertcat(M, A), i - 1)
    }
    vertcatR(M, n)
  }
}

object Standardize {
  case class StandardizationFactors(
    meanOfColumns: DenseMatrix[Double],     // meanOfColumns is a row vector
    stdDevOfColumns: DenseMatrix[Double]    // stdDevOfColumns is a row vector
  )


  ////////////////////// Mean Centering //////////////////////

  // M is a (R x C) matrix
  // meanOfColumnsInM is a (1 x C) matrix
  def meanCenterColumnsWithMeans(M: DenseMatrix[Double], meanOfColumnsInM: DenseMatrix[Double]): DenseMatrix[Double] = M - Utils.vertcat(meanOfColumnsInM, M.rows)

  // The average of each column is subtracted from all the values in corresponding column.
  // As a result of centering, the columns of the centered matrix have a mean of 0.
  // See Applied Predictive Modeling, page 30-31.
  // Returns a pair of matrices (meanCenteredM, meanOfColumnsInM)
  def meanCenterColumns2(M: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val meanOfColumnsInM = Utils.meanColumns(M)
    val meanCenteredM = meanCenterColumnsWithMeans(M, meanOfColumnsInM)
    (meanCenteredM, meanOfColumnsInM)
  }

  // Returns the mean-centered version of M
  def meanCenterColumns(M: DenseMatrix[Double]): DenseMatrix[Double] = meanCenterColumns2(M)._1

  def denormalizeMeanCenteredColumns(meanCenteredM: DenseMatrix[Double], meanOfColumnsInM: DenseMatrix[Double]): DenseMatrix[Double] = {
    meanCenteredM + Utils.vertcat(meanOfColumnsInM, meanCenteredM.rows)
  }

  ////////////////////// Scaling //////////////////////

  // M is a (R x C) matrix
  // meanOfColumnsInM is a (1 x C) matrix
  def scaleColumnsWithStdDevs(M: DenseMatrix[Double], stdDevOfColumnsInM: DenseMatrix[Double]): DenseMatrix[Double] = M :/ Utils.vertcat(stdDevOfColumnsInM, M.rows)

  // To scale the data, every value within each specific column is divided by the column-specific standard deviation.
  // Each column of the scaled matrix has a standard deviation of 1.
  // See Applied Predictive Modeling, page 30-31.
  // Returns a pair of matrices (scaledM, stdDevOfColumnsInM)
  def scaleColumns2(M: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val stdDevOfColumnsInM = Utils.stddevColumns(M)
    val scaledM = scaleColumnsWithStdDevs(M, stdDevOfColumnsInM)
    (scaledM, stdDevOfColumnsInM)
  }

  // Returns the standard-deviation-scaled version of M
  def scaleColumns(M: DenseMatrix[Double]): DenseMatrix[Double] = scaleColumns2(M)._1

  def denormalizeScaledColumns(scaledM: DenseMatrix[Double], stdDevOfColumnsInM: DenseMatrix[Double]): DenseMatrix[Double] = {
    scaledM :* Utils.vertcat(stdDevOfColumnsInM, scaledM.rows)
  }


  ////////////////////// Scaling and Mean Centering //////////////////////

  def centerAndScaleColumnsWithFactors(M: DenseMatrix[Double], meanOfColumnsInM: DenseMatrix[Double], stdDevOfColumnsInM: DenseMatrix[Double]): DenseMatrix[Double] = {
    scaleColumnsWithStdDevs(meanCenterColumnsWithMeans(M, meanOfColumnsInM), stdDevOfColumnsInM)
  }

  // mean center and then scale each column of the given matrix, M, returning the pair (scaledMeanCenteredM, StandardizationFactors(meanOfColumnsInM, stdDevOfMeanCenteredColumnsInM))
  def centerAndScaleColumnsReturningFactors(M: DenseMatrix[Double]): (DenseMatrix[Double], StandardizationFactors) = {
    val (meanCenteredM, meanOfColumnsInM) = meanCenterColumns2(M)
    val (scaledMeanCenteredM, stdDevOfMeanCenteredColumnsInM) = scaleColumns2(meanCenteredM)
    (scaledMeanCenteredM, StandardizationFactors(meanOfColumnsInM, stdDevOfMeanCenteredColumnsInM))
  }

  // mean center and then scale each column of the given matrix, M
  def centerAndScaleColumns(M: DenseMatrix[Double]): DenseMatrix[Double] = scaleColumns(meanCenterColumns(M))

  def denormalizeCenteredAndScaledColumns(standardizedM: DenseMatrix[Double], meanOfColumnsInM: DenseMatrix[Double], stdDevOfColumnsInM: DenseMatrix[Double]): DenseMatrix[Double] = {
    denormalizeMeanCenteredColumns(denormalizeScaledColumns(standardizedM, stdDevOfColumnsInM), meanOfColumnsInM)
  }
}

//abstract class PlsModel[ModelT](var model: ModelT) {
//  // X - predictor variables matrix (N x K)
//  // Y - response variables matrix (N x M)
//  // A - is the number of latent variables (a.k.a. components) to use
//  def train(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): ModelT
//  def predict(X: DenseMatrix[Double]): DenseMatrix[Double]
//}

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
      var i: Int = 0

//      println("***")
//      println(X.t)
//      println(Y)

      var XY: DenseMatrix[Double] = X.t * Y                               // compute the covariance matrices; (K x M) matrix
      val XX: DenseMatrix[Double] = X.t * X                               // (K x K) matrix
      for (a <- 1 to A) {

//        println("XY")
//        println(XY)
//        println("XX")
//        println(XX)

        // A = number of PLS components to compute
        i = a - 1 // i is the zero-based index; a is the 1-based index

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
//        println("before w_a")
//        println(w_a)

        w_a = w_a / sqrt(w_a.t * w_a)                                     // normalize w_a to unity - the denominator is a scalar
        r_a = w_a                                                         // loop to compute r_a
        for (j <- 1 to (a - 1)) {
          r_a = r_a - ( (P(::, j).t * w_a) * R(::, j) )                   // (K x 1) - ( (1 x K) * (K x 1) ) * (K x 1) === (K x 1) - scalar * (K x 1)
        }
        tt = r_a.t * XX * r_a                                             // compute t't - (1 x 1) that is auto-converted to a scalar
        p_a = (r_a.t * XX).t / tt                                         // X-loadings - ((K x 1)' * (K x K))' / tt === (K x 1) / tt
        q_a = (r_a.t * XY).t / tt                                         // Y-loadings - ((K x 1)' * (K x M))' / tt === (M x 1) / tt
        XY = XY - ((p_a * q_a.t) * tt)                                    // XtY deflation

//        println("w_a")
//        println(w_a)
//        println("p_a")
//        println(p_a)
//        println("q_a")
//        println(q_a)
//        println("r_a")
//        println(r_a)

        // update loadings and weights
        W(::, i) := w_a
        P(::, i) := p_a
        Q(::, i) := q_a
        R(::, i) := r_a
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
      val model = train(standardizedX, standardizedY, A)
      StandardizedModel(model, factorsX, factorsY)
    }

    def standardizeAndPredict(standardizedModel: StandardizedModel[Model], X: DenseMatrix[Double]): DenseMatrix[Double] = {
      val factorsX = standardizedModel.standardizationFactorsX
      val factorsY = standardizedModel.standardizationFactorsX
      val standardizedX = Standardize.centerAndScaleColumnsWithFactors(X, factorsX.meanOfColumns, factorsX.stdDevOfColumns)
      val standardizedEstimateY = predict(standardizedModel.model, standardizedX)
      Standardize.denormalizeCenteredAndScaledColumns(standardizedEstimateY, factorsY.meanOfColumns, factorsY.stdDevOfColumns)
    }

  }
}

object Csv {
  // Reads a CSV file into a pair of matrices representing the predictor variables and the response variables
  // Assumes the leftmost M columns in the CSV are the response variables, followed by the remaining columns holding the predictor variables
  // Returns a pair of matrices of the form (predictor matrix, response matrix)
  def read(filename: String, M: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val file = new java.io.File(filename)
    val csvMatrix = csvread(file)

    // M is the number of response variables
    val N = csvMatrix.rows       // number of rows
    val K = csvMatrix.cols - M   // number of predictor variables

    val X = DenseMatrix.zeros[Double](N, K)   // X - predictor variables matrix (N x K)
    val Y = DenseMatrix.zeros[Double](N, M)   // Y - response variables matrix (N x M)

    (0 until M).foreach { c =>
      Y(::, c) := csvMatrix(::, c)
    }

    (M until csvMatrix.cols).foreach { c =>
      X(::, c - M) := csvMatrix(::, c)
    }

    (X, Y)
  }
}
