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
    xMean: DenseVector[Double],
    yMean: DenseVector[Double],
    xStdDev: DenseVector[Double],
    yStdDev: DenseVector[Double]
  )

  // The average of each column is subtracted from all the values in corresponding column.
  // As a result of centering, the columns of the centered matrix have a mean of 0.
  // See Applied Predictive Modeling, page 30-31.
  def meanCenterColumns(M: DenseMatrix[Double]): DenseMatrix[Double] = M - Utils.vertcat(Utils.meanColumns(M), M.rows)

  // To scale the data, every value within each specific column is divided by the column-specific standard deviation.
  // Each column of the scaled matrix has a standard deviaion of 1.
  // See Applied Predictive Modeling, page 30-31.
  def scaleColumns(M: DenseMatrix[Double]): DenseMatrix[Double] = M / Utils.vertcat(Utils.stddevColumns(M), M.rows)

  // mean center and then scale each column of the given matrix, M
  def centerAndScaleColumns(M: DenseMatrix[Double]): DenseMatrix[Double] = scaleColumns(meanCenterColumns(M))
}

//abstract class PlsModel[ModelT](var model: ModelT) {
//  // X - predictor variables matrix (N × K)
//  // Y - response variables matrix (N × M)
//  // A - is the number of latent variables (a.k.a. components) to use
//  def train(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): ModelT
//  def predict(X: DenseMatrix[Double]): DenseMatrix[Double]
//}

trait PlsModel[ModelT] {
  // X - predictor variables matrix (N × K)
  // Y - response variables matrix (N × M)
  // A - is the number of latent variables (a.k.a. components) to use
  def train(X: DenseMatrix[Double], Y: DenseMatrix[Double], A: Int): ModelT
  def predict(model: ModelT, X: DenseMatrix[Double]): DenseMatrix[Double]
}

object DayalMcGregor {
  // Beta - PLS regression coefficients matrix (K × M)
  // W - PLS weights matrix for X (K × A)
  // P - PLS loadings matrix for X (K × A)
  // Q - PLS loadings matrix for Y (M × A)
  // R - PLS weights matrix to compute scores T directly from original X (K × A)
  case class Model(
    Beta: DenseMatrix[Double],
    W: DenseMatrix[Double],
    P: DenseMatrix[Double],
    Q: DenseMatrix[Double],
    R: DenseMatrix[Double]
  )

  object Algorithm2 extends PlsModel[Model]{
    // X - predictor variables matrix (N × K)
    // Y - response variables matrix (N × M)
    // B_PLS - PLS regression coefficients matrix (K × M)
    // W - PLS weights matrix for X (K × A)
    // P - PLS loadings matrix for X (K × A)
    // Q - PLS loadings matrix for Y (M × A)
    // R - PLS weights matrix to compute scores T directly from original X (K × A)
    // T - PLS scores matrix of X (N × A)
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
      val W: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, A)        // (K × A)
      val P: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, A)        // (K × A)
      val Q: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, A)        // (M × A)
      val R: DenseMatrix[Double] = DenseMatrix.zeros[Double](K, A)        // (K × A)
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

        println("XY")
        println(XY)
        println("XX")
        println(XX)

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
          w_a = XY * q_a                                                  // compute X-weights; w_a is (K x 1)
        }
        println("before w_a")
        println(w_a)

        w_a = w_a / sqrt(w_a.t * w_a)                                     // normalize w_a to unity
        r_a = w_a                                                         // loop to compute r_a
        for (j <- 1 to (a - 1)) {
          r_a = r_a - (P(::, j).t * w_a) * R(::, j)
        }
        tt = r_a.t * XX * r_a                                             // compute t't - (1 x 1)
        p_a = (r_a.t * XX).t / tt                                         // X-loadings
        q_a = (r_a.t * XY).t / tt                                         // Y-loadings
        XY = XY - (p_a * q_a.t) * tt                                      // XtY deflation

        println("w_a")
        println(w_a)
        println("p_a")
        println(p_a)
        println("q_a")
        println(q_a)
        println("r_a")
        println(r_a)

        // update loadings and weights
        W(::, i) := w_a
        P(::, i) := p_a
        Q(::, i) := q_a
        R(::, i) := r_a
      }
      val beta = R * Q.t // compute the regression coefficients; (K x M)

      Model(beta, W, P, Q, R)
    }

    // Y = X*B + e
    // predict just ignores the error term, since it only consists of the residuals
    def predict(model: Model, X: DenseMatrix[Double]): DenseMatrix[Double] = {
      X*model.Beta
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

    val X = DenseMatrix.zeros[Double](N, K)   // X - predictor variables matrix (N × K)
    val Y = DenseMatrix.zeros[Double](N, M)   // Y - response variables matrix (N × M)

    (0 until M).foreach { c =>
      Y(::, c) := csvMatrix(::, c)
    }

    (M until csvMatrix.cols).foreach { c =>
      X(::, c - M) := csvMatrix(::, c)
    }

    (X, Y)
  }
}
