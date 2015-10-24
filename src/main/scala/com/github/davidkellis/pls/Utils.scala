package com.github.davidkellis.pls

import breeze.linalg.{*, DenseMatrix, sum}
import breeze.stats.{mean, stddev}

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
