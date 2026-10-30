package com.github.davidkellis.pls

import breeze.linalg.DenseMatrix

object Standardize {
  case class StandardizationFactors(
    meanOfColumns: DenseMatrix[Double],     // meanOfColumns is a row vector
    stdDevOfColumns: DenseMatrix[Double]    // stdDevOfColumns is a row vector
  )


  ////////////////////// Mean Centering //////////////////////

  // M is a (R x C) matrix
  // meanOfColumnsInM is a (1 x C) matrix
  def meanCenterColumnsWithMeans(M: DenseMatrix[Double], meanOfColumnsInM: DenseMatrix[Double]): DenseMatrix[Double] = M - Utils.vertcat(meanOfColumnsInM, M.rows)

  // Center every column, including constants. The stored mean restores the
  // intercept when predictions are transformed back to the original units.
  // Explicit intercept columns are unnecessary with standardizeAndTrain.
  def meanCenterColumns2(M: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    require(M.rows > 0 && M.cols > 0, "Cannot center an empty matrix")
    val meanOfColumnsInM = Utils.meanColumns(M)
    (meanCenterColumnsWithMeans(M, meanOfColumnsInM), meanOfColumnsInM)
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
  // Each column of the scaled matrix has a standard deviation of 1 or 0. The only time a column will have a std. dev. of 0 is when
  // the column has only one distinct value.
  // See Applied Predictive Modeling, page 30-31.
  // Returns a pair of matrices (scaledM, stdDevOfColumnsInM)
  def scaleColumns2(M: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    require(M.rows >= 2 && M.cols > 0, "Scaling needs at least two rows and one column")
    val stdDevOfColumnsInM = Utils.stddevColumns(M)
    // Unit scale leaves zero-variance columns finite and preserves their
    // original values when this scaling-only function is used independently.
    val zeroAdjustedStandardDeviations = stdDevOfColumnsInM.map { stdDev =>
      if (stdDev == 0.0) 1.0 else stdDev
    }

    val scaledM = scaleColumnsWithStdDevs(M, zeroAdjustedStandardDeviations)
    (scaledM, zeroAdjustedStandardDeviations)
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
