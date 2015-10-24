package com.github.davidkellis.pls

import breeze.linalg._
import org.scalatest._

class StandardizeSpec extends FlatSpec {
   "Standardize module" should "mean center columns in a matrix" in {
     val M = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))

     val expectedColumnMeans = DenseMatrix((3.0, 6.0))
     val expectedMeanCenteredM = DenseMatrix((-2.0, -4.0), (-1.0, -2.0), (0.0, 0.0), (1.0, 2.0), (2.0, 4.0))

     assert(Standardize.meanCenterColumns(M) === expectedMeanCenteredM)
     assert(Standardize.meanCenterColumns2(M) === (expectedMeanCenteredM, expectedColumnMeans))
     assert(Standardize.meanCenterColumnsWithMeans(M, expectedColumnMeans) === expectedMeanCenteredM)
   }

   it should "mean-center a column containing one distinct value by treating that column as having mean 0.0" in {
     val M = DenseVector(3.0, 3.0, 3.0).toDenseMatrix.t    // a (3 x 1)
     val expectedMeanCenteredM = DenseVector(3.0, 3.0, 3.0).toDenseMatrix.t    // a (3 x 1)
     val expectedColumnMeans = DenseMatrix((0.0))
     assert(Standardize.meanCenterColumns(M) === expectedMeanCenteredM)
     assert(Standardize.meanCenterColumns2(M) === (expectedMeanCenteredM, expectedColumnMeans))
   }

   it should "un-mean-center (denormalize) the mean-centered matrix" in {
     val meanCenteredM = DenseMatrix((-2.0, -4.0), (-1.0, -2.0), (0.0, 0.0), (1.0, 2.0), (2.0, 4.0))
     val columnMeans = DenseMatrix((3.0, 6.0))
     val expectedDenormalizedM = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))
     assert(Standardize.denormalizeMeanCenteredColumns(meanCenteredM, columnMeans) === expectedDenormalizedM)
   }

   it should "scale the columns in a matrix" in {
     val M = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))

 //    val expectedColumnMeans = Utils.stddevColumns(M)
 //    val expectedMeanCenteredM = M :/ Utils.vertcat(expectedColumnMeans, 5)
     val expectedColumnStdDev = DenseMatrix((1.5811388300841898, 3.1622776601683795))
     val expectedScaledM = DenseMatrix(
       (0.6324555320336759, 0.6324555320336759),
       (1.2649110640673518, 1.2649110640673518),
       (1.8973665961010275, 1.8973665961010275),
       (2.5298221281347035, 2.5298221281347035),
       (3.162277660168379,  3.162277660168379)
     )

     assert(Standardize.scaleColumns(M) === expectedScaledM)
     assert(Standardize.scaleColumns2(M) === (expectedScaledM, expectedColumnStdDev))
     assert(Standardize.scaleColumnsWithStdDevs(M, expectedColumnStdDev) === expectedScaledM)
   }

   it should "scale a column containing one distinct value by dividing the column by the one common value, yielding a scaled column of 1.0s" in {
     val M = DenseVector(3.0, 3.0, 3.0).toDenseMatrix.t    // a (3 x 1)
     val expectedScaledM = DenseVector(1.0, 1.0, 1.0).toDenseMatrix.t    // a (3 x 1)
     val expectedColumnStdDev = DenseMatrix((3.0))
     assert(Standardize.scaleColumns(M) === expectedScaledM)
     assert(Standardize.scaleColumns2(M) === (expectedScaledM, expectedColumnStdDev))
   }

   it should "un-scale (denormalize) the standard-deviation-scaled matrix" in {
     val scaledM = DenseMatrix(
       (0.6324555320336759, 0.6324555320336759),
       (1.2649110640673518, 1.2649110640673518),
       (1.8973665961010275, 1.8973665961010275),
       (2.5298221281347035, 2.5298221281347035),
       (3.162277660168379,  3.162277660168379)
     )
     val columnStandardDeviations = DenseMatrix((1.5811388300841898, 3.1622776601683795))
     val expectedDenormalizedM = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))
     assert(Standardize.denormalizeScaledColumns(scaledM, columnStandardDeviations) === expectedDenormalizedM)
   }

   it should "standardize (mean center and then scale) the columns in a matrix" in {
     val M = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))

     val expectedColumnMeans = DenseMatrix((3.0, 6.0))
     val expectedColumnStandardDeviations = DenseMatrix((1.5811388300841898, 3.1622776601683795))
     val expectedStandardizedM = DenseMatrix(
       (-1.2649110640673518  ,-1.2649110640673518),
       (-0.6324555320336759  ,-0.6324555320336759),
       (0.0                  ,0.0),
       (0.6324555320336759   ,0.6324555320336759),
       (1.2649110640673518   ,1.2649110640673518)
     )

     assert(Standardize.centerAndScaleColumns(M) === expectedStandardizedM)
     assert(Standardize.centerAndScaleColumnsReturningFactors(M) === (expectedStandardizedM, Standardize.StandardizationFactors(expectedColumnMeans, expectedColumnStandardDeviations)))
     assert(Standardize.centerAndScaleColumnsWithFactors(M, expectedColumnMeans, expectedColumnStandardDeviations) === expectedStandardizedM)
   }

   it should "un-standardize (denormalize) the mean-centered and scaled matrix" in {
     val standardizedM = DenseMatrix(
       (-1.2649110640673518  ,-1.2649110640673518),
       (-0.6324555320336759  ,-0.6324555320336759),
       (0.0                  ,0.0),
       (0.6324555320336759   ,0.6324555320336759),
       (1.2649110640673518   ,1.2649110640673518)
     )

     val columnMeans = DenseMatrix((3.0, 6.0))
     val columnStandardDeviations = DenseMatrix((1.5811388300841898, 3.1622776601683795))
     val expectedM = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))
     assert(Standardize.denormalizeCenteredAndScaledColumns(standardizedM, columnMeans, columnStandardDeviations) === expectedM)
   }

 }
