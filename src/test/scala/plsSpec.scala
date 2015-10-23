import org.scalatest._

import breeze.linalg._

import com.github.davidkellis.pls._

class CsvSpec extends FlatSpec {
  "Csv" should "read a CSV of numbers in as a pair of predictor and response matrices" in {
    val (x, y) = Csv.read("data/test.csv", 1)    // has 3 columns
    // test.csv looks like this:
    // 1.0,2.0,3.0
    // 10.0,20.0,30.0
    // 100.0,200.0,300.0

    assert(x.cols === 2)
    assert(x.rows === 3)
    assert(x === DenseMatrix((2.0, 3.0), (20.0, 30.0), (200.0, 300.0)))

    assert(y.cols === 1)
    assert(y.rows === 3)
    assert(y === DenseVector(1.0, 10.0, 100.0).toDenseMatrix.t)
  }
}

class StandardizeSpec extends FlatSpec {
  "Standardize module" should "mean center columns in a matrix" in {
    val M = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))

    val expectedColumnMeans = DenseMatrix((3.0, 6.0))
    val expectedMeanCenteredM = DenseMatrix((-2.0, -4.0), (-1.0, -2.0), (0.0, 0.0), (1.0, 2.0), (2.0, 4.0))

    assert(Standardize.meanCenterColumns(M) === expectedMeanCenteredM)
    assert(Standardize.meanCenterColumns2(M) === (expectedMeanCenteredM, expectedColumnMeans))
    assert(Standardize.meanCenterColumnsWithMeans(M, expectedColumnMeans) === expectedMeanCenteredM)
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

class PlsSpec extends FlatSpec {
  "DayalMcGregor's Algorithm2" should "predict interest rates as well as simple linear regression (see http://www.cyclismo.org/tutorial/R/linearLeastSquares.html)" in {
    val X = DenseMatrix((2000.0, 1.0), (2001.0, 1.0), (2002.0, 1.0), (2003.0, 1.0), (2004.0, 1.0))
    val Y = DenseVector(9.34, 8.50, 7.62, 6.93, 6.60).toDenseMatrix.t             // (5 x 1) matrix

    val model = DayalMcGregor.Algorithm2.train(X, Y, 2)

    val expectedY = DenseVector(-1.367).toDenseMatrix.t
    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((2015.0, 1.0)))

    assert(approxY === expectedY)

    val model2 = DayalMcGregor.Algorithm2.standardizeAndTrain(X, Y, 2)
    val approxY2 = DayalMcGregor.Algorithm2.standardizeAndPredict(model2, DenseMatrix((2015.0, 1.0)))

    assert(approxY2 === expectedY)
  }

  it should "perfectly predict a perfect linear relationship between two predictor variables and one response" in {
    val X = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))
    val Y = DenseVector(2.0, 4.0, 6.0, 8.0, 10.0).toDenseMatrix.t             // (5 x 1) matrix

    val model = DayalMcGregor.Algorithm2.train(X, Y, 2)
    println(model)

    val expectedY = DenseVector(18.0).toDenseMatrix.t
    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((9.0, 18.0)))

    assert(approxY === expectedY)

    val model2 = DayalMcGregor.Algorithm2.standardizeAndTrain(X, Y, 2)
    println(model2)
    val approxY2 = DayalMcGregor.Algorithm2.standardizeAndPredict(model2, DenseMatrix((9.0, 18.0)))
    assert(approxY2 === expectedY)
  }

  it should "predict the gasoline data" in {
    val (trainingX, trainingY) = Csv.read("data/gasoline_train.csv", 1)    // has two columns: octane, NIR
//    println("AAAAAAAAAAAAAAAAAAAAAAAA")
//    println(trainingX)
//    println("BBBBBBBBBBBBBBBBBBBBBBBB")
//    println(trainingY)

//    val model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 2)
    val model = DayalMcGregor.Algorithm2.standardizeAndTrain(trainingX, trainingY, 2)

//    println("========================")
//    println(model.Beta)

    val (testingX, testingY) = Csv.read("data/gasoline_test.csv", 1)
//    println("CCCCCCCCCCCCCCCCCCCCCCCC")
//    println(testingX)
//    println("CCCCCCCCCCCCCCCCCCCCCCCC")
//    println(testingY)

//    val approxY = DayalMcGregor.Algorithm2.predict(model, testingX)
    val approxY = DayalMcGregor.Algorithm2.standardizeAndPredict(model, testingX)

    val expectedApproxY = DenseVector(
      87.94125,
      87.25242,
      88.15832,
      84.96913,
      85.15396,
      84.51415,
      87.56190,
      86.84622,
      89.18925,
      87.09116
    ).toDenseMatrix.t

    assert(approxY === expectedApproxY)
  }
}
