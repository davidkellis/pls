package com.github.davidkellis.pls

import breeze.linalg._
import breeze.numerics.NaN

import org.scalatest._
import Matchers._

trait TestHelpers {
  def beBetween(x: Double, min: Double, max: Double): Boolean = x >= min && x <= max

  def beWithinTolerance(x: Double, base: Double, e: Double): Boolean = beBetween(x, base - e, base + e)

  def beWithinMillionth(x: Double, expectedX: Double): Boolean = beWithinTolerance(x, expectedX, 0.000001)

  def assertMatrixEqualWithinMillionth(x: DenseMatrix[Double], expectedX: DenseMatrix[Double]): Unit = {
    assert(x.cols === expectedX.cols)
    assert(x.rows === expectedX.rows)
    for {
      r <- 0 until x.rows
      c <- 0 until x.cols
    } assert(
      (x(r, c).equals(NaN) && expectedX(r, c).equals(NaN)) ||
      beWithinMillionth(x(r, c), expectedX(r, c))
    )
  }

  def assertVectorEqualWithinMillionth(x: DenseVector[Double], expectedX: DenseVector[Double]): Unit = {
    assert(x.length === expectedX.length)
    for {
      r <- 0 until x.length
    } assert(beWithinMillionth(x(r), expectedX(r)))
  }
}

class PlsSpec extends FlatSpec with TestHelpers {
  "DayalMcGregor's Algorithm1" should "predict interest rates as well as simple linear regression (see http://www.cyclismo.org/tutorial/R/linearLeastSquares.html)" in {
    val X = DenseMatrix((2000.0, 1.0), (2001.0, 1.0), (2002.0, 1.0), (2003.0, 1.0), (2004.0, 1.0))
    val Y = DenseVector(9.34, 8.50, 7.62, 6.93, 6.60).toDenseMatrix.t             // (5 x 1) matrix

    val model = DayalMcGregor.Algorithm1.train(X, Y, 2)

    val approxY = DayalMcGregor.Algorithm1.predict(model, DenseMatrix((2015.0, 1.0)))

    val expectedY = DenseVector(-1.366999998845131).toDenseMatrix.t
    assertMatrixEqualWithinMillionth(approxY, expectedY)
    assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((-0.7049999999111803), (1419.2079998221832)))
    assertMatrixEqualWithinMillionth(model.W, DenseMatrix((0.9999998752271115, -4.995417352414563e-4), (4.995455548717368e-4, 0.9999998752290197)))
    assertMatrixEqualWithinMillionth(model.P, DenseMatrix((0.9999998752497743, -4.994007749060281e-4), (4.995001879364195e-4,  0.9999998752994292)))
    assertMatrixEqualWithinMillionth(model.Q, DenseMatrix((0.003894750669779322, 1419.2081749231177)))
    assertMatrixEqualWithinMillionth(model.R, DenseMatrix((0.9999998752271115, -4.995001879364196e-4), (4.995455548717368e-4, 0.9999998752497744)))


    val model2 = DayalMcGregor.Algorithm1.standardizeAndTrain(X, Y, 2)
    val approxY2 = DayalMcGregor.Algorithm1.standardizeAndPredict(model2, DenseMatrix((2015.0, 1.0)))

    val expectedY2 = DenseVector(-1.366999999999999).toDenseMatrix.t
    assertMatrixEqualWithinMillionth(approxY2, expectedY2)
    assert(approxY2.rows === 1)
    assert(approxY2.cols === 1)
  }

  it should "perfectly predict a simple perfect linear relationship between two predictor variables and one response; also weights,loadings,etc. should match Matlab implementation" in {
    val X = DenseMatrix((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
    val Y = DenseVector(1.0, 2.0, 3.0).toDenseMatrix.t             // (3 x 1) matrix

    val model = DayalMcGregor.Algorithm1.train(X, Y, 1)

    assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((0.5), (0.5)))
    assertMatrixEqualWithinMillionth(model.W, DenseMatrix((0.7071067811865476), (0.7071067811865476)))
    assertMatrixEqualWithinMillionth(model.P, DenseMatrix((0.7071067811865475), (0.7071067811865475)))
    assertMatrixEqualWithinMillionth(model.Q, DenseMatrix((0.7071067811865475)))
    assertMatrixEqualWithinMillionth(model.R, DenseMatrix((0.7071067811865476), (0.7071067811865476)))

    val expectedY = DenseVector(4.0).toDenseMatrix.t
    val approxY = DayalMcGregor.Algorithm1.predict(model, DenseMatrix((4.0, 4.0)))

    assertMatrixEqualWithinMillionth(approxY, expectedY)

    val model2 = DayalMcGregor.Algorithm1.standardizeAndTrain(X, Y, 1)
    val approxY2 = DayalMcGregor.Algorithm1.standardizeAndPredict(model2, DenseMatrix((4.0, 4.0)))

    assertMatrixEqualWithinMillionth(approxY2, expectedY)
  }

  it should "retain the fitted model when duplicate predictors exhaust the usable rank" in {
    val X = DenseMatrix((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
    val Y = DenseMatrix((1.0), (2.0), (3.0))
    val model = DayalMcGregor.Algorithm1.train(X, Y, 2)
    assert(model.R.cols === 1)
    assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((0.5), (0.5)))
    assertMatrixEqualWithinMillionth(
      DayalMcGregor.Algorithm1.predict(model, DenseMatrix((4.0, 4.0))), DenseMatrix((4.0)))
  }

  it should "stop at one component for proportional predictors in raw and standardized models" in {
    val X = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))
    val Y = DenseMatrix((2.0), (4.0), (6.0), (8.0), (10.0))
    for (requested <- Seq(1, 2, 10)) {
      val model = DayalMcGregor.Algorithm1.train(X, Y, requested)
      assert(model.R.cols === 1)
      assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((0.4), (0.8)))
      assertMatrixEqualWithinMillionth(
        DayalMcGregor.Algorithm1.predict(model, DenseMatrix((9.0, 18.0))), DenseMatrix((18.0)))
      val standardized = DayalMcGregor.Algorithm1.standardizeAndTrain(X, Y, requested)
      assert(standardized.model.R.cols === 1)
      assertMatrixEqualWithinMillionth(standardized.model.Beta, DenseMatrix((0.5), (0.5)))
      assertMatrixEqualWithinMillionth(
        DayalMcGregor.Algorithm1.standardizeAndPredict(standardized, DenseMatrix((9.0, 18.0))), DenseMatrix((18.0)))
    }
  }

  it should "predict the gasoline data and yield the same results as matlab" in {
    val (trainingX, trainingY) = Csv.read("data/gasoline_train.csv", Array(0))    // has two columns: octane, NIR

    val model = DayalMcGregor.Algorithm1.train(trainingX, trainingY, 1)

    assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((-1641.8147339889106)))
    assertMatrixEqualWithinMillionth(model.W, DenseMatrix((-1.0)))
    assertMatrixEqualWithinMillionth(model.P, DenseMatrix((-1.0)))
    assertMatrixEqualWithinMillionth(model.Q, DenseMatrix((1641.8147339889106)))
    assertMatrixEqualWithinMillionth(model.R, DenseMatrix((-1.0)))


    val (testingX, testingY) = Csv.read("data/gasoline_test.csv", Array(0))

    val approxY = DayalMcGregor.Algorithm1.predict(model, testingX)

    val expectedApproxY = DenseVector(
      86.415276708772325,       // 88.10 is what's in testingY
      86.523636481215576,       // 87.60 is what's in testingY
      87.663055906603887,       // 88.35 is what's in testingY
      88.877998809755681,       // 85.10 is what's in testingY
      81.471772544731706,       // 85.10 is what's in testingY
      76.974841988336081,       // 84.70 is what's in testingY
      91.211017546753922,       // 87.20 is what's in testingY
      88.153958512066566,       // 86.60 is what's in testingY
      92.452229485649539,       // 89.60 is what's in testingY
      96.546915432217887        // 87.10 is what's in testingY
    ).toDenseMatrix.t

    assertMatrixEqualWithinMillionth(approxY, expectedApproxY)


    val expectedApproxY2 = DenseVector(
      87.22271661912193,        // 88.10 is what's in testingY
      87.2237286040437,         // 87.60 is what's in testingY
      87.23436977882712,        // 88.35 is what's in testingY
      87.24571627643478,        // 85.10 is what's in testingY
      87.1765486403426,         // 85.10 is what's in testingY
      87.13455126608933,        // 84.70 is what's in testingY
      87.2675046184625,         // 87.20 is what's in testingY
      87.23895437718481,        // 86.60 is what's in testingY
      87.27909644574817,        // 89.60 is what's in testingY
      87.317337209307           // 87.10 is what's in testingY
    ).toDenseMatrix.t

    val model2 = DayalMcGregor.Algorithm1.standardizeAndTrain(trainingX, trainingY, 1)
    val approxY2 = DayalMcGregor.Algorithm1.standardizeAndPredict(model2, testingX)

    assertMatrixEqualWithinMillionth(approxY2, expectedApproxY2)
  }

  it should "retain the gasoline regression when more components than predictors are requested" in {
    val (trainingX, trainingY) = Csv.read("data/gasoline_train.csv", Array(0))
    val (testingX, _) = Csv.read("data/gasoline_test.csv", Array(0))
    val baseline = DayalMcGregor.Algorithm1.train(trainingX, trainingY, 1)
    for (requested <- Seq(2, 3, 10)) {
      val model = DayalMcGregor.Algorithm1.train(trainingX, trainingY, requested)
      assert(model.R.cols === 1)
      assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((-1641.8147339889106)))
      assertMatrixEqualWithinMillionth(DayalMcGregor.Algorithm1.predict(model, testingX),
        DayalMcGregor.Algorithm1.predict(baseline, testingX))
    }
  }

  "VIP algorithm with Algorithm1" should "produce VIP values that when squared, sum to the number of predictor values." in {
    var trainingData = Csv.read("data/artificial13.csv", Array(0))    // has 13 predictor variables
    var trainingX = trainingData._1
    var trainingY = trainingData._2

    var model = DayalMcGregor.Algorithm1.train(trainingX, trainingY, 13)
    var vipValues = DayalMcGregor.VIP.computeVIP(model, trainingX, trainingY)
    var sum = vipValues.foldLeft(0.0) { _ + Math.pow(_, 2) }

    assert(sum === (13.0 +- 0.000000001))



    trainingData = Csv.read("data/artificial100.csv", Array(0))    // has 100 predictor variables
    trainingX = trainingData._1
    trainingY = trainingData._2

    model = DayalMcGregor.Algorithm1.train(trainingX, trainingY, 13)
    vipValues = DayalMcGregor.VIP.computeVIP(model, trainingX, trainingY)
    sum = vipValues.foldLeft(0.0) { _ + Math.pow(_, 2) }

    assert(sum === (100.0 +- 0.000000001))
  }

  // Independent scikit-learn 1.9.1 NIPALS reference, two standardized components.
  it should "match independent VIP values on the standardized solubility data" in {
    val filePath = "data/solTrainTrans.csv"
    val A = 2

    val (xHeader, yHeader, x, y) = Csv.readWithHeader(filePath, Array(0))    // assumes the left-most column is the response variable, followed by the predictor columns

    val standardizedModel = DayalMcGregor.Algorithm1.standardizeAndTrain(x, y, A)

    val xFactors = standardizedModel.standardizationFactorsX
    val yFactors = standardizedModel.standardizationFactorsY
    val standardizedX = Standardize.centerAndScaleColumnsWithFactors(x, xFactors.meanOfColumns, xFactors.stdDevOfColumns)
    val standardizedY = Standardize.centerAndScaleColumnsWithFactors(y, yFactors.meanOfColumns, yFactors.stdDevOfColumns)
    val vip = DayalMcGregor.VIP.computeVIP(standardizedModel.model, standardizedX, standardizedY)

    val sortedColumnNameAndVipScorePairs = xHeader.zip(vip.toArray).sortWith( (pair1, pair2) => pair1._2 > pair2._2 ).toArray

    assert(sortedColumnNameAndVipScorePairs.take(4).map(_._1) === Array("MolWeight", "NumCarbon","NumNonHAtoms","NumNonHBonds"))

    assert(sortedColumnNameAndVipScorePairs(0)._2 === 3.0742852550529873 +- 0.000000001)
    assert(sortedColumnNameAndVipScorePairs(1)._2 === 2.8047572729743675 +- 0.000000001)
    assert(sortedColumnNameAndVipScorePairs(2)._2 === 2.697662835874127 +- 0.000000001)
    assert(sortedColumnNameAndVipScorePairs(3)._2 === 2.685881154025871 +- 0.000000001)
  }






  "DayalMcGregor's Algorithm2" should "predict interest rates as well as simple linear regression (see http://www.cyclismo.org/tutorial/R/linearLeastSquares.html)" in {
    val X = DenseMatrix((2000.0, 1.0), (2001.0, 1.0), (2002.0, 1.0), (2003.0, 1.0), (2004.0, 1.0))
    val Y = DenseVector(9.34, 8.50, 7.62, 6.93, 6.60).toDenseMatrix.t             // (5 x 1) matrix

    val model = DayalMcGregor.Algorithm2.train(X, Y, 2)

    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((2015.0, 1.0)))

    val expectedY = DenseVector(-1.366999998845131).toDenseMatrix.t
    assertMatrixEqualWithinMillionth(approxY, expectedY)
    assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((-0.7049999999111803), (1419.2079998221832)))
    assertMatrixEqualWithinMillionth(model.W, DenseMatrix((0.9999998752271115, -4.995417352414563e-4), (4.995455548717368e-4, 0.9999998752290197)))
    assertMatrixEqualWithinMillionth(model.P, DenseMatrix((0.9999998752497743, -4.994007749060281e-4), (4.995001879364195e-4,  0.9999998752994292)))
    assertMatrixEqualWithinMillionth(model.Q, DenseMatrix((0.003894750669779322, 1419.2081749231177)))
    assertMatrixEqualWithinMillionth(model.R, DenseMatrix((0.9999998752271115, -4.995001879364196e-4), (4.995455548717368e-4, 0.9999998752497744)))


    val model2 = DayalMcGregor.Algorithm2.standardizeAndTrain(X, Y, 2)
    val approxY2 = DayalMcGregor.Algorithm2.standardizeAndPredict(model2, DenseMatrix((2015.0, 1.0)))

    val expectedY2 = DenseVector(-1.366999999999999).toDenseMatrix.t
    assertMatrixEqualWithinMillionth(approxY2, expectedY2)
    assert(approxY2.rows === 1)
    assert(approxY2.cols === 1)
  }

  it should "perfectly predict a simple perfect linear relationship between two predictor variables and one response; also weights,loadings,etc. should match Matlab implementation" in {
    val X = DenseMatrix((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
    val Y = DenseVector(1.0, 2.0, 3.0).toDenseMatrix.t             // (3 x 1) matrix

    val model = DayalMcGregor.Algorithm2.train(X, Y, 1)

    assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((0.5), (0.5)))
    assertMatrixEqualWithinMillionth(model.W, DenseMatrix((0.7071067811865476), (0.7071067811865476)))
    assertMatrixEqualWithinMillionth(model.P, DenseMatrix((0.7071067811865475), (0.7071067811865475)))
    assertMatrixEqualWithinMillionth(model.Q, DenseMatrix((0.7071067811865475)))
    assertMatrixEqualWithinMillionth(model.R, DenseMatrix((0.7071067811865476), (0.7071067811865476)))

    val expectedY = DenseVector(4.0).toDenseMatrix.t
    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((4.0, 4.0)))

    assertMatrixEqualWithinMillionth(approxY, expectedY)

    val model2 = DayalMcGregor.Algorithm2.standardizeAndTrain(X, Y, 1)
    val approxY2 = DayalMcGregor.Algorithm2.standardizeAndPredict(model2, DenseMatrix((4.0, 4.0)))

    assertMatrixEqualWithinMillionth(approxY2, expectedY)
  }

  it should "retain the fitted model when duplicate predictors exhaust the usable rank" in {
    val X = DenseMatrix((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
    val Y = DenseMatrix((1.0), (2.0), (3.0))
    val model = DayalMcGregor.Algorithm2.train(X, Y, 2)
    assert(model.R.cols === 1)
    assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((0.5), (0.5)))
    assertMatrixEqualWithinMillionth(
      DayalMcGregor.Algorithm2.predict(model, DenseMatrix((4.0, 4.0))), DenseMatrix((4.0)))
  }

  it should "stop at one component for proportional predictors in raw and standardized models" in {
    val X = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))
    val Y = DenseMatrix((2.0), (4.0), (6.0), (8.0), (10.0))
    for (requested <- Seq(1, 2, 10)) {
      val model = DayalMcGregor.Algorithm2.train(X, Y, requested)
      assert(model.R.cols === 1)
      assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((0.4), (0.8)))
      assertMatrixEqualWithinMillionth(
        DayalMcGregor.Algorithm2.predict(model, DenseMatrix((9.0, 18.0))), DenseMatrix((18.0)))
      val standardized = DayalMcGregor.Algorithm2.standardizeAndTrain(X, Y, requested)
      assert(standardized.model.R.cols === 1)
      assertMatrixEqualWithinMillionth(standardized.model.Beta, DenseMatrix((0.5), (0.5)))
      assertMatrixEqualWithinMillionth(
        DayalMcGregor.Algorithm2.standardizeAndPredict(standardized, DenseMatrix((9.0, 18.0))), DenseMatrix((18.0)))
    }
  }

  it should "predict the gasoline data and yield the same results as matlab" in {
    val (trainingX, trainingY) = Csv.read("data/gasoline_train.csv", Array(0))    // has two columns: octane, NIR

    val model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 1)

    assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((-1641.8147339889106)))
    assertMatrixEqualWithinMillionth(model.W, DenseMatrix((-1.0)))
    assertMatrixEqualWithinMillionth(model.P, DenseMatrix((-1.0)))
    assertMatrixEqualWithinMillionth(model.Q, DenseMatrix((1641.8147339889106)))
    assertMatrixEqualWithinMillionth(model.R, DenseMatrix((-1.0)))


    val (testingX, testingY) = Csv.read("data/gasoline_test.csv", Array(0))

    val approxY = DayalMcGregor.Algorithm2.predict(model, testingX)

    val expectedApproxY = DenseVector(
      86.415276708772325,       // 88.10 is what's in testingY
      86.523636481215576,       // 87.60 is what's in testingY
      87.663055906603887,       // 88.35 is what's in testingY
      88.877998809755681,       // 85.10 is what's in testingY
      81.471772544731706,       // 85.10 is what's in testingY
      76.974841988336081,       // 84.70 is what's in testingY
      91.211017546753922,       // 87.20 is what's in testingY
      88.153958512066566,       // 86.60 is what's in testingY
      92.452229485649539,       // 89.60 is what's in testingY
      96.546915432217887        // 87.10 is what's in testingY
    ).toDenseMatrix.t

    assertMatrixEqualWithinMillionth(approxY, expectedApproxY)


    val expectedApproxY2 = DenseVector(
      87.22271661912193,        // 88.10 is what's in testingY
      87.2237286040437,         // 87.60 is what's in testingY
      87.23436977882712,        // 88.35 is what's in testingY
      87.24571627643478,        // 85.10 is what's in testingY
      87.1765486403426,         // 85.10 is what's in testingY
      87.13455126608933,        // 84.70 is what's in testingY
      87.2675046184625,         // 87.20 is what's in testingY
      87.23895437718481,        // 86.60 is what's in testingY
      87.27909644574817,        // 89.60 is what's in testingY
      87.317337209307           // 87.10 is what's in testingY
    ).toDenseMatrix.t

    val model2 = DayalMcGregor.Algorithm2.standardizeAndTrain(trainingX, trainingY, 1)
    val approxY2 = DayalMcGregor.Algorithm2.standardizeAndPredict(model2, testingX)

    assertMatrixEqualWithinMillionth(approxY2, expectedApproxY2)
  }

  it should "retain the gasoline regression when more components than predictors are requested" in {
    val (trainingX, trainingY) = Csv.read("data/gasoline_train.csv", Array(0))
    val (testingX, _) = Csv.read("data/gasoline_test.csv", Array(0))
    val baseline = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 1)
    for (requested <- Seq(2, 3, 10)) {
      val model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, requested)
      assert(model.R.cols === 1)
      assertMatrixEqualWithinMillionth(model.Beta, DenseMatrix((-1641.8147339889106)))
      assertMatrixEqualWithinMillionth(DayalMcGregor.Algorithm2.predict(model, testingX),
        DayalMcGregor.Algorithm2.predict(baseline, testingX))
    }
  }

  "VIP algorithm with Algorithm2" should "produce VIP values that when squared, sum to the number of predictor values." in {
    var trainingData = Csv.read("data/artificial13.csv", Array(0))    // has 13 predictor variables
    var trainingX = trainingData._1
    var trainingY = trainingData._2

    var model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 13)
    var vipValues = DayalMcGregor.VIP.computeVIP(model, trainingX, trainingY)
    var sum = vipValues.foldLeft(0.0) { _ + Math.pow(_, 2) }

    assert(sum === (13.0 +- 0.000000001))



    trainingData = Csv.read("data/artificial100.csv", Array(0))    // has 100 predictor variables
    trainingX = trainingData._1
    trainingY = trainingData._2

    model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 13)
    vipValues = DayalMcGregor.VIP.computeVIP(model, trainingX, trainingY)
    sum = vipValues.foldLeft(0.0) { _ + Math.pow(_, 2) }

    assert(sum === (100.0 +- 0.000000001))
  }

  // Independent scikit-learn 1.9.1 NIPALS reference, two standardized components.
  it should "match independent VIP values on the standardized solubility data" in {
    val filePath = "data/solTrainTrans.csv"
    val A = 2

    val (xHeader, yHeader, x, y) = Csv.readWithHeader(filePath, Array(0))    // assumes the left-most column is the response variable, followed by the predictor columns

    val standardizedModel = DayalMcGregor.Algorithm2.standardizeAndTrain(x, y, A)

    val xFactors = standardizedModel.standardizationFactorsX
    val yFactors = standardizedModel.standardizationFactorsY
    val standardizedX = Standardize.centerAndScaleColumnsWithFactors(x, xFactors.meanOfColumns, xFactors.stdDevOfColumns)
    val standardizedY = Standardize.centerAndScaleColumnsWithFactors(y, yFactors.meanOfColumns, yFactors.stdDevOfColumns)
    val vip = DayalMcGregor.VIP.computeVIP(standardizedModel.model, standardizedX, standardizedY)

    val sortedColumnNameAndVipScorePairs = xHeader.zip(vip.toArray).sortWith( (pair1, pair2) => pair1._2 > pair2._2 ).toArray

    assert(sortedColumnNameAndVipScorePairs.take(4).map(_._1) === Array("MolWeight", "NumCarbon","NumNonHAtoms","NumNonHBonds"))

    assert(sortedColumnNameAndVipScorePairs(0)._2 === 3.0742852550529873 +- 0.000000001)
    assert(sortedColumnNameAndVipScorePairs(1)._2 === 2.8047572729743675 +- 0.000000001)
    assert(sortedColumnNameAndVipScorePairs(2)._2 === 2.697662835874127 +- 0.000000001)
    assert(sortedColumnNameAndVipScorePairs(3)._2 === 2.685881154025871 +- 0.000000001)
  }

}
