package com.github.davidkellis.pls

import breeze.linalg._
import breeze.numerics.NaN

import org.scalatest._
import Matchers._

trait TestHelpers {
  def beBetween(x: Double, min: Double, max: Double) = x >= min && x <= max

  def beWithinTolerance(x: Double, base: Double, e: Double) = beBetween(x, base - e, base + e)
}

class PlsSpec extends FlatSpec with TestHelpers {
  "DayalMcGregor's Algorithm2" should "predict interest rates as well as simple linear regression (see http://www.cyclismo.org/tutorial/R/linearLeastSquares.html)" in {
    val X = DenseMatrix((2000.0, 1.0), (2001.0, 1.0), (2002.0, 1.0), (2003.0, 1.0), (2004.0, 1.0))
    val Y = DenseVector(9.34, 8.50, 7.62, 6.93, 6.60).toDenseMatrix.t             // (5 x 1) matrix

    val model = DayalMcGregor.Algorithm2.train(X, Y, 2)

    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((2015.0, 1.0)))

    val expectedY = DenseVector(-1.366999998845131).toDenseMatrix.t
    assert(approxY === expectedY)
    assert(model.Beta === DenseMatrix((-0.7049999999111803), (1419.2079998221832)))
    assert(model.W === DenseMatrix((0.9999998752271115, -4.995417352414563e-4), (4.995455548717368e-4, 0.9999998752290197)))
    assert(model.P === DenseMatrix((0.9999998752497743, -4.994007749060281e-4), (4.995001879364195e-4,  0.9999998752994292)))
    assert(model.Q === DenseMatrix((0.003894750669779322, 1419.2081749231177)))
    assert(model.R === DenseMatrix((0.9999998752271115, -4.995001879364196e-4), (4.995455548717368e-4, 0.9999998752497744)))


    val model2 = DayalMcGregor.Algorithm2.standardizeAndTrain(X, Y, 2)
    val approxY2 = DayalMcGregor.Algorithm2.standardizeAndPredict(model2, DenseMatrix((2015.0, 1.0)))

    val expectedY2 = DenseVector(-1.366999999999999).toDenseMatrix.t
    assert(approxY2 === expectedY2)
    assert(approxY2.rows === 1)
    assert(approxY2.cols === 1)
    assert(beWithinTolerance(approxY2(0,0), expectedY2(0,0), 0.01))
  }

  it should "perfectly predict a simple perfect linear relationship between two predictor variables and one response; also weights,loadings,etc. should match Matlab implementation" in {
    val X = DenseMatrix((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
    val Y = DenseVector(1.0, 2.0, 3.0).toDenseMatrix.t             // (3 x 1) matrix

    val model = DayalMcGregor.Algorithm2.train(X, Y, 1)

    assert(model.Beta === DenseMatrix((0.5), (0.5)))
    assert(model.W === DenseMatrix((0.7071067811865476), (0.7071067811865476)))
    assert(model.P === DenseMatrix((0.7071067811865475), (0.7071067811865475)))
    assert(model.Q === DenseMatrix((0.7071067811865475)))
    assert(model.R === DenseMatrix((0.7071067811865476), (0.7071067811865476)))

    val expectedY = DenseVector(4.0).toDenseMatrix.t
    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((4.0, 4.0)))

    assert(approxY === expectedY)

    val model2 = DayalMcGregor.Algorithm2.standardizeAndTrain(X, Y, 1)
    val approxY2 = DayalMcGregor.Algorithm2.standardizeAndPredict(model2, DenseMatrix((4.0, 4.0)))

    assert(approxY2 === expectedY)
  }

  it should "blow up when the matlab implementation blows up; weights,loadings,etc. should match Matlab implementation" in {
    val X = DenseMatrix((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
    val Y = DenseVector(1.0, 2.0, 3.0).toDenseMatrix.t             // (3 x 1) matrix

    val model = DayalMcGregor.Algorithm2.train(X, Y, 2)

    //assert(model.Beta === DenseMatrix((NaN), (NaN)))   // ScalaTest can't === against NaN, but this is what we want to assert
    assert(model.Beta(0, 0).equals(NaN) && model.Beta(1, 0).equals(NaN))    // this is a work-around for the line above

    assert(model.W === DenseMatrix((0.7071067811865476, 0.7071067811865475), (0.7071067811865476, 0.7071067811865475)))

//    assert(model.P === DenseMatrix((0.7071067811865475, NaN), (0.7071067811865475, NaN)))   // ScalaTest can't === against NaN
    assert(model.P(0,0) === 0.7071067811865475)
    assert(model.P(1,0) === 0.7071067811865475)
    assert(model.P(0,1).equals(NaN))
    assert(model.P(1,1).equals(NaN))

//    assert(model.Q === DenseMatrix((0.7071067811865475, NaN)))  // ScalaTest can't === against NaN
    assert(model.Q(0,0) === 0.7071067811865475)
    assert(model.Q(0,1).equals(NaN))

    assert(model.R === DenseMatrix((0.7071067811865476, 0.0), (0.7071067811865476, 0.0)))

    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((4.0, 4.0)))

    // approxY should be a (1 x 1) matrix containing NaN
    assert(approxY(0,0).equals(NaN))
  }

  it should "perfectly predict a perfect linear relationship between two predictor variables and one response" in {
    val X = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))
    val Y = DenseVector(2.0, 4.0, 6.0, 8.0, 10.0).toDenseMatrix.t             // (5 x 1) matrix

    var model = DayalMcGregor.Algorithm2.train(X, Y, 2)
    var approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((9.0, 18.0)))

    val expectedY = DenseVector(18.0).toDenseMatrix.t
    assert(model.Beta === DenseMatrix((0.4), (0.8)))
    assert(model.W === DenseMatrix((0.447213595499958, 0.4472135954999579), (0.894427190999916, 0.8944271909999159)))
    assert(model.P === DenseMatrix((0.44721359549995787, 3.602879701896397e15), (0.8944271909999157, 7.205759403792794e15)))
    assert(model.Q === DenseMatrix((0.8944271909999157, 0.9309090909090909)))
    assert(model.R === DenseMatrix((0.447213595499958, 5.551115123125783e-17), (0.894427190999916, 1.1102230246251565e-16)))
    assert(approxY === expectedY)


    model = DayalMcGregor.Algorithm2.train(X, Y, 1)
    approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((9.0, 18.0)))

    assert(approxY === expectedY)


    var model2 = DayalMcGregor.Algorithm2.standardizeAndTrain(X, Y, 1)
    var approxY2 = DayalMcGregor.Algorithm2.standardizeAndPredict(model2, DenseMatrix((9.0, 18.0)))

    assert(approxY2 === expectedY)


    model2 = DayalMcGregor.Algorithm2.standardizeAndTrain(X, Y, 2)
    approxY2 = DayalMcGregor.Algorithm2.standardizeAndPredict(model2, DenseMatrix((9.0, 18.0)))

    assert(model2.model.Beta(0,0).equals(NaN))
    assert(model2.model.Beta(1,0).equals(NaN))

    assert(model2.model.W(0,0) === 0.7071067811865475)
    assert(model2.model.W(1,0) === 0.7071067811865475)
    assert(model2.model.W(0,1).equals(NaN))
    assert(model2.model.W(1,1).equals(NaN))

    assert(model2.model.P(0,0) === 0.7071067811865476)
    assert(model2.model.P(1,0) === 0.7071067811865476)
    assert(model2.model.P(0,1).equals(NaN))
    assert(model2.model.P(1,1).equals(NaN))

    assert(model2.model.Q(0,0) === 0.7071067811865476)
    assert(model2.model.Q(0,1).equals(NaN))

    assert(model2.model.R(0,0) === 0.7071067811865475)
    assert(model2.model.R(1,0) === 0.7071067811865475)
    assert(model2.model.R(0,1).equals(NaN))
    assert(model2.model.R(1,1).equals(NaN))


  }

  it should "predict the gasoline data and yield the same results as matlab" in {
    val (trainingX, trainingY) = Csv.read("data/gasoline_train.csv", 1)    // has two columns: octane, NIR

    val model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 1)

    assert(model.Beta === DenseMatrix((-1641.8147339889106)))
    assert(model.W === DenseMatrix((-1.0)))
    assert(model.P === DenseMatrix((-1.0)))
    assert(model.Q === DenseMatrix((1641.8147339889106)))
    assert(model.R === DenseMatrix((-1.0)))


    val (testingX, testingY) = Csv.read("data/gasoline_test.csv", 1)

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

    assert(approxY === expectedApproxY)


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

    assert(approxY2 === expectedApproxY2)
  }

  it should "predict the gasoline data and yield the same error results as matlab" in {
    val (trainingX, trainingY) = Csv.read("data/gasoline_train.csv", 1)    // has two columns: octane, NIR
    val (testingX, testingY) = Csv.read("data/gasoline_test.csv", 1)

    var model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 2)
    var approxY = DayalMcGregor.Algorithm2.predict(model, testingX)

    /**
     * matlab results:
     *
     * >> [beta, w, p, q, r] = pls2(x,y,2)
     *
     * beta =
     *
     *    NaN
     *
     * w =
     *
     *     -1   NaN
     *
     * p =
     *
     *     -1   NaN
     *
     * q =
     *
     *    1.0e+03 *
     *
     *     1.6418       NaN
     *
     * r =
     *
     *     -1   NaN
     *
     */

    assert(model.Beta(0, 0).equals(NaN))

    assert(model.W(0, 0) === -1.0)
    assert(model.W(0, 1).equals(NaN))

    assert(model.P(0, 0) === -1.0)
    assert(model.P(0, 1).equals(NaN))

    assert(model.Q(0, 0) === 1641.8147339889106)
    assert(model.Q(0, 1).equals(NaN))

    assert(model.R(0, 0) === -1.0)
    assert(model.R(0, 1).equals(NaN))

    assert(approxY(0, 0).equals(NaN))
    assert(approxY(1, 0).equals(NaN))
    assert(approxY(2, 0).equals(NaN))
    assert(approxY(3, 0).equals(NaN))
    assert(approxY(4, 0).equals(NaN))
    assert(approxY(5, 0).equals(NaN))
    assert(approxY(6, 0).equals(NaN))
    assert(approxY(7, 0).equals(NaN))
    assert(approxY(8, 0).equals(NaN))
    assert(approxY(9, 0).equals(NaN))


    // 3 components (A = 3)
    model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 3)

    assert(model.Beta(0, 0).equals(NaN))

    assert(model.W(0, 0) === -1.0)
    assert(model.W(0, 1).equals(NaN))
    assert(model.W(0, 2).equals(NaN))

    assert(model.P(0, 0) === -1.0)
    assert(model.P(0, 1).equals(NaN))
    assert(model.P(0, 2).equals(NaN))

    assert(model.Q(0, 0) === 1641.8147339889106)
    assert(model.Q(0, 1).equals(NaN))
    assert(model.Q(0, 2).equals(NaN))

    assert(model.R(0, 0) === -1.0)
    assert(model.R(0, 1).equals(NaN))
    assert(model.R(0, 2).equals(NaN))


    // 10 components (A = 10)
    model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 10)

    assert(model.Beta(0, 0).equals(NaN))

    assert(model.W(0, 0) === -1.0)
    assert(model.W(0, 1).equals(NaN))
    assert(model.W(0, 2).equals(NaN))
    assert(model.W(0, 3).equals(NaN))
    assert(model.W(0, 4).equals(NaN))
    assert(model.W(0, 5).equals(NaN))
    assert(model.W(0, 6).equals(NaN))
    assert(model.W(0, 7).equals(NaN))
    assert(model.W(0, 8).equals(NaN))
    assert(model.W(0, 9).equals(NaN))

    assert(model.P(0, 0) === -1.0)
    assert(model.P(0, 1).equals(NaN))
    assert(model.P(0, 2).equals(NaN))
    assert(model.P(0, 3).equals(NaN))
    assert(model.P(0, 4).equals(NaN))
    assert(model.P(0, 5).equals(NaN))
    assert(model.P(0, 6).equals(NaN))
    assert(model.P(0, 7).equals(NaN))
    assert(model.P(0, 8).equals(NaN))
    assert(model.P(0, 9).equals(NaN))

    assert(model.Q(0, 0) === 1641.8147339889106)
    assert(model.Q(0, 1).equals(NaN))
    assert(model.Q(0, 2).equals(NaN))
    assert(model.Q(0, 3).equals(NaN))
    assert(model.Q(0, 4).equals(NaN))
    assert(model.Q(0, 5).equals(NaN))
    assert(model.Q(0, 6).equals(NaN))
    assert(model.Q(0, 7).equals(NaN))
    assert(model.Q(0, 8).equals(NaN))
    assert(model.Q(0, 9).equals(NaN))

    assert(model.R(0, 0) === -1.0)
    assert(model.R(0, 1).equals(NaN))
    assert(model.R(0, 2).equals(NaN))
    assert(model.R(0, 3).equals(NaN))
    assert(model.R(0, 4).equals(NaN))
    assert(model.R(0, 5).equals(NaN))
    assert(model.R(0, 6).equals(NaN))
    assert(model.R(0, 7).equals(NaN))
    assert(model.R(0, 8).equals(NaN))
    assert(model.R(0, 9).equals(NaN))
  }

  "VIP algorithm" should "produce VIP values that when squared, sum to the number of predictor values." in {
    var trainingData = Csv.read("data/artificial13.csv", 1)    // has 13 predictor variables
    var trainingX = trainingData._1
    var trainingY = trainingData._2

    var model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 13)
    var vipValues = DayalMcGregor.Algorithm2.computeVIP(model, trainingX, trainingY)
    var sum = vipValues.foldLeft(0.0) { _ + Math.pow(_, 2) }

    assert(sum === (13.0 +- 0.000000001))



    trainingData = Csv.read("data/artificial100.csv", 1)    // has 100 predictor variables
    trainingX = trainingData._1
    trainingY = trainingData._2

    model = DayalMcGregor.Algorithm2.train(trainingX, trainingY, 13)
    vipValues = DayalMcGregor.Algorithm2.computeVIP(model, trainingX, trainingY)
    sum = vipValues.foldLeft(0.0) { _ + Math.pow(_, 2) }

    assert(sum === (100.0 +- 0.000000001))
  }

  it should "closely approximate the VIP scores documented in Fig. 6.14 of the Applied Predictive Modeling book." in {
    val filePath = "data/solTrainTrans.csv"
    val A = 2

    val (xHeader, yHeader, x, y) = Csv.readWithHeader(filePath, 1)    // assumes the left-most column is the response variable, followed by the predictor columns

    val standardizedModel = DayalMcGregor.Algorithm2.standardizeAndTrain(x, y, A)

    val vip = DayalMcGregor.Algorithm2.computeVIP(standardizedModel.model, x, y)

    val sortedColumnNameAndVipScorePairs = xHeader.zip(vip.toArray).sortWith( (pair1, pair2) => pair1._2 > pair2._2 ).toArray

    assert(sortedColumnNameAndVipScorePairs.take(4).map(_._1) === Array("MolWeight", "NumCarbon","NumNonHAtoms","NumNonHBonds"))

    assert(sortedColumnNameAndVipScorePairs(0)._2 === 2.745480124550425 +- 0.000000001)
    assert(sortedColumnNameAndVipScorePairs(1)._2 === 2.453165729632949 +- 0.000000001)
    assert(sortedColumnNameAndVipScorePairs(2)._2 === 2.3594861202847586 +- 0.000000001)
    assert(sortedColumnNameAndVipScorePairs(3)._2 === 2.312747057203807 +- 0.000000001)

  }
}
