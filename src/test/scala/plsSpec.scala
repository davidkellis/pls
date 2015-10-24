import org.scalatest._

import breeze.linalg._
import breeze.numerics.NaN

import com.github.davidkellis.pls._

trait TestHelpers {
  def beBetween(x: Double, min: Double, max: Double) = x >= min && x <= max

  def beWithinTolerance(x: Double, base: Double, e: Double) = beBetween(x, base - e, base + e)
}

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

    val expectedY2 = DenseVector(-1.3670000000000009).toDenseMatrix.t
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

    val model = DayalMcGregor.Algorithm2.train(X, Y, 2)

    val expectedY = DenseVector(18.0).toDenseMatrix.t
    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((9.0, 18.0)))

    assert(model.Beta === DenseMatrix((0.4), (0.8)))
    assert(model.W === DenseMatrix((0.447213595499958, 0.4472135954999579), (0.894427190999916, 0.8944271909999159)))
    assert(model.P === DenseMatrix((0.44721359549995787, 3.602879701896397e15), (0.8944271909999157, 7.205759403792794e15)))
    assert(model.Q === DenseMatrix((0.8944271909999157, 0.9309090909090909)))
    assert(model.R === DenseMatrix((0.447213595499958, 5.551115123125783e-17), (0.894427190999916, 1.1102230246251565e-16)))
    assert(approxY === expectedY)
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
      86.415276708772325,
      86.523636481215576,
      87.663055906603887,
      88.877998809755681,
      81.471772544731706,
      76.974841988336081,
      91.211017546753922,
      88.153958512066566,
      92.452229485649539,
      96.546915432217887
    ).toDenseMatrix.t

    assert(approxY === expectedApproxY)
  }
}
