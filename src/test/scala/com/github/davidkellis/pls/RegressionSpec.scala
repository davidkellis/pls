package com.github.davidkellis.pls

import breeze.linalg._
import org.scalatest._
import scala.io.Source
import scala.util.parsing.json.JSON

class RegressionSpec extends FlatSpec with TestHelpers {
  private val algorithms: Seq[(String, PlsModel[DayalMcGregor.Model])] = Seq(
    "Algorithm1" -> DayalMcGregor.Algorithm1,
    "Algorithm2" -> DayalMcGregor.Algorithm2
  )

  private def matrix(value: Any): DenseMatrix[Double] = {
    val rows = value.asInstanceOf[List[List[Double]]]
    DenseMatrix.tabulate[Double](rows.length, rows.head.length)((i, j) => rows(i)(j))
  }

  private def assertClose(actual: DenseMatrix[Double], expected: DenseMatrix[Double]): Unit = {
    assert(actual.rows == expected.rows && actual.cols == expected.cols)
    for (i <- 0 until actual.rows; j <- 0 until actual.cols) {
      val value = actual(i, j)
      assert(!value.isNaN && !value.isInfinity, s"Nonfinite prediction at ($i,$j)")
      assert(math.abs(value - expected(i, j)) <= 2e-7 * math.max(1.0, math.abs(expected(i, j))),
        s"($i,$j): $value differed from ${expected(i,j)}")
    }
  }

  private val source = Source.fromInputStream(getClass.getResourceAsStream("/pls-reference.json"), "UTF-8")
  private val cases = try {
    JSON.parseFull(source.mkString).get.asInstanceOf[Map[String, Any]]("cases")
      .asInstanceOf[List[Map[String, Any]]]
  } finally source.close()

  for ((name, algorithm) <- algorithms) {
    for (reference <- cases) {
      name should ("match independent NIPALS predictions and coefficients: " + reference("name")) in {
        val x = matrix(reference("x"))
        val y = matrix(reference("y"))
        val evaluation = matrix(reference("evaluation"))
        val requested = reference("components").asInstanceOf[Double].toInt
        val (prediction, coefficients) = if (reference("scale").asInstanceOf[Boolean]) {
          val fitted = algorithm.standardizeAndTrain(x, y, requested)
          val coefficients = fitted.model.Beta.copy
          for (i <- 0 until x.cols; j <- 0 until y.cols) {
            coefficients(i,j) *= fitted.standardizationFactorsY.stdDevOfColumns(0,j) /
              fitted.standardizationFactorsX.stdDevOfColumns(0,i)
          }
          (algorithm.standardizeAndPredict(fitted, evaluation), coefficients)
        } else {
          val (centeredX, meansX) = Standardize.meanCenterColumns2(x)
          val (centeredY, meansY) = Standardize.meanCenterColumns2(y)
          val fitted = algorithm.train(centeredX, centeredY, requested)
          val predicted = algorithm.predict(fitted, Standardize.meanCenterColumnsWithMeans(evaluation, meansX))
          (Standardize.denormalizeMeanCenteredColumns(predicted, meansY), fitted.Beta)
        }
        assertClose(prediction, matrix(reference("plsPredictions")))
        assertClose(coefficients, matrix(reference("plsCoefficients")))
      }
    }

    name should "restore positive, negative and zero constant responses exactly" in {
      val x = DenseMatrix((1.0, 0.0), (2.0, 1.0), (3.0, 0.0), (4.0, 1.0))
      for (constant <- Seq(9.5, -2.0, 0.0)) {
        val y = DenseMatrix.fill[Double](4, 1)(constant)
        val model = algorithm.standardizeAndTrain(x, y, 10)
        assert(model.model.R.cols == 0)
        assertClose(algorithm.standardizeAndPredict(model, DenseMatrix((10.0, 2.0))), DenseMatrix((constant)))
        val vip = DayalMcGregor.VIP.computeVIP(model.model,
          Standardize.centerAndScaleColumns(x), Standardize.centerAndScaleColumns(y))
        assert(vip == DenseVector.zeros[Double](x.cols))
      }
    }

    name should "use the response mean when every predictor is constant" in {
      val x = DenseMatrix((7.0, 0.0), (7.0, 0.0), (7.0, 0.0), (7.0, 0.0))
      val y = DenseMatrix((1.0), (3.0), (5.0), (7.0))
      val model = algorithm.standardizeAndTrain(x, y, 4)
      assert(model.model.R.cols == 0)
      assertClose(algorithm.standardizeAndPredict(model, DenseMatrix((7.0, 0.0))), DenseMatrix((4.0)))
    }

    name should "stop when predictor-response covariance is zero" in {
      val x = DenseMatrix((-1.0), (0.0), (1.0))
      val y = DenseMatrix((1.0), (-2.0), (1.0))
      val model = algorithm.train(x, y, 3)
      assert(model.R.cols == 0)
      assertClose(algorithm.predict(model, DenseMatrix((4.0))), DenseMatrix((0.0)))
    }

    name should "fit a rank-two, multiple-response problem without spurious extra components" in {
      val pairs = Seq((-2.0, 1.0), (0.0, 3.0), (1.0, -1.0), (4.0, 2.0), (-1.0, -2.0), (2.0, 5.0))
      val x = DenseMatrix.tabulate[Double](pairs.size, 5) { (i, j) =>
        val (a, b) = pairs(i)
        Seq(a, b, a + 2*b, 3*a - b, 0.0)(j)
      }
      val y = DenseMatrix.tabulate[Double](pairs.size, 2) { (i, j) =>
        val (a, b) = pairs(i)
        Seq(2*a - b + 5, 3*a + 2*b - 7)(j)
      }
      val model = algorithm.standardizeAndTrain(x, y, 10)
      assert(model.model.R.cols == 2)
      assertClose(algorithm.standardizeAndPredict(model, DenseMatrix((3.0, 8.0, 19.0, 1.0, 0.0))), DenseMatrix((3.0, 18.0)))
      val t = Standardize.centerAndScaleColumns(x) * model.model.R
      assert(math.abs(t(::, 0).t * t(::, 1)) < 1e-10 * norm(t(::, 0)) * norm(t(::, 1)))
    }

    name should "provide finite VIP scores through both public helper names" in {
      val x = DenseMatrix((-2.0, 1.0), (0.0, 3.0), (1.0, -1.0), (4.0, 2.0))
      val y = DenseMatrix((-5.0), (-3.0), (3.0), (6.0))
      val model = algorithm.train(x, y, 2)
      val vip = DayalMcGregor.VIP.computeVIP(model, x, y)
      assertVectorEqualWithinMillionth(DayalMcGregor.VIP.computeVIP2(model, x, y), vip)
      assert(math.abs(vip.toArray.map(v => v*v).sum - x.cols) < 1e-10)
      intercept[IllegalArgumentException] {
        DayalMcGregor.VIP.computeVIP(model, x, DenseMatrix.horzcat(y, y))
      }
    }

    name should "reject invalid training and prediction inputs" in {
      val x = DenseMatrix((1.0), (2.0))
      val y = DenseMatrix((2.0), (4.0))
      for (components <- Seq(0, -1)) intercept[IllegalArgumentException] { algorithm.train(x, y, components) }
      intercept[IllegalArgumentException] { algorithm.train(x, DenseMatrix((1.0)), 1) }
      intercept[IllegalArgumentException] { algorithm.train(DenseMatrix.zeros[Double](0, 1), DenseMatrix.zeros[Double](0, 1), 1) }
      intercept[IllegalArgumentException] { algorithm.train(DenseMatrix((Double.NaN), (2.0)), y, 1) }
      intercept[IllegalArgumentException] { algorithm.train(x, DenseMatrix((Double.PositiveInfinity), (2.0)), 1) }
      intercept[IllegalArgumentException] { algorithm.standardizeAndTrain(DenseMatrix((1.0)), DenseMatrix((2.0)), 1) }
      val model = algorithm.train(x, y, 1)
      intercept[IllegalArgumentException] { algorithm.predict(model, DenseMatrix((1.0, 2.0))) }
      intercept[IllegalArgumentException] { algorithm.predict(model, DenseMatrix((Double.NaN))) }
    }

    name should "preserve inputs and use training statistics for predictions" in {
      val x = DenseMatrix((1.0, 2.0), (2.0, 1.0), (3.0, 4.0), (4.0, 3.0))
      val y = DenseMatrix((8.0), (10.0), (16.0), (18.0))
      val beforeX = x.copy; val beforeY = y.copy
      val model = algorithm.standardizeAndTrain(x, y, 2)
      val first = DenseMatrix((5.0, 6.0))
      val single = algorithm.standardizeAndPredict(model, first)
      val batch = algorithm.standardizeAndPredict(model, DenseMatrix((5.0, 6.0), (10000.0, -10000.0)))
      assertClose(single, batch(0 to 0, ::).copy)
      assert(x == beforeX && y == beforeY)
    }
  }

  "Matrix assertion helper" should "reject errors in single responses and in the final response column" in {
    intercept[org.scalatest.exceptions.TestFailedException] {
      assertMatrixEqualWithinMillionth(DenseMatrix((123.0)), DenseMatrix((999.0)))
    }
    intercept[org.scalatest.exceptions.TestFailedException] {
      assertMatrixEqualWithinMillionth(DenseMatrix((1.0, 123.0)), DenseMatrix((1.0, 999.0)))
    }
  }
}
