package com.github.davidkellis.pls

import breeze.linalg._
import org.scalatest._

class CsvSpec extends FlatSpec {
  "Csv" should "read a CSV of numbers in as a pair of predictor and response matrices" in {
    val (x, y) = Csv.read("data/test.csv", Array(0))    // has 3 columns
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

  it should "properly identify the predictor and response variables" in {
    val (x, y) = Csv.read("data/test.csv", Array(1))    // has 3 columns
    // test.csv looks like this:
    // 1.0,2.0,3.0
    // 10.0,20.0,30.0
    // 100.0,200.0,300.0

    assert(x.cols === 2)
    assert(x.rows === 3)
    assert(x === DenseMatrix((1.0, 3.0), (10.0, 30.0), (100.0, 300.0)))

    assert(y.cols === 1)
    assert(y.rows === 3)
    assert(y === DenseVector(2.0, 20.0, 200.0).toDenseMatrix.t)
  }

  it should "read a CSV file with column headers properly" in {
    val (predictorNames, responseNames, x, y) = Csv.readWithHeader("data/test2.csv", Array(0))    // has 3 columns
    // test2.csv looks like this:
    // A,B,C
    // 1.0,2.0,3.0
    // 10.0,20.0,30.0
    // 100.0,200.0,300.0

    assert(predictorNames === Array("B", "C"))
    assert(responseNames === Array("A"))

    assert(x.cols === 2)
    assert(x.rows === 3)
    assert(x === DenseMatrix((2.0, 3.0), (20.0, 30.0), (200.0, 300.0)))

    assert(y.cols === 1)
    assert(y.rows === 3)
    assert(y === DenseVector(1.0, 10.0, 100.0).toDenseMatrix.t)
  }

  it should "properly identify the predictor and response variables with 1 response variable" in {
    val (predictorNames, responseNames, x, y) = Csv.readWithHeader("data/test2.csv", Array(1))    // has 3 columns
    // test2.csv looks like this:
    // A,B,C
    // 1.0,2.0,3.0
    // 10.0,20.0,30.0
    // 100.0,200.0,300.0

    assert(predictorNames === Array("A", "C"))
    assert(responseNames === Array("B"))

    assert(x.cols === 2)
    assert(x.rows === 3)
    assert(x === DenseMatrix((1.0, 3.0), (10.0, 30.0), (100.0, 300.0)))

    assert(y.cols === 1)
    assert(y.rows === 3)
    assert(y === DenseVector(2.0, 20.0, 200.0).toDenseMatrix.t)
  }

  it should "properly identify the predictor and response variables with 2 response variables" in {
    val (predictorNames, responseNames, x, y) = Csv.readWithHeader("data/test2.csv", Array(1, 2))    // has 3 columns
    // test2.csv looks like this:
    // A,B,C
    // 1.0,2.0,3.0
    // 10.0,20.0,30.0
    // 100.0,200.0,300.0

    assert(predictorNames === Array("A"))
    assert(responseNames === Array("B", "C"))

    assert(x.cols === 1)
    assert(x.rows === 3)
    assert(x === DenseMatrix((1.0), (10.0), (100.0)))

    assert(y.cols === 2)
    assert(y.rows === 3)
    assert(y === DenseMatrix((2.0, 3.0), (20.0, 30.0), (200.0, 300.0)))
  }
}
