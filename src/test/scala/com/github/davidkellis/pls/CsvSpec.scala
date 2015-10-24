package com.github.davidkellis.pls

import breeze.linalg._
import org.scalatest._

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
