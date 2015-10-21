import org.scalatest._

import breeze.linalg._

import com.github.davidkellis.pls._

class PlsSpec extends FlatSpec {
  "DayalMcGregor's Algorithm2" should "predict interest rates as well as simple linear regression (see http://www.cyclismo.org/tutorial/R/linearLeastSquares.html)" in {
    val X = DenseMatrix((2000.0, 1.0), (2001.0, 1.0), (2002.0, 1.0), (2003.0, 1.0), (2004.0, 1.0))
    val Y = DenseVector(9.34, 8.50, 7.62, 6.93, 6.60).toDenseMatrix.t             // (5 x 1) matrix

    val model = DayalMcGregor.Algorithm2.train(X, Y, 2)

    val expectedY = DenseVector(-1.367).toDenseMatrix.t
    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((2015.0, 1.0)))

    assert(approxY === expectedY)
  }

  it should "perfectly predict a perfect linear relationship between two predictor variables and one response" in {
    val X = DenseMatrix((1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0))
    val Y = DenseVector(2.0, 4.0, 6.0, 8.0, 10.0).toDenseMatrix.t             // (5 x 1) matrix

    val model = DayalMcGregor.Algorithm2.train(X, Y, 2)

    val expectedY = DenseVector(18.0).toDenseMatrix.t
    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((9.0, 18.0)))

    assert(approxY === expectedY)
  }

  it should "predict the gasoline data" in {
    val (X, Y) = Csv.read("data/gasoline_train.csv", 1)    // has two columns: octane, NIR
    
  }
}
