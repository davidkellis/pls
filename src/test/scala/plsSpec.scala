import org.scalatest._

import breeze.linalg._

import com.github.davidkellis.pls._

class PlsSpec extends FlatSpec {
  "pls" should "predict interest rates as well as simple linear regression (see http://www.cyclismo.org/tutorial/R/linearLeastSquares.html)" in {
    val X = DenseMatrix((2000.0, 1.0), (2001.0, 1.0), (2002.0, 1.0), (2003.0, 1.0), (2004.0, 1.0))
    val Y = DenseVector(9.34, 8.50, 7.62, 6.93, 6.60).toDenseMatrix.t             // (5 x 1) matrix

    val model = DayalMcGregor.Algorithm2.train(X, Y, 2)

    val expectedY = DenseVector(-1.367).toDenseMatrix.t
    val approxY = DayalMcGregor.Algorithm2.predict(model, DenseMatrix((2015.0, 1.0)))

    assert(approxY === expectedY)
  }
}
