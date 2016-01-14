package com.github.davidkellis.pls

import breeze.linalg._

object Main {
  def main(args: Array[String]): Unit = {
    if (args.length >= 2) {
      // train and print model
      val filePath = args(0)    // first arg is file path
      val A = args(1).toInt     // second arg is A - the number of components to try to fit the model to
      var predictorColumnIndices = args.drop(2).map(_.toInt)
      predictorColumnIndices = if (predictorColumnIndices.isEmpty)
                                 Array(0)     // assumes the left-most column is the response variable, followed by the predictor columns
                               else
                                 predictorColumnIndices

      val (x, y) = Csv.read(filePath, predictorColumnIndices)
      // val (xHeader, yHeader, x, y) = Csv.readWithHeader(filePath, predictorColumnIndices)


      run(x, y, A)
      // runStandardized(x, y, A)

      // runWithHeader(x, y, A, xHeader, yHeader)
      // runStandardizedWithHeader(x, y, A, xHeader, yHeader)


      // val standardizedModel = DayalMcGregor.Algorithm1.standardizeAndTrain(x, y, A)
      // println(standardizedModel.model.Beta.toString(1000000, 1000))
    }
  }

  def run(x: DenseMatrix[Double], y: DenseMatrix[Double], A: Int): Unit = {
    val model = DayalMcGregor.Algorithm1.train(x, y, A)

    println("model")
    println("Beta")
    println(model.Beta.toString(1000000, 1000))
    println("W")
    println(model.W.toString(1000000, 1000))
    println("P")
    println(model.P.toString(1000000, 1000))
    println("Q")
    println(model.Q.toString(1000000, 1000))
    println("R")
    println(model.R.toString(1000000, 1000))

    val vip = DayalMcGregor.VIP.computeVIP(model, x, y)
    println("vip")
    println(vip.toArray)
  }

  def runStandardized(x: DenseMatrix[Double], y: DenseMatrix[Double], A: Int): Unit = {
    val standardizedModel = DayalMcGregor.Algorithm1.standardizeAndTrain(x, y, A)

    println("model")
    println(standardizedModel.model)

    println("beta")
    println(standardizedModel.model.Beta.toString(1000000, 1000))

    val vip = DayalMcGregor.VIP.computeVIP(standardizedModel.model, x, y)
    println("vip")
    println(vip.toArray)
  }

  def runWithHeader(x: DenseMatrix[Double], y: DenseMatrix[Double], A: Int, xHeader: IndexedSeq[String], yHeader: IndexedSeq[String]): Unit = {
    val model = DayalMcGregor.Algorithm1.train(x, y, A)

    println("model")
    println("Beta")
    println(model.Beta.toString(1000000, 1000))
    println("W")
    println(model.W.toString(1000000, 1000))
    println("P")
    println(model.P.toString(1000000, 1000))
    println("Q")
    println(model.Q.toString(1000000, 1000))
    println("R")
    println(model.R.toString(1000000, 1000))

    val vip = DayalMcGregor.VIP.computeVIP(model, x, y)
    println("vip")
    println(xHeader.zip(vip.toArray).sortWith( (pair1, pair2) => pair1._2 > pair2._2 ))
  }

  def runStandardizedWithHeader(x: DenseMatrix[Double], y: DenseMatrix[Double], A: Int, xHeader: IndexedSeq[String], yHeader: IndexedSeq[String]): Unit = {
    val standardizedModel = DayalMcGregor.Algorithm1.standardizeAndTrain(x, y, A)

    println("model")
    println(standardizedModel.model)

    println("beta")
    println(standardizedModel.model.Beta.toString(1000000, 1000))

    val vip = DayalMcGregor.VIP.computeVIP(standardizedModel.model, x, y)
    println("vip")
    println(xHeader.zip(vip.toArray).sortWith( (pair1, pair2) => pair1._2 > pair2._2 ))
  }

}
