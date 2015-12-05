package com.github.davidkellis.pls

object Main {
  def main(args: Array[String]) {
    if (args.length == 2) {
      // train and print model
      val filePath = args(0)    // first arg is file path
      val A = args(1).toInt     // second arg is A - the number of components to try to fit the model to


      // val (x, y) = Csv.read(filePath, 1)    // assumes the left-most column is the response variable, followed by the predictor columns
      val (xHeader, yHeader, x, y) = Csv.readWithHeader(filePath, 1)    // assumes the left-most column is the response variable, followed by the predictor columns


      // val model = DayalMcGregor.Algorithm2.train(x, y, A)
      val standardizedModel = DayalMcGregor.Algorithm2.standardizeAndTrain(x, y, A)

      println("model")
      // println(model)
      println(standardizedModel.model)

      println("beta")
      // println(model.Beta.toString(1000000, 1000))
      println(standardizedModel.model.Beta.toString(1000000, 1000))

      println("vip")
      // println(DayalMcGregor.Algorithm2.computeVIP(model, x, y))
      println(DayalMcGregor.Algorithm2.computeVIP(standardizedModel.model, x, y))


      // val standardizedModel = DayalMcGregor.Algorithm2.standardizeAndTrain(x, y, A)
      // println(standardizedModel.model.Beta.toString(1000000, 1000))
    }
  }
}
