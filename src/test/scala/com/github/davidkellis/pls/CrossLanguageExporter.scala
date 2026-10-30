package com.github.davidkellis.pls

import breeze.linalg._
import java.io.PrintWriter
import scala.io.Source
import scala.util.parsing.json.JSON

/** Runs the actual Scala APIs on the same JSON datasets as the website port. */
object CrossLanguageExporter {
  private def matrix(value: Any): DenseMatrix[Double] = {
    val rows = value.asInstanceOf[List[List[Double]]]
    DenseMatrix.tabulate[Double](rows.length, rows.head.length)((i, j) => rows(i)(j))
  }

  private def rows(m: DenseMatrix[Double]): Seq[Seq[Double]] =
    (0 until m.rows).map(i => (0 until m.cols).map(j => m(i, j)))

  private def json(value: Any): String = value match {
    case m: Map[_, _] => m.toSeq.sortBy(_._1.toString).map { case (k, v) => json(k.toString) + ":" + json(v) }.mkString("{", ",", "}")
    case seq: Seq[_] => seq.map(json).mkString("[", ",", "]")
    case s: String => "\"" + s.flatMap {
      case '"' => "\\\""
      case '\\' => "\\\\"
      case '\n' => "\\n"
      case '\r' => "\\r"
      case '\t' => "\\t"
      case c if c < ' ' => "\\u%04x".format(c.toInt)
      case c => c.toString
    } + "\""
    case n: Double =>
      require(!n.isNaN && !n.isInfinity, "A nonfinite Scala result cannot pass parity")
      n.toString
    case n: Int => n.toString
    case other => throw new IllegalArgumentException("Unsupported JSON value: " + other)
  }

  def main(args: Array[String]): Unit = {
    require(args.length == 2, "Expected input and output JSON paths")
    val source = Source.fromFile(args(0), "UTF-8")
    val input = try JSON.parseFull(source.mkString).get.asInstanceOf[Map[String, Any]] finally source.close()
    val cases = input("cases").asInstanceOf[List[Map[String, Any]]]
    val algorithms: Seq[(String, PlsModel[DayalMcGregor.Model])] = Seq(
      "Algorithm1" -> DayalMcGregor.Algorithm1, "Algorithm2" -> DayalMcGregor.Algorithm2)
    val results = for (dataset <- cases; (name, algorithm) <- algorithms) yield {
      val x = matrix(dataset("x")); val y = matrix(dataset("y")); val evaluation = matrix(dataset("evaluation"))
      val requested = dataset("components").asInstanceOf[Double].toInt
      val (model, coefficients, meansX, meansY, prediction, trainingPrediction) =
        if (dataset("scale").asInstanceOf[Boolean]) {
          val fit = algorithm.standardizeAndTrain(x, y, requested)
          val beta = fit.model.Beta.copy
          for (i <- 0 until x.cols; j <- 0 until y.cols) {
            beta(i, j) *= fit.standardizationFactorsY.stdDevOfColumns(0, j) /
              fit.standardizationFactorsX.stdDevOfColumns(0, i)
          }
          (fit.model, beta, fit.standardizationFactorsX.meanOfColumns,
            fit.standardizationFactorsY.meanOfColumns,
            algorithm.standardizeAndPredict(fit, evaluation), algorithm.standardizeAndPredict(fit, x))
        } else {
          // The raw Scala API expects preprocessing by its caller. Centering
          // here matches fitPLS(..., { scale: false }), without autoscaling.
          val (centeredX, mx) = Standardize.meanCenterColumns2(x)
          val (centeredY, my) = Standardize.meanCenterColumns2(y)
          val fit = algorithm.train(centeredX, centeredY, requested)
          val predicted = algorithm.predict(fit, Standardize.meanCenterColumnsWithMeans(evaluation, mx))
          val training = algorithm.predict(fit, centeredX)
          (fit, fit.Beta, mx, my, Standardize.denormalizeMeanCenteredColumns(predicted, my),
            Standardize.denormalizeMeanCenteredColumns(training, my))
        }
      val intercept = meansY - meansX * coefficients
      Map[String, Any]("case" -> dataset("name"), "algorithm" -> name,
        "components" -> model.R.cols, "coefficients" -> rows(coefficients),
        "intercept" -> (0 until intercept.cols).map(j => intercept(0, j)),
        "predictions" -> rows(prediction), "trainingPredictions" -> rows(trainingPrediction))
    }
    val out = new PrintWriter(args(1), "UTF-8")
    try out.println(json(Map("scala_version" -> scala.util.Properties.versionNumberString,
      "java_version" -> System.getProperty("java.version"), "results" -> results))) finally out.close()
    println("Exported " + results.size + " Scala fits from " + cases.size + " shared cases")
  }
}
