package com.github.davidkellis.pls

import io.Source

import breeze.linalg._

object Csv {
  // Reads a CSV file into a pair of matrices representing the predictor variables and the response variables
  // The zero-based column indices referenced in <responseColumnIndices> identify the columns in the csv file that hold response variables; all the remaining columns are assumed to be predictor variables
  // Returns a pair of matrices of the form (predictor matrix, response matrix)
  def read(filename: String, responseColumnIndices: Seq[Int]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val responseColumnIndexToColumnInResponseMatrixMap = responseColumnIndices.zipWithIndex.foldLeft(Map.empty[Int, Int]) { (m, pair) => m + (pair._1 -> pair._2) }   // Array(4,5,6) gets transformed to Map(4 -> 0, 5 -> 1, 6 -> 2)
    // val setResponseColumnIndices = responseColumnIndices.toSet
    val file = new java.io.File(filename)
    val lineCount = Source.fromFile(filename).getLines().size

    if (lineCount > 0) {
      val lines = Source.fromFile(filename).getLines()
      var line = lines.next
      var fields = line.split(",").map(_.trim).map(_.toDouble)
      val columnCount = fields.size

      val M = responseColumnIndices.length      // number of response variables
      val N = lineCount                         // number of rows
      val K = columnCount - M                   // number of predictor variables

      val X = DenseMatrix.zeros[Double](N, K)   // X - predictor variables matrix (N x K)
      val Y = DenseMatrix.zeros[Double](N, M)   // Y - response variables matrix (N x M)

      // process the first line that we've already read in
      var row = 0
      var predictorCol = 0
      (0 until columnCount).foreach { c =>
        if (responseColumnIndexToColumnInResponseMatrixMap.contains(c)) {
          Y(row, responseColumnIndexToColumnInResponseMatrixMap(c)) = fields(c)
        } else {
          X(row, predictorCol) = fields(c)
          predictorCol += 1
        }
      }
      row += 1

      // process the remaining lines
      lines.foreach { line =>
        fields = line.split(",").map(_.trim).map(_.toDouble)
        predictorCol = 0
        (0 until columnCount).foreach { c =>
          if (responseColumnIndexToColumnInResponseMatrixMap.contains(c)) {
            Y(row, responseColumnIndexToColumnInResponseMatrixMap(c)) = fields(c)
          } else {
            X(row, predictorCol) = fields(c)
            predictorCol += 1
          }
        }
        row += 1
      }

      (X, Y)
    } else {
      (DenseMatrix.zeros[Double](0, 0), DenseMatrix.zeros[Double](0, 0))
    }
  }

  // Reads a CSV file into a pair of matrices representing the predictor variables and the response variables
  // The zero-based column indices referenced in <responseColumnIndices> identify the columns in the csv file that hold response variables; all the remaining columns are assumed to be predictor variables
  // Returns a 4-tuple of the form (predictor variable names, response variable names, predictor matrix, response matrix)
  def readWithHeader(filename: String, M: Int): (IndexedSeq[String], IndexedSeq[String], DenseMatrix[Double], DenseMatrix[Double]) = {
    val file = new java.io.File(filename)
    val csvMatrix = csvread(file, skipLines = 1)

    val headerLine = Source.fromFile(filename).getLines().take(1).next
    val variableNames = headerLine.split(",").map(trimQuotes _)
    val responseVariableNames = variableNames.take(M)
    val predictorVariableNames = variableNames.drop(M)

    // M is the number of response variables
    val N = csvMatrix.rows       // number of rows
    val K = csvMatrix.cols - M   // number of predictor variables

    val X = DenseMatrix.zeros[Double](N, K)   // X - predictor variables matrix (N x K)
    val Y = DenseMatrix.zeros[Double](N, M)   // Y - response variables matrix (N x M)

    (0 until M).foreach { c =>
      Y(::, c) := csvMatrix(::, c)
    }

    (M until csvMatrix.cols).foreach { c =>
      X(::, c - M) := csvMatrix(::, c)
    }

    (predictorVariableNames, responseVariableNames, X, Y)
  }

  def trimQuotes(str: String): String = str.replaceAll("^(\"|')|(\"|')$", "")
}
