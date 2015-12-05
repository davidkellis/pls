package com.github.davidkellis.pls

import io.Source

import breeze.linalg._

object Csv {
  // Reads a CSV file into a pair of matrices representing the predictor variables and the response variables
  // Assumes the leftmost M columns in the CSV are the response variables, followed by the remaining columns holding the predictor variables
  // Returns a pair of matrices of the form (predictor matrix, response matrix)
  def read(filename: String, M: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val file = new java.io.File(filename)
    val csvMatrix = csvread(file)

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

    (X, Y)
  }

  // Reads a CSV file into a pair of matrices representing the predictor variables and the response variables
  // Assumes the leftmost M columns in the CSV are the response variables, followed by the remaining columns holding the predictor variables
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
