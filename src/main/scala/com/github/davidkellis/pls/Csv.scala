package com.github.davidkellis.pls

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
}
