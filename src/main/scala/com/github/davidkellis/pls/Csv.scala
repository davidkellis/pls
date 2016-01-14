package com.github.davidkellis.pls

import language.implicitConversions
import collection.JavaConversions.asScalaIterator

import java.io.{File, FileReader}
import collection.Iterator
import io.Source
import com.opencsv._

import breeze.linalg._

object Csv {
  def countLines(filename: String): Int = {
    val reader = new CSVReader(new FileReader(filename))
    val size = reader.iterator.size
    reader.close()
    size
  }

  // Reads a CSV file into a pair of matrices representing the predictor variables and the response variables
  // The zero-based column indices referenced in <responseColumnIndices> identify the columns in the csv file that hold response variables; all the remaining columns are assumed to be predictor variables
  // Returns a pair of matrices of the form (predictor matrix, response matrix)
  def read(filename: String, responseColumnIndices: Seq[Int]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val responseColumnIndexToColumnInResponseMatrixMap = responseColumnIndices.zipWithIndex.foldLeft(Map.empty[Int, Int]) { (m, pair) => m + (pair._1 -> pair._2) }   // Array(4,5,6) gets transformed to Map(4 -> 0, 5 -> 1, 6 -> 2)
    // val setResponseColumnIndices = responseColumnIndices.toSet
    val lineCount = countLines(filename)

    if (lineCount > 0) {
      val reader = new CSVReader(new FileReader(filename))
      val rows: Iterator[Array[String]] = reader.iterator
      var row = rows.next
      var fields = row.map(_.trim.toDouble)
      val columnCount = fields.size

      val M = responseColumnIndices.length      // number of response variables
      val N = lineCount                         // number of rows
      val K = columnCount - M                   // number of predictor variables

      val X = DenseMatrix.zeros[Double](N, K)   // X - predictor variables matrix (N x K)
      val Y = DenseMatrix.zeros[Double](N, M)   // Y - response variables matrix (N x M)

      // process the first line that we've already read in
      var rowIndex = 0
      var predictorCol = 0
      (0 until columnCount).foreach { c =>
        if (responseColumnIndexToColumnInResponseMatrixMap.contains(c)) {
          Y(rowIndex, responseColumnIndexToColumnInResponseMatrixMap(c)) = fields(c)
        } else {
          X(rowIndex, predictorCol) = fields(c)
          predictorCol += 1
        }
      }
      rowIndex += 1

      // process the remaining lines
      rows.foreach { row =>
        fields = row.map(_.trim.toDouble)
        predictorCol = 0
        (0 until columnCount).foreach { c =>
          if (responseColumnIndexToColumnInResponseMatrixMap.contains(c)) {
            Y(rowIndex, responseColumnIndexToColumnInResponseMatrixMap(c)) = fields(c)
          } else {
            X(rowIndex, predictorCol) = fields(c)
            predictorCol += 1
          }
        }
        rowIndex += 1
      }

      reader.close

      (X, Y)
    } else {
      (DenseMatrix.zeros[Double](0, 0), DenseMatrix.zeros[Double](0, 0))
    }
  }

  // Reads a CSV file into a pair of matrices representing the predictor variables and the response variables
  // The zero-based column indices referenced in <responseColumnIndices> identify the columns in the csv file that hold response variables; all the remaining columns are assumed to be predictor variables
  // Returns a 4-tuple of the form (predictor variable names, response variable names, predictor matrix, response matrix)
  def readWithHeader(filename: String, responseColumnIndices: Seq[Int]): (IndexedSeq[String], IndexedSeq[String], DenseMatrix[Double], DenseMatrix[Double]) = {
    val responseColumnIndexToColumnInResponseMatrixMap = responseColumnIndices.zipWithIndex.foldLeft(Map.empty[Int, Int]) { (m, pair) => m + (pair._1 -> pair._2) }   // Array(4,5,6) gets transformed to Map(4 -> 0, 5 -> 1, 6 -> 2)
    // val setResponseColumnIndices = responseColumnIndices.toSet
    val lineCount = countLines(filename)

    if (lineCount > 0) {
      val reader = new CSVReader(new FileReader(filename))
      val rows: Iterator[Array[String]] = reader.iterator
      var variableNames = rows.next

      val columnCount = variableNames.size
      var fields = Array.empty[Double]

      val M = responseColumnIndices.length      // number of response variables
      val N = lineCount - 1                     // number of rows
      val K = columnCount - M                   // number of predictor variables

      val responseVariableNames = Array.ofDim[String](M)
      val predictorVariableNames = Array.ofDim[String](K)
      val X = DenseMatrix.zeros[Double](N, K)   // X - predictor variables matrix (N x K)
      val Y = DenseMatrix.zeros[Double](N, M)   // Y - response variables matrix (N x M)

      // process the first line - the header row - that we've already read in
      var predictorCol = 0
      (0 until columnCount).foreach { c =>
        if (responseColumnIndexToColumnInResponseMatrixMap.contains(c)) {
          responseVariableNames(responseColumnIndexToColumnInResponseMatrixMap(c)) = variableNames(c)
        } else {
          predictorVariableNames(predictorCol) = variableNames(c)
          predictorCol += 1
        }
      }

      var rowIndex = 0
      // process the remaining lines
      rows.foreach { row =>
        fields = row.map(_.trim.toDouble)
        predictorCol = 0
        (0 until columnCount).foreach { c =>
          if (responseColumnIndexToColumnInResponseMatrixMap.contains(c)) {
            Y(rowIndex, responseColumnIndexToColumnInResponseMatrixMap(c)) = fields(c)
          } else {
            X(rowIndex, predictorCol) = fields(c)
            predictorCol += 1
          }
        }
        rowIndex += 1
      }

      reader.close

      (predictorVariableNames, responseVariableNames, X, Y)
    } else {
      (Array.empty[String], Array.empty[String], DenseMatrix.zeros[Double](0, 0), DenseMatrix.zeros[Double](0, 0))
    }
  }
}
