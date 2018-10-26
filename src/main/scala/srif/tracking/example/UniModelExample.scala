/*
 *    Copyright 2018, Author: Weidong Chen (chen3000cn@gmail.com)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package srif.tracking.example

import java.io.PrintWriter

import breeze.linalg.{DenseMatrix, DenseVector, det}
import srif.tracking.TargetModel.ConstantVelocityModel
import srif.tracking.example.miscTools.UniModel.calculateEstimationError
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator
import srif.tracking.squarerootkalman.SquareRootInformationFilter.FilterResult
import srif.tracking.squarerootkalman.SquareRootInformationSmoother.SmoothResult
import srif.tracking.squarerootkalman.{BackwardSquareRootInformationFilter, ForwardBackwardSquareRootInformationSmoother, SquareRootInformationFilter, SquareRootInformationSmoother}
import srif.tracking.{FactoredGaussianDistribution, GaussianDistribution, TargetModel}

object UniModelExample {
  def main(args: Array[String]): Unit = {

    val numberOfTestCases: Int = 10
    val numOfEventsPerTestCase: Int = 1000
    val observationStd: Double = 100.0

    val seeds: Vector[Int] = Vector.range(0, numberOfTestCases)

    val model: TargetModel = ConstantVelocityModel(1.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 5.0, 0.0, 5.0)

    seeds.foreach(seed => {

      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEventsPerTestCase, observationStd, seed)

      val observationLst: Vector[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)
      val invStateTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateInvStateTransitionMatrix)

      val forwardInformationFilter = new SquareRootInformationFilter(model, false)
      val backwardInformationFilter = new BackwardSquareRootInformationFilter(model, false)
      val informationSmoother = new SquareRootInformationSmoother(model, false)
      val forwardBackwardSmoother = new ForwardBackwardSquareRootInformationSmoother(model, false)

      val forwardFilterResult: Vector[FilterResult] = forwardInformationFilter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)
      outputSampleResult("ForwardInformationFilter", seed, observationVectorLst, forwardFilterResult.map(_.updatedStateEstimation), stateLst, 1, 0)

      val backwardFilterResult: Vector[FilterResult] = backwardInformationFilter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)
      outputSampleResult("BackwardInformationFilter", seed, observationVectorLst, backwardFilterResult.map(_.updatedStateEstimation), stateLst, 0, 1)

      val smootherResult: Vector[SmoothResult] = informationSmoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)
      outputSampleResult("InformationSmoother", seed, observationVectorLst, smootherResult.map(_.smoothedStateEstimation), stateLst, 0, 0)

      val forwardBackwardSmootherResult: Vector[SmoothResult] = forwardBackwardSmoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)
      outputSampleResult("ForwardBackwardSmoother", seed, observationVectorLst, forwardBackwardSmootherResult.map(_.smoothedStateEstimation), stateLst, 0, 0)

    })

  }

  def outputSampleResult(estimatorName: String,
                         seed: Int,
                         observationVectorLst: Vector[DenseVector[Double]],
                         estimatedState: Vector[FactoredGaussianDistribution],
                         trueStates: Vector[DenseVector[Double]],
                         dropLeft: Int, dropRight: Int) = {

    val error = calculateEstimationError(estimatedState, trueStates, dropLeft, dropRight)
    println(s"$estimatorName, \tState MSE: $error.")

    writeToCSV(trueStates, observationVectorLst, estimatedState, s"$sampleResultFolder/${estimatorName}_result_$seed.csv")

  }

  def writeToCSV(states: Vector[DenseVector[Double]],
                 observationVectorLst: Vector[DenseVector[Double]],
                 estimatedStates: Vector[FactoredGaussianDistribution],
                 fileName: String): Unit = {

    val headers = Seq("STATE_X", "STATE_DOT_X", "STATE_Y", "STATE_DOT_Y", "OBS_X", "OBS_Y",
      "EST_X", "EST_DOT_X", "EST_Y", "EST_DOT_Y", "MSE")

    val records: Seq[Seq[String]] = Vector.range(0, states.length).map(idx => {

      val stateXY = states(idx)
      val observationXY = observationVectorLst(idx)

      if (det(estimatedStates(idx).R) == 0)
        Seq(stateXY(0).toString, stateXY(1).toString, stateXY(2).toString, stateXY(3).toString,
          observationXY(0).toString, observationXY(1).toString,
          "", "", "", "", "")
      else {

        val estimatedXY = estimatedStates(idx).toGaussianDistribution.m

        val errorVector: DenseVector[Double] = estimatedXY - stateXY

        Seq(stateXY(0), stateXY(1), stateXY(2), stateXY(3),
          observationXY(0), observationXY(1),
          estimatedXY(0), estimatedXY(1), estimatedXY(2), estimatedXY(3),
          errorVector.t * errorVector).map(_.toString)
      }
    })

    val allRows: Seq[Seq[String]] = Seq(headers) ++ records

    val csv: String = allRows.map(_.mkString(",")).mkString("\n")

    new PrintWriter(fileName) {
      write(csv)
      close()
    }
  }

}
