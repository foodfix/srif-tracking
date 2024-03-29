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
import srif.tracking.TargetModel.{ConstantPositionModel, ConstantVelocityModel}
import srif.tracking.example.miscTools.MultipleModel.calculateEstimationError
import srif.tracking.example.sampleDataGeneration.MultipleModelTestDataGenerator
import srif.tracking.multipleModel._
import srif.tracking.squarerootkalman.{SquareRootInformationFilter, SquareRootInformationSmoother}
import srif.tracking.{FactoredGaussianDistribution, GaussianDistribution, TargetModel}

object MultipleModelExample {

  def main(args: Array[String]): Unit = {

    val numberOfTestCases: Int = 10
    val numOfEventsPerTestCase: Int = 1000
    val observationStd: Double = 100.0
    val modelSwitchingProbabilityPerUnitTime: Double = 1e-3

    val seeds: Vector[Int] = Vector.range(0, numberOfTestCases)

    // Define a constant velocity model
    val model_0: TargetModel = ConstantVelocityModel(0.5)

    //Define a stationary model
    val model_1: TargetModel = ConstantPositionModel(0.0)

    //initial state for models
    val initialStateLst: Vector[DenseVector[Double]] = Vector(DenseVector(0.0, 5.0, 0.0, 5.0), DenseVector(0.0, 0.0))

    val targetModelLst: Vector[TargetModel] = Vector(model_0, model_1)

    //Define the project matrix between constant velocity model and stationary model
    val modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]] = DenseMatrix(
      (DenseMatrix.eye[Double](model_0.stateDim), DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)).t),
      (DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), DenseMatrix.eye[Double](model_1.stateDim)))

    val filters: Vector[SquareRootInformationFilter] = Vector(new SquareRootInformationFilter(model_0, false), new SquareRootInformationFilter(model_1, false))
    val smoothers: Vector[SquareRootInformationSmoother] = Vector(new SquareRootInformationSmoother(model_0, false), new SquareRootInformationSmoother(model_1, false))


    val immFilter = new SquareRootIMMFilter(filters, modelStateProjectionMatrix, false)

    val viterbiAlg = new SquareRootViterbiAlgorithm(filters, smoothers, modelStateProjectionMatrix)
    val immSmoother = new SquareRootIMMSmoother(smoothers, modelStateProjectionMatrix, false)

    val multipleModel = new MultipleModelStructure(2, 1.0 - modelSwitchingProbabilityPerUnitTime)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) =
        MultipleModelTestDataGenerator(targetModelLst, 1, initialStateLst, numOfEventsPerTestCase, multipleModel, observationStd, modelStateProjectionMatrix, seed)

      val logModelTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)

      val observationLst: Vector[FactoredGaussianDistribution] = observations.map(x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      })
      val squareRootProcessNoiseCovariancePerFilterLst: Vector[Vector[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateSquareRootProcessNoiseCovariance)
      ).transpose
      val stateTransitionMatrixPerFilterLst: Vector[Vector[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateStateTransitionMatrix)
      ).transpose
      val invStateTransitionMatrixPerFilterLst: Vector[Vector[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateInvStateTransitionMatrix)
      ).transpose

      val immFilterResult = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)
      val fusedIMMFilterResult: Vector[MultipleModelEstimationResult] = SquareRootIMMFilter.fuseEstResult(immFilterResult, modelStateProjectionMatrix)
      outputSampleResult("IMMFilter", seed, observations, fusedIMMFilterResult, states, models, modelStateProjectionMatrix, 1, 0)

      val immSmootherResult = immSmoother(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)
      val fusedIMMSmootherResult: Vector[MultipleModelEstimationResult] = SquareRootIMMSmoother.fuseEstResult(immSmootherResult, modelStateProjectionMatrix)
      outputSampleResult("IMMSmoother", seed, observations, fusedIMMSmootherResult, states, models, modelStateProjectionMatrix, 0, 0)

      val viterbiResult = viterbiAlg(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)
      val mapForwardViterbiResult: Vector[MultipleModelEstimationResult] = viterbiAlg.
        smooth(viterbiResult,
          squareRootProcessNoiseCovariancePerFilterLst,
          stateTransitionMatrixPerFilterLst)
      outputSampleResult("ForwardViterbi", seed, observations, mapForwardViterbiResult, states, models, modelStateProjectionMatrix, 1, 0)

    })

  }

  def outputSampleResult(estimatorName: String,
                         seed: Int,
                         observationVectorLst: Vector[DenseVector[Double]],
                         estimatedResult: Vector[MultipleModelEstimationResult],
                         trueStates: Vector[DenseVector[Double]], trueModels: Vector[Int],
                         modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]],
                         dropLeft: Int, dropRight: Int): Unit = {

    val error = calculateEstimationError(estimatedResult, trueStates, trueModels, modelStateProjectionMatrix, dropLeft, dropRight)
    println(s"$estimatorName, \tState MSE: ${error.head}, \tModel Mean Error: ${error.last}.")
    writeToCSV(trueStates, trueModels, observationVectorLst, estimatedResult, modelStateProjectionMatrix, s"$sampleResultFolder/${estimatorName}_result_$seed.csv")

  }

  def writeToCSV(trueStates: Vector[DenseVector[Double]],
                 trueModels: Vector[Int],
                 observationVectorLst: Vector[DenseVector[Double]],
                 estimatedResults: Vector[MultipleModelEstimationResult],
                 modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]],
                 fileName: String): Unit = {

    val headers = Seq("MODEL", "STATE_X", "STATE_DOT_X", "STATE_Y", "STATE_DOT_Y", "OBS_X", "OBS_Y",
      "EST_MODEL", "EST_MODEL_PROB",
      "EST_X", "EST_DOT_X", "EST_Y", "EST_DOT_Y", "MSE")

    val records: Seq[Seq[String]] = Vector.range(0, trueStates.length).map(idx => {

      val model = trueModels(idx)
      val stateXY = modelStateProjectionMatrix(0, model) * trueStates(idx)
      val observationXY = observationVectorLst(idx)

      val estiamtedResult = estimatedResults(idx)
      val estiamtedModel = estiamtedResult.model
      val estiamtedModelProbability = estiamtedResult.modelProbability

      val firstHalfRow: Seq[String] = Seq(model.toString, stateXY(0).toString, stateXY(1).toString, stateXY(2).toString, stateXY(3).toString,
        observationXY(0).toString, observationXY(1).toString,
        estiamtedModel.toString, estiamtedModelProbability.toString)

      if (det(estiamtedResult.state.R) == 0)
        firstHalfRow ++ Seq("", "", "", "", "")
      else {
        val estimatedXY = modelStateProjectionMatrix(0, estiamtedModel) * estiamtedResult.state.toGaussianDistribution.m
        val errorVector: DenseVector[Double] = estimatedXY - stateXY
        firstHalfRow ++ Seq(estimatedXY(0), estimatedXY(1), estimatedXY(2), estimatedXY(3), errorVector.t * errorVector).map(_.toString)
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
