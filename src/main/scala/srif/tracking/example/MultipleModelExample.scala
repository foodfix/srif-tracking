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
import srif.tracking.squarerootkalman.{BackwardSquareRootInformationFilter, SquareRootInformationFilter, SquareRootInformationSmoother}
import srif.tracking.{FactoredGaussianDistribution, GaussianDistribution, TargetModel}

object MultipleModelExample {

  def main(args: Array[String]): Unit = {

    val numberOfTestCases: Int = 10
    val numOfEventsPerTestCase: Int = 1000
    val observationStd: Double = 100.0
    val modelSwitchingProbabilityPerUnitTime: Double = 1e-3

    val seeds: List[Int] = List.range(0, numberOfTestCases)

    // Define a constant velocity model
    val model_0: TargetModel = ConstantVelocityModel(0.5)

    //Define a stationary model
    val model_1: TargetModel = ConstantPositionModel(0.0)

    //initial state for models
    val initialStateLst: List[DenseVector[Double]] = List(DenseVector(0.0, 5.0, 0.0, 5.0), DenseVector(0.0, 0.0))

    val targetModelLst: List[TargetModel] = List(model_0, model_1)

    //Define the project matrix between constant velocity model and stationary model
    val modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]] = DenseMatrix(
      (DenseMatrix.eye[Double](model_0.stateDim), DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)).t),
      (DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), DenseMatrix.eye[Double](model_1.stateDim)))

    val forwardFilters: List[SquareRootInformationFilter] = List(new SquareRootInformationFilter(model_0, false), new SquareRootInformationFilter(model_1, false))
    val backwardFilters: List[BackwardSquareRootInformationFilter] = List(new BackwardSquareRootInformationFilter(model_0, false), new BackwardSquareRootInformationFilter(model_1, false))

    val immFilter = new SquareRootIMMFilter(forwardFilters, modelStateProjectionMatrix, false)
    val forwardViterbiFilter = new ForwardSquareRootViterbiAlgorithm(forwardFilters, modelStateProjectionMatrix, false, false)
    val backwardViterbiFilter = new BackwardSquareRootViterbiFilter(backwardFilters, modelStateProjectionMatrix, false)

    val smoothers: List[SquareRootInformationSmoother] = List(new SquareRootInformationSmoother(model_0, false), new SquareRootInformationSmoother(model_1, false))

    val immSmoother = new SquareRootIMMSmoother(smoothers, modelStateProjectionMatrix, false)
    val viterbiSmoother = new SquareRootViterbiSmoother

    val multipleModel = new MultipleModelStructure(2, 1.0 - modelSwitchingProbabilityPerUnitTime)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) =
        MultipleModelTestDataGenerator(targetModelLst, 1, initialStateLst, numOfEventsPerTestCase, multipleModel, observationStd, modelStateProjectionMatrix, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
      val backwardLogModelTransitionMatrixLst: List[DenseMatrix[Double]] = logModelTransitionMatrixLst.tail ::: List(logModelTransitionMatrixLst.head)

      val observationLst: List[FactoredGaussianDistribution] = observations.map(x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      })
      val squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]] = forwardFilters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateSquareRootProcessNoiseCovariance)
      ).transpose
      val stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = forwardFilters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateStateTransitionMatrix)
      ).transpose
      val invStateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = forwardFilters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateInvStateTransitionMatrix)
      ).transpose

      val immFilterResult = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)
      val fusedIMMFilterResult: List[(FactoredGaussianDistribution, Int, Double)] = SquareRootIMMFilter.fuseEstResult(immFilterResult, modelStateProjectionMatrix)
      outputSampleResult("IMMFilter", seed, observations, fusedIMMFilterResult, states, models, modelStateProjectionMatrix, 1, 0)

      val immSmootherResult = immSmoother(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)
      val fusedIMMSmootherResult: List[(FactoredGaussianDistribution, Int, Double)] = SquareRootIMMSmoother.fuseEstResult(immSmootherResult, modelStateProjectionMatrix)
      outputSampleResult("IMMSmoother", seed, observations, fusedIMMSmootherResult, states, models, modelStateProjectionMatrix, 0, 0)

      val forwardViterbiResult = forwardViterbiFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)
      val mapForwardViterbiResult: List[(FactoredGaussianDistribution, Int, Double)] = forwardViterbiFilter.
        smooth(forwardViterbiResult,
        squareRootProcessNoiseCovariancePerFilterLst,
        stateTransitionMatrixPerFilterLst,
        smoothers)
      outputSampleResult("ForwardViterbi", seed, observations, mapForwardViterbiResult, states, models, modelStateProjectionMatrix, 1, 0)

      val backwardViterbiResult = backwardViterbiFilter(backwardLogModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)
      val mapBackwardViterbiResult: List[(FactoredGaussianDistribution, Int, Double)] = BackwardSquareRootViterbiFilter.mapEstResult(backwardViterbiResult)
      outputSampleResult("BackwardViterbi", seed, observations, mapBackwardViterbiResult, states, models, modelStateProjectionMatrix, 0, 1)

      val viterbiSmootherResult: List[(FactoredGaussianDistribution, Int, Double)] = viterbiSmoother(forwardViterbiResult, backwardViterbiResult)
      outputSampleResult("ViterbiSmoother", seed, observations, viterbiSmootherResult, states, models, modelStateProjectionMatrix, 0, 0)

    })

  }

  def outputSampleResult(estimatorName: String,
                         seed: Int,
                         observationVectorLst: List[DenseVector[Double]],
                         estimatedResult: List[(FactoredGaussianDistribution, Int, Double)],
                         trueStates: List[DenseVector[Double]], trueModels: List[Int],
                         modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]],
                         dropLeft: Int, dropRight: Int) = {

    val error = calculateEstimationError(estimatedResult, trueStates, trueModels, modelStateProjectionMatrix, dropLeft, dropRight)
    println(s"$estimatorName, \tState MSE: ${error.head}, \tModel Mean Error: ${error.last}.")
    writeToCSV(trueStates, trueModels, observationVectorLst, estimatedResult, modelStateProjectionMatrix, s"$sampleResultFolder/${estimatorName}_result_$seed.csv")

  }

  def writeToCSV(trueStates: List[DenseVector[Double]],
                 trueModels: List[Int],
                 observationVectorLst: List[DenseVector[Double]],
                 estimatedResults: List[(FactoredGaussianDistribution, Int, Double)],
                 modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]],
                 fileName: String): Unit = {

    val headers = Seq("MODEL", "STATE_X", "STATE_DOT_X", "STATE_Y", "STATE_DOT_Y", "OBS_X", "OBS_Y",
      "EST_MODEL", "EST_MODEL_PROB",
      "EST_X", "EST_DOT_X", "EST_Y", "EST_DOT_Y", "MSE")

    val records: Seq[Seq[String]] = List.range(0, trueStates.length).map(idx => {

      val model = trueModels(idx)
      val stateXY = modelStateProjectionMatrix(0, model) * trueStates(idx)
      val observationXY = observationVectorLst(idx)

      val estiamtedResult = estimatedResults(idx)
      val estiamtedModel = estiamtedResult._2
      val estiamtedModelProbability = estiamtedResult._3

      val firstHalfRow: Seq[String] = Seq(model.toString, stateXY(0).toString, stateXY(1).toString, stateXY(2).toString, stateXY(3).toString,
        observationXY(0).toString, observationXY(1).toString,
        estiamtedModel.toString, estiamtedModelProbability.toString)

      if (det(estiamtedResult._1.R) == 0)
        firstHalfRow ++ Seq("", "", "", "", "")
      else {
        val estimatedXY = modelStateProjectionMatrix(0, estiamtedModel) * estiamtedResult._1.toGaussianDistribution.m
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
