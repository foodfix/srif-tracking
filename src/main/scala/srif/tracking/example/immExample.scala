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

import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import srif.tracking.TargetModel.{ConstantPositionModel, ConstantVelocityModel}
import srif.tracking.example.sampleDataGeneration.MultipleModelTestDataGenerator
import srif.tracking.multipleModel.SquareRootIMMFilter.IMMFilterResult
import srif.tracking.multipleModel.SquareRootIMMSmoother.IMMSmootherResult
import srif.tracking.multipleModel.{MultipleModelStructure, SquareRootIMMFilter, SquareRootIMMSmoother, calculateGaussianMixtureDistribution}
import srif.tracking.squarerootkalman.{SquareRootInformationFilter, SquareRootInformationSmoother}
import srif.tracking.{FactoredGaussianDistribution, GaussianDistribution, TargetModel}

object immExample {

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

    //Define IMM filter over possible models
    val filters: List[SquareRootInformationFilter] = List(new SquareRootInformationFilter(model_0, false), new SquareRootInformationFilter(model_1, false))
    val immFilter = new SquareRootIMMFilter(filters, modelStateProjectionMatrix, false)

    //Define IMM smoothers over possible models
    val smoothers: List[SquareRootInformationSmoother] = List(new SquareRootInformationSmoother(model_0, false), new SquareRootInformationSmoother(model_1, false))
    val immSmoother = new SquareRootIMMSmoother(smoothers, modelStateProjectionMatrix, false)

    val multipleModel = new MultipleModelStructure(2, 1.0 - modelSwitchingProbabilityPerUnitTime)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 1, initialStateLst, numOfEventsPerTestCase, multipleModel, observationStd, modelStateProjectionMatrix, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
      val observationLst: List[FactoredGaussianDistribution] = observations.map(x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      })
      val squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]] = smoothers.map(
        f => stepSizeLst.map(f.getTargetModel.calculateSquareRootProcessNoiseCovariance)
      ).transpose
      val stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = smoothers.map(
        f => stepSizeLst.map(f.getTargetModel.calculateStateTransitionMatrix)
      ).transpose

      val immFilterResult = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)
      val immSmootherResult = immSmoother(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)

      outputSampleResult(states, models, immFilterResult, immSmootherResult, 0.15, 100, false, modelStateProjectionMatrix, numOfEventsPerTestCase)

    })

  }

  def outputSampleResult(states: List[DenseVector[Double]],
                         models: List[Int],
                         immFilterResult: List[IMMFilterResult],
                         immSmootherResult: List[IMMSmootherResult],
                         modelTol: Double,
                         stateTol: Double,
                         isDebugEnabled: Boolean = false,
                         modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]],
                         numOfEvents: Int): Unit = {


    val numOfSkippedEvent: Int = 1

    val error: List[List[Double]] = List.range(0, states.length).drop(numOfSkippedEvent).reverse.map(idx => {

      //True state
      val state = states(idx)
      //True model
      val model = models(idx)

      //Filtered estimations:
      val filterStates: List[FactoredGaussianDistribution] = immFilterResult(idx).updateResultPerFilter.map(_.updatedStateEstimation)
      val filterStateProbabilities: List[Double] = immFilterResult(idx).updatedLogModeProbability.toArray.toList.map(math.exp)
      val filterModel: Int = argmax(immFilterResult(idx).updatedLogModeProbability)
      val filterFusedState = calculateGaussianMixtureDistribution(filterStates, filterStateProbabilities, modelStateProjectionMatrix(filterModel, ::).t.toArray.toList, filterModel)
      val filterErrorVector = modelStateProjectionMatrix(0, filterModel) * filterFusedState.toGaussianDistribution.m - modelStateProjectionMatrix(0, model) * state

      //Smooth estimations:
      val smoothStates: List[FactoredGaussianDistribution] = immSmootherResult(idx).smoothResultPerSmoother.map(_.smoothedStateEstimation)
      val smoothProbabilities: List[Double] = immSmootherResult(idx).smoothedLogModeProbability.toArray.toList.map(math.exp)
      val smoothModel: Int = argmax(immSmootherResult(idx).smoothedLogModeProbability)
      val smoothFusedState = calculateGaussianMixtureDistribution(smoothStates, smoothProbabilities, modelStateProjectionMatrix(smoothModel, ::).t.toArray.toList, smoothModel)
      val smoothErrorVector = modelStateProjectionMatrix(0, smoothModel) * smoothFusedState.toGaussianDistribution.m - modelStateProjectionMatrix(0, model) * state

      val filterStateError: Double = filterErrorVector.t * filterErrorVector
      val smoothStateError: Double = smoothErrorVector.t * smoothErrorVector

      val filterModelScore: Double = filterStateProbabilities(model)
      val smoothModelScore: Double = smoothProbabilities(model)

      List(filterStateError, smoothStateError, filterModelScore, smoothModelScore)

    }).transpose

    val immFilterStateMSE: Double = error.head.sum / (numOfEvents - numOfSkippedEvent)
    val immSmootherStateMSE: Double = error(1).sum / (numOfEvents - numOfSkippedEvent)

    val immFilterModelScore: Double = error(2).sum / (numOfEvents - numOfSkippedEvent)
    val immSmootherModelScore: Double = error(3).sum / (numOfEvents - numOfSkippedEvent)


    println(s"IMM Filter MSE: $immFilterStateMSE,\tIMM Smoother MSE: $immSmootherStateMSE," +
      s"\tIMM Filter Model Score: $immFilterModelScore,\tIMM Smoother Model Score: $immSmootherModelScore")

  }
}
