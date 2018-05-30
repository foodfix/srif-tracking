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

package srif.tracking.multipleModel

import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.TargetModel.{ConstantPositionModel, ConstantVelocityModel}
import srif.tracking.example.sampleDataGeneration.MultipleModelTestDataGenerator
import srif.tracking.multipleModel.BackwardSquareRootViterbiFilter.BackwardSquareRootViterbiFilterResult
import srif.tracking.squarerootkalman.BackwardSquareRootInformationFilter
import srif.tracking.{FactoredGaussianDistribution, GaussianDistribution, TargetModel}

import scala.util.Random

class BackwardSquareRootViterbiFilterSuite extends FlatSpec with Matchers {

  val dim: Int = 3
  val r: Random = new scala.util.Random(0)

  val seeds: List[Int] = List.range(0, 10)
  val numOfEvents: Int = 1000
  val observationStd: Double = 100.0

  val model_0: TargetModel = ConstantVelocityModel(0.5)
  val model_1: TargetModel = ConstantPositionModel(0.0)

  val targetModelLst: List[TargetModel] = List(model_0, model_1)
  val initialStateLst: List[DenseVector[Double]] = List(DenseVector(0.0, 5.0, 0.0, 5.0), DenseVector(0.0, 0.0))
  val multipleModel = new MultipleModelStructure(2, 1.0)

  val modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]] = DenseMatrix(
    (DenseMatrix.eye[Double](model_0.stateDim), DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)).t),
    (DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), DenseMatrix.eye[Double](model_1.stateDim)))

  val filters: List[BackwardSquareRootInformationFilter] = List(new BackwardSquareRootInformationFilter(model_0, false), new BackwardSquareRootInformationFilter(model_1, false))
  val backwardViterbiFilter = new BackwardSquareRootViterbiFilter(filters, modelStateProjectionMatrix, false)

  def validateBackwardSquareRootViterbiFilterResult(states: List[DenseVector[Double]], models: List[Int],
                                                    backwardViterbiFilterResult: List[BackwardSquareRootViterbiFilterResult], modelTol: Double, stateTol: Double): Unit = {

    val numOfSkippedEvent: Int = 1

    val error: List[List[Double]] = List.range(0, states.length).dropRight(numOfSkippedEvent).map(idx => {

      val state = states(idx)
      val model: Int = models(idx)

      val filterStates: List[FactoredGaussianDistribution] = backwardViterbiFilterResult(idx).updatedEstimatePerFilter
      val filterStatesLikelihood: DenseVector[Double] = backwardViterbiFilterResult(idx).logLikelihoodPerFilter
      val selectedFilterModel: Int = argmax(filterStatesLikelihood)
      val selectedFilterState = filterStates(selectedFilterModel)
      val selectFilterErrorVector = modelStateProjectionMatrix(0, selectedFilterModel) * selectedFilterState.toGaussianDistribution.m - modelStateProjectionMatrix(0, model) * state

      val filterStateError: Double = selectFilterErrorVector.t * selectFilterErrorVector
      val filterModelScore: Double = if (model == selectedFilterModel) 1.0 else 0.0

      List(filterStateError, filterModelScore)

    }).transpose

    val filterStateMSE: Double = error.head.sum / (numOfEvents - numOfSkippedEvent)
    val filterModelScore: Double = error(1).sum / (numOfEvents - numOfSkippedEvent)

    backwardViterbiFilterResult.length should be(numOfEvents)
    filterStateMSE should be <= stateTol * stateTol
    filterModelScore should be >= (1 - modelTol)

  }

  "BackwardSquareRootViterbiFilter" should "detect stationary object" in {

    val multipleModel = new MultipleModelStructure(2, 1.0)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) =
        MultipleModelTestDataGenerator(targetModelLst, 1, initialStateLst, numOfEvents, multipleModel, observationStd, modelStateProjectionMatrix, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
      val backwardLogModelTransitionMatrixLst: List[DenseMatrix[Double]] = logModelTransitionMatrixLst.tail ::: List(logModelTransitionMatrixLst.head)

      val observationLst: List[FactoredGaussianDistribution] = observations.map(x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      })
      val squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateSquareRootProcessNoiseCovariance)
      ).transpose
      val stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateStateTransitionMatrix)
      ).transpose
      val invStateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateInvStateTransitionMatrix)
      ).transpose

      val result = backwardViterbiFilter(backwardLogModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)

      validateBackwardSquareRootViterbiFilterResult(states, models, result, 0.06, 30)

    })

  }

  it should "detect moving object" in {

    val multipleModel = new MultipleModelStructure(2, 1.0)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 0, initialStateLst, numOfEvents, multipleModel, observationStd, modelStateProjectionMatrix, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
      val backwardLogModelTransitionMatrixLst: List[DenseMatrix[Double]] = logModelTransitionMatrixLst.tail ::: List(logModelTransitionMatrixLst.head)

      val observationLst: List[FactoredGaussianDistribution] = observations.map(x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      })
      val squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateSquareRootProcessNoiseCovariance)
      ).transpose
      val stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateStateTransitionMatrix)
      ).transpose
      val invStateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateInvStateTransitionMatrix)
      ).transpose

      val result = backwardViterbiFilter(backwardLogModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)

      validateBackwardSquareRootViterbiFilterResult(states, models, result, 0.05, 100)

    })

  }

  it should "detect object that changes models" in {

    val multipleModel = new MultipleModelStructure(2, 0.999)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 1, initialStateLst, numOfEvents, multipleModel, observationStd, modelStateProjectionMatrix, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
      val backwardLogModelTransitionMatrixLst: List[DenseMatrix[Double]] = logModelTransitionMatrixLst.tail ::: List(logModelTransitionMatrixLst.head)

      val observationLst: List[FactoredGaussianDistribution] = observations.map(x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      })
      val squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateSquareRootProcessNoiseCovariance)
      ).transpose
      val stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateStateTransitionMatrix)
      ).transpose
      val invStateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = filters.map(
        f => stepSizeLst.map(f.getTargetModel.calculateInvStateTransitionMatrix)
      ).transpose

      val result = backwardViterbiFilter(backwardLogModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)

      validateBackwardSquareRootViterbiFilterResult(states, models, result, 0.15, 150)

    })

  }

}
