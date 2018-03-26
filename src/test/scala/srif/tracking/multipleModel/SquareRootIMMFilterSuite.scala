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

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, sum}
import breeze.numerics.{exp, log}
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.TargetModel.{ConstantPositionModel, ConstantVelocityModel}
import srif.tracking.example.sampleDataGeneration.MultipleModelTestDataGenerator
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator._
import srif.tracking.multipleModel.SquareRootIMMFilter.{IMMFilterResult, _}
import srif.tracking.squarerootkalman.SquareRootInformationFilter
import srif.tracking.{TargetModel, _}

import scala.util.Random

class SquareRootIMMFilterSuite extends FlatSpec with Matchers {

  val dim: Int = 3
  val r: Random = new scala.util.Random(0)
  val numOfTestToDo: Int = 100

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

  val filters: List[SquareRootInformationFilter] = List(new SquareRootInformationFilter(model_0, false), new SquareRootInformationFilter(model_1, false))
  val immFilter = new SquareRootIMMFilter(filters, modelStateProjectionMatrix, false)

  def validateIMMFilterResult(states: List[DenseVector[Double]], models: List[Int],
                              immFilterResult: List[IMMFilterResult], modelTol: Double, stateTol: Double): Unit = {

    val numOfSkippedEvent: Int = 1

    val error: List[List[Double]] = List.range(0, states.length).drop(numOfSkippedEvent).reverse.map(idx => {

      val state = states(idx)
      val model = models(idx)

      val filterStates: List[FactoredGaussianDistribution] = immFilterResult(idx).updateResultPerFilter.map(_.updatedStateEstimation)
      val filterStateProbabilities: List[Double] = immFilterResult(idx).updatedLogModeProbability.toArray.toList.map(math.exp)
      val filterModel: Int = argmax(immFilterResult(idx).updatedLogModeProbability)
      val filterFusedState = calculateGaussianMixtureDistribution(filterStates, filterStateProbabilities, modelStateProjectionMatrix(filterModel, ::).t.toArray.toList, filterModel)
      val filterErrorVector = modelStateProjectionMatrix(0, filterModel) * filterFusedState.toGaussianDistribution.m - modelStateProjectionMatrix(0, model) * state

      val filterStateError: Double = filterErrorVector.t * filterErrorVector
      val filterModelScore: Double = filterStateProbabilities(model)

      List(filterStateError, filterModelScore)

    }).transpose

    val immFilterStateMSE: Double = error.head.sum / (numOfEvents - numOfSkippedEvent)
    val immFilterModelScore: Double = error(1).sum / (numOfEvents - numOfSkippedEvent)

    immFilterStateMSE should be <= stateTol * stateTol
    immFilterModelScore should be >= (1 - modelTol)

  }

  "calculateLogMixingWeight" should "compute the mixing weight" in {

    List.range(0, numOfTestToDo).foreach(_ => {
      val previousModeProbabilities: DenseVector[Double] = getNormalizedVector(dim, r)
      val previousLogModeProbabilities: DenseVector[Double] = log(previousModeProbabilities)

      val modelTransitionMatrix: DenseMatrix[Double] = getRandomTransitionMatrix(dim, r)
      val logModelTransitionMatrix: DenseMatrix[Double] = log(modelTransitionMatrix)

      val predictedModelProbabilities = modelTransitionMatrix * previousModeProbabilities

      val m = modelTransitionMatrix(*, ::) * previousModeProbabilities
      val mixingMatrix = m(::, *) / predictedModelProbabilities

      val ret = calculateLogMixingWeight(previousLogModeProbabilities, logModelTransitionMatrix)

      val expMixingMatrix = exp(ret._1)
      val expPredictedModelProbabilities = exp(ret._2)

      isMatrixAlmostEqual(expMixingMatrix, mixingMatrix) should be(true)
      isVectorAlmostEqual(sum(expMixingMatrix(*, ::)), DenseVector.fill(dim, 1.0)) should be(true)
      isVectorAlmostEqual(expPredictedModelProbabilities, predictedModelProbabilities)

    })

  }

  "SquareRootIMMFilter" should "detect stationary object" in {

    val multipleModel = new MultipleModelStructure(2, 1.0)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 1, initialStateLst, numOfEvents, multipleModel, observationStd, modelStateProjectionMatrix, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
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

      val result = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)

      validateIMMFilterResult(states, models, result, 0.05, 30)

    })

  }

  it should "detect moving object" in {

    val multipleModel = new MultipleModelStructure(2, 1.0)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 0, initialStateLst, numOfEvents, multipleModel, observationStd, modelStateProjectionMatrix, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
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

      val result = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)

      validateIMMFilterResult(states, models, result, 0.01, 100)

    })

  }

  it should "detect object that changes models" in {

    val multipleModel = new MultipleModelStructure(2, 0.999)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 1, initialStateLst, numOfEvents, multipleModel, observationStd, modelStateProjectionMatrix, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
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

      val result: List[IMMFilterResult] = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)

      validateIMMFilterResult(states, models, result, 0.15, 100)

    })

  }


}
