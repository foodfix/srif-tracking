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

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.{exp, log}
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.TargetModel.{ConstantPositionModel, ConstantVelocityModel}
import srif.tracking.example.miscTools.MultipleModel.calculateEstimationError
import srif.tracking.example.sampleDataGeneration.MultipleModelTestDataGenerator
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator._
import srif.tracking.multipleModel.SquareRootIMMFilter._
import srif.tracking.squarerootkalman.SquareRootInformationFilter
import srif.tracking.{TargetModel, _}

import scala.util.Random

class SquareRootIMMFilterSuite extends FlatSpec with Matchers {

  val dim: Int = 3
  val r: Random = new scala.util.Random(0)
  val numOfTestToDo: Int = 100

  val seeds: Vector[Int] = Vector.range(0, 10)
  val numOfEvents: Int = 1000
  val observationStd: Double = 100.0

  val model_0: TargetModel = ConstantVelocityModel(0.5)
  val model_1: TargetModel = ConstantPositionModel(0.0)

  val targetModelLst: Vector[TargetModel] = Vector(model_0, model_1)
  val initialStateLst: Vector[DenseVector[Double]] = Vector(DenseVector(0.0, 5.0, 0.0, 5.0), DenseVector(0.0, 0.0))

  val modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]] = DenseMatrix(
    (DenseMatrix.eye[Double](model_0.stateDim), DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)).t),
    (DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), DenseMatrix.eye[Double](model_1.stateDim)))

  val filters: Vector[SquareRootInformationFilter] = Vector(new SquareRootInformationFilter(model_0, false), new SquareRootInformationFilter(model_1, false))
  val immFilter = new SquareRootIMMFilter(filters, modelStateProjectionMatrix, false)

  "calculateLogMixingWeight" should "compute the mixing weight" in {

    Vector.range(0, numOfTestToDo).foreach(_ => {
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

      val result = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)

      val fusedResult = fuseEstResult(result, modelStateProjectionMatrix)

      val error: Vector[Double] = calculateEstimationError(fusedResult, states, models, modelStateProjectionMatrix, 1)

      error.head should be <= 470.0
      error.last should be >= 0.95

    })

  }

  it should "detect moving object" in {

    val multipleModel = new MultipleModelStructure(2, 1.0)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 0, initialStateLst, numOfEvents, multipleModel, observationStd, modelStateProjectionMatrix, seed)

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

      val result = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)

      val fusedResult = fuseEstResult(result, modelStateProjectionMatrix)

      val error: Vector[Double] = calculateEstimationError(fusedResult, states, models, modelStateProjectionMatrix, 1)

      error.head should be <= 7800.0
      error.last should be >= 0.99

    })

  }

  it should "detect object that changes models" in {

    val multipleModel = new MultipleModelStructure(2, 0.999)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 1, initialStateLst, numOfEvents, multipleModel, observationStd, modelStateProjectionMatrix, seed)

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

      val result = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)

      val fusedResult = fuseEstResult(result, modelStateProjectionMatrix)

      val error: Vector[Double] = calculateEstimationError(fusedResult, states, models, modelStateProjectionMatrix, 1)

      error.head should be <= 5700.0
      error.last should be >= 0.87

    })

  }


}
