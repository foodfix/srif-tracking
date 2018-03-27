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

package srif.tracking.squarerootkalman

import breeze.linalg.{DenseMatrix, DenseVector}
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.TargetModel.{ConstantPositionModel, ConstantVelocityModel}
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator.calculateEstimationError
import srif.tracking.{FactoredGaussianDistribution, GaussianDistribution, TargetModel}

class SquareRootInformationSmootherSuite extends FlatSpec with Matchers with LazyLogging {

  val seeds: List[Int] = List.range(0, 10)
  val numOfEvents: Int = 1000
  val observationStd: Double = 100.0

  "SquareRootInformationSmoother" should "tracks target moving in constant speed." in {

    val model: TargetModel = ConstantVelocityModel(1.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 5.0, 0.0, 5.0)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)


      val smoother = new SquareRootInformationSmoother(model, false)

      val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)
      val invStateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateInvStateTransitionMatrix)

      val results = smoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

      val smoothError = calculateEstimationError(results.map(_.smoothedStateEstimation), stateLst)

      results.length should be(numOfEvents)
      smoothError should be <= 55.0 * 55.0

    })
  }

  it should "track stationary target with non-zero process noise." in {
    val model: TargetModel = ConstantPositionModel(1.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)

      val smoother = new SquareRootInformationSmoother(model, false)

      val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)
      val invStateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateInvStateTransitionMatrix)

      val results = smoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

      val smoothError = calculateEstimationError(results.map(_.smoothedStateEstimation), stateLst)

      results.length should be(numOfEvents)
      smoothError should be <= 20.0 * 20.0

    })

  }

  it should "track stationary target with zero process noise." in {
    val model: TargetModel = ConstantPositionModel(0.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)

      val smoother = new SquareRootInformationSmoother(model, false)

      val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)
      val invStateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateInvStateTransitionMatrix)

      val results = smoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

      val smoothError = calculateEstimationError(results.map(_.smoothedStateEstimation), stateLst)

      results.length should be(numOfEvents)
      smoothError should be <= 7.0 * 7.0

    })

  }

  it should "track stationary target with zero process noise with perfect observation." in {
    val model: TargetModel = ConstantPositionModel(0.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)

    val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, 0.0, 0)

    val smoother = new SquareRootInformationSmoother(model, false)

    val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
      x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      }
    )

    val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
    val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)
    val invStateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateInvStateTransitionMatrix)

    val results = smoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

    val smoothError = calculateEstimationError(results.map(_.smoothedStateEstimation), stateLst)

    results.length should be(numOfEvents)
    smoothError should be <= 7.0 * 7.0

  }

}
