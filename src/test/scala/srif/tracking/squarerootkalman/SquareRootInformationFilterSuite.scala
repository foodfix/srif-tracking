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
import breeze.stats.distributions.MultivariateGaussian
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.TargetModel.{ConstantPositionModel, ConstantVelocityModel}
import srif.tracking.example.miscTools.UniModel.calculateEstimationError
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator.getRandomGaussianDistribution
import srif.tracking.{TargetModel, _}

import scala.util.Random

class SquareRootInformationFilterSuite extends FlatSpec with Matchers with LazyLogging {

  val seeds: Vector[Int] = Vector.range(0, 10)
  val numOfEvents: Int = 1000
  val observationStd: Double = 100.0

  "SquareRootInformationFilter" should "tracks target moving in constant speed." in {

    val model: TargetModel = ConstantVelocityModel(1.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 5.0, 0.0, 5.0)
    val filter = new SquareRootInformationFilter(model, false)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)

      val observationLst: Vector[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)
      val invStateTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateInvStateTransitionMatrix)

      val results = filter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

      val updateError = calculateEstimationError(results.map(_.updatedStateEstimation), stateLst, dropLeft = 1)

      results.length should be(numOfEvents)
      updateError should be <= 95.0 * 95.0

    })
  }

  it should "track stationary target with non-zero process noise." in {
    val model: TargetModel = ConstantPositionModel(1.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)
    val filter = new SquareRootInformationFilter(model, false)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)

      val observationLst: Vector[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)
      val invStateTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateInvStateTransitionMatrix)

      val results = filter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

      val updateError = calculateEstimationError(results.map(_.updatedStateEstimation), stateLst, dropLeft = 1)

      results.length should be(numOfEvents)
      updateError should be <= 30.0 * 30.0

    })

  }

  it should "track stationary target with zero process noise." in {
    val model: TargetModel = ConstantPositionModel(0.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)
    val filter = new SquareRootInformationFilter(model, false)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)

      val observationLst: Vector[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)
      val invStateTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateInvStateTransitionMatrix)

      val results = filter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

      val updateError = calculateEstimationError(results.map(_.updatedStateEstimation), stateLst, dropLeft = 1)

      results.length should be(numOfEvents)
      updateError should be <= 25.0 * 25.0

    })

  }

  it should "track stationary target with zero process noise with perfect observation." in {
    val model: TargetModel = ConstantPositionModel(0.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)

    val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, 0.0, 0)

    val filter = new SquareRootInformationFilter(model, false)

    val observationLst: Vector[FactoredGaussianDistribution] = observationVectorLst.map(
      x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      }
    )

    val squareRootProcessNoiseCovarianceLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
    val stateTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)
    val invStateTransitionMatrixLst: Vector[DenseMatrix[Double]] = stepSizeLst.map(model.calculateInvStateTransitionMatrix)

    val results = filter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

    val updateError = calculateEstimationError(results.map(_.updatedStateEstimation), stateLst, dropLeft = 1)

    results.length should be(numOfEvents)
    updateError should be <= 15.0 * 15.0
  }

  "computeLogObservationProbability" should "compute the likelihood of observation" in {
    val r: Random = new scala.util.Random(0)

    Vector.range(0, 100).foreach(_ => {
      val x = getRandomGaussianDistribution(2, 10, r)
      val y = getRandomGaussianDistribution(2, 10, r)
      val m = DenseMatrix.eye[Double](2)

      val measurementResidual: FactoredGaussianDistribution =
        sumFactoredGaussianDistribution(y.toFactoredGaussianDistribution, x.toFactoredGaussianDistribution, -m)._1

      SquareRootInformationFilter.computeLogObservationProbability(y.toFactoredGaussianDistribution, x.toFactoredGaussianDistribution, m) should be(
        MultivariateGaussian(measurementResidual.toGaussianDistribution.m, measurementResidual.toGaussianDistribution.V).logPdf(DenseVector(0.0, 0.0)) +- 1e-8
      )

    })
  }

}
