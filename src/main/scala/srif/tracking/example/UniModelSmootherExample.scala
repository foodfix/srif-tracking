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

import breeze.linalg.{DenseMatrix, DenseVector}
import srif.tracking.TargetModel.ConstantVelocityModel
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator.calculateEstimationError
import srif.tracking.squarerootkalman.SquareRootInformationSmoother.SmoothResult
import srif.tracking.squarerootkalman.{ForwardBackwardSquareRootInformationSmoother, SquareRootInformationSmoother}
import srif.tracking.{FactoredGaussianDistribution, GaussianDistribution, TargetModel}

object UniModelSmootherExample {
  def main(args: Array[String]): Unit = {

    val numberOfTestCases: Int = 10
    val numOfEventsPerTestCase: Int = 1000
    val observationStd: Double = 100.0

    val seeds: List[Int] = List.range(0, numberOfTestCases)

    val model: TargetModel = ConstantVelocityModel(1.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 5.0, 0.0, 5.0)

    seeds.foreach(seed => {

      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEventsPerTestCase, observationStd, seed)

      val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)
      val invStateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateInvStateTransitionMatrix)

      val smoother = new SquareRootInformationSmoother(model, false)
      val forwardBackwardSmoother = new ForwardBackwardSquareRootInformationSmoother(model, false)

      val smootherResult: List[SmoothResult] = smoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)
      val forwardBackwardSmootherResult: List[SmoothResult] = forwardBackwardSmoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

      outputSampleResult(stateLst, smootherResult, forwardBackwardSmootherResult, numOfEventsPerTestCase)

    })

  }

  def outputSampleResult(states: List[DenseVector[Double]],
                         smootherResult: List[SmoothResult],
                         forwardBackwardSmootherResult: List[SmoothResult],
                         numOfEvents: Int): Unit = {

    val smootherMSE: Double = calculateEstimationError(smootherResult.map(_.smoothedStateEstimation), states)
    val forwardBackwardSmootherMSE: Double = calculateEstimationError(forwardBackwardSmootherResult.map(_.smoothedStateEstimation), states)

    println(s"Smoother MSE: $smootherMSE,\tForward-Backward smoother MSE: $forwardBackwardSmootherMSE")

  }
}
