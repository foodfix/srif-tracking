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
import srif.tracking.squarerootkalman.SquareRootInformationFilter.FilterResult
import srif.tracking.squarerootkalman.{BackwardSquareRootInformationFilter, SquareRootInformationFilter}
import srif.tracking.{FactoredGaussianDistribution, GaussianDistribution, TargetModel}

object UniModelFilterExample {
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

      val forwardFilter = new SquareRootInformationFilter(model, false)
      val backwardFilter = new BackwardSquareRootInformationFilter(model, false)

      val forwardFilterResult: List[FilterResult] = forwardFilter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)
      val backwardFilterResult: List[FilterResult] = backwardFilter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

      outputSampleResult(stateLst, forwardFilterResult, backwardFilterResult, numOfEventsPerTestCase)

    })

  }

  def outputSampleResult(states: List[DenseVector[Double]],
                         forwardFilterResult: List[FilterResult],
                         backwardFilterResult: List[FilterResult],
                         numOfEvents: Int): Unit = {

    val numOfSkippedEvent: Int = 1

    val forwardFilterMSE: Double = calculateEstimationError(forwardFilterResult.map(_.updatedStateEstimation), states, numOfSkippedEvent, numOfSkippedEvent)
    val backwardFilterMSE: Double = calculateEstimationError(backwardFilterResult.map(_.updatedStateEstimation), states, numOfSkippedEvent, numOfSkippedEvent)

    println(s"Forward Filter MSE: $forwardFilterMSE,\tBackward Filter MSE: $backwardFilterMSE")

  }
}
