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
import scalaz.State
import srif.tracking.squarerootkalman.SquareRootInformationFilter.{FilterResult, computeLogObservationProbability}
import srif.tracking.{FactoredGaussianDistribution, TargetModel, sequence}

class BackwardSquareRootInformationFilter(targetModel: TargetModel,
                                          isDebugEnabled: Boolean = false) extends
  SquareRootInformationFilter(targetModel, isDebugEnabled) {

  /**
    * Return the filter results.
    *
    * @param observationLst                      : a vector of [[FactoredGaussianDistribution]] presents the observations
    * @param squareRootProcessNoiseCovarianceLst : refer to [[TargetModel.calculateSquareRootProcessNoiseCovariance]]
    * @param stateTransitionMatrixLst            : refer to [[TargetModel.calculateStateTransitionMatrix]]
    * @param invStateTransitionMatrixLst         : refer to [[TargetModel.calculateInvStateTransitionMatrix]]
    * @return filterd results
    */
  override def apply(observationLst: Vector[FactoredGaussianDistribution],
                     squareRootProcessNoiseCovarianceLst: Vector[DenseMatrix[Double]],
                     stateTransitionMatrixLst: Vector[DenseMatrix[Double]],
                     invStateTransitionMatrixLst: Vector[DenseMatrix[Double]]): Vector[FilterResult] = {

    require(observationLst.lengthCompare(squareRootProcessNoiseCovarianceLst.length) == 0)
    require(squareRootProcessNoiseCovarianceLst.lengthCompare(stateTransitionMatrixLst.length) == 0)

    sequence(Vector.range(0, observationLst.length).reverse.map(idx => {
      val observation = observationLst(idx)
      val squareRootProcessNoiseCovariance = squareRootProcessNoiseCovarianceLst(idx)
      val stateTransitionMatrix = stateTransitionMatrixLst(idx)
      val invStateTransitionMatrix = invStateTransitionMatrixLst(idx)

      backwardFilterStep(observation, squareRootProcessNoiseCovariance, stateTransitionMatrix, invStateTransitionMatrix)
    })).eval(FactoredGaussianDistribution(DenseVector.zeros(dim), DenseMatrix.zeros(dim, dim))).reverse

  }

  /**
    * One iteration of backward filter step.
    *
    * @param observation                      a [[FactoredGaussianDistribution]] presents the observation
    * @param squareRootProcessNoiseCovariance refer to [[TargetModel.calculateSquareRootProcessNoiseCovariance]]
    * @param stateTransitionMatrix            refer to [[TargetModel.calculateStateTransitionMatrix]]
    * @param invStateTransitionMatrix         refer to [[TargetModel.calculateInvStateTransitionMatrix]]
    * @return
    */
  def backwardFilterStep(observation: FactoredGaussianDistribution,
                         squareRootProcessNoiseCovariance: DenseMatrix[Double],
                         stateTransitionMatrix: DenseMatrix[Double],
                         invStateTransitionMatrix: DenseMatrix[Double]): State[FactoredGaussianDistribution, FilterResult] =
    State[FactoredGaussianDistribution, FilterResult] {

      nextBackwardPredictedStateEstimation: FactoredGaussianDistribution => {

        val observationLogLikelihood: Double = computeLogObservationProbability(observation, nextBackwardPredictedStateEstimation, targetModel.observationMatrix)
        require(!observationLogLikelihood.isNaN)

        val updatedStateEstimation: FactoredGaussianDistribution = updateStep(nextBackwardPredictedStateEstimation, observation)

        val (predictedStateEstimation, tilda_R_w, tilda_R_wx, tilda_z_w) = predictStep(updatedStateEstimation,
          -invStateTransitionMatrix * squareRootProcessNoiseCovariance,
          invStateTransitionMatrix,
          stateTransitionMatrix)

        (predictedStateEstimation,
          FilterResult(predictedStateEstimation, updatedStateEstimation, tilda_R_w, tilda_R_wx, tilda_z_w, observationLogLikelihood))
      }

    }

}
