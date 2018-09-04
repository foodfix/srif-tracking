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

import breeze.linalg.{DenseMatrix, DenseVector, qr}
import srif.tracking.squarerootkalman.SquareRootInformationFilter.FilterResult
import srif.tracking.squarerootkalman.SquareRootInformationSmoother.SmoothResult
import srif.tracking.{FactoredGaussianDistribution, TargetModel}

class ForwardBackwardSquareRootInformationSmoother(targetModel: TargetModel,
                                                   isDebugEnabled: Boolean = false) {

  lazy val forwardFilter = new SquareRootInformationFilter(targetModel)
  lazy val backwardFilter = new BackwardSquareRootInformationFilter(targetModel)

  val dim: Int = targetModel.stateDim
  val getTargetModel: TargetModel = targetModel

  /**
    * Return the smooth results.
    *
    * @param observationLst                      : a list of [[FactoredGaussianDistribution]] presents the observations
    * @param squareRootProcessNoiseCovarianceLst : refer to [[TargetModel.calculateSquareRootProcessNoiseCovariance]]
    * @param stateTransitionMatrixLst            : refer to [[TargetModel.calculateStateTransitionMatrix]]
    * @param invStateTransitionMatrixLst         refer to [[TargetModel.calculateInvStateTransitionMatrix]]
    * @return filterd results
    */
  def apply(observationLst: List[FactoredGaussianDistribution],
            squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]],
            stateTransitionMatrixLst: List[DenseMatrix[Double]],
            invStateTransitionMatrixLst: List[DenseMatrix[Double]]): List[SmoothResult] = {

    val forwardFilterResultLst = forwardFilter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)
    val backwardFilterResultLst = backwardFilter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst, invStateTransitionMatrixLst)

    SmoothResult(backwardFilterResultLst.head.updatedStateEstimation, backwardFilterResultLst.head.observationLogLikelihood) :: List.range(1, observationLst.length - 1).map({
      idx => {
        val forwardFilterResult = forwardFilterResultLst(idx)
        val backwardFilterResult = backwardFilterResultLst(idx)
        smoothStep(forwardFilterResult, backwardFilterResult)
      }
    }) ::: List(SmoothResult(forwardFilterResultLst.last.updatedStateEstimation, forwardFilterResultLst.last.observationLogLikelihood))
  }

  /**
    * Return the smoothed result at timestamp k
    *
    * @param forwardFilterResult  forward filtered result at timestamp k
    * @param backwardFilterResult backward filtered result at timestamp k
    * @return
    */
  def smoothStep(forwardFilterResult: FilterResult, backwardFilterResult: FilterResult): SmoothResult = {

    val forwardPredictedStateEstimation: FactoredGaussianDistribution = forwardFilterResult.predictedStateEstimation
    val backwardUpdatedStateEstimation: FactoredGaussianDistribution = backwardFilterResult.updatedStateEstimation

    val R0: DenseMatrix[Double] = forwardPredictedStateEstimation.R
    val zeta0: DenseVector[Double] = forwardPredictedStateEstimation.zeta

    val R1: DenseMatrix[Double] = backwardUpdatedStateEstimation.R
    val zeta1: DenseVector[Double] = backwardUpdatedStateEstimation.zeta

    val mRow0: DenseMatrix[Double] = DenseMatrix.horzcat(R0, zeta0.toDenseMatrix.t)
    val mRow1: DenseMatrix[Double] = DenseMatrix.horzcat(R1, zeta1.toDenseMatrix.t)

    val m: DenseMatrix[Double] = DenseMatrix.vertcat(mRow0, mRow1)

    val QR = qr(m)

    val hat_R: DenseMatrix[Double] = QR.r(0 until R0.rows, 0 until R0.cols)
    val hat_zeta: DenseVector[Double] = QR.r(0 until R0.rows, R0.cols until (R0.cols + 1)).toDenseVector

    lazy val e: DenseVector[Double] = QR.r(R0.rows until (R0.rows + R1.rows), R0.cols until (R0.cols + 1)).toDenseVector

    SmoothResult(FactoredGaussianDistribution(hat_zeta, hat_R), forwardFilterResult.observationLogLikelihood)

  }

}
