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

import breeze.linalg.{DenseMatrix, DenseVector, argmax, qr}
import com.typesafe.scalalogging.LazyLogging
import srif.tracking.FactoredGaussianDistribution
import srif.tracking.multipleModel.BackwardSquareRootViterbiFilter.BackwardSquareRootViterbiFilterResult
import srif.tracking.multipleModel.ForwardSquareRootViterbiFilter.ForwardSquareRootViterbiFilterResult
import srif.tracking.multipleModel.SquareRootViterbiSmoother.SquareRootViterbiSmootherResult
import srif.tracking.squarerootkalman.{BackwardSquareRootInformationFilter, SquareRootInformationFilter}

class SquareRootViterbiSmoother(filters: List[SquareRootInformationFilter],
                                backwardFilters: List[BackwardSquareRootInformationFilter],
                                modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]],
                                isDebugEnabled: Boolean = false) extends LazyLogging {

  val numOfFilters: Int = filters.length
  val forwardViterbiFilter = new ForwardSquareRootViterbiFilter(filters, modelStateProjectionMatrix, true, isDebugEnabled)
  val backwardViterbiFilter = new BackwardSquareRootViterbiFilter(backwardFilters, modelStateProjectionMatrix, isDebugEnabled)

  /**
    * Return the forward Viterbi filter result.
    *
    * @param logModelTransitionMatrixLst                  logarithmic model transition matrix for each timestamp
    * @param observationLst                               observation for each timestamp
    * @param squareRootProcessNoiseCovariancePerFilterLst process noise covariance matrix in square root form for each timestamp
    * @param stateTransitionMatrixPerFilterLst            state transition matrix for each timestamp
    * @param invStateTransitionMatrixPerFilterLst         inverse of state transition matrix for each timestamp
    * @return filter result for each timestamp
    */
  def apply(logModelTransitionMatrixLst: List[DenseMatrix[Double]],
            observationLst: List[FactoredGaussianDistribution],
            squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]],
            stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]],
            invStateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]]): List[SquareRootViterbiSmootherResult] = {

    val numOfTimeSteps: Int = observationLst.length

    val forwardResult: List[ForwardSquareRootViterbiFilterResult] =
      forwardViterbiFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)

    val backwardLogModelTransitionMatrixLst: List[DenseMatrix[Double]] = logModelTransitionMatrixLst.tail ::: List(logModelTransitionMatrixLst.head)
    val backwardResult: List[BackwardSquareRootViterbiFilterResult] =
      backwardViterbiFilter(backwardLogModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst, invStateTransitionMatrixPerFilterLst)

    val firstSmoothedState: SquareRootViterbiSmootherResult = {
      val firstSmoothedModelIdx: Int = argmax(backwardResult.head.updatedLogLikelihoodPerFilter)
      SquareRootViterbiSmootherResult(backwardResult.head.updatedEstimatePerFilter(firstSmoothedModelIdx),
        backwardResult.head.updatedLogLikelihoodPerFilter(firstSmoothedModelIdx),
        firstSmoothedModelIdx)
    }

    val lastSmoothedState: SquareRootViterbiSmootherResult = {
      val lastSmoothedModelIdx: Int = argmax(forwardResult.last.updatedLogLikelihoodPerFilter)
      SquareRootViterbiSmootherResult(forwardResult.last.updatedEstimatePerFilter(lastSmoothedModelIdx),
        forwardResult.last.updatedLogLikelihoodPerFilter(lastSmoothedModelIdx),
        lastSmoothedModelIdx)
    }

    List(firstSmoothedState) ::: List.range(1, numOfTimeSteps - 1).map(idx => {
      smoothStep(forwardResult(idx), backwardResult(idx), idx)
    }) ::: List(lastSmoothedState)

  }

  def smoothStep(currentForwardResult: ForwardSquareRootViterbiFilterResult,
                 currentBackwardResult: BackwardSquareRootViterbiFilterResult, idx: Int): SquareRootViterbiSmootherResult = {

    val logLikelihoods: DenseVector[Double] = currentForwardResult.predictedLogLikelihoodPerFilter + currentBackwardResult.updatedLogLikelihoodPerFilter
    val modelIdx = argmax(logLikelihoods)

    val forwardPredictedEstimates: FactoredGaussianDistribution = currentForwardResult.predictedEstimatePerFilter(modelIdx)
    val backwardUpdatedEstimates: FactoredGaussianDistribution = currentBackwardResult.updatedEstimatePerFilter(modelIdx)

    val mRow0: DenseMatrix[Double] = DenseMatrix.horzcat(forwardPredictedEstimates.R, forwardPredictedEstimates.zeta.toDenseMatrix.t)
    val mRow1: DenseMatrix[Double] = DenseMatrix.horzcat(backwardUpdatedEstimates.R, backwardUpdatedEstimates.zeta.toDenseMatrix.t)

    val m: DenseMatrix[Double] = DenseMatrix.vertcat(mRow0, mRow1)
    val QR = qr(m)
    val hat_R: DenseMatrix[Double] = QR.r(0 until forwardPredictedEstimates.R.rows, 0 until forwardPredictedEstimates.R.cols)
    val hat_z: DenseVector[Double] = QR.r(0 until forwardPredictedEstimates.R.rows, forwardPredictedEstimates.R.cols until (forwardPredictedEstimates.R.cols + 1)).toDenseVector

    val smoothedEstimate: FactoredGaussianDistribution = FactoredGaussianDistribution(hat_z, hat_R)

    SquareRootViterbiSmootherResult(smoothedEstimate, logLikelihoods(modelIdx), modelIdx)

  }

}

object SquareRootViterbiSmoother {

  case class SquareRootViterbiSmootherResult(smoothedEstimate: FactoredGaussianDistribution, logLikelihood: Double, modelIdx: Int)

}
