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
import srif.tracking.multipleModel.ForwardSquareRootViterbiAlgorithm.ForwardSquareRootViterbiFilterResult

class SquareRootViterbiSmoother extends LazyLogging {

  def apply(forwardResult: List[ForwardSquareRootViterbiFilterResult],
            backwardResult: List[BackwardSquareRootViterbiFilterResult]): List[(FactoredGaussianDistribution, Int, Double)] = {

    val numOfTimeSteps: Int = forwardResult.length

    val firstSmoothedState: (FactoredGaussianDistribution, Int, Double) = {
      val firstSmoothedModelIdx: Int = argmax(backwardResult.head.updatedLogLikelihoodPerFilter)
      (backwardResult.head.updatedEstimatePerFilter(firstSmoothedModelIdx),
        firstSmoothedModelIdx,
        1.0)
    }

    val lastSmoothedState: (FactoredGaussianDistribution, Int, Double) = {
      val lastSmoothedModelIdx: Int = argmax(forwardResult.last.updatedLogLikelihoodPerFilter)
      (forwardResult.last.filterResultPerFilter(lastSmoothedModelIdx).updatedStateEstimation,
        lastSmoothedModelIdx,
        1.0)
    }

    List(firstSmoothedState) ::: List.range(1, numOfTimeSteps - 1).map(idx => {
      smoothStep(forwardResult(idx), backwardResult(idx), idx)
    }) ::: List(lastSmoothedState)

  }

  def smoothStep(currentForwardResult: ForwardSquareRootViterbiFilterResult,
                 currentBackwardResult: BackwardSquareRootViterbiFilterResult, idx: Int): (FactoredGaussianDistribution, Int, Double) = {

    val logModelLikelihoods: DenseVector[Double] = currentForwardResult.predictedLogLikelihoodPerFilter + currentBackwardResult.updatedLogLikelihoodPerFilter
    val modelIdx = argmax(logModelLikelihoods)

    val forwardPredictedEstimates: FactoredGaussianDistribution = currentForwardResult.filterResultPerFilter(modelIdx).predictedStateEstimation
    val backwardUpdatedEstimates: FactoredGaussianDistribution = currentBackwardResult.updatedEstimatePerFilter(modelIdx)

    val mRow0: DenseMatrix[Double] = DenseMatrix.horzcat(forwardPredictedEstimates.R, forwardPredictedEstimates.zeta.toDenseMatrix.t)
    val mRow1: DenseMatrix[Double] = DenseMatrix.horzcat(backwardUpdatedEstimates.R, backwardUpdatedEstimates.zeta.toDenseMatrix.t)

    val m: DenseMatrix[Double] = DenseMatrix.vertcat(mRow0, mRow1)
    val QR = qr(m)
    val hat_R: DenseMatrix[Double] = QR.r(0 until forwardPredictedEstimates.R.rows, 0 until forwardPredictedEstimates.R.cols)
    val hat_z: DenseVector[Double] = QR.r(0 until forwardPredictedEstimates.R.rows, forwardPredictedEstimates.R.cols until (forwardPredictedEstimates.R.cols + 1)).toDenseVector

    val smoothedEstimate: FactoredGaussianDistribution = FactoredGaussianDistribution(hat_z, hat_R)

    (smoothedEstimate, modelIdx, 1.0)

  }

}
