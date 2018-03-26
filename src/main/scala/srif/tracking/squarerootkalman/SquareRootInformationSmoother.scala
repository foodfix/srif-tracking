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
import com.typesafe.scalalogging.LazyLogging
import scalaz.State
import srif.tracking.squarerootkalman.SquareRootInformationFilter.FilterResult
import srif.tracking.squarerootkalman.SquareRootInformationSmoother.SmoothResult
import srif.tracking.{FactoredGaussianDistribution, TargetModel, sequence}

/**
  * Implementation of Square Root Information Smoother described in Bierman's Book:
  * Factorization Methods for Discrete Sequential Estimation
  *
  * @param targetModel    : the model to describe the target
  * @param isDebugEnabled control is debug information is logged, default is false
  */
class SquareRootInformationSmoother(targetModel: TargetModel,
                                    isDebugEnabled: Boolean = false) extends LazyLogging {

  lazy val filter = new SquareRootInformationFilter(targetModel)
  val dim: Int = targetModel.stateDim
  val getTargetModel: TargetModel = targetModel

  /**
    * Return the smooth results.
    *
    * @param observationLst                      : a list of [[FactoredGaussianDistribution]] presents the observations
    * @param squareRootProcessNoiseCovarianceLst : refer to [[TargetModel.calculateSquareRootProcessNoiseCovariance]]
    * @param stateTransitionMatrixLst            : refer to [[TargetModel.calculateStateTransitionMatrix]]
    * @return filterd results
    */
  def apply(observationLst: List[FactoredGaussianDistribution],
            squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]],
            stateTransitionMatrixLst: List[DenseMatrix[Double]]): List[SmoothResult] = {

    val filterResultLst = filter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst)

    sequence(
      (filterResultLst.drop(1), squareRootProcessNoiseCovarianceLst.drop(1), stateTransitionMatrixLst.drop(1)).
        zipped.map({
        case (filterResult, squareRootProcessNoiseCovariance, stateTransitionMatrix) =>
          smoothStep(filterResult, squareRootProcessNoiseCovariance, stateTransitionMatrix)
      }).reverse
    ).eval(filterResultLst.last.updatedStateEstimation).reverse ::: List(SmoothResult(filterResultLst.last.updatedStateEstimation))

  }

  /**
    * One iteration of filter step.
    * Please refer to X.2.7
    *
    * @param filterResult                     [[FilterResult]] of next time stamp
    * @param squareRootProcessNoiseCovariance process noise in next time stamp
    * @param stateTransitionMatrix            state transit matrix in next time stamp
    * @return
    */
  def smoothStep(filterResult: FilterResult,
                 squareRootProcessNoiseCovariance: DenseMatrix[Double],
                 stateTransitionMatrix: DenseMatrix[Double]): State[FactoredGaussianDistribution, SmoothResult] =
    State[FactoredGaussianDistribution, SmoothResult] {

      previousSmoothedStateEstimation =>

        val R_x1: DenseMatrix[Double] = previousSmoothedStateEstimation.R
        val z_x1: DenseVector[Double] = previousSmoothedStateEstimation.zeta

        val G: DenseMatrix[Double] = squareRootProcessNoiseCovariance
        val phi: DenseMatrix[Double] = stateTransitionMatrix

        val tilda_R_w: DenseMatrix[Double] = filterResult.tilda_R_w
        val tilda_R_wx: DenseMatrix[Double] = filterResult.tilda_R_wx
        val tilda_z_w: DenseVector[Double] = filterResult.tilda_z_w

        val mRow0: DenseMatrix[Double] = DenseMatrix.horzcat(tilda_R_w + tilda_R_wx * G, tilda_R_wx * phi, tilda_z_w.toDenseMatrix.t)
        val mRow1: DenseMatrix[Double] = DenseMatrix.horzcat(R_x1 * G, R_x1 * phi, z_x1.toDenseMatrix.t)
        val m: DenseMatrix[Double] = DenseMatrix.vertcat(mRow0, mRow1)

        val QR = qr(m)

        val R_x0: DenseMatrix[Double] = QR.r(tilda_R_w.rows until (tilda_R_w.rows + R_x1.rows), tilda_R_w.cols until (tilda_R_w.cols + phi.cols))
        val z_x0: DenseVector[Double] = QR.r(tilda_R_w.rows until (tilda_R_w.rows + R_x1.rows),
          (tilda_R_w.cols + phi.cols) until (tilda_R_w.cols + phi.cols + 1)).toDenseVector

        lazy val star_R_wx: DenseMatrix[Double] = QR.r(0 until tilda_R_w.rows, tilda_R_w.cols until (tilda_R_w.cols + phi.cols))
        lazy val star_R_w: DenseMatrix[Double] = QR.r(0 until tilda_R_w.rows, 0 until tilda_R_w.cols)
        lazy val star_z_w: DenseVector[Double] = QR.r(0 until tilda_R_w.rows, (tilda_R_w.cols + phi.cols) until (tilda_R_w.cols + phi.cols + 1)).toDenseVector

        val smoothedStateEstimate: FactoredGaussianDistribution = FactoredGaussianDistribution(z_x0, R_x0)

        (smoothedStateEstimate, SmoothResult(smoothedStateEstimate))
    }

}

object SquareRootInformationSmoother {

  case class SmoothResult(smoothedStateEstimation: FactoredGaussianDistribution)

}

