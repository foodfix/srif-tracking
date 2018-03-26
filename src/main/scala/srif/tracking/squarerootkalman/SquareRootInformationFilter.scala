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

import breeze.linalg.{DenseMatrix, DenseVector, det, inv, qr}
import com.typesafe.scalalogging.LazyLogging
import scalaz.State
import srif.tracking.squarerootkalman.SquareRootInformationFilter._
import srif.tracking.{FactoredGaussianDistribution, TargetModel, sequence, sumFactoredGaussianDistribution}

/**
  * Implementation of Square Root Information Filter described in Bierman's Book:
  * Factorization Methods for Discrete Sequential Estimation
  *
  * @param targetModel    : the model to describe the target
  * @param isDebugEnabled control is debug information is logged, default is false
  */
class SquareRootInformationFilter(targetModel: TargetModel,
                                  isDebugEnabled: Boolean = false) extends LazyLogging {

  val dim: Int = targetModel.stateDim

  val getTargetModel: TargetModel = targetModel

  /**
    * Return the filter results.
    *
    * @param observationLst                      : a list of [[FactoredGaussianDistribution]] presents the observations
    * @param squareRootProcessNoiseCovarianceLst : refer to [[TargetModel.calculateSquareRootProcessNoiseCovariance]]
    * @param stateTransitionMatrixLst            : refer to [[TargetModel.calculateStateTransitionMatrix]]
    * @param invStateTransitionMatrixLst         : refer to [[TargetModel.calculateInvStateTransitionMatrix()]]
    * @return filterd results
    */
  def apply(observationLst: List[FactoredGaussianDistribution],
            squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]],
            stateTransitionMatrixLst: List[DenseMatrix[Double]],
            invStateTransitionMatrixLst: List[DenseMatrix[Double]]): List[FilterResult] = {

    require(observationLst.lengthCompare(squareRootProcessNoiseCovarianceLst.length) == 0)
    require(squareRootProcessNoiseCovarianceLst.lengthCompare(stateTransitionMatrixLst.length) == 0)

    sequence(List.range(0, observationLst.length).map(idx => {
      val observation = observationLst(idx)
      val squareRootProcessNoiseCovariance = squareRootProcessNoiseCovarianceLst(idx)
      val stateTransitionMatrix = stateTransitionMatrixLst(idx)
      val invStateTransitionMatrix = invStateTransitionMatrixLst(idx)

      filterStep(observation, squareRootProcessNoiseCovariance, stateTransitionMatrix, invStateTransitionMatrix)
    })).eval(FactoredGaussianDistribution(DenseVector.zeros(dim), DenseMatrix.zeros(dim, dim)))

  }

  /**
    * One iteration of filter step.
    *
    * @param observation                      a [[FactoredGaussianDistribution]] presents the observation
    * @param squareRootProcessNoiseCovariance refer to [[TargetModel.calculateSquareRootProcessNoiseCovariance]]
    * @param stateTransitionMatrix            refer to [[TargetModel.calculateStateTransitionMatrix]]
    * @param invStateTransitionMatrix         refer to [[TargetModel.calculateInvStateTransitionMatrix]]
    * @return
    */
  def filterStep(observation: FactoredGaussianDistribution,
                 squareRootProcessNoiseCovariance: DenseMatrix[Double],
                 stateTransitionMatrix: DenseMatrix[Double],
                 invStateTransitionMatrix: DenseMatrix[Double]): State[FactoredGaussianDistribution, FilterResult] =
    State[FactoredGaussianDistribution, FilterResult] {

      previousUpdatedStateEstimation: FactoredGaussianDistribution => {

        val (predictedStateEstimation, tilda_R_w, tilda_R_wx, tilda_z_w) = predictStep(previousUpdatedStateEstimation,
          squareRootProcessNoiseCovariance,
          stateTransitionMatrix,
          invStateTransitionMatrix)

        val updatedStateEstimation: FactoredGaussianDistribution = updateStep(predictedStateEstimation, observation)

        val observationLogLikelihood: Double = computeLogObservationProbability(observation, predictedStateEstimation, targetModel.observationMatrix)

        require(!observationLogLikelihood.isNaN) // observationLogLikelihood can be -Inf, means we have no information at all

        (updatedStateEstimation,
          FilterResult(predictedStateEstimation, updatedStateEstimation, tilda_R_w, tilda_R_wx, tilda_z_w, observationLogLikelihood))
      }

    }

  /**
    * Equation VI.2.29
    *
    * @param updatedStateEstimation           updated state estimation from previous iteration
    * @param squareRootProcessNoiseCovariance refer to [[TargetModel.calculateSquareRootProcessNoiseCovariance]]
    * @param stateTransitionMatrix            refer to [[TargetModel.calculateStateTransitionMatrix]]
    * @param invStateTransitionMatrix         refer to [[TargetModel.calculateInvStateTransitionMatrix]]
    * @return the predicted state estimation
    */
  def predictStep(updatedStateEstimation: FactoredGaussianDistribution,
                  squareRootProcessNoiseCovariance: DenseMatrix[Double],
                  stateTransitionMatrix: DenseMatrix[Double],
                  invStateTransitionMatrix: DenseMatrix[Double]): (FactoredGaussianDistribution, DenseMatrix[Double], DenseMatrix[Double], DenseVector[Double]) = {

    val z_w: DenseVector[Double] = DenseVector.zeros(dim)
    val R_w: DenseMatrix[Double] = DenseMatrix.eye(dim)

    val z_0: DenseVector[Double] = updatedStateEstimation.zeta
    val R_0: DenseMatrix[Double] = updatedStateEstimation.R

    val R_1d: DenseMatrix[Double] = R_0 * invStateTransitionMatrix

    val G: DenseMatrix[Double] = squareRootProcessNoiseCovariance

    val mRow0: DenseMatrix[Double] = DenseMatrix.horzcat(R_w, DenseMatrix.zeros[Double](R_w.rows, R_1d.cols), z_w.toDenseMatrix.t)
    val mRow1: DenseMatrix[Double] = DenseMatrix.horzcat(-R_1d * G, R_1d, z_0.toDenseMatrix.t)
    val m: DenseMatrix[Double] = DenseMatrix.vertcat(mRow0, mRow1)

    val QR = qr(m)

    val tilda_R_w: DenseMatrix[Double] = QR.r(0 until R_w.rows, 0 until R_w.cols)
    val tilda_R_wx: DenseMatrix[Double] = QR.r(0 until R_w.rows, R_w.cols until (R_w.cols + R_1d.cols))
    val tilda_z_w: DenseVector[Double] = QR.r(0 until R_w.rows, (R_w.cols + R_1d.cols) until (R_w.cols + R_1d.cols + 1)).toDenseVector

    val tilda_R_1: DenseMatrix[Double] = QR.r(R_w.rows until (R_w.rows + R_1d.rows), R_w.cols until (R_w.cols + R_1d.cols))
    val tilda_z_1: DenseVector[Double] = QR.r(R_w.rows until (R_w.rows + R_1d.rows), (R_w.cols + R_1d.cols) until (R_w.cols + R_1d.cols + 1)).toDenseVector

    if (isDebugEnabled) {
      logger.debug(s"\nThe predicted zeta is \n $tilda_z_1")
      logger.debug(s"\nThe predicted R is \n ($tilda_R_1)")

      if (det(tilda_R_1) > 0) {
        lazy val mean: DenseVector[Double] = tilda_R_1 \ tilda_z_1
        lazy val covariance: DenseMatrix[Double] = inv(tilda_R_1.t * tilda_R_1)
        logger.debug(s"\nThe predicted mean is \n $mean")
        logger.debug(s"\nThe predicted covariance is \n ($covariance)")
      } else logger.debug(s"\nThe predicted R is not invertible.")

    }

    (FactoredGaussianDistribution(tilda_z_1, tilda_R_1), tilda_R_w, tilda_R_wx, tilda_z_w)

  }

  /**
    * Equation 2.28
    *
    * @param predictedStateEstimation predicted state estimation from [[SquareRootInformationFilter.predictStep]]
    * @param observation              a [[FactoredGaussianDistribution]] presents the observation
    * @return the updated state estimation
    */
  def updateStep(predictedStateEstimation: FactoredGaussianDistribution,
                 observation: FactoredGaussianDistribution): FactoredGaussianDistribution = {

    val tilda_R: DenseMatrix[Double] = predictedStateEstimation.R
    val tilda_z: DenseVector[Double] = predictedStateEstimation.zeta

    val A: DenseMatrix[Double] = observation.R * targetModel.observationMatrix
    val z: DenseVector[Double] = observation.zeta

    val mRow0: DenseMatrix[Double] = DenseMatrix.horzcat(tilda_R, tilda_z.toDenseMatrix.t)
    val mRow1: DenseMatrix[Double] = DenseMatrix.horzcat(A, z.toDenseMatrix.t)
    val m: DenseMatrix[Double] = DenseMatrix.vertcat(mRow0, mRow1)

    val QR = qr(m)

    val hat_R: DenseMatrix[Double] = QR.r(0 until tilda_R.rows, 0 until tilda_R.cols)
    val hat_z: DenseVector[Double] = QR.r(0 until tilda_R.rows, tilda_R.cols until (tilda_R.cols + 1)).toDenseVector
    lazy val e: DenseVector[Double] = QR.r(tilda_R.rows until (tilda_R.rows + A.rows), tilda_R.cols until (tilda_R.cols + 1)).toDenseVector

    if (isDebugEnabled) {
      logger.debug(s"\nThe updated zeta is \n $hat_z")
      logger.debug(s"\nThe updated R is \n ($hat_R)")

      if (det(hat_R) > 0) {
        lazy val mean: DenseVector[Double] = hat_R \ hat_z
        lazy val covariance: DenseMatrix[Double] = inv(hat_R.t * hat_R)
        logger.debug(s"\nThe updated mean is \n $mean")
        logger.debug(s"\nThe updated covariance is \n ($covariance)")
      } else logger.debug(s"\nThe updated R is not invertible.")

    }

    FactoredGaussianDistribution(hat_z, hat_R)

  }

}

object SquareRootInformationFilter {

  /**
    * Compute the logarithmic likelihood of a observation
    *
    * @param observation              refer to [[SquareRootInformationFilter.filterStep]]
    * @param predictedStateEstimation refer to [[SquareRootInformationFilter.updateStep]]
    * @param observationMatrix        refer to [[TargetModel.observationMatrix]]
    * @return logarithmic likelihood
    */
  def computeLogObservationProbability(observation: FactoredGaussianDistribution,
                                       predictedStateEstimation: FactoredGaussianDistribution,
                                       observationMatrix: DenseMatrix[Double]): Double = {
    val measurementResidual: FactoredGaussianDistribution =
      sumFactoredGaussianDistribution(observation, predictedStateEstimation, -observationMatrix)._1

    val logLikelihood: Double = measurementResidual.logLikelihood

    logLikelihood
  }

  case class FilterResult(predictedStateEstimation: FactoredGaussianDistribution,
                          updatedStateEstimation: FactoredGaussianDistribution,
                          tilda_R_w: DenseMatrix[Double],
                          tilda_R_wx: DenseMatrix[Double],
                          tilda_z_w: DenseVector[Double],
                          observationLogLikelihood: Double)

}
