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

package srif.tracking

import breeze.linalg.{DenseMatrix, cholesky}

trait TargetModel {

  // Dimension of the state space.
  val stateDim: Int
  val observationDim: Int

  val observationMatrix: DenseMatrix[Double]

  /**
    * Return the state-transition matrix.
    *
    * The Kalman filter model assumes the true state at time k is evolved from the state at (k − 1) according to
    * :math:`x_{k} = F_k x_{k-1} + G * w_k`
    * where :math:`F_k` is the state-transition matrix.
    *
    * @param stepSize number of seconds from time (k-1) to time k.
    * @return the state-transition matrix
    */
  def calculateStateTransitionMatrix(stepSize: Double): DenseMatrix[Double]

  /**
    * Return the inverse of state-transition matrix.
    * :math:`x_{k-1} F_k^{-1} * x_{k} - F_k^{-1} * G * w_k`
    *
    * @param stepSize number of seconds from time (k-1) to time k.
    * @return the inverse of state-transition matrix
    */
  def calculateInvStateTransitionMatrix(stepSize: Double): DenseMatrix[Double] = calculateStateTransitionMatrix(-stepSize)

  /**
    * Return the covariance matrix of process noise.
    *
    * The Kalman filter model assumes the true state at time k is evolved from the state at (k − 1) according to
    * :math:`x_{k} = F_k x_{k-1} + w_k`
    * where :math:`w_k` is the process noise which is assumed to be drawn from
    * a zero mean multivariate normal distribution with covariance matrix return by this function.
    *
    * @param stepSize number of seconds from time (k-1) to time k.
    * @return the covariance matrix of process noise.
    */
  def calculateProcessNoiseCovariance(stepSize: Double): DenseMatrix[Double]

  /**
    * Return square root of the covariance matrix of process noise.
    *
    * @param stepSize number of seconds from time (k-1) to time k.
    * @return
    */
  def calculateSquareRootProcessNoiseCovariance(stepSize: Double): DenseMatrix[Double] = {
    val processNoiseCovariance = calculateProcessNoiseCovariance(stepSize: Double)
    if (processNoiseCovariance.forall(_ == 0)) processNoiseCovariance
    else cholesky(processNoiseCovariance)
  }

}

object TargetModel {

  /**
    * Describe a constant Velocity Model.
    * Two white noise model is implemented: continuous white noise model, and piecewise white noise model.
    *
    * @param sigma           if "continuous" white noise model is used, it is the spectral density of the white noise;
    *                        if "piecewise" white noise model is used, it is the standard deviation of acceleration between each time periods.
    * @param whiteNoiseModel either "continues" or "piecewise". The default model is "continuous"
    */
  case class ConstantVelocityModel(sigma: Double, whiteNoiseModel: String = "continuous") extends TargetModel {

    require((whiteNoiseModel == "continuous") || (whiteNoiseModel == "piecewise"))

    // the state is described as :math:`(x, \dot x, y, \dot y)`
    val stateDim: Int = 4
    val observationDim: Int = 2

    val observationMatrix: DenseMatrix[Double] = DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))

    def calculateStateTransitionMatrix(stepSize: Double): DenseMatrix[Double] = {

      DenseMatrix(
        (1.0, stepSize, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, stepSize),
        (0.0, 0.0, 0.0, 1.0)
      )
    }

    def calculateProcessNoiseCovariance(stepSize: Double): DenseMatrix[Double] = {

      require(stepSize > 0)

      lazy val stepSizePower2 = scala.math.pow(stepSize, 2)
      lazy val stepSizePower3 = scala.math.pow(stepSize, 3)
      lazy val stepSizePower4 = scala.math.pow(stepSize, 4)

      // building blocks for continuous white noise model
      lazy val Q1: DenseMatrix[Double] = DenseMatrix(
        (stepSizePower3 / 3.0, stepSizePower2 / 2.0),
        (stepSizePower2 / 2.0, stepSize))

      // building blocks for piecewise white noise model
      lazy val Q2: DenseMatrix[Double] = DenseMatrix(
        (stepSizePower4 / 4.0, stepSizePower3 / 3.0),
        (stepSizePower3 / 3.0, stepSizePower2))

      if (whiteNoiseModel == "continuous")
        DenseMatrix.vertcat(
          DenseMatrix.horzcat(Q1, DenseMatrix.zeros[Double](Q1.rows, Q1.cols)),
          DenseMatrix.horzcat(DenseMatrix.zeros[Double](Q1.rows, Q1.cols), Q1)
        ) * sigma
      else
        DenseMatrix.vertcat(
          DenseMatrix.horzcat(Q2, DenseMatrix.zeros[Double](Q2.rows, Q2.cols)),
          DenseMatrix.horzcat(DenseMatrix.zeros[Double](Q2.rows, Q2.cols), Q2)
        ) * sigma * sigma

    }

  }

  /**
    * Describe a stationary target.
    *
    * @param speedStd standard deviation of velocity between each time periods
    */
  case class ConstantPositionModel(speedStd: Double) extends TargetModel {

    val stateDim: Int = 2
    val observationDim: Int = 2

    val observationMatrix: DenseMatrix[Double] = DenseMatrix.eye[Double](stateDim)

    def calculateStateTransitionMatrix(stepSize: Double): DenseMatrix[Double] = {
      DenseMatrix.eye[Double](stateDim)
    }

    def calculateProcessNoiseCovariance(stepSize: Double): DenseMatrix[Double] = {

      require(stepSize > 0)

      val speedVariance = speedStd * speedStd
      DenseMatrix.eye[Double](stateDim) * speedVariance * stepSize

    }

  }

}