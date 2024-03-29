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

package srif.tracking.example.sampleDataGeneration

import breeze.linalg.{DenseMatrix, DenseVector, diag, sum}
import breeze.stats.distributions.{MultivariateGaussian, RandBasis}
import srif.tracking.{GaussianDistribution, TargetModel}

import scala.util.Random

object UniModelTestDataGenerator {

  /**
    * Generate states and observation for a given model.
    *
    * @param targetModel    the model used to generate the states
    * @param initialState   the inital state
    * @param numOfEvents    number of states to be generated
    * @param observationStd observation error standard deviation
    * @param seed           the random seed
    * @return vector of state, vector of observation vector, vector of step sizes
    */
  def apply(targetModel: TargetModel,
            initialState: DenseVector[Double],
            numOfEvents: Int,
            observationStd: Double,
            seed: Int): (Vector[DenseVector[Double]], Vector[DenseVector[Double]], Vector[Double]) = {

    val r: Random = new scala.util.Random(seed)

    val (stateLst, stepSizeLst) = UniModelTestDataGenerator.generateState(targetModel, initialState, numOfEvents, r)

    val observationCovarianceLst: Vector[DenseMatrix[Double]] = Vector.fill(numOfEvents)(
      DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
    )

    val observationVectorLst: Vector[DenseVector[Double]] =
      if (observationStd > 0) UniModelTestDataGenerator.generateObservations(targetModel, stateLst, observationCovarianceLst, r)
      else stateLst

    require(stateLst.lengthCompare(numOfEvents) == 0)
    require(observationCovarianceLst.lengthCompare(numOfEvents) == 0)
    require(stepSizeLst.lengthCompare(numOfEvents) == 0)

    (stateLst, observationVectorLst, stepSizeLst)

  }

  /**
    * Generate states for a given model
    *
    * @param targetModel  refer to [[UniModelTestDataGenerator.apply]]
    * @param initialState refer to [[UniModelTestDataGenerator.apply]]
    * @param numOfEvents  refer to [[UniModelTestDataGenerator.apply]]
    * @param r            random number generator
    * @return vector of state, vector of step sizes
    */
  def generateState(targetModel: TargetModel,
                    initialState: DenseVector[Double],
                    numOfEvents: Int,
                    r: Random): (Vector[DenseVector[Double]], Vector[Double]) = {

    val stepSizeLst: Vector[Double] = Vector.fill[Double](numOfEvents - 1)(r.nextInt(10) + 1)


    def getNextState(state: DenseVector[Double], stepSize: Double): DenseVector[Double] = {
      val stateTransitionMatrix = targetModel.calculateStateTransitionMatrix(stepSize)
      val processNoise = targetModel.calculateProcessNoiseCovariance(stepSize)

      if (processNoise.forall(_ == 0)) stateTransitionMatrix * state
      else stateTransitionMatrix * state + MultivariateGaussian(DenseVector.zeros[Double](targetModel.stateDim), processNoise)(RandBasis.withSeed(r.nextInt)).draw
    }

    (stepSizeLst.scanLeft(initialState)(getNextState), 1.0 +: stepSizeLst)

  }

  /**
    * Generate the observations for a vector of states
    *
    * @param targetModel              refer to [[UniModelTestDataGenerator.apply]]
    * @param stateLst                 vector of states generated by [[generateState]]
    * @param observationCovarianceLst vector of observation error covariance matrix
    * @param r                        random number generator
    * @return vector of observation vector,
    */
  def generateObservations(targetModel: TargetModel,
                           stateLst: Vector[DenseVector[Double]],
                           observationCovarianceLst: Vector[DenseMatrix[Double]],
                           r: Random): Vector[DenseVector[Double]] = {

    (stateLst, observationCovarianceLst).zipped.map({
      case (state, observationCovariance) => targetModel.observationMatrix * state +
        MultivariateGaussian(DenseVector.zeros[Double](targetModel.observationDim), observationCovariance)(RandBasis.withSeed(r.nextInt)).draw
    })

  }

  /**
    * Generate a random Gaussian distribution
    *
    * @param dim   dimension of the distribution
    * @param range range of the random number
    * @param r     random generator
    * @return
    */
  def getRandomGaussianDistribution(dim: Int, range: Double, r: Random): GaussianDistribution =
    GaussianDistribution(getRandomMean(dim, range, r),
      getRandomCovarianceMatrix(dim, range, r))

  def getRandomCovarianceMatrix(dim: Int, range: Double, r: Random): DenseMatrix[Double] = {

    val eigenValues: DenseVector[Double] = DenseVector.fill(dim)(r.nextDouble() * range + 0.1)
    val Q: DenseMatrix[Double] = DenseMatrix.fill(dim, dim)(r.nextDouble() * range - range / 2)

    val D: DenseMatrix[Double] = diag(eigenValues)

    Q.t * D * Q
  }

  def getRandomMean(dim: Int, range: Double, r: Random): DenseVector[Double] = DenseVector.fill(dim)(r.nextDouble() * range - range / 2)

  def getRandomTransitionMatrix(dim: Int, r: Random): DenseMatrix[Double] = {

    val randomVectors = Vector.fill(dim)(getNormalizedVector(dim, r).toDenseMatrix.t)

    DenseMatrix.horzcat(randomVectors: _*)

  }

  def getNormalizedVector(dim: Int, r: Random): DenseVector[Double] = {
    val randomVector: DenseVector[Double] = DenseVector.fill(dim)(r.nextDouble() + 1e-8)
    randomVector / sum(randomVector)
  }

}
