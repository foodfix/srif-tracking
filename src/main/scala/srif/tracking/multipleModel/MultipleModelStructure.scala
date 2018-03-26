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

import breeze.linalg.{DenseMatrix, DenseVector, diag, svd}
import breeze.numerics.log

class MultipleModelStructure(numOfModels: Int, sameModelProbabilityPerUnitTime: Double) {

  private val modelTransitionMatrixPerUnitTime: DenseMatrix[Double] = {
    val probabilityToSwitchToOneDifferentModel: Double = (1.0 - sameModelProbabilityPerUnitTime) / (numOfModels - 1)

    DenseMatrix.fill[Double](numOfModels, numOfModels)(probabilityToSwitchToOneDifferentModel) +
      diag(DenseVector.fill[Double](numOfModels)(sameModelProbabilityPerUnitTime - probabilityToSwitchToOneDifferentModel))
  }
  private val modelTransitionMatrixPerUnitTimeSVD: svd.DenseSVD = svd(modelTransitionMatrixPerUnitTime)

  def getNumOfModels: Int = numOfModels

  def getLogModelTransitionMatrix(stepSize: Double): DenseMatrix[Double] =
    log(getModelTransitionMatrix(stepSize))

  /**
    * Compute the model transition matrix from one timestamp to next timestamp
    *
    * @param stepSize number of seconds from one timestamp to next timestamp
    * @return (i, j) is the probability of change from model j to model i
    */
  def getModelTransitionMatrix(stepSize: Double): DenseMatrix[Double] = {
    if (stepSize <= 1.0) modelTransitionMatrixPerUnitTime
    else {
      val svd.SVD(u, s, v) = modelTransitionMatrixPerUnitTimeSVD
      val sn: DenseVector[Double] = s.map(math.pow(_, stepSize))
      u * diag(sn) * v
    }
  }

}
