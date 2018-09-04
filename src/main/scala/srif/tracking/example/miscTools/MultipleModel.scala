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

package srif.tracking.example.miscTools

import breeze.linalg.{DenseMatrix, DenseVector}
import srif.tracking.FactoredGaussianDistribution
import srif.tracking.multipleModel.MultipleModelEstimationResult

object MultipleModel {

  def calculateEstimationError(estimatedResult: Vector[MultipleModelEstimationResult],
                               trueStates: Vector[DenseVector[Double]], trueModels: Vector[Int],
                               modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]],
                               dropLeft: Int = 0, dropRight: Int = 0): Vector[Double] = {

    Vector.range(0, trueStates.length).dropRight(dropRight).drop(dropLeft).map(idx => {

      val trueState = trueStates(idx)
      val trueModel = trueModels(idx)

      val estState: FactoredGaussianDistribution = estimatedResult(idx).state
      val estModel: Int = estimatedResult(idx).model
      val estProbability: Double = estimatedResult(idx).modelProbability

      val errorVector: DenseVector[Double] = modelStateProjectionMatrix(0, estModel) * estState.toGaussianDistribution.m - modelStateProjectionMatrix(0, trueModel) * trueState
      val stateScore: Double = errorVector.t * errorVector

      val modelScore: Double = if (trueModel == estModel) estProbability else 1 - estProbability

      DenseVector(stateScore, modelScore)
    }).reduce(_ + _).toArray.toVector.map(_ / (trueStates.length - dropLeft - dropRight))

  }

}
