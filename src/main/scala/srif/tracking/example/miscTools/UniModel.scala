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

import breeze.linalg.DenseVector
import srif.tracking.FactoredGaussianDistribution

object UniModel {

  def calculateEstimationError(estimatedStates: Vector[FactoredGaussianDistribution],
                               states: Vector[DenseVector[Double]],
                               dropLeft: Int = 0, dropRight: Int = 0): Double = {

    require(estimatedStates.length == states.length)

    (estimatedStates, states).zipped.toVector.dropRight(dropRight).drop(dropLeft).map({
      case (estimatedState, state) =>
        val errorVector: DenseVector[Double] = estimatedState.toGaussianDistribution.m - state
        errorVector.t * errorVector
    }).sum / (states.length - dropRight - dropLeft)
  }

}
