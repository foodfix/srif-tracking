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

import java.io.PrintWriter

import breeze.linalg.{DenseVector, det}
import srif.tracking.FactoredGaussianDistribution

object UniModel {

  def calculateEstimationError(estimatedStates: List[FactoredGaussianDistribution],
                               states: List[DenseVector[Double]],
                               dropLeft: Int = 0, dropRight: Int = 0): Double = {

    require(estimatedStates.length == states.length)

    (estimatedStates, states).zipped.toList.dropRight(dropRight).drop(dropLeft).map({
      case (estimatedState, state) =>
        val errorVector: DenseVector[Double] = estimatedState.toGaussianDistribution.m - state
        errorVector.t * errorVector
    }).sum / (states.length - dropRight - dropLeft)
  }

  def writeToCSV(states: List[DenseVector[Double]],
                 observationVectorLst: List[DenseVector[Double]],
                 estimatedStates: List[FactoredGaussianDistribution],
                 fileName: String): Unit = {

    val headers = Seq("STATE_X", "STATE_DOT_X", "STATE_Y", "STATE_DOT_Y", "OBS_X", "OBS_Y",
      "EST_X", "EST_DOT_X", "EST_Y", "EST_DOT_Y", "MSE")

    val records: Seq[Seq[String]] = List.range(0, states.length).map(idx => {

      val stateXY = states(idx)
      val observationXY = observationVectorLst(idx)

      if (det(estimatedStates(idx).R) == 0)
        Seq(stateXY(0).toString, stateXY(1).toString, stateXY(2).toString, stateXY(3).toString,
          observationXY(0).toString, observationXY(1).toString,
          "", "", "", "", "")
      else {

        val estimatedXY = estimatedStates(idx).toGaussianDistribution.m

        val errorVector: DenseVector[Double] = estimatedXY - stateXY

        Seq(stateXY(0), stateXY(1), stateXY(2), stateXY(3),
          observationXY(0), observationXY(1),
          estimatedXY(0), estimatedXY(1), estimatedXY(2), estimatedXY(3),
          errorVector.t * errorVector).map(_.toString)
      }
    })

    val allRows: Seq[Seq[String]] = Seq(headers) ++ records

    val csv: String = allRows.map(_.mkString(",")).mkString("\n")

    new PrintWriter(fileName) {
      write(csv)
      close()
    }
  }
}
