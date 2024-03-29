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

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{MultivariateGaussian, RandBasis}
import srif.tracking.TargetModel
import srif.tracking.multipleModel.MultipleModelStructure

import scala.util.Random

object MultipleModelTestDataGenerator {

  def apply(targetModelLst: Vector[TargetModel],
            initialModel: Int,
            initialStateLst: Vector[DenseVector[Double]],
            numOfEvents: Int,
            multipleModel: MultipleModelStructure,
            observationStd: Double,
            modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]],
            seed: Int): (Vector[Int], Vector[DenseVector[Double]], Vector[DenseVector[Double]], Vector[Double]) = {

    val r: Random = new scala.util.Random(seed)

    val stepSizeLst: Vector[Double] = Vector.fill[Double](numOfEvents - 1)(r.nextInt(10) + 1) // the length is numberOfEvents - 1

    val modelsWithStates: Vector[(Int, DenseVector[Double])] = stepSizeLst.scanLeft((initialModel, initialStateLst(initialModel)))({
      case ((previousModel: Int, previousState: DenseVector[Double]), stepSize: Double) =>

        val modelTransitionMatrix: DenseMatrix[Double] = multipleModel.getModelTransitionMatrix(stepSize)
        val modeProbabilities: DenseVector[Double] = modelTransitionMatrix(::, previousModel)

        val newModel: Int = sample(modeProbabilities, r)
        val stateTransitionMatrix: DenseMatrix[Double] = targetModelLst(newModel).calculateStateTransitionMatrix(stepSize)
        val processNoise = targetModelLst(newModel).calculateProcessNoiseCovariance(stepSize)

        val newState: DenseVector[Double] =
          if (processNoise.forall(_ == 0))
            stateTransitionMatrix * modelStateProjectionMatrix(newModel, previousModel) * previousState
          else stateTransitionMatrix * modelStateProjectionMatrix(newModel, previousModel) * previousState +
            MultivariateGaussian(DenseVector.zeros[Double](targetModelLst(newModel).stateDim), processNoise)(RandBasis.withSeed(r.nextInt)).draw

        (newModel, newState)

    })

    val models: Vector[Int] = modelsWithStates.map(_._1)
    val states: Vector[DenseVector[Double]] = modelsWithStates.map(_._2)
    val observations: Vector[DenseVector[Double]] = generateObservations(modelsWithStates, targetModelLst, observationStd, r)

    (models, states, observations, 1.0 +: stepSizeLst)

  }

  def generateObservations(modelsWithStates: Vector[(Int, DenseVector[Double])], targetModelLst: Vector[TargetModel], observationStd: Double, r: Random): Vector[DenseVector[Double]] = {
    modelsWithStates.map({
      case (model: Int, state: DenseVector[Double]) =>
        val observationCovariance: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        val observationNoise: DenseVector[Double] = MultivariateGaussian(DenseVector.zeros[Double](targetModelLst(model).observationDim), observationCovariance)(RandBasis.withSeed(r.nextInt)).draw
        targetModelLst(model).observationMatrix * state + observationNoise

    })
  }

  def sample(dist: DenseVector[Double], r: Random): Int = {
    val p = r.nextDouble
    dist.toArray.toVector.scanLeft(0.0)(_ + _).zipWithIndex.filter(_._1 >= p).head._2 - 1
  }

}
