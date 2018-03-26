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

import breeze.linalg.{DenseMatrix, DenseVector, qr}
import breeze.numerics.sqrt

package object multipleModel {

  val MIN_PROBABILITY: Double = 1e-16

  /**
    * Return the [[FactoredGaussianDistribution]], which is obtained by mixing distributionLst
    * with probabilities probabilityLst.
    *
    * @param distributionLst list of distribution to be mixed
    * @param probabilityLst  mixing probabilities for each of the distribution above
    * @return
    */
  def calculateGaussianMixtureDistribution(distributionLst: List[FactoredGaussianDistribution],
                                           probabilityLst: List[Double]): FactoredGaussianDistribution = {

    require(distributionLst.lengthCompare(probabilityLst.length) == 0)

    val modelStateProjectionMatrixLst: List[DenseMatrix[Double]] = List.fill(distributionLst.length)(DenseMatrix.eye[Double](distributionLst.head.zeta.length))

    calculateGaussianMixtureDistribution(distributionLst, probabilityLst, modelStateProjectionMatrixLst, 0)

  }

  /**
    * Mixing a list of [[FactoredGaussianDistribution]], x1, x2, x3, ..., xn
    *
    * @param distributionLst list of distribution to be mixed
    * @param probabilityLst  mixing probabilities for each of the distribution above
    * @param modelStateProjectionMatrixLst
    * @param projectToModelIdx
    * @return
    */
  def calculateGaussianMixtureDistribution(distributionLst: List[FactoredGaussianDistribution],
                                           probabilityLst: List[Double],
                                           modelStateProjectionMatrixLst: List[DenseMatrix[Double]],
                                           projectToModelIdx: Int): FactoredGaussianDistribution = {

    require(distributionLst.lengthCompare(probabilityLst.length) == 0)
    require(distributionLst.lengthCompare(modelStateProjectionMatrixLst.length) == 0)

    val mixingParameterList = (distributionLst, probabilityLst, modelStateProjectionMatrixLst).zipped.toList

    val trunced = mixingParameterList.take(projectToModelIdx) ::: mixingParameterList.drop(projectToModelIdx + 1)

    val targetModelParameter: (FactoredGaussianDistribution, Double, DenseMatrix[Double]) = mixingParameterList(projectToModelIdx)

    require(trunced.lengthCompare(distributionLst.length - 1) == 0)

    trunced.foldLeft(targetModelParameter)({
      case ((x1: FactoredGaussianDistribution, p1: Double, _: DenseMatrix[Double]),
      (x2: FactoredGaussianDistribution, p2: Double, projectMatrix: DenseMatrix[Double])) =>
        val (mixedDistribution, totalProbability) = mixtureTwoDistribution(x1, p1, x2, p2, projectMatrix)
        (mixedDistribution, totalProbability, targetModelParameter._3)
    })._1

  }

  /**
    * Mix two [[FactoredGaussianDistribution]]
    *
    * @param x1 first [[FactoredGaussianDistribution]]
    * @param p1 probability of first [[FactoredGaussianDistribution]]
    * @param x2 second [[FactoredGaussianDistribution]]
    * @param p2 probability of second [[FactoredGaussianDistribution]]
    * @return
    */
  def mixtureTwoDistribution(x1: FactoredGaussianDistribution, p1: Double,
                             x2: FactoredGaussianDistribution, p2: Double): (FactoredGaussianDistribution, Double) =
    mixtureTwoDistribution(x1, p1, x2, p2, DenseMatrix.eye[Double](x2.zeta.length))

  /**
    * Mix two [[FactoredGaussianDistribution]]
    *
    * @param x1 first [[FactoredGaussianDistribution]]
    * @param p1 probability of first [[FactoredGaussianDistribution]]
    * @param x2 second [[FactoredGaussianDistribution]]
    * @param p2 probability of second [[FactoredGaussianDistribution]]
    * @param A  lifting matrix, it lefts x2 to the same dimension of x1
    * @return
    */
  def mixtureTwoDistribution(x1: FactoredGaussianDistribution, p1: Double,
                             x2: FactoredGaussianDistribution, p2: Double, A: DenseMatrix[Double]): (FactoredGaussianDistribution, Double) = {

    require(p1 >= 0, s"p1=$p1, p2=$p2")
    require(p2 >= 0, s"p1=$p1, p2=$p2")

    if (p1 == 0) { // it should not happen
      (FactoredGaussianDistribution(A * x2.zeta, A * x2.R * A.t), p1 + p2)
    } else if (p2 == 0) { // it should not happen
      (x1, p1 + p2)
    } else {

      val p1x1 = FactoredGaussianDistribution(sqrt(p1) * x1.zeta, x1.R / sqrt(p1))
      val p2x2 = FactoredGaussianDistribution(sqrt(p2) * x2.zeta, x2.R / sqrt(p2))

      val (x0, q1, q2) = sumFactoredGaussianDistribution(p1x1, p2x2, A)

      val v1: DenseVector[Double] = q1 / sqrt(p1) - sqrt(p1) * x0.zeta
      val v2: DenseVector[Double] = q2 / sqrt(p2) - sqrt(p2) * x0.zeta

      //another solution by computing the inverse of upper triangular matrix
      //val U = inverseUpperTriangularMatrix(qr(DenseMatrix.vertcat(DenseMatrix.eye[Double](x0.zeta.length), v1.toDenseMatrix, v2.toDenseMatrix)).r(0 until x0.zeta.length, ::)).t
      //(FactoredGaussianDistribution(U * x0.zeta, U * x0.R), p1 + p2)

      val invU = qr(DenseMatrix.vertcat(DenseMatrix.eye[Double](x0.zeta.length), v1.toDenseMatrix, v2.toDenseMatrix)).r(0 until x0.zeta.length, ::).t
      (FactoredGaussianDistribution(invU \ x0.zeta, invU \ x0.R), p1 + p2)

    }

  }

}
