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

import breeze.linalg.{DenseMatrix, max}
import breeze.numerics.abs
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator.getRandomGaussianDistribution
import srif.tracking.minModeProbability

import scala.util.Random

class multipleModelSuite extends FlatSpec with Matchers {
  val r: Random = new scala.util.Random(0)
  val dim: Int = 4
  val numTestCases: Int = 100
  val range: Double = 10
  val tol: Double = 1e-4

  val dimToBeLifted: Int = 2
  val liftingMatrix: DenseMatrix[Double] = DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)).t

  val modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]] = DenseMatrix(
    (DenseMatrix.eye[Double](4), DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)).t),
    (DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), DenseMatrix.eye[Double](2)))

  "calculateGaussianMixtureDistribution" should "return the mixture distribution" in {

    List.range(0, numTestCases).foreach(_ => {
      val x1 = getRandomGaussianDistribution(dim, range, r)
      val m1 = x1.m
      val V1 = x1.V
      val y1 = x1.toFactoredGaussianDistribution
      val p1 = r.nextDouble

      val x2 = getRandomGaussianDistribution(dim, range, r)
      val m2 = x2.m
      val V2 = x2.V
      val y2 = x2.toFactoredGaussianDistribution
      val p2 = 1.0 - p1

      val bar_m = p1 * m1 + p2 * m2
      val factor = p1 * (m1 - bar_m) * (m1 - bar_m).t + p2 * (m2 - bar_m) * (m2 - bar_m).t

      val x0 = calculateGaussianMixtureDistribution(List(y1, y2), List(p1, p2)).toGaussianDistribution

      max(abs(x0.m - bar_m)) should be <= tol
      max(abs(x0.V - p1 * V1 - p2 * V2 - factor)) should be <= tol
    })

  }

  it should "return the mixture distribution with lifting" in {

    List.range(0, numTestCases).foreach(_ => {
      val x1 = getRandomGaussianDistribution(dim, range, r)
      val m1 = x1.m
      val V1 = x1.V
      val y1 = x1.toFactoredGaussianDistribution
      val p1 = r.nextDouble

      val x2 = getRandomGaussianDistribution(dimToBeLifted, range, r)
      val m2 = x2.m
      val V2 = x2.V
      val y2 = x2.toFactoredGaussianDistribution
      val p2 = 1.0 - p1

      val bar_m = p1 * m1 + p2 * liftingMatrix * m2
      val factor = p1 * (m1 - bar_m) * (m1 - bar_m).t + p2 * (liftingMatrix * m2 - bar_m) * (liftingMatrix * m2 - bar_m).t

      val x0 = calculateGaussianMixtureDistribution(List(y1, y2), List(p1, p2), modelStateProjectionMatrix(0, ::).t.toArray.toList, 0).toGaussianDistribution

      max(abs(x0.m - bar_m)) should be <= tol
      max(abs(x0.V - p1 * V1 - p2 * liftingMatrix * V2 * liftingMatrix.t - factor)) should be <= tol
    })

  }

  it should "return the mixture distribution when one probability is minModeProbability" in {

    List.range(0, numTestCases).foreach(_ => {
      val x1 = getRandomGaussianDistribution(dim, range, r)
      val m1 = x1.m
      val V1 = x1.V
      val y1 = x1.toFactoredGaussianDistribution
      val p1 = minModeProbability

      val x2 = getRandomGaussianDistribution(dim, range, r)
      val m2 = x2.m
      val V2 = x2.V
      val y2 = x2.toFactoredGaussianDistribution
      val p2 = 1.0 - p1

      val bar_m = p1 * m1 + p2 * m2
      val factor = p1 * (m1 - bar_m) * (m1 - bar_m).t + p2 * (m2 - bar_m) * (m2 - bar_m).t

      val x0 = calculateGaussianMixtureDistribution(List(y1, y2), List(p1, p2)).toGaussianDistribution

      max(abs(x0.m - bar_m)) should be <= tol
      max(abs(x0.V - p1 * V1 - p2 * V2 - factor)) should be <= tol
    })

  }

  it should "return the mixture distribution when one probability is minModeProbability with lifting" in {

    List.range(0, numTestCases).foreach(_ => {
      val x1 = getRandomGaussianDistribution(dim, range, r)
      val m1 = x1.m
      val V1 = x1.V
      val y1 = x1.toFactoredGaussianDistribution
      val p1 = minModeProbability

      val x2 = getRandomGaussianDistribution(dimToBeLifted, range, r)
      val m2 = x2.m
      val V2 = x2.V
      val y2 = x2.toFactoredGaussianDistribution
      val p2 = 1.0 - p1

      val bar_m = p1 * m1 + p2 * liftingMatrix * m2
      val factor = p1 * (m1 - bar_m) * (m1 - bar_m).t + p2 * (liftingMatrix * m2 - bar_m) * (liftingMatrix * m2 - bar_m).t

      val x0 = calculateGaussianMixtureDistribution(List(y1, y2), List(p1, p2), modelStateProjectionMatrix(0, ::).t.toArray.toList, 0).toGaussianDistribution

      max(abs(x0.m - bar_m)) should be <= tol
      max(abs(x0.V - p1 * V1 - p2 * liftingMatrix * V2 * liftingMatrix.t - factor)) should be <= tol
    })

  }

  it should "return the mixture distribution when one probability is 0.1 * minModeProbability" in {

    List.range(0, numTestCases).foreach(_ => {
      val x1 = getRandomGaussianDistribution(dim, range, r)
      val m1 = x1.m
      val V1 = x1.V
      val y1 = x1.toFactoredGaussianDistribution
      val p1 = minModeProbability * 0.1

      val x2 = getRandomGaussianDistribution(dim, range, r)
      val m2 = x2.m
      val V2 = x2.V
      val y2 = x2.toFactoredGaussianDistribution
      val p2 = 1.0 - p1

      val bar_m = p1 * m1 + p2 * m2
      val factor = p1 * (m1 - bar_m) * (m1 - bar_m).t + p2 * (m2 - bar_m) * (m2 - bar_m).t

      val x0 = calculateGaussianMixtureDistribution(List(y1, y2), List(p1, p2)).toGaussianDistribution

      max(abs(x0.m - bar_m)) should be <= tol
      max(abs(x0.V - p1 * V1 - p2 * V2 - factor)) should be <= tol * 100
    })

  }

  it should "return the mixture distribution when one probability is 0.1 * minModeProbability with lifting" in {

    List.range(0, numTestCases).foreach(_ => {
      val x1 = getRandomGaussianDistribution(dim, range, r)
      val m1 = x1.m
      val V1 = x1.V
      val y1 = x1.toFactoredGaussianDistribution
      val p1 = minModeProbability * 0.1

      val x2 = getRandomGaussianDistribution(dimToBeLifted, range, r)
      val m2 = x2.m
      val V2 = x2.V
      val y2 = x2.toFactoredGaussianDistribution
      val p2 = 1.0 - p1

      val bar_m = p1 * m1 + p2 * liftingMatrix * m2
      val factor = p1 * (m1 - bar_m) * (m1 - bar_m).t + p2 * (liftingMatrix * m2 - bar_m) * (liftingMatrix * m2 - bar_m).t

      val x0 = calculateGaussianMixtureDistribution(List(y1, y2), List(p1, p2), modelStateProjectionMatrix(0, ::).t.toArray.toList, 0).toGaussianDistribution

      max(abs(x0.m - bar_m)) should be <= tol
      max(abs(x0.V - p1 * V1 - p2 * liftingMatrix * V2 * liftingMatrix.t - factor)) should be <= tol
    })

  }
}
