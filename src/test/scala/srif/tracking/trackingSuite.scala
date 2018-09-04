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

import breeze.linalg.{DenseMatrix, DenseVector, max}
import breeze.numerics.abs
import breeze.stats.distributions.MultivariateGaussian
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator.getRandomGaussianDistribution

import scala.util.Random

class trackingSuite extends FlatSpec with Matchers {

  val r: Random = new scala.util.Random(0)
  val dim: Int = 4
  val numTestCases: Int = 100
  val range: Double = 10
  val tol: Double = 1e-6

  val dimToBeLifted: Int = 2
  val liftingMatrix: DenseMatrix[Double] = DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)).t

  "sumFactoredGaussianDistribution" should "sum two FactoredGaussianDistribution" in {

    Vector.range(0, numTestCases).foreach(_ => {
      val x1 = getRandomGaussianDistribution(dim, range, r)
      val m1 = x1.m
      val V1 = x1.V
      val y1 = x1.toFactoredGaussianDistribution

      val x2 = getRandomGaussianDistribution(dim, range, r)
      val m2 = x2.m
      val V2 = x2.V
      val y2 = x2.toFactoredGaussianDistribution

      val x0 = sumFactoredGaussianDistribution(y1, y2)._1.toGaussianDistribution

      max(abs(x0.m - m1 - m2)) should be <= tol
      max(abs(x0.V - V1 - V2)) should be <= tol
    })

  }

  it should "sum two FactoredGaussianDistribution with lifting" in {

    Vector.range(0, numTestCases).foreach(_ => {
      val x1 = getRandomGaussianDistribution(dim, range, r)
      val m1 = x1.m
      val V1 = x1.V
      val y1 = x1.toFactoredGaussianDistribution

      val x2 = getRandomGaussianDistribution(dimToBeLifted, range, r)
      val m2 = x2.m
      val V2 = x2.V
      val y2 = x2.toFactoredGaussianDistribution

      val x0 = sumFactoredGaussianDistribution(y1, y2, liftingMatrix)._1.toGaussianDistribution

      max(abs(x0.m - m1 - liftingMatrix * m2)) should be <= tol
      max(abs(x0.V - V1 - liftingMatrix * V2 * liftingMatrix.t)) should be <= tol
    })

  }

  it should "sum two FactoredGaussianDistribution with downgrading" in {

    Vector.range(0, numTestCases).foreach(_ => {
      val x1 = getRandomGaussianDistribution(dim, range, r)
      val m1 = x1.m
      val V1 = x1.V
      val y1 = x1.toFactoredGaussianDistribution

      val x2 = getRandomGaussianDistribution(dimToBeLifted, range, r)
      val m2 = x2.m
      val V2 = x2.V
      val y2 = x2.toFactoredGaussianDistribution

      val x0 = sumFactoredGaussianDistribution(y2, y1, liftingMatrix.t)._1.toGaussianDistribution

      max(abs(x0.m - liftingMatrix.t * m1 - m2)) should be <= tol
      max(abs(x0.V - V2 - liftingMatrix.t * V1 * liftingMatrix)) should be <= tol
    })

  }

  "FactoredGaussianDistribution.logLikelihood" should "compute the logarithmic likelihood for dim=2" in {

    Vector.range(0, numTestCases).foreach(_ => {
      val x = getRandomGaussianDistribution(2, range, r)
      x.toFactoredGaussianDistribution.logLikelihood should be(MultivariateGaussian(x.m, x.V).logPdf(DenseVector(0.0, 0.0)) +- 1e-8)
    })

  }

  it should "compute the logarithmic likelihood for dim=4" in {

    Vector.range(0, numTestCases).foreach(_ => {
      val x = getRandomGaussianDistribution(4, range, r)
      x.toFactoredGaussianDistribution.logLikelihood should be(MultivariateGaussian(x.m, x.V).logPdf(DenseVector(0.0, 0.0, 0.0, 0.0)) +- 1e-8)
    })

  }

}
