package srif

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, diag, inv, max, qr, upperTriangular}
import breeze.numerics.abs
import breeze.stats.distributions.MultivariateGaussian
import scalaz.State

import scala.annotation.tailrec

package object tracking {

  val minModeProbability: Double = 1e-3
  val verySmallStd: Double = 1e-3

  def sequence[S, A](sas: List[State[S, A]]): State[S, List[A]] = {
    @tailrec
    def go(s: S, actions: List[State[S, A]], acc: List[A]): (S, List[A]) =
      actions match {
        case Nil => (s, acc.reverse)
        case h :: t => h.run(s) match {
          case (s2, a) => go(s2, t, a :: acc)
        }
      }

    State((s: S) => go(s, sas, List()))
  }

  /**
    * Return the inverse of a upper triangular matrix.
    *
    * @param t an upper triangular matrix
    * @return inverse of t
    */
  def inverseUpperTriangularMatrix(t: DenseMatrix[Double]): DenseMatrix[Double] = {

    require(isMatrixAlmostEqual(t, upperTriangular(t)))

    val diagElements: DenseVector[Double] = diag(t)

    require(diagElements.forall(_ != 0))

    val gamma: DenseMatrix[Double] = diag(diagElements)
    val gammaInv: DenseMatrix[Double] = diag(1.0 / diagElements)
    val gammaInvTu: DenseMatrix[Double] = gammaInv * (t - gamma)

    List.range(0, t.rows).scanLeft(DenseMatrix.eye[Double](t.rows))({
      case (m, _) => m * -gammaInvTu
    }).reduce(_ + _) * gammaInv

  }

  /**
    * Return if max_{ij} |a - b| <= tol
    */
  def isMatrixAlmostEqual(a: DenseMatrix[Double], b: DenseMatrix[Double], tolerance: Double = 1e-6): Boolean =
    max(abs(a - b)) <= tolerance

  /**
    * Return if max_{i} |a - b| <= tol
    */
  def isVectorAlmostEqual(a: DenseVector[Double], b: DenseVector[Double], tolerance: Double = 1e-6): Boolean =
    max(abs(a - b)) <= tolerance

  /**
    * Sum two [[FactoredGaussianDistribution]] as x3 = x1 + x2
    *
    * For i = 1, 2, 3,
    * let mi = E(xi), (Ri^T * Ri)^(-1) = Var(xi), zi = Ri*mi
    *
    * @param x1 one [[FactoredGaussianDistribution]]
    * @param x2 another [[FactoredGaussianDistribution]]
    * @return x3, R3*m1, R3*m2
    */
  def sumFactoredGaussianDistribution(x1: FactoredGaussianDistribution, x2: FactoredGaussianDistribution):
  (FactoredGaussianDistribution, DenseVector[Double], DenseVector[Double]) =
    sumFactoredGaussianDistribution(x1, x2, DenseMatrix.eye[Double](x2.zeta.length))

  /**
    * Sum two [[FactoredGaussianDistribution]] as x3 = x1 + A*x2
    *
    * For i = 1, 2, 3,
    * let mi = E(xi), (Ri^T * Ri)^(-1) = Var(xi), zi = Ri*mi
    *
    * @param x1 one [[FactoredGaussianDistribution]]
    * @param x2 another [[FactoredGaussianDistribution]]
    * @return x3, R3*m1, R3*A*m2
    */
  def sumFactoredGaussianDistribution(x1: FactoredGaussianDistribution, x2: FactoredGaussianDistribution, A: DenseMatrix[Double]):
  (FactoredGaussianDistribution, DenseVector[Double], DenseVector[Double]) = {

    require(A.cols == x2.zeta.length, s"${A.cols}!=${x2.zeta.length}")
    require(A.rows == x1.zeta.length, s"${A.rows}!=${x1.zeta.length}")

    val R1: DenseMatrix[Double] = x1.R
    val z1: DenseVector[Double] = x1.zeta

    val R2: DenseMatrix[Double] = x2.R
    val z2: DenseVector[Double] = x2.zeta

    val mRow0: DenseMatrix[Double] = DenseMatrix.horzcat(R2, DenseMatrix.zeros[Double](R2.rows, R1.cols), z2.toDenseMatrix.t)
    val mRow1: DenseMatrix[Double] = DenseMatrix.horzcat(-R1 * A, R1, z1.toDenseMatrix.t)
    val m: DenseMatrix[Double] = DenseMatrix.vertcat(mRow0, mRow1)

    val QR = qr(m)

    val R3: DenseMatrix[Double] = QR.r(R2.rows until (R2.rows + R1.rows), R2.cols until (R2.cols + R1.cols))
    val z3: DenseVector[Double] = QR.r(R2.rows until (R2.rows + R1.rows), (R2.cols + R1.cols) until (R2.cols + R1.cols + 1)).toDenseVector

    val Q: DenseMatrix[Double] = QR.q(::, R2.rows until (R2.rows + R1.rows))

    val R3m1 = Q.t * DenseVector.vertcat(DenseVector.zeros[Double](z2.length), z1)
    val R3Am2 = Q.t * DenseVector.vertcat(z2, DenseVector.zeros[Double](z1.length))

    (FactoredGaussianDistribution(z3, R3), R3m1, R3Am2)

  }

  /**
    * Present a Gaussian distribution.
    * If a Gaussian has mean :math:`m` and covariance matrix :math:`V`.
    * Let :math:`V^{-1}=R^T R`, and :math:`\zeta = Rm`.
    *
    * @param zeta :math:`\zeta = Rm`
    * @param R    :math:`V^{-1}=R^T R`
    */
  case class FactoredGaussianDistribution(zeta: DenseVector[Double], R: DenseMatrix[Double]) {
    def toGaussianDistribution: GaussianDistribution = GaussianDistribution(R \ zeta, inv(R.t * R))

    /**
      * Compute the logarithmic likelihood at mean = 0, var = (R^T * R)^{-1}, x = R \ zeta
      *
      * @return
      */
    def logLikelihood: Double = unnormalizedLogLikelihood - 0.5 * zeta.length * math.log(2 * math.Pi) + 0.5 * math.log(math.pow(diag(R).toArray.toList.filter(_ != 0).product, 2))

    def unnormalizedLogLikelihood: Double = -0.5 * (zeta.t * zeta)

    def multiply(A: DenseMatrix[Double]): FactoredGaussianDistribution = {
      val smallDistribution = FactoredGaussianDistribution(DenseVector.zeros[Double](A.rows), DenseMatrix.eye[Double](A.rows) * (1.0 / verySmallStd))
      sumFactoredGaussianDistribution(smallDistribution, this, A)._1
    }

  }

  /**
    * Present a Gaussian distribution.
    *
    * @param m mean of the Gaussian distribution
    * @param V covariance matrix of the Gaussian distribution
    */
  case class GaussianDistribution(m: DenseVector[Double], V: DenseMatrix[Double]) {
    def toFactoredGaussianDistribution: FactoredGaussianDistribution = {
      val R: DenseMatrix[Double] = cholesky(inv(V)).t
      FactoredGaussianDistribution(R * m, R)
    }

    def multiply(A: DenseMatrix[Double]): GaussianDistribution = GaussianDistribution(A * m, A * V * A.t)

    def logLikelihood: Double = MultivariateGaussian(m, V).logPdf(DenseVector.zeros[Double](m.length))
  }

}
