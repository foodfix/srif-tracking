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
