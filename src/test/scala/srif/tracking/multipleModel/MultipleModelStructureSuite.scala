package srif.tracking.multipleModel

import breeze.linalg.DenseMatrix
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.isMatrixAlmostEqual

class MultipleModelStructureSuite extends FlatSpec with Matchers {

  "modelTransitionMatrixPerUnitTime" should "return the model transition matrix for 1 second step size." in {

    val multipleModel = new MultipleModelStructure(3, 0.94)
    val stepSize: Double = 1.0

    val modeTransitionMatrixPerSecond: DenseMatrix[Double] = DenseMatrix(
      (0.94, 0.03, 0.03), (0.03, 0.94, 0.03), (0.03, 0.03, 0.94)
    )

    isMatrixAlmostEqual(multipleModel.getModelTransitionMatrix(stepSize), modeTransitionMatrixPerSecond) should be(true)

  }

  it should "return the model transition matrix for 5 second step size." in {

    val multipleModel = new MultipleModelStructure(3, 0.94)
    val stepSize: Double = 5.0

    val modeTransitionMatrixPerSecond: DenseMatrix[Double] = DenseMatrix(
      (0.94, 0.03, 0.03), (0.03, 0.94, 0.03), (0.03, 0.03, 0.94)
    )

    val expectedModeTransitionMatrix: DenseMatrix[Double] = List.fill(stepSize.toInt)(modeTransitionMatrixPerSecond).reduce(_ * _)

    isMatrixAlmostEqual(multipleModel.getModelTransitionMatrix(stepSize), expectedModeTransitionMatrix) should be(true)

  }

  it should "return the model transition matrix for very big step size." in {

    val multipleModel = new MultipleModelStructure(3, 0.94)
    val stepSize: Double = 1000.0

    val expectedModeTransitionMatrix: DenseMatrix[Double] = DenseMatrix.fill(3, 3)(1.0 / 3)

    isMatrixAlmostEqual(multipleModel.getModelTransitionMatrix(stepSize), expectedModeTransitionMatrix) should be(true)

  }

}
