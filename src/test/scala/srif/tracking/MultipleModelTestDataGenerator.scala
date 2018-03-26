package srif.tracking

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{MultivariateGaussian, RandBasis}
import srif.tracking.multipleModel.MultipleModelStructure

import scala.util.Random

object MultipleModelTestDataGenerator {

  def apply(targetModelLst: List[TargetModel],
            initialModel: Int,
            initialStateLst: List[DenseVector[Double]],
            numOfEvents: Int,
            multipleModel: MultipleModelStructure,
            observationStd: Double,
            projectionMatrixLst: List[DenseMatrix[Double]],
            seed: Int): (List[Int], List[DenseVector[Double]], List[DenseVector[Double]], List[Double]) = {

    val r: Random = new scala.util.Random(seed)

    val stepSizeLst: List[Double] = List.fill[Double](numOfEvents - 1)(r.nextInt(10) + 1) // the length is numberOfEvents - 1

    val modelsWithStates: List[(Int, DenseVector[Double])] = stepSizeLst.scanLeft((initialModel, initialStateLst(initialModel)))({
      case ((previousModel: Int, previousState: DenseVector[Double]), stepSize: Double) =>

        val modelTransitionMatrix: DenseMatrix[Double] = multipleModel.getModelTransitionMatrix(stepSize)
        val modeProbabilities: DenseVector[Double] = modelTransitionMatrix(::, previousModel)

        val newModel: Int = sample(modeProbabilities, r)
        val stateTransitionMatrix: DenseMatrix[Double] = targetModelLst(newModel).calculateStateTransitionMatrix(stepSize)
        val processNoise = targetModelLst(newModel).calculateProcessNoiseCovariance(stepSize)

        val newState: DenseVector[Double] =
          if (processNoise.forall(_ == 0))
            projectionMatrixLst(newModel).t * (projectionMatrixLst(newModel) * stateTransitionMatrix * projectionMatrixLst(newModel).t *
              projectionMatrixLst(previousModel) * previousState)
          else projectionMatrixLst(newModel).t * (projectionMatrixLst(newModel) * stateTransitionMatrix * projectionMatrixLst(newModel).t *
            projectionMatrixLst(previousModel) * previousState) +
            MultivariateGaussian(DenseVector.zeros[Double](targetModelLst(newModel).stateDim), processNoise)(RandBasis.withSeed(r.nextInt)).draw

        (newModel, newState)

    })

    val models: List[Int] = modelsWithStates.map(_._1)
    val states: List[DenseVector[Double]] = modelsWithStates.map(_._2)
    val observations: List[DenseVector[Double]] = generateObservations(modelsWithStates, targetModelLst, observationStd, r)

    (models, states, observations, 1.0 :: stepSizeLst)

  }

  def generateObservations(modelsWithStates: List[(Int, DenseVector[Double])], targetModelLst: List[TargetModel], observationStd: Double, r: Random): List[DenseVector[Double]] = {
    modelsWithStates.map({
      case (model: Int, state: DenseVector[Double]) =>
        val observationCovariance: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        val observationNoise: DenseVector[Double] = MultivariateGaussian(DenseVector.zeros[Double](targetModelLst(model).observationDim), observationCovariance)(RandBasis.withSeed(r.nextInt)).draw
        targetModelLst(model).observationMatrix * state + observationNoise

    })
  }

  def sample(dist: DenseVector[Double], r: Random): Int = {
    val p = r.nextDouble
    dist.toArray.toList.scanLeft(0.0)(_ + _).zipWithIndex.filter(_._1 >= p).head._2 - 1
  }

}
