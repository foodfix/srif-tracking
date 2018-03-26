package srif.tracking.squarerootkalman

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import breeze.stats.distributions.MultivariateGaussian
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.TargetModel.{ConstantPositionModel, ConstantVelocityModel}
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator.getRandomGaussianDistribution
import srif.tracking.squarerootkalman.SquareRootInformationFilter.FilterResult
import srif.tracking.{TargetModel, _}

import scala.util.Random

class SquareRootInformationFilterSuite extends FlatSpec with Matchers with LazyLogging {

  val seeds: List[Int] = List.range(0, 10)
  val numOfEvents: Int = 1000
  val observationStd: Double = 100.0

  /**
    * Compute the prediction and update error.
    *
    * @param stateLst       list of true states
    * @param results        the filter results
    * @param isDebugEnabled control is debug information is logged, default is false
    * @return mean of prediction error and mean of update error
    */
  def computeError(stateLst: List[DenseVector[Double]],
                   results: List[FilterResult],
                   isDebugEnabled: Boolean = false): (Double, Double) = {

    val (totalPredictionError, totalUpdateError, count) = (stateLst, results).zipped.drop(numOfEvents / 10).map({
      case (state, result) =>
        val predictedResult = result.predictedStateEstimation.toGaussianDistribution
        val updatedResult = result.updatedStateEstimation.toGaussianDistribution

        val predictionError: Double = norm(state - predictedResult.m)
        val updateError: Double = norm(state - updatedResult.m)

        if (isDebugEnabled) {
          logger.debug(s"The real state is \n $state")
          logger.debug(s"The Predicted state estimation mean is \n ${predictedResult.m} with error $predictionError.")
          logger.debug(s"The updated state estimation mean is \n ${updatedResult.m} with error $updateError.")
        }

        (predictionError, updateError, 1.0)
    }).reduce((x1, x2) => (x1._1 + x2._1, x1._2 + x2._2, x1._3 + x2._3))

    (totalPredictionError / count, totalUpdateError / count)
  }


  "SquareRootInformationFilter" should "tracks target moving in constant speed." in {

    val model: TargetModel = ConstantVelocityModel(1.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 5.0, 0.0, 5.0)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)

      val filter = new SquareRootInformationFilter(model, false)

      val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)

      val results = filter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst)

      val (predictionError, updateError) = computeError(stateLst, results)

      predictionError should be <= 130.0
      updateError should be <= 95.0

    })
  }

  it should "track stationary target with non-zero process noise." in {
    val model: TargetModel = ConstantPositionModel(1.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)

      val filter = new SquareRootInformationFilter(model, false)

      val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)

      val results = filter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst)

      val (predictionError, updateError) = computeError(stateLst, results)

      predictionError should be <= 25.0
      updateError should be <= 23.0

    })

  }

  it should "track stationary target with zero process noise." in {
    val model: TargetModel = ConstantPositionModel(0.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)


    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)

      val filter = new SquareRootInformationFilter(model, false)

      val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)

      val results = filter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst)

      val (predictionError, updateError) = computeError(stateLst, results)

      predictionError should be <= 11.0
      updateError should be <= 11.0

    })

  }

  it should "track stationary target with zero process noise with perfect observation." in {
    val model: TargetModel = ConstantPositionModel(0.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)


    val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, 0.0, 0)

    val filter = new SquareRootInformationFilter(model, false)

    val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
      x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      }
    )

    val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
    val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)

    val results = filter(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst)

    val (predictionError, updateError) = computeError(stateLst, results)

    predictionError should be <= 11.0
    updateError should be <= 11.0

  }

  "computeLogObservationProbability" should "compute the likelihood of observation" in {
    val r: Random = new scala.util.Random(0)

    List.range(0, 100).foreach(_ => {
      val x = getRandomGaussianDistribution(2, 10, r)
      val y = getRandomGaussianDistribution(2, 10, r)
      val m = DenseMatrix.eye[Double](2)

      val measurementResidual: FactoredGaussianDistribution =
        sumFactoredGaussianDistribution(y.toFactoredGaussianDistribution, x.toFactoredGaussianDistribution, -m)._1

      SquareRootInformationFilter.computeLogObservationProbability(y.toFactoredGaussianDistribution, x.toFactoredGaussianDistribution, m) should be(
        MultivariateGaussian(measurementResidual.toGaussianDistribution.m, measurementResidual.toGaussianDistribution.V).logPdf(DenseVector(0.0, 0.0)) +- 1e-8
      )

    })
  }

}
