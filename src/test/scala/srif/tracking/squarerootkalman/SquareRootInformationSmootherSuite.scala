package srif.tracking.squarerootkalman

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.TargetModel.{ConstantPositionModel, ConstantVelocityModel}
import srif.tracking.example.sampleDataGeneration.UniModelTestDataGenerator
import srif.tracking.squarerootkalman.SquareRootInformationSmoother.SmoothResult
import srif.tracking.{FactoredGaussianDistribution, GaussianDistribution, TargetModel}

class SquareRootInformationSmootherSuite extends FlatSpec with Matchers with LazyLogging {

  val seeds: List[Int] = List.range(0, 10)
  val numOfEvents: Int = 1000
  val observationStd: Double = 100.0

  /**
    * Compute the prediction and update error.
    *
    * @param stateLst       list of true states
    * @param results        the smoothed results
    * @param isDebugEnabled control is debug information is logged, default is false
    * @return mean of prediction error and mean of update error
    */
  def computeError(stateLst: List[DenseVector[Double]],
                   results: List[SmoothResult],
                   isDebugEnabled: Boolean = false): Double = {

    val (totalSmoothError, count) = (stateLst, results).zipped.map({
      case (state, result) =>
        val smoothedResult = result.smoothedStateEstimation.toGaussianDistribution

        val smoothError: Double = norm(state - smoothedResult.m)

        if (isDebugEnabled) {
          logger.debug(s"The real state is \n $state")
          logger.debug(s"The smoothed state estimation mean is \n ${smoothedResult.m} with error $smoothError.")
        }
        (smoothError, 1.0)
    }).reduce((x1, x2) => (x1._1 + x2._1, x1._2 + x2._2))

    totalSmoothError / count
  }

  "SquareRootInformationSmoother" should "tracks target moving in constant speed." in {

    val model: TargetModel = ConstantVelocityModel(1.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 5.0, 0.0, 5.0)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)


      val smoother = new SquareRootInformationSmoother(model, false)

      val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)

      val results = smoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst)

      val smoothError = computeError(stateLst, results)

      smoothError should be <= 55.0

    })
  }

  it should "track stationary target with non-zero process noise." in {
    val model: TargetModel = ConstantPositionModel(1.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)

      val smoother = new SquareRootInformationSmoother(model, false)

      val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)

      val results = smoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst)

      val smoothError = computeError(stateLst, results)

      smoothError should be <= 16.0

    })

  }

  it should "track stationary target with zero process noise." in {
    val model: TargetModel = ConstantPositionModel(0.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)

    seeds.foreach(seed => {
      val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, observationStd, seed)

      val smoother = new SquareRootInformationSmoother(model, false)

      val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
        x => {
          val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
          GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
        }
      )

      val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
      val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)

      val results = smoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst)

      val smoothError = computeError(stateLst, results)

      smoothError should be <= 7.0

    })

  }

  it should "track stationary target with zero process noise with perfect observation." in {
    val model: TargetModel = ConstantPositionModel(0.0)
    val initialState: DenseVector[Double] = DenseVector(0.0, 0.0)

    val (stateLst, observationVectorLst, stepSizeLst) = UniModelTestDataGenerator(model, initialState, numOfEvents, 0.0, 0)

    val smoother = new SquareRootInformationSmoother(model, false)

    val observationLst: List[FactoredGaussianDistribution] = observationVectorLst.map(
      x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      }
    )

    val squareRootProcessNoiseCovarianceLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateSquareRootProcessNoiseCovariance)
    val stateTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(model.calculateStateTransitionMatrix)

    val results = smoother(observationLst, squareRootProcessNoiseCovarianceLst, stateTransitionMatrixLst)

    val smoothError = computeError(stateLst, results)

    smoothError should be <= 7.0

  }

}
