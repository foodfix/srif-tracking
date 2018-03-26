package srif.tracking.multipleModel

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, sum}
import breeze.numerics.{exp, log}
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{FlatSpec, Matchers}
import srif.tracking.TargetModel.{ConstantPositionModel, ConstantVelocityModel}
import srif.tracking.UniModelTestDataGenerator.{getNormalizedVector, getRandomTransitionMatrix}
import srif.tracking.{TargetModel, _}
import srif.tracking.multipleModel.SquareRootIMMFilter.IMMFilterResult
import srif.tracking.multipleModel.SquareRootIMMSmoother._
import srif.tracking.squarerootkalman.{SquareRootInformationFilter, SquareRootInformationSmoother}

import scala.util.Random

class SquareRootIMMSmootherSuite extends FlatSpec with Matchers with LazyLogging {

  val r: Random = new scala.util.Random(0)
  val dim: Int = 3
  val numOfTestToDo: Int = 100

  val seeds: List[Int] = List.range(0, 10)
  val numOfEvents: Int = 1000
  val observationStd: Double = 100.0

  val model_0: TargetModel = ConstantVelocityModel(0.5)
  val model_1: TargetModel = ConstantPositionModel(0.0)

  val targetModelLst: List[TargetModel] = List(model_0, model_1)
  val initialStateLst: List[DenseVector[Double]] = List(DenseVector(0.0, 5.0, 0.0, 5.0), DenseVector(0.0, 0.0))
  val multipleModel = new MultipleModelStructure(2, 1.0)

  val modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]] = DenseMatrix(
    (DenseMatrix.eye[Double](model_0.stateDim), DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)).t),
    (DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)), DenseMatrix.eye[Double](model_1.stateDim)))

  val projectionMatrixLst: List[DenseMatrix[Double]] = List(DenseMatrix.eye[Double](model_0.stateDim), DenseMatrix((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)).t)

  val filters: List[SquareRootInformationFilter] = List(new SquareRootInformationFilter(model_0, false), new SquareRootInformationFilter(model_1, false))
  val immFilter = new SquareRootIMMFilter(filters, modelStateProjectionMatrix, false)

  val smoothers: List[SquareRootInformationSmoother] = List(new SquareRootInformationSmoother(model_0, false), new SquareRootInformationSmoother(model_1, false))
  val immSmoother = new SquareRootIMMSmoother(smoothers, modelStateProjectionMatrix, false)

  def validateIMMSmootherResult(states: List[DenseVector[Double]],
                                models: List[Int],
                                immFilterResult: List[IMMFilterResult],
                                immSmootherResult: List[IMMSmootherResult],
                                modelTol: Double,
                                stateTol: Double,
                                isDebugEnabled: Boolean = false): Unit = {


    val numOfSkippedEvent: Int = 1

    val error: List[List[Double]] = List.range(0, states.length).drop(numOfSkippedEvent).reverse.map(idx => {

      val state = states(idx)
      val model = models(idx)

      val filterStates: List[FactoredGaussianDistribution] = immFilterResult(idx).updateResultPerFilter.map(_.updatedStateEstimation)
      val filterStateProbabilities: List[Double] = immFilterResult(idx).updatedLogModeProbability.toArray.toList.map(math.exp)
      val filterModel: Int = argmax(immFilterResult(idx).updatedLogModeProbability)
      val filterFusedState = calculateGaussianMixtureDistribution(filterStates, filterStateProbabilities, modelStateProjectionMatrix(filterModel, ::).t.toArray.toList, filterModel)
      val filterErrorVector = modelStateProjectionMatrix(0, filterModel) * filterFusedState.toGaussianDistribution.m - modelStateProjectionMatrix(0, model) * state

      val smoothStates: List[FactoredGaussianDistribution] = immSmootherResult(idx).smoothResultPerSmoother.map(_.smoothedStateEstimation)
      val smoothProbabilities: List[Double] = immSmootherResult(idx).smoothedLogModeProbability.toArray.toList.map(math.exp)
      val smoothModel: Int = argmax(immSmootherResult(idx).smoothedLogModeProbability)
      val smoothFusedState = calculateGaussianMixtureDistribution(smoothStates, smoothProbabilities, modelStateProjectionMatrix(smoothModel, ::).t.toArray.toList, smoothModel)
      val smoothErrorVector = modelStateProjectionMatrix(0, smoothModel) * smoothFusedState.toGaussianDistribution.m - modelStateProjectionMatrix(0, model) * state

      val filterStateError: Double = filterErrorVector.t * filterErrorVector
      val smoothStateError: Double = smoothErrorVector.t * smoothErrorVector

      val filterModelScore: Double = filterStateProbabilities(model)
      val smoothModelScore: Double = smoothProbabilities(model)

      List(filterStateError, smoothStateError, filterModelScore, smoothModelScore)

    }).transpose

    val immFilterStateMSE: Double = error.head.sum / (numOfEvents - numOfSkippedEvent)
    val immSmootherStateMSE: Double = error(1).sum / (numOfEvents - numOfSkippedEvent)

    val immFilterModelScore: Double = error(2).sum / (numOfEvents - numOfSkippedEvent)
    val immSmootherModelScore: Double = error(3).sum / (numOfEvents - numOfSkippedEvent)

    immFilterStateMSE should be <= stateTol * stateTol
    immSmootherStateMSE should be <= stateTol * stateTol

    immFilterModelScore should be >= (1 - modelTol)
    immSmootherModelScore should be >= (1 - modelTol)

    immFilterModelScore should be <= immSmootherModelScore

  }

  "calculateBackwardLogMixingWeight" should "compute the backward mixing weight" in {

    List.range(0, numOfTestToDo).foreach(_ => {

      val previousModeProbabilities: DenseVector[Double] = getNormalizedVector(dim, r)

      val modelTransitionMatrix: DenseMatrix[Double] = getRandomTransitionMatrix(dim, r)

      val predictedModelProbabilities = modelTransitionMatrix * previousModeProbabilities

      val m1 = modelTransitionMatrix(*, ::) * previousModeProbabilities
      val mixingMatrix = m1(::, *) / predictedModelProbabilities
      val logMixingWeight = log(mixingMatrix)

      val smoothedModeProbabilities: DenseVector[Double] = getNormalizedVector(dim, r)
      val smoothedLogModeProbabilities: DenseVector[Double] = log(smoothedModeProbabilities)

      val m2 = mixingMatrix(::, *) *:* smoothedModeProbabilities
      val expectedRet = m2(*, ::) /:/ sum(m2(::, *)).t

      val ret = exp(calculateBackwardLogMixingWeight(logMixingWeight, smoothedLogModeProbabilities))

      isVectorAlmostEqual(sum(ret(::, *)).t, DenseVector.fill(dim, 1.0)) should be(true)
      isMatrixAlmostEqual(expectedRet, ret) should be(true)

    })

  }


  "SquareRootIMMSmoother" should "detect stationary object" in {

    val multipleModel = new MultipleModelStructure(2, 1.0)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 1, initialStateLst, numOfEvents, multipleModel, observationStd, projectionMatrixLst, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
      val observationLst: List[FactoredGaussianDistribution] = observations.map(x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      })
      val squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]] = smoothers.map(
        f => stepSizeLst.map(f.getTargetModel.calculateSquareRootProcessNoiseCovariance)
      ).transpose
      val stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = smoothers.map(
        f => stepSizeLst.map(f.getTargetModel.calculateStateTransitionMatrix)
      ).transpose

      val immFilterResult = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)
      val immSmootherResult = immSmoother(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)

      validateIMMSmootherResult(states, models, immFilterResult, immSmootherResult, 0.05, 30, false)

    })

  }

  it should "detect moving object" in {

    val multipleModel = new MultipleModelStructure(2, 1.0)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 0, initialStateLst, numOfEvents, multipleModel, observationStd, projectionMatrixLst, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
      val observationLst: List[FactoredGaussianDistribution] = observations.map(x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      })
      val squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]] = smoothers.map(
        f => stepSizeLst.map(f.getTargetModel.calculateSquareRootProcessNoiseCovariance)
      ).transpose
      val stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = smoothers.map(
        f => stepSizeLst.map(f.getTargetModel.calculateStateTransitionMatrix)
      ).transpose

      val immFilterResult = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)
      val immSmootherResult = immSmoother(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)

      validateIMMSmootherResult(states, models, immFilterResult, immSmootherResult, 0.01, 100, false)

    })

  }

  it should "detect object that changes models" in {

    val multipleModel = new MultipleModelStructure(2, 0.999)

    seeds.foreach(seed => {

      val (models, states, observations, stepSizeLst) = MultipleModelTestDataGenerator(targetModelLst, 1, initialStateLst, numOfEvents, multipleModel, observationStd, projectionMatrixLst, seed)

      val logModelTransitionMatrixLst: List[DenseMatrix[Double]] = stepSizeLst.map(multipleModel.getLogModelTransitionMatrix)
      val observationLst: List[FactoredGaussianDistribution] = observations.map(x => {
        val covarianceMatrix: DenseMatrix[Double] = DenseMatrix((observationStd * observationStd, 0.0), (0.0, observationStd * observationStd))
        GaussianDistribution(x, covarianceMatrix).toFactoredGaussianDistribution
      })
      val squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]] = smoothers.map(
        f => stepSizeLst.map(f.getTargetModel.calculateSquareRootProcessNoiseCovariance)
      ).transpose
      val stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]] = smoothers.map(
        f => stepSizeLst.map(f.getTargetModel.calculateStateTransitionMatrix)
      ).transpose

      val immFilterResult = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)
      val immSmootherResult = immSmoother(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerFilterLst, stateTransitionMatrixPerFilterLst)

      validateIMMSmootherResult(states, models, immFilterResult, immSmootherResult, 0.15, 100, false)

    })

  }

}
