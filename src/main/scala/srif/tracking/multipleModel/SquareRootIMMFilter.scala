package srif.tracking.multipleModel

import breeze.linalg.{*, DenseMatrix, DenseVector, det, softmax}
import breeze.numerics.exp
import com.typesafe.scalalogging.LazyLogging
import scalaz.State
import srif.tracking.multipleModel.SquareRootIMMFilter._
import srif.tracking.squarerootkalman.SquareRootInformationFilter
import srif.tracking.squarerootkalman.SquareRootInformationFilter.FilterResult
import srif.tracking.{FactoredGaussianDistribution, TargetModel, sequence}

class SquareRootIMMFilter(filters: List[SquareRootInformationFilter], modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]], isDebugEnabled: Boolean = false) extends LazyLogging {

  val numOfFilters: Int = filters.length

  /**
    * Return the IMM filter result.
    *
    * @param logModelTransitionMatrixLst                  logarithmic model transition matrix for each timestamp
    * @param observationLst                               observation for each timestamp
    * @param squareRootProcessNoiseCovariancePerFilterLst process noise covariance matrix in square root form for each timestamp
    * @param stateTransitionMatrixPerFilterLst            state transition matrix for each timestamp
    * @return filter result for each timestamp
    */
  def apply(logModelTransitionMatrixLst: List[DenseMatrix[Double]],
            observationLst: List[FactoredGaussianDistribution],
            squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]],
            stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]]): List[IMMFilterResult] = {

    val numOfTimeSteps: Int = observationLst.length

    require(logModelTransitionMatrixLst.lengthCompare(numOfTimeSteps) == 0)
    require(squareRootProcessNoiseCovariancePerFilterLst.lengthCompare(numOfTimeSteps) == 0)
    require(stateTransitionMatrixPerFilterLst.lengthCompare(numOfTimeSteps) == 0)
    require(squareRootProcessNoiseCovariancePerFilterLst.forall(_.lengthCompare(numOfFilters) == 0))
    require(stateTransitionMatrixPerFilterLst.forall(_.lengthCompare(numOfFilters) == 0))

    require(stateTransitionMatrixPerFilterLst.forall(ms => (ms, filters).zipped.forall((m, f) => m.cols == f.dim)))

    require(modelStateProjectionMatrix.rows == numOfFilters)
    require(modelStateProjectionMatrix.cols == numOfFilters)

    for (i <- List.range(0, numOfFilters);
         j <- List.range(0, numOfFilters)) yield {
      modelStateProjectionMatrix(i, j).rows == filters(j).getTargetModel.stateDim
      modelStateProjectionMatrix(i, j).cols == filters(i).getTargetModel.stateDim
    }

    val initialLogModeProbability: DenseVector[Double] = DenseVector.fill[Double](numOfFilters, math.log(1.0 / numOfFilters))
    val initialFilterResultLst: List[FactoredGaussianDistribution] = filters.map(f => FactoredGaussianDistribution(DenseVector.zeros(f.dim), DenseMatrix.zeros(f.dim, f.dim)))
    val immInitialState: IMMFilterState = IMMFilterState(initialLogModeProbability, initialFilterResultLst)

    sequence(List.range(0, numOfTimeSteps).map(idx =>
      (logModelTransitionMatrixLst(idx),
        observationLst(idx),
        squareRootProcessNoiseCovariancePerFilterLst(idx),
        stateTransitionMatrixPerFilterLst(idx))).
      map(p => filterStep(p._1, p._2, p._3, p._4))).
      eval(immInitialState)

  }

  /**
    * IMM filter step
    *
    * @param logModelTransitionMatrix                  entry (i,j) is the logarithmic probability of change from model j to model i
    * @param observation                               observation :math:`z_k`
    * @param squareRootProcessNoiseCovariancePerFilter [[TargetModel.calculateSquareRootProcessNoiseCovariance]] for each filter at time k
    * @param stateTransitionMatrixPerFilter            [[TargetModel.calculateStateTransitionMatrix]] for each filter at time k
    * @return
    */
  def filterStep(logModelTransitionMatrix: DenseMatrix[Double],
                 observation: FactoredGaussianDistribution,
                 squareRootProcessNoiseCovariancePerFilter: List[DenseMatrix[Double]],
                 stateTransitionMatrixPerFilter: List[DenseMatrix[Double]]): State[IMMFilterState, IMMFilterResult] = State[IMMFilterState, IMMFilterResult] {
    previousIMMState => {

      val (mixedStateEstimatePerFilter: List[FactoredGaussianDistribution], predictedLogModelProbability: DenseVector[Double], logMixingWeight: DenseMatrix[Double]) =
        modelConditionedReinitialization(previousIMMState.updatedLogModeProbability,
          logModelTransitionMatrix, previousIMMState.updateStateEstimationLst)

      val filterResultPerFilter: List[FilterResult] = modelConditionedFiltering(observation,
        squareRootProcessNoiseCovariancePerFilter,
        stateTransitionMatrixPerFilter,
        mixedStateEstimatePerFilter)

      val logModeProbability: DenseVector[Double] = modelProbabilityUpdate(predictedLogModelProbability, filterResultPerFilter.map(_.observationLogLikelihood))
      require(logModeProbability.forall(!_.isNaN))

      if (isDebugEnabled) {

        logger.debug("=== IMM Filter Step ===")

        logger.debug(s"\nModelTransitionMatrix: \n${exp(logModelTransitionMatrix)}")
        logger.debug(s"\nobservation: \n${observation.toGaussianDistribution.m}")
        logger.debug(s"\npredictedLogModelProbability: \n${exp(predictedLogModelProbability)}")

        mixedStateEstimatePerFilter.foreach(est => {
          if (det(est.R) != 0) {
            logger.debug(s"\nmixedStateEstimatePerFilter m: \n${est.toGaussianDistribution.m}")
            logger.debug(s"\nmixedStateEstimatePerFilter V: \n${est.toGaussianDistribution.V}")
          }
        })

        logger.debug(s"\nobservationLogLikelihood: \n${filterResultPerFilter.map(_.observationLogLikelihood)}")
        logger.debug(s"\nupdatedLogModeProbability: \n${exp(logModeProbability)}")

        filterResultPerFilter.map(_.updatedStateEstimation).foreach(est => {
          if (det(est.R) != 0) {
            val m = est.toGaussianDistribution.m
            val V = est.toGaussianDistribution.V

            logger.debug(s"\nupdatedStateEstimation m: \n$m")
            logger.debug(s"\nupdatedStateEstimation V: \n$V")
          }
        })

      }

      (IMMFilterState(logModeProbability, filterResultPerFilter.map(_.updatedStateEstimation)),
        IMMFilterResult(logModeProbability, logMixingWeight, filterResultPerFilter))
    }
  }


  /**
    * Model-set conditioned (re)initialization
    *
    * @param previousLogModelProbabilities    :math:`P(m_{k-1} | z_1,...,z_{k-1})`
    * @param logModelTransitionMatrix         refer to [[SquareRootIMMFilter.filterStep]]
    * @param previousStateEstimationPerFilter state estimation :math:`P(x_{k-1} | m_{k-1}=j, z_1,...,z_{k-1})` for each j
    * @return mixed state estimation :math:`P(x_{k-1} | m_k=i, z_1,...,z_{k-1})` for each i
    *         predicted logarithmic model probability, return of [[SquareRootIMMFilter.calculateLogMixingWeight]]
    *         logarithmic of mixing weight, return of [[SquareRootIMMFilter.calculateLogMixingWeight]]
    *
    */
  def modelConditionedReinitialization(previousLogModelProbabilities: DenseVector[Double],
                                       logModelTransitionMatrix: DenseMatrix[Double],
                                       previousStateEstimationPerFilter: List[FactoredGaussianDistribution]):
  (List[FactoredGaussianDistribution], DenseVector[Double], DenseMatrix[Double]) = {

    val (logMixingWeight: DenseMatrix[Double], predictedLogModelProbability: DenseVector[Double]) =
      calculateLogMixingWeight(previousLogModelProbabilities, logModelTransitionMatrix)

    require(predictedLogModelProbability.forall(!_.isNaN))

    val mixedStateEstimationPerFilter: List[FactoredGaussianDistribution] = List.range(0, numOfFilters).map(filterIndex => {
      val logMixingWeightPerFilter: DenseVector[Double] = logMixingWeight(filterIndex, ::).t

      calculateGaussianMixtureDistribution(
        previousStateEstimationPerFilter,
        exp(logMixingWeightPerFilter).toArray.toList,
        modelStateProjectionMatrix(filterIndex, ::).t.toArray.toList,
        filterIndex)
    })

    (mixedStateEstimationPerFilter,
      predictedLogModelProbability,
      logMixingWeight)
  }

  /**
    * Model-conditioned filtering
    *
    * @param observation                               refer to [[SquareRootIMMFilter.filterStep]]
    * @param squareRootProcessNoiseCovariancePerFilter refer to [[SquareRootIMMFilter.filterStep]]
    * @param stateTransitionMatrixPerFilter            refer to [[SquareRootIMMFilter.filterStep]]
    * @param stateEstimationPerFilter                  return of [[SquareRootIMMFilter.modelConditionedReinitialization]]
    * @return state estimation :math:`P(x_k | m_k=i, z_1,...,z_k)` for each i
    */
  def modelConditionedFiltering(observation: FactoredGaussianDistribution,
                                squareRootProcessNoiseCovariancePerFilter: List[DenseMatrix[Double]],
                                stateTransitionMatrixPerFilter: List[DenseMatrix[Double]],
                                stateEstimationPerFilter: List[FactoredGaussianDistribution]): List[FilterResult] = {
    List.range(0, numOfFilters).map(filterIndex => {
      filters(filterIndex).filterStep(observation,
        squareRootProcessNoiseCovariancePerFilter(filterIndex),
        stateTransitionMatrixPerFilter(filterIndex)).
        eval(stateEstimationPerFilter(filterIndex))
    })
  }

  /**
    * Model probability update
    *
    * @param predictedLogModelProbability   return of [[SquareRootIMMFilter.calculateLogMixingWeight]]
    * @param observationLikelihoodPerFilter refer to [[SquareRootInformationFilter.computeLogObservationProbability]]
    * @return :math:`P(m_k | z_1,...,z_k)`
    */
  def modelProbabilityUpdate(predictedLogModelProbability: DenseVector[Double],
                             observationLikelihoodPerFilter: List[Double]): DenseVector[Double] = {

    val modeLikelihoods: DenseVector[Double] =
      if (observationLikelihoodPerFilter.exists(_.isInfinite)) predictedLogModelProbability
      else if (observationLikelihoodPerFilter.exists(_.isNaN)) predictedLogModelProbability
      else DenseVector(observationLikelihoodPerFilter: _*) + predictedLogModelProbability

    (modeLikelihoods - softmax(modeLikelihoods)).map(_ max math.log(MIN_PROBABILITY))
  }

}

object SquareRootIMMFilter {

  /**
    * Calculate the mixing weight
    *
    * @param logModelProbabilities    refer to [[SquareRootIMMFilter.modelConditionedReinitialization]]
    * @param logModelTransitionMatrix refer to [[SquareRootIMMFilter.modelConditionedReinitialization]]
    * @return mixing weight, entry (i,j) is :math:`P(m_{k-1}=j | m_k=i, z_1,...,z_{k-1})`
    *         predicted logarithmic model probabilities for each model, :math:`P(m_k | z_1,...,z_{k-1})`
    */
  def calculateLogMixingWeight(logModelProbabilities: DenseVector[Double],
                               logModelTransitionMatrix: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
    val m = logModelTransitionMatrix(*, ::) + logModelProbabilities
    val predictedLogModelProbability = softmax(m(*, ::))
    (m(::, *) - predictedLogModelProbability, predictedLogModelProbability)
  }

  case class IMMFilterResult(updatedLogModeProbability: DenseVector[Double],
                             logMixingWeight: DenseMatrix[Double],
                             updateResultPerFilter: List[FilterResult])

  case class IMMFilterState(updatedLogModeProbability: DenseVector[Double], updateStateEstimationLst: List[FactoredGaussianDistribution])

}