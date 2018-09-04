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

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, det, softmax}
import breeze.numerics.exp
import com.typesafe.scalalogging.LazyLogging
import scalaz.State
import srif.tracking._
import srif.tracking.multipleModel.SquareRootIMMFilter.IMMFilterResult
import srif.tracking.multipleModel.SquareRootIMMSmoother._
import srif.tracking.squarerootkalman.SquareRootInformationFilter.FilterResult
import srif.tracking.squarerootkalman.SquareRootInformationSmoother
import srif.tracking.squarerootkalman.SquareRootInformationSmoother.SmoothResult

/**
  * Square Root IMM smoother
  *
  * @param smoothers                  list of smoothers.
  * @param modelStateProjectionMatrix used to project the state of one model to another model
  * @param isDebugEnabled             true if show debug message
  */
class SquareRootIMMSmoother(smoothers: List[SquareRootInformationSmoother], modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]], isDebugEnabled: Boolean = false) extends LazyLogging {

  lazy val immFilter: SquareRootIMMFilter = new SquareRootIMMFilter(smoothers.map(_.filter), modelStateProjectionMatrix, isDebugEnabled)
  val numOfSmoothers: Int = smoothers.length

  /**
    * Return the IMM Smoother result.
    *
    * @param logModelTransitionMatrixLst                    logarithmic model transition matrix for each timestamp
    * @param observationLst                                 observation for each timestamp
    * @param squareRootProcessNoiseCovariancePerSmootherLst process noise covariance matrix in square root form for each timestamp
    * @param stateTransitionMatrixPerSmootherLst            state transition matrix for each timestamp
    * @param invStateTransitionMatrixPerFilterLst           inverse of state transition matrix for each timestamp
    * @return smooth result for each timestamp
    */
  def apply(logModelTransitionMatrixLst: List[DenseMatrix[Double]],
            observationLst: List[FactoredGaussianDistribution],
            squareRootProcessNoiseCovariancePerSmootherLst: List[List[DenseMatrix[Double]]],
            stateTransitionMatrixPerSmootherLst: List[List[DenseMatrix[Double]]],
            invStateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]]): List[IMMSmootherResult] = {

    val numOfTimeSteps: Int = observationLst.length

    val immFilterResult: List[IMMFilterResult] = immFilter(logModelTransitionMatrixLst, observationLst, squareRootProcessNoiseCovariancePerSmootherLst, stateTransitionMatrixPerSmootherLst, invStateTransitionMatrixPerFilterLst)
    val updatedLogModeProbabilityLst = immFilterResult.map(_.updatedLogModeProbability)

    val initialSmoothResultPerSmoother: List[SmoothResult] = immFilterResult.last.updateResultPerFilter.map(x => SmoothResult(x.updatedStateEstimation, x.observationLogLikelihood))

    val immSmootherResult: List[IMMSmootherResult] = sequence(List.range(1, numOfTimeSteps).map(idx => {
      (immFilterResult(idx),
        squareRootProcessNoiseCovariancePerSmootherLst(idx),
        stateTransitionMatrixPerSmootherLst(idx),
        updatedLogModeProbabilityLst(idx - 1),
        logModelTransitionMatrixLst(idx))
    }).reverse.map(p => smoothStep(p._1, p._2, p._3, p._4, p._5))).
      eval(IMMSmootherState(immFilterResult.last.updatedLogModeProbability, initialSmoothResultPerSmoother)).
      reverse ::: List(IMMSmootherResult(immFilterResult.last.updatedLogModeProbability, initialSmoothResultPerSmoother))

    immSmootherResult
  }

  /**
    * IMM smooth step at time stamp k
    *
    * @param nextImmFilterResult                         IMM filter step at time k+1
    * @param squareRootProcessNoiseCovariancePerSmoother [[TargetModel.calculateSquareRootProcessNoiseCovariance]] at time k+1
    * @param stateTransitionMatrixPerSmoother            [[TargetModel.calculateStateTransitionMatrix]] at time k+1
    * @param updatedLogModelProbability                  jth entry is :math:`P(m_k=j | z_1,...,z_k)`
    * @param logModelTransitionMatrix                    entry (i, j) is :math:`P(m_{k+1} = i | m_k = j)`
    * @return
    */
  def smoothStep(nextImmFilterResult: IMMFilterResult,
                 squareRootProcessNoiseCovariancePerSmoother: List[DenseMatrix[Double]],
                 stateTransitionMatrixPerSmoother: List[DenseMatrix[Double]],
                 updatedLogModelProbability: DenseVector[Double],
                 logModelTransitionMatrix: DenseMatrix[Double]): State[IMMSmootherState, IMMSmootherResult] = State[IMMSmootherState, IMMSmootherResult] {

    nextSmoothedState => {

      val nextSmoothedLogModelProbabilities: DenseVector[Double] = nextSmoothedState.smoothedLogModeProbability
      val logMixingWeight: DenseMatrix[Double] = nextImmFilterResult.logMixingWeight
      val nextSmoothResultPerSmoother: List[SmoothResult] = nextSmoothedState.smoothResultPerSmoother
      val nextFilterResultPerSmoother: List[FilterResult] = nextImmFilterResult.updateResultPerFilter

      val backwardLogMixingMatrix: DenseMatrix[Double] = calculateBackwardLogMixingWeight(logMixingWeight, nextSmoothedLogModelProbabilities)
      val mixedStateEstimationPerSmoother: List[FactoredGaussianDistribution] = modelConditionedReinitialization(backwardLogMixingMatrix, nextSmoothResultPerSmoother.map(_.smoothedStateEstimation))

      val smoothResultPerSmoother: List[SmoothResult] = modelConditionedSmoothing(nextFilterResultPerSmoother, squareRootProcessNoiseCovariancePerSmoother, stateTransitionMatrixPerSmoother, mixedStateEstimationPerSmoother)

      val smoothedLogModelProbability: DenseVector[Double] = modelProbabilitySmooth(updatedLogModelProbability, logModelTransitionMatrix, nextFilterResultPerSmoother, nextSmoothResultPerSmoother)

      smoothResultPerSmoother.foreach(x => {
        require(x.smoothedStateEstimation.zeta.forall(!_.isNaN))
      })

      mixedStateEstimationPerSmoother.foreach(x => {
        require(x.zeta.forall(!_.isNaN))
      })

      if (isDebugEnabled) {

        logger.debug("=== IMM Smoother Step ===")

        logger.debug(s"\nlogMixingWeight: \n${exp(logMixingWeight)}")

        mixedStateEstimationPerSmoother.foreach(est => {
          if (det(est.R) != 0) {
            logger.debug(s"\nmixedStateEstimationPerSmoother m: \n${est.toGaussianDistribution.m}")
            logger.debug(s"\nmixedStateEstimationPerSmoother V: \n${est.toGaussianDistribution.V}")
          }
        })

        smoothResultPerSmoother.map(_.smoothedStateEstimation).foreach(est => {
          if (det(est.R) != 0) {
            val m = est.toGaussianDistribution.m
            val V = est.toGaussianDistribution.V

            logger.debug(s"\nsmoothResultPerSmoother m: \n$m")
            logger.debug(s"\nsmoothResultPerSmoother V: \n$V")
          }
        })

        logger.debug(s"\nsmoothedLogModelProbability: \n${exp(smoothedLogModelProbability)}")

      }

      (IMMSmootherState(smoothedLogModelProbability, smoothResultPerSmoother),
        IMMSmootherResult(smoothedLogModelProbability, smoothResultPerSmoother))

    }

  }

  /**
    * Return the mixed state estimate
    *
    * @param backwardLogMixingMatrix    return of [[SquareRootIMMSmoother.calculateBackwardLogMixingWeight]]
    * @param stateEstimationPerSmoother ith entry is :math:`P(x_{k+1} | m_{k+1}=i, z_1,...,z_T)`
    * @return jth entry is :math:`P(x_{k+1} | m_k=j z_1,...,z_T)`
    */
  def modelConditionedReinitialization(backwardLogMixingMatrix: DenseMatrix[Double],
                                       stateEstimationPerSmoother: List[FactoredGaussianDistribution]): List[FactoredGaussianDistribution] = {

    List.range(0, numOfSmoothers).map(smootherIndex => {
      val logMixingWeightPerFilter: DenseVector[Double] = backwardLogMixingMatrix(::, smootherIndex)
      calculateGaussianMixtureDistribution(
        stateEstimationPerSmoother,
        exp(logMixingWeightPerFilter).toArray.toList,
        modelStateProjectionMatrix(smootherIndex, ::).t.toArray.toList,
        smootherIndex)
    })

  }

  /**
    * Model-conditioned smoothering
    *
    * @param filterResultPerSmoother                     ith element is to describe :math:`P(x_{k+1} | m_{k+1} = i, z_1,...,z_{k+1})`
    * @param squareRootProcessNoiseCovariancePerSmoother refer to [[SquareRootIMMSmoother.smoothStep]]
    * @param stateTransitionMatrixPerSmoother            refer to [[SquareRootIMMSmoother.smoothStep]]
    * @param stateEstimationPerSmoother                  return of [[SquareRootIMMSmoother.modelConditionedReinitialization]]
    * @return jth element is :math:`P(x_k | m_k = j, z_1,...,z_T)`
    */
  def modelConditionedSmoothing(filterResultPerSmoother: List[FilterResult],
                                squareRootProcessNoiseCovariancePerSmoother: List[DenseMatrix[Double]],
                                stateTransitionMatrixPerSmoother: List[DenseMatrix[Double]],
                                stateEstimationPerSmoother: List[FactoredGaussianDistribution]): List[SmoothResult] = {
    List.range(0, numOfSmoothers).map(smootherIndex => {
      smoothers(smootherIndex).
        smoothStep(filterResultPerSmoother(smootherIndex),
          squareRootProcessNoiseCovariancePerSmoother(smootherIndex),
          stateTransitionMatrixPerSmoother(smootherIndex)).
        eval(stateEstimationPerSmoother(smootherIndex))
    })
  }

  /**
    * Model probability smooth
    *
    * @param updatedLogModelProbability  refer to [[SquareRootIMMSmoother.smoothStep]]
    * @param logModelTransitionMatrix    refer to [[SquareRootIMMSmoother.smoothStep]]
    * @param nextFilterResultPerSmoother ith element is :math:`P(x_{k+1} | m_{k+1} = i, z_1,...,z_{k+1})`
    * @param nextSmoothResultPerSmoother ith element is :math:`P(x_{k+1} | m_{k+1} = i, z_1,...,z_T)`
    * @return jth entry is :math:`P(m_k=j | z_1,...,z_T)`
    */
  def modelProbabilitySmooth(updatedLogModelProbability: DenseVector[Double],
                             logModelTransitionMatrix: DenseMatrix[Double],
                             nextFilterResultPerSmoother: List[FilterResult],
                             nextSmoothResultPerSmoother: List[SmoothResult]): DenseVector[Double] = {

    val u = List.range(0, numOfSmoothers).map(j => {
      val v = List.range(0, numOfSmoothers).
        map(i => {

          val nextSmoothedStateEstimate: FactoredGaussianDistribution = nextSmoothResultPerSmoother(i).smoothedStateEstimation
          val nextPredictedStateEstimate: FactoredGaussianDistribution = nextFilterResultPerSmoother(j).predictedStateEstimation

          sumFactoredGaussianDistribution(nextSmoothedStateEstimate, nextPredictedStateEstimate, -modelStateProjectionMatrix(i, j))._1.multiply(smoothers(i).getTargetModel.observationMatrix).logLikelihood

        })
      softmax(DenseVector(v: _*) + logModelTransitionMatrix(::, j))
    })

    val unnormalizedLogProbability: DenseVector[Double] = DenseVector(u: _*) + updatedLogModelProbability
    (unnormalizedLogProbability - softmax(unnormalizedLogProbability)).map(_ max math.log(MIN_PROBABILITY))
  }

}

object SquareRootIMMSmoother {

  /**
    * Return the logarithmic of the backward mixing weight, entry (i, j) is :math:`P(m_{k+1} = i | m_k = j, z_1,...,z_T )`
    *
    * @param logMixingWeight             entry (i, j) is :math:`P(m_k=j | m_{k+1}=i, z_1,...,z_k)`
    * @param smoothedLogModelProbability :math:`P(m_{k+1} | z_1,...,z_T)`
    * @return
    */
  def calculateBackwardLogMixingWeight(logMixingWeight: DenseMatrix[Double], smoothedLogModelProbability: DenseVector[Double]): DenseMatrix[Double] = {

    val m = logMixingWeight(::, *) + smoothedLogModelProbability
    m(*, ::) - softmax(m(::, *)).t
  }

  /**
    * Fuse the estimation result.
    *
    * @param estimationResults          return of [[SquareRootIMMSmoother]]
    * @param modelStateProjectionMatrix refer to [[SquareRootIMMSmoother]]
    * @return fused estimation states
    *         estimated model index
    *         estimtaed model probability
    */
  def fuseEstResult(estimationResults: List[IMMSmootherResult],
                    modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]]): List[(FactoredGaussianDistribution, Int, Double, Double)] = {

    estimationResults.map(estimationResult => {

      val selectedModel: Int = argmax(estimationResult.smoothedLogModeProbability)

      val estStates: List[FactoredGaussianDistribution] = estimationResult.smoothResultPerSmoother.map(_.smoothedStateEstimation)
      val estStateProbabilities: List[Double] = estimationResult.smoothedLogModeProbability.toArray.toList.map(math.exp)
      val fusedEstimationState = calculateGaussianMixtureDistribution(estStates, estStateProbabilities, modelStateProjectionMatrix(selectedModel, ::).t.toArray.toList, selectedModel)

      (fusedEstimationState, selectedModel, estStateProbabilities(selectedModel),
        estimationResult.smoothResultPerSmoother(selectedModel).observationLogLikelihood)

    })

  }

  case class IMMSmootherState(smoothedLogModeProbability: DenseVector[Double], smoothResultPerSmoother: List[SmoothResult])

  case class IMMSmootherResult(smoothedLogModeProbability: DenseVector[Double],
                               smoothResultPerSmoother: List[SmoothResult])

}