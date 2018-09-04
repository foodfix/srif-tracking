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

import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import com.typesafe.scalalogging.LazyLogging
import scalaz.State
import srif.tracking.multipleModel.SquareRootViterbiAlgorithm.{SquareRootViterbiFilterResult, SquareRootViterbiFilterState}
import srif.tracking.squarerootkalman.SquareRootInformationFilter.FilterResult
import srif.tracking.squarerootkalman.{SquareRootInformationFilter, SquareRootInformationSmoother}
import srif.tracking.{FactoredGaussianDistribution, TargetModel, sequence}

class SquareRootViterbiAlgorithm(filters: List[SquareRootInformationFilter],
                                 smoothers: List[SquareRootInformationSmoother],
                                 modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]],
                                 isDebugEnabled: Boolean = false) extends LazyLogging {

  val numOfFilters: Int = filters.length

  /**
    * Return the forward Viterbi filter result.
    *
    * @param logModelTransitionMatrixLst                  logarithmic model transition matrix for each timestamp
    * @param observationLst                               observation for each timestamp
    * @param squareRootProcessNoiseCovariancePerFilterLst process noise covariance matrix in square root form for each timestamp
    * @param stateTransitionMatrixPerFilterLst            state transition matrix for each timestamp
    * @param invStateTransitionMatrixPerFilterLst         inverse of state transition matrix for each timestamp
    * @return filter result for each timestamp
    */
  def apply(logModelTransitionMatrixLst: List[DenseMatrix[Double]],
            observationLst: List[FactoredGaussianDistribution],
            squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]],
            stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]],
            invStateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]],
            initialState: Option[SquareRootViterbiFilterState] = None): List[SquareRootViterbiFilterResult] = {

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
      require(modelStateProjectionMatrix(i, j).rows == filters(i).getTargetModel.stateDim)
      require(modelStateProjectionMatrix(i, j).cols == filters(j).getTargetModel.stateDim)
    }

    val initialViterbiFilterState = initialState.getOrElse({
      val initialLogLikelihoodPerFilter: DenseVector[Double] = DenseVector.fill[Double](numOfFilters, 0)
      val initialFilterResultLst: List[FactoredGaussianDistribution] = filters.map(f => FactoredGaussianDistribution(DenseVector.zeros(f.dim), DenseMatrix.zeros(f.dim, f.dim)))
      SquareRootViterbiFilterState(initialLogLikelihoodPerFilter, initialFilterResultLst)
    })

    sequence(List.range(0, numOfTimeSteps).map(idx =>
      (logModelTransitionMatrixLst(idx),
        observationLst(idx),
        squareRootProcessNoiseCovariancePerFilterLst(idx),
        stateTransitionMatrixPerFilterLst(idx),
        invStateTransitionMatrixPerFilterLst(idx))).
      map(p => filterStep(p._1, p._2, p._3, p._4, p._5))).
      eval(initialViterbiFilterState)

  }

  /**
    * Forward Viterbi filter step at time k.
    *
    * @param logModelTransitionMatrix                  entry (i,j) is the logarithmic probability of change from model j to model i
    * @param observation                               observation :math:`z_k`
    * @param squareRootProcessNoiseCovariancePerFilter [[TargetModel.calculateSquareRootProcessNoiseCovariance]] for each filter at time k
    * @param stateTransitionMatrixPerFilter            [[TargetModel.calculateStateTransitionMatrix]] for each filter at time k
    * @param invStateTransitionMatrixPerFilter         [[TargetModel.calculateInvStateTransitionMatrix()]] for each filter at time k
    * @return
    */
  def filterStep(logModelTransitionMatrix: DenseMatrix[Double],
                 observation: FactoredGaussianDistribution,
                 squareRootProcessNoiseCovariancePerFilter: List[DenseMatrix[Double]],
                 stateTransitionMatrixPerFilter: List[DenseMatrix[Double]],
                 invStateTransitionMatrixPerFilter: List[DenseMatrix[Double]]): State[SquareRootViterbiFilterState, SquareRootViterbiFilterResult] =
    State[SquareRootViterbiFilterState, SquareRootViterbiFilterResult] {
      previousForwardViterbiFilterState => {

        List.range(0, numOfFilters).map(currentFilterIdx => {

          val filterResultBeforeSwitching: List[FilterResult] = List.range(0, numOfFilters).map(previousFilterIdx => {

            val previousFilterUpdatedEstimate: FactoredGaussianDistribution = previousForwardViterbiFilterState.updatedEstimatePerFilter(previousFilterIdx)

            val projectedPreviousFilterUpdatedEstimate: FactoredGaussianDistribution =
              if (previousFilterIdx == currentFilterIdx) previousFilterUpdatedEstimate
              else previousFilterUpdatedEstimate.multiply(modelStateProjectionMatrix(currentFilterIdx, previousFilterIdx))

            filters(currentFilterIdx).
              filterStep(observation,
                squareRootProcessNoiseCovariancePerFilter(currentFilterIdx),
                stateTransitionMatrixPerFilter(currentFilterIdx),
                invStateTransitionMatrixPerFilter(currentFilterIdx)).
              eval(projectedPreviousFilterUpdatedEstimate)
          })

          val transitionLogLikelihood: DenseVector[Double] = logModelTransitionMatrix(currentFilterIdx, ::).t

          val predictedLogLikelihoodPerFilter: DenseVector[Double] = previousForwardViterbiFilterState.updatedLogLikelihoodPerFilter + transitionLogLikelihood

          val observationLogLikelihoodPerFilter: DenseVector[Double] = DenseVector(filterResultBeforeSwitching.map(_.observationLogLikelihood): _*)

          val updatedLogLikelihoodPerFilter: DenseVector[Double] = predictedLogLikelihoodPerFilter + observationLogLikelihoodPerFilter

          val selectedModelIdx: Int = argmax(updatedLogLikelihoodPerFilter)

          List(
            filterResultBeforeSwitching(selectedModelIdx),
            predictedLogLikelihoodPerFilter(selectedModelIdx),
            observationLogLikelihoodPerFilter(selectedModelIdx),
            updatedLogLikelihoodPerFilter(selectedModelIdx),
            selectedModelIdx)

        }).transpose match {
          case (filterResultPerFilter: List[FilterResult]) :: (predictedLogLikelihoodPerFilter: List[Double]) :: (observationLogLikelihoodPerFilter: List[Double]) :: (updatedLogLikelihoodPerFilter: List[Double]) :: (previousModelPerFilter: List[Int]) :: Nil =>
            val currentViterbiFilterResult = SquareRootViterbiFilterResult(
              DenseVector(observationLogLikelihoodPerFilter: _*),
              DenseVector(updatedLogLikelihoodPerFilter: _*),
              filterResultPerFilter,
              Some(previousModelPerFilter)
            )
            (currentViterbiFilterResult.toState, currentViterbiFilterResult)
        }

      }
    }

  /**
    * Perform viterbi smoothing.
    *
    * @param filterResults                                filter result for each timestamp, return of [[SquareRootViterbiAlgorithm.apply]]
    * @param squareRootProcessNoiseCovariancePerFilterLst refer to [[SquareRootViterbiAlgorithm.apply]]
    * @param stateTransitionMatrixPerFilterLst            refer to [[SquareRootViterbiAlgorithm.apply]]
    * @return smooth result for each timestamp
    */
  def smooth(filterResults: List[SquareRootViterbiFilterResult],
             squareRootProcessNoiseCovariancePerFilterLst: List[List[DenseMatrix[Double]]],
             stateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]]): List[(FactoredGaussianDistribution, Int, Double)] = {

    val lastEstimatedModel: Int = argmax(filterResults.last.updatedLogLikelihoodPerFilter)
    val numberOfEvents: Int = filterResults.length

    sequence(List.range(0, numberOfEvents - 1).reverse.map(idx => {
      val nextViterbiFilterResult: SquareRootViterbiFilterResult = filterResults(idx + 1)
      val squareRootProcessNoiseCovariancePerFilter: List[DenseMatrix[Double]] = squareRootProcessNoiseCovariancePerFilterLst(idx + 1)
      val stateTransitionMatrixPerFilter: List[DenseMatrix[Double]] = stateTransitionMatrixPerFilterLst(idx + 1)

      smoothStep(nextViterbiFilterResult, squareRootProcessNoiseCovariancePerFilter, stateTransitionMatrixPerFilter)

    })).eval(
      filterResults.last.filterResultPerFilter(lastEstimatedModel).updatedStateEstimation,
      lastEstimatedModel
    ).reverse ::: List((filterResults.last.filterResultPerFilter(lastEstimatedModel).updatedStateEstimation,
      lastEstimatedModel,
      1.0))

  }

  /**
    * Vertibi smooth step at time k
    *
    * @param viterbiFilterResult                       filter result at time k + 1
    * @param squareRootProcessNoiseCovariancePerFilter [[TargetModel.calculateSquareRootProcessNoiseCovariance]] for each filter at time k + 1
    * @param stateTransitionMatrixPerFilter            [[TargetModel.calculateStateTransitionMatrix]] for each filter at time k + 1
    * @return
    */
  def smoothStep(viterbiFilterResult: SquareRootViterbiFilterResult,
                 squareRootProcessNoiseCovariancePerFilter: List[DenseMatrix[Double]],
                 stateTransitionMatrixPerFilter: List[DenseMatrix[Double]]):
  State[(FactoredGaussianDistribution, Int), (FactoredGaussianDistribution, Int, Double)] =
    State[(FactoredGaussianDistribution, Int), (FactoredGaussianDistribution, Int, Double)] {
      case (nextSmoothedDistribution, nextSelectModel) =>
        val currentSelectModel: Int = viterbiFilterResult.previousModelPerFilter.get(nextSelectModel)

        val currentSmoothedDistributionNotProjected: FactoredGaussianDistribution = smoothers(nextSelectModel).smoothStep(
          viterbiFilterResult.filterResultPerFilter(nextSelectModel),
          squareRootProcessNoiseCovariancePerFilter(nextSelectModel),
          stateTransitionMatrixPerFilter(nextSelectModel)).eval(nextSmoothedDistribution).
          smoothedStateEstimation

        val currentSmoothedDistribution = if (currentSelectModel == nextSelectModel) currentSmoothedDistributionNotProjected
        else currentSmoothedDistributionNotProjected.multiply(modelStateProjectionMatrix(currentSelectModel, nextSelectModel))

        ((currentSmoothedDistribution, currentSelectModel), (currentSmoothedDistribution, currentSelectModel, 1.0))
    }

}

object SquareRootViterbiAlgorithm {

  case class SquareRootViterbiFilterResult(observationLogLikelihoodPerFilter: DenseVector[Double],
                                           updatedLogLikelihoodPerFilter: DenseVector[Double],
                                           filterResultPerFilter: List[FilterResult],
                                           previousModelPerFilter: Option[List[Int]]) {
    def toState: SquareRootViterbiFilterState = SquareRootViterbiFilterState(
      updatedLogLikelihoodPerFilter,
      filterResultPerFilter.map(_.updatedStateEstimation)
    )
  }

  case class SquareRootViterbiFilterState(updatedLogLikelihoodPerFilter: DenseVector[Double],
                                          updatedEstimatePerFilter: List[FactoredGaussianDistribution])

}
