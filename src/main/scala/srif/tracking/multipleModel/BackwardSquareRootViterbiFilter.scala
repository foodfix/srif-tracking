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
import srif.tracking.multipleModel.BackwardSquareRootViterbiFilter.BackwardSquareRootViterbiFilterResult
import srif.tracking.squarerootkalman.BackwardSquareRootInformationFilter
import srif.tracking.squarerootkalman.SquareRootInformationFilter.FilterResult
import srif.tracking.{FactoredGaussianDistribution, TargetModel, sequence}

class BackwardSquareRootViterbiFilter(backwardFilters: List[BackwardSquareRootInformationFilter],
                                      modelStateProjectionMatrix: DenseMatrix[DenseMatrix[Double]],
                                      isDebugEnabled: Boolean = false) extends LazyLogging {
  val numOfFilters: Int = backwardFilters.length

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
            invStateTransitionMatrixPerFilterLst: List[List[DenseMatrix[Double]]]): List[BackwardSquareRootViterbiFilterResult] = {

    val numOfTimeSteps: Int = observationLst.length

    require(logModelTransitionMatrixLst.lengthCompare(numOfTimeSteps) == 0)
    require(squareRootProcessNoiseCovariancePerFilterLst.lengthCompare(numOfTimeSteps) == 0)
    require(stateTransitionMatrixPerFilterLst.lengthCompare(numOfTimeSteps) == 0)
    require(squareRootProcessNoiseCovariancePerFilterLst.forall(_.lengthCompare(numOfFilters) == 0))
    require(stateTransitionMatrixPerFilterLst.forall(_.lengthCompare(numOfFilters) == 0))

    require(stateTransitionMatrixPerFilterLst.forall(ms => (ms, backwardFilters).zipped.forall((m, f) => m.cols == f.dim)))

    require(modelStateProjectionMatrix.rows == numOfFilters)
    require(modelStateProjectionMatrix.cols == numOfFilters)

    for (i <- List.range(0, numOfFilters);
         j <- List.range(0, numOfFilters)) yield {
      require(modelStateProjectionMatrix(i, j).rows == backwardFilters(i).getTargetModel.stateDim)
      require(modelStateProjectionMatrix(i, j).cols == backwardFilters(j).getTargetModel.stateDim)
    }

    val initialLogLikelihoodPerFilter: DenseVector[Double] = DenseVector.fill[Double](numOfFilters, 0)
    val initialFilterResultLst: List[FactoredGaussianDistribution] = backwardFilters.map(f => FactoredGaussianDistribution(DenseVector.zeros(f.dim), DenseMatrix.zeros(f.dim, f.dim)))
    val initialViterbiFilterResult = BackwardSquareRootViterbiFilterResult(initialLogLikelihoodPerFilter, initialFilterResultLst, initialFilterResultLst, None)

    sequence(List.range(0, numOfTimeSteps).reverse.map(idx =>
      (logModelTransitionMatrixLst(idx),
        observationLst(idx),
        squareRootProcessNoiseCovariancePerFilterLst(idx),
        stateTransitionMatrixPerFilterLst(idx),
        invStateTransitionMatrixPerFilterLst(idx))).
      map(p => filterStep(p._1, p._2, p._3, p._4, p._5))).
      eval(initialViterbiFilterResult).reverse

  }

  /**
    * Backward Viterbi filter step
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
                 invStateTransitionMatrixPerFilter: List[DenseMatrix[Double]]): State[BackwardSquareRootViterbiFilterResult, BackwardSquareRootViterbiFilterResult] =
    State[BackwardSquareRootViterbiFilterResult, BackwardSquareRootViterbiFilterResult] {

      nextBackwardFilterResult => {

        List.range(0, numOfFilters).map(currentFilterIdx => {

          val filterResultBeforeSwitchingPerModel: List[FilterResult] = List.range(0, numOfFilters).map(nextFilterIdx => {

            val nextFilterPredictedEstimate: FactoredGaussianDistribution = nextBackwardFilterResult.predictedEstimatePerFilter(nextFilterIdx)

            val projectedNextFilterPredictedEstimate: FactoredGaussianDistribution =
              if (currentFilterIdx == nextFilterIdx) nextFilterPredictedEstimate
              else nextFilterPredictedEstimate.multiply(modelStateProjectionMatrix(currentFilterIdx, nextFilterIdx))

            backwardFilters(currentFilterIdx).
              backwardFilterStep(observation,
                squareRootProcessNoiseCovariancePerFilter(currentFilterIdx),
                stateTransitionMatrixPerFilter(currentFilterIdx),
                invStateTransitionMatrixPerFilter(currentFilterIdx)).
              eval(projectedNextFilterPredictedEstimate)
          })

          val transitionLogLikelihood: DenseVector[Double] = logModelTransitionMatrix(::, currentFilterIdx)

          val logLikelihoodPerFilter: DenseVector[Double] =
            nextBackwardFilterResult.updatedLogLikelihoodPerFilter +
              DenseVector(filterResultBeforeSwitchingPerModel.map(_.observationLogLikelihood): _*) +
              transitionLogLikelihood

          val selectedNextFilterIdx: Int = argmax(logLikelihoodPerFilter)

          List(filterResultBeforeSwitchingPerModel(selectedNextFilterIdx),
            logLikelihoodPerFilter(selectedNextFilterIdx),
            selectedNextFilterIdx)

        }).transpose match {
          case (filterResultPerFilter: List[FilterResult]) :: (logLikelihoodPerFilter: List[Double]) :: (nextModelPerFilter: List[Int]) :: Nil =>
            val currentViterbiFilterResult = BackwardSquareRootViterbiFilterResult(
              DenseVector(logLikelihoodPerFilter: _*),
              filterResultPerFilter.map(_.predictedStateEstimation),
              filterResultPerFilter.map(_.updatedStateEstimation),
              Some(nextModelPerFilter)
            )
            (currentViterbiFilterResult, currentViterbiFilterResult)
        }

      }

    }

}

object BackwardSquareRootViterbiFilter {

  def mapEstResult(estimationResults: List[BackwardSquareRootViterbiFilterResult]): List[(FactoredGaussianDistribution, Int, Double)] = {

    val firstEstimatedModel: Int = argmax(estimationResults.head.updatedLogLikelihoodPerFilter)

    sequence(estimationResults.
      map(filterResult => State[Option[Int], (FactoredGaussianDistribution, Int, Double)] {
        selectModel =>
          val nextModel: Option[Int] = filterResult.nextModelPerFilter.map(_ (selectModel.get))
          val estimatedModel: Int = selectModel.get
          val estimatedState: FactoredGaussianDistribution = filterResult.updatedEstimatePerFilter(estimatedModel)

          (nextModel, (estimatedState, estimatedModel, 1.0))
      })).eval(Some(firstEstimatedModel))

  }

  case class BackwardSquareRootViterbiFilterResult(updatedLogLikelihoodPerFilter: DenseVector[Double],
                                                   predictedEstimatePerFilter: List[FactoredGaussianDistribution],
                                                   updatedEstimatePerFilter: List[FactoredGaussianDistribution],
                                                   nextModelPerFilter: Option[List[Int]])

}
