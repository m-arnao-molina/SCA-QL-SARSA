import numpy as np
import time
from datetime import datetime
from database.DatabaseORM import Database
from Metrics import Diversidad as dv

from metaheuristics.Metaheuristic import Metaheuristic
from problems.Kp import Kp

db = Database()


class ScaKp(Metaheuristic):
    def process(self, *args, **kwargs):
        db.startExecution(self.executionId, datetime.now())

        if not self.validateInstance():
            return False

        kp = Kp(self.instanceName, self.instancePath, self.populationSize, self.discretizationScheme)
        kp.readInstance()
        kp.process()

        maxDiversities = np.zeros(7)  # es tamaño 7 porque calculamos 7 diversidades
        diversities, maxDiversities, explorationPercentage, exploitationPercentage, state = dv.ObtenerDiversidadYEstado(
            kp.binMatrix, maxDiversities
        )

        a = 2
        bestFitness = None
        bestSolution = None
        startDatetime = datetime.now()
        iterationsData = []
        for iterationNumber in range(0, self.maxIterations):
            iterationData = {
                'iterationNumber': iterationNumber,
                'executionId': self.executionId,
                'startDatetime': datetime.now()
            }
            iterationTimeStart = time.time()
            processTimeStart = time.process_time()

            # SCA
            r1 = a - iterationNumber * (a / self.maxIterations)
            r4 = np.random.uniform(low=0.0, high=1.0, size=kp.population.shape[0])
            r2 = (2 * np.pi) * np.random.uniform(low=0.0, high=1.0, size=kp.population.shape)
            r3 = np.random.uniform(low=0.0, high=2.0, size=kp.population.shape)

            bestSolutionIndex = kp.solutionsRanking[0]
            bestSolutionOldFitness = kp.fitness[bestSolutionIndex]
            bestSolutionOldBin = kp.binMatrix[bestSolutionIndex]
            bestSolution = kp.population[bestSolutionIndex]
            bestFitness = np.max(kp.fitness)
            avgFitness = np.average(kp.fitness)

            # print(f'---{iterationNumber}---')
            # print('kp.fitness:', kp.fitness)
            # print('bestSolutionOldFitness:', bestSolutionOldFitness)
            # print('bestSolutionOldBin:', bestSolutionOldBin)
            # print('bestSolution:', bestSolution)
            # print('kp.solutionsRanking:', kp.solutionsRanking)
            # print('------')

            kp.population[r4 < 0.5] = kp.population[r4 < 0.5] + np.multiply(
                r1, np.multiply(np.sin(r2[r4 < 0.5]), np.abs(
                    np.multiply(r3[r4 < 0.5], bestSolution) - kp.population[r4 < 0.5]
                ))
            )
            kp.population[r4 >= 0.5] = kp.population[r4 >= 0.5] + np.multiply(
                r1, np.multiply(np.cos(r2[r4 >= 0.5]), np.abs(
                    np.multiply(r3[r4 >= 0.5], bestSolution) - kp.population[r4 >= 0.5]
                ))
            )

            # Se binariza y evalua el fitness de todas las soluciones de la iteración t
            kp.process()

            # Se convierte el Best pisándolo
            if np.max(kp.fitness) <= bestSolutionOldFitness:
                kp.fitness[bestSolutionIndex] = bestSolutionOldFitness
                kp.binMatrix[bestSolutionIndex] = bestSolutionOldBin
            else:
                print(f'fitness[solutionsRank[0]]***: {kp.fitness[bestSolutionIndex]}')

            diversities, maxDiversities, explorationPercentage, exploitationPercentage, state = dv.ObtenerDiversidadYEstado(
                kp.binMatrix, maxDiversities
            )

            iterationTimeEnd = np.round(time.time() - iterationTimeStart, 6)
            processTimeEnd = np.round(time.process_time() - processTimeStart, 6)
            iterationData.update({
                'bestFitness': int(bestFitness),
                'avgFitness': float(avgFitness),
                'fitnessBestIteration': 0,
                'parameters': {
                    'fitness': int(np.max(kp.fitness)),
                    'iterationTime': iterationTimeEnd,
                    'processTime': processTimeEnd,
                    'repairsQuantity': int(kp.repairsQuantity),
                    'diversities': diversities.tolist(),
                    'explorationPercentage': explorationPercentage.tolist(),
                    'exploitationPercentage': exploitationPercentage.tolist(),
                    'state': state
                },
                'endDatetime': datetime.now(),
            })
            iterationsData.append(iterationData)

            if iterationNumber % 100 == 0 and db.insertIterations(iterationsData):
                iterationsData.clear()

        # Si es que queda alguna iteración para insertar
        if len(iterationsData) > 0:
            db.insertIterations(iterationsData)

        # Actualiza la tabla execution_results, sin mejor_solucion
        db.insertExecutionResult(
            fitness=int(bestFitness),
            bestSolution=bestSolution.tolist(),
            executionId=self.executionId,
            startDatetime=startDatetime,
            endDatetime=datetime.now()
        )

        if not db.endExecution(self.executionId, datetime.now()):
            return False

        return True
