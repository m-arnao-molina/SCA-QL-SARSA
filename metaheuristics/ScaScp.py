import numpy as np
import time
from datetime import datetime
from database.DatabaseORM import Database
from Metrics import Diversidad as dv

from metaheuristics.Metaheuristic import Metaheuristic
from problems.Scp import Scp

db = Database()


class ScaScp(Metaheuristic):
    def process(self, *args, **kwargs):
        db.startExecution(self.executionId, datetime.now())

        if not self.validateInstance():
            return False

        scp = Scp(self.instanceName, self.instancePath, self.populationSize, self.discretizationScheme, self.repairType)
        scp.readInstance()
        scp.process()

        maxDiversities = np.zeros(7)  # es tamaño 7 porque calculamos 7 diversidades
        diversities, maxDiversities, explorationPercentage, exploitationPercentage, state = dv.ObtenerDiversidadYEstado(
            scp.binMatrix, maxDiversities
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
            r4 = np.random.uniform(low=0.0, high=1.0, size=scp.population.shape[0])
            r2 = (2 * np.pi) * np.random.uniform(low=0.0, high=1.0, size=scp.population.shape)
            r3 = np.random.uniform(low=0.0, high=2.0, size=scp.population.shape)
            bestSolutionIndex = scp.solutionsRanking[0]
            bestSolutionOldBin = scp.binMatrix[bestSolutionIndex]
            bestSolution = scp.population[bestSolutionIndex]
            bestFitness = np.min(scp.fitness)
            avgFitness = np.average(scp.fitness)

            # print(f'---{iterationNumber}---')
            # print('scp.fitness:', scp.fitness)
            # #print('bestSolutionOldFitness:', bestSolutionOldFitness)
            # print('bestSolutionOldBin:', bestSolutionOldBin)
            # print('bestSolution:', bestSolution)
            # print('scp.solutionsRanking:', scp.solutionsRanking)
            # print('------')

            scp.population[r4 < 0.5] = scp.population[r4 < 0.5] + np.multiply(
                r1, np.multiply(np.sin(r2[r4 < 0.5]), np.abs(
                    np.multiply(r3[r4 < 0.5], bestSolution) - scp.population[r4 < 0.5]
                ))
            )
            scp.population[r4 >= 0.5] = scp.population[r4 >= 0.5] + np.multiply(
                r1, np.multiply(np.cos(r2[r4 >= 0.5]), np.abs(
                    np.multiply(r3[r4 >= 0.5], bestSolution) - scp.population[r4 >= 0.5]
                ))
            )

            # Se binariza y evalua el fitness de todas las soluciones de la iteración t
            scp.process()

            # Conservo el Best
            if scp.fitness[bestSolutionIndex] > bestFitness:
                scp.fitness[bestSolutionIndex] = bestFitness
                scp.binMatrix[bestSolutionIndex] = bestSolutionOldBin

            diversities, maxDiversities, explorationPercentage, exploitationPercentage, state = dv.ObtenerDiversidadYEstado(
                scp.binMatrix, maxDiversities
            )

            iterationTimeEnd = np.round(time.time() - iterationTimeStart, 6)
            processTimeEnd = np.round(time.process_time() - processTimeStart, 6)
            iterationData.update({
                'bestFitness': int(bestFitness),
                'avgFitness': float(avgFitness),
                'fitnessBestIteration': 0,
                'parameters': {
                    'fitness': int(np.min(scp.fitness)),
                    'iterationTime': iterationTimeEnd,
                    'processTime': processTimeEnd,
                    'repairsQuantity': int(scp.repairsQuantity),
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
