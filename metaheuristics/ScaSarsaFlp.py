import numpy as np
import time
from datetime import datetime
from database.DatabaseORM import Database
from Metrics import Diversidad as dv

from metaheuristics.Metaheuristic import Metaheuristic
from machineLearnings.Sarsa import Sarsa
from problems.Flp import Flp

db = Database()

transferFunctions = ['S1', 'S2', 'S3', 'S4', 'V1', 'V2', 'V3', 'V4']
binarizationOperators = ['STANDARD', 'COMPLEMENT', 'STATIC', 'ELITIST', 'ELITIST_ROULETTE']
dsActions = [(tf, ob) for tf in transferFunctions for ob in binarizationOperators]


class ScaSarsaFlp(Metaheuristic):
    def __init__(
            self, executionId, instanceName, instanceFile, instanceDirectory, populationSize, maxIterations,
            discretizationScheme, repairType, policy, rewardType, qlAlpha, qlGamma, qlAlphaType
    ):
        super().__init__(
            executionId, instanceName, instanceFile, instanceDirectory, populationSize, maxIterations,
            discretizationScheme, repairType
        )
        self.policy = policy
        self.rewardType = rewardType
        self.qlAlpha = qlAlpha
        self.qlGamma = qlGamma
        self.qlAlphaType = qlAlphaType

    def process(self, *args, **kwargs):
        db.startExecution(self.executionId, datetime.now())

        if not self.validateInstance():
            return False

        # QLearning
        agent = Sarsa(
            self.qlGamma, self.qlAlphaType, self.rewardType, self.maxIterations, dsActions, 2, qlAlpha=self.qlAlpha
        )
        dsIndex = agent.getAction(0, self.policy)
        self.discretizationScheme = {
            'transferFunction': dsActions[dsIndex][0],
            'binarizationOperator': dsActions[dsIndex][1]
        }

        flp = Flp(self.instanceName, self.instancePath, self.populationSize, self.discretizationScheme, self.repairType)
        flp.readInstance()
        flp.process()

        maxDiversities = np.zeros(7)  # es tamaño 7 porque calculamos 7 diversidades
        diversities, maxDiversities, explorationPercentage, exploitationPercentage, state = dv.ObtenerDiversidadYEstado(
            flp.binMatrix, maxDiversities
        )

        # Sarsa
        currentState = state[0]  # Estamos midiendo según Diversidad "DimensionalHussain"

        a = 2
        globalBestFitness = 0
        # bestSolution = None
        globalBestSolutionOldBin = np.array([])
        startDatetime = datetime.now()
        iterationsData = []
        for iterationNumber in range(0, self.maxIterations):
            # print('iterationNumber:', iterationNumber)
            iterationData = {
                'iterationNumber': iterationNumber,
                'executionId': self.executionId,
                'startDatetime': datetime.now()
            }
            iterationTimeStart = time.time()
            processTimeStart = time.process_time()

            # SCA
            r1 = a - iterationNumber * (a / self.maxIterations)
            r2 = (2 * np.pi) * np.random.uniform(low=0.0, high=1.0, size=flp.population.shape)
            r3 = 2 * np.random.uniform(low=0.0, high=1.0, size=flp.population.shape)
            r4 = np.random.uniform(low=0.0, high=1.0, size=flp.population.shape[0])

            bestSolutionIndex = flp.solutionsRanking[0]
            bestSolutionOldFitness = flp.fitness[bestSolutionIndex]
            bestSolutionOldBin = flp.binMatrix[bestSolutionIndex]
            bestSolution = flp.population[bestSolutionIndex]
            bestFitness = np.max(flp.fitness)
            avgFitness = np.average(flp.fitness)

            flp.population[r4 < 0.5] = flp.population[r4 < 0.5] + np.multiply(
                r1, np.multiply(np.sin(r2[r4 < 0.5]), np.abs(
                    np.multiply(r3[r4 < 0.5], bestSolution) - flp.population[r4 < 0.5]
                ))
            )
            flp.population[r4 >= 0.5] = flp.population[r4 >= 0.5] + np.multiply(
                r1, np.multiply(np.cos(r2[r4 >= 0.5]), np.abs(
                    np.multiply(r3[r4 >= 0.5], bestSolution) - flp.population[r4 >= 0.5]
                ))
            )

            # Escogemos esquema desde SARSA
            dsIndex = agent.getAction(currentState, self.policy)
            oldState = currentState  # Rescatamos estado actual
            self.discretizationScheme = {
                'transferFunction': dsActions[dsIndex][0],
                'binarizationOperator': dsActions[dsIndex][1]
            }
            flp.discretizationScheme = self.discretizationScheme

            # Se binariza y evalua el fitness de todas las soluciones de la iteración t
            flp.process()

            # Se convierte el Best pisándolo
            if bestFitness <= bestSolutionOldFitness:
                flp.fitness[bestSolutionIndex] = bestSolutionOldFitness
                flp.binMatrix[bestSolutionIndex] = bestSolutionOldBin
            else:
                print(f'fitness[solutionsRank[0]]***: {flp.fitness[bestSolutionIndex]}')

            if bestFitness > globalBestFitness:
                globalBestFitness = np.max(flp.fitness)
                globalBestSolutionOldBin = bestSolutionOldBin

            diversities, maxDiversities, explorationPercentage, exploitationPercentage, state = dv.ObtenerDiversidadYEstado(
                flp.binMatrix, maxDiversities
            )

            # Observamos, y recompensa/castigo. Actualizamos Tabla Q
            currentState = state[0]
            agent.updateQtable(bestFitness, dsIndex, currentState, oldState, iterationNumber)

            iterationTimeEnd = np.round(time.time() - iterationTimeStart, 6)
            processTimeEnd = np.round(time.process_time() - processTimeStart, 6)
            iterationData.update({
                'bestFitness': int(bestFitness),
                'avgFitness': float(avgFitness),
                'fitnessBestIteration': int(globalBestFitness),
                'parameters': {
                    'fitness': int(np.max(flp.fitness)),
                    'iterationTime': iterationTimeEnd,
                    'processTime': processTimeEnd,
                    'repairsQuantity': int(flp.repairsQuantity),
                    'discretizationScheme': self.discretizationScheme,
                    'diversities': diversities.tolist(),
                    'explorationPercentage': explorationPercentage.tolist(),
                    'exploitationPercentage': exploitationPercentage.tolist(),
                    'state': state
                },
                'endDatetime': datetime.now()
            })
            iterationsData.append(iterationData)

            if iterationNumber % 100 == 0 and db.insertIterations(iterationsData):
                iterationsData.clear()

        # Si es que queda alguna iteración para insertar
        if len(iterationsData) > 0:
            db.insertIterations(iterationsData)

        # Actualiza la tabla execution_results, sin mejor_solucion
        db.insertExecutionResult(
            fitness=int(globalBestFitness),
            # bestSolution=bestSolution.tolist(),
            bestSolution=globalBestSolutionOldBin.tolist(),
            # bestSolution=agent.getQTable().tolist(),
            executionId=self.executionId,
            startDatetime=startDatetime,
            endDatetime=datetime.now()
        )

        if not db.endExecution(self.executionId, datetime.now()):
            return False

        return True
