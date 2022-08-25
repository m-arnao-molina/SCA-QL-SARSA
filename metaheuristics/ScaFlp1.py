import numpy as np
import time
from datetime import datetime
from database.DatabaseORM import Database
from Metrics import Diversidad as dv

from metaheuristics.Metaheuristic import Metaheuristic
from problems.Flp import Flp
from metaheuristics.ScaMlp import ScaMlp

db = Database()


class ScaFlp(Metaheuristic):
    def process(self, *args, **kwargs):
        db.startExecution(self.executionId, datetime.now())

        if not self.validateInstance():
            return False

        print('--- Paso 1: Instanciando FLP (elección de tiendas) ---')
        flp = Flp(self.instanceName, self.instancePath, self.populationSize, self.discretizationScheme, self.repairType)
        print('--- Paso 2 ---')
        flp.readInstance()
        print('--- Paso 3 ---')
        flp.process()
        print('--- Paso 4: Instanciando SCA MLP (fijación de precios) ---')
        scaMlp = ScaMlp(
            self.executionId, self.populationSize, self.maxIterations, self.discretizationScheme,
            self.repairType, flp.population, flp.facilities, flp.customers, flp.openFacilities, flp.costs,
            flp.customerBudgets, flp.binMatrix, flp.solutionsRanking[0]
        )

        print('flp.population:', flp.population)
        print('flp.population.shape:', flp.population.shape)
        print('flp.facilities:', flp.facilities)
        print('flp.customers:', flp.customers)
        print('flp.openFacilities:', flp.openFacilities)
        print('flp.costs:', flp.costs)
        print('flp.customerBudgets:', flp.customerBudgets)
        print('flp.binMatrix:', flp.binMatrix)
        print('flp.binMatrix.shape:', flp.binMatrix.shape)  # 40 de población x 40 tiendas


        print('--- Paso 5 ---')
        scaMlp.process()
        # exit(0)
        print('--- Paso 6 ---')
        maxDiversities = np.zeros(7)  # es tamaño 7 porque calculamos 7 diversidades
        diversities, maxDiversities, explorationPercentage, exploitationPercentage, state = dv.ObtenerDiversidadYEstado(
            flp.binMatrix, maxDiversities
        )

        a = 2
        globalBestFitness = 0
        globalBestSolutionBin = np.array([])
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

            oldBestSolutionIndex = flp.solutionsRanking[0]
            oldBestSolutionFitness = flp.fitness[oldBestSolutionIndex]
            oldBestSolutionBin = flp.binMatrix[oldBestSolutionIndex]
            oldBestSolution = flp.population[oldBestSolutionIndex]

            flp.population[r4 < 0.5] = flp.population[r4 < 0.5] + np.multiply(
                r1, np.multiply(np.sin(r2[r4 < 0.5]), np.abs(
                    np.multiply(r3[r4 < 0.5], oldBestSolution) - flp.population[r4 < 0.5]
                ))
            )
            flp.population[r4 >= 0.5] = flp.population[r4 >= 0.5] + np.multiply(
                r1, np.multiply(np.cos(r2[r4 >= 0.5]), np.abs(
                    np.multiply(r3[r4 >= 0.5], oldBestSolution) - flp.population[r4 >= 0.5]
                ))
            )

            # Se binariza y evalua el fitness de todas las soluciones de la iteración t
            flp.process()

            scaMlp = ScaMlp(
                self.executionId, self.populationSize, self.maxIterations, self.discretizationScheme,
                self.repairType, flp.population, flp.facilities, flp.customers, flp.openFacilities, flp.costs,
                flp.customerBudgets, flp.binMatrix, flp.solutionsRanking[0]
            )
            scaMlp.process()

            # Se convierte el best pisándolo
            # if np.max(flp.fitness) <= oldBestSolutionFitness:                       # Se está maximizando el costo de transporte, no será que se deba minimizar?
            if np.min(flp.fitness) > oldBestSolutionFitness:
                flp.fitness[oldBestSolutionIndex] = oldBestSolutionFitness
                flp.binMatrix[oldBestSolutionIndex] = oldBestSolutionBin
            else:
                print(f'FLP fitness[solutionsRank[0]]: {flp.fitness[flp.solutionsRanking[0]]}')

            # if np.max(flp.fitness) > globalBestFitness:                             # Se está maximizando el costo de transporte, no será que se deba minimizar?
            if np.min(flp.fitness) < globalBestFitness:
                # globalBestFitness = np.max(flp.fitness)
                globalBestFitness = np.min(flp.fitness)
                globalBestSolutionBin = flp.binMatrix[flp.solutionsRanking[0]]

            diversities, maxDiversities, explorationPercentage, exploitationPercentage, state = dv.ObtenerDiversidadYEstado(
                flp.binMatrix, maxDiversities
            )

            iterationTimeEnd = np.round(time.time() - iterationTimeStart, 6)
            processTimeEnd = np.round(time.process_time() - processTimeStart, 6)
            iterationData.update({
                # 'bestFitness': int(np.max(flp.fitness)),
                'bestFitness': int(np.min(flp.fitness)),
                'avgFitness': float(np.average(flp.fitness)),
                'fitnessBestIteration': int(globalBestFitness),
                'parameters': {
                    # 'fitness': int(np.max(flp.fitness)),
                    'problem': 'FLP',
                    'fitness': int(np.min(flp.fitness)),
                    'currentBestSolution': flp.binMatrix[flp.solutionsRanking[0]].tolist(),
                    'oldBestSolutionFitness': oldBestSolutionFitness,
                    'oldBestSolutionBin': oldBestSolutionBin.tolist(),
                    'iterationTime': iterationTimeEnd,
                    'processTime': processTimeEnd,
                    'repairsQuantity': int(flp.repairsQuantity),
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

        # Actualiza la tabla execution_results
        db.insertExecutionResult(
            fitness=int(globalBestFitness),
            bestSolution=globalBestSolutionBin.tolist(),
            executionId=self.executionId,
            startDatetime=startDatetime,
            endDatetime=datetime.now()
        )

        if not db.endExecution(self.executionId, datetime.now()):
            return False

        return True
