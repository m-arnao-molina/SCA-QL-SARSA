import numpy as np
import time
from datetime import datetime
from database.DatabaseORM import Database
from Metrics import Diversidad as dv

from problems.Mlp import Mlp

db = Database()


class ScaMlp:
    def __init__(
        self, executionId, populationSize, maxIterations, discretizationScheme, repairType, population,
            facilities, customers, openFacilities, costs, customerBudgets, flpSolutions, flpBestSolution
    ):
        self.executionId = executionId
        self.populationSize = populationSize
        self.maxIterations = maxIterations
        self.discretizationScheme = discretizationScheme
        self.repairType = repairType
        self.population = population

        self.facilities = facilities
        self.customers = customers
        self.openFacilities = openFacilities
        self.costs = costs
        self.customerBudgets = customerBudgets
        self.flpSolutions = flpSolutions                # Matriz de solucines binarias de FLP
        self.flpBestSolution = flpBestSolution          # Solución con mejor fitness

    def process(self, *args, **kwargs):
        # db.startExecution(self.executionId, datetime.now())
        print('--- Paso 5.1: Instanciando MLP ---')
        mlp = Mlp(
            self.populationSize, self.discretizationScheme, self.repairType, self.facilities, self.customers,
            self.openFacilities, self.costs, self.customerBudgets, self.flpSolutions, self.flpBestSolution
        )
        print('--- Paso 5.2 Procesando MLP ---')
        mlp.process()
        # exit(0)

        #maxDiversities = np.zeros(7)  # es tamaño 7 porque calculamos 7 diversidades
        #diversities, maxDiversities, explorationPercentage, exploitationPercentage, state = dv.ObtenerDiversidadYEstado(
        #    flp.binMatrix, maxDiversities
        #)

        print('--- Paso 5.3 Comenzando iteraciones ---')
        a = 2
        globalBestFitness = 0
        globalBestSolutionDisc = np.array([])
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
            r2 = (2 * np.pi) * np.random.uniform(low=0.0, high=1.0, size=mlp.population.shape)
            r3 = 2 * np.random.uniform(low=0.0, high=1.0, size=mlp.population.shape)
            r4 = np.random.uniform(low=0.0, high=1.0, size=mlp.population.shape[0])

            oldBestSolutionIndex = mlp.solutionsRanking[0]
            oldBestSolutionFitness = mlp.fitness[oldBestSolutionIndex]
            oldBestSolutionDisc = mlp.discMatrix[oldBestSolutionIndex]
            oldBestSolution = mlp.population[oldBestSolutionIndex]

            mlp.population[r4 < 0.5] = mlp.population[r4 < 0.5] + np.multiply(
                r1, np.multiply(np.sin(r2[r4 < 0.5]), np.abs(
                    np.multiply(r3[r4 < 0.5], oldBestSolution) - mlp.population[r4 < 0.5]
                ))
            )
            mlp.population[r4 >= 0.5] = mlp.population[r4 >= 0.5] + np.multiply(
                r1, np.multiply(np.cos(r2[r4 >= 0.5]), np.abs(
                    np.multiply(r3[r4 >= 0.5], oldBestSolution) - mlp.population[r4 >= 0.5]
                ))
            )

            # Se binariza y evalua el fitness de todas las soluciones de la iteración t
            mlp.process()

            # Se convierte el best pisándolo
            if np.max(mlp.fitness) <= oldBestSolutionFitness:
                mlp.fitness[oldBestSolutionIndex] = oldBestSolutionFitness
                mlp.discMatrix[oldBestSolutionIndex] = oldBestSolutionDisc
            else:
                # print(f'MLP fitness[solutionsRank[0]]: {mlp.fitness[mlp.solutionsRanking[0]]}')
                pass

            if np.max(mlp.fitness) > globalBestFitness:
                globalBestFitness = np.max(mlp.fitness)
                globalBestSolutionDisc = mlp.discMatrix[mlp.solutionsRanking[0]]

            #diversities, maxDiversities, explorationPercentage, exploitationPercentage, state = dv.ObtenerDiversidadYEstado(
            #    flp.binMatrix, maxDiversities
            #)

            iterationTimeEnd = np.round(time.time() - iterationTimeStart, 6)
            processTimeEnd = np.round(time.process_time() - processTimeStart, 6)
            iterationData.update({
                'bestFitness': int(np.max(mlp.fitness)),
                'avgFitness': float(np.average(mlp.fitness)),
                'fitnessBestIteration': int(globalBestFitness),
                'parameters': {
                    'problem': 'MLP',
                    'fitness': int(np.max(mlp.fitness)),
                    'roundPrices': [round(price * 100) for price in mlp.selectPrices[mlp.solutionsRanking[0]]],
                    'prices': mlp.selectPrices[mlp.solutionsRanking[0]].tolist(),
                    'customersSpending': mlp.customersSpending[mlp.solutionsRanking[0]].tolist(),
                    'selectPricesByCustomer': mlp.selectPricesByCustomer[mlp.solutionsRanking[0]].tolist(),
                    'currentBestSolution': mlp.discMatrix[mlp.solutionsRanking[0]].tolist(),
                    'oldBestSolutionFitness': oldBestSolutionFitness,
                    'oldBestSolutionDisc': oldBestSolutionDisc.tolist(),
                    'flpBestSolution': self.flpSolutions[self.flpBestSolution].tolist(),
                    'iterationTime': iterationTimeEnd,
                    'processTime': processTimeEnd,
                    'repairsQuantity': int(mlp.repairsQuantity),
                    #'diversities': diversities.tolist(),
                    #'explorationPercentage': explorationPercentage.tolist(),
                    #'exploitationPercentage': exploitationPercentage.tolist(),
                    #'state': state
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
            bestSolution=globalBestSolutionDisc.tolist(),
            executionId=self.executionId,
            startDatetime=startDatetime,
            endDatetime=datetime.now()
        )

        # if not db.endExecution(self.executionId, datetime.now()):
        #    return False

        return True
