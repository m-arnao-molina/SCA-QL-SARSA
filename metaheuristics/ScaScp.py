import numpy as np
import time
from datetime import datetime
from database.DatabaseORM import Database
from Problem.util import read_instance as Instance
from Problem import SCP as Problem
from Metrics import Diversidad as dv
# RepairGPU
from Problem.util import SCPProblem

from metaheuristics.Metaheuristic import Metaheuristic
from problems.Scp import Scp

db = Database()

class ScaScp(Metaheuristic):
    def process(self, *args, **kwargs):
        #db.startExecution(self.executionId, datetime.now())

        if not self.validateInstance():
            return False

        scp = Scp(self.instanceName, self.instancePath)
        scp.readInstance()

        # problemaGPU = SCPProblem.SCPProblem(self.instancePath)
        # pondRestricciones = 1 / np.sum(problemaGPU.instance.get_r(), axis=1)

        weightConstraints = 1 / np.sum(scp.constraints, axis=1)  # pondRestricciones
        coverageMatrix = scp.constraints  # coverageMatrix
        dim = len(scp.costs)
        a = 2

        # Variables de diversidad
        diversidades = []
        maxDiversidades = np.zeros(7)  # es tamaño 7 porque calculamos 7 diversidades
        PorcentajeExplor = []
        PorcentajeExplot = []
        state = []

        # Generar población inicial
        population = np.random.uniform(low=-1.0, high=1.0, size=(self.populationSize, dim))
        # matrixBin = np.zeros((self.populationSize,dim))
        binMatrix = np.random.randint(low=0, high=2, size=(self.populationSize, dim))
        fitness = np.zeros(self.populationSize)
        solutionsRanking = np.zeros(self.populationSize)
        # binMatrix, fitness, solutionsRanking, numReparaciones = Problem.SCP(
        #     population, binMatrix, solutionsRanking, scp.costs, scp.constraints, self.discretizationScheme,
        #     self.repairType, problemaGPU, weightConstraints
        # )

        binMatrix, fitness, solutionsRanking, repairsQuantity = scp.process(
            population, binMatrix, solutionsRanking, scp.costs, scp.constraints, self.discretizationScheme,
            self.repairType, weightConstraints
        )

        print({
            'binMatrix': binMatrix,
            'fitness': fitness,
            'solutionsRanking': solutionsRanking,
            'repairsQuantity': repairsQuantity
        })


        return True

        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(
            binMatrix,maxDiversidades
        )

        startDatetime = datetime.now()
        timerStartResult = time.time()
        iterationsData = []
        for iterationNumber in range(0, self.maxIterations):
            iterationData = {
                'iterationNumber': iterationNumber,
                'executionId': self.executionId,
                'startDatetime': datetime.now()
            }

            # print(f'iterationNumber: {iterationNumber}')
            processTime = time.process_time()
            timerStart = time.time()

            # SCA
            r1 = a - iterationNumber * (a / self.maxIterations)
            r4 = np.random.uniform(low=0.0, high=1.0, size=population.shape[0])
            r2 = (2 * np.pi) * np.random.uniform(low=0.0, high=1.0, size=population.shape)
            r3 = np.random.uniform(low=0.0, high=2.0, size=population.shape)
            bestRowAux = solutionsRanking[0]
            Best = population[bestRowAux]
            BestBinary = binMatrix[bestRowAux]
            BestFitness = np.min(fitness)
            population[r4 < 0.5] = population[r4 < 0.5] + np.multiply(r1, np.multiply(np.sin(r2[r4 < 0.5]), np.abs(
                np.multiply(r3[r4 < 0.5], Best) - population[r4 < 0.5])))
            population[r4 >= 0.5] = population[r4 >= 0.5] + np.multiply(r1, np.multiply(np.cos(r2[r4 >= 0.5]), np.abs(
                np.multiply(r3[r4 >= 0.5], Best) - population[r4 >= 0.5])))
            # population[bestRow] = Best

            # Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
            binMatrix, fitness, solutionsRanking, repairsQuantity = Problem.SCP(
                population, binMatrix, solutionsRanking, scp.costs, coverageMatrix, self.discretizationScheme,
                self.repairType, problemaGPU, weightConstraints
            )

            # Conservo el Best
            if fitness[bestRowAux] > BestFitness:
                fitness[bestRowAux] = BestFitness
                binMatrix[bestRowAux] = BestBinary

            diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(
                binMatrix, maxDiversidades)
            BestFitnes = str(np.min(fitness))

            walltimeEnd = np.round(time.time() - timerStart, 6)
            processTimeEnd = np.round(time.process_time() - processTime, 6)

            iterationData.update({
                'bestFitness': 0,
                'avgFitness': 0,
                'fitnessBestIteration': 0,
                'parameters': {
                    'fitness': BestFitnes,
                    'clock_Time': walltimeEnd,
                    'process_time': processTimeEnd,
                    'discretizationScheme': self.discretizationScheme,
                    'diversities': str(diversidades),
                    'exploration_percentage': str(PorcentajeExplor),
                    # 'exploitation_percentage': str(PorcentajeExplot),
                    'repairsQuantity': str(repairsQuantity)
                    # 'state': str(state)
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
            fitness=float(BestFitness),
            bestSolution=None,
            executionId=self.executionId,
            startDatetime=startDatetime,
            endDatetime=datetime.now()
        )

        if not db.endExecution(self.executionId, datetime.now()):
            return False

        return True