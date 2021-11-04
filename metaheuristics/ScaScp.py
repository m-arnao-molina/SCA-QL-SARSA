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

        a = 2

        # Variables de diversidad
        diversidades = []
        maxDiversidades = np.zeros(7)  # es tamaño 7 porque calculamos 7 diversidades
        PorcentajeExplor = []
        PorcentajeExplot = []
        state = []

        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(
            scp.binMatrix, maxDiversidades
        )

        #print({
        #    'diversidades': diversidades,
        #    'maxDiversidades': maxDiversidades,
        #    'PorcentajeExplor': PorcentajeExplor,
        #    'PorcentajeExplot': PorcentajeExplot,
        #    'state': state
        #})

        startDatetime = datetime.now()
        iterationsData = []
        for iterationNumber in range(0, self.maxIterations):
            iterationData = {
                'iterationNumber': iterationNumber,
                'executionId': self.executionId,
                'startDatetime': datetime.now()
            }

            processTime = time.process_time()
            timerStart = time.time()

            # SCA
            r1 = a - iterationNumber * (a / self.maxIterations)
            r4 = np.random.uniform(low=0.0, high=1.0, size=scp.population.shape[0])
            r2 = (2 * np.pi) * np.random.uniform(low=0.0, high=1.0, size=scp.population.shape)
            r3 = np.random.uniform(low=0.0, high=2.0, size=scp.population.shape)
            bestRowAux = scp.solutionsRanking[0]
            Best = scp.population[bestRowAux]
            BestBinary = scp.binMatrix[bestRowAux]
            BestFitness = np.min(scp.fitness)
            scp.population[r4 < 0.5] = scp.population[r4 < 0.5] + np.multiply(r1, np.multiply(np.sin(r2[r4 < 0.5]), np.abs(
                np.multiply(r3[r4 < 0.5], Best) - scp.population[r4 < 0.5])))
            scp.population[r4 >= 0.5] = scp.population[r4 >= 0.5] + np.multiply(r1, np.multiply(np.cos(r2[r4 >= 0.5]), np.abs(
                np.multiply(r3[r4 >= 0.5], Best) - scp.population[r4 >= 0.5])))
            # scp.population[bestRow] = Best

            # Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t

            scp.process()


            # Conservo el Best
            if scp.fitness[bestRowAux] > BestFitness:
                scp.fitness[bestRowAux] = BestFitness
                scp.binMatrix[bestRowAux] = BestBinary

            diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(
                scp.binMatrix, maxDiversidades
            )
            BestFitnes = str(np.min(scp.fitness))

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
                    'repairsQuantity': str(scp.repairsQuantity)
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
