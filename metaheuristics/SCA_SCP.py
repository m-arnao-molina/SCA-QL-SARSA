import os
import numpy as np
import time
import json
from database.Database import Database
from envs import env
from datetime import datetime
from Problem.util import read_instance as Instance
from Problem import SCP as Problem
from Metrics import Diversidad as dv

# RepairGPU
from Problem.util import SCPProblem

# Definicion Environments Vars
workdir = os.path.abspath(os.getcwd())
workdirInstance = workdir + env('DIR_INSTANCES')

database = Database()


def SCA_SCP(executionId, instanceFile, instanceDirectory, population, maxIterations, discretizacionScheme, repair):
    database.startExecution(executionId, datetime.now())

    instancePath = workdirInstance + instanceDirectory + instanceFile

    if not os.path.exists(instancePath):
        print(f'No se encontró la instancia: {instancePath}')
        return False

    instance = Instance.Read(instancePath)

    problemaGPU = SCPProblem.SCPProblem(instancePath)
    pondRestricciones = 1 / np.sum(problemaGPU.instance.get_r(), axis=1)

    matrizCobertura = np.array(instance.get_r())
    vectorCostos = np.array(instance.get_c())

    dim = len(vectorCostos)
    pob = population
    maxIterations = maxIterations
    DS = discretizacionScheme  # [v1,Standard]
    a = 2

    # Variables de diversidad
    diversidades = []
    maxDiversidades = np.zeros(7)  # es tamaño 7 porque calculamos 7 diversidades
    PorcentajeExplor = []
    PorcentajeExplot = []
    state = []

    # Generar población inicial
    poblacion = np.random.uniform(low=-1.0, high=1.0, size=(pob, dim))
    # matrixBin = np.zeros((pob,dim))
    matrixBin = np.random.randint(low=0, high=2, size=(pob, dim))
    fitness = np.zeros(pob)
    solutionsRanking = np.zeros(pob)
    matrixBin, fitness, solutionsRanking, numReparaciones = Problem.SCP(poblacion, matrixBin, solutionsRanking,
                                                                        vectorCostos, matrizCobertura, DS, repair,
                                                                        problemaGPU, pondRestricciones)
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,
                                                                                                           maxDiversidades)

    inicio = datetime.now()
    timerStartResult = time.time()
    iterationsData = []
    for iterationNumber in range(0, maxIterations):
        # print(f'iterationNumber: {iterationNumber}')
        processTime = time.process_time()
        timerStart = time.time()

        # SCA
        r1 = a - iterationNumber * (a / maxIterations)
        r4 = np.random.uniform(low=0.0, high=1.0, size=poblacion.shape[0])
        r2 = (2 * np.pi) * np.random.uniform(low=0.0, high=1.0, size=poblacion.shape)
        r3 = np.random.uniform(low=0.0, high=2.0, size=poblacion.shape)
        bestRowAux = solutionsRanking[0]
        Best = poblacion[bestRowAux]
        BestBinary = matrixBin[bestRowAux]
        BestFitness = np.min(fitness)
        poblacion[r4 < 0.5] = poblacion[r4 < 0.5] + np.multiply(r1, np.multiply(np.sin(r2[r4 < 0.5]), np.abs(
            np.multiply(r3[r4 < 0.5], Best) - poblacion[r4 < 0.5])))
        poblacion[r4 >= 0.5] = poblacion[r4 >= 0.5] + np.multiply(r1, np.multiply(np.cos(r2[r4 >= 0.5]), np.abs(
            np.multiply(r3[r4 >= 0.5], Best) - poblacion[r4 >= 0.5])))
        # poblacion[bestRow] = Best

        # Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixBin, fitness, solutionsRanking, numReparaciones = Problem.SCP(poblacion, matrixBin, solutionsRanking,
                                                                            vectorCostos, matrizCobertura, DS, repair,
                                                                            problemaGPU, pondRestricciones)

        # Conservo el Best
        if fitness[bestRowAux] > BestFitness:
            fitness[bestRowAux] = BestFitness
            matrixBin[bestRowAux] = BestBinary

        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(
            matrixBin, maxDiversidades)
        BestFitnes = str(np.min(fitness))

        walltimeEnd = np.round(time.time() - timerStart, 6)
        processTimeEnd = np.round(time.process_time() - processTime, 6)

        iterationData = {
            'execution_id': executionId,
            'iteration_number': iterationNumber,
            'best_fitness': BestFitnes,
            'parameters': json.dumps({
                'fitness': BestFitnes,
                'clock_Time': walltimeEnd,
                'process_time': processTimeEnd,
                'DS': DS,
                'diversities': str(diversidades),
                'exploration_percentage': str(PorcentajeExplor),
                # 'exploitation_percentage': str(PorcentajeExplot),
                'repairs_number': str(numReparaciones)
                # 'state': str(state)
            })
        }

        iterationsData.append(iterationData)
        if iterationNumber % 100 == 0 and database.insertIteration(iterationsData):
            iterationsData.clear()

    # Si es que queda alguna iteración para insertar
    if len(iterationsData) > 0:
        database.insertIteration(iterationsData)

    # Actualiza la tabla execution_results, sin mejor_solucion
    executionResultData = {
        'execution_id': executionId,
        'fitness': BestFitnes,
        'start_datetime': inicio,
        'end_datetime': datetime.now()
    }
    database.insertExecutionResult(executionResultData)

    if not database.endExecution(executionId, datetime.now()):
        return False

    return True
