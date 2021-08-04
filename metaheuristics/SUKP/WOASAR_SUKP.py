# Utils

import sys
import os
import settings
from envs import env
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# SQL
import sqlalchemy as db
import psycopg2
import json
import pickle
import zlib

import database.Database as Database

# Problema
from Problem import SUKP as Problem

#Diversidad
from Metrics import Diversidad as dv

# ML
from MachineLearning.SARSA import SARSA as SAR

# Definicion Environments Vars
workdir = os.path.abspath(os.getcwd())
workdirInstance = workdir+env('DIR_INSTANCES')

connect = Database.Database()

transferFunction = ['V1', 'V2', 'V3', 'V4', 'S1', 'S2', 'S3', 'S4']
operatorBinarization = ['Standard','Complement','Elitist','Static','ElitistRoulette']

DS_actions = [tf + "," + ob for tf in transferFunction for ob in operatorBinarization]

def WOASAR_SUKP(id,instance_file,instance_dir,population,maxIter,discretizacionScheme,ql_alpha,ql_gamma,repair,policy,rewardType,qlAlphaType):

    instance_path = workdirInstance + instance_dir + instance_file

    if not os.path.exists(instance_path):
        print(f'No se encontró la instancia: {instance_path}')
        return False

    #Lectura de instancias
    numItem, numElement, knapsackSize, relationMatrix, weight, profit = Problem.read_instance_SUKP(instance_file,instance_dir,workdirInstance)
    
    dim = numItem
    pob = population
    maxIter = maxIter
    DS = discretizacionScheme
    a = 2
    solutionsRank = np.zeros(pob)


    #Variables de diversidad
    diversidades = []
    maxDiversidades = np.zeros(7) #es tamaño 7 porque calculamos 7 diversidades
    PorcentajeExplor = []
    PorcentajeExplot = []
    state = []

    #Generar población inicial
    poblacion = np.random.uniform(0,1,size=(pob,dim))
    #matrixBin = np.zeros((pob,dim))
    matrixBin = np.random.randint(low=0, high=2, size = (pob,dim))
    fitness = np.zeros(pob)
    solutionsRank = np.zeros(pob)
    
    
    #Parámetros fijos de WOA
    b = 1 #Según código

    # SARSA 
    agente = SAR(ql_gamma, DS_actions, 2, qlAlphaType, rewardType, maxIter,  qlAlpha = ql_alpha)
    DS = agente.getAccion(0,policy) 

    matrixBin,fitness,solutionsRank,numReparaciones = Problem.SUKP(poblacion,matrixBin,solutionsRank,DS_actions[DS],repair,numItem, numElement, knapsackSize, relationMatrix, weight, profit)
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)

    #SARSA
    CurrentState = state[0] #Estamos midiendo según Diversidad "DimensionalHussain"

    inicio = datetime.now()
    timerStartResult = time.time()
    memory = []
    for iter in range(0, maxIter):

        processTime = time.process_time()             
        timerStart = time.time()


        #WOA
        a = 2 - ((2*iter)/maxIter)
        A = np.random.uniform(low=-a,high=a,size=(pob,dim)) #vector rand de tam (pob,dim)
        Aabs = np.abs(A[0]) # Vector de A absoluto en tam pob
        C = np.random.uniform(low=0,high=2,size=(pob,dim)) #vector rand de tam (pob,dim)
        l = np.random.uniform(low=-1,high=1,size=(pob,dim)) #vector rand de tam (pob,dim)
        p = np.random.uniform(low=0,high=0.4,size=pob) #vector rand de tam pob ***

        bestSolutionOldFitness = fitness[solutionsRank[0]]
        bestSolutionOldBin  = matrixBin[solutionsRank[0]]
        Best = poblacion[solutionsRank[0]]

        #ecu 2.1 Pero el movimiento esta en 2.2
        indexCond2_2 = np.intersect1d(np.argwhere(p<0.5),np.argwhere(Aabs<1)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 2.2
        if indexCond2_2.shape[0] != 0:
            poblacion[indexCond2_2] = Best - np.multiply(A[indexCond2_2],np.abs(np.multiply(C[indexCond2_2],Best)-poblacion[indexCond2_2]))

        #ecu 2.8
        indexCond2_8 = np.intersect1d(np.argwhere(p<0.5),np.argwhere(Aabs>=1)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 2.1
        if indexCond2_8.shape[0] != 0:
            Xrand = poblacion[np.random.randint(low=0, high=pob, size=indexCond2_8.shape[0])] #Me entrega un conjunto de soluciones rand de tam indexCond2_2.shape[0] (osea los que cumplen la cond11)

            poblacion[indexCond2_8] = Xrand - np.multiply(A[indexCond2_8],np.abs(np.multiply(C[indexCond2_8],Xrand)-poblacion[indexCond2_8]))

        #ecu 2.5
        indexCond2_5 = np.intersect1d(np.argwhere(p>=0.5),np.argwhere(p>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 2.1
        if indexCond2_5.shape[0] != 0:
            poblacion[indexCond2_5] = np.multiply(np.multiply(np.abs(Best - poblacion[indexCond2_5]),np.exp(b*l)),np.cos(2*np.pi*l)) + Best

        
        # Escogemos esquema desde SARSA
        DS = agente.getAccion(CurrentState,policy)
        oldState = CurrentState #Rescatamos estado actual

        #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixBin,fitness,solutionsRank,numReparaciones  = Problem.SUKP(poblacion,matrixBin,solutionsRank,DS_actions[DS],repair,numItem, numElement, knapsackSize, relationMatrix, weight, profit)
        
        #Convierto el Best - Pisándolo
        if np.max(fitness) <= bestSolutionOldFitness:
            fitness[solutionsRank[0]] = bestSolutionOldFitness
            matrixBin[solutionsRank[0]] = bestSolutionOldBin
        else:
            print(f'fitness[solutionsRank[0]]** **: {fitness[solutionsRank[0]]}')

        #Calcular Diversidad y Estado
        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)
        BestFitness = str(np.max(fitness))

        CurrentState = state[0]
        # Observamos, y recompensa/castigo.  Actualizamos Tabla Q
        agente.updateQtable(np.max(fitness), DS, CurrentState, oldState, iter)


        walltimeEnd = np.round(time.time() - timerStart,6)
        processTimeEnd = np.round(time.process_time()-processTime,6) 
        
        dataIter = {
            "id_ejecucion": id,
            "numero_iteracion":iter,
            "fitness_mejor": BestFitness,
            "parametros_iteracion": json.dumps({
                "fitness": BestFitness,
                "clockTime": walltimeEnd,
                "processTime": processTimeEnd,
                "DS":str(DS),
                "Diversidades":  str(diversidades),
                "PorcentajeExplor": str(PorcentajeExplor)
                #"PorcentajeExplot": str(PorcentajeExplot),
                #"state": str(state)
                })
                }

        memory.append(dataIter)
       

        if iter % 100 == 0:
            memory = connect.insertMemory(memory)

    # Si es que queda algo en memoria para insertar
    if(len(memory)>0):
        memory = connect.insertMemory(memory)

    #Actualizamos la tabla resultado_ejecucion, sin mejor_solucion
    memory2 = []
    fin = datetime.now()    
    qtable = agente.getQtable()
    dataResult = {
        "id_ejecucion": id,
        "fitness": BestFitness,
        "inicio": inicio,
        "fin": fin,
        "mejor_solucion": json.dumps(qtable.tolist())
        }
    memory2.append(dataResult)
    dataResult = connect.insertMemoryBest(memory2)

    # Update ejecucion
    if not connect.endEjecucion(id,datetime.now(),'terminado'):
        return False

    return True