import numpy as np
import random

from Discretization import DiscretizationScheme as DS

#action : esquema de discretizacion DS
def SUKP(poblacion,matrixBin,solutionsRanking,ds,repairType,numItem, numElement, knapsackSize, relationMatrix, weight, profit):

    #Binarización de 2 pasos
    ds = ds.split(",")
    ds = DS.DiscretizationScheme(poblacion,matrixBin,solutionsRanking,ds[0],ds[1])
    matrixBin = ds.binariza()


    #Evaluamos restricción, sila función evaluar_restriccion_SUKP devuelve un 1 es factible, si devuelve un 0 no es factible y hay que reparar
    EvaluarReparacion, PesosDeItems = evaluar_restriccion_SUKP(matrixBin,relationMatrix,weight,knapsackSize,numItem)

    #Reparamos si corresponde
    numReparacionesVector = []
    for i in range(len(EvaluarReparacion)):
        if EvaluarReparacion[i] == 0:
            if repairType == 1: #Reparación "Simple" con un orden según peso de los items
                vectorSinReparar = matrixBin[i]
                matrixBin[i] = reparar1(matrixBin[i], knapsackSize, relationMatrix, weight, profit,PesosDeItems)   #*** Ver que más necesita la función para reparar
                vectorReparada = matrixBin[i]
                numReparacionesVector.append(np.sum(vectorReparada - vectorSinReparar))
            if repairType == 2: #Reparación según ponderación p_i/R_i del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)
                vectorSinReparar = matrixBin[i]
                matrixBin[i] = reparar2(matrixBin[i], knapsackSize, relationMatrix, weight, profit,PesosDeItems,numItem)   #*** Ver que más necesita la función para reparar
                vectorReparada = matrixBin[i]
                numReparacionesVector.append(np.sum(vectorReparada - vectorSinReparar))
            if repairType == 3: #Reparación según ponderación p_i/R_i del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)
                vectorSinReparar = matrixBin[i]
                matrixBin[i] = reparar3(matrixBin[i], knapsackSize, relationMatrix, weight, profit,PesosDeItems,numItem)   #*** Ver que más necesita la función para reparar
                vectorReparada = matrixBin[i]
                numReparacionesVector.append(np.sum(vectorReparada - vectorSinReparar))
            if repairType == 4: #Reparación según ponderación p_i/R_i del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)
                vectorSinReparar = matrixBin[i]
                matrixBin[i] = reparar3(matrixBin[i], knapsackSize, relationMatrix, weight, profit,PesosDeItems,numItem)   #*** Ver que más necesita la función para reparar
                vectorReparada = matrixBin[i]
                numReparacionesVector.append(np.sum(vectorReparada - vectorSinReparar))
    numReparaciones = np.sum(numReparacionesVector)


    #Calculamos Fitness
    # print(f'matrixBin: {matrixBin}')
    # print(f'matrixBin.sum: {np.sum(matrixBin)}')
    fitness = np.sum(np.multiply(matrixBin,profit),axis =1)
    solutionsRanking = (-fitness).argsort() # rankings de los mejores fitness

    return matrixBin,fitness,solutionsRanking,numReparaciones

def evaluar_restriccion_SUKP(matrixBin,relationMatrix,weight,knapsackSize,numItem):
    
    PesosDeMatrizRelacion = np.multiply(relationMatrix,weight)
    PesosDeItems = np.sum(PesosDeMatrizRelacion,axis=1)
    PesosParaEvaluarFactibilidad = np.sum(np.multiply(matrixBin,PesosDeItems),axis=1)
    EvaluarReparacion = np.where(PesosParaEvaluarFactibilidad <= knapsackSize, 1, 0)



    return EvaluarReparacion, PesosDeItems


def reparar1(vectorBin, knapsackSize, relationMatrix, weight, profit,PesosDeItems): #Reparación "Simple" con un orden según peso de los items
    
    indicadorReparacion = np.divide(profit,PesosDeItems) #Nos entregará un vector de indicadores de la reción profit/weight, en donde el menor valor corresponde a los items menos importantes
    RankIndicadorReparacion= (-indicadorReparacion).argsort()

    for i in range(len(RankIndicadorReparacion)):
        if np.sum(np.multiply(vectorBin,PesosDeItems)) > knapsackSize:
            vectorBin[RankIndicadorReparacion[i]] = 0
        else:
            break

    
    return vectorBin

def reparar2(vectorBin, knapsackSize, relationMatrix, weight, profit,PesosDeItems,numItem): #Reparación según ponderación p_i/R_i del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)
    
    frecuenciasDeElementos = np.divide(np.sum(relationMatrix, axis=0),numItem)
    PesosDeMatrizRelacion = np.multiply(relationMatrix,weight)
    R = np.sum(np.divide(PesosDeMatrizRelacion,frecuenciasDeElementos), axis=1) #Vector de tamaño items
    indicadorReparacion = np.divide(profit,R) #Vector tamaño items

    indicadorReparacion = np.multiply(indicadorReparacion,vectorBin)
    
    maxindicador = np.sum(indicadorReparacion)
    indicadorReparacion = np.where(indicadorReparacion == 0, indicadorReparacion-maxindicador, indicadorReparacion)
    # print(f'indicadorReparacion: {indicadorReparacion}')

    RankIndicadorReparacion= (-indicadorReparacion).argsort()

    cant = np.sum(np.where(indicadorReparacion == -maxindicador, 1, 0)) #Cantidad de "0" en nuestro indicador, es decir items no elegidos en nuestro vector solución evaluado

    RankIndicadorReparacion = RankIndicadorReparacion[:-cant]
    
    # print(f'RankIndicadorReparacion.shape: {RankIndicadorReparacion.shape}')

    vectorBinZ = np.zeros(vectorBin.shape)

    #Linea 4 del algoritmo 2 del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)
    for i in range(len(RankIndicadorReparacion)):
        if PesosDeItems[RankIndicadorReparacion[i]] <= (knapsackSize - np.sum(np.multiply(vectorBinZ,PesosDeItems))): #Comprobamos si el items entra en el espacio disponible en la mochila
            vectorBinZ[RankIndicadorReparacion[i]] = 1
        else:
            continue


    #Linea 8 del algoritmo 2 del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)

    vectorBinZInvertida = np.where(vectorBinZ == 1, 0, 1) #Es decir que los 1 serán 0 y los 0 serán 1.

    indicadorReparacion = np.multiply(indicadorReparacion,vectorBinZInvertida)
    
    maxindicador = np.sum(indicadorReparacion)
    indicadorReparacion = np.where(indicadorReparacion == 0, indicadorReparacion-maxindicador, indicadorReparacion)

    RankIndicadorReparacion= (-indicadorReparacion).argsort()

    cant = np.sum(np.where(RankIndicadorReparacion == -maxindicador, 1, 0)) #Cantidad de "0" en nuestro indicador, es decir items no elegidos en nuestro vector solución evaluado

    RankIndicadorReparacion = RankIndicadorReparacion[:-cant]

    for i in range(len(RankIndicadorReparacion)):
        if PesosDeItems[RankIndicadorReparacion[i]] <= (knapsackSize - np.sum(np.multiply(vectorBinZ,PesosDeItems))): #Comprobamos si el items entra en el espacio disponible en la mochila
            vectorBinZ[RankIndicadorReparacion[i]] = 1
        else:
            continue    


    vectorBin = vectorBinZ
    # print(f'Peso de la mochila: {np.sum(np.multiply(vectorBin,PesosDeItems))}')
    
    return vectorBin

def reparar3(vectorBin, knapsackSize, relationMatrix, weight, profit,PesosDeItems,numItem): #Reparación según ponderación p_i/R_i del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)
    
    frecuenciasDeElementos = np.divide(np.sum(relationMatrix, axis=0),numItem)
    PesosDeMatrizRelacion = np.multiply(relationMatrix,weight)
    R = np.sum(np.divide(PesosDeMatrizRelacion,frecuenciasDeElementos), axis=1) #Vector de tamaño items
    indicadorReparacion = np.divide(profit,R) #Vector tamaño items

    # indicadorReparacion = np.multiply(indicadorReparacion,vectorBin)
    
    # maxindicador = np.sum(indicadorReparacion)
    # indicadorReparacion = np.where(indicadorReparacion == 0, indicadorReparacion-maxindicador, indicadorReparacion)
    # print(f'indicadorReparacion: {indicadorReparacion}')

    RankIndicadorReparacion= (indicadorReparacion).argsort()

    # cant = np.sum(np.where(indicadorReparacion == -maxindicador, 1, 0)) #Cantidad de "0" en nuestro indicador, es decir items no elegidos en nuestro vector solución evaluado

    # RankIndicadorReparacion = RankIndicadorReparacion[:-cant]
    
    # print(f'RankIndicadorReparacion.shape: {RankIndicadorReparacion.shape}')

    #vectorBinZ = np.zeros(vectorBin.shape)
    vectorBinZ = vectorBin

    #Linea 4 del algoritmo 2 del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)
    for i in range(len(RankIndicadorReparacion)):
        # print(f'np.sum(np.multiply(vectorBinZ,PesosDeItems)): {np.sum(np.multiply(vectorBinZ,PesosDeItems))}')
        # print(f'knapsackSize: {knapsackSize}')
        if np.sum(np.multiply(vectorBinZ,PesosDeItems)) >= knapsackSize: #Comprobamos si el items entra en el espacio disponible en la mochila
            vectorBinZ[RankIndicadorReparacion[i]] = 0
        else:
            continue


    #Linea 8 del algoritmo 2 del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)

    vectorBinZInvertida = np.where(vectorBinZ == 1, 0, 1) #Es decir que los 1 serán 0 y los 0 serán 1.

    # indicadorReparacion = np.multiply(indicadorReparacion,vectorBinZInvertida)
    
    # maxindicador = np.sum(indicadorReparacion)
    # indicadorReparacion = np.where(indicadorReparacion == 0, indicadorReparacion-maxindicador, indicadorReparacion)

    RankIndicadorReparacion= (-indicadorReparacion).argsort()

    # cant = np.sum(np.where(RankIndicadorReparacion == -maxindicador, 1, 0)) #Cantidad de "0" en nuestro indicador, es decir items no elegidos en nuestro vector solución evaluado

    # RankIndicadorReparacion = RankIndicadorReparacion[:-cant]

    for i in range(len(RankIndicadorReparacion)):
        # print("*************************************************")
        # print(f'np.sum(np.multiply(vectorBinZ,PesosDeItems)): {np.sum(np.multiply(vectorBinZ,PesosDeItems))}')
        # print(f'knapsackSize - np.sum(np.multiply(vectorBinZ,PesosDeItems)): {knapsackSize - np.sum(np.multiply(vectorBinZ,PesosDeItems))}')

        if PesosDeItems[RankIndicadorReparacion[i]] <= (knapsackSize - np.sum(np.multiply(vectorBinZ,PesosDeItems))): #Comprobamos si el items entra en el espacio disponible en la mochila
            vectorBinZ[RankIndicadorReparacion[i]] = 1
        else:
            continue    


    vectorBin = vectorBinZ
    # print(f'Peso de la mochila: {np.sum(np.multiply(vectorBin,PesosDeItems))}')
    
    return vectorBin

def reparar4(vectorBin, knapsackSize, relationMatrix, weight, profit,PesosDeItems,numItem): #Reparación según ponderación p_i/R_i del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)
    
    frecuenciasDeElementos = np.divide(np.sum(relationMatrix, axis=0),numItem)
    PesosDeMatrizRelacion = np.multiply(relationMatrix,weight)
    R = np.sum(np.divide(PesosDeMatrizRelacion,frecuenciasDeElementos), axis=1) #Vector de tamaño items
    indicadorReparacion = np.divide(profit,R) #Vector tamaño items

    RankIndicadorReparacion= (-indicadorReparacion).argsort()

    # vectorBinZ = vectorBin
    vectorBinZ = np.zeros(numItem)

    #Linea 4 del algoritmo 2 del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)
    for i in range(len(RankIndicadorReparacion)):
        # if np.sum(np.multiply(vectorBinZ,PesosDeItems)) >= knapsackSize: #Comprobamos si el items entra en el espacio disponible en la mochila
        if PesosDeItems[RankIndicadorReparacion[i]] <= (knapsackSize - np.sum(np.multiply(vectorBinZ,PesosDeItems))) and vectorBin[RankIndicadorReparacion[i]] == 1: #Comprobamos si el items entra en el espacio disponible en la mochila
            rand = np.random.rand()
            if rand <= 1:
                vectorBinZ[RankIndicadorReparacion[i]] = 1
        else:
            continue
   #Linea 8 del algoritmo 2 del paper (https://sci-hub.se/https://doi.org/10.1016/j.future.2017.05.044)
    RankIndicadorReparacion= (-indicadorReparacion).argsort()

    for i in range(len(RankIndicadorReparacion)):

        if PesosDeItems[RankIndicadorReparacion[i]] <= (knapsackSize - np.sum(np.multiply(vectorBinZ,PesosDeItems))) and vectorBinZ[RankIndicadorReparacion[i]] == 0: #Comprobamos si el items entra en el espacio disponible en la mochila
            rand = np.random.rand()
            if rand <= 1:
                vectorBinZ[RankIndicadorReparacion[i]] = 1
        else:
            continue    


    vectorBin = vectorBinZ
    # print(f'Peso de la mochila: {np.sum(np.multiply(vectorBin,PesosDeItems))}')
    
    return vectorBin


def read_instance_SUKP(file,instance_dir,workdirInstance):
    # print(f'file: {file}')
    path = workdirInstance + instance_dir

    numItem = 0 # i, el numero de itemes.
    numElement = 0 # j, el numero de elemento.
    knapsackSize = 0 # Esta es la restriccion.
    profit = np.zeros(numItem) # m
    weight = np.zeros(numElement) # n
    fo = open(path + file , 'r')
    fo, line  = read_line(3,fo)
    posicion = line.find('=')
    numItem = int(line[posicion+1:posicion+5])
    posicion = line.find('=',posicion+1)
    numElement = int(line[posicion+1:posicion+5])
    relationMatrix = np.zeros((numItem, numElement))
    posicion = line.find('=',posicion+1)
    knapsackSize = int(line[posicion+1:posicion+7])
    fo, line  = read_line(3,fo)
    line = line.replace(' \n','').split(' ')
    line = [int(i) for i in line] 
    profit = np.asarray(line)
    fo, line  = read_line(3,fo)
    line = line.replace(' \n','').split(' ')
    line = [int(i) for i in line] 
    weight = np.asarray(line)
    fo, line  = read_line(2,fo)
    for i in range(0,numItem):
        relation = fo.readline().replace(' \n','').split(' ')
        for j in range(0,numElement):
            relationMatrix[i,j] = int(relation[j])
    print(relationMatrix.shape)
    fo.close()
    return numItem, numElement, knapsackSize, relationMatrix, weight, profit

def read_line(n,fo):
    for i in range(0,n):
        line = fo.readline()
    return fo, line 
