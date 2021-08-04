import numpy as np
import random

from sqlalchemy.sql.expression import true

from Discretization import DiscretizationScheme as DS

def evaluar_restriccion_KP(totalPesos,knapsackSize):
    #print(matrixBin,weightItems)
    if(totalPesos<=knapsackSize):
        return True
    elif(totalPesos>knapsackSize):
        return False

#action : esquema de discretizacion DS
def KP(poblacion,matrixBin,solutionsRanking,ds,repair,numItem,knapsackSize, weightItems,valueItems):

    #Binarización de 2 pasos
    ds = ds.split(",")
    # print(ds)
    # input()
    ds = DS.DiscretizationScheme(poblacion,matrixBin,solutionsRanking,ds[0],ds[1])
    matrixBin = ds.binariza()
    
    #Se obtiene tipo de reparacion, si es factible la solucion repairType = True y si no es factibler la solucion repairType = False 
    #print(matrixBin)
    

    #Se crea un vector con la densidad de los items solucion y se ordenan de manera no decreciente, moviendo los valores de la matrixBin.
    density = np.divide(valueItems,weightItems)
    #Se selecciona metodo de reparacion
    #print(matrixBin)
    fitness = np.zeros(len(matrixBin))
    # print('Como entra la matriz:',matrixBin[0])
    # print('Como entran los pesos:',weightItems)
    # print('Como entran los valores:',valueItems)
    # input()
    for i in range(len(matrixBin)):
        density = np.multiply(matrixBin[i],density)
        for j in range(len(matrixBin[i])-1):
            for k in range(len(matrixBin[i])-j-1):
                if density[k] > density[k + 1] :
                    density[k], density[k + 1] = density[k + 1], density[k]
                    weightItems[k],weightItems[k+1] = weightItems[k+1],weightItems[k]
                    valueItems[k],valueItems[k+1] = valueItems[k+1],valueItems[k]
                    matrixBin[i][k],matrixBin[i][k+1] = matrixBin[i][k+1],matrixBin[i][k]
            # print(weightItems)
            # print(valueItems)
            # input()
        pesosItems = np.multiply(matrixBin[i],weightItems)
        pesosTotales = np.sum(pesosItems)
        repairType = evaluar_restriccion_KP(pesosTotales,knapsackSize)
        # print('Como sale la matriz:',matrixBin[0])
        # print('Como salen los pesos:',weightItems)
        # print('Como salen los valores:',valueItems)
        # print(matrixBin[i],pesosTotales)
        # print()
        # input()
        if(repairType == True):
            # print('entra TRUE')
            # input()
            for j in range(len(matrixBin[i])):
                if matrixBin[i][j] == 0 and pesosTotales + weightItems[j] <= knapsackSize:
                    matrixBin[i][j] = 1
                    pesosTotales = pesosTotales + weightItems[j]
  
        elif(repairType == False):    
            # print('entra FALSE')
            # input()
            # print(range(len(matrixBin[i])-1,-1,-1))
            for j in range(len(matrixBin[i])-1,-1,-1):
                # print(matrixBin[i][j])
                if matrixBin[i][j] == 1 and pesosTotales > knapsackSize:
                    matrixBin[i][j] = 0
                    pesosTotales = pesosTotales - weightItems[j]
                    # print(pesosTotales,weightItems[j])
                    # input()

        fitness[i]=np.sum(np.multiply(matrixBin[i],valueItems))
        # print(fitness)
        # input()

    #Calculamos Fitness
    # print(f'matrixBin: {matrixBin}')
    # print(f'matrixBin.sum: {np.sum(matrixBin)}')
    
    solutionsRanking = (-fitness).argsort() # rankings de los mejores fitness
    # print(fitness,fitness[0])
    # input()
    return matrixBin,fitness,solutionsRanking




def read_instance_KP(file,instance_dir,workdirInstance):
    # print(f'file: {file}')
    path = workdirInstance + instance_dir
    valueItems = [] #Arreglo con el valor de cada item
    weightItems = [] #Arreglo con el peso de cada item
    with open(path + file,'r') as f:
        line =f.readline().split()
        numItem = int(line[0]) #Cantidad de items
        knapsackSize = int(line[1]) #Peso maximo
        #print('Numero de items: ',numItem,'Capacidad de mochila: ',knapsackSize)
        
        for line in f:
            vector = line.split()
            valueItems.append(float(vector[0]))
            weightItems.append(float(vector[1]))



    #print(valueItem,weightItem)
    f.close()
    return numItem, knapsackSize, valueItems, weightItems


