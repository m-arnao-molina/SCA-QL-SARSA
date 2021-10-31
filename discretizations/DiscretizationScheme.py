import numpy as np
from scipy import special as scyesp

"""
Realiza una transformación de una matriz en el dominio de los continuos a una matriz binaria.
Usada para discretización de soluciones en metaheurísticas. 
Donde cada fila es una solución (individuo de la población), y las columnas de la matriz corresponden a las dimensiones de la solución.
Params
----------
matrixCont : matriz de continuos.
matrixBin  : matriz de poblacion binarizada en t-1. 
bestRowBin : individuo (fila) de mejor fitness.
solutionsRanking: Lista de indices ordenados por fitness, en la posición 0 esta el best
transferFunction: Funciones de transferencia (V1,..,V4, S1,..,S4)
binarizationOperator: Operador de binarización (Standard,Complement, Elitist, Static, Roulette)
Returns
-------
matrixBinOut: matriz binaria.

Definiciones
---------
Para t=0, es decir, si matrixBin es vacía, se utiliza por defecto binarizationOperator = Standard.
"""


class DiscretizationScheme:
    def __init__(self, continuousMatrix, binMatrix, solutionsRanking, transferFunction, binarizationOperator):
        self.continuousMatrix = continuousMatrix
        self.binMatrix = binMatrix
        self.solutionsRanking = solutionsRanking
        self.transferFunction = transferFunction
        self.binarizationOperator = binarizationOperator
        self.bestRow = np.argmin(solutionsRanking)

        # Output
        self.probMatrix = np.zeros(self.continuousMatrix.shape)
        self.binMatrixOut = np.zeros(self.binMatrix.shape)

    # Transfer functions
    def tV1(self):
        self.probMatrix = np.abs(scyesp.erf(np.divide(np.sqrt(np.pi), 2) * self.continuousMatrix))

    def tV2(self):
        self.probMatrix = np.abs(np.tanh(self.continuousMatrix))

    def tV3(self):
        self.probMatrix = np.abs(np.divide(self.continuousMatrix, np.sqrt(1 + np.power(self.continuousMatrix, 2))))

    def tV4(self):
        self.probMatrix = np.abs(np.divide(2, np.pi) * np.arctan(np.divide(np.pi, 2) * self.continuousMatrix))

    def tS1(self):
        self.probMatrix = np.divide(1, (1 + np.exp(-2 * self.continuousMatrix)))

    def tS2(self):
        self.probMatrix = np.divide(1, (1 + np.exp(-1 * self.continuousMatrix)))

    def tS3(self):
        self.probMatrix = np.divide(1, (1 + np.exp(np.divide(-1 * self.continuousMatrix, 2))))

    def tS4(self):
        self.probMatrix = np.divide(1, (1 + np.exp(np.divide(-1 * self.continuousMatrix, 3))))

    # Binarization methods
    def bStandard(self):
        randMatrix = np.random.uniform(low=0.0, high=1.0, size=self.continuousMatrix.shape)
        self.binMatrixOut = np.greater(self.probMatrix, randMatrix).astype(int)

    def bComplement(self):
        randMatrix = np.random.uniform(low=0.0, high=1.0, size=self.continuousMatrix.shape)
        complementMatrix = np.abs(1 - self.binMatrix)
        # cambie greater, por greater_equal (por ecuacion)
        self.binMatrixOut = np.multiply(np.greater_equal(self.probMatrix, randMatrix).astype(int), complementMatrix)
        return self.binMatrixOut

    def bElitist(self):
        randMatrix = np.random.uniform(low=0.0, high=1.0, size=self.continuousMatrix.shape)
        # greater, porque es estricto en la ecuacion.
        conditionMatrix = np.greater(self.probMatrix, randMatrix)
        #todo: validar que el index exista
        bestIndividual = self.binMatrix[self.bestRow]
        # si ProbT > Rand() , then bestIndividualBin, else 0
        self.binMatrixOut = np.where(conditionMatrix == True, bestIndividual, 0)

    def bStatic(self):
        alfa = 1/3
        self.binMatrixOut[self.probMatrix <= alfa] = 0
        self.binMatrixOut[(self.probMatrix > alfa) & (self.probMatrix <= 0.5 * (1 + alfa))] = \
            self.binMatrix[(self.probMatrix > alfa) & (self.probMatrix <= 0.5 * (1 + alfa))]
        self.binMatrixOut[self.probMatrix >= 0.5 * (1 + alfa)] = 1

    def bElitistRoulette(self):
        randMatrix = np.random.uniform(low=0.0, high=1.0, size=self.continuousMatrix.shape)
        #greater, porque es estricto en la ecuacion.
        conditionMatrix = np.greater(self.probMatrix, randMatrix)
        # todo: validar que el index exista

        alfa = 0.2
        # condicion sum()==0, para el caso en que entregamos lista de rank con [0 0 0 0 0 0 0 0 ... 0 0 0 0] -> cuando generamos poblacion inicial
        if (self.solutionsRanking.shape[0] * alfa < 1) or (self.solutionsRanking.sum() == 0):
            #bestIndividual = self.matrixBin[self.SolutionRanking[0]]
            bestIndividual = self.binMatrix[0]
        else:
            bestSolutionRaking = int(self.solutionsRanking.shape[0] * alfa)
            random = np.random.randint(low=0, high=bestSolutionRaking)
            bestIndividual = self.binMatrix[random]

        # si ProbT > Rand() , then bestIndividualBin, else 0
        self.binMatrixOut = np.where(conditionMatrix == True, bestIndividual, 0)

    def binarize(self):
        transferFunctionsSwitcher = {
            'V1': self.tV1(),
            'V2': self.tV2(),
            'V3': self.tV3(),
            'V4': self.tV4(),
            'S1': self.tS1(),
            'S2': self.tS2(),
            'S3': self.tS3(),
            'S4': self.tS4()
        }
        transferFunctionsSwitcher.get(self.transferFunction.upper(), self.tV1())

        binarizationMethodsSwitcher = {
            'STANDARD': self.bStandard(),
            'COMPLEMENT': self.bComplement(),
            'ELITIST': self.bElitist(),
            'STATIC': self.bStatic(),
            'ELITIST_ROULETTE': self.bElitistRoulette()
        }
        binarizationMethodsSwitcher.get(self.binarizationOperator.upper(), self.bStandard())

        return self.binMatrixOut
