import numpy as np
from discretizations.DiscretizationScheme import DiscretizationScheme
from problems.Problem import Problem


class Kp(Problem):
    def __init__(self, instanceName, instancePath, populationSize, discretizationScheme):
        super().__init__(instanceName, instancePath)
        self.populationSize = populationSize
        self.discretizationScheme = discretizationScheme
        self.valueItems = np.array([], dtype=np.float64)  # Arreglo con el valor de cada item
        self.weightItems = np.array([], dtype=np.float64)  # Arreglo con el peso de cada item
        self.itemsQuantity = 0
        self.knapsackSize = 0
        self.repairsQuantity = 0 # ?
        self.globalOptimum = self.getGlobalOptimum()
        self.population = np.random.uniform(low=-1.0, high=1.0, size=(self.populationSize, self.itemsQuantity))
        self.binMatrix = np.random.randint(low=0, high=2, size=(self.populationSize, self.itemsQuantity))
        self.fitness = np.zeros(self.populationSize)
        self.solutionsRanking = np.zeros(self.populationSize)

    def getGlobalOptimum(self):
        instances = {
            'f1_l-d_kp_10_269': [0, 0],
            'f2_l-d_kp_20_878': [1, 0],
            'f3_l-d_kp_4_20': [2, 0],
            'f4_l-d_kp_4_11': [3, 0],
            'f5_l-d_kp_15_375': [4, 0],
            'f6_l-d_kp_10_60': [5, 0],
        }

        for instanceName in instances:
            if instanceName in self.instanceName:
                return instances[instanceName][1]

        return None

    def readInstance(self):
        file = open(self.instancePath, 'r')

        line = file.readline().split()
        self.itemsQuantity = int(line[0])  # Cantidad de items
        self.knapsackSize = int(line[1])  # Peso maximo

        row = 0
        while line != '' and row < self.itemsQuantity:
            line = file.readline().split()
            self.valueItems = np.append(self.valueItems, float(line[0]))
            self.weightItems = np.append(self.weightItems, float(line[1]))
            row += 1
        file.close()
        self.refreshComputedAttributes()

    def refreshComputedAttributes(self):
        self.population = np.random.uniform(low=-1.0, high=1.0, size=(self.populationSize, self.itemsQuantity))
        self.binMatrix = np.random.randint(low=0, high=2, size=(self.populationSize, self.itemsQuantity))
        self.fitness = np.zeros(self.populationSize)
        self.solutionsRanking = np.zeros(self.populationSize)

    def process(self, *args, **kwargs):
        # Binarización de 2 pasos
        self.binMatrix = DiscretizationScheme(
            self.population, self.binMatrix, self.solutionsRanking, self.discretizationScheme['transferFunction'],
            self.discretizationScheme['binarizationOperator']
        ).binarize()

        # Se crea un vector con la densidad de los items solucion y se ordenan de manera no decreciente, moviendo los valores de la matrixBin
        density = np.divide(self.valueItems, self.weightItems)
        for i in range(self.binMatrix.shape[0]):
            density = np.multiply(self.binMatrix[i], density)
            for j in range(len(self.binMatrix[i]) - 1):
                for k in range(len(self.binMatrix[i]) - j - 1):
                    if density[k] > density[k + 1]:
                        density[k], density[k + 1] = density[k + 1], density[k]
                        self.weightItems[k], self.weightItems[k + 1] = self.weightItems[k + 1], self.weightItems[k]
                        self.valueItems[k], self.valueItems[k + 1] = self.valueItems[k + 1], self.valueItems[k]
                        self.binMatrix[i][k], self.binMatrix[i][k + 1] = self.binMatrix[i][k + 1], self.binMatrix[i][k]

            # Si es factible la solucion no se repara, de lo contrario se repara
            weightItems = np.multiply(self.binMatrix[i], self.weightItems)
            totalWeights = np.sum(weightItems)
            if totalWeights <= self.knapsackSize:
                for j in range(len(self.binMatrix[i])):
                    if self.binMatrix[i][j] == 0 and totalWeights + self.weightItems[j] <= self.knapsackSize:
                        self.binMatrix[i][j] = 1
                        totalWeights = totalWeights + self.weightItems[j]
            else:
                for j in range(len(self.binMatrix[i]) - 1, -1, -1):
                    if self.binMatrix[i][j] == 1 and totalWeights > self.knapsackSize:
                        self.binMatrix[i][j] = 0
                        totalWeights = totalWeights - self.weightItems[j]

            self.fitness[i] = np.sum(np.multiply(self.binMatrix[i], self.valueItems))

        self.solutionsRanking = (-self.fitness).argsort()  # Ranking de los mejores fitness
        bestSolutionOldFitness = self.fitness[self.solutionsRanking[0]]
        bestSolutionOldBin = self.binMatrix[self.solutionsRanking[0]]
