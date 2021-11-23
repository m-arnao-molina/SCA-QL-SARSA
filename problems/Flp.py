import numpy as np
from discretizations.DiscretizationScheme import DiscretizationScheme
from problems.Problem import Problem
from problems.repairs.RepairScp import RepairScp
from Problem.repair import cumpleRestricciones as cumpleGPU
from Problem.repair.ReparaStrategy import ReparaStrategy

class Flp(Problem):
    def __init__(self, instanceName, instancePath, populationSize, discretizationScheme, repairType):
        super().__init__(instanceName, instancePath)
        self.populationSize = populationSize
        self.discretizationScheme = discretizationScheme
        self.repairType = repairType

        self.costs = np.array([[]], dtype=np.int32)
        self.customerBudgets = np.array([], dtype=np.int32)
        self.facilities = 0
        self.customers = 0
        self.openFacilities = 0

        self.repairsQuantity = 0
        self.globalOptimum = self.getGlobalOptimum()
        self.population = np.random.uniform(low=-1.0, high=1.0, size=(self.populationSize, self.costs.shape[0]))
        self.binMatrix = np.random.randint(low=0, high=2, size=(self.populationSize, self.costs.shape[0]))
        self.fitness = np.zeros(self.populationSize)
        self.solutionsRanking = np.zeros(self.populationSize)
        self.weightConstraints = np.array([], dtype=np.float64)

    def getGlobalOptimum(self):
        instances = {
            'FLPr_100_40_01': [0, 2245],
            'FLPr_100_40_02': [1, 2259],
            'FLPr_100_40_03': [2, 2019],
            'FLPr_100_40_04': [3, 1533],
            'FLPr_100_40_05': [4, 2386],
            'FLPr_100_40_06': [5, 1960],
            'FLPr_100_40_07': [6, 2179],
            'FLPr_100_40_08': [7, 2139],
            'FLPr_100_40_09': [8, 1895],
            'FLPr_100_40_10': [9, 2209],
            'FLPr_100_100_01': [10, 2235],
            'FLPr_100_100_02': [11, 2240],
            'FLPr_100_100_03': [12, 1923],
            'FLPr_100_100_04': [13, 2133],
            'FLPr_100_100_05': [14, 2099],
            'FLPr_100_100_06': [15, 2237],
            'FLPr_100_100_07': [16, 1888],
            'FLPr_100_100_08': [17, 1825],
            'FLPr_100_100_09': [18, 1767],
            'FLPr_100_100_10': [19, 2368]
        }

        for instanceName in instances:
            if instanceName in self.instanceName:
                return instances[instanceName][1]

        return None

    def readInstance(self):
        file = open(self.instancePath, 'r')

        # Leer dimensiones
        line = file.readline().split()
        self.facilities = int(line[0])  # m
        self.customers = int(line[1])  # n
        self.openFacilities = int(line[2])  # p

        # Leer costos
        self.costs = np.zeros((self.facilities, self.customers), dtype=np.int32)
        for j in range(0, self.facilities):
            line = file.readline().split()
            self.costs[j] = np.array(line, dtype=np.int32)

        print(self.costs)
        print(self.costs.shape)

        # Leer restricciones (?)
        line = file.readline().split()
        self.customerBudgets = np.array(line, dtype=np.int32)

        print(self.customerBudgets)
        print(self.customerBudgets.shape)

        file.close()
        self.refreshComputedAttributes()

    def refreshComputedAttributes(self):
        self.population = np.random.uniform(low=-1.0, high=1.0, size=(self.populationSize, self.costs.shape[0]))
        self.binMatrix = np.random.randint(low=0, high=2, size=(self.populationSize, self.costs.shape[0]))
        self.fitness = np.zeros(self.populationSize)
        self.solutionsRanking = np.zeros(self.populationSize)
        self.weightConstraints = 1 / np.sum(self.customerBudgets)  # <<<

    def process(self, *args, **kwargs):
        # Binarización de 2 pasos
        self.binMatrix = DiscretizationScheme(
            self.population, self.binMatrix, self.solutionsRanking, self.discretizationScheme['transferFunction'],
            self.discretizationScheme['binarizationOperator']
        ).binarize()

        # Reparación
        if self.repairType == 3:  # Si se repara con GPU
            unrepairedBinMatrix = self.binMatrix.copy()
            self.binMatrix = cumpleGPU.reparaSoluciones(
                self.binMatrix, self.constraints, self.costs, self.weightConstraints
            )
            self.repairsQuantity = np.sum(self.binMatrix - unrepairedBinMatrix)
        else:  # Si se repara con CPU simple o compleja
            repairScp = RepairScp(self.repairType, self.constraints, self.costs)
            unrepairedBinMatrix = self.binMatrix.copy()
            for solution in range(self.binMatrix.shape[0]):
                if not repairScp.meets(self.binMatrix[solution]):
                    self.binMatrix[solution] = repairScp.repair(self.binMatrix[solution])[0]
            self.repairsQuantity = np.sum(self.binMatrix - unrepairedBinMatrix)

        # Calculamos Fitness
        self.fitness = np.sum(np.multiply(self.binMatrix, self.costs), axis=1)
        self.solutionsRanking = np.argsort(self.fitness)  # rankings de los mejores fitness

    def repair(self):
        pass