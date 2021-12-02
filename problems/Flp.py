import random
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

        print('self.costs:', self.costs)
        print('self.costs.shape:', self.costs.shape)
        print('self.costs.shape[0]:', self.costs.shape[0])

        # Leer restricciones (?)
        line = file.readline().split()
        self.customerBudgets = np.array(line, dtype=np.int32)

        print('self.customerBudgets:', self.customerBudgets)
        print('self.customerBudgets.shape:', self.customerBudgets.shape)

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
        print('----------START process----------')
        for solution in range(self.binMatrix.shape[0]):
            openFacilities = np.sum(self.binMatrix[solution])
            print(f'[antes] solution: {solution} - open facilities: {openFacilities}')
            print('[antes] self.binMatrix[solution]:', self.binMatrix[solution])

            if openFacilities > self.openFacilities:  # si hay mas de p tiendas abiertas
                potentialCloseFacilities = np.where(self.binMatrix[solution] == 1)[0]
                randIndexes = random.sample(set(potentialCloseFacilities), k=(openFacilities - self.openFacilities))
                self.binMatrix[solution][randIndexes] = 0
            elif openFacilities < self.openFacilities:  # si hay menos de p tiendas abiertas
                potentialOpenNewFacilities = np.where(self.binMatrix[solution] == 0)[0]
                randIndexes = random.sample(set(potentialOpenNewFacilities), k=(self.openFacilities - openFacilities))
                self.binMatrix[solution][randIndexes] = 1

            openFacilities = np.sum(self.binMatrix[solution])
            print(f'[despues] solution: {solution} - open facilities: {openFacilities}')
            print('[despues] self.binMatrix[solution]:', self.binMatrix[solution])

            facilitiesSelectedCosts = np.zeros((self.openFacilities, self.customers), dtype=np.int32)
            i = 0
            for facilityOpen in range(self.binMatrix[solution].shape[0]):
                if self.binMatrix[solution][facilityOpen] == 1:
                    facilitiesSelectedCosts[i] = self.costs[facilityOpen]
                    i += 1
            print('facilitiesSelectedCosts:', facilitiesSelectedCosts)

            self.fitness[solution] = 0
            for customer in range(self.customerBudgets.shape[0]):
                facilitiesCosts = np.sort(facilitiesSelectedCosts[:, customer])
                if facilitiesCosts[0] <= self.customerBudgets[customer]:
                    self.fitness[solution] += facilitiesCosts[0]

            print('self.fitness[solution]:', self.fitness[solution])

        self.solutionsRanking = (-self.fitness).argsort()

        print('self.fitness:', self.fitness)
        print('self.solutionsRanking:', self.solutionsRanking)

        bestSolutionIdx = self.solutionsRanking[0]
        bestSolution = self.binMatrix[bestSolutionIdx]
        facilitiesSelectedCosts = np.zeros((self.openFacilities, self.customers), dtype=np.int32)
        i = 0
        for facilityOpen in range(self.binMatrix[bestSolutionIdx].shape[0]):
            if self.binMatrix[bestSolutionIdx][facilityOpen] == 1:
                print('self.costs[facilityOpen]:', self.costs[facilityOpen])
                facilitiesSelectedCosts[i] = self.costs[facilityOpen]
                i += 1

        print('bestSolutionFitness:', self.fitness[bestSolutionIdx])
        print('bestSolutionIdx:', bestSolutionIdx)
        print('bestSolution:', bestSolution)
        print('facilitiesSelectedCosts:', facilitiesSelectedCosts)
        print('self.customerBudgets:', self.customerBudgets)


    def process2(self, *args, **kwargs):
        # Binarización de 2 pasos
        self.binMatrix = DiscretizationScheme(
            self.population, self.binMatrix, self.solutionsRanking, self.discretizationScheme['transferFunction'],
            self.discretizationScheme['binarizationOperator']
        ).binarize()

        print('self.binMatrix:', self.binMatrix)
        print('self.binMatrix.shape:', self.binMatrix.shape)

        for solution in range(self.binMatrix.shape[0]):
            openFacilities = np.sum(self.binMatrix[solution])
            print(f'[antes] solution: {solution} - open facilities: {openFacilities}')
            if openFacilities < self.openFacilities:
                print(f'>>>>>>>>>>>>>>>>>>>>>>><{openFacilities}<<<<<<<<<<<<<<<<<<<<<<<<')
            if openFacilities > self.openFacilities:  # si hay mas de p tiendas abiertas
                facilitiesOpenCount = 0
                for facilityOpen in range(self.binMatrix[solution].shape[0]):
                    if self.binMatrix[solution][facilityOpen] == 1:  # reparacion basica (cierra las primeras tiendas abiertas, podria aplicarse random)
                        if facilitiesOpenCount == self.openFacilities:
                            self.binMatrix[solution][facilityOpen] = 0
                        else:
                            facilitiesOpenCount += 1
            elif openFacilities < self.openFacilities:  # si hay menos de p tiendas abiertas
                facilitiesOpenCount = openFacilities
                for facilityOpen in range(self.binMatrix[solution].shape[0]):
                    if self.binMatrix[solution][facilityOpen] == 0:  # reparacion basica (abre las primeras tiendas cerradas hasta completar, podria aplicarse random)
                        if facilitiesOpenCount < self.openFacilities:
                            self.binMatrix[solution][facilityOpen] = 1
                            facilitiesOpenCount += 1

            openFacilities = np.sum(self.binMatrix[solution])
            print(f'[despues] solution: {solution} - open facilities: {openFacilities}')
            #if not repairScp.meets(self.binMatrix[solution]):
            #    self.binMatrix[solution] = repairScp.repair(self.binMatrix[solution])[0]

            print('self.binMatrix[solution]:', self.binMatrix[solution])
            #print('self.costs:', self.costs)

            totalFitness = 0
            facilitiesSelectedCosts = np.zeros((self.openFacilities, self.customers), dtype=np.int32)
            i = 0
            for facilityOpen in range(self.binMatrix[solution].shape[0]):
                if self.binMatrix[solution][facilityOpen] == 1:
                    totalFitness = np.sum(self.costs[facilityOpen]) # deberia considerar solo los mejores precios para los clientes de las tiendas selccionadas
                    facilitiesSelectedCosts[i] = self.costs[facilityOpen]
                    i += 1

            #self.fitness[solution] = totalFitness
            #print('totalFitness:', totalFitness)
            print('facilitiesSelectedCosts:', facilitiesSelectedCosts)

            self.fitness[solution] = 0
            customersIdxServed = np.array([], dtype=np.int)
            for customer in range(self.customerBudgets.shape[0]):
                #facilitiesCosts = np.argsort(facilitiesSelectedCosts[:, customer]) # retorna indices, puede servir para saber cuales tiendas no sirven
                facilitiesCosts = np.sort(facilitiesSelectedCosts[:, customer])
                if facilitiesCosts[0] <= self.customerBudgets[customer]:
                    self.fitness[solution] += facilitiesCosts[0]
                    customersIdxServed = np.append(customersIdxServed, customer)
                # if facilitiesCosts[0] > self.customerBudgets[customer]:  # verificar si ninguna tienda de las seleccionadas respeta el presupuesto del cliente
                #     self.fitness[solution] = 0  # por ahora fijo el fitness en 0 de aquellas soluciones inviables
                # else:
                #     self.fitness[solution] += facilitiesCosts[0]

                print('customer:', customer)
                print('customerBudget:', self.customerBudgets[customer])
                print('facilitiesCosts:', facilitiesCosts)

            print('customersIdxServed:', customersIdxServed)
            print('self.fitness[solution]:', self.fitness[solution])

            budgets = 0
            for customerIdxServed in customersIdxServed:
                budgets += self.customerBudgets[customerIdxServed]
            print('budgets:', budgets)

            #if self.fitness[solution] != 0:
            #    print('solucion factible encontrada')
            #    exit(0)




            #self.fitness[solution] = np.sum(np.multiply(self.binMatrix[solution], self.costs), axis=1)


        # Calculamos Fitness
        #self.fitness = np.sum(np.multiply(self.binMatrix, self.costs), axis=1)
        #self.solutionsRanking = np.argsort(self.fitness)  # rankings de los mejores fitness
        self.solutionsRanking = (-self.fitness).argsort()

        print('self.fitness:', self.fitness)
        print('self.solutionsRanking:', self.solutionsRanking)

        bestSolutionIdx = self.solutionsRanking[0]
        bestSolution = self.binMatrix[bestSolutionIdx]
        facilitiesSelectedCosts = np.zeros((self.openFacilities, self.customers), dtype=np.int32)
        i = 0
        for facilityOpen in range(self.binMatrix[bestSolutionIdx].shape[0]):
            if self.binMatrix[bestSolutionIdx][facilityOpen] == 1:
                print('self.costs[facilityOpen]:', self.costs[facilityOpen])
                facilitiesSelectedCosts[i] = self.costs[facilityOpen]
                i += 1

        print('bestSolutionFitness:', self.fitness[bestSolutionIdx])
        print('bestSolutionIdx:', bestSolutionIdx)
        print('bestSolution:', bestSolution)
        print('facilitiesSelectedCosts:', facilitiesSelectedCosts)
        print('self.customerBudgets:', self.customerBudgets)

    def repair(self):
        pass