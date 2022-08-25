import numpy as np
from discretizations.DiscretizationScheme import DiscretizationScheme


class Mlp:
    def __init__(
        self, populationSize, discretizationScheme, repairType, facilities, customers, openFacilities, costs,
        customerBudgets, flpSolutions, flpBestSolution
    ):
        self.populationSize = populationSize
        self.discretizationScheme = discretizationScheme
        self.repairType = repairType

        # self.costs = np.array([[]], dtype=np.int32)
        # self.customerBudgets = np.array([], dtype=np.int32)
        # self.facilities = 0
        # self.customers = 0
        # self.openFacilities = 0

        self.facilities = facilities
        self.customers = customers
        self.openFacilities = openFacilities
        self.costs = costs
        self.customerBudgets = customerBudgets
        self.flpSolutions = flpSolutions
        self.flpBestSolution = flpBestSolution

        self.repairsQuantity = 0
        self.population = np.random.uniform(low=-1.0, high=1.0, size=(self.populationSize, self.facilities))  # 40x40 => 40x5
        self.discMatrix = np.random.randint(low=0.0, high=1.0, size=(self.populationSize, self.facilities))  # 40x40 => 40x5
        # self.binMatrix = np.random.randint(low=0, high=2, size=(self.populationSize, self.costs.shape[0]))
        self.fitness = np.zeros(self.populationSize)
        self.solutionsRanking = np.zeros(self.populationSize)
        self.selectPrices = np.zeros((self.populationSize, self.facilities), dtype=np.float32)
        self.customersSpending = np.zeros((self.populationSize, self.customers))
        self.selectPricesByCustomer = np.zeros((self.populationSize, self.customers))

    def process(self, *args, **kwargs):
        # print('--- Paso 5.2.1 process ---')
        # Discretización
        #self.discMatrix = DiscretizationScheme(
        #    self.population, None, self.solutionsRanking, self.discretizationScheme['transferFunction'],
        #    None, discMatrix=self.discMatrix
        #).discretize()

        self.discMatrix = np.random.uniform(low=0.0, high=1.0, size=(self.populationSize, self.facilities))

        # print('self.population:', self.population)
        # print('self.population.shape:', self.population.shape)
        # print('self.discMatrix:', self.discMatrix)
        # print('self.discMatrix.shape:', self.discMatrix.shape)
        # print('self.solutionsRanking:', self.solutionsRanking)
        #exit(0)

        # print('self.flpSolutions:', self.flpSolutions)
        # print('self.flpBestSolution:', self.flpBestSolution)
        # print('self.flpSolutions[self.flpBestSolution]:', self.flpSolutions[self.flpBestSolution])

        flpBestSolutionBin = self.flpSolutions[self.flpBestSolution]
        #flpBestSolutionBin = self.flpSolutions[10]
        #print('flpBestSolutionBin:', flpBestSolutionBin)
        for solution in range(self.discMatrix.shape[0]):                                        # Tomar los precios solo de aquellas tiendas que están abiertas
            i = 0
            for facilityOpen in range(flpBestSolutionBin.shape[0]):
                if flpBestSolutionBin[facilityOpen] == 1:  # Comprobar si la tienda está abierta
                    self.selectPrices[solution][i] = self.discMatrix[solution][facilityOpen]  # Agregar al arreglo los precios de la tienda
                i += 1
            # print('self.selectPrices[solution]:', self.selectPrices[solution])
            # print('self.discMatrix[solution]:', self.discMatrix[solution])

            self.fitness[solution] = 0
            self.selectPricesByCustomer[solution] = np.zeros(self.customers)
            self.customersSpending[solution] = np.zeros(self.customers)
            for customer in range(self.customerBudgets.shape[0]):                               # Recorrer arreglo de presupuestos de clientes
                # print('Cliente:', customer)

                facilitiesPricesIdxs = (-self.selectPrices[solution]).argsort()[:self.openFacilities]  # Ordenar precios de mayor a menor
                # print('facilitiesPricesIdxs:', facilitiesPricesIdxs)

                for facilitiesPriceIdx in facilitiesPricesIdxs:
                    price = round(self.selectPrices[solution][facilitiesPriceIdx] * 100)
                    #print('facilitiesPriceIdx:', facilitiesPriceIdx)
                    #print('self.selectPrices[solution]:', self.selectPrices[solution])
                    #print('self.selectPrices[solution][facilitiesPriceIdx]:', self.selectPrices[solution][facilitiesPriceIdx])

                    #print('Precio elegido:', price)
                    #print('Costo de transporte:', self.costs[facilitiesPriceIdx][customer])
                    #print('Presupuesto del cliente:', self.customerBudgets[customer])

                    flag = False
                    if self.selectPricesByCustomer[solution][customer] != 0:
                        flag = True
                        print('[antes 1] self.selectPricesByCustomer[solution][customer]:', self.selectPricesByCustomer[solution][customer])

                    if price + self.costs[facilitiesPriceIdx][customer] <= self.customerBudgets[customer]:  # Se elige el mayor precio + el costo de transporte y
                        # se comprueba si esta suma es menor o igual al presupuesto del cliente
                        # Se debe tomar el costo de transporte de la tienda elegida
                        self.fitness[solution] += price                                                     # Sumar al fitness de la solución
                        self.customersSpending[solution][customer] = price + self.costs[facilitiesPriceIdx][customer]
                        #print('[antes] self.selectPricesByCustomer[solution][customer]:', self.selectPricesByCustomer[solution][customer])
                        self.selectPricesByCustomer[solution][customer] = price
                        if flag:
                            print('[despues] self.selectPricesByCustomer[solution][customer]:', self.selectPricesByCustomer[solution][customer])
                        break

            print('self.fitness[solution]:', self.fitness[solution])

            if self.fitness[solution] > 2100: # 2100
                print('SOLUCION ALTA:', self.fitness[solution], '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                print('Tiendas abiertas:', flpBestSolutionBin)
                print('Precios (brutos):', self.selectPrices[solution])
                print('Precios (procesados):', [round(price * 100) for price in self.selectPrices[solution]])
                print('Gastos de los clientes:', self.customersSpending[solution])
                print('Precios elegidos por los clientes:', self.selectPricesByCustomer[solution])

            if self.fitness[solution] == 2245:  # 2245
                print('MEJOR SOLUCION ENCONTRADA:', self.fitness[solution], '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('Tiendas abiertas:', flpBestSolutionBin)
                print('Precios (brutos):', self.selectPrices[solution])
                print('Precios (procesados):', [round(price * 100) for price in self.selectPrices[solution]])
                print('Gastos de los clientes:', self.customersSpending[solution])
                print('Precios elegidos por los clientes:', self.selectPricesByCustomer[solution])

                #exit(0)

        self.solutionsRanking = (-self.fitness).argsort()
        #print('MLP self.solutionsRanking:', self.solutionsRanking)

    def repair(self):
        pass