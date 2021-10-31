import numpy as np
from discretizations.DiscretizationScheme import DiscretizationScheme
from problems.Problem import Problem
from Problem.repair import cumpleRestricciones as cumpleGPU
from Problem.repair.ReparaStrategy import ReparaStrategy

class Scp(Problem):
    def __init__(self, instanceName, instancePath):
        super().__init__(instanceName, instancePath)
        self.costs = np.array([], dtype=np.int32)
        self.constraints = np.array([[]], dtype=np.int32)

    def readInstance(self):
        self.globalOptimum = self.getGlobalOptimum()
        file = open(self.instancePath, 'r')

        # Leer dimensiones
        line = file.readline().split()
        rows = int(line[0])
        columns = int(line[1])

        # Leer costos
        line = file.readline()
        costValuesCount = 0
        while line != '' and costValuesCount < columns:
            costValues = np.array(line.split(), dtype=np.int32)
            self.costs = np.concatenate((self.costs, costValues))
            costValuesCount += len(costValues)
            line = file.readline()

        # Leer restricciones
        self.constraints = np.zeros((rows, columns), dtype=np.int32)
        row = 0
        while line != '':
            oneValuesQuantity = int(line)
            oneValuesCount = 0
            line = file.readline()
            line = line.replace('\n', '').replace("\\n'", '')
            while line != '' and oneValuesCount < oneValuesQuantity:
                columns = line.split()
                for i in range(len(columns)):
                    column = int(columns[i]) - 1
                    self.constraints[row][column] = 1
                    oneValuesCount += 1
                line = file.readline()
            row += 1
        file.close()

    def getGlobalOptimum(self):
        instances = {
            'scp41': [0, 429],
            'scp42': [1, 512],
            'scp43': [2, 516],
            'scp44': [3, 494],
            'scp45': [4, 512],
            'scp46': [5, 560],
            'scp47': [6, 430],
            'scp48': [7, 492],
            'scp49': [8, 641],
            'scp410': [9, 514],
            'scp51': [10, 253],
            'scp52': [11, 302],
            'scp53': [12, 226],
            'scp54': [13, 242],
            'scp55': [14, 211],
            'scp56': [15, 213],
            'scp57': [16, 293],
            'scp58': [17, 288],
            'scp59': [18, 279],
            'scp510': [19, 265],
            'scp61': [20, 138],
            'scp62': [21, 146],
            'scp63': [22, 145],
            'scp64': [23, 131],
            'scp65': [24, 161],
            'scpa1': [25, 253],
            'scpa2': [26, 252],
            'scpa3': [27, 232],
            'scpa4': [28, 234],
            'scpa5': [29, 236],
            'scpb1': [30, 69],
            'scpb2': [31, 76],
            'scpb3': [32, 80],
            'scpb4': [33, 79],
            'scpb5': [34, 72],
            'scpc1': [35, 227],
            'scpc2': [36, 219],
            'scpc3': [37, 243],
            'scpc4': [38, 219],
            'scpc5': [39, 215],
            'scpd1': [40, 60],
            'scpd2': [41, 66],
            'scpd3': [42, 72],
            'scpd4': [43, 62],
            'scpd5': [44, 61],
            'scpnre1': [45, 29],
            'scpnre2': [46, 30],
            'scpnre3': [47, 27],
            'scpnre4': [48, 28],
            'scpnre5': [49, 28],
            'scpnrf1': [50, 14],
            'scpnrf2': [51, 15],
            'scpnrf3': [52, 14],
            'scpnrf4': [53, 14],
            'scpnrf5': [54, 13],
            'scpnrg1': [55, 176],
            'scpnrg2': [56, 154],
            'scpnrg3': [57, 166],
            'scpnrg4': [58, 168],
            'scpnrg5': [59, 168],
            'scpnrh1': [60, 63],
            'scpnrh2': [61, 63],
            'scpnrh3': [62, 59],
            'scpnrh4': [63, 58],
            'scpnrh5': [64, 55]
        }

        for instanceName in instances:
            if instanceName in self.instanceName:
                return instances[instanceName][1]

        return None

    # action : esquema de discretizacion DS
    def process(
        self, population, binMatrix, solutionsRanking, costs, constraints, discretizationScheme, repairType,
        weightConstraints, *args, **kwargs
    ):
        # Binarización de 2 pasos
        binMatrix = DiscretizationScheme(
            population, binMatrix, solutionsRanking, discretizationScheme['transferFunction'],
            discretizationScheme['binarizationOperator']
        ).binarize()

        # Reparamos
        if repairType == 3:  # Si reparamos con GPU
            matrizSinReparar = binMatrix
            binMatrix = cumpleGPU.reparaSoluciones(binMatrix, constraints, costs, weightConstraints)
            matrizReparada = binMatrix
            repairsQuantity = np.sum(matrizReparada - matrizSinReparar)

        else:  # Si reparamos con CPU simple o complejo
            repair = ReparaStrategy(constraints, costs, constraints.shape[0], constraints.shape[1])
            matrizSinReparar = binMatrix
            for solucion in range(binMatrix.shape[0]):
                if repair.cumple(binMatrix[solucion]) == 0:
                    binMatrix[solucion] = \
                    repair.repara_one(binMatrix[solucion], repairType, None, weightConstraints)[0]
            matrizReparada = binMatrix
            repairsQuantity = np.sum(matrizReparada - matrizSinReparar)

        # Calculamos Fitness
        fitness = np.sum(np.multiply(binMatrix, costs), axis=1)
        solutionsRanking = np.argsort(fitness)  # rankings de los mejores fitness

        return binMatrix, fitness, solutionsRanking, repairsQuantity
