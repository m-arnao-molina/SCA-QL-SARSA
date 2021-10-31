import os
from dotenv import dotenv_values
from database.DatabaseORM import Database

db = Database()

class Metaheuristic:
    def __init__(
            self, executionId, instanceName, instanceFile, instanceDirectory, populationSize, maxIterations,
            discretizationScheme, repairType
    ):
        config = dotenv_values('.env')
        workdir = os.path.abspath(os.getcwd())
        workdirInstance = workdir + config['DIR_INSTANCES']
        self.instancePath = workdirInstance + instanceDirectory + instanceFile

        self.instanceName = instanceName
        self.executionId = executionId
        self.populationSize = populationSize
        self.maxIterations = maxIterations
        self.discretizationScheme = discretizationScheme
        self.repairType = repairType

    def validateInstance(self):
        if not os.path.exists(self.instancePath):
            print(f'No se encontró la instancia: {self.instancePath}')
            return False
        return True

    def process(self, *args, **kwargs):
        pass
