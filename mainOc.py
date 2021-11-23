import json
from database.DatabaseORM import Database
from metaheuristics.ScaScp import ScaScp
from metaheuristics.ScaKp import ScaKp

db = Database()

while True:
    lastPendingExecution = db.getLastPendingExecution()
    if not lastPendingExecution:
        print('No hay ejecuciones pendientes')
        exit(0)

    print('--------------------------------------------------------------------------------------------------------\n')
    print(f'Execution ID: {lastPendingExecution.id} - {lastPendingExecution.algorithmCodeName}')
    print(json.dumps(lastPendingExecution.parameters, indent=4))
    print('--------------------------------------------------------------------------------------------------------\n')

    """
    if 'SCA_SCP' in lastPendingExecution.algorithmCodeName:
        if SCA_SCP(lastPendingExecution.id,
                   lastPendingExecution.parameters['instanceFile'],
                   lastPendingExecution.parameters['instanceDirectory'],
                   lastPendingExecution.parameters['populationSize'],
                   lastPendingExecution.parameters['maxIterations'],
                   lastPendingExecution.parameters['discretizationScheme'],
                   lastPendingExecution.parameters['repairType']
                   ):
            print(f'Execution ID: {lastPendingExecution.id} completed')
    """
    if 'SCA_SCP' in lastPendingExecution.algorithmCodeName:
        scaScp = ScaScp(
            executionId=lastPendingExecution.id,
            instanceName=lastPendingExecution.parameters['instanceName'],
            instanceFile=lastPendingExecution.parameters['instanceFile'],
            instanceDirectory=lastPendingExecution.parameters['instanceDirectory'],
            populationSize=lastPendingExecution.parameters['populationSize'],
            maxIterations=lastPendingExecution.parameters['maxIterations'],
            discretizationScheme=lastPendingExecution.parameters['discretizationScheme'],
            repairType=lastPendingExecution.parameters['repairType']
        )
        if scaScp.process():
            print(f'Execution ID: {lastPendingExecution.id} completed')
    # """

    if 'SCA_KP' in lastPendingExecution.algorithmCodeName:
        scaKp = ScaKp(
            executionId=lastPendingExecution.id,
            instanceName=lastPendingExecution.parameters['instanceName'],
            instanceFile=lastPendingExecution.parameters['instanceFile'],
            instanceDirectory=lastPendingExecution.parameters['instanceDirectory'],
            populationSize=lastPendingExecution.parameters['populationSize'],
            maxIterations=lastPendingExecution.parameters['maxIterations'],
            discretizationScheme=lastPendingExecution.parameters['discretizationScheme'],
            repairType=lastPendingExecution.parameters['repairType']
        )
        if scaKp.process():
            print(f'Execution ID: {lastPendingExecution.id} completed')
