import json
from database.DatabaseORM import Database
from metaheuristics.ScaFlp import ScaFlp
from metaheuristics.ScaKp import ScaKp
from metaheuristics.ScaScp import ScaScp


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

    if 'SCA_FLP' in lastPendingExecution.algorithmCodeName:
        scaFlp = ScaFlp(
            executionId=lastPendingExecution.id,
            instanceName=lastPendingExecution.parameters['instanceName'],
            instanceFile=lastPendingExecution.parameters['instanceFile'],
            instanceDirectory=lastPendingExecution.parameters['instanceDirectory'],
            populationSize=lastPendingExecution.parameters['populationSize'],
            maxIterations=lastPendingExecution.parameters['maxIterations'],
            discretizationScheme=lastPendingExecution.parameters['discretizationScheme'],
            repairType=lastPendingExecution.parameters['repairType']
        )
        if scaFlp.process():
            print(f'Execution ID: {lastPendingExecution.id} completed')

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