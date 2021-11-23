import json
from database.DatabaseORM import Database
from metaheuristics.ScaFlp import ScaFlp
from metaheuristics.ScaScp import ScaScp
from metaheuristics.ScaKp import ScaKp

# Algorithms
from metaheuristics.SCA_SCP import SCA_SCP
# from metaheuristics.SCAQL_SCP import SCAQL_SCP
# # from metaheuristics.SCASAR_SCP import SCASAR_SCP  <<< no existe
# # from metaheuristics.SCA_KP import SCA_KP          <<< no existe
# from metaheuristics.SCAQL_KP import SCAQL_KP
# from metaheuristics.SCASAR_KP import SCASAR_KP

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
    if lastPendingExecution.algorithmCodeName in ['SCA_SCP_BCL1', 'SCA_SCP_MIR2']:
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
    if lastPendingExecution.algorithmCodeName in ['SCA_SCP_BCL1', 'SCA_SCP_MIR2']:
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

    if lastPendingExecution.algorithmCodeName in ['SCA_FLP']:
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

    if lastPendingExecution.algorithmCodeName in ['SCA_KP']:
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
    exit(0)

    if lastPendingExecution.algorithmCodeName in ['SCA_SCP_QL1', 'SCA_SCP_QL2', 'SCA_SCP_QL3', 'SCA_SCP_QL4',
                                                       'SCA_SCP_QL5']:
        if SCAQL_SCP(lastPendingExecution.id,
                     lastPendingExecution.parameters['instance_file'],
                     lastPendingExecution.parameters['instance_directory'],
                     lastPendingExecution.parameters['population'],
                     lastPendingExecution.parameters['max_iterations'],
                     lastPendingExecution.parameters['discretization_scheme'],
                     lastPendingExecution.parameters['ql_alpha'],
                     lastPendingExecution.parameters['ql_gamma'],
                     lastPendingExecution.parameters['repair'],
                     lastPendingExecution.parameters['policy'],
                     lastPendingExecution.parameters['reward_type'],
                     lastPendingExecution.parameters['ql_alpha_type']
                     ):
            print(f"Execution ID: {lastPendingExecution.id} completed")

    if lastPendingExecution.algorithmCodeName in ['SCA_KPP_QL1', 'SCA_KPP_QL2', 'SCA_KPP_QL3', 'SCA_KPP_QL4',
                                                       'SCA_KPP_QL5']:
        if SCAQL_KP(lastPendingExecution.id,
                    lastPendingExecution.parameters['instance_file'],
                    lastPendingExecution.parameters['instance_directory'],
                    lastPendingExecution.parameters['population'],
                    lastPendingExecution.parameters['max_iterations'],
                    lastPendingExecution.parameters['discretization_scheme'],
                    lastPendingExecution.parameters['ql_alpha'],
                    lastPendingExecution.parameters['ql_gamma'],
                    lastPendingExecution.parameters['repair'],
                    lastPendingExecution.parameters['policy'],
                    lastPendingExecution.parameters['reward_type'],
                    lastPendingExecution.parameters['ql_alpha_type']
                    ):
            print(f"Execution ID: {lastPendingExecution.id} completed")

    if lastPendingExecution.algorithmCodeName in ['SCA_KPP_SA1', 'SCA_KPP_SA2', 'SCA_KPP_SA3', 'SCA_KPP_SA4',
                                                       'SCA_KPP_SA5']:
        if SCASAR_KP(lastPendingExecution.id,
                     lastPendingExecution.parameters['instance_file'],
                     lastPendingExecution.parameters['instance_directory'],
                     lastPendingExecution.parameters['population'],
                     lastPendingExecution.parameters['max_iterations'],
                     lastPendingExecution.parameters['discretization_scheme'],
                     lastPendingExecution.parameters['ql_alpha'],
                     lastPendingExecution.parameters['ql_gamma'],
                     lastPendingExecution.parameters['repair'],
                     lastPendingExecution.parameters['policy'],
                     lastPendingExecution.parameters['reward_type'],
                     lastPendingExecution.parameters['ql_alpha_type']
                     ):
            print(f"Execution ID: {lastPendingExecution.id} completed")
