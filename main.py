import json
from database.Database import Database

# Algorithms
from metaheuristics.SCA_SCP import SCA_SCP
from metaheuristics.SCAQL_SCP import SCAQL_SCP
# from metaheuristics.SCASAR_SCP import SCASAR_SCP  <<< no existe
# from metaheuristics.SCA_KP import SCA_KP          <<< no existe
from metaheuristics.SCAQL_KP import SCAQL_KP
from metaheuristics.SCASAR_KP import SCASAR_KP

database = Database()

while True:
    lastPendingExecution = database.getLastPendingExecution()
    if not lastPendingExecution:
        print('No hay más ejecuciones pendientes')
        exit(0)

    print("--------------------------------------------------------------------------------------------------------\n")
    print(f"Execution ID: {lastPendingExecution['id']} - {lastPendingExecution['algorithm_code_name']}")
    print(json.dumps(lastPendingExecution['parameters'], indent=4))
    print("--------------------------------------------------------------------------------------------------------\n")

    if lastPendingExecution['algorithm_code_name'] in ['SCA_SCP_BCL1', 'SCA_SCP_MIR2']:
        if SCA_SCP(lastPendingExecution['id'],
                   lastPendingExecution['parameters']['instance_file'],
                   lastPendingExecution['parameters']['instance_directory'],
                   lastPendingExecution['parameters']['population'],
                   lastPendingExecution['parameters']['max_iterations'],
                   lastPendingExecution['parameters']['discretization_scheme'],
                   lastPendingExecution['parameters']['repair']
                   ):
            print(f"Execution ID: {lastPendingExecution['id']} completed")
    exit(0)

    if lastPendingExecution['algorithm_code_name'] in ['SCA_SCP_QL1', 'SCA_SCP_QL2', 'SCA_SCP_QL3', 'SCA_SCP_QL4',
                                                       'SCA_SCP_QL5']:
        if SCAQL_SCP(lastPendingExecution['id'],
                     lastPendingExecution['parameters']['instance_file'],
                     lastPendingExecution['parameters']['instance_directory'],
                     lastPendingExecution['parameters']['population'],
                     lastPendingExecution['parameters']['max_iterations'],
                     lastPendingExecution['parameters']['discretization_scheme'],
                     lastPendingExecution['parameters']['ql_alpha'],
                     lastPendingExecution['parameters']['ql_gamma'],
                     lastPendingExecution['parameters']['repair'],
                     lastPendingExecution['parameters']['policy'],
                     lastPendingExecution['parameters']['reward_type'],
                     lastPendingExecution['parameters']['ql_alpha_type']
                     ):
            print(f"Execution ID: {lastPendingExecution['id']} completed")

    if lastPendingExecution['algorithm_code_name'] in ['SCA_KPP_QL1', 'SCA_KPP_QL2', 'SCA_KPP_QL3', 'SCA_KPP_QL4',
                                                       'SCA_KPP_QL5']:
        if SCAQL_KP(lastPendingExecution['id'],
                    lastPendingExecution['parameters']['instance_file'],
                    lastPendingExecution['parameters']['instance_directory'],
                    lastPendingExecution['parameters']['population'],
                    lastPendingExecution['parameters']['max_iterations'],
                    lastPendingExecution['parameters']['discretization_scheme'],
                    lastPendingExecution['parameters']['ql_alpha'],
                    lastPendingExecution['parameters']['ql_gamma'],
                    lastPendingExecution['parameters']['repair'],
                    lastPendingExecution['parameters']['policy'],
                    lastPendingExecution['parameters']['reward_type'],
                    lastPendingExecution['parameters']['ql_alpha_type']
                    ):
            print(f"Execution ID: {lastPendingExecution['id']} completed")

    if lastPendingExecution['algorithm_code_name'] in ['SCA_KPP_SA1', 'SCA_KPP_SA2', 'SCA_KPP_SA3', 'SCA_KPP_SA4',
                                                       'SCA_KPP_SA5']:
        if SCASAR_KP(lastPendingExecution['id'],
                     lastPendingExecution['parameters']['instance_file'],
                     lastPendingExecution['parameters']['instance_directory'],
                     lastPendingExecution['parameters']['population'],
                     lastPendingExecution['parameters']['max_iterations'],
                     lastPendingExecution['parameters']['discretization_scheme'],
                     lastPendingExecution['parameters']['ql_alpha'],
                     lastPendingExecution['parameters']['ql_gamma'],
                     lastPendingExecution['parameters']['repair'],
                     lastPendingExecution['parameters']['policy'],
                     lastPendingExecution['parameters']['reward_type'],
                     lastPendingExecution['parameters']['ql_alpha_type']
                     ):
            print(f"Execution ID: {lastPendingExecution['id']} completed")
