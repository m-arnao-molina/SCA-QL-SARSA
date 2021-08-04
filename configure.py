import json
from database.Database import Database

database = Database()

algorithms = {
    'SCP': [
        {'code_name': 'SCA_SCP_BCL1', 'reward_type': None},
        # {'code_name': 'SCA_SCP_MIR2', 'reward_type': None},
        # {'code_name': 'SCA_SCP_QL1', 'reward_type': 'withPenalty1'},
        # {'code_name': 'SCA_SCP_QL2', 'reward_type': 'withoutPenalty1'},
        # {'code_name': 'SCA_SCP_QL3', 'reward_type': 'globalBest'},
        # {'code_name': 'SCA_SCP_QL4', 'reward_type': 'rootAdaptation'},
        # {'code_name': 'SCA_SCP_QL5', 'reward_type': 'escalatingMultiplicativeAdaptation'}
    ],
    'KP': [
        # {'code_name': 'SCA_KPP_QL1', 'reward_type': 'withPenalty1'},
        # {'code_name': 'SCA_KPP_QL2', 'reward_type': 'withoutPenalty1'},
        # {'code_name': 'SCA_KPP_QL3', 'reward_type': 'globalBest'},
        # {'code_name': 'SCA_KPP_QL4', 'reward_type': 'rootAdaptation'},
        # {'code_name': 'SCA_KPP_QL5', 'reward_type': 'escalatingMultiplicativeAdaptation'},
        # {'code_name': 'SCA_KPP_SA1', 'reward_type': 'withPenalty1'},
        # {'code_name': 'SCA_KPP_SA2', 'reward_type': 'withoutPenalty1'},
        # {'code_name': 'SCA_KPP_SA3', 'reward_type': 'globalBest'},
        # {'code_name': 'SCA_KPP_SA4', 'reward_type': 'rootAdaptation'},
        # {'code_name': 'SCA_KPP_SA5', 'reward_type': 'escalatingMultiplicativeAdaptation'},
    ]
}

instances = {
    'SCP': {
        'directory': 'MSCP/',
        'names': [
            'mscp42', # 'mscp43', 'mscp44', 'mscp45', 'mscp46', 'mscp47', 'mscp62', 'mscp63', 'mscp64', 'mscp65',
        ]
    },
    'KP': {
        'directory': 'KP/',
        'names': [
            'f1_l-d_kp_10_269', 'f2_l-d_kp_20_878', 'f3_l-d_kp_4_20', 'f4_l-d_kp_4_11', 'f5_l-d_kp_15_375',
            'f6_l-d_kp_10_60', 'knapPI_1_100_1000_1', 'knapPI_2_100_1000_1', 'knapPI_3_100_1000_1',
            'knapPI_1_200_1000_1'
        ]
    }
}

runs = 1
# runs = 11
population = 40
maxIterations = 500
qlAlpha = 0.1
qlGamma = 0.4
policy = 'softMax-rulette-elitist'      # Puede ser 'e-greedy', 'greedy', 'e-soft', 'softMax-rulette', 'softMax-rulette-elitist'
qlAlphaType = 'static'                  # Puede ser 'static', 'iteration', 'visits'
repair = 2                              # 1: Simple; 2: Compleja; 3: RepairGPU

for run in range(runs):
    for problem, problemAlgorithms in algorithms.items():
        for problemAlgorithm in problemAlgorithms:
            for instanceName in instances[problem]['names']:
                executionData = {
                    'algorithm_code_name': problemAlgorithm['code_name'],
                    'parameters': json.dumps({
                        'instance_name': instanceName,
                        'instance_file': instanceName + '.txt',
                        'instance_directory': instances[problem]['directory'],
                        'population': population,
                        'max_iterations': maxIterations,
                        'discretization_scheme': 'V4,Elitist',
                        'repair': repair,
                        'policy': policy,
                        'reward_type': problemAlgorithm['reward_type'],
                        'ql_alpha': qlAlpha,
                        'ql_gamma': qlGamma,
                        'ql_alpha_type': qlAlphaType
                    }),
                    'status': 'PENDING'
                }
                result = database.insertExecution(executionData)
                executionId = result.fetchone()[0]
                print(f'Execution ID: {executionId}')

print('Executions created successfully')
