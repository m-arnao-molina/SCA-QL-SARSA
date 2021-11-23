from database.DatabaseORM import Database

db = Database()

algorithms = {
    'SCP': [
        # {'codeName': 'SCA_SCP_BCL1', 'rewardType': None},
        # {'codeName': 'SCA_SCP_MIR2', 'rewardType': None},
        # {'codeName': 'SCA_SCP_QL1', 'rewardType': 'withPenalty1'},
        # {'codeName': 'SCA_SCP_QL2', 'rewardType': 'withoutPenalty1'},
        # {'codeName': 'SCA_SCP_QL3', 'rewardType': 'globalBest'},
        # {'codeName': 'SCA_SCP_QL4', 'rewardType': 'rootAdaptation'},
        # {'codeName': 'SCA_SCP_QL5', 'rewardType': 'escalatingMultiplicativeAdaptation'}
    ],
    'KP': [
        {'codeName': 'SCA_KP', 'rewardType': None},
        # {'codeName': 'SCA_KPP_QL1', 'rewardType': 'withPenalty1'},
        # {'codeName': 'SCA_KPP_QL2', 'rewardType': 'withoutPenalty1'},
        # {'codeName': 'SCA_KPP_QL3', 'rewardType': 'globalBest'},
        # {'codeName': 'SCA_KPP_QL4', 'rewardType': 'rootAdaptation'},
        # {'codeName': 'SCA_KPP_QL5', 'rewardType': 'escalatingMultiplicativeAdaptation'},
        # {'codeName': 'SCA_KPP_SA1', 'rewardType': 'withPenalty1'},
        # {'codeName': 'SCA_KPP_SA2', 'rewardType': 'withoutPenalty1'},
        # {'codeName': 'SCA_KPP_SA3', 'rewardType': 'globalBest'},
        # {'codeName': 'SCA_KPP_SA4', 'rewardType': 'rootAdaptation'},
        # {'codeName': 'SCA_KPP_SA5', 'rewardType': 'escalatingMultiplicativeAdaptation'},
    ],
    'FLP': [
        # {'codeName': 'SCA_FLP', 'rewardType': None},
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
            'f1_l-d_kp_10_269', # 'f2_l-d_kp_20_878', 'f3_l-d_kp_4_20', 'f4_l-d_kp_4_11', 'f5_l-d_kp_15_375',
            # 'f6_l-d_kp_10_60', 'knapPI_1_100_1000_1', 'knapPI_2_100_1000_1', 'knapPI_3_100_1000_1',
            # 'knapPI_1_200_1000_1'
        ]
    },
    'FLP': {
        'directory': 'FLP/',
        'names': [
            'FLPr_100_40_01', # 'FLPr_100_40_02', 'FLPr_100_40_03', 'FLPr_100_40_04', 'FLPr_100_40_05', 'FLPr_100_40_06',
            # 'FLPr_100_40_07', 'FLPr_100_40_08', 'FLPr_100_40_09', 'FLPr_100_40_10', 'FLPr_100_100_01',
            # 'FLPr_100_100_02', 'FLPr_100_100_03', 'FLPr_100_100_04', 'FLPr_100_100_05', 'FLPr_100_100_06',
            # 'FLPr_100_100_07', 'FLPr_100_100_08', 'FLPr_100_100_09', 'FLPr_100_100_10'
        ]
    }
}

runs = 1
# runs = 11
populationSize = 40
maxIterations = 500
qlAlpha = 0.1
qlGamma = 0.4
policy = 'softMax-rulette-elitist'      # Puede ser 'e-greedy', 'greedy', 'e-soft', 'softMax-rulette', 'softMax-rulette-elitist'
qlAlphaType = 'static'                  # Puede ser 'static', 'iteration', 'visits'
repairType = 2                          # 1: Simple; 2: Compleja; 3: RepairGPU
# """
for run in range(runs):
    for problem, problemAlgorithms in algorithms.items():
        for problemAlgorithm in problemAlgorithms:
            for instanceName in instances[problem]['names']:
                execution = db.insertExecution(
                    algorithmCodeName=problemAlgorithm['codeName'],
                    parameters={
                        'instanceName': instanceName,
                        'instanceFile': instanceName + '.txt',
                        'instanceDirectory': instances[problem]['directory'],
                        'populationSize': populationSize,
                        'maxIterations': maxIterations,
                        'discretizationScheme': {
                            'transferFunction': 'V4',
                            'binarizationOperator': 'ELITIST'
                        },
                        'repairType': repairType,
                        'policy': policy,
                        'rewardType': problemAlgorithm.get('rewardType'),
                        'qlAlpha': qlAlpha,
                        'qlGamma': qlGamma,
                        'qlAlphaType': qlAlphaType
                    },
                    status='PENDING'
                )
                print(f'Execution ID: {execution.id}')

"""
for run in range(runs):
    for problem, problemAlgorithms in algorithms.items():
        for problemAlgorithm in problemAlgorithms:
            for instanceName in instances[problem]['names']:
                execution = db.insertExecution(
                    algorithmCodeName=problemAlgorithm['codeName'],
                    parameters={
                        'instanceName': instanceName,
                        'instanceFile': instanceName + '.txt',
                        'instanceDirectory': instances[problem]['directory'],
                        'populationSize': populationSize,
                        'maxIterations': maxIterations,
                        'discretizationScheme': 'V4,Elitist',
                        'repairType': repairType,
                        'policy': policy,
                        'rewardType': problemAlgorithm['rewardType'],
                        'qlAlpha': qlAlpha,
                        'qlGamma': qlGamma,
                        'qlAlphaType': qlAlphaType
                    },
                    status='PENDING'
                )
                print(f'Execution ID: {execution.id}')
"""
print('Executions created successfully')
