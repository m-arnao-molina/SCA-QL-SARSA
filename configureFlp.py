from database.DatabaseORM import Database

db = Database()
db.sync()

# transferFunctions = ['S1', 'S2', 'S3', 'S4', 'V1', 'V2', 'V3', 'V4', 'Z1', 'Z2', 'Z3', 'Z4']
transferFunctions = ['V4']
binarizationOperators = [('ELITIST', 'ELT')]
# binarizationOperators = [
#     ('STANDARD', 'STD'), ('COMPLEMENT', 'CPT'), ('STATIC', 'STC'), ('ELITIST', 'ELT'), ('ELITIST_ROULETTE', 'ELR')
# ]

algorithms = dict()
algorithms['FLP'] = list()
for transferFunction in transferFunctions:
    for binarizationOperator, alias in binarizationOperators:
        algorithms['FLP'].append({
            'codeName': f'SCA_FLP_{transferFunction}_{alias}',
            'discretizationScheme': {
                'transferFunction': transferFunction,
                'binarizationOperator': binarizationOperator
            }
        })

instances = {
    'FLP': {
        'directory': 'FLP/',
        'names': [
            ('FLPr_100_40_01', 'FLP1'), ('FLPr_100_40_02', 'FLP2'), ('FLPr_100_40_03', 'FLP3'),
            ('FLPr_100_40_04', 'FLP4'), ('FLPr_100_40_05', 'FLP5'), ('FLPr_100_40_06', 'FLP6'),
            ('FLPr_100_40_07', 'FLP7'), ('FLPr_100_40_08', 'FLP8'), ('FLPr_100_40_09', 'FLP9'),
            ('FLPr_100_40_10', 'FLP10'), ('FLPr_100_100_01', 'FLP11'), ('FLPr_100_100_02', 'FLP12'),
            ('FLPr_100_100_03', 'FLP13'), ('FLPr_100_100_04', 'FLP14'), ('FLPr_100_100_05', 'FLP15'),
            ('FLPr_100_100_06', 'FLP16'), ('FLPr_100_100_07', 'FLP17'), ('FLPr_100_100_08', 'FLP18'),
            ('FLPr_100_100_09', 'FLP19'), ('FLPr_100_100_10', 'FLP20')
        ]
    },
    'KP': {
        'directory': 'KP/',
        'names': [('f1_l-d_kp_10_269', 'KP1')]
    },
    'SCP': {
        'directory': 'MSCP/',
        'names': [('mscp42', 'SCP1')]
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

for run in range(runs):
    for problem, problemAlgorithms in algorithms.items():
        for problemAlgorithm in problemAlgorithms:
            for instanceName, alias in instances[problem]['names']:
                execution = db.insertExecution(
                    algorithmCodeName=f'{problemAlgorithm["codeName"]}_{alias}',
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

print('Executions created successfully')
