from database.DatabaseORM import Database

db = Database()

problems = ['SCP', 'KP']
transferFunctions = ['S1', 'S2', 'S3', 'S4', 'V1', 'V2', 'V3', 'V4', 'Z1', 'Z2', 'Z3', 'Z4']
binarizationOperators = [
    ('STANDARD', 'STD'), ('COMPLEMENT', 'CPT'), ('STATIC', 'STC'), ('ELITIST', 'ELT'), ('ELITIST_ROULETTE', 'ELR')
]

algorithms = dict()
for problem in problems:
    algorithms[problem] = list()
    for transferFunction in transferFunctions:
        for binarizationOperator, alias in binarizationOperators:
            algorithms[problem].append({
                'codeName': f'SCA_{problem}_{transferFunction}_{alias}',
                'discretizationScheme': {
                    'transferFunction': transferFunction,
                    'binarizationOperator': binarizationOperator
                }
            })

"""
algorithms = {
    'SCP': [
        # {'codeName': 'SCA_SCP_V4_ELT', 'discretizationScheme': {'transferFunction': 'V4', 'binarizationOperator': 'ELITIST'}},
    ],
    'KP': [
        {'codeName': 'SCA_KP_V4_ELT', 'discretizationScheme': {'transferFunction': 'V4', 'binarizationOperator': 'ELITIST'}},
    ]
}
"""

instances = {
    'SCP': {
        'directory': 'MSCP/',
        'names': [
            'mscp42', 'mscp43', 'mscp44', 'mscp45', 'mscp46', # 'mscp47', 'mscp62', 'mscp63', 'mscp64', 'mscp65',
        ]
    },
    'KP': {
        'directory': 'KP/',
        'names': [
            'f1_l-d_kp_10_269', 'f2_l-d_kp_20_878',  # 'f3_l-d_kp_4_20', 'f4_l-d_kp_4_11',
            'f5_l-d_kp_15_375', 'f6_l-d_kp_10_60', 'knapPI_1_100_1000_1', # 'knapPI_2_100_1000_1', 'knapPI_3_100_1000_1',
            'knapPI_1_200_1000_1'
        ]
    }
}

runs = 10  # 31
populationSize = 40
maxIterations = 500
repairType = 2  # 1: Simple; 2: Compleja; 3: RepairGPU
# """
for run in range(runs):
    for problem, problemAlgorithms in algorithms.items():
        for problemAlgorithm in problemAlgorithms:
            for instanceName in instances[problem]['names']:
                execution = db.insertExecution(
                    algorithmCodeName=problemAlgorithm['codeName'],
                    parameters={
                        'instanceName': instanceName,
                        'instanceFile': f'{instanceName}.txt',
                        'instanceDirectory': instances[problem]['directory'],
                        'populationSize': populationSize,
                        'maxIterations': maxIterations,
                        'discretizationScheme': problemAlgorithm['discretizationScheme'],
                        'repairType': repairType
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
                        'repairType': repairType
                    },
                    status='PENDING'
                )
                print(f'Execution ID: {execution.id}')
"""
print('Executions created successfully')
