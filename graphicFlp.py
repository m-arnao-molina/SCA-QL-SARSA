from database.DatabaseORM import Database, Execution, ExecutionResult, Iteration
from sqlalchemy import asc, desc
import matplotlib.pylab as plt
import numpy as np

db = Database()

problems = ['FLP']
# transferFunctions = ['S1', 'S2', 'S3', 'S4', 'V1', 'V2', 'V3', 'V4', 'Z1', 'Z2', 'Z3', 'Z4']
transferFunctions = ['V4']
# binarizationOperators = [
#     #('STANDARD', 'Estándar'), ('COMPLEMENT', 'Complemento'), ('STATIC', 'Probabilidad Estática'),
#     ('ELITIST', 'Elitista'), # ('ELITIST_ROULETTE', 'Ruleta Elitista')
# ]
binarizationOperators = [('COMPLEMENT', 'Complemento'), ('ELITIST', 'Elitista')]
instances = {
    'FLP': [
        ('FLPr_100_40_01', 'FLP1'), ('FLPr_100_40_02', 'FLP2'), ('FLPr_100_40_03', 'FLP3'),
        ('FLPr_100_40_04', 'FLP4'), ('FLPr_100_40_05', 'FLP5'), ('FLPr_100_40_06', 'FLP6'),
        ('FLPr_100_40_07', 'FLP7'), ('FLPr_100_40_08', 'FLP8'), ('FLPr_100_40_09', 'FLP9'),
        ('FLPr_100_40_10', 'FLP10'), ('FLPr_100_100_01', 'FLP11'), ('FLPr_100_100_02', 'FLP12'),
        ('FLPr_100_100_03', 'FLP13'), ('FLPr_100_100_04', 'FLP14'), ('FLPr_100_100_05', 'FLP15'),
        ('FLPr_100_100_06', 'FLP16'), ('FLPr_100_100_07', 'FLP17'), ('FLPr_100_100_08', 'FLP18'),
        ('FLPr_100_100_09', 'FLP19'), ('FLPr_100_100_10', 'FLP20')
    ],
}
diversities = [
    'Dimensional-Hussain' #, 'Pesos de Inercia', 'Leung-Gao-Xu', 'Entrópica', 'Hamming por Frecuencias', 'Momento de Inercia'
]

"""
for problem in problems:
    for transferFunction in transferFunctions:
        for binarizationOperator, binarizationOperatorAlias in binarizationOperators:
            for instance, instanceAlias in instances[problem]:
                print(f'Generando: {instanceAlias} {transferFunction} {binarizationOperatorAlias}...')
                executionResult = db.session.query(ExecutionResult). \
                    join(ExecutionResult.execution). \
                    filter(Execution.id > 50991). \
                    filter(Execution.parameters['instanceName'].astext == instance). \
                    filter(Execution.parameters['discretizationScheme']['transferFunction'].astext == transferFunction). \
                    filter(Execution.parameters['discretizationScheme']['binarizationOperator'].astext == binarizationOperator). \
                    order_by(desc(ExecutionResult.fitness)). \
                    limit(1). \
                    one()

                for idx, diversity in enumerate(diversities):
                    iterations = executionResult.execution.iterations
                    explorations = np.array([])
                    exploitations = np.array([])
                    for iteration in iterations:
                        explorations = np.append(explorations, iteration.parameters['explorationPercentage'][idx])
                        exploitations = np.append(exploitations, iteration.parameters['exploitationPercentage'][idx])

                    # print(explorations)
                    # print(exploitations)

                    figure = plt.figure()
                    ax = figure.add_subplot()
                    ax.plot(explorations)
                    ax.plot(exploitations)
                    ax.legend([
                        f'Exploración {round(np.average(explorations), 2)} %',
                        f'Explotación {round(np.average(exploitations), 2)} %'
                    ])
                    plt.title(f'{diversity}\n{instanceAlias} {transferFunction} {binarizationOperatorAlias}', figure=figure)
                    figure.savefig(f'outputFiles/{instanceAlias}_{transferFunction}_{binarizationOperator}_{diversity}_{int(executionResult.fitness)}_{executionResult.executionId}.png')
                    plt.close(figure)
                    plt.close('all')
                    # plt.show()
"""

machineLearnings = ['QL', 'SARSA']
rewardTypes = [
    ('withPenalty1', 'Con Penalización'), ('withoutPenalty1', 'Sin Penalización'), ('globalBest', 'Global Best'),
    ('rootAdaptation', 'Root Adaption'), ('escalatingMultiplicativeAdaptation', 'Escalating Multiplicative Adaptation')
]

for machineLearning in machineLearnings:
    for rewardType, rewardTypeAlias in rewardTypes:
        for instance, instanceAlias in instances['FLP']:
            print(f'Generando: {instanceAlias} {machineLearning} {rewardTypeAlias}...')
            executionResult = db.session.query(ExecutionResult). \
                join(ExecutionResult.execution). \
                filter(Execution.id > 50991). \
                filter(Execution.parameters['instanceName'].astext == instance). \
                filter(Execution.parameters['rewardType'].astext == rewardType). \
                filter(Execution.algorithmCodeName.like(f'%{machineLearning}%')). \
                order_by(desc(ExecutionResult.fitness)). \
                limit(1). \
                one()

            for idx, diversity in enumerate(diversities):
                iterations = executionResult.execution.iterations
                explorations = np.array([])
                exploitations = np.array([])
                for iteration in iterations:
                    explorations = np.append(explorations, iteration.parameters['explorationPercentage'][idx])
                    exploitations = np.append(exploitations, iteration.parameters['exploitationPercentage'][idx])

                # print(explorations)
                # print(exploitations)

                plt.margins(0.2)
                figure = plt.figure()
                ax = figure.add_subplot()
                ax.plot(explorations)
                ax.plot(exploitations)
                ax.legend([
                    f'Exploración {round(np.average(explorations), 2)} %',
                    f'Explotación {round(np.average(exploitations), 2)} %'
                ])
                plt.title(f'{diversity}\n{instanceAlias} {machineLearning}\n{rewardTypeAlias}', figure=figure)
                figure.tight_layout()
                figure.savefig(
                    f'outputFiles/{instanceAlias}_{machineLearning}_{rewardType}_{diversity}_{int(executionResult.fitness)}_{executionResult.executionId}.png')
                plt.close(figure)
                plt.close('all')

exit(0)


executionResults = db.session.query(ExecutionResult). \
    join(ExecutionResult.execution). \
    filter(Execution.id > 50981). \
    all()

# .in_([50959])
for executionResult in executionResults:
    iterations = executionResult.execution.iterations
    explorations = np.array([])
    exploitations = np.array([])
    for iteration in iterations:
        explorations = np.append(explorations, iteration.parameters['explorationPercentage'][0])
        exploitations = np.append(exploitations, iteration.parameters['exploitationPercentage'][0])

    # print(explorations)
    # print(exploitations)

    figure = plt.figure()
    ax = figure.add_subplot()
    ax.plot(explorations)
    ax.plot(exploitations)
    ax.legend([
        f'Exploración {round(np.average(explorations), 2)} %',
        f'Explotación {round(np.average(exploitations), 2)} %'
    ])
    plt.title('Dimensional Hussain', figure=figure)
    figure.savefig(f'outputFiles/{executionResult.execution.algorithmCodeName}_{int(executionResult.fitness)}_{executionResult.executionId}.png')
    plt.close(figure)
    plt.close('all')
    # plt.show()
