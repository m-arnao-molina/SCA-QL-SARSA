from database.DatabaseORM import Database, Execution, ExecutionResult, Iteration
from sqlalchemy import asc, desc
import matplotlib.pylab as plt
import numpy as np

db = Database()

problems = ['FLP']
# transferFunctions = ['S1', 'S2', 'S3', 'S4', 'V1', 'V2', 'V3', 'V4', 'Z1', 'Z2', 'Z3', 'Z4']
transferFunctions = ['V4']
binarizationOperators = [
    #('STANDARD', 'Estándar'), ('COMPLEMENT', 'Complemento'), ('STATIC', 'Probabilidad Estática'),
    ('ELITIST', 'Elitista'), # ('ELITIST_ROULETTE', 'Ruleta Elitista')
]
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
"""
for problem in problems:
    for transferFunction in transferFunctions:
        for binarizationOperator, binarizationOperatorAlias in binarizationOperators:
            for instance, instanceAlias in instances[problem]:
                print(f'Generando: {instanceAlias} {transferFunction} {binarizationOperatorAlias}...')
                executionResult = db.session.query(ExecutionResult). \
                    join(ExecutionResult.execution). \
                    filter(Execution.id >= 10). \
                    filter(Execution.parameters['instanceName'].astext == instance). \
                    filter(Execution.parameters['discretizationScheme']['transferFunction'].astext == transferFunction). \
                    filter(Execution.parameters['discretizationScheme']['binarizationOperator'].astext == binarizationOperator). \
                    order_by(desc(ExecutionResult.fitness)). \
                    limit(1). \
                    one()

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
                plt.title(f'Dimensional Hussain\n{instanceAlias} {transferFunction} {binarizationOperatorAlias}', figure=figure)
                figure.savefig(f'outputFiles/{instanceAlias}_{transferFunction}_{binarizationOperator}_{int(executionResult.fitness)}_{executionResult.executionId}.png')
                plt.close(figure)
                plt.close('all')
                # plt.show()


exit(0)
"""

executionResult = db.session.query(ExecutionResult). \
    join(ExecutionResult.execution). \
    filter(Execution.id == 55). \
    limit(1). \
    one()

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