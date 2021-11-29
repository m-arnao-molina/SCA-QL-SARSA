from database.DatabaseORM import Database, Execution, ExecutionResult, Iteration
from sqlalchemy import asc, desc
import matplotlib.pylab as plt
import numpy as np

db = Database()

# problems = ['SCP', 'KP']
problems = ['KP']
# transferFunctions = ['S1', 'S2', 'S3', 'S4', 'V1', 'V2', 'V3', 'V4', 'Z1', 'Z2', 'Z3', 'Z4']
transferFunctions = ['Z4']
binarizationOperators = [
    ('STANDARD', 'Estándar'), ('COMPLEMENT', 'Complemento'), ('STATIC', 'Probabilidad Estática'),
    ('ELITIST', 'Elitista'), ('ELITIST_ROULETTE', 'Ruleta Elitista')
]
instances = {
    'SCP': [
        ('mscp42', 'SCP1'), ('mscp43', 'SCP2'), ('mscp44', 'SCP3'), ('mscp45', 'SCP4'), ('mscp46', 'SCP5')
    ],
    'KP': [
        ('f1_l-d_kp_10_269', 'KP1'), ('f2_l-d_kp_20_878', 'KP2'), ('f5_l-d_kp_15_375', 'KP3'),
        ('f6_l-d_kp_10_60', 'KP4'), ('knapPI_1_100_1000_1', 'KP5'), ('knapPI_1_200_1000_1', 'KP6')
    ]
}

for problem in problems:
    for transferFunction in transferFunctions:
        for binarizationOperator, binarizationOperatorAlias in binarizationOperators:
            for instance, instanceAlias in instances[problem]:
                print(f'Generando: {instanceAlias} {transferFunction} {binarizationOperatorAlias}...')
                executionResult = db.session.query(ExecutionResult). \
                    join(ExecutionResult.execution). \
                    filter(Execution.id >= 88). \
                    filter(Execution.parameters['instanceName'].astext == instance). \
                    filter(Execution.parameters['discretizationScheme']['transferFunction'].astext == transferFunction). \
                    filter(Execution.parameters['discretizationScheme']['binarizationOperator'].astext == binarizationOperator)

                if problem == 'SCP':
                    executionResult = executionResult.order_by(asc(ExecutionResult.fitness))
                elif problem == 'KP':
                    executionResult = executionResult.order_by(desc(ExecutionResult.fitness))

                executionResult = executionResult.limit(1).one()
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


# print(algorithms)

exit(0)



executionResult = db.session.query(ExecutionResult). \
    join(ExecutionResult.execution). \
    filter(Execution.parameters['instanceName'].astext == 'knapPI_1_100_1000_1'). \
    filter(Execution.parameters['discretizationScheme']['transferFunction'].astext == 'Z4'). \
    filter(Execution.parameters['discretizationScheme']['binarizationOperator'].astext == 'ELITIST'). \
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
plt.title('Dimensional Hussain', figure=figure)
plt.show()
# figure.savefig('chart.png')

exit(0)
# print(query.statement.compile(compile_kwargs={"literal_binds": True}))
# print(query.all())
