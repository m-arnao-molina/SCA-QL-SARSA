from datetime import datetime
from dotenv import dotenv_values
from sqlalchemy import Column, create_engine, DateTime, Float, ForeignKey, Integer, LargeBinary, select, String
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.exc import SQLAlchemyError, NoResultFound
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from threading import Lock


class DatabaseMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Database(metaclass=DatabaseMeta):
    def __init__(self):
        config = dotenv_values('.env')
        dbMotor = config['DB_MOTOR']
        dbUsername = config['DB_USERNAME']
        dbPassword = config['DB_PASSWORD']
        dbHost = config['DB_HOST']
        dbPort = config['DB_PORT']
        dbDatabase = config['DB_DATABASE']

        self.engine = create_engine(
            f'{dbMotor}://{dbUsername}:{dbPassword}@{dbHost}:{dbPort}/{dbDatabase}', echo=False, future=True  # echo=True
        )
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.Base = declarative_base()
        self.Base.metadata.create_all(self.engine)

    def insertExecution(self, algorithmCodeName, parameters, status, startDatetime=None, endDatetime=None):
        try:
            execution = Execution(algorithmCodeName, parameters, status, startDatetime, endDatetime)
            self.session.add(execution)
            self.session.commit()
            return execution
        except SQLAlchemyError as error:
            print(error)
            exit(1)

    def getLastPendingExecution(self):
        try:
            try:
                return self.session.execute(select(Execution).filter_by(status='PENDING').limit(1)).scalar_one()
            except NoResultFound:
                return None
        except SQLAlchemyError as error:
            print(error)
            exit(1)

    def startExecution(self, executionId, startDatetime=datetime.now()):
        try:
            execution = self.session.execute(select(Execution).filter_by(id=executionId)).scalar_one()
            execution.startDatetime = startDatetime
            execution.status = 'RUNNING'
            self.session.commit()
            return execution
        except SQLAlchemyError as error:
            print(error)
            return False

    def insertIterations(self, iterationsData):
        try:
            iterations = list()
            for iterationData in iterationsData:
                iteration = Iteration(
                    iterationNumber=iterationData.get('iterationNumber'),
                    bestFitness=iterationData.get('bestFitness'),
                    avgFitness=iterationData.get('avgFitness'),
                    fitnessBestIteration=iterationData.get('fitnessBestIteration'),
                    parameters=iterationData.get('parameters'),
                    startDatetime=iterationData.get('startDatetime'),
                    endDatetime=iterationData.get('endDatetime'),
                    internalData=iterationData.get('internalData')
                )
                iteration.executionId = iterationData.get('executionId')
                iterations.append(iteration)

            self.session.bulk_save_objects(iterations)
            self.session.commit()
            return iterations
        except SQLAlchemyError as error:
            print(error)
            return False

    def insertIteration(
        self, iterationNumber, bestFitness, avgFitness, fitnessBestIteration, parameters, executionId,
        startDatetime=None, endDatetime=None, internalData=None
    ):
        try:
            iteration = Iteration(
                iterationNumber, bestFitness, avgFitness, fitnessBestIteration, parameters, startDatetime, endDatetime,
                internalData
            )
            iteration.executionId = executionId
            self.session.add(iteration)
            self.session.commit()
            return iteration
        except SQLAlchemyError as error:
            print(error)
            return False

    def insertExecutionResult(self, fitness, bestSolution, executionId, startDatetime=None, endDatetime=None):
        try:
            executionResult = ExecutionResult(fitness, bestSolution, startDatetime, endDatetime)
            executionResult.executionId = executionId
            self.session.add(executionResult)
            self.session.commit()
            return executionResult
        except SQLAlchemyError as error:
            print(error)
            return False

    def endExecution(self, executionId, endDatetime=datetime.now()):
        try:
            execution = self.session.execute(select(Execution).filter_by(id=executionId)).scalar_one()
            execution.endDatetime = endDatetime
            execution.status = 'FINISHED'
            self.session.commit()
            return execution
        except SQLAlchemyError as error:
            print(error)
            return False


class Execution(Database().Base):
    __tablename__ = 'executions'

    id = Column(Integer, primary_key=True)
    algorithmCodeName = Column(String(100))
    parameters = Column(JSON)
    status = Column(String(20))
    startDatetime = Column(DateTime)
    endDatetime = Column(DateTime)

    iterations = relationship('Iteration', back_populates='execution', order_by='Iteration.iterationNumber.asc()')
    executionResults = relationship('ExecutionResult', back_populates='execution')

    def __init__(self, algorithmCodeName, parameters, status, startDatetime=None, endDatetime=None):
        self.algorithmCodeName = algorithmCodeName
        self.parameters = parameters
        self.status = status
        self.startDatetime = startDatetime
        self.endDatetime = endDatetime

    def __repr__(self):
        return f'{self.__class__.__name__}({self.id}, {self.algorithmCodeName}, {self.status}, {self.startDatetime}, ' \
               f'{self.endDatetime})'


class Iteration(Database().Base):
    __tablename__ = 'iterations'

    id = Column(Integer, primary_key=True)
    iterationNumber = Column(Integer)
    bestFitness = Column(Float)
    avgFitness = Column(Float)
    fitnessBestIteration = Column(Float)
    parameters = Column(JSON)
    startDatetime = Column(DateTime)
    endDatetime = Column(DateTime)
    internalData = Column(LargeBinary)

    executionId = Column(Integer, ForeignKey('executions.id'), nullable=False)
    execution = relationship('Execution', back_populates='iterations')

    def __init__(
        self, iterationNumber, bestFitness, avgFitness, fitnessBestIteration, parameters, startDatetime=None,
        endDatetime=None, internalData=None
    ):
        self.iterationNumber = iterationNumber
        self.bestFitness = bestFitness
        self.avgFitness = avgFitness
        self.fitnessBestIteration = fitnessBestIteration
        self.parameters = parameters
        self.startDatetime = startDatetime
        self.endDatetime = endDatetime
        self.internalData = internalData

    def __repr__(self):
        return f'{self.__class__.__name__}({self.id}, {self.iterationNumber}, {self.startDatetime}, ' \
               f'{self.endDatetime})'


class ExecutionResult(Database().Base):
    __tablename__ = 'execution_results'

    id = Column(Integer, primary_key=True)
    fitness = Column(Float)
    bestSolution = Column(JSON)
    startDatetime = Column(DateTime)
    endDatetime = Column(DateTime)

    executionId = Column(Integer, ForeignKey('executions.id'), nullable=False)
    execution = relationship('Execution', back_populates='executionResults')

    def __init__(self, fitness, bestSolution, startDatetime=None, endDatetime=None):
        self.fitness = fitness
        self.bestSolution = bestSolution
        self.startDatetime = startDatetime
        self.endDatetime = endDatetime

    def __repr__(self):
        return f'{self.__class__.__name__}({self.id}, {self.fitness}, {self.bestSolution}, {self.startDatetime}, ' \
               f'{self.endDatetime})'
