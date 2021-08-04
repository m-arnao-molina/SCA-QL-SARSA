import json
from datetime import datetime
from dotenv import dotenv_values
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy.exc import SQLAlchemyError


class Database:

    def __init__(self):
        config = dotenv_values('.env')
        dbMotor = config['DB_MOTOR']
        dbUsername = config['DB_USERNAME']
        dbPassword = config['DB_PASSWORD']
        dbHost = config['DB_HOST']
        dbPort = config['DB_PORT']
        dbDatabase = config['DB_DATABASE']

        self.engine = create_engine(f'{dbMotor}://{dbUsername}:{dbPassword}@{dbHost}:{dbPort}/{dbDatabase}')
        self.metaData = MetaData()
        self.executionsTable = Table('executions', self.metaData, autoload=True, autoload_with=self.engine)
        self.iterationsTable = Table('iterations', self.metaData, autoload=True, autoload_with=self.engine)
        self.executionResultsTable = Table('execution_results', self.metaData, autoload=True, autoload_with=self.engine)

    def getConnection(self):
        try:
            return self.engine.connect()
        except SQLAlchemyError as error:
            exit(str(error.__dict__['orig']))

    def insertExecution(self, executionData):
        try:
            insert = self.executionsTable.insert().returning(self.executionsTable.c.id)
            return self.getConnection().execute(insert, executionData)
        except SQLAlchemyError as error:
            print(error)
            exit(1)

    def getLastPendingExecution(self):
        try:
            select = self.executionsTable.select().where(self.executionsTable.c.status == 'PENDING')
            lastPendingExecution = self.getConnection().execute(select).fetchone()
            if lastPendingExecution:
                return {
                    'id': lastPendingExecution['id'],
                    'algorithm_code_name': lastPendingExecution['algorithm_code_name'],
                    'parameters': json.loads(lastPendingExecution['parameters'])
                }
            else:
                return None
        except SQLAlchemyError as error:
            print(error)
            exit(1)

    def startExecution(self, executionId, startDatetime=datetime.now()):
        try:
            update = self.executionsTable.update().where(self.executionsTable.c.id == executionId)
            return self.getConnection().execute(update, {
                'start_datetime': startDatetime,
                'status': 'RUNNING'
            })
        except SQLAlchemyError as error:
            print(error)
            return False

    def insertIteration(self, iterationData):
        try:
            insert = self.iterationsTable.insert()
            return self.getConnection().execute(insert, iterationData)
        except SQLAlchemyError as error:
            print(error)
            return False

    def insertExecutionResult(self, executionResultData):
        try:
            insert = self.executionResultsTable.insert()
            return self.getConnection().execute(insert, executionResultData)
        except SQLAlchemyError as error:
            print(error)
            return False

    def endExecution(self, executionId, endDatetime=datetime.now()):
        try:
            update = self.executionsTable.update().where(self.executionsTable.c.id == executionId)
            return self.getConnection().execute(update, {
                'end_datetime': endDatetime,
                'status': 'FINISHED'
            })
        except SQLAlchemyError as error:
            print(error)
            return False
