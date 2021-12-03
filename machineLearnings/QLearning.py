import math
import numpy as np


class QLearning:
    # Initialize alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, qlGamma, qlAlphaType, rewardType, maxIterations, actions, stateQ, epsilon=0.05, qlAlpha=0.1):
        self.qlGamma = qlGamma
        self.qlAlphaType = qlAlphaType
        self.rewardType = rewardType
        self.maxIterations = maxIterations
        self.epsilon = epsilon
        self.qlAlpha = qlAlpha

        self.bestMetric = -999999999  # Esto en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
        self.w = 10
        self.qValues = np.zeros(shape=(stateQ, len(actions)))  # state, actions
        self.visits = np.zeros(shape=(stateQ, len(actions)))  # state, actions

    def rWithPenalty1(self, metric):
        if self.bestMetric < metric:  # Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
            self.bestMetric = metric
            return 1
        return -1

    def rWithoutPenalty1(self, metric):
        if self.bestMetric < metric:  # Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
            self.bestMetric = metric
            return 1
        return 0

    def rGlobalBest(self, metric):
        if self.bestMetric < metric:  # Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
            self.bestMetric = metric
            return self.w / self.bestMetric
        return 0

    def rRootAdaptation(self, metric):
        if self.bestMetric < metric:  # Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
            self.bestMetric = metric
            return math.sqrt(metric)
        return 0

    def rEscalatingMultiplicativeAdaptation(self, metric):
        if self.bestMetric < metric:  # Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
            self.bestMetric = metric
            return self.w * self.bestMetric
        return 0

    def getReward(self, metric):
        rewardFunctionsSwitcher = {
            'withPenalty1': self.rWithPenalty1,
            'withoutPenalty1': self.rWithoutPenalty1,
            'globalBest': self.rGlobalBest,
            'rootAdaptation': self.rRootAdaptation,
            'escalatingMultiplicativeAdaptation': self.rEscalatingMultiplicativeAdaptation
        }
        return rewardFunctionsSwitcher.get(self.rewardType, self.rWithPenalty1)(metric)

    def getAction(self, state, policy):
        # e-greedy
        if policy == "e-greedy":
            probabilidad = np.random.uniform(low=0.0, high=1.0) #numero aleatorio [0,1]
            if probabilidad <= self.epsilon: #seleccion aleatorio
                return np.random.randint(low=0, high=self.qValues.shape[0]) #seleccion aleatoria de una accion
            else: #selecion de Q_Value mayor        
                maximo = np.amax(self.qValues, axis=1) # retorna el elemento mayor por fila
                indices = np.where(self.qValues[state, :] == maximo[state])[0]  #retorna los indices donde se ubica el maximo en la fila estado
                return np.random.choice(indices) # funciona tanto cuando hay varios iguales como cuando hay solo uno 
        
        # greedy
        elif policy == "greedy":
            return np.argmax(self.qValues[state])

        # e-soft 
        elif policy == "e-soft":
            probabilidad = np.random.uniform(low=0.0, high=1.0) #numero aleatorio [0,1]
            if probabilidad > self.epsilon: #seleccion aleatorio
                return np.random.randint(low=0, high=self.qValues.shape[0]) #seleccion aleatoria de una accion
            else: #selecion de Q_Value mayor        
                maximo = np.amax(self.qValues, axis=1) # retorna el elemento mayor por fila
                indices = np.where(self.qValues[state, :] == maximo[state])[0]  #retorna los indices donde se ubica el maximo en la fila estado
                return np.random.choice(indices) # funciona tanto cuando hay varios iguales como cuando hay solo uno 

        # softMax seleccion ruleta
        elif policy == "softMax-rulette":
            #*** Falta generar una normalización de las probabilidades que sumen 1, para realizar el choice
            Qtable_normalizada = np.nan_to_num(self.qValues[state] / np.linalg.norm(self.qValues[state])) # normalizacion de valores
            #La suma de las prob = 1
            seleccionado = np.random.choice(self.qValues[state], p=Qtable_normalizada)
            indices = np.where(self.qValues[state, :] == seleccionado)[0]
            return np.random.choice(indices)
    
        # softmax seleccion ruleta elitista (25% mejores acciones)
        elif policy == "softMax-rulette-elitist":
            sort = np.argsort(self.qValues[state]) # argumentos ordenados
            cant_mejores = int(sort.shape[0]*0.25) # obtenemos el 25% de los mejores argumentos
            rulette_elitist = sort[0:cant_mejores] # tiene el 25% de los mejores argumentos
            return np.random.choice(rulette_elitist)

    def updateVisits(self, action, state):  # ACTUALIZACION DE LAS VISITAS
        self.visits[state, action] = self.visits[state, action] + 1

    def getAlpha(self, state, action, iterationNumber):
        if self.qlAlphaType == 'static':
            return self.qlAlpha
        elif self.qlAlphaType == 'iteration':
            return 1 - (0.9 * (iterationNumber / self.maxIterations))
        elif self.qlAlphaType == 'visits':
            return 1 / (1 + self.visits[state, action])

    def updateQtable(self, metric, action, currentState, oldState, iterationNumber):
        # Revisar
        reward = self.getReward(metric)
        alpha = self.getAlpha(oldState, action, iterationNumber)
        qNew = ((1 - alpha) * self.qValues[oldState][action]) + alpha * \
            (reward + (self.qlGamma * max(self.qValues[currentState])))

        self.updateVisits(action, oldState)  # Actuzación de visitas
        self.qValues[currentState][action] = qNew

    def getQTable(self):
        return self.qValues
