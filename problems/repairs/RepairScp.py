import random
import numpy as np
from problems.repairs import solution as sl
from problems.repairs import heuristic as he


class RepairScp:
    def __init__(self, repairType, constraints, costs):
        self.repairType = repairType
        self.constraints = constraints
        self.costs = costs

        self.rHeuristic = he.getRowHeuristics(self.constraints)
        self.dictCol = he.getColumnRow(self.constraints)
        self.dictcHeuristics = {}
        self.cHeuristic = []
        self.lSolution = []
        self.dict = he.getRowColumn(self.constraints)

    def meets(self, solution):
        for i in range(self.constraints.shape[0]):
            if np.sum(self.constraints[i] * solution) < 1:
                return False
        return True

    def repair(self, solution):
        if self.repairType == 1:
            return self.simpleRepair(solution)
        elif self.repairType == 2:
            return self.complexRepair(solution)

    def simpleRepair(self, solution):
        repairsQuantity = 0
        indices = list(range(self.constraints.shape[0]))
        random.shuffle(indices)
        for i in indices:
            if np.sum(self.constraints[i] * solution) < 1:
                constraintIdx = np.argwhere((self.constraints[i]) > 0)
                lowerWeightIdx = constraintIdx[np.argmin(self.costs[constraintIdx])]
                solution[lowerWeightIdx[0]] = 1
                repairsQuantity += 1
        return solution, repairsQuantity

    def complexRepair(self, solution):
        lSolution = [i for i in range(len(solution)) if solution[i] == 1]
        lSolution, repairsQuantity = sl.generaSolucion(
            lSolution, self.constraints, self.costs, self.rHeuristic, self.dictcHeuristics, self.dict, self.cHeuristic,
            self.dictCol
        )
        sol = np.zeros(self.constraints.shape[1], dtype=np.float64)
        sol[lSolution] = 1
        return sol.tolist(), repairsQuantity
