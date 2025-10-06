from instance import Instance
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys


class ExactMethod:
    instance: Instance
    model: gp.Model
    n: int
    X: np.ndarray
    D: np.ndarray

    def __init__(self, instance: Instance):
        self.instance = instance
        self.model = gp.Model("ttp_exact")

    def _build_patterns(self):
        # TODO: Generalize for any n
        # Some heuristic? All patterns?
        self.X = np.array([
            [0, 2, 3, 1, 0],
            [1, 3, 0, 2, 1],
            [2, 1, 3, 0, 2],
            [3, 0, 1, 2, 3],
        ])

    def _build_distances(self):
        self.D = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.D[i, j] = self.instance.distances.get((i, i))
                else:
                    self.D[i, j] = 0

    def _build_variables(self):
        self.n = self.instance.n
        self.teams = list(range(self.n))
        self._build_patterns()
        self._build_distances()

    def _build_decision_variables(self):
        # Y
        self.Y = self.model.addVars(self.teams, range(len(self.teams)), vtype=GRB.BINARY, name="y")
        self.H = self.model.addVars(self.teams, range(len(self.teams)), vtype=GRB.INTEGER, name="h",
                                    lb=self.instance.lower_bound, ub=self.instance.upper_bound)

    def _build_objective(self):
        self.obj = gp.quicksum(
            (1-self.Y[t, j]) * self.D[self.X[t, j-1], self.X[t,j]] +
            self.Y[t, j] * self.D[self.X[t, j-1], self.X[t, j]]
            for t in self.teams for j in range(1, self.n)
        )
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def _build_constraints(self):
        # H and Y relationship
        big_M = 1e6
        self.model.addConstrs(
            (self.H[t, j] <= self.Y[t, j] * big_M for t in self.teams for j in range(len(self.teams))),
            name="H_Y_rel_1")
        self.model.addConstrs(
            (self.H[t, j] - self.Y[t, j] >= 0 for t in self.teams for j in range(len(self.teams))),
            name="H_Y_rel_2")
        # Making sure there are no more than upper_bound consecutive away games
        self.model.addConstrs(
            (gp.quicksum(self.Y[t, j + k] for j in range(1, self.n-self.instance.upper_bound + 1)
                         for k in range(self.instance.upper_bound)) >= 1 for t in self.teams),
            name="max_away_games"
        )
        # Right number of home games
        self.model.addConstrs(
            ((gp.quicksum(self.H[t, j] for j in range(len(self.teams))) == len(self.teams) - 1)
             for t in self.teams),
            name="no_home_games"
        )

    def build_model(self):
        self._build_variables()
        self._build_decision_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def solve(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print(f"Optimal objective: {self.model.objVal}")
            for v in self.model.getVars():
                if v.X > 0.1:  # Only print variables that are set to 1
                    print(f"{v.VarName}: {v.X}")
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible")
        elif self.model.status == GRB.UNBOUNDED:
            print("Model is unbounded")
        else:
            print(f"Optimization ended with status {self.model.status}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python exact_method.py <instance_file>")
        sys.exit(1)

    instance_file = sys.argv[1]
    instance = Instance.from_file(instance_file)
    instance.print_summary()
    method = ExactMethod(instance)
    method.build_model()
    method.solve()

