from instance import Instance
import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np
import sys
import math


class ExactMethod:
    instance: Instance
    model: gp.Model
    n: int
    A: np.ndarray
    D: np.ndarray

    def __init__(self, instance: Instance):
        self.instance = instance
        self.J = range(len(self.instance.teams))
        self.I = range(np.math.factorial(len(self.instance.teams) - 1))
        self.T = range(len(self.instance.teams))
        self.model = gp.Model("ttp_exact")

    def _build_patterns(self):
        patterns = np.array([
            list(perm) + [perm[0]]
            for perm in itertools.permutations(self.T)
        ])
        # Group patterns by their first element for easy access: A[t][i] gives the i-th pattern for team t
        self.A = {t: [] for t in self.T}
        for row in self.A:
            self.A_grouped[row[0]].append(row)

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
        self.S = self.model.addVars(self.teams, self.I, vtype=GRB.BINARY, name="s")
        self.Alpha = self.model.addVars(self.teams, self.I, self.J, vtype=GRB.BINARY, name="alpha")
        self.Beta = self.model.addVars(self.teams, self.I, self.J, vtype=GRB.BINARY, name="beta")

    def _build_objective(self):
        self.obj = gp.quicksum(
            self.Alpha[t, i, j] * self.D[self.A[i, j - 1], self.A[i, j]] +
            self.Beta[t, i, j] * (self.D[self.A[i, j - 1], t] + self.D[t, self.A[i, j]])
            for t in self.teams for j in range(1, self.n) for i in range(1, len(self.A))
        )
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def _build_constraints(self):
        # H and Y relationship
        big_M = 1e6
        # === Constraints from equations (alpha_beta_valid) to (no_home_games) ===
        # (0) \sum_{i=1}^m s_{\vec a_{t,i}} = 1
        self.model.addConstrs(
            (gp.quicksum(self.s[i] for i in self.I) == 1
             for t in self.T),
            name="one_pattern_per_team"
        )

        # (1) α_{t,i,j} ≤ 1 - y_{t,j}
        self.model.addConstrs(
            (self.alpha[t, i, j] <= 1 - self.y[t, j]
             for t in self.T for i in self.I for j in self.J),
            name="alpha_le_1_minus_y"
        )

        # (2) α_{t,i,j} ≤ s_i
        self.model.addConstrs(
            (self.alpha[t, i, j] <= self.s[i]
             for t in self.T for i in self.I for j in self.J),
            name="alpha_le_s"
        )

        # (3) α_{t,i,j} ≥ s_i - y_{i,j}
        self.model.addConstrs(
            (self.alpha[t, i, j] >= self.s[i] - self.y[i, j]
             for t in self.T for i in self.I for j in self.J),
            name="alpha_ge_s_minus_y"
        )

        # (4) β_{t,i,j} ≤ y_{i,j}
        self.model.addConstrs(
            (self.beta[t, i, j] <= self.y[i, j]
             for t in self.T for i in self.I for j in self.J),
            name="beta_le_y"
        )

        # (5) β_{t,i,j} ≤ s_i
        self.model.addConstrs(
            (self.beta[t, i, j] <= self.s[i]
             for t in self.T for i in self.I for j in self.J),
            name="beta_le_s"
        )

        # (6) β_{t,i,j} ≥ s_i + y_{i,j} - 1
        self.model.addConstrs(
            (self.beta[t, i, j] >= self.s[i] + self.y[i, j] - 1
             for t in self.T for i in self.I for j in self.J),
            name="beta_ge_s_plus_y_minus_1"
        )

        # (7) h_{t, j} ≤ y_{t, j} * M
        self.model.addConstrs(
            (self.h[t, j] <= self.y[t, j] * M
             for t in self.T for j in self.J),
            name="home_le_yM"
        )

        # (8) h_{t, j} - y_{t, j} ≥ 0
        self.model.addConstrs(
            (self.h[t, j] - self.y[t, j] >= 0
             for t in self.T for j in self.J),
            name="h_minus_y_ge_0"
        )

        # (9) ∑_{j=1}^{n-u} ∑_{k=0}^{u-1} y_{t, j+k} ≥ 1
        self.model.addConstrs(
            (gp.quicksum(self.y[t, j + k] for j in range(1, n - u + 1) for k in range(u)) >= 1
             for t in self.T),
            name="max_away"
        )

        # (10) ∑_{j=0}^{n-1} h_{t,j} = n - 1
        self.model.addConstrs(
            (gp.quicksum(self.h[t, j] for j in range(self.n)) == self.n - 1
             for t in self.T),
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
                if v.A > 0.1:  # Only print variables that are set to 1
                    print(f"{v.VarName}: {v.A}")
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
    method._build_patterns()
    # print(method.A)
    method.build_model()
    method.solve()

