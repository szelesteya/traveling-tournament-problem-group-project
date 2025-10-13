from instance import Instance
import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np
import sys


def setup_gurobi_options(lic_content: str) -> dict[str, str]:
    lines = lic_content.strip().split("\n")

    wls_access_prefix = "WLSACCESSID="
    wls_access_id = [
        line[len(wls_access_prefix) :]
        for line in lines
        if line.startswith(wls_access_prefix)
    ][0]

    wls_secret_prefix = "WLSSECRET="
    wls_secret = [
        line[len(wls_access_prefix) :]
        for line in lines
        if line.startswith(wls_secret_prefix)
    ][0]

    license_id_prefix = "LICENSEID="
    license_id = [
        int(line[len(license_id_prefix) :])
        for line in lines
        if line.startswith(license_id_prefix)
    ][0]

    return {
        "WLSAccessID": wls_access_id,
        "WLSSecret": wls_secret,
        "LicenseID": license_id,
    }


# noinspection PyTypeChecker
class ExactMethod:
    instance: Instance
    model: gp.Model
    n: int
    A: np.ndarray
    D: np.ndarray

    def __init__(self, ttp_instance: Instance, gurobi_lic: str = None):
        if gurobi_lic is not None:
            self.gurobi_options = setup_gurobi_options(gurobi_lic)
        self.instance = ttp_instance
        self.J = range(len(self.instance.teams))
        self.T = range(len(self.instance.teams))
        self.lb = self.instance.lower_bound
        self.ub = self.instance.upper_bound

    @property
    def I(self):
        if not hasattr(self, "A"):
            raise AttributeError("Patterns have not been built yet.")
        return range(len(self.A[0]))  # Number of patterns per team

    def _build_patterns(self, no_patterns_per_team: int = 6):
        all_patterns = np.array(
            [list(perm) + [perm[0]] for perm in itertools.permutations(self.T)]
        )

        pattern_len = len(self.T) + 1
        team_patterns = [all_patterns[all_patterns[:, 0] == t] for t in self.T]

        if no_patterns_per_team is None:
            max_patterns = max(len(p) for p in team_patterns)
            sampled_per_team = team_patterns
        else:
            max_patterns = no_patterns_per_team
            rng = np.random.default_rng()
            sampled_per_team = []
            for p in team_patterns:
                if len(p) == 0:
                    sampled_per_team.append(np.empty((0, pattern_len), dtype=int))
                    continue
                if len(p) >= no_patterns_per_team:
                    idx = rng.choice(len(p), size=no_patterns_per_team, replace=False)
                else:
                    idx = rng.choice(len(p), size=no_patterns_per_team, replace=True)
                sampled_per_team.append(p[idx])

        self.A = np.full((len(self.T), max_patterns, pattern_len), -1, dtype=int)
        for t, sp in enumerate(sampled_per_team):
            for i, row in enumerate(sp):
                self.A[t, i, :] = row

    def _build_distances(self):
        self.D = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.D[i, j] = self.instance.distances.get((i, j))

    def _build_variables(self):
        self.n = self.instance.n
        self.teams = list(range(self.n))
        self._build_patterns()
        self._build_distances()

    def _build_decision_variables(self):
        # Y
        self.Y = self.model.addVars(self.T, self.J, vtype=GRB.BINARY, name="y")
        self.H = self.model.addVars(
            self.T,
            self.J,
            vtype=GRB.INTEGER,
            name="h",
            lb=0,
            ub=self.instance.upper_bound,
        )
        self.S = self.model.addVars(self.T, self.I, vtype=GRB.BINARY, name="s")
        self.Alpha = self.model.addVars(
            self.T, self.I, self.J, vtype=GRB.BINARY, name="alpha"
        )
        self.Beta = self.model.addVars(
            self.T, self.I, self.J, vtype=GRB.BINARY, name="beta"
        )

    def _build_objective(self):
        self.model.setObjective(
            gp.quicksum(
                self.Alpha[t, i, j] * self.D[self.A[t, i, j - 1], self.A[t, i, j]]
                + self.Beta[t, i, j]
                * (self.D[self.A[t, i, j - 1], t] + self.D[t, self.A[t, i, j]])
                for t in self.T
                for j in self.J
                for i in self.I
            ),
            GRB.MINIMIZE,
        )

    def _build_constraints(self):
        # (0) \sum_{i=1}^m s_{\vec a_{t,i}} = 1
        self.model.addConstrs(
            (gp.quicksum(self.S[t, i] for i in self.I) == 1 for t in self.T),
            name="one_pattern_per_team",
        )

        # (1) α_{t,i,j} ≤ 1 - y_{t,j}
        self.model.addConstrs(
            (
                self.Alpha[t, i, j] <= 1 - self.Y[t, j]
                for t in self.T
                for i in self.I
                for j in self.J
            ),
            name="alpha_le_1_minus_y",
        )

        # (2) α_{t,i,j} ≤ s_i
        self.model.addConstrs(
            (
                self.Alpha[t, i, j] <= self.S[t, i]
                for t in self.T
                for i in self.I
                for j in self.J
            ),
            name="alpha_le_s",
        )

        # (3) α_{t,i,j} ≥ s_i - y_{i,j}
        self.model.addConstrs(
            (
                self.Alpha[t, i, j] >= self.S[t, i] - self.Y[t, j]
                for t in self.T
                for i in self.I
                for j in self.J
            ),
            name="alpha_ge_s_minus_y",
        )

        # (4) β_{t,i,j} ≤ y_{i,j}
        self.model.addConstrs(
            (
                self.Beta[t, i, j] <= self.Y[t, j]
                for t in self.T
                for i in self.I
                for j in self.J
            ),
            name="beta_le_y",
        )

        # (5) β_{t,i,j} ≤ s_i
        self.model.addConstrs(
            (
                self.Beta[t, i, j] <= self.S[t, i]
                for t in self.T
                for i in self.I
                for j in self.J
            ),
            name="beta_le_s",
        )

        # (6) β_{t,i,j} ≥ s_i + y_{i,j} - 1
        self.model.addConstrs(
            (
                self.Beta[t, i, j] >= self.S[t, i] + self.Y[t, j] - 1
                for t in self.T
                for i in self.I
                for j in self.J
            ),
            name="beta_ge_s_plus_y_minus_1",
        )

        # (7) h_{t, j} ≤ y_{t, j} * M
        self.model.addConstrs(
            (self.H[t, j] <= self.Y[t, j] * self.ub for t in self.T for j in self.J),
            name="home_le_yM",
        )

        # (8) h_{t, j} - y_{t, j} ≥ 0
        self.model.addConstrs(
            (self.H[t, j] - self.Y[t, j] >= 0 for t in self.T for j in self.J),
            name="h_minus_y_ge_0",
        )

        # (9) ∑_{j=1}^{n-u} ∑_{k=0}^{u-1} y_{t, j+k} ≥ 1
        self.model.addConstrs(
            (
                gp.quicksum(
                    self.Y[t, j + k]
                    for j in range(1, self.n - self.ub + 1)
                    for k in range(self.ub)
                )
                >= 1
                for t in self.T
            ),
            name="max_away",
        )

        # (10) ∑_{j=0}^{n-1} h_{t,j} = n - 1
        self.model.addConstrs(
            (
                gp.quicksum(self.H[t, j] for j in range(self.n)) == self.n - 1
                for t in self.T
            ),
            name="no_home_games",
        )

    def _build_schedule(self):
        self.schedule = np.empty((self.n, 2 * (self.n - 1)), dtype=int)
        chosen_schedules = [
            (t, i) for t in self.T for i in self.I if self.S[t, i].X > 0.5
        ]

        for t, i in chosen_schedules:
            rounds = 0
            for j in range(self.n - 1):
                if self.H[t, j].X > 0.5:
                    for k in range(int(self.H[t, j].X)):
                        self.schedule[t, rounds] = t
                        rounds += 1
                self.schedule[t, rounds] = self.A[t, i, j + 1]
                rounds += 1
            if self.H[t, self.n - 1].X > 0.5:
                for k in range(int(self.H[t, self.n - 1].X)):
                    self.schedule[t, rounds] = t
                    rounds += 1
            assert rounds == 2 * (self.n - 1)

    def print_summary(self):
        print(f"Objective value: {self.model.objVal}")
        self._print_schedule()

    def _print_schedule(self):
        print("\nSchedule:")

        header = "Round | " + " | ".join(
            [f"{self.instance.teams[t]}" for t in self.teams]
        )
        print(header)
        print("-" * len(header))

        for j in range(2 * (self.n - 1)):
            row = [f"{j + 1:>5}"]
            for t in self.teams:
                team = self.instance.teams[t]
                if self.schedule[t, j] == t:
                    row.append(team)
                else:
                    row.append(self.instance.teams[self.schedule[t, j]])
            print(" | ".join(row))

    def solve(self):
        if not hasattr(self, "gurobi_options"):
            self.gurobi_options = {}
        print(self.gurobi_options)
        with gp.Env(empty=True) as env:
            env.start()
            self.model = gp.Model("TTP", env=env)
            self._build_patterns()
            self._build_variables()
            self._build_decision_variables()
            self._build_objective()
            self._build_constraints()
            self.model.update()
            self._solve()

    def _solve(self):
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
        self._build_schedule()


if __name__ == "__main__":
    gurobi_lic = None
    if len(sys.argv) < 2:
        print("Usage: python exact_method.py <instance_file> [gurobi license file]")
        sys.exit(1)
    if len(sys.argv) > 2:
        gurobi_lic_path = sys.argv[2]
        with open(gurobi_lic_path, "r") as f:
            lic_content = f.read()
    else:
        lic_content = None
    instance_file = sys.argv[1]
    instance = Instance.from_file(instance_file)
    instance.print_summary()
    method = ExactMethod(instance, gurobi_lic=lic_content)
    method.solve()
    if method.model.status == GRB.OPTIMAL:
        method.print_summary()
