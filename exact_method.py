from instance import Instance
import gurobipy as gp
from gurobipy import GRB

import itertools
import numpy as np
import sys

PATTERNS_PER_TEAM_DEFAULT = 7
RANDOM_SEED_DEFAULT = 42


def setup_gurobi_options(lic_content: str) -> dict[str, str]:
    params = {}
    for line in lic_content.strip().split("\n"):
        if "=" in line:
            key, value = line.split("=", 1)
            params[key] = int(value) if key == "LICENSEID" else value

    return {
        "WLSACCESSID": params["WLSACCESSID"],
        "WLSSECRET": params["WLSSECRET"],
        "LICENSEID": params["LICENSEID"],
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
    def I(self):  # noqa: E743 ambiguous function name
        if not hasattr(self, "A"):
            raise AttributeError("Patterns have not been built yet.")
        return range(len(self.A[0]))  # Number of patterns per team

    def _add_optimal_patterns(self, sampled: np.ndarray, n):
        if n == 4:
            """optimal_patterns = np.array(
                [[[0, 2, 1, 3]], [[1, 0, 2, 3]], [[2, 0, 3, 1]], [[3, 1, 2, 0]]]
            )"""
            return sampled
        if n == 6:
            """optimal_patterns = np.array(
                [
                    [0, 2, 3, 5, 1, 4],
                    [1, 5, 0, 4, 2, 3],
                    [2, 3, 5, 0, 4, 1],
                    [3, 5, 2, 1, 0, 4],
                    [4, 0, 2, 1, 3, 5],
                    [5, 0, 4, 1, 2, 3],
                ],
                dtype=int,
            )"""
            return sampled
        if n == 8:
            optimal_patterns = np.array(
                [
                    [0, 4, 6, 5, 7, 2, 1, 3],
                    [1, 5, 6, 7, 4, 0, 3, 2],
                    [2, 3, 7, 6, 5, 0, 4, 1],
                    [3, 0, 4, 1, 7, 6, 5, 2],
                    [4, 2, 1, 3, 5, 6, 7, 0],
                    [5, 4, 0, 7, 6, 2, 1, 3],
                    [6, 7, 2, 1, 3, 5, 4, 0],
                    [7, 5, 3, 1, 2, 0, 4, 6]
                    
                ],
                dtype=int,
            )
            #return sampled

        # Overwrite one sampled pattern per start team with the desired away order
        # Keep shapes consistent: sampled has shape (n_teams, patterns_per_team, n_teams)
        for pattern in optimal_patterns:
            start_team = pattern[0]
            sampled[start_team, 0, :] = pattern
        return sampled

    def _build_patterns(
        self,
        no_patterns_per_team: int = PATTERNS_PER_TEAM_DEFAULT,
        random_seed: int = RANDOM_SEED_DEFAULT,
    ):
        all_patterns = np.array([list(perm) for perm in itertools.permutations(self.T)])
        # Get unique starting values (first elements)
        unique_starts = np.unique(all_patterns[:, 0])

        # Group into a 3D array — one subarray per unique starting value
        grouped_patterns = [
            all_patterns[all_patterns[:, 0] == start] for start in unique_starts
        ]
        grouped_patterns = np.array(grouped_patterns, dtype=int)

        # Determine how many patterns to sample per team (per start group)
        max_patterns = min(grouped_patterns.shape[1], no_patterns_per_team)

        # Randomly sample patterns per group with the provided seed
        rng = np.random.default_rng(random_seed)
        if grouped_patterns.shape[1] == max_patterns:
            sampled = grouped_patterns
        else:
            sampled_list = []
            for group in grouped_patterns:
                indices = rng.choice(group.shape[0], size=max_patterns, replace=False)
                sampled_list.append(group[indices])
            sampled = np.stack(sampled_list, axis=0)

        # Add optimal patterns to the sampled patterns
        sampled = self._add_optimal_patterns(sampled, self.n)

        self.A = sampled

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
        # gamma[t,i,j] = s[t,i] AND y[t,j-1] (j = 1..n-1)
        self.Gamma = self.model.addVars(
            self.T, self.I, range(1, self.n), vtype=GRB.BINARY, name="gamma"
        )

    def _build_objective(self):
        sum_distances = gp.quicksum(
            # Base edge cost when no home break between consecutive away opponents
            self.S[t, i] * self.D[self.A[t, i, j - 1], self.A[t, i, j]]
            # If there is a home block (gamma=1), add the extra detour via home
            + self.Gamma[t, i, j]
            * (
                self.D[self.A[t, i, j - 1], t]
                + self.D[t, self.A[t, i, j]]
                - self.D[self.A[t, i, j - 1], self.A[t, i, j]]
            )
            for t in self.T
            for j in range(1, len(self.T))
            for i in self.I
        )
        sum_back_travel = gp.quicksum(
            self.S[t, i] * self.D[self.A[t, i, -1], t] for t in self.T for i in self.I
        )
        self.model.setObjective(sum_distances + sum_back_travel, GRB.MINIMIZE)

    def _build_constraints(self):
        # (1) \sum_{i=1}^m s_{\vec a_{t,i}} = 1
        self.model.addConstrs(
            (gp.quicksum(self.S[t, i] for i in self.I) == 1 for t in self.T),
            name="one_pattern_per_team",
        )

        # (2) gamma = s AND y_{t,j-1} for j=1..n-1
        self.model.addConstrs(
            (
                self.Gamma[t, i, j] <= self.Y[t, j - 1]
                for t in self.T
                for i in self.I
                for j in range(1, self.n)
            ),
            name="gamma_le_y",
        )

        self.model.addConstrs(
            (
                self.Gamma[t, i, j] <= self.S[t, i]
                for t in self.T
                for i in self.I
                for j in range(1, self.n)
            ),
            name="gamma_le_s",
        )

        self.model.addConstrs(
            (
                self.Gamma[t, i, j] >= self.S[t, i] + self.Y[t, j - 1] - 1
                for t in self.T
                for i in self.I
                for j in range(1, self.n)
            ),
            name="gamma_ge_s_plus_y_minus_1",
        )

        # (3) h_{t, j} ≤ y_{t, j} * M
        self.model.addConstrs(
            (self.H[t, j] <= self.Y[t, j] * self.ub for t in self.T for j in self.J),
            name="home_le_yM",
        )

        # (4) h_{t, j} - y_{t, j} ≥ 0
        self.model.addConstrs(
            (self.H[t, j] - self.Y[t, j] >= 0 for t in self.T for j in self.J),
            name="h_minus_y_ge_0",
        )

        # (5) ∑_{j=0}^{n-1-u} ∑_{k=0}^{u} y_{t, j+k} ≥ 1
        self.model.addConstrs(
            (
                gp.quicksum(
                    self.Y[t, j + k]
                    for j in range(0, self.n - self.ub)
                    for k in range(self.ub + 1)
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
        self._print_travel()

    def _print_travel(self):
        print("\nTravels")
        sum_sum_travel = 0
        for t in self.teams:
            sum_travel = 0
            current_loc = t
            travel_str = f"{self.instance.teams[t]}: {self.instance.teams[t]}"
            for r in range(2 * (len(self.T) - 1)):
                next_loc = self.schedule[t, r]
                distance = self.D[current_loc, next_loc]
                sum_travel += distance
                current_loc = next_loc
                travel_str += f" = {distance} => {self.instance.teams[next_loc]}"
            back_travel = self.D[self.schedule[t, 2 * (len(self.T) - 1) - 1], t]
            sum_travel += back_travel
            sum_sum_travel += sum_travel
            travel_str += f" = {back_travel} => {self.instance.teams[t]}: {sum_travel}"
            print(travel_str)
        print(f"Full travel distance: {sum_sum_travel}")

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

    def validity_of_schedules(self):
        """
        Orchestrator for Section 2.1.4.
        Requires: self.A, self.S, self.H, self.T, self.n built already.
        Creates: self.r_pos, self.q, self.w_and, self.delta
        Adds constraints: (13)–(23)
        """
        self._init_sets()
        self._build_tournament_variables()

        self._add_r_constraints()  # (13)–(15)
        self._add_q_constraints()  # (16)–(17)
        self._add_and_constraints()  # (18)
        self._add_delta_mapping()  # (19)
        self._add_tournament_constraints()  # (20)–(23)

    def _init_sets(self):
        self.Rounds = list(range(1, 2 * (self.n - 1) + 1))
        self.AwayPositions = list(range(1, self.n))

        I_all = list(range(self.A.shape[1]))
        self.I_map = {t: I_all for t in self.T}

    def _build_tournament_variables(self):

        # r and q
        self.r_pos = self.model.addVars(
            self.T, self.AwayPositions, vtype=GRB.CONTINUOUS, name="r"
        )
        self.q = self.model.addVars(
            self.T, self.AwayPositions, self.Rounds, vtype=GRB.BINARY, name="q"
        )

        # w = S AND q
        self.w_and = gp.tupledict()
        for t in self.T:
            for i in self.I_map[t]:
                for j in self.AwayPositions:
                    for rnd in self.Rounds:
                        self.w_and[t, i, j, rnd] = self.model.addVar(
                            vtype=GRB.BINARY, name=f"w[{t},{i},{j},{rnd}]"
                        )

        # delta (away-at)
        self.delta = self.model.addVars(
            self.T, self.T, self.Rounds, vtype=GRB.BINARY, name="delta"
        )

        self.model.update()

    def _add_r_constraints(self):
        # (8) r_{t,1} = 1 + h_{t,0}
        self.model.addConstrs(
            (self.r_pos[t, 1] == 1 + self.H[t, 0] for t in self.T), name="c8"
        )

        # (14) r_{t,j+1} = r_{t,j} + 1 + h_{t,j}  for j=1..n-2
        if self.n >= 3:
            self.model.addConstrs(
                (
                    self.r_pos[t, j + 1] == self.r_pos[t, j] + 1 + self.H[t, j]
                    for t in self.T
                    for j in range(1, self.n - 1)
                ),
                name="c9",
            )

        # (10) bounds
        self.model.addConstrs((self.r_pos[t, 1] >= 1 for t in self.T), name="c15_lower")
        self.model.addConstrs(
            (self.r_pos[t, self.n - 1] <= 2 * (self.n - 1) for t in self.T),
            name="c10_upper",
        )

    def _add_q_constraints(self):
        # (11)
        self.model.addConstrs(
            (
                gp.quicksum(self.q[t, j, rnd] for rnd in self.Rounds) == 1
                for t in self.T
                for j in self.AwayPositions
            ),
            name="c16",
        )
        # (12)
        self.model.addConstrs(
            (
                gp.quicksum(rnd * self.q[t, j, rnd] for rnd in self.Rounds)
                == self.r_pos[t, j]
                for t in self.T
                for j in self.AwayPositions
            ),
            name="c12",
        )

    def _add_and_constraints(self):
        # (13) w = S AND q
        for t in self.T:
            for i in self.I_map[t]:
                for j in self.AwayPositions:
                    for rnd in self.Rounds:
                        self.model.addConstr(
                            self.w_and[t, i, j, rnd] <= self.S[t, i],
                            name=f"c18a[{t},{i},{j},{rnd}]",
                        )
                        self.model.addConstr(
                            self.w_and[t, i, j, rnd] <= self.q[t, j, rnd],
                            name=f"c18b[{t},{i},{j},{rnd}]",
                        )
                        self.model.addConstr(
                            self.w_and[t, i, j, rnd]
                            >= self.S[t, i] + self.q[t, j, rnd] - 1,
                            name=f"c18c[{t},{i},{j},{rnd}]",
                        )

    def _opponent_at(self, t: int, i: int, j: int) -> int:
        # A[t,i,0]=t and A[t,i,n]=t
        return int(self.A[t, i, j])

    def _add_delta_mapping(self):
        # (19)  delta_{t,t',r} = sum_{i} sum_{j: a_{t,i,j}=t'} w_{t,i,j,r}
        for t in self.T:
            for t2 in self.T:
                if t == t2:
                    continue
                for rnd in self.Rounds:
                    self.model.addConstr(
                        self.delta[t, t2, rnd]
                        == gp.quicksum(
                            self.w_and[t, i, j, rnd]
                            for i in self.I_map[t]
                            for j in self.AwayPositions
                            if self._opponent_at(t, i, j) == t2
                        ),
                        name=f"c19[{t},{t2},{rnd}]",
                    )

    def _add_tournament_constraints(self):
        # (20) one game per round per team
        self.model.addConstrs(
            (
                gp.quicksum(
                    self.delta[t, t2, rnd] + self.delta[t2, t, rnd]
                    for t2 in self.T
                    if t2 != t
                )
                == 1
                for t in self.T
                for rnd in self.Rounds
            ),
            name="c20",
        )
        # (21) double round-robin (once away, once home) – combined per unordered pair
        self.model.addConstrs(
            (
                gp.quicksum(
                    self.delta[t, t2, rnd] + self.delta[t2, t, rnd]
                    for rnd in self.Rounds
                )
                == 2
                for t in self.T
                for t2 in self.T
                if t2 > t
            ),
            name="c21",
        )
        # (22) no immediate rematch
        if len(self.Rounds) >= 2:
            self.model.addConstrs(
                (
                    (self.delta[t, t2, rnd] + self.delta[t2, t, rnd])
                    + (self.delta[t, t2, rnd + 1] + self.delta[t2, t, rnd + 1])
                    <= 1
                    for t in self.T
                    for t2 in self.T
                    if t2 > t
                    for rnd in self.Rounds[:-1]
                ),
                name="c22",
            )
        # (23) no self matches
        self.model.addConstrs(
            (self.delta[t, t, rnd] == 0 for t in self.T for rnd in self.Rounds),
            name="c23",
        )

    def solve(self):
        if not hasattr(self, "gurobi_options"):
            self.gurobi_options = {}
        print(self.gurobi_options)
        with gp.Env(params=self.gurobi_options) as env:
            env.start()
            self.model = gp.Model("TTP", env=env)
            self._build_variables()
            self._build_decision_variables()
            self._build_objective()
            self._build_constraints()
            self.validity_of_schedules()
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
        method.model.write("models/model.lp")
