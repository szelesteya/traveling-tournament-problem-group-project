from instance import Instance
import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np
import sys
import math


def setup_gurobi_options(lic_content: str) -> dict[str, str]:
    lines = lic_content.strip().split("\n")

    wls_access_prefix = "WLSACCESSID="
    wls_access_id = [line[len(wls_access_prefix):] for line in lines if line.startswith(wls_access_prefix)][0]

    wls_secret_prefix = "WLSSECRET="
    wls_secret = [line[len(wls_secret_prefix):] for line in lines if line.startswith(wls_secret_prefix)][0]

    license_id_prefix = "LICENSEID="
    license_id = [int(line[len(license_id_prefix):]) for line in lines if line.startswith(license_id_prefix)][0]

    return {
        "WLSACCESSID": wls_access_id,
        "WLSSECRET": wls_secret,
        "LICENSEID": license_id,
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

    
    def _build_patterns(self, K_per_team: int = 6, seed: int | None = 0):
        """
        Build opponent-order patterns per team.
        A[t,i,0] = t, A[t,i,n] = t; A[t,i,1..n-1] = permutation of opponents.
        Always creates at least one canonical pattern; fills the rest with random valid ones.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        n = len(self.T)
        pat_len = n + 1
        self.A = np.full((n, K_per_team, pat_len), -1, dtype=int)
        for t in self.T:
            # canonical cyclic order: t+1, t+2, ..., t+n-1 (mod n)
            opps_canonical = [(t + k) % n for k in range(1, n)]
            self.A[t, 0, :] = [t] + opps_canonical + [t]
            if K_per_team > 1:
                seen = {tuple(opps_canonical)}
                opps = [u for u in self.T if u != t]
                i = 1
                attempts = 0
                while i < K_per_team and attempts < 1000 * K_per_team:
                    rng.shuffle(opps)
                    key = tuple(opps)
                    attempts += 1
                    if key in seen:
                        continue
                    seen.add(key)
                    self.A[t, i, :] = [t] + list(opps) + [t]
                    i += 1
                while i < K_per_team:
                    self.A[t, i, :] = [t] + opps_canonical + [t]
                    i += 1


    def _validate_patterns(self):
        """
        Ensure each A[t,i,1..n-1] is a permutation of opponents (no missing/duplicate opponents, no t).
        Raises on error so you fail fast with a helpful message instead of an IIS later.
        """
        for t in self.T:
            opps = set(self.T) - {t}
            for i in self.I:
                row = [int(self.A[t, i, j]) for j in range(1, self.n)]
                if set(row) != opps or len(set(row)) != (self.n - 1) or any(x == t for x in row):
                    raise ValueError(f"Invalid pattern for team {t}, i={i}: {row}")

    def _build_distances(self):
        self.D = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.D[i, j] = 0.0
                else:
                    val = self.instance.distances.get((i, j))
                    if val is None:
                        raise ValueError(f"Missing distance ({i},{j}) in instance.distances")
                    self.D[i, j] = float(val)


    def _build_variables(self):
        self.n = self.instance.n
        self.T = range(self.n)
        # Positions:
        self.J_home = range(self.n)      # j = 0..n-1  (for Y, H)
        self.J_pos  = range(1, self.n)   # j = 1..n-1  (for Alpha, Beta, objective)
        self.teams = list(self.T)

        self._build_patterns()
        self._validate_patterns()
        self._build_distances()

    def _build_decision_variables(self):
        # Y, H over j = 0..n-1
        self.Y = self.model.addVars(self.T, self.J_home, vtype=GRB.BINARY,  name="y")
        self.H = self.model.addVars(self.T, self.J_home, vtype=GRB.INTEGER, lb=0, ub=self.instance.upper_bound, name="h")

        # S over patterns (unchanged)
        self.S = self.model.addVars(self.T, self.I, vtype=GRB.BINARY, name="s")

        # Alpha, Beta over away positions j = 1..n-1
        self.Alpha = self.model.addVars(self.T, self.I, self.J_pos, vtype=GRB.BINARY, name="alpha")
        self.Beta  = self.model.addVars(self.T, self.I, self.J_pos, vtype=GRB.BINARY, name="beta")


    def _build_objective(self):
        # α/β terms only for away positions j = 1..n-1
        sum_trips = gp.quicksum(
            self.Alpha[t, i, j] * self.D[self.A[t, i, j - 1], self.A[t, i, j]] +
            self.Beta[t,  i, j] * ( self.D[self.A[t, i, j - 1], t] + self.D[t, self.A[t, i, j]] )
            for t in self.T for i in self.I for j in self.J_pos
        )
        # Final return-home: from last away a_{n-1} back to t  (D[a_{n-1}, t])
        sum_return = gp.quicksum(
            self.S[t, i] * self.D[self.A[t, i, self.n - 1], t]
            for t in self.T for i in self.I
        )
        self.model.setObjective(sum_trips + sum_return, GRB.MINIMIZE)

    def _build_constraints(self):
        # (1) one pattern per team
        self.model.addConstrs(
            (gp.quicksum(self.S[t, i] for i in self.I) == 1 for t in self.T),
            name="one_pattern_per_team"
        )

        # (2)–(4) Alpha logic on j = 1..n-1
        self.model.addConstrs(
            (self.Alpha[t, i, j] <= 1 - self.Y[t, j] for t in self.T for i in self.I for j in self.J_pos),
            name="alpha_le_1_minus_y"
        )
        self.model.addConstrs(
            (self.Alpha[t, i, j] <= self.S[t, i] for t in self.T for i in self.I for j in self.J_pos),
            name="alpha_le_s"
        )
        self.model.addConstrs(
            (self.Alpha[t, i, j] >= self.S[t, i] - self.Y[t, j] for t in self.T for i in self.I for j in self.J_pos),
            name="alpha_ge_s_minus_y"
        )

        # (5)–(7) Beta logic on j = 1..n-1
        self.model.addConstrs(
            (self.Beta[t, i, j] <= self.Y[t, j] for t in self.T for i in self.I for j in self.J_pos),
            name="beta_le_y"
        )
        self.model.addConstrs(
            (self.Beta[t, i, j] <= self.S[t, i] for t in self.T for i in self.I for j in self.J_pos),
            name="beta_le_s"
        )
        self.model.addConstrs(
            (self.Beta[t, i, j] >= self.S[t, i] + self.Y[t, j] - 1 for t in self.T for i in self.I for j in self.J_pos),
            name="beta_ge_s_plus_y_minus_1"
        )

        # (8) H ≤ U·Y  on j = 0..n-1  (already correct)
        self.model.addConstrs(
            (self.H[t, j] <= self.Y[t, j] * self.ub for t in self.T for j in self.J_home),
            name="home_le_yU"
        )

        # (9) H ≥ L·Y  (fix: replace H - Y ≥ 0)
        self.model.addConstrs(
            (self.H[t, j] >= self.Y[t, j] * self.lb for t in self.T for j in self.J_home),
            name="home_ge_yL"
        )

        # (10) Sliding window: in any U consecutive positions, at least one Y = 1
        #     windows start at p = 0..n-U
        self.model.addConstrs(
            (gp.quicksum(self.Y[t, j] for j in range(p, p + self.ub)) >= 1
            for t in self.T for p in range(0, self.n - self.ub + 1)),
            name="max_away_run"
        )

        # (11) Sum of home-insertion lengths equals n-1
        self.model.addConstrs(
            (gp.quicksum(self.H[t, j] for j in self.J_home) == self.n - 1 for t in self.T),
            name="total_home_blocks"
        )


    def _build_schedule(self):
        self.schedule = np.empty((self.n, 2 * (self.n - 1)), dtype=int)
        chosen_schedules = [(t, i) for t in self.T for i in self.I if self.S[t, i].X > 0.5]

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

        header = "Round | " + " | ".join([f"{self.instance.teams[t]}" for t in self.teams])
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

    def validity_of_schedules(self, patterns_per_team = None):
        """
        Orchestrator for Section 2.1.4.
        Requires: self.A, self.S, self.H, self.T, self.n built already.
        Creates: self.r_pos, self.q, self.w_and, self.delta
        Adds constraints: (13)–(23)
        """
        self._init_sets(patterns_per_team)
        self._build_tournament_variables()

        self._add_r_constraints()         # (13)–(15)
        self._add_q_constraints()         # (16)–(17)
        self._add_and_constraints()       # (18)
        self._add_delta_mapping()         # (19)
        self._add_tournament_constraints()# (20)–(23)
        
    def _init_sets(self, patterns_per_team=None):
        self.Rounds = list(range(1, 2 * (self.n - 1) + 1))  
        self.AwayPositions = list(range(1, self.n))         

        if patterns_per_team is None:
            I_all = list(range(self.A.shape[1]))
            self.I_map = {t: I_all for t in self.T}
        else:
            self.I_map = {t: list(patterns_per_team[t]) for t in self.T}   

    def _build_tournament_variables(self):

        # (13)–(17): r and q
        self.r_pos = self.model.addVars(self.T, self.AwayPositions, vtype=GRB.CONTINUOUS, name="r")
        self.q     = self.model.addVars(self.T, self.AwayPositions, self.Rounds, vtype=GRB.BINARY, name="q")

        # (18): w = S AND q
        self.w_and = gp.tupledict()
        for t in self.T:
            for i in self.I_map[t]:
                for j in self.AwayPositions:
                    for rnd in self.Rounds:
                        self.w_and[t, i, j, rnd] = self.model.addVar(vtype=GRB.BINARY, name=f"w[{t},{i},{j},{rnd}]")

        # (19)–(23): delta (away-at)
        self.delta = self.model.addVars(self.T, self.T, self.Rounds, vtype=GRB.BINARY, name="delta")

        self.model.update()

    def _add_r_constraints(self):
        # (13) r_{t,1} = 1 + h_{t,0}
        self.model.addConstrs((self.r_pos[t, 1] == 1 + self.H[t, 0] for t in self.T), name="c13")

        # (14) r_{t,j+1} = r_{t,j} + 1 + h_{t,j}  for j=1..n-2
        if self.n >= 3:
            self.model.addConstrs(
                (self.r_pos[t, j+1] == self.r_pos[t, j] + 1 + self.H[t, j]
                for t in self.T for j in range(1, self.n - 1)),
                name="c14"
            )

        # (15) bounds
        self.model.addConstrs((self.r_pos[t, 1] >= 1 for t in self.T), name="c15_lower")
        self.model.addConstrs((self.r_pos[t, self.n - 1] <= 2 * (self.n - 1) for t in self.T), name="c15_upper")


    def _add_q_constraints(self):
        # (16) 
        self.model.addConstrs(
            (gp.quicksum(self.q[t, j, rnd] for rnd in self.Rounds) == 1
            for t in self.T for j in self.AwayPositions),
            name="c16"
        )
        # (17) 
        self.model.addConstrs(
            (gp.quicksum(rnd * self.q[t, j, rnd] for rnd in self.Rounds) == self.r_pos[t, j]
            for t in self.T for j in self.AwayPositions),
            name="c17"
        )

    def _add_and_constraints(self):
        # (18) w = S AND q
        for t in self.T:
            for i in self.I_map[t]:
                for j in self.AwayPositions:
                    for rnd in self.Rounds:
                        self.model.addConstr(self.w_and[t, i, j, rnd] <= self.S[t, i],        name=f"c18a[{t},{i},{j},{rnd}]")
                        self.model.addConstr(self.w_and[t, i, j, rnd] <= self.q[t, j, rnd],    name=f"c18b[{t},{i},{j},{rnd}]")
                        self.model.addConstr(self.w_and[t, i, j, rnd] >= self.S[t, i] + self.q[t, j, rnd] - 1,
                                            name=f"c18c[{t},{i},{j},{rnd}]")

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
                        self.delta[t, t2, rnd] == gp.quicksum(
                            self.w_and[t, i, j, rnd]
                            for i in self.I_map[t]
                            for j in self.AwayPositions
                            if self._opponent_at(t, i, j) == t2
                        ),
                        name=f"c19[{t},{t2},{rnd}]"
                    )

    def _add_tournament_constraints(self):
        # (20) one game per round per team
        self.model.addConstrs(
            (gp.quicksum(self.delta[t, t2, rnd] + self.delta[t2, t, rnd] for t2 in self.T if t2 != t) == 1
            for t in self.T for rnd in self.Rounds),
            name="c20"
        )
        # (21) double round-robin (once away, once home)
        self.model.addConstrs(
            (gp.quicksum(self.delta[t, t2, rnd] for rnd in self.Rounds) == 1
            for t in self.T for t2 in self.T if t2 != t),
            name="c21a"
        )
        self.model.addConstrs(
            (gp.quicksum(self.delta[t2, t, rnd] for rnd in self.Rounds) == 1
            for t in self.T for t2 in self.T if t2 != t),
            name="c21b"
        )
        # (22) no immediate rematch
        if len(self.Rounds) >= 2:
            self.model.addConstrs(
                ((self.delta[t, t2, rnd] + self.delta[t2, t, rnd]) +
                (self.delta[t, t2, rnd + 1] + self.delta[t2, t, rnd + 1]) <= 1
                for t in self.T for t2 in self.T if t2 != t for rnd in self.Rounds[:-1]),
                name="c22"
            )
        # (23) no self matches
        self.model.addConstrs(
            (self.delta[t, t, rnd] == 0 for t in self.T for rnd in self.Rounds),
            name="c23"
        )




    def solve(self):
        if not hasattr(self, "gurobi_options"):
            self.gurobi_options = {}
        with gp.Env(params=self.gurobi_options) as env:
            self.model = gp.Model("TTP", env=env)
            self._build_variables()
            self._build_decision_variables()
            self._build_objective()
            self._build_constraints()
            self.validity_of_schedules()
            self.model.update()
            self._solve()


    def _bounds_sanity_check(self):
        import math
        L = max(1, int(self.lb))
        U = int(self.ub)
        minY = math.ceil(self.n / U)
        maxY = (self.n - 1) // L
        print(f"[Bounds check] need Y >= {minY}, allowed Y <= {maxY} (n={self.n}, L={L}, U={U})")
        if minY > maxY:
            raise ValueError(f"Infeasible bounds: need at least {minY} insertions from U={U}, "
                            f"but at most {maxY} fit into total home rounds with L={L}.")

    def _solve(self):
        self._bounds_sanity_check()
        self.model.optimize()

        # ---- guard: only read .X if we actually have a solution ----
        if self.model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
            if self.model.SolCount >= 1:
                # OK to read variable values
                self._build_schedule()    # this is where you use self.S[t,i].X etc.
            else:
                print("No feasible solution found (SolCount=0).")
                self._diagnose_infeasibility()
                return
        else:
            print(f"Model status: {self.model.Status} (no solution available).")
            self._diagnose_infeasibility()
            return

    def _diagnose_infeasibility(self):
        try:
            self.model.computeIIS()
            self.model.write("infeasible.ilp")   # or "model.ilp" then "model.ilp"
            self.model.write("iis.ilp")
            print("Wrote IIS to iis.ilp (open it to see which rows/vars cause the conflict).")
        except Exception as e:
            print(f"Could not compute IIS: {e}")


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


