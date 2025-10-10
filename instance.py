import lxml.etree as etree
from pathlib import Path
from typing import Any
import sys


class Instance:
    teams: dict[int, str]
    distances: dict[(str, str), int]
    n: int
    lower_bound: int
    upper_bound: int

    def __init__(self, teams: dict[int, str], distances: dict[(str, str), int],
                 lower_bound: int = 1, upper_bound: int = 2):
        self.teams = teams
        self.distances = distances
        self.n = len(teams)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @classmethod
    def _load_teams(cls, root: Any) -> dict[int, str]:
        return {int(team_elem.get("id")): team_elem.get("name") for team_elem in root.xpath('.//Teams/team')}

    @classmethod
    def _load_distances(cls, root: Any) -> dict[(str, str), int]:
        distances = {}
        for dist_elem in root.xpath('.//Distances/distance'):
            team1 = int(dist_elem.get("team1"))
            team2 = int(dist_elem.get("team2"))
            value = int(dist_elem.get("dist"))
            distances[(team1, team2)] = value
            distances[(team2, team1)] = value  # Assuming symmetry
        return distances

    @classmethod
    def _load_bound_limits(cls, root: Any, n:int, lower_bound: int | None, upper_bound: int | None) -> (int, int):
        constraints = root.xpath(".//Constraints/SeparationConstraints/SE1")
        lower_bound_limit = min(int(constraints[0].get("min", 1)), n - 1) if constraints else n - 1
        if lower_bound is None:
            lower_bound = lower_bound_limit
        assert lower_bound >= lower_bound_limit, \
            f"Provided lower bound {lower_bound} is below instance limit {lower_bound_limit}"
        upper_bound_limit = min(int(constraints[0].get("max", 2)), n-1) if constraints else n - 1
        if upper_bound is None:
            upper_bound = upper_bound_limit
        assert upper_bound <= upper_bound_limit, \
            f"Provided upper bound {upper_bound} exceeds instance limit {upper_bound_limit}"
        return lower_bound, upper_bound

    @classmethod
    def from_file(cls, path: str, lb: int = None, ub: int = None) -> 'Instance':
        tree = etree.parse(Path(path))
        root = tree.getroot()
        teams = cls._load_teams(root)
        bounds = cls._load_bound_limits(root, len(teams), lb, ub)

        return cls(
            teams=teams,
            distances=cls._load_distances(root),
            lower_bound=bounds[0],
            upper_bound=bounds[1],
        )

    def _print_distances(self):
        # Print distances as a pretty matrix
        team_ids = list(self.teams.keys())
        team_names = [self.teams[tid] for tid in team_ids]
        # Header row
        print("Distances matrix:")
        print(" " * 12 + " ".join(f"{name:>10}" for name in team_names))
        for tid1 in team_ids:
            row = [f"{self.teams[tid1]:>10}"]
            for tid2 in team_ids:
                dist = self.distances.get((tid1, tid2), 0)
                row.append(f"{dist:10}")
            print(" ".join(row))

    def print_summary(self):
        print(f"Number of teams: {self.n}")
        print(f"Teams: {self.teams}")
        self._print_distances()
        print(f"Lower Bound: {self.lower_bound}, Upper Bound: {self.upper_bound}")


# Read XML data
if __name__ == "__main__":
    if len(sys.argv) < 4:
        instance = Instance.from_file(sys.argv[1])
    else:
        instance = Instance.from_file(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    instance.print_summary()
