import gurobipy as gp
from gurobipy import GRB
import numpy as np
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

    def __init__(self, teams: dict[int, str], distances: dict[(str, str), int], lower_bound: int = 1, upper_bound: int = 2):
        self.teams = teams
        self.distances = distances
        self.n = len(teams)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @classmethod
    def _load_teams(cls, root: Any) -> dict[int, str]:
        return {team_elem.get("id"): team_elem.get("name") for team_elem in root.xpath('.//Teams/team')}

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
    def _load_bounds(cls, root: Any) -> (int, int):
        constraints = root.xpath(".//Constraints/SeparationConstraints/SE1")
        lower_bound = int(constraints[0].get("min", 1)) if constraints else 1
        upper_bound = int(constraints[0].get("max", 2)) if constraints else 2
        return lower_bound, upper_bound

    @classmethod
    def from_file(cls, path: str) -> 'Instance':
        tree = etree.parse(Path(path))
        root = tree.getroot()
        bounds = cls._load_bounds(root)

        return cls(
            teams=cls._load_teams(root),
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
    instance = Instance.from_file(sys.argv[1])
    instance.print_summary()