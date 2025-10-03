import numpy as np
from lxml import etree as ET

import xml.etree.ElementTree as ET
import numpy as np

class Loader:
    def _init_(self, xml_path):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

    # -------------------------
    # Teams
    # -------------------------
    def get_teams(self):
        teams = []
        for t in self.root.findall(".//Teams/team"):
            tid = int(t.get("id"))
            name = t.get("name")
            teams.append((tid, name))
        teams.sort(key=lambda x: x[0])
        return teams

    def get_team_ids(self):
        return [tid for tid, _ in self.get_teams()]

    def get_team_names(self):
        return {tid: name for tid, name in self.get_teams()}

    def get_num_teams(self):
        return len(self.get_team_ids())

    # -------------------------
    # Distances
    # -------------------------
    def get_distances(self):
        n = self.get_num_teams()
        D = np.zeros((n, n), dtype=int)
        for d in self.root.findall(".//Distances/distance"):
            i = int(d.get("team1"))
            j = int(d.get("team2"))
            dist = int(d.get("dist"))
            D[i, j] = dist
        return D

    # -------------------------
    # Capacity Constraints
    # -------------------------
    def get_max_consecutive(self):
        max_consec = None
        for ca3 in self.root.findall(".//CapacityConstraints/CA3"):
            try:
                max_consec = int(ca3.get("max"))
            except:
                pass
        if max_consec is None:
            max_consec = 3  # fallback default
        return max_consec