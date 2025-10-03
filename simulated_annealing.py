"""
Simple Simulated Annealing solver for the Traveling Tournament Problem (TTP)
"""

from data_loader import Loader


# Get data from XML file
n = Loader.get_num_teams()
team_ids = Loader.get_team_ids
team_names = Loader.get_team_names()
D = Loader.get_distances()
max_consec = Loader.get_max_consecutive()





