from simulated_annealing import build_double_round_robin,simulated_annealing,parse_args
from data_loader import Loader
import numpy as np
import argparse

def run_experiments(instance,num_trials):
    XML_PATH = instance

    loader = Loader(XML_PATH)
    n = loader.get_num_teams()
    team_ids = loader.get_team_ids()
    team_names = loader.get_team_names()
    D = loader.get_distances()
    max_consec = loader.get_max_consecutive()

    initial = build_double_round_robin(n)
    results = []

    for i in range(num_trials):
        print(f"\n=== Trial {i+1}/{num_trials} ===")
        best, best_travel, best_viol, best_score = simulated_annealing(
        initial, n, D, max_consec,
        penalty_weight=10000,
        T0=1000.0, cooling=0.99999, iterations=100000,
        )    
        print(f"Trial {i+1}: score={best_score:.2f}, travel={best_travel}, viol={best_viol}")
        results.append(best_score)


    results = np.array(results)
    print("\n=== Summary ===")
    print(f"Mean score: {results.mean():.2f}")
    print(f"Std dev:     {results.std():.2f}")
    print(f"Best:        {results.min():.2f}")
    print(f"Worst:       {results.max():.2f}")
    return results

def main():
    args = parse_args()

    run_experiments(args.xml_path, args.trials)

if __name__ == "__main__":
    main()