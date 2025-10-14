"""
Simple Simulated Annealing solver for the Traveling Tournament Problem (TTP)
"""
import argparse
from data_loader import Loader
import random
import copy
import math

# -------------------------
# Parse command-line arguments
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Simulated Annealing TTP solver")
    parser.add_argument(
        "xml_path",
        help="Path to the XML instance file, e.g., instances/NL4.xml"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of SA runs to perform (default: 5)"
    )
    return parser.parse_args()

def round_robin_pairings(n):
    """
    Return list of rounds; each round is list of pairs (a,b) (undirected).
    Uses circle method; expects even n. If n odd, normally add a dummy team.
    """
    assert n % 2 == 0, "Round-robin generator expects even number of teams"
    teams = list(range(n))
    rounds = []
    for r in range(n - 1): 
        # In a single round-robin each team must play every other team once; 
        # each team has n-1 opponents, and they play one opponent per round 
        # → n-1 rounds needed
        pairs = []
        for i in range(n // 2):
            # We pair the first element with the last, the second with the second-last, etc.
            # This yields n/2 disjoint pairs so every team appears exactly once that round
            a = teams[i]
            b = teams[n - 1 - i]
            pairs.append((a, b))
        # Rotate except the first element
        # Moves the last element into position 1 and shifts the middle block right by one
        teams = [teams[0]] + [teams[-1]] + teams[1:-1] 
        rounds.append(pairs)
    return rounds  # (n-1) rounds, each with n/2 matches


def build_double_round_robin(n):
    """
    Construct a double round-robin schedule as list of rounds.
    For the second half we mirror opponents and swap home/away.
    Representation: rounds is a list; each round is list of matches (home, away)
    """
    first_half = round_robin_pairings(n)  # (n-1) rounds
    rounds = []
    # initial home/away assignment: alternate to try some balance
    for r, pairs in enumerate(first_half):
        round_matches = []
        for a, b in pairs:
            # heuristically assign home by parity of (round + min(a,b))
            if ((r + min(a, b)) % 2) == 0:
                # Because rounds increment and min(a, b) changes depending on the pair, 
                # the result flips between even/odd over the season.
                # This prevents one team from always being home or always away in the first half
                round_matches.append((a, b))
            else:
                round_matches.append((b, a))
        rounds.append(round_matches)
    # second half: mirror opponents and flip home/away
    for r, pairs in enumerate(first_half):
        round_matches = []
        for a, b in pairs:
            # find who was home in corresponding first half round
            if ((r + min(a, b)) % 2) == 0:
                # first half had (a,b) as (home, away)
                round_matches.append((b, a))
            else:
                round_matches.append((a, b))
        rounds.append(round_matches)
    return rounds  # length 2*(n-1)


def compute_travel_and_violations(rounds, n, D, max_consecutive, count_repeat=True, check_mirror = True):
    """
    rounds: list of rounds; each round is list of matches (home, away)
    returns total_travel_cost, total_violations (sum of consecutive home/away > max_consecutive
    plus repeat-match violations if count_repeat=True)
    """
    # Build per-team sequence of (opponent, is_home)
    seq = {t: [] for t in range(n)}
    for rnd in rounds:
        for h, a in rnd:
            seq[h].append((a, True))
            seq[a].append((h, False))

    total_travel = 0
    violations = 0
    mirror_vio = 0

    # detect repeating matches in consecutive rounds (A vs B in round r and again in r+1)
    if count_repeat:
        pairs_per_round = []
        for rnd in rounds:
            # --- frozenset prevents A vs B and B vs A as being seen differently
            pairs_per_round.append(set(frozenset((h, a)) for h, a in rnd))
        repeat_violations = 0
        for r in range(len(pairs_per_round) - 1):
            repeats = pairs_per_round[r].intersection(pairs_per_round[r + 1])
            repeat_violations += len(repeats)
        violations += repeat_violations
    else:
        repeat_violations = 0

    # check that second half mirrors first half with flipped home/away
    if check_mirror:
        m = len(rounds) // 2
        if len(rounds) != 2 * m:
            # odd number of rounds -> can't check strict mirror, skip or handle differently
            pass
        else:
            # We'll compare round r with round r+m: expecting reversed ordered pairs
            for r in range(m):
                # build map of unordered->ordered for round r for quick lookup
                first = {(h, a): (h, a) for (h, a) in rounds[r]}
                # build set of ordered tuples in the mirrored round
                second_ordered = set(rounds[r + m])
                # For each ordered pair in first, we expect reversed pair in second_ordered
                for (h, a) in rounds[r]:
                    if (a, h) not in second_ordered:
                        # if the reversed ordered pair is missing, count as a non-mirror violation
                        violations += 1
                        mirror_vio += 1



    # Per-team travel + consecutive home/away violations (existing behavior)
    for t in range(n):
        cur_loc = t  # start at home city index
        consecutive_home = 0
        consecutive_away = 0
        for opp, at_home in seq[t]:
            if at_home:
                cur_loc = t
                consecutive_home += 1
                consecutive_away = 0
            else:
                total_travel += D[cur_loc, opp]
                cur_loc = opp
                consecutive_away += 1
                consecutive_home = 0
            # violations counting (per-game style)
            if consecutive_home > max_consecutive:
                violations += 1
            if consecutive_away > max_consecutive:
                violations += 1
        # return to home at end
        if cur_loc != t:
            total_travel += D[cur_loc, t]

    # Optionally, you can return repeat_violations separately if you want more details:
    # return total_travel, violations, repeat_violations
    return total_travel, violations

# helper function
def schedule_to_team_sequences(rounds, n):
    seq = {t: [] for t in range(n)}
    for rnd in rounds:
        for h, a in rnd:
            seq[h].append((a, True))
            seq[a].append((h, False))
    return seq

# printer
def print_schedule(rounds, team_names):
    print("Schedule (round -> matches):")
    for r, rnd in enumerate(rounds):
        matches_str = ", ".join(f"{team_names[h]}(H) vs {team_names[a]}(A)" for h,a in rnd)
        print(f" R{r+1:02d}: {matches_str}")
        
        
# By trying many neighbors, the algorithm explores the search space and hopefully 
# finds one with lower travel distance or fewer violations
def random_neighbor(rounds):
    """
    Return a neighbor schedule by applying one of several moves:
     - swap two matches' opponents between rounds (swap opponent partner)
     - swap home/away of a random match
     - swap entire rounds
    This implementation tries to keep validity (no self match) but does not always guarantee feasibility.
    We'll repair trivial self-matches (shouldn't occur with these moves).
    """
    s = copy.deepcopy(rounds)
    n_rounds = len(s)
    # choose move type
    r = random.random()
    if r < 0.4:
        # swap two matches between two rounds: pick two rounds and two matches and swap the entire match entries
        r1, r2 = random.sample(range(n_rounds), 2)
        m1 = random.randrange(len(s[r1]))
        m2 = random.randrange(len(s[r2]))
        s[r1][m1], s[r2][m2] = s[r2][m2], s[r1][m1]
        # after swap, ensure no team plays twice in same round -> simple check & repair by swapping back if violation
        if not round_is_valid(s[r1]) or not round_is_valid(s[r2]):
            s[r1][m1], s[r2][m2] = s[r2][m2], s[r1][m1]  # revert
    elif r < 0.75:
        # flip home/away in a random round/match
        rr = random.randrange(n_rounds)
        mm = random.randrange(len(s[rr]))
        h, a = s[rr][mm]
        s[rr][mm] = (a, h)
        # if this created duplicate team in the round, revert
        if not round_is_valid(s[rr]):
            s[rr][mm] = (h, a)
    else:
        # swap two whole rounds
        r1, r2 = random.sample(range(n_rounds), 2)
        s[r1], s[r2] = s[r2], s[r1]
    return s

def round_is_valid(round_matches):
    # ensure teams in that round are unique (no team plays twice in same round) and no self matches
    seen = set()
    for h, a in round_matches:
        if h == a:
            return False
        if h in seen or a in seen:
            return False
        seen.add(h)
        seen.add(a)
    return True


# -------------------------
# Simulated Annealing
# -------------------------
def simulated_annealing(initial, n, D, max_consec,
                        penalty_weight=10000,
                        T0=1000.0, cooling=0.995,
                        iterations=20000):
    best = copy.deepcopy(initial)
    best_travel, best_viol = compute_travel_and_violations(best, n, D, max_consec, check_mirror = True)
    best_score = best_travel + penalty_weight * best_viol

    cur = copy.deepcopy(best)
    cur_travel, cur_viol = best_travel, best_viol
    cur_score = best_score

    T = T0
    best_improvements = 0 
    
    for it in range(iterations):
        cand = random_neighbor(cur)
        cand_travel, cand_viol = compute_travel_and_violations(cand, n, D, max_consec, check_mirror= True)
        # cand_score = cand_travel + penalty_weight * cand_viol
        cand_score = math.sqrt(cand_travel**2 + (penalty_weight * cand_viol)**2)


        delta = cand_score - cur_score
        accept = False
        if delta < 0:
            accept = True
        else:
            if random.random() < math.exp(-delta / max(1e-9, T)):
                accept = True

        if accept:
            cur = cand
            cur_travel, cur_viol = cand_travel, cand_viol
            cur_score = cand_score
            if cur_score < best_score and cur_viol == 0:
                best = copy.deepcopy(cur)
                best_travel, best_viol = cur_travel, cur_viol
                best_score = cur_score
                best_improvements += 1


        # cooling
        T *= cooling

    print(f"Number of times best improved: {best_improvements}")
    # final evaluation of best (might be infeasible if none feasible found)
    return best, best_travel, best_viol, best_score


def main():
    args = parse_args()
    XML_PATH = args.xml_path

    loader = Loader(XML_PATH)
    n = loader.get_num_teams()
    team_ids = loader.get_team_ids()
    team_names = loader.get_team_names()
    D = loader.get_distances()
    max_consec = loader.get_max_consecutive()
    
    print(f"Teams: {n}, max consecutive home/away allowed ~ {max_consec}")
    print("Building initial double round-robin schedule...")
    initial = build_double_round_robin(n)
    init_travel, init_viol = compute_travel_and_violations(initial, n, D, max_consec, check_mirror=True)
    init_score = init_travel + 10000 * init_viol
    print(f"Initial travel cost: {init_travel}, violations: {init_viol}, score: {init_score}\n")
    print_schedule(initial, team_names)
    print("\nRunning Simulated Annealing ... (this may take a little time)\n")

    best, best_travel, best_viol, best_score = simulated_annealing(
        initial, n, D, max_consec,
        penalty_weight=10000,
        T0=1000.0, cooling=0.99999, iterations=100000,
    )

    print("\n--- RESULT ---")
    print(f"Best travel cost: {best_travel}; violations: {best_viol}; score: {best_score}")
    print_schedule(best, team_names)
    if best_viol == 0:
        print("Feasible solution found!")
    else:
        print(f"Infeasible — {best_viol} violations remain.")


if __name__ == "__main__":
    main()




