"""
Simple Simulated Annealing solver for the Traveling Tournament Problem (TTP)
"""

from data_loader import Loader

XML_PATH = "instances/NL4.xml"


# Get data from XML file
loader = Loader(XML_PATH)
n = loader.get_num_teams()
team_ids = loader.get_team_ids()
team_names = loader.get_team_names()
D = loader.get_distances()
max_consec = loader.get_max_consecutive()


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


def compute_travel_and_violations(rounds, n, D, max_consecutive):
    """
    rounds: list of rounds; each round is list of matches (home, away)
    returns total_travel_cost, total_violations (sum of consecutive home/away > max_consecutive)
    Travel model:
      - each team starts at home (their own city)
      - if playing away, travel from current location to opponent's city
      - if playing home, stay in home city (no travel)
      - at the end of the season, return to home from current location (if not already home)
    """
    # Build per-team sequence of (opponent, is_home)
    seq = {t: [] for t in range(n)}
    for rnd in rounds:
        # each match is (home, away)
        for h, a in rnd:
            seq[h].append((a, True))
            seq[a].append((h, False))

    total_travel = 0
    violations = 0

    for t in range(n):
        cur_loc = t  # start at home city index
        consecutive_home = 0
        consecutive_away = 0
        for opp, at_home in seq[t]:
            if at_home:
                # at home: no travel, location becomes home
                cur_loc = t
                consecutive_home += 1
                consecutive_away = 0
            else:
                # away: travel from current location to opp
                total_travel += D[cur_loc, opp]
                cur_loc = opp
                consecutive_away += 1
                consecutive_home = 0
            # violations counting
            if consecutive_home > max_consecutive:
                violations += 1
            if consecutive_away > max_consecutive:
                violations += 1
        # return to home at end
        if cur_loc != t:
            total_travel += D[cur_loc, t]

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


def main():
    print(f"Teams: {n}, max consecutive home/away allowed ~ {max_consec}")
    print("Building initial double round-robin schedule...")
    initial = build_double_round_robin(n)
    init_travel, init_viol = compute_travel_and_violations(initial, n, D, max_consec)
    init_score = init_travel + 10000 * init_viol
    print(f"Initial travel cost: {init_travel}, violations: {init_viol}, score: {init_score}\n")
    print_schedule(initial, team_names)


if __name__ == "__main__":
    main()




