# IP with Column Generation for TTP and Simmulated Annealing for mirorred-TTP

This repository provides two solvers for the Traveling Tournament Problem (TTP):
an exact integer-programming approach with column generation (via Gurobi) for
double round-robin schedules under separation constraints, and a simulated
annealing heuristic for the mirrored-TTP variant. It includes XML instance
parsers, experiment scripts, and utilities to inspect schedules and travel
distances.

## Setup Instructions

1. Clone the repository

    ```sh
    git clone https://github.com/szelesteya/traveling-tournament-problem-group-project
    cd traveling-tournament-problem-group-project
    ```

2. Install `uv` (Python package manager)

    ```sh
    pip install uv
    ```

3. Create virtual environments and install dependencies

    ```sh
    uv venv
    uv sync
    ```

4. Start Jupyter Notebook

    ```sh
    uv run jupyter notebook
    ```

5. Setting venv as interpreter in IDE

You can set the virtual environment as the interpreter in your IDE.

For instance, in VSCode:
- Open Command Palette (Ctrl+Shift+P)
- Type and select `Python: Select Interpreter`
- Choose the interpreter from the `.venv` folder in your project directory

## Prerequisites

- Python (see version in `pyproject.toml`)
- Gurobi installed and licensed for the exact method
  - Option A: Provide a WLS license via `--gurobi-license`
  - Option B: Use an already-configured Gurobi environment/license

## Quickstart

- Exact method (minimal):

  ```sh
  uv run exact_method.py --instance instances/NL4.xml --gurobi-license gurobi.lic
  ```

- Approximate method (minimal):

  ```sh
  uv run simulated_annealing.py instances/NL12.xml
  ```

## Usage

### Exact method
1. Run the exact solver

    ```sh
    uv run exact_method.py --instance instances/NL8.xml \
      --gurobi-license path/to/gurobi-wls.lic \
      --add-optimal-patterns \
      --patterns-per-team 6 \
      --random-seed 42 \
      --lower-bound 1 \
      --upper-bound 3
    ```

    - The script prints the model summary and writes the model file to `models/model.lp` when an optimal solution is found.

2. Flags

    - `-i, --instance` (string): Path to the XML instance file. Default: `instances/NL4.xml`.
    - `-g, --gurobi-license` (string): Optional path to a Gurobi WLS license file. If omitted, the default Gurobi environment is used.
    - `--add-optimal-patterns` (boolean flag): If present, force-inject known optimal patterns for supported `n`.
    - `--patterns-per-team` (int): Number of sampled patterns per starting team. Default: `6`.
    - `--random-seed` (int): Random seed used for pattern sampling. Default: `42`.
    - `--lower-bound` (int): Minimum length of home blocks between away games. If omitted, uses the instance's SE1 min limit.
    - `--upper-bound` (int): Maximum length of home blocks between away games. If omitted, uses the instance's SE1 max limit.

#### Interpreting results

- Objective value: total travel distance of the constructed double round-robin schedule.
- Schedule: per-round opponent or home slot for each team.
- Travels: per-team distances and totals. On optimal solutions, the `.lp` model is saved to `models/model.lp`.


### Approximate method
1. Generate a schedule with an approximate solution

    ```sh
    uv run simulated_annealing.py instances/NLx.xml
    ```
    Replace NLx with the desired number of teams.

2. Run experiments for a given instance
    
    ```sh
    uv run experiments.py instances/NL8.xml --trials 50
    ```
    Specify the number of trials (in this example, 50).
    The script will return the mean, standard deviation, and the best and worst scores obtained across all trials.

### Data loader and instance files

- `data_loader.py`: Lightweight XML reader used by the approximate method. It exposes helpers like `get_team_ids`, `get_team_names`, `get_distances`, and `get_max_consecutive` to prepare inputs for simulated annealing.

- `instance.py`: Typed parser used by the exact method. `Instance.from_file(path, lb=None, ub=None)` loads teams, symmetric distances, and SE1 separation constraints. If `lb`/`ub` are not provided via flags, the instance limits are used; if provided, they are checked against the instance's limits.

### Instances

- XML benchmarks in `instances/` (e.g., `NL4.xml`, `NL6.xml`, `NL8.xml`, ...)
- Each file defines teams, symmetric distances, and separation constraints used by the solver.

## Troubleshooting

- Gurobi license errors: verify `--gurobi-license` path or environment license setup.
- lxml import issues: ensure dependencies installed via `uv` and system XML libs are available.
- Large instances slow to solve: start with `NL4`/`NL6`, then scale up.

## Reproducibility

- Exact method: set `--random-seed` and keep `--patterns-per-team` fixed to reproduce sampled patterns.
- Experiments: record `--trials` and any seeds used in your runner.

## Performance tips for exact method

- Increase `--patterns-per-team` gradually; higher values expand search but increase time.
- Validate environment on small instances before scaling up.
- Inspect `models/model.lp` when diagnosing model structure.
