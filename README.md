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

3. Create and activate a virtual environment

    ```sh
    uv venv
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

## Run commands
1. Generate a schedule with an approximate solution

    ```sh
    uv run .\simulated_annealing.py .\instances\NLx.xml
    ```
    Replace NLx with the desired number of teams.

2. Run experiments for a given instance
    
    ```sh
    uv run .\experiments.py .\instances\NL8.xml --trials 50
    ```
    Specify the number of trials (in this example, 50).
    The script will return the mean, standard deviation, and the best and worst scores obtained across all trials.
