## Setup Instructions

1. Clone the repository

    ```sh
    git clone https://github.com/<your-username>/traveling-tournament-problem-group-project.git
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