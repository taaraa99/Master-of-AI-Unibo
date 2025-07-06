# Unified Optimization Solver

This project provides a unified interface to run various optimization models (CP, MIP, SAT, SMT) on a common set of problem instances. It is designed to be extensible and easy to use, with support for Docker to ensure a consistent and portable environment.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python**: Version 3.10 or higher.
* **Docker Desktop**: Required for building and running the containerized version of the solver.
* **Git Bash** (for Windows users): Required to run the `.sh` helper script. Can be installed with [Git for Windows](https://git-scm.com/download/win).

## Project Structure

The project is organized as follows:

```
.
├── Instances/      # Contains all the .dat instance files.
├── models/         # Contains the Python source code for each solver model.
├── res/            # The output directory where results are saved.
├── unified_solver.py # The main script that orchestrates the solving process.
├── Dockerfile      # Recipe for building the Docker image.
├── requirements.txt# Python package dependencies.
└── run_solver.sh   # A helper script to simplify running the solver with Docker.
```

## How to Run the Solver

There are three ways to run the solver, from the easiest to the most manual.

### Method 1: Using the `run_solver.sh` Helper Script (Easiest Method)

This is the recommended method for most users. The helper script automates the process of building the Docker image (if needed) and running the container with the correct parameters.

**1. Make the script executable:**
(You only need to do this once). Open a **Git Bash** terminal and run:
```bash
chmod +x run_solver.sh
```

**2. Run the solver:**
Execute the script with the desired model type. Here are the commands for all available models:

* **To run the Constraint Programming (CP) solver:**
    ```bash
    ./run_solver.sh cp
    ```
* **To run the Mixed-Integer Programming (MIP) solver:**
    ```bash
    ./run_solver.sh mip
    ```
* **To run the Satisfiability (SAT) solver:**
    ```bash
    ./run_solver.sh sat
    ```
* **To run the Satisfiability Modulo Theories (SMT) solver:**
    ```bash
    ./run_solver.sh smt
    ```

### Method 2: Using Docker Manually

This method gives you more control and is useful if you want to understand the underlying Docker commands.

**1. Build the Docker Image:**
From the project's root directory, run the build command. This will create an image named `solver-app`.
```bash
docker build -t solver-app .
```

**2. Run the Container:**
Execute the `docker run` command, making sure to mount the `Instances` and `res` directories as volumes. This allows the container to read your instance files and write the results back to your machine.

Replace `<model_type>` with `cp`, `mip`, `sat`, or `smt`.

* **On Windows (in Git Bash):**
    ```bash
    # Example for CP
    docker run --rm -v "${PWD}/Instances:/app/Instances" -v "${PWD}/res:/app/res" solver-app cp
    ```
* **On macOS or Linux:**
    ```bash
    # Example for MIP
    docker run --rm -v "$(pwd)/Instances:/app/Instances" -v "$(pwd)/res:/app/res" solver-app mip
    ```

### Method 3: Direct Execution with Python (Manual Setup)

This method runs the script directly on your machine without Docker. It requires you to have all the necessary solvers (like MiniZinc) installed and configured on your system, which can be complex.

**1. Create a Virtual Environment:**
It's highly recommended to use a virtual environment to manage dependencies.
```bash
# Create the environment
python -m venv venv

# Activate the environment (on Windows PowerShell)
.\venv\Scripts\activate

# Activate the environment (on macOS/Linux or Windows Git Bash)
source venv/bin/activate
```

**2. Install Dependencies:**
Install all the required Python packages into your active virtual environment.
```bash
pip install -r requirements.txt
```

**3. Run the Script:**
Execute the `unified_solver.py` script directly, passing the desired model type as an argument.

* **To run the Constraint Programming (CP) solver:**
    ```bash
    python unified_solver.py cp
    ```
* **To run the Mixed-Integer Programming (MIP) solver:**
    ```bash
    python unified_solver.py mip
    ```
* **To run the Satisfiability (SAT) solver:**
    ```bash
    python unified_solver.py sat
    ```
* **To run the Satisfiability Modulo Theories (SMT) solver:**
    ```bash
    python unified_solver.py smt
    ```

## Viewing Results

Regardless of the method used, the output JSON files will be saved in the `res/` directory, organized into subfolders based on the model type.

For example, after running `./run_solver.sh cp`, the results will be located in:

`res/CP/01.json`
`res/CP/02.json`
...and so on.
