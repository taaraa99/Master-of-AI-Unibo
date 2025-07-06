#!/bin/bash

# A simple script to run the unified solver using Docker.

# --- Configuration ---
IMAGE_NAME="solver-app"

# --- Script Logic ---

# 1. Check that the user provided exactly one argument.
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_type>"
    echo "  <model_type>: Choose from 'cp', 'mip', 'sat', or 'smt'"
    exit 1
fi

MODEL_TYPE="$1"

# 2. Check if the Docker image exists. If not, build it.
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "--> Docker image '$IMAGE_NAME' not found. Building from Dockerfile..."
    docker build -t "$IMAGE_NAME" .
    echo "--> Docker image '$IMAGE_NAME' built successfully."
fi

# 3. Print a clear message about what's about to run.
echo "========================================"
echo "  Running Solver"
echo "----------------------------------------"
echo "  Model Type: $MODEL_TYPE"
echo "========================================"
echo "--> Starting Docker container..."

# 4. Execute the Docker container.
#    --rm          : Automatically remove the container when it exits.
#    -v ...        : Mounts local folders into the container so it can read instances and write results.
#    $IMAGE_NAME   : The name of the image to run.
#    $MODEL_TYPE   : The argument passed to our unified_solver.py script.
#    NOTE: Adding an extra '/' before ${PWD} to force correct path interpretation on Windows + Git Bash.
docker run --rm \
    -v "/${PWD}/Instances:/app/Instances" \
    -v "/${PWD}/res:/app/res" \
    "$IMAGE_NAME" "$MODEL_TYPE"

echo ""
echo "========================================"
echo "  Solver execution finished."
echo "  Check the 'res/${MODEL_TYPE^^}' folder for results."
echo "========================================"
