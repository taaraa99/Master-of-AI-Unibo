# Start from the official MiniZinc base image to get the toolchain.
FROM minizinc/minizinc:latest

# Set the working directory inside the container.
WORKDIR /app

# --- Install Python and Pip ---
# The base image is minimal. We need to explicitly install Python and Pip
# ourselves. This guarantees that the `python3` and `pip3` commands
# will be available for the next steps.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python Dependencies ---
# Now that pip3 is installed, this command will work.
# We add --break-system-packages to override the "externally-managed-environment" error.
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# --- Copy Project Files ---
# Copy your main script and the 'models' directory into the container.
COPY unified_solver.py .
COPY models/ ./models/

# --- Configure Container Execution ---
# Use 'python3' since that's what we explicitly installed.
ENTRYPOINT ["python3", "unified_solver.py"]
CMD ["cp"]