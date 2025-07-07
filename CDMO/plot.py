import json
import matplotlib.pyplot as plt
from pathlib import Path
import math # Import math for math.isnan

# Define the directory where your SMT results are stored
RESULTS_DIR = Path("res") / "SMT"

# Define the configurations (search strategies) to plot
# These keys must match the keys used in your JSON result files
CONFIGURATIONS = [
    "smt_linear",
    "smt_binary",
    "smt_z3",
    "smt_binary_knn6",
    "smt_lns_knn6"
]

def plot_run_times():
    """
    Reads run times from SMT model result JSON files and plots them.
    """
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory '{RESULTS_DIR}' not found.")
        print("Please ensure your SMT model script has run and generated results in this directory.")
        return

    # Dictionary to store run times for each configuration
    # Key: configuration name (e.g., 'smt_linear')
    # Value: List of run times for that configuration across instances
    run_times_data = {config: [] for config in CONFIGURATIONS}
    
    # List to store instance IDs (for x-axis)
    instance_ids = []

    # Get all JSON files in the results directory, sorted by name (instance ID)
    # The key function handles both '1.json' and '01.json' formats
    json_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: int(f.stem) if f.stem.isdigit() else float('inf'))

    if not json_files:
        print(f"No JSON result files found in '{RESULTS_DIR}'.")
        return

    print(f"Found {len(json_files)} result files. Processing...")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract instance ID from filename (e.g., '01.json' -> 1)
                instance_id = int(file_path.stem)
                instance_ids.append(instance_id)

                for config in CONFIGURATIONS:
                    # Check if the configuration exists in the current JSON data
                    if config in data and "time" in data[config]:
                        run_times_data[config].append(data[config]["time"])
                    else:
                        # If a config is missing for an instance, append a placeholder (e.g., NaN)
                        # This ensures all lists have the same length for plotting
                        run_times_data[config].append(float('nan')) # Not a Number
                        print(f"Warning: '{config}' or its 'time' data not found in {file_path.name}. Using NaN.")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path.name}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path.name}: {e}. Skipping.")

    # Check if we actually collected any data
    # Ensure that instance_ids is not empty and at least one config has non-NaN data
    if not instance_ids or all(all(math.isnan(t) for t in times) for times in run_times_data.values()):
        print("No valid run time data collected for plotting.")
        return

    # --- Plotting ---
    plt.figure(figsize=(10, 6)) # Set figure size for better readability

    for config in CONFIGURATIONS:
        # Plot only if there's actual data for this configuration
        if any(not math.isnan(t) for t in run_times_data[config]):
            plt.plot(instance_ids, run_times_data[config], marker='o', label=config)

    plt.xlabel("Instance ID")
    plt.ylabel("Run Time (seconds)")
    plt.title("SMT Model Run Time Comparison Across Configurations")
    plt.yscale("log") # Use logarithmic scale for Y-axis as seen in the example plot
    plt.xticks(instance_ids) # Ensure x-ticks are at each instance ID
    plt.grid(True, which="both", ls="-", alpha=0.7) # Add grid for better readability
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

if __name__ == "__main__":
    plot_run_times()