# Active Abstraction Learning (AAL)

This repository implements **Active Abstraction Learning** for Interventionally Consistent Surrogates, extending the work of [NeurIPS 2024](https://arxiv.org/).

The goal is to train a fast surrogate model that is not only observationally accurate but **interventionally consistent** with a complex ground-truth simulator (ABM), using Active Learning to efficiently explore the parameter space.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd active-abstraction-learning
    ```

2.  **Create Environment**:
    ```bash
    conda env create -f environment.yaml
    conda activate active_abstraction
    ```

3.  **Install Package**:
    ```bash
    pip install -e .
    ```

## Usage

The entire experiment pipeline is controlled via **`configuration.yaml`**.

### 1. Configure
Edit `configuration.yaml` to set your experiment parameters:
```yaml
experiment:
  budget: 20           # Number of active learning steps
  seeds: 3             # Number of random seeds to run per method
  initial_samples: 5   # Size of initial random dataset
  epochs: 100          # Training epochs per step
  families: [lodernn]  # Surrogate family: lode or lodernn
  methods: [random, uncertainty_sampling, bald] # Comparison methods
```

### 2. Run
Execute the main experiment script. This will run all configured methods and seeds:
```bash
python experiments/experiment.py
```

### 3. Visualize
The script automatically generates visualizations in the `experiments/` directory:
-   **`final_performance_{family}.png`**: Comparison of Test NLL (Negative Log Likelihood) over time. Lower is better.
-   **`trajectory_grid.png`**: visualization of Surrogate vs Simulator trajectories across diverse scenarios (Outbreak, Suppression, Reinfection).

## Repository Structure

-   **`src/active_abstraction/`**: Core logic.
    -   `loop.py`: `ActiveLearner` class implementing the active learning cycle.
    -   `acquisition.py`: `AcquisitionManager` implementing scoring functions (BALD, Uncertainty).
    -   `models.py`: Surrogate model definitions with MC Dropout support.
-   **`experiments/`**: Experiment orchestration.
    -   `experiment.py`: Main entry point.
    -   `run_aal.py`: Single-run execution script.
    -   `visualize_trajectories.py`: Visualization logic.
-   **`neurips_ics4csm_original/`**: Vendor code from the original paper (Simulator, Training Utils).

## Alignment with NeurIPS Paper
This implementation extends **Algorithm 1** of the original paper by replacing the random Data Collection loop with an **Active Selection** step. See `paper_alignment.md` (if available) for details.
