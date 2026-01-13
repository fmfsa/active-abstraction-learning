# Active Abstraction Learning (AAL) for Causal Abstractions

## Overview
This project implements **Active Abstraction Learning (AAL)**, a framework to efficiently learn causal abstractions of complex simulation models (like Spatial SIRS). Instead of uniformly sampling interventions, AAL uses an **active acquisition strategy** driven by epistemic uncertainty to select the most informative interventions.

This repository extends the work from [neurips_ics4csm](https://github.com/joelnmdyer/neurips_ics4csm).

## Project Structure
- `src/active_abstraction/`: Core AAL implementation (Acquisition, Models, Interaction Loop).
- `experiments/`: Experiment scripts and results.
- `neurips_ics4csm_original/`: The original legacy codebase (untouched).
- `configuration.yaml`: Configuration for experiments.

## Key Components
1.  **Epistemic Uncertainty**: We wrap the surrogate models (MLP, RNN) with **MC Dropout** to estimate uncertainty in the causal structure.
2.  **Acquisition Manager**: Selects interventions based on:
    - **Uncertainty Sampling**: Picking candidates where the model is least confident.
    - **Disagreement**: Picking candidates where committee members disagree.
3.  **Active Loop**: Automates the cycle of *Screening -> Acquisition -> Query (Ground Truth) -> Update*.

## Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    conda env create -f environment.yaml
    conda activate active_abstraction
    ```
3.  Ensure `neurips_ics4csm_original` is installed (handled by environment.yaml), or installed in editable mode manually if needed (`pip install -e neurips_ics4csm_original`).

## Running Experiments
To run the main comparison (Random vs Uncertainty Sampling):

1.  Edit `configuration.yaml` to adjust settings (budget, seeds, etc.).
2.  Run the experiment script:
    ```bash
    python experiments/experiment.py
    ```
3.  Check `experiments/` for plots:
    - `final_performance.png`: Test NLL over time.
    - `uncertainty_landscape.png`: Visualization of the uncertainty in intervention space.

## Reference
The original codebase is located in `neurips_ics4csm_original/` and is provided by [Joel Dyer et al.](https://github.com/joelnmdyer/neurips_ics4csm).
