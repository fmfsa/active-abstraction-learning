# Active Abstraction Learning (AAL) - Project Overview

## 1. Executive Summary
**Active Abstraction Learning (AAL)** is a framework designed to efficiently learn **causal abstractions** of complex simulation models (such as Agent-Based Models). Instead of training surrogate models on randomly sampled simulation runs (which is computationally expensive and data-inefficient), AAL uses **active learning** to intelligently select the most informative interventions to run.

**Goal**: Minimize the number of expensive ground-truth simulation queries required to train an accurate surrogate model by focusing on regions of the parameter space where the model is uncertain.

## 2. High-Level Architecture
The system operates as a closed feedback loop:

1.  **Surrogate Model**: A differentiable approximation (Neural ODE/RNN) of the complex simulator. It estimates both predictions and *epistemic uncertainty* (via MC Dropout).
2.  **In Silico Screening**: The system generates a large pool of potential interventions (candidate parameters).
3.  **Acquisition Function**: The system evaluates these candidates using the surrogate's uncertainty estimates. It selects the candidates that maximize an information gain metric (e.g., highest uncertainty, committee disagreement).
4.  **Ground Truth Query**: The selected candidates are run in the actual detailed simulator (Spatial SIRS).
5.  **Abstraction Update**: The surrogate model is retrained on the augmented dataset.

## 3. Key Components

### A. The Controller: `ActiveLearner`
**Location**: `src/active_abstraction/loop.py`
The central class that orchestrates the AAL cycle.
- **State**: Maintains the dataset of observed simulation trajectories (`states`, `agg_ts`).
- **Loop (`step`)**:
    1.  Generates a random pool of `candidate_pool` (intervention parameters).
    2.  Calls `AcquisitionManager` to select the best candidates.
    3.  Runs the simulator (`_query_simulator`) for those candidates.
    4.  Retrains the surrogate model (`_train_surrogate`) using the legacy `eurips_ics4csm` training routines.

### B. The Strategist: `AcquisitionManager`
**Location**: `src/active_abstraction/acquisition.py`
Determines *what* to learn next.
- **Method**: Implements strategies like `uncertainty_sampling` (variance of predictions), `entropy`, and `disagreement`.
- **Logic**: It receives a `PredictiveWrapper` (the model) and a pool of candidates. It runs the model multiple times (using MC Dropout) to get a distribution of outputs for each candidate. It then computes a score (e.g., variance) and selects the top-k candidates.
- **Current Status**: The implementations of different metrics needs refinement (currently they all default to similar variance calculations), as noted in the improvement plan.

### C. The Brain: Surrogate Models
**Location**: `src/active_abstraction/models.py`
Differentiable models that mimic the simulator.
- **Architecture**: A composite model consisting of:
    - **Omega (MLP)**: Maps simulation parameters (Alpha, Beta, Gamma) to ODE parameters.
    - **ODE/RNN**: Simulates the dynamics over time.
    - **Emission**: Maps latent states to observed outputs.
- **Uncertainty**: Implements **MC Dropout** (Monte Carlo Dropout).
    - `DropoutMLP` / `DropoutRNN`: Custom layers that keep dropout active.
    - `MCDropoutWrapper`: Forces the model into `train()` mode during inference so that stochastic dropout masks are applied, generating a distribution of predictions.

### D. The Simulator: Spatial SIRS
**Location**: `neurips_ics4csm_original/...` (Legacy Code)
The "Ground Truth" oracle.
- A Spatial SIRS (Susceptible-Infected-Recovered-Susceptible) agent-based model.
- **Input**: Parameters ($\alpha, \beta, \gamma$) and Intervention Time.
- **Output**: Time-series of infection counts.
- **Cost**: Assumed to be expensive, hence the need for AAL.

## 4. Codebase Structure

- **`src/active_abstraction/`**: The core AAL implementation.
    - `loop.py`: Main `ActiveLearner` logic.
    - `acquisition.py`: Acquisition strategies.
    - `models.py`: Surrogate architectures with dropout.
- **`experiments/`**: Scripts to run investigations.
    - `run_aal.py`: The entry point for running a full AAL experiment.
    - `experiment.py`: Config-driven experimental runner.
- **`neurips_ics4csm_original/`**: The frozen legacy codebase providing the base models, simulator, and training utilities.
- **`IMPROVEMENT_PLAN.md`**: A detailed roadmap of features to be implemented (e.g., AT-UCB support, fixing acquisition functions).

## 5. Data Flow

1.  **Configuration**: User sets budget, seeds, and family in `configuration.yaml` or CLI args.
2.  **Initialization**: `ActiveLearner` collects $N$ random samples.
    - Calls `run_spatial_intervention` $\to$ Returns trajectories.
    - Stores in `self.states`.
3.  **Training**: `_train_surrogate` calls `train_epi` (from legacy code) to fit `obs_omega` and `obs_rnn_net` to the data.
4.  **Active Step**:
    - **Pool**: Random vectors $[\alpha, \beta, \gamma, t_{int}]$ generated.
    - **Scoring**: `PredictiveWrapper` runs $M$ passes per candidate.
        - `omega(params)` $\to$ ODE parameters.
        - `ODE(y0, params)` $\to$ Dynamic trajectory.
        - `RNN(trajectory)` $\to$ Emission parameters.
        - **Variance** of these Emission parameters is the *Uncertainty Score*.
    - **Selection**: Top candidate chosen.
5.  **Query**: Simulator run for selected candidate $\to$ Data added to buffer.
6.  **Loop**: Process repeats until budget exhausted.

## 6. Current Status & Future Roadmap
The project is in an active development phase. While the core loop is functional, several capabilities are planned for immediate implementation (as detailed in `IMPROVEMENT_PLAN.md`):
1.  **AT-UCB (Active Testing + UCB)**: Integrating Upper Confidence Bound logic to focus not just on uncertainty, but on identifying specific boundaries in the parameter space.
2.  **Refined Acquisition**: Distinct implementations for Entropy vs. Disagreement vs. Variance Reduction.
3.  **Robustness**: Improved error handling and result logging.
