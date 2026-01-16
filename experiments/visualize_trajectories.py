import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import yaml

# Add legacy code path (removed as package is installed)
# sys.path.append(os.path.abspath("neurips_ics4csm_original"))

from neurips_ics4csm.utils import run_spatial_intervention, generate_dists, instantiate_model
from active_abstraction.models import MCDropoutWrapper
import neurips_ics4csm.utils as utils_module
from neurips_ics4csm.models import sirs_spatial
utils_module.sirs_spatial = sirs_spatial

def load_config():
    with open("configuration.yaml", "r") as f:
        return yaml.safe_load(f)

def load_surrogate(experiment_dir, family, config):
    # Initialize Model components
    from active_abstraction.models import generate_dropout_networks
    rnn_net, omega = generate_dropout_networks(kind=family, seed=42, dropout_prob=config['model']['dropout_prob'])
    
    # Load Weights
    # Look for {family}_uncertainty_sampling_seed0
    # Fallback to any seed if seed0 missing, or random/other method if needed.
    
    candidate_dir = os.path.join(experiment_dir, f"{family}_uncertainty_sampling_seed0")
    model_path = os.path.join(candidate_dir, "final_model.pt")
    
    if not os.path.exists(model_path):
        # Try finding ANY valid model for this family
        print(f"Warning: {model_path} not found. Searching for any {family} model...")
        for root, dirs, files in os.walk(experiment_dir):
            if f"{family}_" in root and "final_model.pt" in files:
                model_path = os.path.join(root, "final_model.pt")
                break
    
    if os.path.exists(model_path):
        print(f"Loading trained {family} model from {model_path}")
        state = torch.load(model_path)
        omega.load_state_dict(state['omega'])
        if state['rnn'] is not None:
            rnn_net.load_state_dict(state['rnn'])
    else:
        print(f"WARNING: No trained model found for {family}. Using initialized weights!")
    
    T = 50
    model = instantiate_model(torch.linspace(0, T, T+1))
    
    return omega, rnn_net, model

def plot_trajectory_grid(families, scenarios, experiment_dir, config):
    rows = len(families)
    cols = len(scenarios)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
    if rows == 1: axes = axes.reshape(1, -1)
    if cols == 1: axes = axes.reshape(-1, 1)
    
    T = 50
    L = 50 
    N = L**2
    from neurips_ics4csm.utils import create_instantiate_emission
    
    t_steps = np.arange(T+1)
    labels = ['Susceptible', 'Infected', 'Recovered']
    colors = ['blue', 'red', 'green']
    
    for i, family in enumerate(families):
        omega, rnn_net, model = load_surrogate(experiment_dir, family, config)
        instantiate_emission = create_instantiate_emission(N, family)
        
        # Enable dropout
        omega.train() 
        rnn_net.train()
        
        for j, (name, params, intervention) in enumerate(scenarios):
            ax = axes[i, j]
            if i == 0:
                print(f"Plotting Ground Truth for scenario: {name}...")
            
            alpha, beta, gamma = params
            
            # Get IC from config
            ic_val = config['experiment'].get('initial_condition', {}).get('value', 0.01)
            
            # 1. Ground Truth (only calculate once per col ideally, but cheap enough)
            init_state, x_gt = run_spatial_intervention(torch.tensor([alpha, beta, gamma]), intervention, ic_val, T, L)
            
            # 2. Surrogate Prediction
            n_mc = 20
            predictions = []
            
            # y0 must match ic_val. y0 is [S, I, R]. R=0.
            # I = ic_val. S = 1 - ic_val.
            y0 = torch.tensor([1.0 - ic_val, ic_val, 0.0]).double()
            
            for _ in range(n_mc):
                e_dists = generate_dists(instantiate_emission, omega, torch.tensor([alpha, beta, gamma]), model, y0, intervention, rnn_net)
                traj = torch.stack([d.mean for d in e_dists]) / N 
                predictions.append(traj.detach())
                
            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
            
            # 3. Plot
            for dim in range(3):
                # GT
                ax.plot(t_steps, x_gt[:, dim], '--', color=colors[dim], alpha=0.6)
                # Surrogate
                ax.plot(t_steps, mean_pred[:, dim], '-', color=colors[dim], label=f'{labels[dim]}' if (i==0 and j==0) else "")
                # Uncertainty
                lower = mean_pred[:, dim] - 2 * std_pred[:, dim]
                upper = mean_pred[:, dim] + 2 * std_pred[:, dim]
                ax.fill_between(t_steps, lower, upper, color=colors[dim], alpha=0.2)
                
            if i == 0:
                ax.set_title(name)
            if j == 0:
                ax.set_ylabel(f"{family.upper()}\nFraction")
            
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend()
            
    # Add custom legend for line styles
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', linestyle='-'),
                    Line2D([0], [0], color='black', linestyle='--')]
    fig.legend(custom_lines, ['Surrogate (Mean)', 'Ground Truth'], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = os.path.abspath("experiments/trajectory_grid.png")
    plt.savefig(save_path)
    print(f"Trajectory grid saved to {save_path}")

def main():
    config = load_config()
    
    # Define diverse parameters
    # [alpha, beta, gamma]
    p_outbreak = [0.9, 0.1, 0.0]  # Fast spread, no reinfection
    p_suppress = [0.2, 0.5, 0.0]  # Low spread, moderate recovery
    p_reinfect = [0.6, 0.1, 0.3]  # Moderate spread, high reinfection
    
    i_none = 0 # No intervention
    i_lock = 2 # Intervention ~ t=10
    
    scenarios = [
        ("Outbreak (No Int)",    p_outbreak, i_none),
        ("Outbreak (Lockdown)",  p_outbreak, i_lock),
        ("Suppression (No Int)", p_suppress, i_none),
        ("Reinfection (Lockdown)", p_reinfect, i_lock),
    ]
    
    # Rows: LODERNN, LODE, RNN
    families = ["lodernn"]
    
    plot_trajectory_grid(families, scenarios, "results_experiment_final", config)

if __name__ == "__main__":
    main()
