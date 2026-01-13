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

def load_surrogate(experiment_dir, config):
    # Initialize Model components
    # We must use generate_dropout_networks because the saved model contains DropoutMLP/RNN
    from active_abstraction.models import generate_dropout_networks
    rnn_net, omega = generate_dropout_networks(kind=config['model']['family'], seed=42, dropout_prob=config['model']['dropout_prob'])
    
    # Load Weights
    # We look for the best run in the experiment dir (e.g. results_experiment_final/uncertainty_sampling_seed0)
    # The experiment_dir passed might be the base dir. Let's find a valid run.
    
    model_path = None
    if os.path.exists(os.path.join(experiment_dir, "final_model.pt")):
         model_path = os.path.join(experiment_dir, "final_model.pt")
    else:
        # Search for a seed subdir
        for root, dirs, files in os.walk(experiment_dir):
            if "final_model.pt" in files:
                model_path = os.path.join(root, "final_model.pt")
                break
    
    if model_path:
        print(f"Loading trained model from {model_path}")
        state = torch.load(model_path)
        omega.load_state_dict(state['omega'])
        if state['rnn'] is not None:
            rnn_net.load_state_dict(state['rnn'])
    else:
        print("WARNING: No 'final_model.pt' found. Using untrained model!")
    
    T = 50
    model = instantiate_model(torch.linspace(0, T, T+1))
    
    return omega, rnn_net, model

def plot_trajectory_grid(omega, rnn_net, model, scenarios, config):
    n_scenarios = len(scenarios)
    cols = 3
    rows = (n_scenarios + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    T = 50
    L = 50 
    N = L**2
    from neurips_ics4csm.utils import create_instantiate_emission
    instantiate_emission = create_instantiate_emission(N, config['model']['family'])
    t_steps = np.arange(T+1)
    labels = ['Susceptible', 'Infected', 'Recovered']
    colors = ['blue', 'red', 'green']
    
    # Enable dropout
    omega.train() 
    rnn_net.train()
    
    for idx, (name, params, intervention) in enumerate(scenarios):
        ax = axes[idx]
        print(f"Plotting scenario: {name}...")
        
        alpha, beta, gamma = params
        
        # 1. Ground Truth
        init_state, x_gt = run_spatial_intervention(torch.tensor([alpha, beta, gamma]), intervention, 0.1, T, L)
        
        # 2. Surrogate
        n_mc = 20
        predictions = []
        y0 = torch.tensor([0.9, 0.1, 0.0]).double()
        
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
            ax.plot(t_steps, mean_pred[:, dim], '-', color=colors[dim], label=f'{labels[dim]}' if idx==0 else "")
            # Uncertainty
            lower = mean_pred[:, dim] - 2 * std_pred[:, dim]
            upper = mean_pred[:, dim] + 2 * std_pred[:, dim]
            ax.fill_between(t_steps, lower, upper, color=colors[dim], alpha=0.2)
            
        ax.set_title(f"{name}\n(i={intervention}, $\\alpha$={alpha}, $\\beta$={beta})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Fraction")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
            
    # Add custom legend for line styles
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', linestyle='-'),
                    Line2D([0], [0], color='black', linestyle='--')]
    fig.legend(custom_lines, ['Surrogate (Mean)', 'Ground Truth'], loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.abspath("experiments/trajectory_grid.png")
    plt.savefig(save_path)
    print(f"Trajectory grid saved to {save_path}")

def main():
    config = load_config()
    
    # Define interesting scenarios
    # Params: alpha (inf), beta (rec), gamma (resus)
    # High: alpha=0.9, beta=0.1
    # Med:  alpha=0.6, beta=0.2
    # Low:  alpha=0.3, beta=0.3
    # Interventions: None=0, Early=1, Late=4
    
    p_high = [0.9, 0.1, 0.1]
    p_med  = [0.6, 0.2, 0.1]
    p_low  = [0.3, 0.3, 0.1]
    
    i_none = 0
    i_early = 1
    i_late = 4
    
    scenarios = [
        ("High Inf, No Int", p_high, i_none),
        ("High Inf, Early",  p_high, i_early),
        ("High Inf, Late",   p_high, i_late),
        ("Med Inf, No Int",  p_med, i_none),
        ("Med Inf, Early",   p_med, i_early),
        ("Med Inf, Late",    p_med, i_late),
        ("Low Inf, No Int",  p_low, i_none),
        ("Low Inf, Early",   p_low, i_early),
        ("Low Inf, Late",    p_low, i_late)
    ]
    
    # Load Model
    omega, rnn_net, model = load_surrogate("results_experiment_final", config)
    
    plot_trajectory_grid(omega, rnn_net, model, scenarios, config)

if __name__ == "__main__":
    main()
