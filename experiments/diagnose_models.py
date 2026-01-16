import os
import torch
import numpy as np
import yaml
from neurips_ics4csm.utils import run_spatial_intervention, generate_dists, instantiate_model
from neurips_ics4csm.utils import create_instantiate_emission
import neurips_ics4csm.utils as utils_module
from neurips_ics4csm.models import sirs_spatial
utils_module.sirs_spatial = sirs_spatial

def load_checkpoint(experiment_dir, family, seed, step, config):
    from active_abstraction.models import generate_dropout_networks
    rnn_net, omega = generate_dropout_networks(kind=family, seed=42, dropout_prob=config['model']['dropout_prob'])
    
    run_dir = os.path.join(experiment_dir, f"{family}_uncertainty_sampling_seed{seed}")
    model_path = os.path.join(run_dir, f"model_size_{step}.pt")
    
    if os.path.exists(model_path):
        print(f"Loading {family} Size {step} from {model_path}")
        state = torch.load(model_path)
        omega.load_state_dict(state['omega'])
        if state['rnn'] is not None and family != 'lode':
             rnn_net.load_state_dict(state['rnn'])
    else:
        print(f"Check point not found: {model_path}")
        return None, None
        
    return omega, rnn_net

def diagnose():
    with open("configuration.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    families = ["lode", "lodernn"]
    step = 1
    
    T = 50
    L = 50 
    N = L**2
    model = instantiate_model(torch.linspace(0, T, T+1))
    alpha, beta, gamma = 0.9, 0.1, 0.1
    intervention = 4
    y0 = torch.tensor([0.9, 0.1, 0.0]).double()
    
    for family in families:
        print(f"\n--- Diagnosing {family} at Size {step} ---")
        omega, rnn_net = load_checkpoint("results_experiment_final", family, 0, step, config)
        
        if omega is None:
            continue
            
        omega.train()
        if hasattr(rnn_net, 'train'): rnn_net.train()
        
        instantiate_emission = create_instantiate_emission(N, family)
        
        predictions = []
        n_mc = 20
        
        print("Running MC Dropout...")
        for i in range(n_mc):
            try:
                e_dists = generate_dists(instantiate_emission, omega, torch.tensor([alpha, beta, gamma]), model, y0, intervention, rnn_net)
                traj = torch.stack([d.mean for d in e_dists]) / N 
                predictions.append(traj.detach())
            except Exception as e:
                print(f"Error in MC sample {i}: {e}")
                
        if len(predictions) > 0:
            predictions = torch.stack(predictions)
            print(f"Predictions Shape: {predictions.shape}") # [20, 51, 3]
            
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
            
            print(f"Mean Pred (First 5 steps):\n{mean_pred[:5]}")
            print(f"Std Pred (First 5 steps):\n{std_pred[:5]}")
            
            if torch.isnan(mean_pred).any():
                print("!!! WARNING: NaNs detected in Mean !!!")
            if torch.isnan(std_pred).any():
                print("!!! WARNING: NaNs detected in Std !!!")
                
            print(f"Max Std: {std_pred.max().item()}")
            print(f"Min Mean: {mean_pred.min().item()}, Max Mean: {mean_pred.max().item()}")
        else:
            print("No predictions generated.")

if __name__ == "__main__":
    diagnose()
