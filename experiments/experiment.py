import os
import yaml
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.decomposition import PCA
import torch

def load_config():
    with open("configuration.yaml", "r") as f:
        return yaml.safe_load(f)

def run_single_seed(method, seed, config, dirname):
    print(f"--- Running {method} (Seed {seed}) ---")
    
    cmd = [
        "python", "experiments/run_aal.py",
        "--acquisition_method", method,
        "--dirname", dirname,
        "--seed", str(seed),
        "--family", config['model']['family'],
        "--dropout_prob", str(config['model']['dropout_prob']),
        "--budget", str(config['experiment']['budget']),
        "--initial_samples", str(config['experiment']['initial_samples']),
        "--n_acquire", str(config['experiment']['n_acquire']),
        "--pool_size", str(config['acquisition']['pool_size']),
        "--mc_samples", str(config['acquisition']['mc_samples']),
        "--epochs", "100" 
    ]
    
    if config['simulation'].get('fixed_params'):
        # Convert list [0.8, 0.1, 0.1] to string "0.8,0.1,0.1"
        fp_str = ",".join(map(str, config['simulation']['fixed_params']))
        cmd.extend(["--fixed_params", fp_str])
    subprocess.check_call(cmd)

def aggregate_results(base_exp_dir, method, seeds):
    all_nlls = []
    steps = None
    
    for seed in range(seeds):
        path = os.path.join(base_exp_dir, f"{method}_seed{seed}", "metrics.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                all_nlls.append(data["test_nll"])
                if steps is None:
                    steps = data["dataset_size"]
    
    if not all_nlls:
        return None, None, None

    arr = np.array(all_nlls)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return steps, mean, std

def plot_performance(methods_data, save_path):
    plt.figure(figsize=(10, 6))
    
    for method, (steps, mean, std) in methods_data.items():
        plt.plot(steps, mean, marker='o', label=method)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
        
    plt.xlabel("Dataset Size")
    plt.ylabel("Test NLL")
    plt.title("AAL Performance (Mean ± Std)")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Performance plot saved to {save_path}")

def visualize_uncertainty_space(learner, config):
    # This requires access to the learner instance or re-creating it.
    # We will simulate this by mocking the pool generation and scoring,
    # OR better, we simply run a short script that loads the model and visualizes.
    # For now, let's just use random data to demonstrate the concept if we can't easily load the model.
    # Actually, we can just instantiate ActiveLearner in this script if we set imports right.
    
    from active_abstraction.loop import ActiveLearner
    from argparse import Namespace
    
    args = Namespace(
        family=config['model']['family'],
        dropout_prob=config['model']['dropout_prob'],
        acquisition_method="uncertainty_sampling",
        seed=42,
        pool_size=200, # Dense pool for viz
        mc_samples=config['acquisition']['mc_samples']
    )
    
    # We need a trained model ideally.
    # Let's just init a random model and show the "initial" uncertainty landscape.
    # Or load one from the experiments if possible.
    
    print("Generating Uncertainty Landscape visualization...")
    learner = ActiveLearner(args)
    # Generate large pool
    pool = torch.rand((200, 4))
    pool[:, 3] = torch.randint(0, 5, (200,)).float()
    
    wrapper_model = learner._create_predictive_model()
    
    # Score
    scores = learner.acquisition_manager._score_candidates(pool, wrapper_model, n_samples=10)
    scores = scores.cpu().numpy()
    
    # PCA projection of params (first 3 dims: alpha, beta, gamma)
    params = pool[:, :3].numpy()
    pca = PCA(n_components=2)
    proj = pca.fit_transform(params)
    
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=scores, cmap='viridis')
    plt.colorbar(sc, label='Uncertainty Score')
    plt.title("Uncertainty Landscape (PCA of Parameter Space)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_path = os.path.abspath("experiments/uncertainty_landscape.png")
    plt.savefig(save_path)
    print(f"Uncertainty plot saved to {save_path}")

def main():
    config = load_config()
    
    base_dir = "results_experiment_final"
    os.makedirs(base_dir, exist_ok=True)
    
    methods = ["random", "uncertainty_sampling"]
    seeds = config['experiment']['seeds']
    
    methods_data = {}
    
    # Run Experiments
    for method in methods:
        for seed in range(seeds):
            run_dir = os.path.join(base_dir, f"{method}_seed{seed}")
            if not os.path.exists(os.path.join(run_dir, "metrics.json")):
                 run_single_seed(method, seed, config, run_dir)
        
        # Aggregate
        steps, mean, std = aggregate_results(base_dir, method, seeds)
        if steps is not None:
            methods_data[method] = (steps, mean, std)
            
    # Plot Performance
    plot_performance(methods_data, "experiments/final_performance.png")
    
    # Visualize Uncertainty
    if config['visualization']['plot_uncertainty_space']:
        visualize_uncertainty_space(None, config)

if __name__ == "__main__":
    main()
