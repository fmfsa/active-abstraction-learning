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

def run_single_seed(method, family, seed, config, dirname):
    print(f"--- Running {method} (Seed {seed}) [Family: {family}] ---")
    
    cmd = [
        "python", "experiments/run_aal.py",
        "--acquisition_method", method,
        "--dirname", dirname,
        "--seed", str(seed),
        "--family", family,
        "--dropout_prob", str(config['model']['dropout_prob']),
        "--budget", str(config['experiment']['budget']),
        "--initial_samples", str(config['experiment']['initial_samples']),
        "--n_acquire", str(config['experiment']['n_acquire']),
        "--pool_size", str(config['acquisition']['pool_size']),
        "--mc_samples", str(config['acquisition']['mc_samples']),
        "--epochs", str(config['experiment'].get('epochs', 30)),
        "--initial_condition_value", str(config['experiment'].get('initial_condition', {}).get('value', 0.01))
    ]
    
    if config.get('simulation') and config['simulation'].get('fixed_params'):
        fp_str = ",".join(map(str, config['simulation']['fixed_params']))
        cmd.extend(["--fixed_params", fp_str])
        
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    subprocess.check_call(cmd, env=env)

def aggregate_results(base_exp_dir, family, method, seeds):
    all_nlls = []
    steps = None
    
    for seed in range(seeds):
        path = os.path.join(base_exp_dir, f"{family}_{method}_seed{seed}", "metrics.json")
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

def plot_performance(methods_data, save_path, family_name):
    plt.figure(figsize=(10, 6))
    
    for method, (steps, mean, std) in methods_data.items():
        plt.plot(steps, mean, marker='o', label=method)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
        
    plt.xlabel("Dataset Size")
    plt.ylabel("Test NLL")
    plt.title(f"AAL Performance ({family_name}) (Mean ± Std)")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Performance plot saved to {save_path}")

def main():
    config = load_config()
    
    base_dir = "results_experiment_final"
    os.makedirs(base_dir, exist_ok=True)
    
    families = config['experiment']['families']
    methods = config['experiment']['methods']
    seeds = config['experiment']['seeds']
    
    for family in families:
        print(f"\n=== Processing Family: {family} ===")
        methods_data = {}
        
        for method in methods:
            for seed in range(seeds):
                run_dir = os.path.join(base_dir, f"{family}_{method}_seed{seed}")
                if not os.path.exists(os.path.join(run_dir, "metrics.json")):
                     run_single_seed(method, family, seed, config, run_dir)
            
            # Aggregate
            steps, mean, std = aggregate_results(base_dir, family, method, seeds)
            if steps is not None:
                methods_data[method] = (steps, mean, std)
        
        # Plot Performance per Family
        plot_performance(methods_data, f"experiments/final_performance_{family}.png", family)

    # Visualize Trajectories
    print("Generating Trajectory Grid...")
    subprocess.check_call(["python", "experiments/visualize_trajectories.py"])



if __name__ == "__main__":
    main()
