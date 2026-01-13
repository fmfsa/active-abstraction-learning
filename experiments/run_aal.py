import argparse
import time
import os
import torch
import numpy as np
import argparse
import time
import os
import torch
import numpy as np
import yaml
import json

from active_abstraction.loop import ActiveLearner

def main():
    parser = argparse.ArgumentParser(description="Active Abstraction Learning Experiment")
    
    # Model args
    parser.add_argument("--family", type=str, default="lode", choices=["lode", "lodernn", "lrnn"], help="Surrogate family")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout probability for MC Dropout")
    
    # AL args
    parser.add_argument("--acquisition_method", type=str, default="uncertainty_sampling", 
                        choices=["random", "uncertainty_sampling", "disagreement", "entropy"], 
                        help="Acquisition function")
    parser.add_argument("--pool_size", type=int, default=100, help="Size of In Silico Screening pool")
    parser.add_argument("--mc_samples", type=int, default=10, help="Number of MC Dropout samples for scoring")
    parser.add_argument("--budget", type=int, default=5, help="Number of acquisition steps")
    parser.add_argument("--n_acquire", type=int, default=1, help="Number of candidates to acquire per step")
    parser.add_argument("--initial_samples", type=int, default=10, help="Number of initial random samples")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per update (kept low for demo/speed)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dirname", type=str, default=f"results_aal_{int(time.time())}", help="Output directory")
    parser.add_argument("--fixed_params", type=str, default=None, help="Fixed parameters as comma-separated string: alpha,beta,gamma")

    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.dirname, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse fixed params if provided
    if args.fixed_params:
        try:
            args.fixed_params = [float(x) for x in args.fixed_params.split(',')]
            print(f"Using FIXED PARAMETERS: {args.fixed_params}")
        except ValueError:
            print("Error parsing fixed_params. Using random parameters.")
            args.fixed_params = None
    else:
        args.fixed_params = None
    
    print(f"Starting AAL Experiment: {args.acquisition_method} on {args.family}")
    print(f"Output directory: {args.dirname}")
    
    learner = ActiveLearner(args)
    learner.generate_test_set(n_test=50) # Small test set for speed in this demo
    
    results = {
        "step": [],
        "dataset_size": [],
        "test_nll": []
    }
    
    def log_step(step_idx):
        nll = learner.evaluate()
        ds_size = len(learner.states)
        print(f"-> Step {step_idx}: Dataset Size={ds_size}, Test NLL={nll:.4f}")
        results["step"].append(step_idx)
        results["dataset_size"].append(ds_size)
        results["test_nll"].append(nll)

    # 1. Initialization
    print(">>> Phase 1: Initialization")
    learner.collect_initial_data(n_samples=args.initial_samples)
    learner._train_surrogate() # Train on initial data
    log_step(0)
    
    # 2. AL Loop
    print(">>> Phase 2: Active Loop")
    for step in range(args.budget):
        print(f"\n--- AL Step {step+1}/{args.budget} ---")
        learner.step(n_acquire=args.n_acquire)
        log_step(step + 1)
        
    print("\n>>> Experiment Complete.")
    print(f"Final dataset size: {len(learner.states)}")
    
    import json
    with open(os.path.join(args.dirname, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Save Final Model
    # learner.model is the Wrapper/Surrogate. 
    # We need to save the underlying torch modules: omega and rnn_net
    # Access them from learner state if possible, or usually learner.obs_omega and learner.obs_rnn_net
    if hasattr(learner, 'obs_omega'):
        torch.save({
            'omega': learner.obs_omega.state_dict(),
            'rnn': learner.obs_rnn_net.state_dict() if hasattr(learner, 'obs_rnn_net') else None
        }, os.path.join(args.dirname, "final_model.pt"))
        print(f"Final model saved to {os.path.join(args.dirname, 'final_model.pt')}")


if __name__ == "__main__":
    main()
