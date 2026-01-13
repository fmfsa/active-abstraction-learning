import torch
import numpy as np

class AcquisitionManager:
    def __init__(self, method='random', seed=0):
        """
        method: 'random', 'uncertainty_sampling', 'disagreement', 'variance_reduction'
        """
        self.method = method
        self.rng = np.random.RandomState(seed)

    def select_next_intervention(self, candidate_pool, model, n_samples=10, batch_size=1):
        """
        Selects the best intervention(s) from the candidate pool.
        
        Args:
            candidate_pool: Tensor of shape (P, D) where P is pool size, D is dimension of intervention params.
            model: The surrogate model (wrapped in MCDropoutWrapper if needed).
            n_samples: Number of MC samples to take for uncertainty estimation.
            batch_size: Number of candidates to select.
            
        Returns:
            selected_indices: Indices of the selected candidates in the pool.
            selected_candidates: The actual candidate vectors.
        """
        if self.method == 'random':
            indices = self.rng.choice(len(candidate_pool), size=batch_size, replace=False)
            return indices, candidate_pool[indices]
        
        scores = self._score_candidates(candidate_pool, model, n_samples)
        
        # Select top k scores
        # Note: We assume higher score = better candidate (more uncertainty/utility)
        _, top_indices = torch.topk(scores, k=batch_size)
        top_indices = top_indices.cpu().numpy()
        
        return top_indices, candidate_pool[top_indices]

    def _score_candidates(self, candidate_pool, model, n_samples):
        """
        Computes a score for each candidate.
        """
        model.eval() # Ensure eval mode logic is handled (MCDropoutWrapper forces train mode internally if needed)
        
        # We need to run the model multiple times
        # candidate_pool is (P, D)
        # We want outputs (P, n_samples, OutputDim)
        
        # Depending on how the model forward works, we might need a loop or batching
        # The 'model' here is likely an Omega/RNN system.
        # For simplicity, let's assume model(x) returns a distribution or point estimate.
        # But wait, in ics4csm, the model takes (y0, params). 
        # The 'candidate_pool' are the 'params'.
        
        # We need to interface with the specific way ics4csm models work.
        # Omega takes (grids, parameters). Grids is often None for simple cases or handled internally.
        # Actually, looking at `utils.py/generate_dists`:
        # new_params = omega(None, params.unsqueeze(0).double())[0]
        # So we can pass (None, candidates).
        
        scores = []
        
        # Doing this in a loop for now to avoid OOM if pool is large, 
        # but ideal would be batch processing if Omega supports it.
        # Omega seems to support batching: x = self.ffn(x).
        
        with torch.no_grad():
            # Get predictions for all samples
            # Repetitive forward passes for MC Dropout
            
            # (P, D)
            candidates = candidate_pool.double() 
            
            outputs_list = []
            for _ in range(n_samples):
                # model here should refer to the 'omega' usually, 
                # but the full surrogate involves omega -> ode -> rnn.
                # Uncertainty comes from omega (if it has dropout) and rnn (if it has dropout).
                # If we just want parameter uncertainty (Epistemic on structure), we care about Omega's output variance?
                # OR the final predicted distribution entropy?
                # AAL usually targets "Uncertainty in Causal Abstraction", so uncertainty in Omega's mapping is key.
                # However, the user said "surrogate must output... distribution over distributions".
                # So we should look at the final output `e_pars` (emission params) variance or entropy.
                
                # We need the full forward pass from params to e_pars.
                # Let's assume 'model' passed here is a callable that takes params and returns e_pars (logits/probs).
                # We will construct a helper for this in the loop.
                
                outs = model(candidates) # Expected shape (P, ...)
                outputs_list.append(outs)
                
            # Stack: (n_samples, P, ...)
            outputs = torch.stack(outputs_list)
            
            # Calculate score based on method
            if self.method == 'uncertainty_sampling':
                # Variance of the mean prediction or Entropy
                # Let's use Variance of the logits for now as a proxy for epistemic uncertainty
                # outputs: (n_samples, P, OutputDim)
                # Compute variance across samples
                var = torch.var(outputs, dim=0).mean(dim=-1) # (P,) - mean over output dims
                return var
                
            elif self.method == 'entropy':
                # Entropy of the expected predictive distribution
                # If outputs are logits, convert to probs
                # This is "Predictive Entropy"
                # But for MC Dropout, we want "Mutual Information" usually (BALD), 
                # or just simple Variance.
                # Let's stick to simple variance of logits for 'uncertainty_sampling'
                # and maybe add a specific 'bald' if needed.
                return torch.var(outputs, dim=0).mean(dim=-1)
                
            elif self.method == 'disagreement':
                # KL Divergence between committees
                # Here we treat MC samples as committee members
                # Compute pairwise KL or Variance (approximation)
                # Variance is a good proxy for disagreement in regression-like outputs (logits).
                 return torch.var(outputs, dim=0).mean(dim=-1)
                 
        return torch.zeros(len(candidate_pool))
