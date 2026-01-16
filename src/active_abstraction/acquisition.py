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
        Computes an information gain score for each candidate in the pool.
        
        Supported Methods:
        - random: (Handled in select_next_intervention)
        - uncertainty_sampling: Variance of model predictions.
        - bald: Mutual Information (Entropy of Expected - Expected Entropy).
        - entropy: Entropy of Expected Predictions.
        """
        model.eval() 
        
        with torch.no_grad():
            candidates = candidate_pool.double() 
            
            # 1. MC Sampling
            outputs_list = []
            for _ in range(n_samples):
                outs = model(candidates) # (P, OutputFeatures)
                outputs_list.append(outs)
            
            # Stack: (n_samples, P, OutputFeatures)
            outputs = torch.stack(outputs_list)
            S, P, F = outputs.shape
            
            if self.method == 'uncertainty_sampling':
                # Variance of Mean Logits (Uncertainty in Model Parameters)
                return torch.var(outputs, dim=0).mean(dim=-1)
                
            elif self.method == 'bald':
                # BALD: H[y|x, D] - E_w[H[y|x, w]]
                # Requires probabilistic output (Softmax)
                if F % 3 == 0:
                    outputs_reshaped = outputs.view(S, P, -1, 3)
                    probs = torch.softmax(outputs_reshaped, dim=-1) # (S, P, T, 3)
                    
                    # A. Entropy of Expected Prediction
                    mean_probs = torch.mean(probs, dim=0) # (P, T, 3)
                    entropy_predictive = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1).mean(dim=-1)
                    
                    # B. Expected Entropy of Predictions
                    entropy_samples = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) # (S, P, T)
                    mean_entropy = torch.mean(entropy_samples, dim=0).mean(dim=-1)
                    
                    return entropy_predictive - mean_entropy
                else:
                    # Fallback to variance if output shape implies not classification
                    return torch.var(outputs, dim=0).mean(dim=-1)

            elif self.method == 'entropy':
                if F % 3 == 0:
                    outputs_reshaped = outputs.view(S, P, -1, 3)
                    probs = torch.softmax(outputs_reshaped, dim=-1) 
                    mean_probs = torch.mean(probs, dim=0)
                    entropy_predictive = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1).mean(dim=-1)
                    return entropy_predictive
                return torch.var(outputs, dim=0).mean(dim=-1)
                
            elif self.method == 'disagreement':
                 return torch.var(outputs, dim=0).mean(dim=-1)
                 
        return torch.zeros(len(candidate_pool))
