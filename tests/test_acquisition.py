import torch
import torch.nn as nn
from src.active_abstraction.acquisition import AcquisitionManager

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # x is (Batch, Dim)
        # return random outputs depending on input magnitude
        # Higher input magnitude -> Higher variance (to simulating uncertainty)
        
        batch_size = x.shape[0]
        # Noise proportional to x mean
        noise_scale = x.mean(dim=1, keepdim=True).abs() # (Batch, 1)
        
        # Base prediction is constant 0
        prediction = torch.zeros(batch_size, 1).double()
        
        # Add noise
        noise = torch.randn(batch_size, 1).double() * noise_scale
        
        return prediction + noise

def test_acquisition():
    torch.manual_seed(42)
    manager = AcquisitionManager(method='uncertainty_sampling')
    
    # Pool of 5 candidates with increasing magnitude
    # Candidate 0: small magnitude -> low variance (low score)
    # Candidate 4: large magnitude -> high variance (high score)
    pool = torch.tensor([
        [0.1],
        [1.0],
        [2.0],
        [5.0],
        [10.0]
    ]).double()
    
    model = MockModel()
    
    # We expect the one with 10.0 to be selected because MockModel produces higher variance for it
    indices, selected = manager.select_next_intervention(pool, model, n_samples=20, batch_size=1)
    
    print(f"Selected index: {indices}")
    print(f"Selected candidate: {selected}")
    
    assert indices[0] == 4, f"Expected index 4 (highest variance), got {indices[0]}"
    
    # Test Random
    manager_rand = AcquisitionManager(method='random', seed=42)
    indices_rand, _ = manager_rand.select_next_intervention(pool, model, batch_size=2)
    assert len(indices_rand) == 2

    print("Test passed!")

if __name__ == "__main__":
    test_acquisition()
