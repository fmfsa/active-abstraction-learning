import torch
from src.active_abstraction.models import DropoutMLP, MCDropoutWrapper

def test_dropout_mlp():
    torch.manual_seed(42)
    mlp = DropoutMLP(input_dim=10, hidden_dims=[10, 10], output_dim=1, dropout_prob=0.5).double()
    
    x = torch.randn(1, 10).double()
    
    # Test 1: Standard eval mode should be deterministic
    mlp.eval()
    y1 = mlp(x)
    y2 = mlp(x)
    assert torch.allclose(y1, y2), "Eval mode should be deterministic"
    
    # Test 2: MCDropoutWrapper should be stochastic
    wrapped_mlp = MCDropoutWrapper(mlp)
    
    # Note: Wrapped model forces training mode for dropout
    y_mc1 = wrapped_mlp(x)
    y_mc2 = wrapped_mlp(x)
    
    print(f"y1: {y1.item()}")
    print(f"y_mc1: {y_mc1.item()}")
    print(f"y_mc2: {y_mc2.item()}")
    
    assert not torch.allclose(y_mc1, y_mc2), "MCDropoutWrapper should be stochastic"
    
    print("Test passed!")

if __name__ == "__main__":
    test_dropout_mlp()
