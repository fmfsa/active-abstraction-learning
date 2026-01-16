import torch
import torch.nn as nn
from neurips_ics4csm.networks import MLP, RNN, count_pars

class DropoutMLP(MLP):
    def __init__(self, 
                 input_dim=64,
                 hidden_dims=[32, 32],
                 output_dim=1,
                 final_nonlinearity=torch.nn.Identity(),
                 dropout_prob=0.1):
        
        # Initialize parent class slightly differently because we want to inject dropout
        # creating a dummy parent initialization to set basic attributes, but we will override _layers
        super().__init__(input_dim, hidden_dims, output_dim, final_nonlinearity)
        
        self.dropout_prob = dropout_prob
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Re-build layers with dropout
        self._layers = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), self.relu, self.dropout)
        for i in range(len(hidden_dims) - 1):
            self._layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self._layers.append(self.relu)
            self._layers.append(self.dropout)
        self._layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self._final_nl = final_nonlinearity
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize final layer with small weights to keep initial outputs reasonable
        if isinstance(self._layers[-1], nn.Linear):
            nn.init.normal_(self._layers[-1].weight, mean=0.0, std=0.1)
            nn.init.constant_(self._layers[-1].bias, 0.0)

class DropoutRNN(RNN):
    def __init__(self,
                 input_size=3,
                 hidden_size=32,
                 num_layers=1,
                 final_ff=nn.Identity(),
                 nonlinearity='tanh',
                 flavour='gru',
                 dropout_prob=0.1):
        
        # Initialize parent
        super().__init__(input_size, hidden_size, num_layers, final_ff, nonlinearity, flavour)
        
        self.dropout_prob = dropout_prob
        # If num_layers > 1, we can add dropout to the RNN itself
        if num_layers > 1:
            if flavour == 'gru':
                self._rnn = nn.GRU(input_size, hidden_size, num_layers,
                                   batch_first=True, dropout=dropout_prob)
            else:
                self._rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=nonlinearity,
                                   batch_first=True, dropout=dropout_prob)
        # Note: For num_layers=1, RNN dropout argument is ignored by PyTorch unless we do it manually on outputs
        
        # We can also wrap the final feedforward network if it's an MLP
        if isinstance(self._fff, MLP) and not isinstance(self._fff, DropoutMLP):
            # Ideally we would want the final_ff to be a DropoutMLP already, 
            # but if it's passed in, we might be able to leave it as is 
            # or rely on the caller to pass a DropoutMLP.
            pass

    def forward(self, x, h=None):
        # We can add dropout to the input or output if needed, but usually it's inside the layers
        if h is None:
            out, _ = self._rnn(x)
        else:
            out, _ = self._rnn(x, h)
            
        logits = self._fff(out)
        # Clamp logits to prevent numerical instability
        return torch.clamp(logits, min=-20.0, max=20.0)

class MCDropoutWrapper(nn.Module):
    def __init__(self, model):
        """
        Wraps a model to enable dropout during inference (eval mode).
        """
        super().__init__()
        self.model = model
        
    def forward(self, *args, **kwargs):
        # Force dropout layers to be in training mode
        self.model.train() 
        # But we might want batch norm to stay in eval mode? 
        # The base models don't seem to use BatchNorm, so this simple switch is likely fine.
        # However, a safer way for general models is to iterate children.
        # But for this specific codebase, self.model.train() is consistent with "MC Dropout".
        return self.model(*args, **kwargs)

def generate_dropout_networks(kind='lodernn', seed=0, dropout_prob=0.1):
    torch.manual_seed(seed)
    if kind in ['lodernn', 'lrnn']:
        # This is to map from hidden state of RNN to logits
        mlp_net = DropoutMLP(input_dim=32, output_dim=3, hidden_dims=[32, 64, 32, 16], 
                             final_nonlinearity=torch.nn.Identity(), dropout_prob=dropout_prob).double()
        # This is to map from ODE output to hidden state of RNN
        rnn_net = DropoutRNN(input_size=3, final_ff=mlp_net, flavour='gru', dropout_prob=dropout_prob).double()
        # This maps from ABM parameters to parameters of ODE
        mlp = DropoutMLP(input_dim=3, output_dim=3, hidden_dims=[32, 64, 32], 
                         final_nonlinearity=torch.nn.Sigmoid(), dropout_prob=dropout_prob).double()
    elif kind in ['lodernn_small', 'lrnn_small']:
        mlp_net = DropoutMLP(input_dim=32, output_dim=3, hidden_dims=[32, 64, 32], 
                             final_nonlinearity=torch.nn.Identity(), dropout_prob=dropout_prob).double()
        rnn_net = DropoutRNN(input_size=3, final_ff=mlp_net, flavour='gru', dropout_prob=dropout_prob).double()
        mlp = DropoutMLP(input_dim=3, output_dim=3, hidden_dims=[32, 32], 
                         final_nonlinearity=torch.nn.Sigmoid(), dropout_prob=dropout_prob).double()
    elif kind == 'lode':
        rnn_net = torch.nn.Identity()
        mlp = DropoutMLP(input_dim=3, output_dim=3, hidden_dims=[32, 64, 64, 64, 32], 
                         final_nonlinearity=torch.nn.Sigmoid(), dropout_prob=dropout_prob).double()
    
    # We need to import Omega from utils, but utils imports models... circular dependency risk?
    # neurips_ics4csm.utils imports sirs_ode, networks. 
    # generate_networks is in neurips_ics4csm.utils.
    # We can probably import Omega from neurips_ics4csm.utils OK.
    from neurips_ics4csm.utils import Omega
    
    omega = Omega(mlp)
    print("Total trainable parameters =", 
          count_pars(omega) + 
          count_pars(rnn_net)
          )
    return rnn_net, omega
