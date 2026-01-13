import torch
import numpy as np
import copy
from neurips_ics4csm.training_ode_grid_sirs import train_epi
from neurips_ics4csm.utils import run_spatial_intervention, generate_networks, create_instantiate_emission, create_nll, create_instantiate_sirsrnn, instantiate_model
from active_abstraction.models import generate_dropout_networks, MCDropoutWrapper
from active_abstraction.acquisition import AcquisitionManager

class ActiveLearner:
    def __init__(self, 
                 args, 
                 device='cpu'):
        
        self.args = args
        self.device = device
        
        # Dimensions
        self.L = 50
        self.N = self.L ** 2
        self.T = 50
        
        # Acquisition Manager
        self.acquisition_manager = AcquisitionManager(method=args.acquisition_method, seed=args.seed)
        
        # Data storage
        # We need to store:
        # - states: (R, N_grid, N_grid, 3) 
        # - agg_ts: (R, T+1, 3)
        # - thi: (R, 4) -> (alpha, beta, gamma, intervention_time)
        self.states = []
        self.agg_ts = [] # This is 'xs' in utils.run_spatial_intervention
        self.thi = []
        
        # Current model
        self.omega = None
        self.rnn_net = None # If using RNN based wrapper
        self.model = None # The ODE model (or wrapper)
        
        # Initialize surrogate structure
        self._init_surrogate_structure()
        
    def _init_surrogate_structure(self):
        # Generate networks with dropout
        # We assume args.family is like 'lodernn'
        self.obs_rnn_net, self.obs_omega = generate_dropout_networks(kind=self.args.family, seed=self.args.seed, dropout_prob=self.args.dropout_prob)
        
        # Instantiate emission and NLL
        self.instantiate_emission = create_instantiate_emission(self.N, kind=self.args.family)
        self.negative_log_likelihood = create_nll(self.instantiate_emission, self.N)
        
        # Model instantiation function (ODE)
        if self.args.family == "lrnn":
            self.obs_inst_mod = create_instantiate_sirsrnn(self.obs_rnn_net)
        else:
            self.obs_inst_mod = instantiate_model

        # Optimiser will be created during training to clear state? Or keep state?
        # Usually retraining from scratch or fine-tuning. 
        # "Retrain the surrogate ... by minimizing" implies potential scratch retraining or warm start.
        # We'll default to warm start (keeping weights) but new optimizer usually.
    
    def collect_initial_data(self, n_samples=10):
        print(f"Collecting {n_samples} initial random samples...")
        self._collect_data(n_samples, method='random')
        
    def step(self, n_acquire=1):
        """
        One cycle of AL:
        1. In Silico Screening (Generate Pool)
        2. Acquisition (Rank & Select)
        3. Ground Truth Query (Run Simulator)
        4. update (Retrain)
        """
        # 1. Generate Pool
        pool_size = self.args.pool_size
        candidate_pool = torch.rand((pool_size, 4)) # [alpha, beta, gamma, intervention]
        
        # Check for fixed params
        if hasattr(self.args, 'fixed_params') and self.args.fixed_params is not None:
            fp = self.args.fixed_params
            # Assign alpha, beta, gamma
            candidate_pool[:, 0] = fp[0]
            candidate_pool[:, 1] = fp[1]
            candidate_pool[:, 2] = fp[2]
            
        # Intervention is random int 0-4
        candidate_pool[:, 3] = torch.randint(0, 5, (pool_size,)).float()
        
        # 2. Acquisition
        # We need to wrap the current model to produce predictions from candidates
        # The 'model' passed to acquisition manager needs to map candidates -> output variance/entropy.
        # Our surrogate definition is complex (Omega -> ODE -> RNN -> Emission).
        # We need a callable that takes params (candidate) and returns the emission distribution parameters (logits).
        
        wrapper_model = self._create_predictive_model()
        
        indices, selected_candidates = self.acquisition_manager.select_next_intervention(
            candidate_pool, 
            wrapper_model, 
            n_samples=self.args.mc_samples, 
            batch_size=n_acquire
        )
        
        print(f"Acquired candidates: {selected_candidates}")
        
        # 3. Ground Truth Query
        # Run simulator for selected candidates
        self._query_simulator(selected_candidates)
        
        # 4. Abstraction Update
        self._train_surrogate()
        
    def _collect_data(self, n, method='random'):
        params = torch.rand((n, 3))
        if hasattr(self.args, 'fixed_params') and self.args.fixed_params is not None:
            fp = self.args.fixed_params
            params[:, 0] = fp[0]
            params[:, 1] = fp[1]
            params[:, 2] = fp[2]
        intervention = torch.randint(0, 5, (n,)).float()
        candidates = torch.cat([params, intervention.unsqueeze(1)], dim=1)
        self._query_simulator(candidates)

    def _query_simulator(self, candidates):
        # candidates: (K, 4)
        for i in range(len(candidates)):
            cand = candidates[i]
            params = cand[:3] # alpha, beta, gamma
            interv_time = int(cand[3].item())
            
            # i0 is random dist in original collect_data?
            # "Draw initial proportion of infected individuals from Uniform(0,1)" in utils.collect_data
            # We should probably sample i0 randomly here too, OR include it in the candidate?
            # The prompt says "intervention i*". Usually intervention refers to the do-operator parameters.
            # Initial state might be part of the context or fixed.
            # In utils.collect_data, i0 is random.
            i0 = torch.rand(1)
            
            # Run wrapper
            init_state, x = run_spatial_intervention(params, interv_time, i0, self.T, self.L)
            
            self.states.append(init_state)
            self.agg_ts.append(x)
            self.thi.append(cand)
            
    def _create_predictive_model(self):
        # Returns a vector-to-vector callable model for AcquisitionManager
        # Input: (Batch, 4) -> [alpha, beta, gamma, int_time]
        # Output: (Batch, feature_dim) -> e.g. logits of the emission
        
        # We need to handle the fact that forward pass requires y0 (initial state).
        # But we don't know y0 for the candidates yet (it's randomly sampled during query).
        # HOWEVER, the 'epistemic uncertainty' we care about is in the learned MAP `omega` (and RNN).
        # If we fix a 'canonical' y0, we can measure uncertainty in the dynamics.
        # Or we can integrate over y0. 
        # For simplicity, let's fix a canonical y0 = [0.99, 0.01, 0.0].
        
        canonical_y0 = torch.tensor([0.99, 0.01, 0.0]).double()
        t_seq = torch.linspace(0, self.T, self.T+1)
        
        # Ensure we are using the current best weights
        omega = self.obs_omega
        rnn_net = self.obs_rnn_net
        
        # Determine model structure (ODE or RNN)
        if self.args.family == 'lrnn':
            # This is complex because lrnn creates a new model per time step or something? 
            # create_instantiate_sirsrnn(obs_rnn_net) returns a function that returns SIRSRNN.
            # SIRSRNN takes (y0, params).
            pass 
        
        # Let's define the forward pass callable
        class PredictiveWrapper(torch.nn.Module):
            def __init__(self, omega, rnn_net, family, inst_mod, t_seq, y0):
                super().__init__()
                self.omega = MCDropoutWrapper(omega)
                self.rnn_net = MCDropoutWrapper(rnn_net)
                self.family = family
                self.inst_mod = inst_mod
                self.t_seq = t_seq
                self.y0 = y0
                
            def forward(self, candidates):
                # candidates: (Batch, 4)
                # Split params and intervention time
                # Check `utils.py/generate_dists` logic
                
                # We need to output something whose variance represents uncertainty.
                # Emission logits are good.
                
                outputs = []
                for j in range(len(candidates)):
                    cand = candidates[j]
                    params = cand[:3]
                    interv_time = cand[3] # Treated as float
                    
                    # Omega forward: (grids, params)
                    # utils.py: new_params = omega(None, params.unsqueeze(0).double())[0]
                    new_params = self.omega(None, params.unsqueeze(0).double())[0]
                    
                    # Model forward
                    # We need to construct the model instance.
                    if self.family == 'lrnn':
                        # inst_mod is a factory function
                        model = self.inst_mod(self.t_seq)
                        # forward(y0, params)
                        # SIRSRNN forward expects params.
                        # Wait, SIRSODE_naive_int uses 'interv_time' (which is passed in y0[-1]?? NO)
                        # Let's check sirs_ode.py
                        # SIRSODE_naive_int: forward(y0, params). i = y0[-1].
                        # So intervention time is passed as part of y0 in the naive formulation?
                        # utils.py ln 103: model(torch.cat((y0, torch.tensor([i])), dim=-1), new_params)
                        pass
                        
                    # Prepare y0_aug
                    y0_aug = torch.cat((self.y0, interv_time.view(1)), dim=-1)
                    
                    if self.family != 'lrnn':
                        model = self.inst_mod(self.t_seq) # standard ODE
                        y_mac = model(y0_aug, new_params) # -> (T+1, 4) ?
                        # sirs_ode.py: return y (sequence)
                        y_mac = y_mac[:, :-1] # Drop aux dimension if needed? 
                        # utils.py ln 103: [:, :-1]
                        
                    else:
                        model = self.inst_mod(self.t_seq)
                        y = model(y0_aug, new_params)
                        y_mac = y[:, :-1]
                    
                    # RNN net forward
                    # e_pars = rnn_net(y_mac.double())
                    e_pars = self.rnn_net(y_mac.double())
                    
                    # e_pars is (T, output_dim)
                    # Flatten to (T*dim)
                    outputs.append(e_pars.flatten())
                
                return torch.stack(outputs)

        return PredictiveWrapper(omega, rnn_net, self.args.family, self.obs_inst_mod, t_seq, canonical_y0)

    def _train_surrogate(self):
        print(f"Training surrogate on {len(self.states)} samples...")
        
        # Prepare data tensors
        # states -> stack
        # agg_ts -> stack
        # thi -> stack
        
        train_abm_states = torch.stack(self.states).double()
        train_abm_agg_ts = torch.stack(self.agg_ts).double()
        train_abm_thi = torch.stack(self.thi).double()
        
        # Create Optimiser
        optimiser = torch.optim.Adam(
            list(self.obs_rnn_net.parameters()) + list(self.obs_omega.parameters()),
            lr=1e-2
        )
        
        # Call train_epi
        # Note: train_epi returns (best_omega, best_calib, loss_hist)
        # We need to handle checks if it returns model too (arg full_node)
        
        # We need to provide 'calib' (rnn_net) and 'ode' (inst_mod)
        
        # train_epi signature:
        # train_epi(omega, calib, abm_states, abm_agg_ts, abm_thi, ode, loss_fn, optimiser, ...)
        
        # What is 'ode' argument expectation?
        # In job_script.py: obs_inst_mod
        # obs_inst_mod = create_instantiate_sirsrnn(obs_rnn_net) OR instantiate_model
        
        # Call train_epi
        full_node = (self.args.family == 'lrnn')
        
        result = train_epi(
            omega=self.obs_omega,
            calib=self.obs_rnn_net,
            abm_states=train_abm_states,
            abm_agg_ts=train_abm_agg_ts,
            abm_thi=train_abm_thi,
            ode=self.obs_inst_mod,
            loss_fn=self.negative_log_likelihood,
            optimiser=optimiser,
            batch_size=min(50, len(self.states)),
            max_epochs=self.args.epochs, 
            notebook=False,
            full_node=full_node
        )
        
        if full_node:
             self.obs_omega, self.obs_rnn_net, self.model, _ = result
        else:
             self.obs_omega, self.obs_rnn_net, _ = result

    def generate_test_set(self, n_test=100):
        print(f"Generating fixed test set of size {n_test}...")
        # Use utils.collect_data logic but adapted
        params = torch.rand((n_test, 3))
        if hasattr(self.args, 'fixed_params') and self.args.fixed_params is not None:
            fp = self.args.fixed_params
            params[:, 0] = fp[0]
            params[:, 1] = fp[1]
            params[:, 2] = fp[2]
        intervention = torch.randint(0, 5, (n_test,)).float()
        candidates = torch.cat([params, intervention.unsqueeze(1)], dim=1)
        
        self.test_states = []
        self.test_agg_ts = []
        self.test_thi = []
        
        for i in range(len(candidates)):
            cand = candidates[i]
            params = cand[:3]
            interv_time = int(cand[3].item())
            i0 = torch.rand(1)
            init_state, x = run_spatial_intervention(params, interv_time, i0, self.T, self.L)
            self.test_states.append(init_state)
            self.test_agg_ts.append(x)
            self.test_thi.append(cand)
            
        self.test_states = torch.stack(self.test_states).double()
        self.test_agg_ts = torch.stack(self.test_agg_ts).double()
        self.test_thi = torch.stack(self.test_thi).double()

    def evaluate(self):
        # Compute NLL on test set
        if not hasattr(self, 'test_states'):
             return float('nan')
             
        # Reuse produce_loss logic from training_ode_grid_sirs but for test
        # produce_loss(omega, calib, states, aggs, ths, i_s, model, loss_fn)
        from neurips_ics4csm.training_ode_grid_sirs import produce_loss
        
        # We need the ODE model instance
        model = self.obs_inst_mod(torch.linspace(0, self.T, self.T+1))
        
        th, i = self.test_thi[:, :-1], self.test_thi[:, -1]
        
        with torch.no_grad():
            loss = produce_loss(
                self.obs_omega, 
                self.obs_rnn_net, 
                self.test_states, 
                self.test_agg_ts, 
                th, 
                i, 
                model, 
                self.negative_log_likelihood
            )
        return loss.item()

