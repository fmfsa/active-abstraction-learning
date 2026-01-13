# AAL Implementation - Improvement Plan

This document contains prioritized recommendations for enhancing the Active Abstraction Learning implementation to fully meet the original requirements.

## Priority 0 (Critical) 🔴

### 1. Implement AT-UCB Integration
**Effort**: 2-3 days | **Impact**: Critical

The original specification requested AT-UCB integration, which is completely missing from the current implementation.

**Tasks**:
- [ ] Create `ATUCBActiveLearner` class extending `ActiveLearner`
- [ ] Implement `compute_ucb_scores()` method: UCB = μ(a) + β * σ(a)
- [ ] Implement `identify_optimality_boundary()` to find candidates where UCB ≥ threshold
- [ ] Add "imaginative rollouts" using surrogate for rapid candidate screening (10k+ candidates)
- [ ] Modify acquisition to focus on optimality boundary rather than uniform pool
- [ ] Add command-line arguments: `--use_atucb`, `--threshold`, `--beta`
- [ ] Document the AT-UCB approach in docstrings and README

**Key Files to Modify**:
- `src/active_abstraction/loop.py` - Add `ATUCBActiveLearner` class
- `experiments/run_aal.py` - Add AT-UCB configuration options

### 2. Fix Acquisition Function Implementations
**Effort**: 1 day | **Impact**: High

All acquisition methods currently compute the same variance metric. Each needs distinct implementation.

**Tasks**:
- [ ] **Entropy method**: Implement proper predictive entropy
  - Convert outputs to probabilities
  - Compute H(p) = -Σ p log p for mean predictions
- [ ] **Disagreement method**: Implement Query-By-Committee with KL divergence
  - Compute KL divergence between each MC sample and mean prediction
  - Return mean KL divergence across samples
- [ ] **Variance Reduction**: Implement expected variance reduction strategy
  - Approximate Fisher information or use BALD (Bayesian Active Learning by Disagreement)
  - Return expected reduction in model uncertainty
- [ ] Add `ValueError` for unknown methods (replace silent `torch.zeros()` return)

**Key Files to Modify**:
- `src/active_abstraction/acquisition.py` - Fix `_score_candidates()` method

### 3. Add Error Handling and Input Validation
**Effort**: 4 hours | **Impact**: High

Prevent runtime failures from invalid inputs.

**Tasks**:
- [ ] Add validation in `AcquisitionManager.select_next_intervention()`:
  - Check `len(candidate_pool) > 0`
  - Check `batch_size <= len(candidate_pool)`
  - Check `n_samples >= 1`
- [ ] Add validation in `ActiveLearner.__init__()`:
  - Validate `args.family` is in ['lode', 'lodernn', 'lrnn']
  - Validate `args.dropout_prob` is in [0, 1]
- [ ] Add device consistency checks in `PredictiveWrapper.forward()`
- [ ] Add try-except blocks around simulator calls with informative error messages

**Key Files to Modify**:
- `src/active_abstraction/acquisition.py`
- `src/active_abstraction/loop.py`

---

## Priority 1 (Should Fix) 🟡

### 4. Add Result Logging and Tracking
**Effort**: 1 day | **Impact**: Medium

Enable reproducibility and analysis of experiments.

**Tasks**:
- [ ] Add budget tracking:
  - Track `queries_used` vs `query_budget`
  - Raise error if budget exceeded
- [ ] Log metrics at each AL step:
  - Training loss
  - Validation loss
  - Abstraction error (KL divergence between surrogate and ground truth)
  - Selected candidates history
- [ ] Save artifacts:
  - Model checkpoints (`omega_step{i}.pt`, `rnn_step{i}.pt`)
  - Metrics CSV file
  - Loss curves plot
- [ ] Add `compute_abstraction_error()` method to compute KL(M || M')
- [ ] Create results summary at end of experiment

**Key Files to Modify**:
- `src/active_abstraction/loop.py` - Add tracking attributes and `compute_abstraction_error()`
- `experiments/run_aal.py` - Add saving logic after each step

### 5. Improve Documentation
**Effort**: 1-2 days | **Impact**: Medium

Make the codebase accessible to future users.

**Tasks**:
- [ ] Add module-level docstrings explaining AAL algorithm to:
  - `src/active_abstraction/acquisition.py`
  - `src/active_abstraction/models.py`
  - `src/active_abstraction/loop.py`
- [ ] Complete function docstrings with Args/Returns/Examples for:
  - `ActiveLearner._create_predictive_model()`
  - `ActiveLearner._train_surrogate()`
  - `AcquisitionManager._score_candidates()`
- [ ] Add inline comments explaining:
  - τ-ω transformation logic
  - Emission parameter handling
  - Why `canonical_y0` is fixed vs random `i0` in queries
- [ ] Create `docs/aal_tutorial.md` with:
  - Quick start guide
  - Example usage
  - Interpretation of results
- [ ] Update main `README.md` with AAL usage instructions

**Key Files to Create/Modify**:
- All files in `src/active_abstraction/`
- `docs/aal_tutorial.md` (new)
- `README.md`

### 6. Expand Test Coverage
**Effort**: 2 days | **Impact**: Medium

Prevent regressions and ensure correctness.

**Tasks**:
- [ ] Create `tests/test_loop.py`:
  - Test `ActiveLearner.step()` completes without errors
  - Test data accumulation (states/agg_ts/thi grow correctly)
  - Test budget tracking
- [ ] Expand `tests/test_acquisition.py`:
  - Add tests for each acquisition method showing distinct outputs
  - Test batch_size parameter
  - Test error handling for invalid inputs
- [ ] Create `tests/test_integration.py`:
  - End-to-end test with small dataset (5 initial + 3 AL steps)
  - Test all family options ('lode', 'lodernn', 'lrnn')
  - Verify surrogate improves over iterations
- [ ] Add regression tests:
  - Freeze a "known good" run with seed
  - Assert future runs match within tolerance

**Key Files to Create**:
- `tests/test_loop.py` (new)
- `tests/test_integration.py` (new)
- Expand `tests/test_acquisition.py`

---

## Priority 2 (Nice-to-Have) 🟢

### 7. Implement Ensemble Approach
**Effort**: 2 days | **Impact**: Low

Provide alternative to MC Dropout for epistemic uncertainty.

**Tasks**:
- [ ] Create `EnsembleWrapper` class:
  - Accepts list of models
  - Forward pass returns stacked outputs from all ensemble members
- [ ] Create `generate_ensemble_networks()` function:
  - Train N models with different random initializations
  - Return list of models
- [ ] Modify `ActiveLearner` to support ensemble mode:
  - Add `--uncertainty_method` argument: ['mc_dropout', 'ensemble']
  - Conditionally wrap models in `MCDropoutWrapper` or `EnsembleWrapper`
- [ ] Update acquisition to handle ensemble outputs (already supports stacking)

**Key Files to Modify**:
- `src/active_abstraction/models.py` - Add `EnsembleWrapper`
- `src/active_abstraction/loop.py` - Add ensemble support
- `experiments/run_aal.py` - Add `--uncertainty_method` argument

### 8. Performance Optimizations
**Effort**: 1 day | **Impact**: Low

Speed up candidate scoring for large pools.

**Tasks**:
- [ ] Batch process candidates in `PredictiveWrapper.forward()`:
  - Replace sequential loop with batched tensor operations
  - Target: Handle 100+ candidates simultaneously
- [ ] Add LRU cache for surrogate predictions:
  - Cache (candidate → prediction) mappings
  - Useful if same candidates appear in multiple pools
- [ ] Profile code to identify bottlenecks:
  - Use `torch.profiler` or `cProfile`
  - Optimize identified hotspots
- [ ] Add option to run on GPU:
  - Ensure all tensors respect `device` argument
  - Add `.to(device)` calls where missing

**Key Files to Modify**:
- `src/active_abstraction/loop.py` - Optimize `PredictiveWrapper`
- `src/active_abstraction/acquisition.py` - Add caching

### 9. Make Architectures Configurable
**Effort**: 4 hours | **Impact**: Low

Allow experimentation with network architectures without code changes.

**Tasks**:
- [ ] Add command-line arguments:
  - `--omega_hidden_dims` (e.g., "32,64,32")
  - `--rnn_hidden_size`
  - `--rnn_num_layers`
- [ ] Parse list arguments in `run_aal.py`
- [ ] Pass architecture configs to `generate_dropout_networks()`
- [ ] Update `generate_dropout_networks()` to accept architecture kwargs
- [ ] Optionally: Support config files (YAML/JSON) for complex setups

**Key Files to Modify**:
- `experiments/run_aal.py` - Add architecture arguments
- `src/active_abstraction/models.py` - Make `generate_dropout_networks()` flexible

---

## Quick Reference

### Implementation Order Recommendation
1. **AT-UCB Integration** (P0.1) - Core missing feature
2. **Fix Acquisition Functions** (P0.2) - Currently broken
3. **Error Handling** (P0.3) - Prevent crashes
4. **Result Logging** (P1.4) - Enable analysis
5. **Documentation** (P1.5) - Enable usage
6. Remaining items as time permits

### Testing Strategy
After each P0 item:
- Write unit tests
- Run integration test
- Verify no regressions

After all P0 items complete:
- Run full benchmark comparing AAL vs random baseline
- Document performance improvements

### Success Metrics
- [ ] All acquisition methods produce different scores on same pool
- [ ] AT-UCB reduces queries needed by 30%+ vs random
- [ ] Abstraction error decreases over AL iterations
- [ ] Zero runtime errors on valid inputs
- [ ] 80%+ test coverage

---

## Prompt Template for Future Agent

```
Please implement the improvements listed in IMPROVEMENT_PLAN.md.

Focus on Priority 0 items first:
1. AT-UCB Integration
2. Fix Acquisition Functions  
3. Add Error Handling

For each item:
- Implement the changes described
- Add unit tests
- Update documentation
- Verify existing tests still pass

Start with item P0.1 (AT-UCB Integration).
```
