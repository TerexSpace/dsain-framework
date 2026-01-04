#!/usr/bin/env python3
"""
Integration Patch for real_experiments.py
==========================================

This file contains the code snippets to integrate Byzantine attacks and
baseline aggregators into real_experiments.py

Usage:
1. Add these imports at the top of real_experiments.py
2. Replace the aggregation section with the new implementation
3. Update FederatedClient to use sophisticated attacks

Author: Almas Ospanov
"""

# ============================================================================
# STEP 1: ADD THESE IMPORTS AT THE TOP OF real_experiments.py
# ============================================================================

"""
# Add after existing imports:
from byzantine_attacks import get_byzantine_attack, ByzantineAttack
from baseline_aggregators import (
    get_aggregator,
    krum_aggregation,
    bulyan_aggregation,
    trimmed_mean_aggregation,
    median_aggregation,
    fltrust_aggregation,
    centered_clipping_aggregation
)
"""

# ============================================================================
# STEP 2: REPLACE FLConfig with enhanced version
# ============================================================================

ENHANCED_FL_CONFIG = """
@dataclass
class FLConfig:
    '''Federated learning configuration with Byzantine attack support.'''
    num_clients: int = 100
    participation_rate: float = 0.1
    local_epochs: int = 2
    local_batch_size: int = 32
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    num_rounds: int = 200

    # DSAIN-specific
    compression_ratio: float = 0.1
    dp_epsilon: float = float('inf')
    dp_delta: float = 1e-5
    gradient_clip: float = 10.0
    byzantine_frac: float = 0.0
    byzantine_threshold: float = 5.0

    # NEW: Byzantine attack configuration
    attack_type: str = 'sign_flipping'  # Type of Byzantine attack
    attack_scale: float = 5.0  # Attack strength parameter

    # Experiment
    seed: int = 42
    eval_every: int = 5
"""

# ============================================================================
# STEP 3: ENHANCED FederatedClient with sophisticated attacks
# ============================================================================

ENHANCED_CLIENT = """
class FederatedClient:
    '''Federated learning client with sophisticated Byzantine attacks.'''

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        model_fn: Callable[[], nn.Module],
        config: FLConfig,
        is_byzantine: bool = False,
        num_byzantine: int = 0,
        num_honest: int = 0
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.model_fn = model_fn
        self.config = config
        self.is_byzantine = is_byzantine

        # Initialize Byzantine attack if needed
        self.attack = None
        if is_byzantine:
            self.attack = get_byzantine_attack(
                config.attack_type,
                num_byzantine=num_byzantine,
                num_honest=num_honest,
                scale=config.attack_scale,
                total_rounds=config.num_rounds
            )
            logger.info(f"Client {client_id}: Initialized {config.attack_type} attack")

    def local_train(self, global_model: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict]:
        '''Perform local training and return model update (delta).'''
        # ... existing local training code ...

        # Compute delta (local - global)
        delta = {}
        with torch.no_grad():
            for name, param in local_model.named_parameters():
                delta[name] = param.data.clone() - global_state[name]

        # Apply Byzantine attack if client is malicious
        if self.is_byzantine and self.attack is not None:
            delta = self.attack.attack(delta)
            self.attack.increment_round()  # Track rounds for adaptive attacks
            logger.debug(f"Client {self.client_id}: Applied {self.attack.attack_name}")

        avg_loss = total_loss / max(total_samples, 1)

        return delta, {'loss': avg_loss, 'samples': total_samples}
"""

# ============================================================================
# STEP 4: REPLACE aggregation section in FederatedServer
# ============================================================================

ENHANCED_AGGREGATION = """
def aggregate_updates(
    self,
    deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    selected_ids: List[int]
) -> Dict[str, torch.Tensor]:
    '''
    Aggregate client updates using specified method.

    Args:
        deltas: List of client model updates
        weights: Client weights (typically by data size)
        selected_ids: IDs of selected clients

    Returns:
        Aggregated update
    '''
    num_byzantine = sum(1 for cid in selected_ids if self.clients[cid].is_byzantine)

    # Standard FedAvg
    if self.method == 'fedavg':
        aggregated = fedavg_aggregate(deltas, weights)

    # Krum (single or multi)
    elif self.method == 'krum':
        aggregator_fn = get_aggregator('krum', num_byzantine=num_byzantine, multi_krum=False)
        aggregated = aggregator_fn(deltas, num_byzantine=num_byzantine)

    elif self.method == 'multi_krum':
        aggregator_fn = get_aggregator('krum', num_byzantine=num_byzantine, multi_krum=True)
        aggregated = aggregator_fn(deltas, num_byzantine=num_byzantine, multi_krum=True)

    # Bulyan
    elif self.method == 'bulyan':
        aggregator_fn = get_aggregator('bulyan', num_byzantine=num_byzantine)
        aggregated = aggregator_fn(deltas, num_byzantine=num_byzantine)

    # Trimmed Mean
    elif self.method == 'trimmed_mean':
        trim_ratio = num_byzantine / len(deltas) if len(deltas) > 0 else 0.1
        aggregator_fn = get_aggregator('trimmed_mean', trim_ratio=trim_ratio)
        aggregated = aggregator_fn(deltas, trim_ratio=trim_ratio)

    # Coordinate-wise Median
    elif self.method == 'median':
        aggregator_fn = get_aggregator('median')
        aggregated = aggregator_fn(deltas)

    # FLTrust (requires server validation set)
    elif self.method == 'fltrust':
        # Compute server update on validation set
        server_update = self._compute_server_update()
        aggregator_fn = get_aggregator('fltrust', server_update=server_update)
        aggregated = aggregator_fn(deltas, server_update=server_update)

    # Centered Clipping
    elif self.method == 'centered_clipping':
        aggregator_fn = get_aggregator('centered_clipping', clip_threshold=2.0)
        aggregated = aggregator_fn(deltas, clip_threshold=2.0)

    # DSAIN/ByzFed
    elif self.method == 'dsain':
        aggregated = byzfed_aggregate(deltas, weights, self.config.byzantine_threshold)

    else:
        raise ValueError(f"Unknown aggregation method: {self.method}")

    return aggregated

def _compute_server_update(self) -> Dict[str, torch.Tensor]:
    '''Compute server update on validation set (for FLTrust).'''
    # Use a small held-out validation set
    # For simplicity, use test set (in practice, separate validation set)
    self.model.train()
    optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.config.learning_rate,
        momentum=self.config.momentum
    )
    criterion = nn.CrossEntropyLoss()

    # Save current state
    original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

    # Train for 1 epoch on validation set
    for data, target in self.test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = self.model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        break  # Just one batch for server update

    # Compute delta
    server_update = {}
    with torch.no_grad():
        for name, param in self.model.named_parameters():
            server_update[name] = param.data.clone() - original_state[name]

    # Restore original state
    self.model.load_state_dict(original_state)

    return server_update
"""

# ============================================================================
# STEP 5: UPDATE train_round method
# ============================================================================

UPDATED_TRAIN_ROUND = """
def train_round(self) -> Dict:
    '''Execute one training round with enhanced aggregation.'''
    selected_ids = self.select_clients()

    # Collect client updates
    deltas = []
    weights = []
    total_loss = 0
    total_samples = 0

    for cid in selected_ids:
        client = self.clients[cid]
        delta, metrics = client.local_train(self.model)

        # Apply compression (DSAIN only)
        if self.method == 'dsain' and self.config.compression_ratio < 1.0:
            delta = topk_compress(delta, self.config.compression_ratio)

        # Apply DP noise (DSAIN only)
        if self.method == 'dsain' and self.noise_scale > 0:
            delta = add_dp_noise(delta, self.config.gradient_clip, self.noise_scale)

        deltas.append(delta)
        weights.append(metrics['samples'])
        total_loss += metrics['loss'] * metrics['samples']
        total_samples += metrics['samples']

    # Aggregate using appropriate method
    aggregated = self.aggregate_updates(deltas, weights, selected_ids)

    # Update global model
    with torch.no_grad():
        for name, param in self.model.named_parameters():
            param.add_(aggregated[name].to(DEVICE))

    # Count Byzantine clients selected
    num_byzantine_selected = sum(1 for cid in selected_ids if self.clients[cid].is_byzantine)

    return {
        'train_loss': total_loss / max(total_samples, 1),
        'num_participants': len(selected_ids),
        'num_byzantine_selected': num_byzantine_selected
    }
"""

# ============================================================================
# STEP 6: HELPER FUNCTIONS
# ============================================================================

HELPER_FUNCTIONS = """
def fedavg_aggregate(deltas: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    '''Standard FedAvg aggregation.'''
    total_weight = sum(weights)
    aggregated = {}

    for name in deltas[0].keys():
        aggregated[name] = sum(w * d[name] for d, w in zip(deltas, weights)) / total_weight

    return aggregated


def byzfed_aggregate(
    deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
    threshold: float = 5.0
) -> Dict[str, torch.Tensor]:
    '''ByzFed aggregation from dsain.py, adapted for PyTorch.'''
    n = len(deltas)

    if n < 3:
        # Not enough clients for robust aggregation
        return fedavg_aggregate(deltas, weights)

    # Flatten for geometric median computation
    flat_deltas = []
    for delta in deltas:
        flat = torch.cat([d.flatten().float() for d in delta.values()])
        flat_deltas.append(flat)

    # Compute geometric median
    median = geometric_median_torch(flat_deltas)

    # Compute distances from median
    distances = [torch.norm(f - median).item() for f in flat_deltas]

    # Robust scale estimate (MAD)
    median_dist = np.median(distances)

    if median_dist < 1e-8:
        return fedavg_aggregate(deltas, weights)

    scale = median_dist * 1.4826  # MAD to std conversion

    # Filter outliers
    filtered_indices = [i for i, d in enumerate(distances) if d <= threshold * scale]

    if len(filtered_indices) < n // 2:
        filtered_indices = list(range(n))

    # Weighted average of filtered updates
    filtered_deltas = [deltas[i] for i in filtered_indices]
    filtered_weights = [weights[i] for i in filtered_indices]

    return fedavg_aggregate(filtered_deltas, filtered_weights)


def geometric_median_torch(points: List[torch.Tensor], max_iter: int = 100, tol: float = 1e-6) -> torch.Tensor:
    '''Compute geometric median using Weiszfeld algorithm.'''
    if len(points) == 0:
        raise ValueError("No points provided")
    if len(points) == 1:
        return points[0].clone()

    stacked = torch.stack(points)
    y = stacked.mean(dim=0)

    for _ in range(max_iter):
        distances = torch.stack([torch.norm(p - y).clamp(min=1e-10) for p in points])
        weights = 1.0 / distances
        weights = weights / weights.sum()
        y_new = sum(w * p for w, p in zip(weights, points))

        if torch.norm(y_new - y) < tol:
            break
        y = y_new

    return y
"""

# ============================================================================
# STEP 7: COMPLETE INTEGRATION EXAMPLE
# ============================================================================

def print_integration_guide():
    """Print step-by-step integration guide."""
    guide = """
    ═══════════════════════════════════════════════════════════════════
    INTEGRATION GUIDE: real_experiments.py Enhancement
    ═══════════════════════════════════════════════════════════════════

    Follow these steps to integrate Byzantine attacks and baseline aggregators:

    STEP 1: Add Imports
    ────────────────────
    At the top of real_experiments.py, add:

        from byzantine_attacks import get_byzantine_attack
        from baseline_aggregators import get_aggregator


    STEP 2: Update FLConfig
    ───────────────────────
    Add attack configuration fields:

        attack_type: str = 'sign_flipping'
        attack_scale: float = 5.0


    STEP 3: Enhance FederatedClient.__init__
    ────────────────────────────────────────
    Add parameters:
        num_byzantine: int = 0
        num_honest: int = 0

    Initialize attack:
        if is_byzantine:
            self.attack = get_byzantine_attack(...)


    STEP 4: Modify FederatedClient.local_train
    ───────────────────────────────────────────
    After computing delta, before returning:

        if self.is_byzantine and self.attack is not None:
            delta = self.attack.attack(delta)
            self.attack.increment_round()


    STEP 5: Add aggregate_updates method to FederatedServer
    ────────────────────────────────────────────────────────
    Replace simple aggregation with method dispatch.
    See ENHANCED_AGGREGATION above.


    STEP 6: Update run_cifar10_experiment
    ──────────────────────────────────────
    When creating clients, pass num_byzantine and num_honest:

        for i, indices in enumerate(client_indices):
            client = FederatedClient(
                client_id=i,
                train_loader=loader,
                model_fn=lambda: SimpleCNN(num_classes=10),
                config=config,
                is_byzantine=(i in byzantine_ids),
                num_byzantine=num_byzantine,
                num_honest=num_clients - num_byzantine
            )


    STEP 7: Add new experiment modes
    ─────────────────────────────────
    Add to argparse:
        --attack_type (choices: sign_flipping, little_is_enough, etc.)
        --aggregation_method (choices: fedavg, krum, bulyan, dsain, etc.)


    TESTING:
    ────────
    Test with small configuration first:

        python real_experiments.py --mode single --method bulyan \\
            --attack_type little_is_enough --byzantine_frac 0.2 \\
            --num_clients 20 --num_rounds 30

    Expected: Should run without errors and show Byzantine resilience.

    ═══════════════════════════════════════════════════════════════════
    """
    print(guide)


if __name__ == "__main__":
    print_integration_guide()

    print("\n" + "="*70)
    print("Code snippets available in this file:")
    print("="*70)
    print("1. ENHANCED_FL_CONFIG")
    print("2. ENHANCED_CLIENT")
    print("3. ENHANCED_AGGREGATION")
    print("4. UPDATED_TRAIN_ROUND")
    print("5. HELPER_FUNCTIONS")
    print("\nCopy-paste these into real_experiments.py as needed.")
    print("="*70)
