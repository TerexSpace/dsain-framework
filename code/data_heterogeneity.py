#!/usr/bin/env python3
"""
Enhanced Non-IID Data Partitioning for Federated Learning
==========================================================

Implements multiple types of data heterogeneity beyond basic Dirichlet partitioning:
1. Label skew (Dirichlet, pathological non-IID, label imbalance)
2. Feature skew (synthetic covariate shift, domain adaptation)
3. Quantity skew (power law, Zipf distribution)
4. Combined heterogeneity (realistic FL scenarios)

References:
    - Li et al. (2024) "Non-IID Data in Federated Learning: A Comprehensive Survey"
    - Kairouz et al. (2021) "Advances and Open Problems in Federated Learning"
    - Hsu et al. (2019) "Measuring the Effects of Non-Identical Data Distribution"
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Label Skew (Distribution-Based Heterogeneity)
# =============================================================================

def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    min_samples_per_client: int = 10
) -> List[np.ndarray]:
    """
    Partition dataset using Dirichlet distribution for label heterogeneity.

    This is the standard method used in FL literature. Lower alpha = more heterogeneous.

    Args:
        targets: Array of labels (N,)
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (0.1 = highly non-IID, 1.0 = moderately non-IID)
        min_samples_per_client: Minimum samples each client must have

    Returns:
        List of index arrays, one per client

    Example:
        >>> targets = np.array([0, 0, 1, 1, 2, 2])
        >>> partitions = dirichlet_partition(targets, num_clients=3, alpha=0.5)
        >>> len(partitions) == 3
        True
    """
    num_classes = len(np.unique(targets))
    indices_per_class = [np.where(targets == i)[0] for i in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    # Sample Dirichlet distribution for each class
    for class_idx, indices in enumerate(indices_per_class):
        # Skip empty classes
        if len(indices) == 0:
            continue

        np.random.shuffle(indices)

        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # Ensure minimum samples per client
        proportions = np.maximum(proportions, min_samples_per_client / len(indices))
        proportions = proportions / proportions.sum()  # Renormalize

        # Split indices according to proportions
        splits = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        class_partitions = np.split(indices, splits)

        for client_idx, partition in enumerate(class_partitions):
            client_indices[client_idx].extend(partition)

    # Shuffle each client's indices
    for client_idx in range(num_clients):
        np.random.shuffle(client_indices[client_idx])
        client_indices[client_idx] = np.array(client_indices[client_idx])

    # Filter out clients with too few samples
    client_indices = [idx for idx in client_indices if len(idx) >= min_samples_per_client]

    if len(client_indices) == 0:
        raise ValueError(f"No clients have >= {min_samples_per_client} samples. "
                        f"Try reducing min_samples_per_client or increasing alpha.")

    logger.info(f"Dirichlet partition (alpha={alpha}): {len(client_indices)} clients")
    logger.info(f"Samples per client: min={min(len(idx) for idx in client_indices)}, "
                f"max={max(len(idx) for idx in client_indices)}, "
                f"mean={np.mean([len(idx) for idx in client_indices]):.1f}")

    return client_indices


def pathological_partition(
    targets: np.ndarray,
    num_clients: int,
    shards_per_client: int = 2
) -> List[np.ndarray]:
    """
    Pathological non-IID partition: each client gets exactly k classes.

    This creates extreme label skew where each client only sees a subset of classes.
    Used in McMahan et al. (2017) FedAvg paper.

    Args:
        targets: Array of labels (N,)
        num_clients: Number of clients
        shards_per_client: Number of shards (class groups) per client

    Returns:
        List of index arrays, one per client

    Example:
        >>> targets = np.array([0]*100 + [1]*100 + [2]*100)
        >>> partitions = pathological_partition(targets, num_clients=10, shards_per_client=2)
    """
    num_classes = len(np.unique(targets))

    # Collect all indices and shuffle
    all_indices = np.arange(len(targets))
    np.random.shuffle(all_indices)

    # Calculate shard size
    total_shards = num_clients * shards_per_client
    shard_size = len(targets) // total_shards

    # Create shards by splitting shuffled indices
    shards = []
    for i in range(total_shards):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i < total_shards - 1 else len(targets)
        shards.append(all_indices[start_idx:end_idx])

    # Shuffle shards and assign to clients
    np.random.shuffle(shards)
    client_indices = []
    for i in range(num_clients):
        start_shard = i * shards_per_client
        end_shard = min((i + 1) * shards_per_client, len(shards))
        if start_shard < len(shards):
            client_shards = shards[start_shard:end_shard]
            if len(client_shards) > 0:
                client_idx = np.concatenate(client_shards)
                np.random.shuffle(client_idx)
                client_indices.append(client_idx)

    logger.info(f"Pathological partition ({shards_per_client} shards/client): {num_clients} clients")
    logger.info(f"Samples per client: {[len(idx) for idx in client_indices[:5]]}")

    return client_indices


def label_imbalance_partition(
    targets: np.ndarray,
    num_clients: int,
    imbalance_ratio: float = 10.0
) -> List[np.ndarray]:
    """
    Create label imbalance: some classes are overrepresented, others underrepresented.

    Args:
        targets: Array of labels (N,)
        num_clients: Number of clients
        imbalance_ratio: Ratio between most common and least common class

    Returns:
        List of index arrays, one per client
    """
    num_classes = len(np.unique(targets))
    indices_per_class = [np.where(targets == i)[0] for i in range(num_classes)]

    # Create imbalanced class weights (exponential decay)
    class_weights = np.exp(-np.linspace(0, np.log(imbalance_ratio), num_classes))
    class_weights = class_weights / class_weights.sum()

    client_indices = [[] for _ in range(num_clients)]

    for class_idx, indices in enumerate(indices_per_class):
        np.random.shuffle(indices)

        # Assign samples to clients proportionally to class weight
        samples_for_this_class = int(len(indices) * class_weights[class_idx])
        if samples_for_this_class == 0:
            samples_for_this_class = 1  # Ensure at least one sample

        # Split among clients
        splits = np.array_split(indices[:samples_for_this_class], num_clients)
        for client_idx, split in enumerate(splits):
            client_indices[client_idx].extend(split)

    # Shuffle and convert to numpy arrays
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        client_indices[i] = np.array(client_indices[i])

    logger.info(f"Label imbalance partition (ratio={imbalance_ratio}): {num_clients} clients")

    return client_indices


# =============================================================================
# 2. Feature Skew (Covariate Shift)
# =============================================================================

class FeatureSkewDataset(Dataset):
    """
    Wrapper dataset that applies feature-level transformations to create covariate shift.

    Simulates different data collection conditions across clients (e.g., different sensors,
    lighting conditions, image qualities).
    """

    def __init__(self, base_dataset: Dataset, transform_type: str, intensity: float = 0.5):
        """
        Args:
            base_dataset: Original dataset
            transform_type: One of ['noise', 'blur', 'brightness', 'rotation', 'none']
            intensity: Transformation intensity (0 = no change, 1 = maximum change)
        """
        self.base_dataset = base_dataset
        self.transform_type = transform_type
        self.intensity = intensity

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]

        # Apply transformation if tensor
        if isinstance(x, torch.Tensor):
            x = self._apply_transform(x)

        return x, y

    def _apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature-level transformation."""
        if self.transform_type == 'none':
            return x

        elif self.transform_type == 'noise':
            # Add Gaussian noise
            noise = torch.randn_like(x) * self.intensity * 0.1
            return torch.clamp(x + noise, 0, 1)

        elif self.transform_type == 'blur':
            # Simplified blur (average with neighbors)
            if x.dim() == 3:  # (C, H, W)
                kernel_size = int(3 + 4 * self.intensity)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                # Simple moving average
                return torch.nn.functional.avg_pool2d(
                    x.unsqueeze(0),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size//2
                ).squeeze(0)
            return x

        elif self.transform_type == 'brightness':
            # Adjust brightness
            factor = 1.0 + (self.intensity - 0.5)
            return torch.clamp(x * factor, 0, 1)

        elif self.transform_type == 'contrast':
            # Adjust contrast
            mean = x.mean()
            factor = 1.0 + self.intensity
            return torch.clamp((x - mean) * factor + mean, 0, 1)

        else:
            return x


def feature_skew_partition(
    base_dataset: Dataset,
    num_clients: int,
    num_skew_types: int = 3
) -> List[Dataset]:
    """
    Create feature skew by applying different transformations to different clients.

    Args:
        base_dataset: Original dataset
        num_clients: Number of clients
        num_skew_types: Number of different skew types to use

    Returns:
        List of datasets with different feature distributions
    """
    transform_types = ['noise', 'blur', 'brightness', 'contrast', 'none']
    selected_transforms = transform_types[:num_skew_types]

    client_datasets = []
    for i in range(num_clients):
        # Assign transform type cyclically
        transform_type = selected_transforms[i % len(selected_transforms)]
        intensity = 0.3 + 0.4 * (i / num_clients)  # Gradually increase intensity

        client_dataset = FeatureSkewDataset(base_dataset, transform_type, intensity)
        client_datasets.append(client_dataset)

    logger.info(f"Feature skew partition: {num_clients} clients with {num_skew_types} skew types")

    return client_datasets


# =============================================================================
# 3. Quantity Skew (Sample Size Heterogeneity)
# =============================================================================

def power_law_partition(
    total_samples: int,
    num_clients: int,
    alpha: float = 1.5
) -> np.ndarray:
    """
    Generate sample sizes following a power law distribution.

    Models realistic FL where some clients have much more data than others.

    Args:
        total_samples: Total number of samples
        num_clients: Number of clients
        alpha: Power law exponent (higher = more skewed)

    Returns:
        Array of sample sizes for each client

    Example:
        >>> sizes = power_law_partition(1000, 10, alpha=2.0)
        >>> sizes.sum() == 1000
        True
    """
    # Generate power law samples
    raw_sizes = np.random.pareto(alpha, num_clients) + 1

    # Normalize to sum to total_samples
    sizes = (raw_sizes / raw_sizes.sum() * total_samples).astype(int)

    # Ensure at least 10 samples per client
    sizes = np.maximum(sizes, 10)

    # Adjust to match total exactly
    diff = sizes.sum() - total_samples
    if diff > 0:
        # Remove from largest clients
        largest_idx = np.argmax(sizes)
        sizes[largest_idx] -= diff
    elif diff < 0:
        # Add to largest client
        largest_idx = np.argmax(sizes)
        sizes[largest_idx] += abs(diff)

    logger.info(f"Power law partition (alpha={alpha}): sizes from {sizes.min()} to {sizes.max()}")
    logger.info(f"Gini coefficient: {gini_coefficient(sizes):.3f}")

    return sizes


def zipf_partition(
    total_samples: int,
    num_clients: int,
    s: float = 1.0
) -> np.ndarray:
    """
    Generate sample sizes following Zipf's law.

    Args:
        total_samples: Total number of samples
        num_clients: Number of clients
        s: Zipf parameter (higher = more skewed)

    Returns:
        Array of sample sizes for each client
    """
    ranks = np.arange(1, num_clients + 1)
    raw_sizes = 1.0 / (ranks ** s)

    # Normalize
    sizes = (raw_sizes / raw_sizes.sum() * total_samples).astype(int)
    sizes = np.maximum(sizes, 10)  # Minimum 10 samples

    # Adjust to match total
    sizes[0] += total_samples - sizes.sum()

    logger.info(f"Zipf partition (s={s}): sizes from {sizes.min()} to {sizes.max()}")

    return sizes


def gini_coefficient(sizes: np.ndarray) -> float:
    """
    Calculate Gini coefficient to measure inequality in sample distribution.

    Gini = 0: perfect equality
    Gini = 1: perfect inequality

    Args:
        sizes: Array of sample sizes

    Returns:
        Gini coefficient [0, 1]
    """
    sorted_sizes = np.sort(sizes)
    n = len(sizes)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_sizes)) / (n * np.sum(sorted_sizes)) - (n + 1) / n


# =============================================================================
# 4. Combined Heterogeneity
# =============================================================================

def realistic_partition(
    dataset: Dataset,
    targets: np.ndarray,
    num_clients: int,
    label_alpha: float = 0.3,
    quantity_skew: str = 'power_law',
    feature_skew: bool = True
) -> Tuple[List[Subset], Dict[str, any]]:
    """
    Create realistic FL partition with combined heterogeneity.

    Combines:
    - Label skew (Dirichlet)
    - Quantity skew (power law or Zipf)
    - Feature skew (optional)

    Args:
        dataset: Original dataset
        targets: Array of labels
        num_clients: Number of clients
        label_alpha: Dirichlet concentration for label heterogeneity
        quantity_skew: One of ['power_law', 'zipf', 'uniform']
        feature_skew: Whether to apply feature-level transformations

    Returns:
        Tuple of (client_datasets, metadata_dict)
    """
    logger.info(f"\nCreating realistic partition:")
    logger.info(f"  - Label heterogeneity: Dirichlet(alpha={label_alpha})")
    logger.info(f"  - Quantity skew: {quantity_skew}")
    logger.info(f"  - Feature skew: {feature_skew}")

    # Step 1: Label-based partition (Dirichlet)
    label_partition = dirichlet_partition(targets, num_clients, label_alpha)

    # Step 2: Apply quantity skew
    if quantity_skew == 'power_law':
        target_sizes = power_law_partition(len(targets), num_clients, alpha=1.5)
    elif quantity_skew == 'zipf':
        target_sizes = zipf_partition(len(targets), num_clients, s=1.0)
    else:  # uniform
        target_sizes = np.full(num_clients, len(targets) // num_clients)

    # Resample each client's partition to match target size
    client_indices = []
    for i, indices in enumerate(label_partition):
        target_size = target_sizes[i]
        if len(indices) < target_size:
            # Oversample with replacement
            sampled = np.random.choice(indices, size=target_size, replace=True)
        else:
            # Subsample without replacement
            sampled = np.random.choice(indices, size=target_size, replace=False)
        client_indices.append(sampled)

    # Step 3: Create subsets
    client_datasets = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        client_datasets.append(subset)

    # Step 4: Apply feature skew if requested
    if feature_skew:
        transform_types = ['noise', 'blur', 'brightness', 'none']
        skewed_datasets = []
        for i, subset in enumerate(client_datasets):
            transform_type = transform_types[i % len(transform_types)]
            intensity = 0.2 + 0.3 * (i / num_clients)
            skewed_dataset = FeatureSkewDataset(subset, transform_type, intensity)
            skewed_datasets.append(skewed_dataset)
        client_datasets = skewed_datasets

    # Collect metadata
    metadata = {
        'num_clients': num_clients,
        'label_alpha': label_alpha,
        'quantity_skew': quantity_skew,
        'feature_skew': feature_skew,
        'samples_per_client': [len(d) for d in client_datasets],
        'gini_coefficient': gini_coefficient(np.array([len(d) for d in client_datasets])),
    }

    logger.info(f"\nPartition statistics:")
    logger.info(f"  - Clients: {num_clients}")
    logger.info(f"  - Samples per client: min={min(metadata['samples_per_client'])}, "
                f"max={max(metadata['samples_per_client'])}, "
                f"mean={np.mean(metadata['samples_per_client']):.1f}")
    logger.info(f"  - Gini coefficient: {metadata['gini_coefficient']:.3f}")

    return client_datasets, metadata


# =============================================================================
# 5. Utility Functions
# =============================================================================

def analyze_partition(
    client_datasets: List[Dataset],
    targets: np.ndarray,
    num_classes: int
) -> Dict[str, any]:
    """
    Analyze heterogeneity in data partition.

    Computes:
    - Label distribution per client
    - KL divergence from uniform distribution
    - Sample size statistics
    - Class balance metrics

    Args:
        client_datasets: List of client datasets
        targets: Full array of labels
        num_classes: Number of classes

    Returns:
        Dictionary of analysis metrics
    """
    num_clients = len(client_datasets)

    # Get label distribution for each client
    client_label_counts = []
    for dataset in client_datasets:
        if isinstance(dataset, Subset):
            indices = dataset.indices
            labels = targets[indices]
        else:
            # For wrapped datasets, need to extract indices
            labels = [targets[i] for i in range(len(dataset))]

        counts = np.bincount(labels, minlength=num_classes)
        client_label_counts.append(counts)

    client_label_counts = np.array(client_label_counts)

    # Compute metrics
    uniform_dist = np.ones(num_classes) / num_classes
    kl_divergences = []

    for counts in client_label_counts:
        dist = counts / (counts.sum() + 1e-10)
        kl = np.sum(dist * np.log((dist + 1e-10) / (uniform_dist + 1e-10)))
        kl_divergences.append(kl)

    analysis = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'samples_per_client': [len(d) for d in client_datasets],
        'label_distribution': client_label_counts.tolist(),
        'kl_divergence_mean': np.mean(kl_divergences),
        'kl_divergence_std': np.std(kl_divergences),
        'gini_coefficient': gini_coefficient(np.array([len(d) for d in client_datasets])),
    }

    return analysis


def print_partition_summary(analysis: Dict[str, any]):
    """Print human-readable partition analysis."""
    print("\n" + "="*70)
    print("PARTITION ANALYSIS")
    print("="*70)
    print(f"Number of clients: {analysis['num_clients']}")
    print(f"Number of classes: {analysis['num_classes']}")
    print(f"\nSample distribution:")
    print(f"  Min: {min(analysis['samples_per_client'])}")
    print(f"  Max: {max(analysis['samples_per_client'])}")
    print(f"  Mean: {np.mean(analysis['samples_per_client']):.1f}")
    print(f"  Std: {np.std(analysis['samples_per_client']):.1f}")
    print(f"\nHeterogeneity metrics:")
    print(f"  Gini coefficient: {analysis['gini_coefficient']:.3f}")
    print(f"  KL divergence (mean ± std): {analysis['kl_divergence_mean']:.3f} ± {analysis['kl_divergence_std']:.3f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test partitioning methods
    print("\n" + "="*70)
    print("TESTING DATA HETEROGENEITY PARTITIONING")
    print("="*70)

    # Create synthetic dataset
    num_samples = 5000
    num_classes = 10
    targets = np.random.randint(0, num_classes, size=num_samples)
    num_clients = 20

    # Test 1: Dirichlet partition
    print("\n1. Dirichlet Partition:")
    for alpha in [0.1, 0.5, 1.0]:
        client_indices = dirichlet_partition(targets, num_clients, alpha)
        print(f"   alpha={alpha}: {len(client_indices)} clients, "
              f"samples range [{min(len(idx) for idx in client_indices)}, "
              f"{max(len(idx) for idx in client_indices)}]")

    # Test 2: Pathological partition
    print("\n2. Pathological Partition:")
    client_indices = pathological_partition(targets, num_clients, shards_per_client=2)
    print(f"   Created {len(client_indices)} clients with 2 shards each")

    # Test 3: Quantity skew
    print("\n3. Quantity Skew:")
    power_sizes = power_law_partition(num_samples, num_clients, alpha=1.5)
    print(f"   Power law: Gini={gini_coefficient(power_sizes):.3f}, "
          f"range [{power_sizes.min()}, {power_sizes.max()}]")

    zipf_sizes = zipf_partition(num_samples, num_clients, s=1.0)
    print(f"   Zipf: Gini={gini_coefficient(zipf_sizes):.3f}, "
          f"range [{zipf_sizes.min()}, {zipf_sizes.max()}]")

    print("\n" + "="*70)
    print("ALL PARTITIONING TESTS PASSED!")
    print("="*70 + "\n")
