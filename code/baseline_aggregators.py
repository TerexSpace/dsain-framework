#!/usr/bin/env python3
"""
Byzantine-Resilient Aggregation Baselines
==========================================

Implementations of state-of-the-art Byzantine-resilient aggregation methods
for comparison with DSAIN/ByzFed.

Methods Implemented:
- Krum (Blanchard et al., 2017)
- Bulyan (Mhamdi et al., 2018)
- Trimmed Mean (Yin et al., 2018)
- Median (Yin et al., 2018)
- FLTrust (Cao et al., 2021)
- Centered Clipping (Karimireddy et al., 2021)

Author: Almas Ospanov
License: MIT
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def krum_aggregation(
    deltas: List[Dict[str, torch.Tensor]],
    num_byzantine: int,
    multi_krum: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Krum aggregation (Blanchard et al., 2017).

    Selects the update closest to its neighbors (measured by sum of distances
    to n-f-2 nearest neighbors).

    Args:
        deltas: List of client updates
        num_byzantine: Number of Byzantine clients (f)
        multi_krum: If True, average top k updates instead of selecting one

    Returns:
        Aggregated update
    """
    n = len(deltas)
    f = num_byzantine

    # Flatten deltas for distance computation
    flat_deltas = []
    for delta in deltas:
        flat = torch.cat([d.flatten() for d in delta.values()])
        flat_deltas.append(flat)

    # Compute pairwise distances
    distances = torch.zeros(n, n)
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(flat_deltas[i] - flat_deltas[j]).item()
            distances[i, j] = dist
            distances[j, i] = dist

    # Krum score: sum of n-f-2 closest distances
    scores = []
    for i in range(n):
        sorted_dists = torch.sort(distances[i])[0]
        # Exclude self (0) and f+1 furthest
        score = sorted_dists[1:n-f-1].sum().item()
        scores.append(score)

    if not multi_krum:
        # Select single update with minimum score
        selected = np.argmin(scores)
        return deltas[selected]
    else:
        # Multi-Krum: average top (n-f) updates
        num_selected = n - f
        top_indices = np.argsort(scores)[:num_selected]

        # Average selected updates
        aggregated = {}
        for name in deltas[0].keys():
            stacked = torch.stack([deltas[i][name] for i in top_indices])
            aggregated[name] = stacked.mean(dim=0)

        return aggregated


def bulyan_aggregation(
    deltas: List[Dict[str, torch.Tensor]],
    num_byzantine: int
) -> Dict[str, torch.Tensor]:
    """
    Bulyan aggregation (Mhamdi et al., 2018).

    Combines Multi-Krum selection with Trimmed Mean.

    Steps:
    1. Use Multi-Krum to select n-2f updates
    2. Apply coordinate-wise trimmed mean on selected updates

    Args:
        deltas: List of client updates
        num_byzantine: Number of Byzantine clients (f)

    Returns:
        Aggregated update
    """
    n = len(deltas)
    f = num_byzantine

    # Step 1: Multi-Krum selection
    # Select top (n-2f) updates using Krum scoring
    flat_deltas = []
    for delta in deltas:
        flat = torch.cat([d.flatten() for d in delta.values()])
        flat_deltas.append(flat)

    distances = torch.zeros(n, n)
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(flat_deltas[i] - flat_deltas[j]).item()
            distances[i, j] = dist
            distances[j, i] = dist

    scores = []
    for i in range(n):
        sorted_dists = torch.sort(distances[i])[0]
        score = sorted_dists[1:n-f-1].sum().item()
        scores.append(score)

    # Select n-2f updates with best scores
    num_selected = n - 2*f
    selected_indices = np.argsort(scores)[:num_selected]

    # Step 2: Coordinate-wise trimmed mean
    # Trim f/2 from each end
    trim_count = f // 2

    aggregated = {}
    for name in deltas[0].keys():
        selected_tensors = [deltas[i][name] for i in selected_indices]
        stacked = torch.stack(selected_tensors)

        # Coordinate-wise trimming
        if trim_count > 0:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_count:-trim_count]
        else:
            trimmed = stacked

        aggregated[name] = trimmed.mean(dim=0)

    return aggregated


def trimmed_mean_aggregation(
    deltas: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Trimmed mean aggregation (Yin et al., 2018).

    Remove extreme values (top and bottom trim_ratio) before averaging.

    Args:
        deltas: List of client updates
        trim_ratio: Fraction to trim from each end (0 to 0.5)

    Returns:
        Aggregated update
    """
    n = len(deltas)
    trim_count = int(n * trim_ratio)

    aggregated = {}
    for name in deltas[0].keys():
        stacked = torch.stack([d[name] for d in deltas])

        if trim_count > 0:
            # Sort along client dimension
            sorted_vals, _ = torch.sort(stacked, dim=0)
            # Trim top and bottom
            trimmed = sorted_vals[trim_count:n-trim_count]
        else:
            trimmed = stacked

        aggregated[name] = trimmed.mean(dim=0)

    return aggregated


def median_aggregation(
    deltas: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise median aggregation (Yin et al., 2018).

    Most robust but can be slow for high-dimensional models.

    Args:
        deltas: List of client updates

    Returns:
        Aggregated update
    """
    aggregated = {}
    for name in deltas[0].keys():
        stacked = torch.stack([d[name] for d in deltas])
        aggregated[name] = stacked.median(dim=0)[0]

    return aggregated


def fltrust_aggregation(
    deltas: List[Dict[str, torch.Tensor]],
    server_update: Dict[str, torch.Tensor],
    clip_threshold: float = 2.0
) -> Dict[str, torch.Tensor]:
    """
    FLTrust aggregation (Cao et al., 2021).

    Uses a clean server-side validation dataset to bootstrap trust.
    Clients are weighted by cosine similarity to server update.

    Args:
        deltas: List of client updates
        server_update: Update from server's validation set
        clip_threshold: Trust score clipping threshold

    Returns:
        Aggregated update
    """
    # Flatten server update
    server_flat = torch.cat([v.flatten() for v in server_update.values()])
    server_norm = torch.norm(server_flat)

    if server_norm < 1e-10:
        logger.warning("Server update has near-zero norm, falling back to mean")
        return fedavg_aggregation(deltas, weights=None)

    # Compute trust scores (cosine similarity with server)
    trust_scores = []
    for delta in deltas:
        delta_flat = torch.cat([v.flatten() for v in delta.values()])
        delta_norm = torch.norm(delta_flat)

        if delta_norm < 1e-10:
            trust_scores.append(0.0)
        else:
            cosine_sim = torch.dot(server_flat, delta_flat) / (server_norm * delta_norm)
            # ReLU to filter out negative similarities
            trust = max(0.0, cosine_sim.item())
            # Clip to threshold
            trust = min(trust, clip_threshold)
            trust_scores.append(trust)

    # Normalize trust scores
    total_trust = sum(trust_scores)
    if total_trust < 1e-10:
        logger.warning("All clients have zero trust, falling back to mean")
        return fedavg_aggregation(deltas, weights=None)

    weights = [t / total_trust for t in trust_scores]

    # Weighted aggregation
    aggregated = {}
    for name in deltas[0].keys():
        weighted_sum = sum(w * deltas[i][name] for i, w in enumerate(weights))
        aggregated[name] = weighted_sum

    return aggregated


def centered_clipping_aggregation(
    deltas: List[Dict[str, torch.Tensor]],
    clip_threshold: float = 2.0
) -> Dict[str, torch.Tensor]:
    """
    Centered clipping aggregation (Karimireddy et al., 2021).

    Clip updates that are too far from the median.

    Args:
        deltas: List of client updates
        clip_threshold: Clipping threshold (in units of MAD)

    Returns:
        Aggregated update
    """
    # Compute median
    median_update = median_aggregation(deltas)

    # Flatten for distance computation
    median_flat = torch.cat([v.flatten() for v in median_update.values()])

    # Compute distances to median
    distances = []
    for delta in deltas:
        delta_flat = torch.cat([v.flatten() for v in delta.values()])
        dist = torch.norm(delta_flat - median_flat).item()
        distances.append(dist)

    # Compute MAD (Median Absolute Deviation)
    mad = np.median(distances)
    if mad < 1e-10:
        mad = 1.0  # Avoid division by zero

    # Clip updates
    clipped_deltas = []
    for delta, dist in zip(deltas, distances):
        if dist <= clip_threshold * mad:
            clipped_deltas.append(delta)
        else:
            # Clip towards median
            clip_factor = (clip_threshold * mad) / dist
            clipped = {}
            for name in delta.keys():
                median_val = median_update[name]
                clipped[name] = median_val + clip_factor * (delta[name] - median_val)
            clipped_deltas.append(clipped)

    # Average clipped updates
    return fedavg_aggregation(clipped_deltas, weights=None)


def fedavg_aggregation(
    deltas: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None
) -> Dict[str, torch.Tensor]:
    """
    Standard FedAvg aggregation (baseline, not Byzantine-resilient).

    Args:
        deltas: List of client updates
        weights: Optional weights (default: uniform)

    Returns:
        Aggregated update
    """
    if weights is None:
        weights = [1.0 / len(deltas)] * len(deltas)

    aggregated = {}
    for name in deltas[0].keys():
        weighted_sum = sum(w * delta[name] for w, delta in zip(weights, deltas))
        aggregated[name] = weighted_sum

    return aggregated


# Factory function
def get_aggregator(method: str, **kwargs):
    """
    Factory function for aggregation methods.

    Args:
        method: Name of aggregation method
        **kwargs: Method-specific parameters

    Returns:
        Aggregation function
    """
    methods = {
        'fedavg': lambda deltas, **kw: fedavg_aggregation(
            deltas, weights=kw.get('weights')
        ),
        'krum': lambda deltas, **kw: krum_aggregation(
            deltas,
            num_byzantine=kw.get('num_byzantine', 1),
            multi_krum=kw.get('multi_krum', False)
        ),
        'bulyan': lambda deltas, **kw: bulyan_aggregation(
            deltas,
            num_byzantine=kw.get('num_byzantine', 1)
        ),
        'trimmed_mean': lambda deltas, **kw: trimmed_mean_aggregation(
            deltas,
            trim_ratio=kw.get('trim_ratio', 0.1)
        ),
        'median': lambda deltas, **kw: median_aggregation(deltas),
        'fltrust': lambda deltas, **kw: fltrust_aggregation(
            deltas,
            server_update=kw.get('server_update'),
            clip_threshold=kw.get('clip_threshold', 2.0)
        ),
        'centered_clipping': lambda deltas, **kw: centered_clipping_aggregation(
            deltas,
            clip_threshold=kw.get('clip_threshold', 2.0)
        ),
    }

    if method not in methods:
        raise ValueError(f"Unknown aggregation method: {method}. "
                        f"Available: {list(methods.keys())}")

    return methods[method]
