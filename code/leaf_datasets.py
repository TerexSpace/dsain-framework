#!/usr/bin/env python3
"""
LEAF Dataset Loaders for DSAIN Framework
=========================================

This module provides loaders for real federated datasets from the LEAF benchmark:
- FEMNIST: Federated EMNIST (handwritten characters, 62 classes, 3,550 users)
- Shakespeare: Next character prediction from The Complete Works of Shakespeare

These datasets provide realistic non-IID federated learning scenarios.

References:
    LEAF: A Benchmark for Federated Settings
    Caldas et al., 2018
    https://leaf.cmu.edu/

Requirements:
    pip install numpy pillow requests

Usage:
    from leaf_datasets import FEMNISTLoader, ShakespeareLoader
    
    # Load FEMNIST
    femnist = FEMNISTLoader(data_dir='./data/femnist')
    clients_data = femnist.load()
    
    # Load Shakespeare
    shakespeare = ShakespeareLoader(data_dir='./data/shakespeare')
    clients_data = shakespeare.load()
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientData:
    """Container for a single client's data."""
    client_id: str
    x: np.ndarray  # Features
    y: np.ndarray  # Labels
    num_samples: int
    
    def __post_init__(self):
        self.num_samples = len(self.y)


class LEAFBaseLoader:
    """Base class for LEAF dataset loaders."""
    
    DATASET_NAME = "base"
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _download_if_needed(self) -> bool:
        """Check if data exists, provide instructions if not."""
        raise NotImplementedError
        
    def load(self, max_clients: Optional[int] = None) -> List[ClientData]:
        """Load dataset and return list of ClientData objects."""
        raise NotImplementedError
        
    def get_statistics(self, clients: List[ClientData]) -> Dict:
        """Compute dataset statistics."""
        num_clients = len(clients)
        total_samples = sum(c.num_samples for c in clients)
        samples_per_client = [c.num_samples for c in clients]
        
        return {
            'num_clients': num_clients,
            'total_samples': total_samples,
            'mean_samples_per_client': np.mean(samples_per_client),
            'std_samples_per_client': np.std(samples_per_client),
            'min_samples': min(samples_per_client),
            'max_samples': max(samples_per_client)
        }


class FEMNISTLoader(LEAFBaseLoader):
    """
    Federated EMNIST (FEMNIST) Dataset Loader.
    
    FEMNIST is a federated variant of the Extended MNIST dataset where data is
    partitioned by writer. This creates natural non-IID splits based on
    handwriting styles.
    
    - 62 classes: digits (0-9), uppercase letters (A-Z), lowercase letters (a-z)
    - ~3,550 users
    - ~805,263 samples total
    - Images are 28x28 grayscale
    
    The dataset exhibits natural heterogeneity as each user has their own
    handwriting style, making it ideal for federated learning research.
    """
    
    DATASET_NAME = "femnist"
    IMAGE_SIZE = 28
    NUM_CLASSES = 62
    
    def __init__(self, data_dir: str = "./data/femnist"):
        super().__init__(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        
    def _download_if_needed(self) -> bool:
        """
        Check if FEMNIST data exists.
        
        Note: LEAF datasets require cloning the repository and running
        preprocessing scripts. We provide instructions here.
        """
        if not self.train_dir.exists():
            logger.warning(f"FEMNIST data not found at {self.data_dir}")
            logger.info("=" * 60)
            logger.info("To download FEMNIST, run:")
            logger.info("  git clone https://github.com/TalwalkarLab/leaf.git")
            logger.info("  cd leaf/data/femnist")
            logger.info("  ./preprocess.sh -s niid --sf 0.1 -k 0 -t sample")
            logger.info("Then copy data/femnist/data to this directory")
            logger.info("=" * 60)
            return False
        return True
    
    def _generate_synthetic_femnist(self, 
                                     num_clients: int = 100,
                                     samples_per_client: Tuple[int, int] = (50, 500),
                                     seed: int = 42) -> List[ClientData]:
        """
        Generate synthetic FEMNIST-like data for testing when real data unavailable.
        
        This creates non-IID data by:
        1. Each client has a preferred subset of classes
        2. Sample counts vary per client
        3. Adding client-specific "style" variations
        """
        np.random.seed(seed)
        logger.info(f"Generating synthetic FEMNIST-like data for {num_clients} clients")
        
        clients = []
        
        for i in range(num_clients):
            # Non-IID: each client has 2-5 preferred classes
            num_preferred = np.random.randint(2, 6)
            preferred_classes = np.random.choice(self.NUM_CLASSES, num_preferred, replace=False)
            
            # Variable samples per client
            num_samples = np.random.randint(samples_per_client[0], samples_per_client[1])
            
            # 80% samples from preferred classes, 20% random
            num_preferred_samples = int(0.8 * num_samples)
            num_random_samples = num_samples - num_preferred_samples
            
            y_preferred = np.random.choice(preferred_classes, num_preferred_samples)
            y_random = np.random.choice(self.NUM_CLASSES, num_random_samples)
            y = np.concatenate([y_preferred, y_random])
            np.random.shuffle(y)
            
            # Generate synthetic images (simple patterns based on class)
            x = []
            client_style = np.random.randn(self.IMAGE_SIZE, self.IMAGE_SIZE) * 0.1
            
            for label in y:
                # Create base pattern from class
                img = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE))
                
                # Simple class-based pattern (diagonal stripes offset by class)
                offset = label % self.IMAGE_SIZE
                for j in range(self.IMAGE_SIZE):
                    row = (j + offset) % self.IMAGE_SIZE
                    img[row, j] = 0.8 + np.random.randn() * 0.1
                
                # Add client style (simulates handwriting variation)
                img = img + client_style
                img = np.clip(img, 0, 1)
                
                x.append(img.flatten())
            
            x = np.array(x)
            
            clients.append(ClientData(
                client_id=f"user_{i:04d}",
                x=x,
                y=y.astype(np.int64),
                num_samples=len(y)
            ))
        
        return clients
    
    def load(self, max_clients: Optional[int] = None, use_synthetic: bool = True) -> List[ClientData]:
        """
        Load FEMNIST dataset.
        
        Args:
            max_clients: Maximum number of clients to load (None = all)
            use_synthetic: If True and real data not available, generate synthetic
            
        Returns:
            List of ClientData objects
        """
        if self._download_if_needed():
            return self._load_real_data(max_clients)
        elif use_synthetic:
            logger.info("Using synthetic FEMNIST-like data for demonstration")
            return self._generate_synthetic_femnist(
                num_clients=max_clients or 100
            )
        else:
            raise FileNotFoundError(f"FEMNIST data not found at {self.data_dir}")
    
    def _load_real_data(self, max_clients: Optional[int] = None) -> List[ClientData]:
        """Load real FEMNIST data from JSON files."""
        clients = []
        
        json_files = sorted(self.train_dir.glob("*.json"))
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for user_id in data['users']:
                if max_clients and len(clients) >= max_clients:
                    break
                    
                user_idx = data['users'].index(user_id)
                x = np.array(data['user_data'][user_id]['x'])
                y = np.array(data['user_data'][user_id]['y'])
                
                clients.append(ClientData(
                    client_id=user_id,
                    x=x,
                    y=y.astype(np.int64),
                    num_samples=len(y)
                ))
            
            if max_clients and len(clients) >= max_clients:
                break
        
        logger.info(f"Loaded {len(clients)} clients from FEMNIST")
        return clients


class ShakespeareLoader(LEAFBaseLoader):
    """
    Shakespeare Dataset Loader for Next Character Prediction.
    
    This dataset is derived from The Complete Works of William Shakespeare,
    where each speaking role is treated as a different client. The task is
    next character prediction.
    
    - 422 users (speaking roles)
    - ~4,226,158 characters total
    - 80 character vocabulary
    - Sequence length: configurable (default 80)
    
    The natural heterogeneity comes from different characters having
    distinct speaking patterns and vocabulary.
    """
    
    DATASET_NAME = "shakespeare"
    VOCAB_SIZE = 80
    SEQ_LEN = 80
    
    # Character vocabulary (printable ASCII + special tokens)
    VOCAB = list(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~")
    
    def __init__(self, data_dir: str = "./data/shakespeare", seq_len: int = 80):
        super().__init__(data_dir)
        self.seq_len = seq_len
        self.char_to_idx = {c: i for i, c in enumerate(self.VOCAB)}
        self.idx_to_char = {i: c for i, c in enumerate(self.VOCAB)}
        
    def _download_if_needed(self) -> bool:
        """Check if Shakespeare data exists."""
        train_file = self.data_dir / "train" / "all_data_0_0_0.json"
        if not train_file.exists():
            logger.warning(f"Shakespeare data not found at {self.data_dir}")
            logger.info("=" * 60)
            logger.info("To download Shakespeare, run:")
            logger.info("  git clone https://github.com/TalwalkarLab/leaf.git")
            logger.info("  cd leaf/data/shakespeare")
            logger.info("  ./preprocess.sh -s niid --sf 0.1 -k 0 -t sample")
            logger.info("Then copy data/shakespeare/data to this directory")
            logger.info("=" * 60)
            return False
        return True
    
    def _generate_synthetic_shakespeare(self,
                                         num_clients: int = 100,
                                         samples_per_client: Tuple[int, int] = (100, 1000),
                                         seed: int = 42) -> List[ClientData]:
        """
        Generate synthetic Shakespeare-like data for testing.
        
        Each client has a distinct "style" based on:
        1. Preferred character distributions
        2. Common n-grams
        """
        np.random.seed(seed)
        logger.info(f"Generating synthetic Shakespeare-like data for {num_clients} clients")
        
        # Some common English patterns
        common_patterns = [
            "the ", "and ", "ing ", "tion", "er ", "ed ", "of ", "to ", "in ", "is ",
            "that", "it ", "was ", "for ", "on ", "are ", "as ", "with", "his ", "they"
        ]
        
        clients = []
        
        for i in range(num_clients):
            # Each client has preferred patterns
            num_patterns = np.random.randint(3, 8)
            client_patterns = list(np.random.choice(common_patterns, num_patterns, replace=False))
            
            # Generate text
            num_samples = np.random.randint(samples_per_client[0], samples_per_client[1])
            total_chars_needed = (num_samples + 1) * self.seq_len
            
            text = []
            while len(text) < total_chars_needed:
                if np.random.random() < 0.4 and client_patterns:
                    # Use preferred pattern
                    pattern = np.random.choice(client_patterns)
                    text.extend(list(pattern))
                else:
                    # Random character from vocabulary
                    char_idx = np.random.randint(len(self.VOCAB))
                    text.append(self.VOCAB[char_idx])
            
            text = text[:total_chars_needed]
            
            # Create sequences
            x = []
            y = []
            
            for j in range(num_samples):
                start = j * self.seq_len
                seq = text[start:start + self.seq_len]
                next_char = text[start + self.seq_len]
                
                # Convert to indices
                seq_idx = [self.char_to_idx.get(c, 0) for c in seq]
                next_idx = self.char_to_idx.get(next_char, 0)
                
                x.append(seq_idx)
                y.append(next_idx)
            
            clients.append(ClientData(
                client_id=f"role_{i:04d}",
                x=np.array(x),
                y=np.array(y, dtype=np.int64),
                num_samples=len(y)
            ))
        
        return clients
    
    def load(self, max_clients: Optional[int] = None, use_synthetic: bool = True) -> List[ClientData]:
        """
        Load Shakespeare dataset.
        
        Args:
            max_clients: Maximum number of clients to load
            use_synthetic: Use synthetic data if real data unavailable
            
        Returns:
            List of ClientData objects
        """
        if self._download_if_needed():
            return self._load_real_data(max_clients)
        elif use_synthetic:
            logger.info("Using synthetic Shakespeare-like data for demonstration")
            return self._generate_synthetic_shakespeare(
                num_clients=max_clients or 100
            )
        else:
            raise FileNotFoundError(f"Shakespeare data not found at {self.data_dir}")
    
    def _load_real_data(self, max_clients: Optional[int] = None) -> List[ClientData]:
        """Load real Shakespeare data from JSON files."""
        clients = []
        train_dir = self.data_dir / "train"
        
        for json_file in sorted(train_dir.glob("*.json")):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for user_id in data['users']:
                if max_clients and len(clients) >= max_clients:
                    break
                
                raw_x = data['user_data'][user_id]['x']
                raw_y = data['user_data'][user_id]['y']
                
                # Convert characters to indices
                x = []
                y = []
                
                for seq, label in zip(raw_x, raw_y):
                    seq_idx = [self.char_to_idx.get(c, 0) for c in seq]
                    label_idx = self.char_to_idx.get(label, 0)
                    x.append(seq_idx)
                    y.append(label_idx)
                
                clients.append(ClientData(
                    client_id=user_id,
                    x=np.array(x),
                    y=np.array(y, dtype=np.int64),
                    num_samples=len(y)
                ))
            
            if max_clients and len(clients) >= max_clients:
                break
        
        logger.info(f"Loaded {len(clients)} clients from Shakespeare")
        return clients


def compute_heterogeneity_metrics(clients: List[ClientData]) -> Dict:
    """
    Compute metrics quantifying the heterogeneity of the dataset.
    
    Returns:
        Dict with heterogeneity metrics including:
        - label_distribution_divergence: Average KL divergence from global distribution
        - sample_size_gini: Gini coefficient of sample sizes
        - earth_mover_distance: Average EMD between client distributions
    """
    # Get all unique labels
    all_labels = set()
    for client in clients:
        all_labels.update(client.y.tolist())
    num_classes = max(all_labels) + 1
    
    # Compute global label distribution
    global_counts = np.zeros(num_classes)
    for client in clients:
        for label in client.y:
            global_counts[label] += 1
    global_dist = global_counts / global_counts.sum()
    
    # Compute per-client distributions and KL divergence
    kl_divergences = []
    client_dists = []
    
    for client in clients:
        counts = np.zeros(num_classes)
        for label in client.y:
            counts[label] += 1
        dist = counts / counts.sum()
        client_dists.append(dist)
        
        # KL divergence (with smoothing to avoid inf)
        eps = 1e-10
        dist_smooth = dist + eps
        dist_smooth /= dist_smooth.sum()
        global_smooth = global_dist + eps
        global_smooth /= global_smooth.sum()
        
        kl = np.sum(dist_smooth * np.log(dist_smooth / global_smooth))
        kl_divergences.append(kl)
    
    # Gini coefficient of sample sizes
    sample_sizes = np.array([c.num_samples for c in clients])
    sorted_sizes = np.sort(sample_sizes)
    n = len(sorted_sizes)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_sizes) - (n + 1) * np.sum(sorted_sizes)) / (n * np.sum(sorted_sizes))
    
    return {
        'num_classes': num_classes,
        'mean_kl_divergence': float(np.mean(kl_divergences)),
        'max_kl_divergence': float(np.max(kl_divergences)),
        'sample_size_gini': float(gini),
        'global_label_entropy': float(-np.sum(global_dist * np.log(global_dist + 1e-10)))
    }


if __name__ == "__main__":
    """Demonstration of dataset loaders."""
    
    print("=" * 70)
    print("LEAF Dataset Loaders for DSAIN Framework")
    print("=" * 70)
    
    # Test FEMNIST loader
    print("\n1. FEMNIST Dataset")
    print("-" * 40)
    femnist = FEMNISTLoader()
    femnist_clients = femnist.load(max_clients=100, use_synthetic=True)
    
    stats = femnist.get_statistics(femnist_clients)
    print(f"   Clients: {stats['num_clients']}")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Mean samples/client: {stats['mean_samples_per_client']:.1f}")
    print(f"   Std samples/client: {stats['std_samples_per_client']:.1f}")
    
    hetero = compute_heterogeneity_metrics(femnist_clients)
    print(f"   Mean KL divergence: {hetero['mean_kl_divergence']:.3f}")
    print(f"   Sample size Gini: {hetero['sample_size_gini']:.3f}")
    
    # Test Shakespeare loader
    print("\n2. Shakespeare Dataset")
    print("-" * 40)
    shakespeare = ShakespeareLoader()
    shakespeare_clients = shakespeare.load(max_clients=50, use_synthetic=True)
    
    stats = shakespeare.get_statistics(shakespeare_clients)
    print(f"   Clients: {stats['num_clients']}")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Mean samples/client: {stats['mean_samples_per_client']:.1f}")
    print(f"   Std samples/client: {stats['std_samples_per_client']:.1f}")
    
    print("\n" + "=" * 70)
    print("Datasets ready for DSAIN experiments!")
    print("=" * 70)
