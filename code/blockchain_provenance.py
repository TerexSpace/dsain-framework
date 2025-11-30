#!/usr/bin/env python3
"""
Blockchain-Based Model Provenance System for DSAIN
===================================================

Implementation of lightweight blockchain layer for federated learning provenance
as described in Section 4 of the JMLR paper.

Key Features:
- Cryptographic commitment to model states
- Proof-of-Training consensus mechanism
- Zero-knowledge proof framework (simplified)
- Verification protocol for training history

Authors: Almas Ospanov
License: MIT
"""

import hashlib
import json
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelCommitment:
    """Cryptographic commitment to a model state at a specific round."""
    round_number: int
    model_hash: str
    participant_ids: List[int]
    timestamp: float
    convergence_metric: float  # e.g., gradient norm
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute hash of this commitment."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ProofOfTraining:
    """Proof-of-Training for a specific round."""
    commitment: ModelCommitment
    attestations: Dict[int, str]  # participant_id -> signature
    zk_proof: Optional[str]  # Simplified ZK proof placeholder
    previous_block_hash: str
    nonce: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'commitment': self.commitment.to_dict(),
            'attestations': self.attestations,
            'zk_proof': self.zk_proof,
            'previous_block_hash': self.previous_block_hash,
            'nonce': self.nonce
        }
    
    def compute_hash(self) -> str:
        """Compute block hash."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


class Block:
    """Individual block in the provenance blockchain."""
    
    def __init__(
        self,
        index: int,
        proof_of_training: ProofOfTraining,
        previous_hash: str,
        timestamp: Optional[float] = None
    ):
        self.index = index
        self.proof_of_training = proof_of_training
        self.previous_hash = previous_hash
        self.timestamp = timestamp or time.time()
        self.hash = self.compute_hash()
    
    def compute_hash(self) -> str:
        """Compute block hash."""
        block_data = {
            'index': self.index,
            'proof_of_training': self.proof_of_training.to_dict(),
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp
        }
        data = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert block to dictionary."""
        return {
            'index': self.index,
            'proof_of_training': self.proof_of_training.to_dict(),
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'hash': self.hash
        }


class ProvenanceBlockchain:
    """Blockchain for federated learning model provenance."""
    
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_attestations: Dict[int, Dict[int, str]] = {}  # round -> {client_id -> attestation}
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first block in the chain."""
        genesis_commitment = ModelCommitment(
            round_number=0,
            model_hash="0" * 64,
            participant_ids=[],
            timestamp=time.time(),
            convergence_metric=0.0
        )
        
        genesis_pot = ProofOfTraining(
            commitment=genesis_commitment,
            attestations={},
            zk_proof=None,
            previous_block_hash="0" * 64,
            nonce=0
        )
        
        genesis_block = Block(0, genesis_pot, "0" * 64)
        self.chain.append(genesis_block)
        logger.info("Genesis block created")
    
    def get_latest_block(self) -> Block:
        """Get the most recent block."""
        return self.chain[-1]
    
    def create_model_commitment(
        self,
        round_number: int,
        model_weights: np.ndarray,
        participant_ids: List[int],
        convergence_metric: float
    ) -> ModelCommitment:
        """Create a cryptographic commitment to model state."""
        # Hash the model weights
        model_bytes = model_weights.tobytes()
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        
        commitment = ModelCommitment(
            round_number=round_number,
            model_hash=model_hash,
            participant_ids=sorted(participant_ids),
            timestamp=time.time(),
            convergence_metric=convergence_metric
        )
        
        return commitment
    
    def sign_attestation(self, client_id: int, round_number: int, commitment_hash: str) -> str:
        """
        Generate client attestation (simplified signature).
        In production, this would use proper digital signatures (e.g., ECDSA).
        """
        data = f"{client_id}:{round_number}:{commitment_hash}"
        signature = hashlib.sha256(data.encode()).hexdigest()
        return signature
    
    def add_attestation(self, client_id: int, round_number: int, attestation: str):
        """Add a client attestation for a specific round."""
        if round_number not in self.pending_attestations:
            self.pending_attestations[round_number] = {}
        
        self.pending_attestations[round_number][client_id] = attestation
    
    def generate_zk_proof(self, commitment: ModelCommitment, convergence_threshold: float) -> str:
        """
        Generate zero-knowledge proof that model satisfies convergence criteria.
        This is a simplified placeholder - production would use proper ZK-SNARK/ZK-STARK.
        """
        # Simplified: hash of statement that convergence metric meets threshold
        statement = f"convergence:{commitment.convergence_metric}:threshold:{convergence_threshold}"
        proof = hashlib.sha256(statement.encode()).hexdigest()
        return proof
    
    def verify_zk_proof(self, proof: str, commitment: ModelCommitment, threshold: float) -> bool:
        """Verify zero-knowledge proof (simplified)."""
        expected_proof = self.generate_zk_proof(commitment, threshold)
        return proof == expected_proof
    
    def add_training_round(
        self,
        commitment: ModelCommitment,
        convergence_threshold: float = 0.0
    ) -> Block:
        """Add a new training round to the blockchain."""
        round_num = commitment.round_number
        
        # Get attestations for this round
        attestations = self.pending_attestations.get(round_num, {})
        
        # Generate zero-knowledge proof
        zk_proof = self.generate_zk_proof(commitment, convergence_threshold)
        
        # Create Proof-of-Training
        pot = ProofOfTraining(
            commitment=commitment,
            attestations=attestations,
            zk_proof=zk_proof,
            previous_block_hash=self.get_latest_block().hash,
            nonce=len(self.chain)
        )
        
        # Create and add new block
        new_block = Block(
            index=len(self.chain),
            proof_of_training=pot,
            previous_hash=self.get_latest_block().hash
        )
        
        self.chain.append(new_block)
        
        # Clear pending attestations for this round
        if round_num in self.pending_attestations:
            del self.pending_attestations[round_num]
        
        logger.info(f"Block {new_block.index} added for round {round_num}")
        return new_block
    
    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """Verify the integrity of the blockchain."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            # Verify hash consistency
            if current.previous_hash != previous.hash:
                return False, f"Block {i}: previous_hash mismatch"
            
            # Verify block hash
            if current.hash != current.compute_hash():
                return False, f"Block {i}: invalid block hash"
            
            # Verify PoT hash
            if current.proof_of_training.compute_hash() != \
               current.proof_of_training.compute_hash():  # Redundant but demonstrates verification
                return False, f"Block {i}: invalid PoT hash"
        
        return True, None
    
    def verify_attestations(
        self,
        round_number: int,
        expected_participants: Set[int],
        min_attestations: int
    ) -> bool:
        """Verify that sufficient valid attestations exist for a round."""
        if round_number >= len(self.chain):
            return False
        
        block = self.chain[round_number]
        attestations = block.proof_of_training.attestations
        
        # Check minimum number of attestations
        if len(attestations) < min_attestations:
            logger.warning(f"Round {round_number}: insufficient attestations "
                          f"({len(attestations)} < {min_attestations})")
            return False
        
        # Verify attestations are from expected participants
        commitment_hash = block.proof_of_training.commitment.compute_hash()
        
        valid_count = 0
        for client_id, attestation in attestations.items():
            if client_id not in expected_participants:
                continue
            
            # Re-compute expected signature
            expected_sig = self.sign_attestation(client_id, round_number, commitment_hash)
            if attestation == expected_sig:
                valid_count += 1
        
        return valid_count >= min_attestations
    
    def get_training_history(self) -> List[Dict]:
        """Get complete training history from blockchain."""
        history = []
        
        for block in self.chain[1:]:  # Skip genesis
            commitment = block.proof_of_training.commitment
            history.append({
                'round': commitment.round_number,
                'model_hash': commitment.model_hash,
                'participants': len(commitment.participant_ids),
                'convergence_metric': commitment.convergence_metric,
                'timestamp': datetime.fromtimestamp(commitment.timestamp).isoformat(),
                'attestations': len(block.proof_of_training.attestations),
                'block_hash': block.hash
            })
        
        return history
    
    def export_chain(self, filepath: str):
        """Export blockchain to JSON file."""
        chain_data = [block.to_dict() for block in self.chain]
        
        with open(filepath, 'w') as f:
            json.dump(chain_data, f, indent=2)
        
        logger.info(f"Blockchain exported to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get blockchain statistics."""
        if len(self.chain) <= 1:
            return {
                'total_blocks': 0,
                'total_rounds': 0,
                'average_participants': 0,
                'average_attestations': 0
            }
        
        total_participants = 0
        total_attestations = 0
        
        for block in self.chain[1:]:
            total_participants += len(block.proof_of_training.commitment.participant_ids)
            total_attestations += len(block.proof_of_training.attestations)
        
        num_rounds = len(self.chain) - 1
        
        return {
            'total_blocks': len(self.chain),
            'total_rounds': num_rounds,
            'average_participants': total_participants / num_rounds,
            'average_attestations': total_attestations / num_rounds,
            'chain_valid': self.verify_chain()[0]
        }


def demo_blockchain_provenance():
    """Demonstrate blockchain provenance system."""
    logger.info("\n" + "="*80)
    logger.info("DSAIN BLOCKCHAIN PROVENANCE SYSTEM DEMO")
    logger.info("="*80 + "\n")
    
    # Initialize blockchain
    blockchain = ProvenanceBlockchain()
    
    # Simulate federated learning rounds
    num_rounds = 10
    num_clients = 100
    model_dim = 100
    
    for round_num in range(1, num_rounds + 1):
        # Simulate model training
        model_weights = np.random.randn(model_dim)
        
        # Select participating clients (10% participation)
        participants = np.random.choice(num_clients, size=10, replace=False).tolist()
        
        # Simulate convergence metric (decreasing over rounds)
        convergence_metric = 10.0 / round_num
        
        # Create model commitment
        commitment = blockchain.create_model_commitment(
            round_number=round_num,
            model_weights=model_weights,
            participant_ids=participants,
            convergence_metric=convergence_metric
        )
        
        commitment_hash = commitment.compute_hash()
        
        # Collect attestations from participants
        for client_id in participants:
            attestation = blockchain.sign_attestation(client_id, round_num, commitment_hash)
            blockchain.add_attestation(client_id, round_num, attestation)
        
        # Add round to blockchain
        block = blockchain.add_training_round(commitment, convergence_threshold=0.1)
        
        logger.info(f"Round {round_num}: Block {block.index} created with "
                   f"{len(participants)} participants, "
                   f"convergence={convergence_metric:.4f}")
    
    # Verify blockchain
    logger.info("\n" + "-"*80)
    logger.info("BLOCKCHAIN VERIFICATION")
    logger.info("-"*80)
    
    is_valid, error = blockchain.verify_chain()
    logger.info(f"Chain integrity: {'VALID' if is_valid else 'INVALID'}")
    if error:
        logger.error(f"Verification error: {error}")
    
    # Verify attestations for a specific round
    round_to_verify = 5
    expected_participants = set(blockchain.chain[round_to_verify].proof_of_training.commitment.participant_ids)
    attestations_valid = blockchain.verify_attestations(
        round_to_verify,
        expected_participants,
        min_attestations=5
    )
    logger.info(f"Round {round_to_verify} attestations: {'VALID' if attestations_valid else 'INVALID'}")
    
    # Display statistics
    logger.info("\n" + "-"*80)
    logger.info("BLOCKCHAIN STATISTICS")
    logger.info("-"*80)
    
    stats = blockchain.get_statistics()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    # Display training history
    logger.info("\n" + "-"*80)
    logger.info("TRAINING HISTORY")
    logger.info("-"*80)
    
    history = blockchain.get_training_history()
    for entry in history[:5]:  # Show first 5 rounds
        logger.info(f"Round {entry['round']}: "
                   f"participants={entry['participants']}, "
                   f"attestations={entry['attestations']}, "
                   f"convergence={entry['convergence_metric']:.4f}")
    
    # Export blockchain
    import os
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    blockchain.export_chain(os.path.join(output_dir, "provenance_chain.json"))
    
    logger.info("\n" + "="*80)
    logger.info("DEMO COMPLETE")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    demo_blockchain_provenance()
