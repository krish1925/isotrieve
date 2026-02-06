"""
Agent Embedding Communication Protocol (AECP) Implementation

Implements the full protocol with:
- Handshake and capability negotiation
- Calibration with train/val split
- Transfer matrix computation and validation
- Semantic transfer with quality monitoring
- Error handling and fallbacks
"""

import numpy as np
import json
import hashlib
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from matrix_transfer import compute_transfer_matrices, cosine_similarity


@dataclass
class AgentCapabilities:
    """Agent capabilities and configuration."""
    agent_id: str
    embedding_model: str
    dimensions: int
    max_batch_size: int = 1000
    min_quality_threshold: float = 0.75
    protocol_version: str = "1.0"


@dataclass
class CalibrationRequest:
    """Request for calibration between agents."""
    vocabulary_size: int
    validation_size: int
    quality_threshold: float
    timestamp: str


@dataclass
class TransferMatrix:
    """Transfer matrix with quality metrics."""
    matrix_AB: np.ndarray
    matrix_BA: np.ndarray
    training_similarity: float
    validation_similarity: float
    worst_case_similarity: float
    valid_until: str
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict (excluding large matrices)."""
        return {
            "matrix_AB_shape": self.matrix_AB.shape,
            "matrix_BA_shape": self.matrix_BA.shape,
            "training_similarity": float(self.training_similarity),
            "validation_similarity": float(self.validation_similarity),
            "worst_case_similarity": float(self.worst_case_similarity),
            "valid_until": self.valid_until
        }


@dataclass
class SemanticTransfer:
    """Semantic content transfer."""
    transfer_id: str
    embedding: np.ndarray
    source_agent: str
    target_agent: str
    original_norm: float
    expected_similarity: float
    timestamp: str


class ProtocolHandler:
    """
    Handles AECP protocol operations.
    """
    
    def __init__(self, agent_id: str, embedder, model_name: str, dimensions: int):
        """Initialize protocol handler."""
        self.capabilities = AgentCapabilities(
            agent_id=agent_id,
            embedding_model=model_name,
            dimensions=dimensions
        )
        self.embedder = embedder
        self.transfer_matrices: Dict[str, TransferMatrix] = {}
        self.transfer_log: List[Dict] = []
        
    def send_handshake(self) -> Dict:
        """Send handshake message."""
        return {
            "message_type": "handshake",
            "protocol_version": self.capabilities.protocol_version,
            "agent_id": self.capabilities.agent_id,
            "embedding_model": {
                "name": self.capabilities.embedding_model,
                "dimensions": self.capabilities.dimensions
            },
            "capabilities": {
                "max_batch_size": self.capabilities.max_batch_size,
                "min_quality_threshold": self.capabilities.min_quality_threshold
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def receive_handshake(self, handshake: Dict) -> bool:
        """Receive and validate handshake."""
        if handshake.get("protocol_version") != self.capabilities.protocol_version:
            print(f"⚠️  Protocol version mismatch: {handshake.get('protocol_version')} vs {self.capabilities.protocol_version}")
            return False
        
        print(f"✓ Handshake received from {handshake.get('agent_id')}")
        print(f"  Model: {handshake['embedding_model']['name']}")
        print(f"  Dimensions: {handshake['embedding_model']['dimensions']}")
        return True
    
    def create_calibration_request(
        self,
        vocabulary_size: int,
        validation_size: int,
        quality_threshold: float = 0.80
    ) -> CalibrationRequest:
        """Create calibration request."""
        return CalibrationRequest(
            vocabulary_size=vocabulary_size,
            validation_size=validation_size,
            quality_threshold=quality_threshold,
            timestamp=datetime.now().isoformat()
        )
    
    def calibrate(
        self,
        partner_handler: 'ProtocolHandler',
        train_vocabulary: List[str],
        val_vocabulary: List[str],
        quality_threshold: float = 0.80
    ) -> TransferMatrix:
        """
        Perform calibration with partner agent.
        
        Args:
            partner_handler: Partner's protocol handler
            train_vocabulary: Training vocabulary (for matrix computation)
            val_vocabulary: Validation vocabulary (held-out, for quality check)
            quality_threshold: Minimum acceptable quality
            
        Returns:
            TransferMatrix with computed matrices and quality metrics
        """
        print(f"\n{'='*70}")
        print(f"CALIBRATION: {self.capabilities.agent_id} <-> {partner_handler.capabilities.agent_id}")
        print(f"{'='*70}")
        
        # Encode training vocabulary with both agents
        print(f"\nEncoding training vocabulary ({len(train_vocabulary):,} items)...")
        print(f"  Agent A ({self.capabilities.agent_id})...")
        emb_A_train = self.embedder.encode(train_vocabulary, show_progress_bar=True, batch_size=128)
        
        print(f"  Agent B ({partner_handler.capabilities.agent_id})...")
        emb_B_train = partner_handler.embedder.encode(train_vocabulary, show_progress_bar=True, batch_size=128)
        
        # Compute transfer matrices
        print(f"\nComputing transfer matrices...")
        W_AB, W_BA = compute_transfer_matrices(emb_A_train, emb_B_train)
        
        # Training quality - use round-trip for consistency with validation
        # NOTE: We use round-trip (A→B→A) for training similarity to match validation,
        # which ensures fair comparison. Forward transfer (A→B) is typically slightly lower
        # because it doesn't benefit from returning to the original space.
        print(f"\nEvaluating training quality (round-trip on training vocabulary)...")
        train_transferred = emb_A_train @ W_AB
        train_roundtrip = train_transferred @ W_BA
        
        # Sample for speed (use larger sample for better accuracy)
        # Use at least 10k samples for reliable statistics
        sample_size = min(10000, len(emb_A_train))
        train_rt_sims = [cosine_similarity(emb_A_train[i], train_roundtrip[i])
                        for i in range(sample_size)]
        training_similarity = float(np.mean(train_rt_sims))
        
        # Also compute forward similarity for reference (not used in comparison)
        train_forward_sims = [cosine_similarity(train_transferred[i], emb_B_train[i]) 
                             for i in range(min(1000, len(train_transferred)))]
        train_forward_similarity = float(np.mean(train_forward_sims))
        
        print(f"  Training round-trip similarity: {training_similarity:.4f} (on {sample_size:,} samples)")
        print(f"  Training forward similarity: {train_forward_similarity:.4f} (reference only)")
        
        # Validation on held-out data
        print(f"\nValidating on held-out vocabulary ({len(val_vocabulary):,} items)...")
        print(f"  Agent A encoding...")
        emb_A_val = self.embedder.encode(val_vocabulary, show_progress_bar=True, batch_size=128)
        
        print(f"  Agent B encoding...")
        emb_B_val = partner_handler.embedder.encode(val_vocabulary, show_progress_bar=True, batch_size=128)
        
        # Validation round-trip
        val_transferred = emb_A_val @ W_AB
        val_roundtrip = val_transferred @ W_BA
        val_sims = [cosine_similarity(emb_A_val[i], val_roundtrip[i]) 
                   for i in range(len(emb_A_val))]
        
        validation_similarity = float(np.mean(val_sims))
        worst_case_similarity = float(np.min(val_sims))
        
        print(f"  Validation round-trip similarity: {validation_similarity:.4f}")
        print(f"  Worst-case similarity: {worst_case_similarity:.4f}")
        
        # Check quality threshold
        if validation_similarity < quality_threshold:
            print(f"\n⚠️  WARNING: Validation quality ({validation_similarity:.4f}) below threshold ({quality_threshold:.4f})")
            print(f"     Consider: 1) Larger vocabulary, 2) Different models, 3) Non-linear transfer")
        else:
            print(f"\n✓ Quality threshold met ({validation_similarity:.4f} >= {quality_threshold:.4f})")
        
        # Create transfer matrix object
        valid_until = (datetime.now() + timedelta(days=7)).isoformat()
        transfer_matrix = TransferMatrix(
            matrix_AB=W_AB,
            matrix_BA=W_BA,
            training_similarity=training_similarity,
            validation_similarity=validation_similarity,
            worst_case_similarity=worst_case_similarity,
            valid_until=valid_until
        )
        
        # Store in both handlers
        key = f"{self.capabilities.agent_id}_{partner_handler.capabilities.agent_id}"
        self.transfer_matrices[key] = transfer_matrix
        
        partner_key = f"{partner_handler.capabilities.agent_id}_{self.capabilities.agent_id}"
        partner_handler.transfer_matrices[partner_key] = TransferMatrix(
            matrix_AB=W_BA,  # Reversed for partner
            matrix_BA=W_AB,
            training_similarity=training_similarity,
            validation_similarity=validation_similarity,
            worst_case_similarity=worst_case_similarity,
            valid_until=valid_until
        )
        
        return transfer_matrix
    
    def transfer_to(
        self,
        partner_agent_id: str,
        text: str
    ) -> SemanticTransfer:
        """
        Transfer semantic content to partner agent.
        
        Args:
            partner_agent_id: Target agent ID
            text: Text to transfer
            
        Returns:
            SemanticTransfer object
        """
        # Get transfer matrix
        key = f"{self.capabilities.agent_id}_{partner_agent_id}"
        if key not in self.transfer_matrices:
            raise ValueError(f"No calibration found for {key}. Run calibrate() first.")
        
        transfer_matrix = self.transfer_matrices[key]
        
        # Encode text
        embedding = self.embedder.encode(text)
        
        # Transform to partner's space
        transferred = embedding @ transfer_matrix.matrix_AB
        
        # Create transfer object
        transfer_id = hashlib.md5(f"{text}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        semantic_transfer = SemanticTransfer(
            transfer_id=transfer_id,
            embedding=transferred,
            source_agent=self.capabilities.agent_id,
            target_agent=partner_agent_id,
            original_norm=float(np.linalg.norm(embedding)),
            expected_similarity=transfer_matrix.validation_similarity,
            timestamp=datetime.now().isoformat()
        )
        
        # Log transfer
        self.transfer_log.append({
            "transfer_id": transfer_id,
            "source": self.capabilities.agent_id,
            "target": partner_agent_id,
            "timestamp": semantic_transfer.timestamp,
            "expected_quality": transfer_matrix.validation_similarity
        })
        
        return semantic_transfer
    
    def receive_transfer(
        self,
        transfer: SemanticTransfer,
        original_text: str = None
    ) -> Dict:
        """
        Receive and validate transferred semantic content.
        
        Args:
            transfer: Received semantic transfer
            original_text: Optional original text for validation
            
        Returns:
            Acknowledgment with quality metrics
        """
        # Get reverse transfer matrix
        key = f"{self.capabilities.agent_id}_{transfer.source_agent}"
        if key not in self.transfer_matrices:
            return {
                "status": "error",
                "message": "No calibration found"
            }
        
        transfer_matrix = self.transfer_matrices[key]
        
        # Compute received norm
        received_norm = float(np.linalg.norm(transfer.embedding))
        
        # If original text provided, check quality
        quality_metric = None
        if original_text:
            # Encode with our embedder
            our_embedding = self.embedder.encode(original_text)
            
            # Transform received embedding back to source space then to our space
            # This tests the full round-trip
            back_to_source = transfer.embedding @ transfer_matrix.matrix_AB
            
            # Or directly compare with our encoding
            quality_metric = cosine_similarity(transfer.embedding @ transfer_matrix.matrix_AB, our_embedding)
        
        acknowledgment = {
            "message_type": "acknowledgment",
            "transfer_id": transfer.transfer_id,
            "status": "success",
            "received_norm": received_norm,
            "original_norm": transfer.original_norm,
            "norm_ratio": received_norm / transfer.original_norm if transfer.original_norm > 0 else 0,
            "quality_metric": float(quality_metric) if quality_metric else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return acknowledgment
    
    def get_calibration_stats(self, partner_agent_id: str) -> Optional[Dict]:
        """Get calibration statistics for partner."""
        key = f"{self.capabilities.agent_id}_{partner_agent_id}"
        if key not in self.transfer_matrices:
            return None
        
        return self.transfer_matrices[key].to_dict()
    
    def requires_recalibration(self, partner_agent_id: str) -> bool:
        """Check if recalibration is needed."""
        key = f"{self.capabilities.agent_id}_{partner_agent_id}"
        if key not in self.transfer_matrices:
            return True
        
        transfer_matrix = self.transfer_matrices[key]
        
        # Check expiration
        valid_until = datetime.fromisoformat(transfer_matrix.valid_until)
        if datetime.now() > valid_until:
            return True
        
        # Check quality
        if transfer_matrix.validation_similarity < self.capabilities.min_quality_threshold:
            return True
        
        return False


def demonstrate_protocol():
    """Demonstrate the protocol with simple example."""
    from sentence_transformers import SentenceTransformer
    
    print("="*70)
    print("PROTOCOL DEMONSTRATION")
    print("="*70)
    
    # Create two agents
    print("\nInitializing agents...")
    embedder_a = SentenceTransformer('all-MiniLM-L6-v2')
    embedder_b = SentenceTransformer('all-mpnet-base-v2')
    
    agent_a = ProtocolHandler("agent_a", embedder_a, "all-MiniLM-L6-v2", 384)
    agent_b = ProtocolHandler("agent_b", embedder_b, "all-mpnet-base-v2", 768)
    
    # Handshake
    print("\n--- HANDSHAKE PHASE ---")
    handshake_a = agent_a.send_handshake()
    print(f"Agent A sent handshake")
    
    success = agent_b.receive_handshake(handshake_a)
    print(f"Agent B received handshake: {'✓' if success else '✗'}")
    
    # Calibration with small vocabulary for demo
    print("\n--- CALIBRATION PHASE ---")
    demo_vocab = ["hello", "world", "test", "example", "data", "model"]
    demo_val = ["validation", "check", "quality"]
    
    transfer_matrix = agent_a.calibrate(agent_b, demo_vocab, demo_val)
    print(f"\n✓ Calibration complete")
    
    # Transfer
    print("\n--- TRANSFER PHASE ---")
    text = "This is a test message"
    transfer = agent_a.transfer_to("agent_b", text)
    print(f"Agent A transferred: '{text}'")
    print(f"  Transfer ID: {transfer.transfer_id}")
    print(f"  Expected quality: {transfer.expected_similarity:.4f}")
    
    # Receive
    ack = agent_b.receive_transfer(transfer, text)
    print(f"\nAgent B acknowledgment:")
    print(f"  Status: {ack['status']}")
    print(f"  Norm ratio: {ack['norm_ratio']:.4f}")
    if ack['quality_metric']:
        print(f"  Quality: {ack['quality_metric']:.4f}")
    
    print("\n" + "="*70)
    print("PROTOCOL DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    demonstrate_protocol()
