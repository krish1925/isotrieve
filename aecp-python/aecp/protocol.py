"""
AECP Protocol Implementation

Agent Embedding Communication Protocol (AECP) v1.0 implementation.
Enables AI agents with different embedding models to communicate
with high semantic fidelity.

Production-ready with:
- Circuit breaker for failing connections
- Graceful fallback to English text
- Real-time debug monitoring
- Comprehensive error handling
- Thread-safe operations
"""

import hashlib
import time
import threading
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .types import (
    EmbeddingProvider,
    AgentCapabilities,
    TransferMatrix,
    SemanticTransfer,
    CalibrationRequest,
)
from .matrix import (
    compute_transfer_matrices,
    transfer_embedding,
    cosine_similarity,
    evaluate_transfer_quality,
)
from .vocabulary import get_default_vocabulary
from .errors import (
    AECPError,
    CalibrationError,
    TransferError,
    AgentNotCalibratedError,
    MatrixExpiredError,
    QualityBelowThresholdError,
    CircuitBreaker,
    CircuitOpenError,
    RetryPolicy,
    GracefulDegradation,
)

logger = logging.getLogger("aecp.protocol")


@dataclass
class CalibrationResult:
    """
    Result of a calibration between two agents.

    Attributes:
        success: Whether calibration succeeded
        transfer_matrix: The computed transfer matrix (if successful)
        training_similarity: Quality on training data
        validation_similarity: Quality on held-out data
        worst_case_similarity: Minimum observed similarity
        calibration_time_ms: Time taken for calibration in milliseconds
        vocabulary_size: Number of vocabulary items used
        error_message: Error message (if failed)
    """
    success: bool
    transfer_matrix: Optional[TransferMatrix] = None
    training_similarity: float = 0.0
    validation_similarity: float = 0.0
    worst_case_similarity: float = 0.0
    calibration_time_ms: float = 0.0
    vocabulary_size: int = 0
    error_message: Optional[str] = None

    def meets_threshold(self, threshold: float = 0.75) -> bool:
        """Check if calibration meets quality threshold."""
        return self.success and self.validation_similarity >= threshold

    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            "success": self.success,
            "training_similarity": self.training_similarity,
            "validation_similarity": self.validation_similarity,
            "worst_case_similarity": self.worst_case_similarity,
            "calibration_time_ms": self.calibration_time_ms,
            "vocabulary_size": self.vocabulary_size,
            "error_message": self.error_message,
        }


class ProtocolHandler:
    """
    Handles AECP protocol operations for a single agent.

    This class manages the complete lifecycle of agent communication:
    - Handshake and capability exchange
    - Calibration with other agents
    - Semantic transfer of embeddings
    - Quality monitoring and recalibration
    - Circuit breaking for failing connections
    - Graceful fallback to English text

    Example:
        >>> from aecp import ProtocolHandler
        >>> from aecp.adapters import OpenAIAdapter
        >>>
        >>> adapter = OpenAIAdapter(api_key="sk-...")
        >>> agent = ProtocolHandler("agent_1", adapter)
        >>>
        >>> # Calibrate with another agent
        >>> result = agent.calibrate(other_agent, vocabulary)
        >>>
        >>> # Transfer embeddings
        >>> transfer = agent.transfer_to("other_agent", "Hello world")
    """

    def __init__(
        self,
        agent_id: str,
        embedder: EmbeddingProvider,
        max_batch_size: int = 1000,
        min_quality_threshold: float = 0.75,
        matrix_validity_days: int = 7,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        retry_max: int = 3,
        retry_base_delay: float = 1.0,
    ):
        """
        Initialize protocol handler.

        Args:
            agent_id: Unique identifier for this agent
            embedder: Embedding provider (adapter) to use
            max_batch_size: Maximum batch size for embedding operations
            min_quality_threshold: Minimum acceptable transfer quality
            matrix_validity_days: How long transfer matrices remain valid
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_timeout: Seconds before circuit half-opens
            retry_max: Maximum retry attempts
            retry_base_delay: Base delay between retries

        Raises:
            ValueError: If parameters are invalid
            TypeError: If embedder doesn't implement EmbeddingProvider
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(embedder, EmbeddingProvider):
            raise TypeError("embedder must implement EmbeddingProvider")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not 0 < min_quality_threshold <= 1:
            raise ValueError("min_quality_threshold must be between 0 and 1")

        self.capabilities = AgentCapabilities(
            agent_id=agent_id,
            embedding_model=embedder.get_model_id(),
            dimensions=embedder.get_dimensions(),
            max_batch_size=max_batch_size,
            min_quality_threshold=min_quality_threshold,
        )
        self.embedder = embedder
        self.matrix_validity_days = matrix_validity_days
        self.transfer_matrices: Dict[str, TransferMatrix] = {}
        self.transfer_log: List[Dict] = []

        # Thread safety
        self._lock = threading.RLock()

        # Circuit breakers (per partner)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._cb_threshold = circuit_breaker_threshold
        self._cb_timeout = circuit_breaker_timeout

        # Retry policy
        self._retry_policy = RetryPolicy(
            max_retries=retry_max,
            base_delay=retry_base_delay,
        )

        # Graceful degradation
        self._degradation = GracefulDegradation(default_language="en")

        # Debug monitor integration
        self._monitor = None

    @property
    def agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return self.capabilities.agent_id

    @property
    def monitor(self):
        """Get the debug monitor (lazily from global)."""
        if self._monitor is None:
            from .debug import DebugMonitor
            self._monitor = DebugMonitor.get_global()
        return self._monitor

    @monitor.setter
    def monitor(self, value):
        """Set the debug monitor."""
        self._monitor = value

    def _get_circuit_breaker(self, partner_id: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a partner."""
        if partner_id not in self._circuit_breakers:
            cb = CircuitBreaker(
                failure_threshold=self._cb_threshold,
                reset_timeout=self._cb_timeout,
            )
            cb._agent_id = f"{self.agent_id}->{partner_id}"
            self._circuit_breakers[partner_id] = cb
        return self._circuit_breakers[partner_id]

    def _emit_event(self, **kwargs):
        """Emit a debug event if monitor is available."""
        if self.monitor:
            from .debug import DebugEvent, EventType
            event_type_str = kwargs.pop("event_type", "embed")
            event_type = getattr(EventType, event_type_str.upper(), EventType.EMBED)
            event = DebugEvent(
                event_type=event_type,
                timestamp=time.time(),
                agent_id=self.agent_id,
                **kwargs,
            )
            self.monitor.log_event(event)

    def send_handshake(self) -> Dict:
        """
        Create a handshake message to send to another agent.

        Returns:
            Dictionary containing agent capabilities and metadata
        """
        handshake = {
            "message_type": "handshake",
            "protocol_version": self.capabilities.protocol_version,
            "agent_id": self.capabilities.agent_id,
            "embedding_model": {
                "name": self.capabilities.embedding_model,
                "dimensions": self.capabilities.dimensions,
            },
            "capabilities": {
                "max_batch_size": self.capabilities.max_batch_size,
                "min_quality_threshold": self.capabilities.min_quality_threshold,
            },
            "supports_aecp": True,
            "fallback_language": "en",
            "timestamp": datetime.now().isoformat(),
        }

        self._emit_event(
            event_type="handshake",
            metadata={"action": "sent"},
        )

        return handshake

    def receive_handshake(self, handshake: Dict) -> Tuple[bool, Optional[str]]:
        """
        Receive and validate a handshake from another agent.

        If the other agent doesn't support AECP, returns gracefully
        indicating text-based communication should be used.

        Args:
            handshake: Handshake message from another agent

        Returns:
            Tuple of (success, error_message)
        """
        # Check if this is even an AECP handshake
        if not isinstance(handshake, dict):
            return False, "Invalid handshake format: expected dictionary"

        # If agent doesn't support AECP, that's okay - we fall back to text
        if not handshake.get("supports_aecp", True):
            logger.info(
                f"Agent {handshake.get('agent_id', 'unknown')} does not support AECP. "
                "Communication will use plain English text."
            )
            return False, "Agent does not support AECP. Using English text fallback."

        # Validate required fields
        required_fields = ["protocol_version", "agent_id", "embedding_model"]
        for field in required_fields:
            if field not in handshake:
                return False, f"Missing required field: {field}"

        # Check protocol version compatibility
        if handshake.get("protocol_version") != self.capabilities.protocol_version:
            return False, (
                f"Protocol version mismatch: "
                f"{handshake.get('protocol_version')} vs "
                f"{self.capabilities.protocol_version}. "
                "Falling back to English text communication."
            )

        # Validate embedding model info
        embedding_model = handshake.get("embedding_model", {})
        if not isinstance(embedding_model, dict):
            return False, "Invalid embedding_model format"
        if "dimensions" not in embedding_model:
            return False, "Missing embedding dimensions"
        if not isinstance(embedding_model["dimensions"], (int, float)) or embedding_model["dimensions"] <= 0:
            return False, "Invalid embedding dimensions"

        self._emit_event(
            event_type="handshake",
            partner_id=handshake.get("agent_id"),
            metadata={"action": "received", "valid": True},
        )

        return True, None

    def create_calibration_request(
        self,
        vocabulary_size: int = 10000,
        validation_ratio: float = 0.1,
        quality_threshold: float = 0.80,
    ) -> CalibrationRequest:
        """
        Create a calibration request.

        Args:
            vocabulary_size: Total vocabulary size to use
            validation_ratio: Fraction to hold out for validation
            quality_threshold: Minimum acceptable quality

        Returns:
            CalibrationRequest object
        """
        validation_size = int(vocabulary_size * validation_ratio)
        return CalibrationRequest(
            vocabulary_size=vocabulary_size,
            validation_size=validation_size,
            quality_threshold=quality_threshold,
        )

    def calibrate(
        self,
        partner_handler: 'ProtocolHandler',
        vocabulary: Optional[List[str]] = None,
        validation_vocabulary: Optional[List[str]] = None,
        quality_threshold: float = 0.80,
        verbose: bool = True,
    ) -> CalibrationResult:
        """
        Perform calibration with a partner agent.

        This computes transfer matrices that enable embedding transfer
        between the two agents' embedding spaces.

        Args:
            partner_handler: The partner agent's protocol handler
            vocabulary: Training vocabulary (uses default if None)
            validation_vocabulary: Held-out validation vocabulary
            quality_threshold: Minimum acceptable quality
            verbose: Whether to print progress

        Returns:
            CalibrationResult with transfer matrices and quality metrics
        """
        start_time = time.time()

        self._emit_event(
            event_type="calibration_start",
            partner_id=partner_handler.agent_id,
        )

        try:
            # Use default vocabulary if not provided
            if vocabulary is None:
                train_vocab, val_vocab = get_default_vocabulary()
                vocabulary = train_vocab
                if validation_vocabulary is None:
                    validation_vocabulary = val_vocab

            if validation_vocabulary is None:
                # Split vocabulary into train/val
                split_idx = int(len(vocabulary) * 0.9)
                train_vocab = vocabulary[:split_idx]
                val_vocab = vocabulary[split_idx:]
            else:
                train_vocab = vocabulary
                val_vocab = validation_vocabulary

            if verbose:
                print(f"\n{'='*60}")
                print(f"CALIBRATION: {self.agent_id} <-> {partner_handler.agent_id}")
                print(f"{'='*60}")
                print(f"Training vocabulary: {len(train_vocab):,} items")
                print(f"Validation vocabulary: {len(val_vocab):,} items")

            # Encode training vocabulary with both agents
            if verbose:
                print(f"\nEncoding training vocabulary...")
                print(f"  {self.agent_id}...")

            emb_A_train = self._encode_batch(train_vocab)

            if verbose:
                print(f"  {partner_handler.agent_id}...")

            emb_B_train = partner_handler._encode_batch(train_vocab)

            # Compute transfer matrices
            if verbose:
                print(f"\nComputing transfer matrices...")

            W_AB, W_BA = compute_transfer_matrices(
                emb_A_train, emb_B_train, method="ridge"
            )

            # Evaluate on training data
            train_metrics = evaluate_transfer_quality(
                emb_A_train, emb_B_train, W_AB, W_BA, sample_size=1000
            )
            training_similarity = train_metrics["roundtrip_mean_similarity"]

            if verbose:
                print(f"  Training forward: {train_metrics['forward_mean_similarity']:.4f}")
                print(f"  Training round-trip: {training_similarity:.4f}")

            # Validate on held-out data
            if verbose:
                print(f"\nValidating on held-out vocabulary...")

            emb_A_val = self._encode_batch(val_vocab)
            emb_B_val = partner_handler._encode_batch(val_vocab)

            val_metrics = evaluate_transfer_quality(
                emb_A_val, emb_B_val, W_AB, W_BA
            )
            validation_similarity = val_metrics["roundtrip_mean_similarity"]
            worst_case = val_metrics["roundtrip_min_similarity"]

            if verbose:
                print(f"  Validation round-trip: {validation_similarity:.4f}")
                print(f"  Worst-case: {worst_case:.4f}")

            # Check quality threshold
            if validation_similarity < quality_threshold:
                if verbose:
                    print(f"\n⚠️  WARNING: Quality ({validation_similarity:.4f}) "
                          f"below threshold ({quality_threshold:.4f})")
                    print("  Communication will still work but with reduced fidelity.")
                    print("  Consider using a larger calibration vocabulary.")
            else:
                if verbose:
                    print(f"\n✓ Quality threshold met "
                          f"({validation_similarity:.4f} >= {quality_threshold:.4f})")

            # Create and store transfer matrix
            valid_until = (
                datetime.now() + timedelta(days=self.matrix_validity_days)
            ).isoformat()

            transfer_matrix = TransferMatrix(
                matrix_AB=W_AB,
                matrix_BA=W_BA,
                training_similarity=training_similarity,
                validation_similarity=validation_similarity,
                worst_case_similarity=worst_case,
                valid_until=valid_until,
            )

            # Store in both handlers (thread-safe)
            with self._lock:
                key = f"{self.agent_id}_{partner_handler.agent_id}"
                self.transfer_matrices[key] = transfer_matrix

            with partner_handler._lock:
                partner_key = f"{partner_handler.agent_id}_{self.agent_id}"
                partner_handler.transfer_matrices[partner_key] = TransferMatrix(
                    matrix_AB=W_BA,  # Reversed for partner
                    matrix_BA=W_AB,
                    training_similarity=training_similarity,
                    validation_similarity=validation_similarity,
                    worst_case_similarity=worst_case,
                    valid_until=valid_until,
                )

            calibration_time = (time.time() - start_time) * 1000

            if verbose:
                print(f"\n✓ Calibration complete in {calibration_time:.0f}ms")

            result = CalibrationResult(
                success=True,
                transfer_matrix=transfer_matrix,
                training_similarity=training_similarity,
                validation_similarity=validation_similarity,
                worst_case_similarity=worst_case,
                calibration_time_ms=calibration_time,
                vocabulary_size=len(train_vocab),
            )

            self._emit_event(
                event_type="calibration_end",
                partner_id=partner_handler.agent_id,
                duration_ms=calibration_time,
                quality_score=validation_similarity,
                metadata={
                    "training_similarity": training_similarity,
                    "worst_case": worst_case,
                    "vocab_size": len(train_vocab),
                },
            )

            # Reset circuit breaker on successful calibration
            cb = self._get_circuit_breaker(partner_handler.agent_id)
            cb.reset()

            return result

        except Exception as e:
            calibration_time = (time.time() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"

            logger.error(f"Calibration failed: {error_msg}")

            self._emit_event(
                event_type="error",
                partner_id=partner_handler.agent_id,
                duration_ms=calibration_time,
                error=error_msg,
                metadata={"operation": "calibration"},
            )

            return CalibrationResult(
                success=False,
                error_message=error_msg,
                calibration_time_ms=calibration_time,
            )

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts with the embedder.

        Args:
            texts: List of texts to encode

        Returns:
            numpy array of embeddings [n_texts, dimensions]
        """
        start_time = time.time()

        # Process in batches to respect max_batch_size
        all_embeddings = []
        batch_size = self.capabilities.max_batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                embeddings = self.embedder.embed_batch(batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Batch encoding failed at index {i}: {e}")
                raise

        duration_ms = (time.time() - start_time) * 1000

        self._emit_event(
            event_type="embed_batch",
            duration_ms=duration_ms,
            cost_estimate=self._estimate_embed_cost(len(texts)),
            tokens_used=len(texts) * 10,  # rough estimate
            metadata={"batch_size": len(texts)},
        )

        return np.array(all_embeddings)

    def transfer_to(
        self,
        partner_agent_id: str,
        text: str,
    ) -> SemanticTransfer:
        """
        Transfer semantic content to a partner agent.

        Uses circuit breaker to prevent cascading failures.
        Falls back gracefully if transfer fails.

        Args:
            partner_agent_id: ID of the target agent
            text: Text to transfer

        Returns:
            SemanticTransfer object with the transferred embedding

        Raises:
            AgentNotCalibratedError: If not calibrated with the partner
            MatrixExpiredError: If the transfer matrix has expired
            CircuitOpenError: If the circuit breaker is open
        """
        start_time = time.time()

        # Check circuit breaker
        cb = self._get_circuit_breaker(partner_agent_id)

        def _do_transfer():
            # Get transfer matrix (thread-safe)
            with self._lock:
                key = f"{self.agent_id}_{partner_agent_id}"
                if key not in self.transfer_matrices:
                    raise AgentNotCalibratedError(self.agent_id, partner_agent_id)

                matrix = self.transfer_matrices[key]

            # Check if matrix is expired
            if matrix.is_expired():
                raise MatrixExpiredError(key, matrix.valid_until)

            # Encode text
            embedding = np.array(self.embedder.embed(text))

            # Transform to partner's space
            transferred = transfer_embedding(embedding, matrix.matrix_AB)

            # Create transfer object
            transfer_id = self._generate_transfer_id(text)

            return SemanticTransfer(
                transfer_id=transfer_id,
                embedding=transferred,
                source_agent=self.agent_id,
                target_agent=partner_agent_id,
                original_norm=float(np.linalg.norm(embedding)),
                expected_similarity=matrix.validation_similarity,
                timestamp=datetime.now().isoformat(),
            )

        try:
            semantic_transfer = cb.call(_do_transfer)

            duration_ms = (time.time() - start_time) * 1000

            # Log transfer
            with self._lock:
                self.transfer_log.append({
                    "transfer_id": semantic_transfer.transfer_id,
                    "source": self.agent_id,
                    "target": partner_agent_id,
                    "timestamp": semantic_transfer.timestamp,
                    "expected_quality": semantic_transfer.expected_similarity,
                    "duration_ms": duration_ms,
                })

            self._emit_event(
                event_type="transfer",
                partner_id=partner_agent_id,
                duration_ms=duration_ms,
                quality_score=semantic_transfer.expected_similarity,
                cost_estimate=self._estimate_embed_cost(1),
            )

            return semantic_transfer

        except CircuitOpenError:
            self._emit_event(
                event_type="circuit_break",
                partner_id=partner_agent_id,
                error="Circuit breaker open",
            )
            raise
        except (AgentNotCalibratedError, MatrixExpiredError):
            raise
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._emit_event(
                event_type="error",
                partner_id=partner_agent_id,
                duration_ms=duration_ms,
                error=str(e),
                metadata={"operation": "transfer"},
            )
            raise TransferError(
                f"Transfer to {partner_agent_id} failed: {e}",
                recoverable=True,
            )

    def transfer_embedding_to(
        self,
        partner_agent_id: str,
        embedding: np.ndarray,
    ) -> SemanticTransfer:
        """
        Transfer a pre-computed embedding to a partner agent.

        Args:
            partner_agent_id: ID of the target agent
            embedding: Pre-computed embedding vector

        Returns:
            SemanticTransfer object with the transferred embedding
        """
        with self._lock:
            key = f"{self.agent_id}_{partner_agent_id}"
            if key not in self.transfer_matrices:
                raise AgentNotCalibratedError(self.agent_id, partner_agent_id)

            matrix = self.transfer_matrices[key]

        if matrix.is_expired():
            raise MatrixExpiredError(key, matrix.valid_until)

        embedding = np.asarray(embedding)
        transferred = transfer_embedding(embedding, matrix.matrix_AB)

        transfer_id = self._generate_transfer_id(str(embedding[:5]))

        return SemanticTransfer(
            transfer_id=transfer_id,
            embedding=transferred,
            source_agent=self.agent_id,
            target_agent=partner_agent_id,
            original_norm=float(np.linalg.norm(embedding)),
            expected_similarity=matrix.validation_similarity,
            timestamp=datetime.now().isoformat(),
        )

    def receive_transfer(
        self,
        transfer: SemanticTransfer,
        original_text: Optional[str] = None,
    ) -> Dict:
        """
        Receive and validate a transferred embedding.

        Args:
            transfer: The received semantic transfer
            original_text: Optional original text for quality validation

        Returns:
            Acknowledgment dictionary with quality metrics
        """
        # Validate transfer
        if transfer.target_agent != self.agent_id:
            return {
                "status": "error",
                "message": (
                    f"Transfer intended for {transfer.target_agent}, "
                    f"not {self.agent_id}"
                ),
            }

        # Get reverse transfer matrix
        with self._lock:
            key = f"{self.agent_id}_{transfer.source_agent}"
            if key not in self.transfer_matrices:
                return {
                    "status": "error",
                    "message": "No calibration found for source agent",
                }

            matrix = self.transfer_matrices[key]

        # Compute received norm
        received_norm = float(np.linalg.norm(transfer.embedding))

        # Validate quality if original text provided
        quality_metric = None
        if original_text:
            try:
                our_embedding = np.array(self.embedder.embed(original_text))
                quality_metric = cosine_similarity(transfer.embedding, our_embedding)
            except Exception as e:
                logger.warning(f"Quality validation failed: {e}")

        self._emit_event(
            event_type="receive",
            partner_id=transfer.source_agent,
            quality_score=quality_metric,
        )

        return {
            "message_type": "acknowledgment",
            "transfer_id": transfer.transfer_id,
            "status": "success",
            "received_norm": received_norm,
            "original_norm": transfer.original_norm,
            "norm_ratio": transfer.get_norm_ratio(),
            "quality_metric": float(quality_metric) if quality_metric else None,
            "timestamp": datetime.now().isoformat(),
        }

    def send_message(
        self,
        partner_agent_id: str,
        message: str,
        fallback_to_text: bool = True,
    ) -> Dict:
        """
        Send a message to a partner agent with automatic fallback.

        Tries AECP first. If that fails (not calibrated, circuit open, etc.),
        falls back to sending the message as plain English text.

        Args:
            partner_agent_id: Target agent ID
            message: Message to send
            fallback_to_text: Whether to fall back to text on failure

        Returns:
            Dictionary with either AECP transfer or text fallback
        """
        def _aecp_transfer():
            transfer = self.transfer_to(partner_agent_id, message)
            return {
                "method": "aecp",
                "transfer_id": transfer.transfer_id,
                "embedding": transfer.embedding.tolist(),
                "source_agent": transfer.source_agent,
                "target_agent": transfer.target_agent,
                "expected_similarity": transfer.expected_similarity,
                "timestamp": transfer.timestamp,
            }

        if fallback_to_text:
            return self._degradation.send(
                message=message,
                aecp_func=_aecp_transfer,
            )
        else:
            return _aecp_transfer()

    def embed(self, text: str) -> np.ndarray:
        """
        Embed text using this agent's embedder.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        start_time = time.time()

        try:
            result = np.array(self.embedder.embed(text))
        except Exception as e:
            self._emit_event(
                event_type="error",
                error=str(e),
                metadata={"operation": "embed"},
            )
            raise

        duration_ms = (time.time() - start_time) * 1000

        self._emit_event(
            event_type="embed",
            duration_ms=duration_ms,
            cost_estimate=self._estimate_embed_cost(1),
        )

        return result

    def get_calibration_stats(self, partner_agent_id: str) -> Optional[Dict]:
        """
        Get calibration statistics for a partner agent.

        Args:
            partner_agent_id: ID of the partner agent

        Returns:
            Dictionary with calibration stats, or None if not calibrated
        """
        with self._lock:
            key = f"{self.agent_id}_{partner_agent_id}"
            if key not in self.transfer_matrices:
                return None

            matrix = self.transfer_matrices[key]

        stats = matrix.to_dict()
        stats["circuit_breaker"] = self._get_circuit_breaker(partner_agent_id).get_status()
        return stats

    def requires_recalibration(self, partner_agent_id: str) -> bool:
        """
        Check if recalibration is needed with a partner.

        Args:
            partner_agent_id: ID of the partner agent

        Returns:
            True if recalibration is needed
        """
        with self._lock:
            key = f"{self.agent_id}_{partner_agent_id}"
            if key not in self.transfer_matrices:
                return True

            matrix = self.transfer_matrices[key]

        # Check expiration
        if matrix.is_expired():
            return True

        # Check quality
        if not matrix.meets_quality_threshold(self.capabilities.min_quality_threshold):
            return True

        return False

    def get_connection_health(self) -> Dict[str, Dict]:
        """
        Get health status of all connections.

        Returns:
            Dictionary mapping partner IDs to health status
        """
        health = {}
        with self._lock:
            for key, matrix in self.transfer_matrices.items():
                parts = key.split("_", 1)
                if len(parts) == 2 and parts[0] == self.agent_id:
                    partner_id = parts[1]
                    cb = self._get_circuit_breaker(partner_id)
                    health[partner_id] = {
                        "calibrated": True,
                        "expired": matrix.is_expired(),
                        "quality": matrix.validation_similarity,
                        "circuit_breaker": cb.state.value,
                        "valid_until": matrix.valid_until,
                    }
        return health

    def _generate_transfer_id(self, content: str) -> str:
        """Generate a unique transfer ID."""
        data = f"{content}{datetime.now().isoformat()}{self.agent_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _estimate_embed_cost(self, num_texts: int) -> float:
        """Estimate embedding cost."""
        if self.monitor:
            return self.monitor.estimate_cost(
                self.capabilities.embedding_model,
                num_texts,
            )
        return 0.0


# Alias for simpler API
class AECP(ProtocolHandler):
    """
    AECP - Agent Embedding Communication Protocol

    Main interface for the AECP library. This is an alias for ProtocolHandler
    with a simpler API.

    Features:
    - Automatic calibration between different embedding models
    - Circuit breaker for failing connections
    - Graceful fallback to English text when AECP is unavailable
    - Real-time debug monitoring and cost tracking
    - Thread-safe for multi-agent environments

    Example:
        >>> from aecp import AECP
        >>> from aecp.adapters import OpenAIAdapter
        >>>
        >>> agent = AECP(OpenAIAdapter(api_key="sk-..."), agent_id="my_agent")
        >>>
        >>> # Calibrate with another agent
        >>> agent.calibrate_with(other_agent)
        >>>
        >>> # Transfer text (falls back to English if AECP unavailable)
        >>> result = agent.send_message(other_agent.agent_id, "Hello world")
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        agent_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize AECP agent.

        Args:
            embedder: Embedding provider (adapter)
            agent_id: Unique identifier (auto-generated if not provided)
            **kwargs: Additional arguments passed to ProtocolHandler
        """
        if agent_id is None:
            agent_id = f"agent_{hashlib.md5(str(id(embedder)).encode()).hexdigest()[:8]}"

        super().__init__(agent_id, embedder, **kwargs)

    def calibrate_with(
        self,
        other: 'AECP',
        vocabulary: Optional[List[str]] = None,
        **kwargs,
    ) -> CalibrationResult:
        """
        Calibrate with another AECP agent.

        Args:
            other: The other AECP agent
            vocabulary: Optional custom vocabulary
            **kwargs: Additional arguments passed to calibrate()

        Returns:
            CalibrationResult
        """
        return self.calibrate(other, vocabulary=vocabulary, **kwargs)
