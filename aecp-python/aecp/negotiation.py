"""
AECP Auto-Negotiation Module

Enables automatic protocol negotiation between agents.
If both agents support AECP, they will use it automatically.
Otherwise, they fall back to plain text communication.
"""

from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
import warnings

from .protocol import AECP, CalibrationResult
from .types import EmbeddingProvider


@dataclass
class CommunicationMethod:
    """
    Represents the negotiated communication method between two agents.
    
    Attributes:
        uses_aecp: Whether AECP is being used
        agent1_supports: Whether agent 1 supports AECP
        agent2_supports: Whether agent 2 supports AECP
        calibration_result: Calibration result if AECP is used
        fallback_reason: Reason for fallback if not using AECP
    """
    uses_aecp: bool
    agent1_supports: bool
    agent2_supports: bool
    calibration_result: Optional[CalibrationResult] = None
    fallback_reason: Optional[str] = None


class AECPNegotiator:
    """
    Handles automatic AECP negotiation between agents.
    
    This class wraps agent communication and automatically:
    1. Detects if both agents support AECP
    2. Calibrates if both support it
    3. Falls back to text if one doesn't support it
    4. Provides clear feedback about the negotiation
    
    Example:
        >>> from aecp import AECP
        >>> from aecp.adapters import MockAdapter
        >>> from aecp.negotiation import AECPNegotiator
        >>> 
        >>> # Agent 1 with AECP support
        >>> agent1 = AECP(MockAdapter(dimensions=384))
        >>> 
        >>> # Agent 2 without AECP (just a dict or None)
        >>> agent2 = None
        >>> 
        >>> # Auto-negotiate
        >>> method = AECPNegotiator.negotiate(agent1, agent2)
        >>> # Output: "⚠️  AECP not available: Agent 2 does not support AECP. Falling back to text communication."
        >>> 
        >>> # Both with AECP
        >>> agent2_aecp = AECP(MockAdapter(dimensions=768))
        >>> method = AECPNegotiator.negotiate(agent1, agent2_aecp)
        >>> # Output: "✓ AECP enabled with quality 0.9734"
    """
    
    @staticmethod
    def is_aecp_agent(agent: Any) -> bool:
        """
        Check if an agent supports AECP.
        
        Args:
            agent: The agent to check
            
        Returns:
            True if agent supports AECP
        """
        return isinstance(agent, AECP)
    
    @staticmethod
    def negotiate(
        agent1: Union[AECP, Any],
        agent2: Union[AECP, Any],
        auto_calibrate: bool = True,
        verbose: bool = True,
        **calibration_kwargs
    ) -> CommunicationMethod:
        """
        Automatically negotiate communication method between two agents.
        
        This method:
        1. Checks if both agents support AECP
        2. If yes, calibrates them and uses AECP
        3. If no, falls back to text and shows a warning
        
        Args:
            agent1: First agent
            agent2: Second agent
            auto_calibrate: Whether to automatically calibrate if both support AECP
            verbose: Whether to print negotiation messages
            **calibration_kwargs: Additional arguments passed to calibrate()
            
        Returns:
            CommunicationMethod describing the negotiated method
        """
        agent1_supports = AECPNegotiator.is_aecp_agent(agent1)
        agent2_supports = AECPNegotiator.is_aecp_agent(agent2)
        
        # Case 1: Both support AECP
        if agent1_supports and agent2_supports:
            if auto_calibrate:
                if verbose:
                    print("\n🤝 Both agents support AECP. Calibrating...")
                
                calibration_result = agent1.calibrate_with(
                    agent2, 
                    verbose=verbose,
                    **calibration_kwargs
                )
                
                if calibration_result.success:
                    quality = calibration_result.validation_similarity
                    if verbose:
                        print(f"✓ AECP enabled with {quality:.1%} semantic fidelity")
                    
                    return CommunicationMethod(
                        uses_aecp=True,
                        agent1_supports=True,
                        agent2_supports=True,
                        calibration_result=calibration_result,
                    )
                else:
                    if verbose:
                        warnings.warn(
                            f"⚠️  AECP calibration failed: {calibration_result.error_message}. "
                            "Falling back to text communication.",
                            UserWarning
                        )
                    return CommunicationMethod(
                        uses_aecp=False,
                        agent1_supports=True,
                        agent2_supports=True,
                        fallback_reason=f"Calibration failed: {calibration_result.error_message}",
                    )
            else:
                if verbose:
                    print("🤝 Both agents support AECP (auto-calibrate disabled)")
                
                return CommunicationMethod(
                    uses_aecp=False,
                    agent1_supports=True,
                    agent2_supports=True,
                    fallback_reason="Auto-calibration disabled",
                )
        
        # Case 2: Only one or neither supports AECP
        else:
            missing = []
            if not agent1_supports:
                missing.append("Agent 1")
            if not agent2_supports:
                missing.append("Agent 2")
            
            reason = f"{' and '.join(missing)} do{'es' if len(missing) == 1 else ''} not support AECP"
            
            if verbose:
                warnings.warn(
                    f"⚠️  AECP not available: {reason}. "
                    "Falling back to text communication.",
                    UserWarning
                )
            
            return CommunicationMethod(
                uses_aecp=False,
                agent1_supports=agent1_supports,
                agent2_supports=agent2_supports,
                fallback_reason=reason,
            )
    
    @staticmethod
    def send_message(
        sender: Union[AECP, Any],
        receiver: Union[AECP, Any],
        message: str,
        method: Optional[CommunicationMethod] = None,
        auto_calibrate: bool = True,
        verbose: bool = True,
        **calibration_kwargs
    ) -> Union[Dict, str]:
        """
        Send a message from one agent to another using the negotiated method.
        
        Args:
            sender: Sending agent
            receiver: Receiving agent
            message: Message to send
            method: Pre-negotiated communication method (will auto-negotiate if None)
            auto_calibrate: Whether to automatically calibrate if both support AECP
            verbose: Whether to print negotiation messages
            **calibration_kwargs: Additional calibration arguments
            
        Returns:
            If AECP: SemanticTransfer object as dict
            If text: The original message string
        """
        # Auto-negotiate if method not provided
        if method is None:
            method = AECPNegotiator.negotiate(
                sender, receiver, 
                auto_calibrate=auto_calibrate,
                verbose=verbose,
                **calibration_kwargs
            )
        
        # Use AECP if available
        if method.uses_aecp:
            transfer = sender.transfer_to(receiver.agent_id, message)
            return {
                'transfer_id': transfer.transfer_id,
                'embedding': transfer.embedding.tolist(),
                'source_agent': transfer.source_agent,
                'target_agent': transfer.target_agent,
                'expected_similarity': transfer.expected_similarity,
                'timestamp': transfer.timestamp,
                'method': 'aecp',
            }
        else:
            # Fall back to plain text
            return {
                'message': message,
                'method': 'text',
                'fallback_reason': method.fallback_reason,
            }


def enable_aecp_for_agent(
    agent: Any,
    embedder: Optional[EmbeddingProvider] = None,
    agent_id: Optional[str] = None,
    **kwargs
) -> Union[AECP, Any]:
    """
    Enable AECP support for an agent.
    
    If the agent already supports AECP, returns it unchanged.
    Otherwise, wraps it in an AECP instance.
    
    Args:
        agent: The agent to enable AECP for
        embedder: Embedding provider to use (required if agent doesn't support AECP)
        agent_id: Optional agent ID
        **kwargs: Additional AECP configuration
        
    Returns:
        AECP-enabled agent
        
    Raises:
        ValueError: If agent doesn't support AECP and no embedder provided
    """
    if isinstance(agent, AECP):
        return agent
    
    if embedder is None:
        raise ValueError(
            "Cannot enable AECP: agent does not support AECP and no embedder provided"
        )
    
    return AECP(embedder, agent_id=agent_id, **kwargs)


__all__ = [
    'AECPNegotiator',
    'CommunicationMethod',
    'enable_aecp_for_agent',
]
