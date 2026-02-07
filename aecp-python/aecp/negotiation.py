"""
AECP Auto-Negotiation Module

Enables automatic protocol negotiation between agents.
If both agents support AECP, they will use it automatically.
Otherwise, they fall back to plain English text communication.

Key features:
- Auto-detects AECP support on both sides
- Graceful English text fallback (default language)
- Circuit breaker integration
- Debug monitor integration
- Comprehensive error handling
"""

import time
import logging
import warnings
from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass, field

from .protocol import AECP, CalibrationResult
from .types import EmbeddingProvider
from .errors import (
    AECPError,
    NegotiationError,
    CircuitOpenError,
    GracefulDegradation,
    handle_aecp_errors,
)

logger = logging.getLogger("aecp.negotiation")


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
        fallback_language: Language used for text fallback (default: English)
        negotiation_time_ms: Time taken for negotiation
    """
    uses_aecp: bool
    agent1_supports: bool
    agent2_supports: bool
    calibration_result: Optional[CalibrationResult] = None
    fallback_reason: Optional[str] = None
    fallback_language: str = "en"
    negotiation_time_ms: float = 0.0

    @property
    def method_name(self) -> str:
        """Get human-readable method name."""
        if self.uses_aecp:
            quality = (
                self.calibration_result.validation_similarity
                if self.calibration_result
                else 0
            )
            return f"AECP (quality: {quality:.1%})"
        return f"English text ({self.fallback_language})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "uses_aecp": self.uses_aecp,
            "agent1_supports": self.agent1_supports,
            "agent2_supports": self.agent2_supports,
            "fallback_reason": self.fallback_reason,
            "fallback_language": self.fallback_language,
            "negotiation_time_ms": self.negotiation_time_ms,
            "method": self.method_name,
        }


class AECPNegotiator:
    """
    Handles automatic AECP negotiation between agents.

    This class wraps agent communication and automatically:
    1. Detects if both agents support AECP
    2. Calibrates if both support it
    3. Falls back to English text if one doesn't support it
    4. Provides clear feedback about the negotiation
    5. Integrates with circuit breakers and debug monitoring

    The default fallback language is English, ensuring that agents
    can always communicate even without AECP.

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
        >>> # Auto-negotiate - will fall back to English
        >>> method = AECPNegotiator.negotiate(agent1, agent2)
        >>> print(method.method_name)
        'English text (en)'
        >>>
        >>> # Both with AECP
        >>> agent2_aecp = AECP(MockAdapter(dimensions=768))
        >>> method = AECPNegotiator.negotiate(agent1, agent2_aecp)
        >>> print(method.method_name)
        'AECP (quality: 97.3%)'
    """

    @staticmethod
    def is_aecp_agent(agent: Any) -> bool:
        """
        Check if an agent supports AECP.

        Checks for:
        1. Instance of AECP class
        2. Has required AECP methods (duck typing)
        3. Has supports_aecp attribute

        Args:
            agent: The agent to check

        Returns:
            True if agent supports AECP
        """
        if agent is None:
            return False

        # Direct instance check
        if isinstance(agent, AECP):
            return True

        # Duck typing check - does it have the required methods?
        required_methods = ["calibrate_with", "transfer_to", "embed", "send_handshake"]
        has_methods = all(hasattr(agent, m) and callable(getattr(agent, m)) for m in required_methods)
        if has_methods:
            return True

        # Check for explicit flag
        if getattr(agent, "supports_aecp", False):
            return True

        return False

    @staticmethod
    def negotiate(
        agent1: Union[AECP, Any],
        agent2: Union[AECP, Any],
        auto_calibrate: bool = True,
        verbose: bool = True,
        fallback_language: str = "en",
        **calibration_kwargs,
    ) -> CommunicationMethod:
        """
        Automatically negotiate communication method between two agents.

        This method:
        1. Checks if both agents support AECP
        2. If yes, calibrates them and uses AECP
        3. If no, falls back to text (English by default)
        4. Never raises exceptions - always returns a valid method

        Args:
            agent1: First agent
            agent2: Second agent
            auto_calibrate: Whether to automatically calibrate if both support AECP
            verbose: Whether to print negotiation messages
            fallback_language: Language for text fallback (default: "en" for English)
            **calibration_kwargs: Additional arguments passed to calibrate()

        Returns:
            CommunicationMethod describing the negotiated method
        """
        start_time = time.time()

        # Emit debug event
        monitor = None
        try:
            from .debug import DebugMonitor, DebugEvent, EventType
            monitor = DebugMonitor.get_global()
        except ImportError:
            pass

        agent1_supports = AECPNegotiator.is_aecp_agent(agent1)
        agent2_supports = AECPNegotiator.is_aecp_agent(agent2)

        def _make_result(**kwargs) -> CommunicationMethod:
            elapsed = (time.time() - start_time) * 1000
            result = CommunicationMethod(
                negotiation_time_ms=elapsed,
                fallback_language=fallback_language,
                **kwargs,
            )

            # Emit debug event
            if monitor:
                from .debug import DebugEvent, EventType
                agent_id = getattr(agent1, "agent_id", "agent1") if agent1_supports else "agent1"
                partner_id = getattr(agent2, "agent_id", "agent2") if agent2_supports else "agent2"
                monitor.log_event(DebugEvent(
                    event_type=EventType.NEGOTIATION if result.uses_aecp else EventType.FALLBACK,
                    timestamp=time.time(),
                    agent_id=str(agent_id),
                    partner_id=str(partner_id),
                    duration_ms=elapsed,
                    quality_score=(
                        result.calibration_result.validation_similarity
                        if result.calibration_result and result.calibration_result.success
                        else None
                    ),
                    metadata={
                        "uses_aecp": result.uses_aecp,
                        "fallback_reason": result.fallback_reason,
                    },
                ))

            return result

        # Case 1: Both support AECP
        if agent1_supports and agent2_supports:
            if auto_calibrate:
                if verbose:
                    print("\n Both agents support AECP. Calibrating...")

                try:
                    calibration_result = agent1.calibrate_with(
                        agent2,
                        verbose=verbose,
                        **calibration_kwargs,
                    )

                    if calibration_result.success:
                        quality = calibration_result.validation_similarity
                        if verbose:
                            print(f"✓ AECP enabled with {quality:.1%} semantic fidelity")

                        return _make_result(
                            uses_aecp=True,
                            agent1_supports=True,
                            agent2_supports=True,
                            calibration_result=calibration_result,
                        )
                    else:
                        reason = f"Calibration failed: {calibration_result.error_message}"
                        if verbose:
                            print(
                                f"⚠️  {reason}. "
                                f"Falling back to English text communication."
                            )
                        return _make_result(
                            uses_aecp=False,
                            agent1_supports=True,
                            agent2_supports=True,
                            fallback_reason=reason,
                        )
                except Exception as e:
                    reason = f"Calibration error: {type(e).__name__}: {e}"
                    if verbose:
                        print(
                            f"⚠️  {reason}. "
                            f"Falling back to English text communication."
                        )
                    logger.error(f"Negotiation calibration error: {e}")
                    return _make_result(
                        uses_aecp=False,
                        agent1_supports=True,
                        agent2_supports=True,
                        fallback_reason=reason,
                    )
            else:
                if verbose:
                    print(" Both agents support AECP (auto-calibrate disabled)")

                return _make_result(
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

            reason = (
                f"{' and '.join(missing)} "
                f"do{'es' if len(missing) == 1 else ''} not support AECP"
            )

            if verbose:
                print(
                    f"\n {reason}. "
                    f"Using English text communication (default)."
                )
                if not agent1_supports and not agent2_supports:
                    print(
                        "   Both agents will communicate in plain English. "
                        "No embedding transfer needed."
                    )
                else:
                    supporting = "Agent 1" if agent1_supports else "Agent 2"
                    print(
                        f"   {supporting} supports AECP but the other doesn't. "
                        f"Using English text for compatibility."
                    )

            return _make_result(
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
        fallback_language: str = "en",
        **calibration_kwargs,
    ) -> Dict[str, Any]:
        """
        Send a message from one agent to another using the best available method.

        If AECP is available, sends as embedding transfer.
        If not, sends as plain English text.

        This method NEVER raises exceptions - it always returns a valid result.

        Args:
            sender: Sending agent
            receiver: Receiving agent
            message: Message to send
            method: Pre-negotiated communication method (will auto-negotiate if None)
            auto_calibrate: Whether to automatically calibrate if both support AECP
            verbose: Whether to print negotiation messages
            fallback_language: Language for text fallback
            **calibration_kwargs: Additional calibration arguments

        Returns:
            Dictionary with message result:
            - If AECP: {'method': 'aecp', 'transfer_id': ..., 'embedding': ..., ...}
            - If text: {'method': 'text', 'message': ..., 'language': 'en', ...}
        """
        # Auto-negotiate if method not provided
        if method is None:
            method = AECPNegotiator.negotiate(
                sender, receiver,
                auto_calibrate=auto_calibrate,
                verbose=verbose,
                fallback_language=fallback_language,
                **calibration_kwargs,
            )

        # Use AECP if available
        if method.uses_aecp:
            try:
                if hasattr(sender, 'send_message'):
                    result = sender.send_message(receiver.agent_id, message)
                    if result.get("method") == "aecp" or result.get("data"):
                        return result
                    # send_message already fell back to text
                    return result

                transfer = sender.transfer_to(receiver.agent_id, message)
                return {
                    "method": "aecp",
                    "transfer_id": transfer.transfer_id,
                    "embedding": transfer.embedding.tolist(),
                    "source_agent": transfer.source_agent,
                    "target_agent": transfer.target_agent,
                    "expected_similarity": transfer.expected_similarity,
                    "timestamp": transfer.timestamp,
                }
            except Exception as e:
                # AECP failed, fall back to text
                logger.warning(
                    f"AECP transfer failed ({e}), falling back to English text"
                )
                return {
                    "method": "text",
                    "message": message,
                    "language": fallback_language,
                    "fallback": True,
                    "fallback_reason": f"AECP transfer failed: {e}",
                }
        else:
            # Fall back to plain English text
            return {
                "method": "text",
                "message": message,
                "language": fallback_language,
                "fallback_reason": method.fallback_reason,
                "note": (
                    "Message sent as plain English text. "
                    "Both agents can understand this format."
                ),
            }

    @staticmethod
    def batch_send(
        sender: Union[AECP, Any],
        receiver: Union[AECP, Any],
        messages: List[str],
        method: Optional[CommunicationMethod] = None,
        verbose: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Send multiple messages efficiently.

        Args:
            sender: Sending agent
            receiver: Receiving agent
            messages: List of messages to send
            method: Pre-negotiated method (negotiates once if None)
            verbose: Whether to print messages
            **kwargs: Additional arguments

        Returns:
            List of message results
        """
        if method is None:
            method = AECPNegotiator.negotiate(
                sender, receiver, verbose=verbose, **kwargs
            )

        results = []
        for msg in messages:
            result = AECPNegotiator.send_message(
                sender, receiver, msg,
                method=method,
                verbose=False,
            )
            results.append(result)

        return results


def enable_aecp_for_agent(
    agent: Any,
    embedder: Optional[EmbeddingProvider] = None,
    agent_id: Optional[str] = None,
    **kwargs,
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
            "Cannot enable AECP: agent does not support AECP and no embedder provided. "
            "Please provide an embedder (e.g., OpenAIAdapter, MockAdapter) to enable AECP."
        )

    return AECP(embedder, agent_id=agent_id, **kwargs)


__all__ = [
    "AECPNegotiator",
    "CommunicationMethod",
    "enable_aecp_for_agent",
]
