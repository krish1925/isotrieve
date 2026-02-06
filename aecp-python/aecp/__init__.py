"""
AECP - Agent Embedding Communication Protocol

Enable AI agents with different embedding models to communicate
with 97% semantic fidelity preservation.

Example:
    >>> from aecp import AECP
    >>> from aecp.adapters import OpenAIAdapter, VoyageAdapter
    >>> 
    >>> # Initialize agents with different models
    >>> agent1 = AECP(OpenAIAdapter(api_key="sk-..."))
    >>> agent2 = AECP(VoyageAdapter(api_key="pa-..."))
    >>> 
    >>> # One-time calibration
    >>> agent1.calibrate_with(agent2)
    >>> 
    >>> # Transfer embeddings between agents
    >>> embedding = agent1.embed("machine learning")
    >>> transferred = agent1.transfer_to(agent2, embedding)

For more information, see: https://github.com/yourusername/aecp
"""

from .protocol import AECP, ProtocolHandler, CalibrationResult
from .matrix import (
    compute_transfer_matrices,
    transfer_embedding,
    cosine_similarity,
    evaluate_transfer_quality,
)
from .types import (
    EmbeddingProvider,
    AgentCapabilities,
    TransferMatrix,
    SemanticTransfer,
)
from .plugin import (
    register_adapter,
    get_adapter,
    create_adapter,
    list_adapters,
    adapter,
)
from .negotiation import (
    AECPNegotiator,
    CommunicationMethod,
    enable_aecp_for_agent,
)

__version__ = "1.0.0"
__author__ = "AECP Contributors"
__license__ = "MIT"

__all__ = [
    # Main classes
    "AECP",
    "ProtocolHandler",
    "CalibrationResult",
    # Matrix operations
    "compute_transfer_matrices",
    "transfer_embedding",
    "cosine_similarity",
    "evaluate_transfer_quality",
    # Types
    "EmbeddingProvider",
    "AgentCapabilities",
    "TransferMatrix",
    "SemanticTransfer",
    # Plugin system
    "register_adapter",
    "get_adapter",
    "create_adapter",
    "list_adapters",
    "adapter",
    # Auto-negotiation
    "AECPNegotiator",
    "CommunicationMethod",
    "enable_aecp_for_agent",
]
