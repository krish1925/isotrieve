"""
AECP Agent Base Class

Provides the foundational ``AECPAgent`` that decouples the embedding layer
(handled by AECP) from the LLM generation layer.  This is the key
architectural insight: AECP only needs to control the *embedding* layer,
while the LLM can remain coupled to any provider/framework.

Scenario mapping:
     Won't work — embedding tied to LLM:
        agent = Agent('openai:gpt-4')  # Can't access embeddings separately

     Will work — decoupled architecture:
        agent = AECPAgent(
            llm_provider='openai:gpt-4',
            embedder=OpenAIAdapter(model='text-embedding-3-small'),
        )

Example:
    >>> from aecp.integrations import AECPAgent
    >>> from aecp.adapters import OpenAIAdapter
    >>>
    >>> agent = AECPAgent(
    ...     llm_provider="openai:gpt-4",
    ...     embedder=OpenAIAdapter(model="text-embedding-3-small"),
    ...     system_prompt="You are a helpful assistant.",
    ... )
    >>>
    >>> # Embed with AECP (separate from LLM)
    >>> emb = agent.embed("test query")
    >>>
    >>> # Calibrate with another agent
    >>> agent.calibrate_with(other_agent)
    >>>
    >>> # Transfer embedding to another agent's space
    >>> transferred = agent.transfer_to(other_agent, emb)
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..protocol import AECP, CalibrationResult
from ..types import EmbeddingProvider, SemanticTransfer
from ..matrix import cosine_similarity

logger = logging.getLogger("aecp.integrations.base")


class AECPAgent:
    """
    Base class for AECP-enabled agents with decoupled LLM + embedder.

    The embedding layer is fully controlled by AECP, enabling:
    - Cross-model calibration and embedding transfer
    - Semantic fidelity preservation (97%+)
    - Framework-agnostic LLM usage

    The LLM provider string (e.g. ``"openai:gpt-4"``) is stored but
    **not** used directly by this class — subclasses or the caller are
    responsible for creating the actual LLM client.  This keeps the
    base class dependency-free (no pydantic-ai, langchain, etc.).

    Attributes:
        llm_provider: LLM provider identifier string.
        aecp: The underlying AECP protocol handler.
        system_prompt: System prompt for the LLM.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        llm_provider: Optional[str] = None,
        system_prompt: str = "",
        agent_id: Optional[str] = None,
        **aecp_kwargs: Any,
    ):
        """
        Initialise an AECP-enabled agent.

        Args:
            embedder: AECP-compatible embedding provider (adapter).
            llm_provider: LLM provider identifier (e.g. ``"openai:gpt-4"``).
                          Stored for reference; not used by this base class.
            system_prompt: System prompt for the LLM.
            agent_id: Unique agent identifier (auto-generated if omitted).
            **aecp_kwargs: Extra keyword arguments forwarded to ``AECP()``.
        """
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt

        # The AECP protocol handler owns the embedder
        self.aecp = AECP(embedder, agent_id=agent_id, **aecp_kwargs)

    # ── Convenience proxies ──────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        """Agent's unique identifier."""
        return self.aecp.agent_id

    @property
    def embedder(self) -> EmbeddingProvider:
        """The underlying embedding provider."""
        return self.aecp.embedder

    def embed(self, text: str) -> np.ndarray:
        """
        Embed text using AECP (not the LLM).

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        return self.aecp.embed(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts.

        Args:
            texts: Texts to embed.

        Returns:
            Embedding matrix ``[n_texts, dimensions]``.
        """
        return self.aecp._encode_batch(texts)

    def calibrate_with(
        self,
        other: Union["AECPAgent", AECP],
        vocabulary: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CalibrationResult:
        """
        Calibrate AECP transfer matrices with another agent.

        Args:
            other: Another ``AECPAgent`` or raw ``AECP`` instance.
            vocabulary: Custom calibration vocabulary.
            **kwargs: Extra arguments forwarded to ``AECP.calibrate_with``.

        Returns:
            CalibrationResult with quality metrics.
        """
        target_aecp = other.aecp if isinstance(other, AECPAgent) else other
        return self.aecp.calibrate_with(target_aecp, vocabulary=vocabulary, **kwargs)

    def transfer_to(
        self,
        other: Union["AECPAgent", AECP],
        embedding: np.ndarray,
    ) -> SemanticTransfer:
        """
        Transfer a pre-computed embedding to another agent's space.

        Args:
            other: Target agent (``AECPAgent`` or ``AECP``).
            embedding: Embedding in *this* agent's space.

        Returns:
            SemanticTransfer with the transformed embedding.
        """
        target_aecp = other.aecp if isinstance(other, AECPAgent) else other
        return self.aecp.transfer_embedding_to(target_aecp.agent_id, embedding)

    def send_message(
        self,
        other: Union["AECPAgent", AECP],
        message: str,
        fallback_to_text: bool = True,
    ) -> Dict[str, Any]:
        """
        Send a message to another agent (AECP with text fallback).

        Args:
            other: Target agent.
            message: Message text.
            fallback_to_text: Fall back to plain text on failure.

        Returns:
            Dictionary with transfer result or text fallback.
        """
        target_aecp = other.aecp if isinstance(other, AECPAgent) else other
        return self.aecp.send_message(
            target_aecp.agent_id, message, fallback_to_text=fallback_to_text
        )

    def find_similar(
        self,
        query_embedding: np.ndarray,
        candidates: List[np.ndarray],
        top_k: int = 5,
    ) -> List[int]:
        """
        Find the ``top_k`` most similar candidate embeddings.

        Args:
            query_embedding: Query vector.
            candidates: List of candidate vectors.
            top_k: Number of results to return.

        Returns:
            List of indices into ``candidates``, sorted by descending
            similarity.
        """
        similarities = [
            cosine_similarity(query_embedding, c) for c in candidates
        ]
        sorted_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )
        return sorted_indices[:top_k]

    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata."""
        return {
            "agent_id": self.agent_id,
            "llm_provider": self.llm_provider,
            "embedding_model": self.aecp.capabilities.embedding_model,
            "embedding_dimensions": self.aecp.capabilities.dimensions,
            "system_prompt_preview": (
                self.system_prompt[:80] + "…"
                if len(self.system_prompt) > 80
                else self.system_prompt
            ),
            "supports_aecp": True,
        }

    def __repr__(self) -> str:
        return (
            f"AECPAgent(id={self.agent_id!r}, "
            f"llm={self.llm_provider!r}, "
            f"embedder={self.aecp.capabilities.embedding_model!r})"
        )
