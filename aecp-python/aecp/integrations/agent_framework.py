"""
AECP-Enabled Agent Framework

Provides ``AECPEnabledAgent`` — a higher-level agent that adds a local
knowledge base with RAG (Retrieval-Augmented Generation) on top of the
base ``AECPAgent``.

This is the concrete implementation of Scenario 2 (Pydantic AI / framework
integration):

    ┌──────────────────────────────┐
    │  AECPEnabledAgent            │
    │  ┌────────────┐  ┌────────┐ │
    │  │ LLM (gen.) │  │ AECP   │ │
    │  │ (Pydantic, │  │ (embed │ │
    │  │  LangChain)│  │  layer)│ │
    │  └────────────┘  └────────┘ │
    └──────────────────────────────┘

The LLM is used **only** for text generation.
AECP is used **only** for semantic operations (embed, search, transfer).

Example:
    >>> from aecp.integrations import AECPEnabledAgent
    >>> from aecp.adapters import OpenAIAdapter
    >>>
    >>> agent = AECPEnabledAgent(
    ...     llm_provider="openai:gpt-4",
    ...     embedder=OpenAIAdapter(model="text-embedding-3-small"),
    ... )
    >>>
    >>> # Build knowledge base
    >>> agent.add_knowledge([
    ...     "Dopamine is a neurotransmitter.",
    ...     "Serotonin regulates mood.",
    ... ])
    >>>
    >>> # Retrieve relevant context
    >>> results = agent.retrieve("What regulates mood?", top_k=1)
    >>> print(results)
    [('Serotonin regulates mood.', 0.87)]
    >>>
    >>> # Cross-agent communication
    >>> other = AECPEnabledAgent(
    ...     embedder=OpenAIAdapter(model="text-embedding-3-large"),
    ... )
    >>> agent.calibrate_with(other)
    >>> transferred = agent.communicate_with(other, "dopamine pathways")
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..protocol import AECP, CalibrationResult
from ..types import EmbeddingProvider, SemanticTransfer
from ..matrix import cosine_similarity
from .base import AECPAgent

logger = logging.getLogger("aecp.integrations.agent_framework")


class AECPEnabledAgent(AECPAgent):
    """
    Agent with decoupled LLM + AECP embeddings and a local knowledge base.

    Extends ``AECPAgent`` with:
    - A simple in-memory knowledge base (texts + embeddings).
    - Retrieval (semantic search) over the knowledge base.
    - Cross-agent communication via AECP transfer.

    The LLM generation side is intentionally left as a thin shell so
    that callers can plug in *any* framework (Pydantic AI, LangChain,
    bare OpenAI client, etc.).

    Attributes:
        knowledge_texts: Stored knowledge texts.
        knowledge_embeddings: Corresponding embedding vectors.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        llm_provider: Optional[str] = None,
        system_prompt: str = "",
        use_compression: bool = False,
        agent_id: Optional[str] = None,
        **aecp_kwargs: Any,
    ):
        """
        Initialise an AECP-enabled agent with knowledge base.

        Args:
            embedder: AECP-compatible embedding provider.
            llm_provider: LLM identifier (e.g. ``"openai:gpt-4"``).
            system_prompt: System prompt for LLM generation.
            agent_id: Unique agent identifier.
            **aecp_kwargs: Extra kwargs forwarded to ``AECP()``.
        """
        super().__init__(
            embedder=embedder,
            llm_provider=llm_provider,
            system_prompt=system_prompt,
            use_compression=use_compression,
            agent_id=agent_id,
            **aecp_kwargs,
        )

        # In-memory knowledge base
        self.knowledge_texts: List[str] = []
        self.knowledge_embeddings: List[np.ndarray] = []

    # ── Knowledge Base ───────────────────────────────────────────────

    def add_knowledge(self, texts: List[str]) -> int:
        """
        Add texts to the knowledge base.

        Each text is embedded with AECP and stored alongside the
        original text for later retrieval.

        Args:
            texts: Texts to add.

        Returns:
            Total number of items in the knowledge base after insertion.
        """
        embeddings = self.embed_batch(texts)
        for i, text in enumerate(texts):
            self.knowledge_texts.append(text)
            self.knowledge_embeddings.append(embeddings[i])

        logger.info(
            "Added %d items to knowledge base (total: %d)",
            len(texts),
            len(self.knowledge_texts),
        )
        return len(self.knowledge_texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant knowledge items for a query.

        Args:
            query: Query text.
            top_k: Number of results to return.

        Returns:
            List of ``(text, similarity_score)`` tuples sorted by
            descending similarity.
        """
        if not self.knowledge_embeddings:
            return []

        query_emb = self.embed(query)
        similarities = [
            cosine_similarity(query_emb, emb)
            for emb in self.knowledge_embeddings
        ]

        sorted_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )

        results = []
        for idx in sorted_indices[:top_k]:
            results.append((self.knowledge_texts[idx], similarities[idx]))
        return results

    def build_context(self, query: str, top_k: int = 3) -> str:
        """
        Build a context string from the most relevant knowledge items.

        Useful for RAG: feed the returned string into the LLM prompt.

        Args:
            query: Query text.
            top_k: Number of context items.

        Returns:
            Newline-separated context string.
        """
        results = self.retrieve(query, top_k=top_k)
        return "\n".join(text for text, _score in results)

    def clear_knowledge(self) -> None:
        """Remove all items from the knowledge base."""
        self.knowledge_texts.clear()
        self.knowledge_embeddings.clear()

    # ── Cross-Agent Communication ────────────────────────────────────

    def communicate_with(
        self,
        other: Union["AECPEnabledAgent", "AECPAgent", AECP],
        query: str,
    ) -> Dict[str, Any]:
        """
        Send a query to another agent via AECP embedding transfer.

        The query is embedded in *this* agent's space, then transferred
        to the *other* agent's space, preserving semantic fidelity.

        Args:
            other: Target agent.
            query: Query text to transfer.

        Returns:
            Dictionary with:
            - ``transferred_embedding``: The embedding in the other
              agent's space.
            - ``query``: The original query text.
            - ``source_agent``: This agent's ID.
            - ``target_agent``: The other agent's ID.
            - ``expected_similarity``: Expected quality from calibration.
        """
        query_emb = self.embed(query)
        transfer = self.transfer_to(other, query_emb)

        return {
            "transferred_embedding": transfer.embedding,
            "query": query,
            "source_agent": self.agent_id,
            "target_agent": transfer.target_agent,
            "expected_similarity": transfer.expected_similarity,
        }

    def process_transferred_query(
        self,
        transferred_embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Process an embedding that was transferred from another agent.

        Searches the local knowledge base using the transferred
        embedding (which is already in *this* agent's embedding space).

        Args:
            transferred_embedding: Embedding in this agent's space.
            top_k: Number of results.

        Returns:
            List of ``(text, similarity_score)`` tuples.
        """
        if not self.knowledge_embeddings:
            return []

        similarities = [
            cosine_similarity(transferred_embedding, emb)
            for emb in self.knowledge_embeddings
        ]

        sorted_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )

        results = []
        for idx in sorted_indices[:top_k]:
            results.append((self.knowledge_texts[idx], similarities[idx]))
        return results

    # ── Stats ────────────────────────────────────────────────────────

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Return knowledge base statistics."""
        return {
            "knowledge_size": len(self.knowledge_texts),
            "embedding_dimensions": (
                self.knowledge_embeddings[0].shape[0]
                if self.knowledge_embeddings
                else 0
            ),
            **self.get_info(),
        }

    def __repr__(self) -> str:
        return (
            f"AECPEnabledAgent(id={self.agent_id!r}, "
            f"llm={self.llm_provider!r}, "
            f"embedder={self.aecp.capabilities.embedding_model!r}, "
            f"knowledge={len(self.knowledge_texts)} items)"
        )
