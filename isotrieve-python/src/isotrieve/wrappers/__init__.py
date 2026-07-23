"""Query-time wrappers for embedding frameworks.

These wrappers map new-model queries into legacy-model space on-the-fly,
with zero writes to the vector store.  They are the 5-minute trial path.

Modules
-------
llamaindex
    LlamaIndex ``BaseEmbedding`` subclass (query-only).
openai_shim
    OpenAI client wrapper preserving call signature.
"""

from __future__ import annotations


class IsotrieveWrapperUsageError(Exception):
    """Raised when a wrapper is used in a way it was not designed for.

    Specifically, calling ``embed_documents`` on a query-only wrapper
    raises this with a message explaining the wrapper is query-time only
    and pointing to the migration docs.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "This wrapper is query-time only. "
            "Document embeddings must come from the legacy model. "
            "To migrate your corpus, use 'isotrieve migrate' — see docs/migration.md"
        )
