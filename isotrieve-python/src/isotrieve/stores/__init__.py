"""Store adapters."""

from isotrieve.stores.base import VectorRecord, VectorStore
from isotrieve.stores.numpy_files import NumpyFileStore

__all__ = ["VectorRecord", "VectorStore", "NumpyFileStore"]
