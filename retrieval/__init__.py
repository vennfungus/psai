"""Dense semantic retrieval for PSAI document chunks."""

from .config import RetrievalConfig
from .retriever import Retriever

__all__ = ["RetrievalConfig", "Retriever"]
