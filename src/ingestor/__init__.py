"""Ingestor - Comprehensive media-to-markdown ingestion for LLM RAG and fine-tuning."""

__version__ = "0.1.0"

from .types import MediaType, ExtractedImage, ExtractionResult, IngestConfig
from .core import (
    FileDetector,
    CharsetHandler,
    ExtractorRegistry,
    create_default_registry,
    Router,
)

__all__ = [
    # Version
    "__version__",
    # Types
    "MediaType",
    "ExtractedImage",
    "ExtractionResult",
    "IngestConfig",
    # Core
    "FileDetector",
    "CharsetHandler",
    "ExtractorRegistry",
    "create_default_registry",
    "Router",
]
