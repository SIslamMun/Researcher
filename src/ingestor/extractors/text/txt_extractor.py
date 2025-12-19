"""Plain text file extractor with charset detection."""

from pathlib import Path
from typing import Union

from ...core.charset import CharsetHandler
from ...types import ExtractionResult, MediaType
from ..base import BaseExtractor


class TxtExtractor(BaseExtractor):
    """Extract content from plain text files.

    Uses charset_normalizer for automatic encoding detection,
    supporting non-UTF8 files in various languages.
    """

    media_type = MediaType.TXT

    # Extensions we handle
    EXTENSIONS = {".txt", ".md", ".rst", ".text", ".log"}

    def __init__(self):
        """Initialize the extractor."""
        self.charset_handler = CharsetHandler()

    async def extract(self, source: Union[str, Path]) -> ExtractionResult:
        """Extract text content from a file.

        Args:
            source: Path to the text file

        Returns:
            Extraction result with markdown content
        """
        path = Path(source)

        # Read with charset detection
        text, encoding = self.charset_handler.read_text(path)

        # For markdown files, return as-is
        # For other text files, wrap in code block if it looks like code
        if path.suffix.lower() in {".md", ".rst"}:
            markdown = text
        else:
            markdown = text

        return ExtractionResult(
            markdown=markdown,
            title=path.stem,
            source=str(path),
            media_type=MediaType.TXT,
            charset=encoding,
            metadata={
                "encoding": encoding,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
            },
        )

    def supports(self, source: Union[str, Path]) -> bool:
        """Check if this extractor can handle the source.

        Args:
            source: Path to check

        Returns:
            True if this is a text file we can handle
        """
        path = Path(source)
        return path.suffix.lower() in self.EXTENSIONS or path.suffix == ""
