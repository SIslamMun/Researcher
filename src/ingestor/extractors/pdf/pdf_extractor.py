"""PDF extractor placeholder for Docling integration.

This is a placeholder that will be integrated with the existing
paper-to-md pipeline using Docling for PDF extraction.
"""

from pathlib import Path
from typing import Union

from ...types import ExtractionResult, MediaType
from ..base import BaseExtractor


class PdfExtractor(BaseExtractor):
    """Extract content from PDF files.

    NOTE: This is a placeholder. PDF extraction will be handled by
    integrating the existing paper-to-md pipeline using Docling.

    For now, this extractor returns a message indicating that
    PDF support requires the Docling integration.
    """

    media_type = MediaType.PDF

    async def extract(self, source: Union[str, Path]) -> ExtractionResult:
        """Extract content from a PDF file.

        Args:
            source: Path to the PDF file

        Returns:
            Extraction result (placeholder message)
        """
        path = Path(source)

        # Placeholder message
        markdown = f"""# PDF: {path.name}

> **Note:** PDF extraction requires the Docling integration.
>
> This file will be processed by the paper-to-md pipeline
> once it is integrated into the ingestor.

**Source:** {path}

To enable PDF extraction:
1. Ensure Docling is installed
2. Configure the paper-to-md pipeline
3. The integration will automatically process PDF files
"""

        return ExtractionResult(
            markdown=markdown,
            title=path.stem,
            source=str(path),
            media_type=MediaType.PDF,
            images=[],
            metadata={
                "status": "placeholder",
                "note": "PDF extraction requires Docling integration",
            },
        )

    def supports(self, source: Union[str, Path]) -> bool:
        """Check if this extractor handles the source.

        Args:
            source: Path to check

        Returns:
            True if this is a PDF file
        """
        return str(source).lower().endswith(".pdf")
