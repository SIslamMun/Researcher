"""Word document extractor using docx2python and mammoth."""

from pathlib import Path
from typing import List, Union

from markdownify import markdownify

from ...types import ExtractionResult, ExtractedImage, MediaType
from ..base import BaseExtractor


class DocxExtractor(BaseExtractor):
    """Extract content and images from Word documents.

    Uses:
    - docx2python for structured extraction and images
    - mammoth for HTML conversion
    - markdownify for HTML to Markdown
    """

    media_type = MediaType.DOCX

    async def extract(self, source: Union[str, Path]) -> ExtractionResult:
        """Extract content from a DOCX file.

        Args:
            source: Path to the DOCX file

        Returns:
            Extraction result with markdown and images
        """
        from docx2python import docx2python
        import mammoth

        path = Path(source)

        # Use docx2python for structured extraction and images
        doc = docx2python(str(path))

        # Extract images (always)
        images = self._extract_images(doc)

        # Use mammoth for HTML conversion (better formatting)
        with open(path, "rb") as f:
            result = mammoth.convert_to_html(f)
            html = result.value

        # Convert HTML to Markdown
        markdown = markdownify(html, heading_style="ATX", strip=["script", "style"])

        # Clean up markdown
        markdown = self._clean_markdown(markdown)

        # Get title from first heading or filename
        title = self._extract_title(markdown) or path.stem

        return ExtractionResult(
            markdown=markdown,
            title=title,
            source=str(path),
            media_type=MediaType.DOCX,
            images=images,
            metadata={
                "image_count": len(images),
                "tables_count": len(doc.tables) if hasattr(doc, "tables") else 0,
            },
        )

    def _extract_images(self, doc) -> List[ExtractedImage]:
        """Extract images from the document.

        Args:
            doc: docx2python document object

        Returns:
            List of extracted images
        """
        images = []

        if hasattr(doc, "images") and doc.images:
            for name, data in doc.images.items():
                # Determine format from filename
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else "png"
                if ext == "jpg":
                    ext = "jpeg"

                images.append(ExtractedImage(
                    filename=name,
                    data=data,
                    format=ext,
                ))

        return images

    def _extract_title(self, markdown: str) -> str:
        """Extract title from first heading.

        Args:
            markdown: Markdown content

        Returns:
            Title or empty string
        """
        for line in markdown.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return ""

    def _clean_markdown(self, markdown: str) -> str:
        """Clean up markdown output.

        Args:
            markdown: Raw markdown

        Returns:
            Cleaned markdown
        """
        lines = []
        prev_blank = False

        for line in markdown.split("\n"):
            # Remove excessive blank lines
            is_blank = not line.strip()
            if is_blank:
                if not prev_blank:
                    lines.append("")
                prev_blank = True
            else:
                lines.append(line.rstrip())
                prev_blank = False

        return "\n".join(lines).strip()

    def supports(self, source: Union[str, Path]) -> bool:
        """Check if this extractor handles the source.

        Args:
            source: Path to check

        Returns:
            True if this is a DOCX file
        """
        return str(source).lower().endswith((".docx", ".doc"))
