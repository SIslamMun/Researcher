"""EPUB ebook extractor using ebooklib."""

from pathlib import Path
from typing import List, Union

from markdownify import markdownify

from ...types import ExtractionResult, ExtractedImage, MediaType
from ..base import BaseExtractor


class EpubExtractor(BaseExtractor):
    """Extract content and images from EPUB ebooks.

    Uses ebooklib for EPUB2/EPUB3 support.
    """

    media_type = MediaType.EPUB

    async def extract(self, source: Union[str, Path]) -> ExtractionResult:
        """Extract content from an EPUB file.

        Args:
            source: Path to the EPUB file

        Returns:
            Extraction result with markdown and images
        """
        from ebooklib import epub, ITEM_DOCUMENT, ITEM_IMAGE

        path = Path(source)
        book = epub.read_epub(str(path))

        chapters = []
        images: List[ExtractedImage] = []

        # Extract metadata
        title = None
        title_meta = book.get_metadata("DC", "title")
        if title_meta:
            title = title_meta[0][0]

        author = None
        author_meta = book.get_metadata("DC", "creator")
        if author_meta:
            author = author_meta[0][0]

        # Extract chapters/documents
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            try:
                content = item.get_content().decode("utf-8")
                # Convert HTML to markdown
                md = markdownify(content, heading_style="ATX", strip=["script", "style"])
                md = self._clean_markdown(md)
                if md.strip():
                    chapters.append(md)
            except Exception:
                pass

        # Extract images (always)
        for item in book.get_items_of_type(ITEM_IMAGE):
            try:
                filename = item.get_name().split("/")[-1]
                media_type = item.media_type  # e.g., "image/jpeg"
                ext = media_type.split("/")[-1] if "/" in media_type else "png"
                if ext == "jpg":
                    ext = "jpeg"

                images.append(ExtractedImage(
                    filename=filename,
                    data=item.get_content(),
                    format=ext,
                ))
            except Exception:
                pass

        # Build markdown with chapter separators
        markdown_parts = []
        if title:
            markdown_parts.append(f"# {title}")
            if author:
                markdown_parts.append(f"\n**Author:** {author}\n")
            markdown_parts.append("")

        markdown_parts.append("\n\n---\n\n".join(chapters))

        return ExtractionResult(
            markdown="\n".join(markdown_parts),
            title=title or path.stem,
            source=str(path),
            media_type=MediaType.EPUB,
            images=images,
            metadata={
                "author": author,
                "chapter_count": len(chapters),
                "image_count": len(images),
            },
        )

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
            True if this is an EPUB file
        """
        return str(source).lower().endswith(".epub")
