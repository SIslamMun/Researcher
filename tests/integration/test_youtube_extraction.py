"""Integration tests for YouTube extraction."""

import pytest
from pathlib import Path

from ingestor.types import MediaType


class TestYouTubeExtraction:
    """Integration tests for YouTube video extraction."""

    @pytest.fixture
    def extractor(self):
        try:
            from ingestor.extractors.youtube import YouTubeExtractor
            return YouTubeExtractor()
        except ImportError:
            pytest.skip("yt-dlp or youtube-transcript-api not installed")

    def test_supports_youtube_urls(self, extractor):
        """Test supports various YouTube URL formats."""
        assert extractor.supports("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert extractor.supports("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert extractor.supports("https://youtu.be/dQw4w9WgXcQ")
        assert extractor.supports("https://youtube.com/embed/dQw4w9WgXcQ")
        assert extractor.supports("https://youtube.com/shorts/dQw4w9WgXcQ")

    def test_does_not_support_other_urls(self, extractor):
        """Test does not support non-YouTube URLs."""
        assert not extractor.supports("https://example.com")
        assert not extractor.supports("https://vimeo.com/123")

    def test_extract_video_id(self, extractor):
        """Test video ID extraction from various URL formats."""
        assert extractor._extract_video_id("https://youtube.com/watch?v=abc123def45") == "abc123def45"
        assert extractor._extract_video_id("https://youtu.be/abc123def45") == "abc123def45"
        assert extractor._extract_video_id("https://youtube.com/embed/abc123def45") == "abc123def45"
        assert extractor._extract_video_id("abc123def45") == "abc123def45"

    def test_extract_video_id_invalid(self, extractor):
        """Test invalid URL returns None."""
        assert extractor._extract_video_id("https://example.com") is None
        assert extractor._extract_video_id("not a url") is None

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_extract_youtube_video(self, extractor):
        """Test extracting a YouTube video (requires network and yt-dlp)."""
        try:
            import yt_dlp
        except ImportError:
            pytest.skip("yt-dlp not installed (install youtube extra)")

        # Use a well-known, stable video
        url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - first YouTube video

        result = await extractor.extract(url)

        assert result.media_type == MediaType.YOUTUBE
        assert result.markdown
        # Should have video structure
        assert "Transcript" in result.markdown or "Channel" in result.markdown

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_youtube_metadata(self, extractor):
        """Test YouTube metadata extraction."""
        try:
            import yt_dlp
        except ImportError:
            pytest.skip("yt-dlp not installed (install youtube extra)")

        url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

        result = await extractor.extract(url)

        assert "video_id" in result.metadata
        # May have title, channel, etc. if yt-dlp works
        assert result.title or result.metadata.get("title")


class TestYouTubeConfig:
    """Tests for YouTube extractor configuration."""

    def test_caption_type_config(self):
        """Test caption type configuration."""
        try:
            from ingestor.extractors.youtube import YouTubeExtractor
        except ImportError:
            pytest.skip("youtube dependencies not installed")

        extractor = YouTubeExtractor(caption_type="manual")
        assert extractor.caption_type == "manual"

    def test_language_config(self):
        """Test language preference configuration."""
        try:
            from ingestor.extractors.youtube import YouTubeExtractor
        except ImportError:
            pytest.skip("youtube dependencies not installed")

        extractor = YouTubeExtractor(languages=["es", "en"])
        assert extractor.languages == ["es", "en"]
