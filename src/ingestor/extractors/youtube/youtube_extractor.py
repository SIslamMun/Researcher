"""YouTube extractor using yt-dlp and youtube-transcript-api."""

import re
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import parse_qs, urlparse

from ...types import ExtractionResult, MediaType
from ..base import BaseExtractor


class YouTubeExtractor(BaseExtractor):
    """Extract transcripts and metadata from YouTube videos.

    Uses:
    - yt-dlp for video metadata
    - youtube-transcript-api for transcripts
    """

    media_type = MediaType.YOUTUBE

    def __init__(
        self,
        caption_type: str = "auto",
        include_playlist: bool = False,
        languages: Optional[List[str]] = None,
    ):
        """Initialize YouTube extractor.

        Args:
            caption_type: Caption preference (auto, manual)
            include_playlist: Process entire playlist if URL is playlist
            languages: Preferred languages for transcripts
        """
        self.caption_type = caption_type
        self.include_playlist = include_playlist
        self.languages = languages or ["en"]

    async def extract(self, source: Union[str, Path]) -> ExtractionResult:
        """Extract content from a YouTube video.

        Args:
            source: YouTube URL or video ID

        Returns:
            Extraction result with transcript and metadata
        """
        url = str(source)

        # Extract video ID
        video_id = self._extract_video_id(url)
        if not video_id:
            return ExtractionResult(
                markdown=f"# Error\n\nInvalid YouTube URL: {url}",
                title="Error",
                source=url,
                media_type=MediaType.YOUTUBE,
                images=[],
                metadata={"error": "Invalid URL"},
            )

        # Get metadata
        metadata = await self._get_metadata(video_id)

        # Get transcript
        transcript = await self._get_transcript(video_id)

        # Build markdown
        markdown = self._build_markdown(metadata, transcript)

        return ExtractionResult(
            markdown=markdown,
            title=metadata.get("title", video_id),
            source=url,
            media_type=MediaType.YOUTUBE,
            images=[],
            metadata=metadata,
        )

    async def extract_playlist(self, source: Union[str, Path]) -> List[ExtractionResult]:
        """Extract content from all videos in a playlist.

        Args:
            source: YouTube playlist URL

        Returns:
            List of extraction results for each video
        """
        import yt_dlp

        url = str(source)
        results = []

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if not info:
                return results

            entries = info.get("entries", [])
            for entry in entries:
                if entry and entry.get("id"):
                    video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                    result = await self.extract(video_url)
                    results.append(result)

        return results

    async def _get_metadata(self, video_id: str) -> dict:
        """Get video metadata using yt-dlp.

        Args:
            video_id: YouTube video ID

        Returns:
            Metadata dictionary
        """
        import yt_dlp

        url = f"https://www.youtube.com/watch?v={video_id}"

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                if not info:
                    return {"video_id": video_id}

                return {
                    "video_id": video_id,
                    "title": info.get("title", ""),
                    "description": info.get("description", ""),
                    "channel": info.get("channel", info.get("uploader", "")),
                    "channel_id": info.get("channel_id", ""),
                    "upload_date": info.get("upload_date", ""),
                    "duration": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "like_count": info.get("like_count", 0),
                    "tags": info.get("tags", []),
                    "categories": info.get("categories", []),
                    "thumbnail": info.get("thumbnail", ""),
                }
        except Exception as e:
            return {"video_id": video_id, "error": str(e)}

    async def _get_transcript(self, video_id: str) -> str:
        """Get video transcript.

        Args:
            video_id: YouTube video ID

        Returns:
            Transcript text
        """
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to get transcript based on preference
            transcript = None

            if self.caption_type == "manual":
                # Prefer manually created
                try:
                    transcript = transcript_list.find_manually_created_transcript(self.languages)
                except NoTranscriptFound:
                    pass

            if transcript is None:
                # Try auto-generated
                try:
                    transcript = transcript_list.find_generated_transcript(self.languages)
                except NoTranscriptFound:
                    pass

            if transcript is None:
                # Fall back to any available
                try:
                    transcript = transcript_list.find_transcript(self.languages)
                except NoTranscriptFound:
                    # Try any language
                    for t in transcript_list:
                        transcript = t
                        break

            if transcript:
                # Fetch the transcript
                entries = transcript.fetch()
                # Combine all text
                return " ".join(entry.get("text", "") for entry in entries)

            return ""

        except (TranscriptsDisabled, VideoUnavailable):
            return ""
        except Exception:
            return ""

    def _build_markdown(self, metadata: dict, transcript: str) -> str:
        """Build markdown from metadata and transcript.

        Args:
            metadata: Video metadata
            transcript: Transcript text

        Returns:
            Markdown content
        """
        lines = []

        # Title
        title = metadata.get("title", "YouTube Video")
        lines.append(f"# {title}")
        lines.append("")

        # Metadata
        if metadata.get("channel"):
            lines.append(f"**Channel:** {metadata['channel']}")
        if metadata.get("upload_date"):
            date = metadata["upload_date"]
            formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date
            lines.append(f"**Uploaded:** {formatted_date}")
        if metadata.get("duration"):
            duration = metadata["duration"]
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours:
                lines.append(f"**Duration:** {hours}h {minutes}m {seconds}s")
            else:
                lines.append(f"**Duration:** {minutes}m {seconds}s")
        if metadata.get("view_count"):
            lines.append(f"**Views:** {metadata['view_count']:,}")

        lines.append("")
        lines.append(f"**URL:** https://www.youtube.com/watch?v={metadata.get('video_id', '')}")
        lines.append("")

        # Description
        if metadata.get("description"):
            lines.append("## Description")
            lines.append("")
            lines.append(metadata["description"])
            lines.append("")

        # Transcript
        if transcript:
            lines.append("## Transcript")
            lines.append("")
            lines.append(transcript)
        else:
            lines.append("## Transcript")
            lines.append("")
            lines.append("*No transcript available for this video.*")

        return "\n".join(lines)

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL.

        Args:
            url: YouTube URL

        Returns:
            Video ID or None
        """
        # Handle various YouTube URL formats
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com/v/([a-zA-Z0-9_-]{11})",
            r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # Check if it's already just an ID
        if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
            return url

        # Try parsing as URL
        try:
            parsed = urlparse(url)
            if parsed.query:
                params = parse_qs(parsed.query)
                if "v" in params:
                    return params["v"][0]
        except Exception:
            pass

        return None

    def supports(self, source: Union[str, Path]) -> bool:
        """Check if this extractor handles the source.

        Args:
            source: Source to check

        Returns:
            True if this is a YouTube URL
        """
        source_str = str(source)

        # Check for YouTube URLs
        youtube_patterns = [
            r"youtube\.com/watch",
            r"youtu\.be/",
            r"youtube\.com/embed/",
            r"youtube\.com/v/",
            r"youtube\.com/shorts/",
            r"youtube\.com/playlist",
        ]

        for pattern in youtube_patterns:
            if re.search(pattern, source_str):
                return True

        return False
