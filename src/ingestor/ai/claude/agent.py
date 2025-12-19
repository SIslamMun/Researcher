"""Claude agent for markdown cleanup and enhancement."""

from pathlib import Path
from typing import Optional

from ...types import ExtractionResult


class ClaudeAgent:
    """Claude agent for cleaning up and enhancing extracted content.

    Uses the Claude Code SDK to process markdown output with
    intelligent cleanup, formatting, and restructuring.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a markdown content editor. Your task is to clean up and enhance extracted markdown content while preserving all important information.

Guidelines:
1. Fix formatting issues (headers, lists, tables)
2. Remove extraction artifacts and noise
3. Improve readability and structure
4. Preserve all meaningful content
5. Maintain image references exactly as they are
6. Keep the document's original meaning and intent
7. Do not add new information or commentary"""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
    ):
        """Initialize Claude agent.

        Args:
            system_prompt: Custom system prompt for the agent
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._sdk = None

    def _get_sdk(self):
        """Get or create Claude SDK client."""
        if self._sdk is None:
            try:
                from claude_code_sdk import ClaudeCode
                self._sdk = ClaudeCode()
            except ImportError:
                raise ImportError(
                    "Claude Code SDK not installed. "
                    "Install with: pip install claude-code-sdk"
                )
        return self._sdk

    async def cleanup(
        self,
        content: str,
        context: Optional[str] = None,
    ) -> str:
        """Clean up markdown content using Claude.

        Args:
            content: Markdown content to clean
            context: Optional context about the content source

        Returns:
            Cleaned markdown content
        """
        sdk = self._get_sdk()

        prompt = "Clean up the following extracted markdown content:\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"```markdown\n{content}\n```\n\n"
        prompt += "Return only the cleaned markdown, no explanations."

        response = await sdk.query(
            prompt=prompt,
            system=self.system_prompt,
        )

        # Extract markdown from response
        cleaned = response.strip()

        # Remove markdown code block if present
        if cleaned.startswith("```markdown"):
            cleaned = cleaned[11:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()

    async def cleanup_result(
        self,
        result: ExtractionResult,
    ) -> ExtractionResult:
        """Clean up an extraction result.

        Args:
            result: Extraction result to clean

        Returns:
            Result with cleaned markdown
        """
        context = f"Source: {result.source}, Type: {result.media_type.value if result.media_type else 'unknown'}"

        cleaned_markdown = await self.cleanup(
            result.markdown,
            context=context,
        )

        return ExtractionResult(
            markdown=cleaned_markdown,
            title=result.title,
            source=result.source,
            media_type=result.media_type,
            images=result.images,
            metadata=result.metadata,
            charset=result.charset,
        )

    async def summarize(
        self,
        content: str,
        max_length: Optional[int] = None,
    ) -> str:
        """Generate a summary of the content.

        Args:
            content: Content to summarize
            max_length: Maximum summary length in words

        Returns:
            Summary text
        """
        sdk = self._get_sdk()

        prompt = "Summarize the following content"
        if max_length:
            prompt += f" in approximately {max_length} words"
        prompt += f":\n\n{content}"

        response = await sdk.query(
            prompt=prompt,
            system="You are a concise summarizer. Provide clear, accurate summaries.",
        )

        return response.strip()

    async def extract_key_points(
        self,
        content: str,
        max_points: int = 10,
    ) -> str:
        """Extract key points from content.

        Args:
            content: Content to analyze
            max_points: Maximum number of points

        Returns:
            Markdown list of key points
        """
        sdk = self._get_sdk()

        prompt = (
            f"Extract the {max_points} most important key points from "
            f"the following content. Format as a markdown bulleted list:\n\n{content}"
        )

        response = await sdk.query(
            prompt=prompt,
            system="You extract key information concisely and accurately.",
        )

        return response.strip()


async def cleanup_markdown(
    content: str,
    context: Optional[str] = None,
) -> str:
    """Convenience function to clean up markdown content.

    Args:
        content: Content to clean
        context: Optional context

    Returns:
        Cleaned content
    """
    agent = ClaudeAgent()
    return await agent.cleanup(content, context)
