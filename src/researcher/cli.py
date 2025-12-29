"""CLI for the Researcher module - Deep Research using Gemini API."""

import asyncio
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


def load_config(config_path: Path | None = None) -> dict:
    """Load research config from YAML file (searches default locations if not specified)."""
    import yaml
    
    # Default search paths
    search_paths = [
        Path("./research.yaml"),
        Path("./configs/research.yaml"),
        Path.home() / ".config" / "researcher" / "research.yaml",
    ]
    
    if config_path:
        search_paths.insert(0, Path(config_path))
    
    for path in search_paths:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
    
    return {}


@click.group()
@click.version_option(version="0.1.0", prog_name="researcher")
def cli():
    """Researcher CLI - AI-powered deep research using Gemini.

    API key: Set via --api-key, GOOGLE_API_KEY env var, or configs/research.yaml
    Prompts: Customize in configs/prompts.yaml (citation format, follow-up instructions)
    """
    pass


@cli.command()
@click.argument("query", type=str)
@click.option("-o", "--output", type=click.Path(), default="./output", help="Output directory")
@click.option("--format", "output_format", type=str, default=None, help="Output format instructions")
@click.option("--no-stream", is_flag=True, help="Disable streaming (use polling)")
@click.option("--max-wait", type=int, default=3600, help="Max wait time in seconds (default: 3600)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output with thinking steps")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None, help="Path to config file")
@click.option("--api-key", type=str, default=None, help="Google API key (overrides config/env)")
def research(query: str, output: str, output_format: str | None,
             no_stream: bool, max_wait: int, verbose: bool,
             config_path: str | None, api_key: str | None):
    """Conduct deep research on a topic using Gemini Deep Research Agent.

    Searches multiple sources, synthesizes findings, and produces a comprehensive
    report with structured citations (arXiv IDs, DOIs, GitHub URLs).

    Output files (saved to <output>/research/):
      - research_report.md: Main report with citations
      - research_metadata.json: Query, timing, interaction info
      - thinking_steps.md: Agent reasoning (if --verbose)

    Customize citation format in configs/prompts.yaml

    Examples:
      researcher research "Latest advances in quantum computing"
      researcher research "Compare transformers" --format "Include tables" -v
    """
    from .deep_research import DeepResearcher, ResearchConfig

    # Load config file
    config_data = load_config(Path(config_path) if config_path else None)
    
    # Resolve API key: CLI > env > config file
    resolved_api_key = api_key or os.environ.get("GOOGLE_API_KEY") or config_data.get("api_key")
    
    if not resolved_api_key:
        click.echo(click.style("Error: No API key found", fg="red"), err=True)
        click.echo("Set via: --api-key, GOOGLE_API_KEY env, or configs/research.yaml", err=True)
        click.echo("Get key from: https://aistudio.google.com/", err=True)
        sys.exit(1)

    async def run():
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Configure research
        config = ResearchConfig(
            output_format=output_format,
            max_wait_time=max_wait,
            enable_streaming=not no_stream,
            enable_thinking=verbose,
        )

        researcher = DeepResearcher(api_key=resolved_api_key, config=config)

        # Stream file for live output
        stream_file = output_path / "research_stream.md" if verbose else None
        if stream_file:
            stream_file.write_text("# Research Stream Log\n\n")  # Initialize with header

        # Progress callback
        def on_progress(text: str):
            if verbose:
                # Write to stream file
                if stream_file:
                    with open(stream_file, "a") as f:
                        f.write(text + "\n\n")

                # Also print to console
                if text.startswith("[Thinking]"):
                    click.echo(click.style(text, dim=True))
                else:
                    click.echo(text, nl=False)

        # Show query (truncated if too long)
        query_display = query if len(query) <= 80 else f"{query[:77]}..."
        click.echo(f"🔍 Researching: {query_display}")
        click.echo(f"📁 Output: {output_path / 'research'}")
        click.echo()

        try:
            result = await researcher.research(query, on_progress=on_progress if verbose else None)
        except Exception as e:
            click.echo(click.style(f"Error: {e}", fg="red"), err=True)
            sys.exit(1)

        if result.succeeded:
            # Save results
            result.save(output_path / "research")

            click.echo()
            click.echo(click.style("✓ Research completed!", fg="green", bold=True))
            click.echo(f"Duration: {result.duration_seconds:.1f}s")
            click.echo(f"Report: {output_path / 'research' / 'research_report.md'}")

            # Preview in verbose mode
            if verbose and result.report:
                click.echo()
                preview = result.report[:300] + "..." if len(result.report) > 300 else result.report
                click.echo(click.style("Preview:", dim=True))
                click.echo(preview)
        else:
            click.echo()
            click.echo(click.style(f"✗ Research failed: {result.error}", fg="red"), err=True)
            sys.exit(1)

    asyncio.run(run())


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
