# Researcher

> AI-powered deep research agent using Google Gemini  
> Automated multi-step research with comprehensive knowledge synthesis

**CLI Tool:**
- `researcher` - Conduct deep research on any topic using Google's Gemini Deep Research Agent

Produces detailed, cited reports with references to academic papers, code repositories, and other resources.

## Features

- **Autonomous Research**: Multi-step research tasks executed autonomously
- **Comprehensive Reports**: Detailed markdown reports with structured analysis
- **Citation Extraction**: Automatic extraction of arXiv IDs, DOIs, GitHub URLs
- **Streaming Progress**: Real-time progress updates with thinking steps
- **Follow-up Support**: Ask follow-up questions about completed research
- **Configurable Output**: Custom formatting instructions for reports

## Quick Start

```bash
# Install
uv sync

# Set up API key
export GOOGLE_API_KEY="your-google-api-key"

# Conduct research
researcher research "What are the latest advances in quantum computing?"
```

## Commands

| Command | Description |
|---------|-------------|
| `researcher research "<query>"` | Conduct deep research on a topic |
| `researcher research "<query>" -o ./output` | Specify output directory |
| `researcher research "<query>" -v` | Verbose mode with thinking steps |
| `researcher research "<query>" --format "..."` | Custom output format instructions |
| `researcher research "<query>" --no-stream` | Use polling instead of streaming |
| `researcher research "<query>" --max-wait 7200` | Set max wait time (seconds) |
| `researcher --help` | Show help |
| `researcher --version` | Show version |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd researcher

# Install dependencies
uv sync

# For development
uv sync --extra dev
```

## Configuration

### API Key Setup

You can set your Google API key in four ways (in order of priority):

**1. CLI option**:
```bash
researcher research "query" --api-key "your-google-api-key"
```

**2. Environment variable**:
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

**3. `.env` file** (auto-loaded):
```bash
# .env
GOOGLE_API_KEY=your-google-api-key
```

**4. Config file**:
```yaml
# configs/research.yaml
api_key: "your-google-api-key"
```

You can get a Google API key from the [Google AI Studio](https://aistudio.google.com()).

## Usage

### Basic Research

```bash
# Simple research query
researcher research "What is RAG (Retrieval Augmented Generation)?"

# With custom output directory
researcher research "Compare transformer architectures" -o ./my_research

# With verbose output (shows thinking steps)
researcher research "Survey of LLM agents" -v
```

### Advanced Options

```bash
# Custom output format instructions
researcher research "Renewable energy trends" --format "Include comparison table and bullet points"

# Disable streaming (use polling instead)
researcher research "Climate change solutions" --no-stream

# Set maximum wait time (default: 3600 seconds)
researcher research "AI safety research" --max-wait 7200

# Verbose mode with real-time thinking steps
researcher research "Quantum machine learning" -v -o ./quantum_research
```

### Command Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output directory (default: ./output) |
| `--format` | Custom output format instructions |
| `--no-stream` | Disable streaming, use polling |
| `--max-wait` | Maximum wait time in seconds (default: 3600) |
| `-v, --verbose` | Verbose output with thinking steps |

## Output Structure

```
output/research/
├── research_report.md       # Main research report
├── research_metadata.json   # Query, citations, timing info
└── thinking_steps.md        # Agent reasoning (if --verbose)
```

### Report Format

The research report includes:
- Comprehensive analysis of the topic
- Structured sections with findings
- References section with:
  - Academic papers (with arXiv IDs or DOIs)
  - Code repositories (GitHub URLs)
  - Websites & documentation
  - YouTube videos
  - Other resources

### Example Output

```markdown
# Research Report: Quantum Computing Advances

## Executive Summary
...

## Key Findings
...

## Technical Analysis
...

## References

### Papers
- [arXiv:2301.12345] Title of Paper 1
- [DOI:10.1038/nature12373] Title of Paper 2

### Code Repositories
- https://github.com/owner/repo - Description

### Websites & Documentation
- https://example.com/docs - Description
```

## Programmatic Usage

```python
import asyncio
from researcher.deep_research import DeepResearcher, ResearchConfig

async def main():
    # Configure research
    config = ResearchConfig(
        output_format="Include comparison tables",
        max_wait_time=3600,
        enable_streaming=True,
        enable_thinking=True,
    )
    
    # Create researcher
    researcher = DeepResearcher(config=config)
    
    # Progress callback
    def on_progress(text: str):
        if text.startswith("[Thinking]"):
            print(f"Thinking: {text}")
        else:
            print(text, end="")
    
    # Conduct research
    result = await researcher.research(
        "What are the latest advances in quantum computing?",
        on_progress=on_progress
    )
    
    if result.succeeded:
        print(f"\nResearch completed in {result.duration_seconds:.1f}s")
        print(f"Report length: {len(result.report)} characters")
        
        # Save results
        result.save("./output/research")
    else:
        print(f"Research failed: {result.error}")

asyncio.run(main())
```

### Convenience Function

```python
from researcher.deep_research import deep_research

result = await deep_research(
    "Key trends in AI safety research",
    output_format="Format as an executive summary with bullet points."
)
print(result.report)
```

## API Reference

### ResearchConfig

Configuration for deep research tasks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_format` | `str \| None` | `None` | Custom formatting instructions |
| `max_wait_time` | `int` | `3600` | Maximum wait time in seconds |
| `poll_interval` | `int` | `10` | Interval between status checks |
| `enable_streaming` | `bool` | `True` | Enable streaming progress |
| `enable_thinking` | `bool` | `True` | Show agent's thinking process |
| `include_identifiers` | `bool` | `True` | Request arXiv IDs, DOIs in output |

### ResearchResult

Result of a deep research task.

| Property | Type | Description |
|----------|------|-------------|
| `query` | `str` | Original research query |
| `report` | `str` | Generated research report (markdown) |
| `status` | `ResearchStatus` | Final status |
| `interaction_id` | `str \| None` | Gemini Interaction ID |
| `citations` | `list[dict]` | List of cited sources |
| `thinking_steps` | `list[str]` | Agent's reasoning steps |
| `duration_seconds` | `float` | Time taken |
| `error` | `str \| None` | Error message if failed |
| `succeeded` | `bool` | Whether research completed successfully |

### Methods

#### `DeepResearcher.research(query, on_progress=None)`

Conduct a deep research task.

- `query`: The research topic or question
- `on_progress`: Optional callback for progress updates
- Returns: `ResearchResult`

#### `DeepResearcher.follow_up(question, interaction_id)`

Ask a follow-up question about completed research.

- `question`: Follow-up question
- `interaction_id`: ID of the completed research interaction
- Returns: Response text

#### `ResearchResult.save(output_path)`

Save research result to files.

- `output_path`: Directory to save results

## Architecture

### How It Works

```
User Query
    │
    ▼
┌─────────────────┐
│  DeepResearcher │ ← Configures research parameters
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gemini Agent   │ ← deep-research-pro-preview-12-2025
│  Interactions   │
│     API         │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────────┐
│ Web   │ │ Knowledge │
│Search │ │   Base    │
└───────┘ └───────────┘
         │
         ▼
┌─────────────────┐
│   Synthesize    │ ← Multi-step reasoning
│   & Report      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ResearchResult  │ ← Report + Citations + Metadata
└─────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `DeepResearcher` | Main class for conducting research |
| `ResearchConfig` | Configuration dataclass |
| `ResearchResult` | Result dataclass with report and metadata |
| `deep_research()` | Convenience async function |

### Gemini Integration

The researcher uses Google's Gemini Deep Research Agent via the Interactions API:

- **Agent**: `deep-research-pro-preview-12-2025`
- **Background execution**: Long-running tasks with async completion
- **Streaming**: Real-time progress updates with thinking summaries
- **Reconnection**: Automatic stream reconnection handling

## Development

### Setup

```bash
git clone <repo>
cd researcher
uv sync --extra dev
```

### Run Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=researcher

# Specific test file
uv run pytest tests/unit/test_deep_research.py -v
```

### Code Quality

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Type check
uv run mypy src/
```

## Requirements

- Python 3.11+
- Google API key with Gemini access
- `google-genai` package

## License

MIT
