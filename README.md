# Researcher

> AI-powered deep research agent using Google Gemini Deep Research API

Conduct autonomous multi-step research tasks and generate comprehensive, cited reports with references to academic papers, code repositories, and other resources.

## Features

- **Autonomous Research**: Multi-step research executed autonomously by Gemini
- **Structured Citations**: Automatic extraction of arXiv IDs, DOIs, GitHub URLs
- **Streaming Progress**: Real-time updates with agent thinking steps
- **Follow-up Questions**: Continue research with contextual follow-ups
- **Configurable Prompts**: All prompts externalized to `configs/prompts.yaml`

## Quick Start

```bash
# Install dependencies
uv sync

# Set API key (get one from https://aistudio.google.com/)
export GOOGLE_API_KEY="your-api-key"

# Run research
researcher research "What are the latest advances in quantum computing?"

# Output:
# 🔍 Researching: What are the latest advances in quantum computing?
# 📁 Output: ./output/research
#
# ✓ Research completed!
# Duration: 245.3s
# Report: ./output/research/research_report.md
```

## Installation

```bash
git clone <repo-url>
cd researcher
uv sync                    # Production
uv sync --extra dev        # Development
```

## Configuration

### API Key

Set your Google API key (priority order):
1. CLI: `--api-key "your-key"`
2. Environment: `export GOOGLE_API_KEY="your-key"`
3. File: `.env` or `configs/research.yaml`

### Prompts

**All prompts are externalized** to `configs/prompts.yaml` for easy customization:

**Available prompts:**
- `default_output_format` - Citation format and reference structure for main research
- `follow_up_system_prompt` - System instructions for follow-up questions

**To customize:**
```bash
# Edit prompts without touching code
nano configs/prompts.yaml

# Changes take effect immediately
researcher research "Your query"
```

Benefits:
- ✅ No code editing required
- ✅ A/B test different prompt formats
- ✅ Version control your prompts
- ✅ Single source of truth

## Usage

### CLI

```bash
# Basic research
researcher research "Your research query"

# With options
researcher research "Compare transformer architectures" \
  -o ./my_research \
  -v \
  --format "Include comparison tables"

# Common options:
#   -o, --output DIR     Output directory (default: ./output)
#   -v, --verbose        Show thinking steps and preview
#   --format TEXT        Custom format instructions
#   --no-stream          Use polling instead of streaming
#   --max-wait SECONDS   Max wait time (default: 3600)
#   --api-key KEY        Google API key

# Get help
researcher --help
researcher research --help
```

### Programmatic

```python
from researcher.deep_research import DeepResearcher, ResearchConfig

# Basic usage
researcher = DeepResearcher()
result = await researcher.research("What is quantum computing?")
print(result.report)

# With config
config = ResearchConfig(
    output_format="Include comparison tables",
    enable_streaming=True,
    enable_thinking=True
)
researcher = DeepResearcher(config=config)
result = await researcher.research("AI safety research")

# Save results
result.save("./output/research")

# Follow-up questions
answer = await researcher.follow_up(
    "Can you elaborate on error correction?",
    result.interaction_id
)
```

### Convenience Function

```python
from researcher.deep_research import deep_research

result = await deep_research(
    "Latest AI trends",
    output_format="Executive summary with bullet points"
)
```

## Output

### What You Get

When you run a research query, you'll see:

```bash
🔍 Researching: What are the latest advances in quantum computing?
📁 Output: ./output/research

# Research progress (if --verbose)...

✓ Research completed!
Duration: 245.3s
Report: ./output/research/research_report.md
```

### File Structure

```
output/research/
├── research_report.md       # Main report with structured citations
├── research_metadata.json   # Query, timing, interaction ID
└── thinking_steps.md        # Agent reasoning (only with -v flag)
```

### Report Contents

Reports include comprehensive analysis with **structured references** organized by type:

| Category | Identifiers Included |
|----------|---------------------|
| Published Papers | DOI, full citation, venue |
| Preprints | arXiv ID, bioRxiv, SSRN |
| Code Repositories | GitHub URLs, stars, description |
| Datasets | Hugging Face, Kaggle, format info |
| Websites & Docs | URLs, access dates |
| Books & Textbooks | ISBN, publisher, year |
| Videos | YouTube URLs, duration |

All formatted in **markdown tables** with **YAML metadata** for easy programmatic extraction.

## Development

```bash
# Run tests
uv run pytest
uv run pytest --cov=researcher

# Code quality
uv run ruff check src/
uv run ruff format src/
uv run mypy src/
```

## Requirements

- Python 3.11+
- Google API key with Gemini access
- Dependencies: `google-genai`, `click`, `rich`, `pyyaml`

## Architecture

Uses Google's Gemini Deep Research Agent (`deep-research-pro-preview-12-2025`) via the Interactions API with:
- Background execution for long-running tasks
- Streaming for real-time progress updates
- Automatic stream reconnection handling

## License

MIT
