# Research-to-RAG Interactive Pipeline

A fully interactive pipeline that guides you through all configuration options before executing the 4-phase research-to-RAG workflow.

## Quick Start

```bash
cd /path/to/ingestor
source .env  # Load API keys
uv run python scripts/pipeline_interactive.py
```

## Features

- **Interactive Configuration**: Prompts for ALL options before running
- **Skip Any Phase**: Resume from any point by skipping completed phases
- **No Limits by Default**: Press Enter on limit prompts to process all items
- **Auto-Cleanup**: Removes existing folders before re-running (only for non-skipped phases)
- **Multiple Git Modes**: clone (full), api (README+metadata), readme (fastest)
- **Deep Web Crawling**: BFS/DFS/BestFirst strategies with configurable depth

## Pipeline Phases

### Phase 1: Deep Research (`researcher research`)
Uses Google Gemini to conduct comprehensive research on your topic.

**Options:**
- `Skip research`: Use existing research file
- `Verbose output`: Show thinking steps
- `Custom format`: Custom format instructions
- `Disable streaming`: Use polling instead
- `Max wait time`: Timeout in seconds (default: 3600)

**Output:** `{output}/1_research/research/research_report.md`

### Phase 2: Parse References (`parser parse-refs`)
Extracts all references from the research report.

**Extracts:**
- DOIs (`10.xxxx/...`)
- arXiv IDs (`arXiv:YYMM.NNNNN`)
- GitHub repos (`github.com/owner/repo`)
- YouTube URLs
- General URLs

**Options:**
- `Skip parsing`: Use existing references.json
- `Output format`: json, md, or both

**Output:** `{output}/2_references/references.json`

### Phase 3: Paper Retrieval & BibTeX (`parser retrieve/doi2bib/verify`)
Downloads papers and generates BibTeX citations.

**Sources:** arXiv, Unpaywall, PMC, Semantic Scholar, OpenAlex, bioRxiv, CrossRef

**Options:**
- `Skip retrieval`: Skip this entire phase
- `Maximum papers`: Enter number or press Enter for unlimited
- `Email`: For API access (CrossRef, Unpaywall)
- `Semantic Scholar API key`: Optional
- `Skip existing`: Don't re-download existing PDFs
- `BibTeX format`: bibtex, json, markdown
- `Verify BibTeX`: Cross-check against CrossRef/arXiv

**Output:** 
- `{output}/3_papers/*.pdf`
- `{output}/3_bibtex/*.bib`

### Phase 4: Ingest to Markdown (`ingestor ingest/clone/crawl`)
Converts everything to markdown for RAG ingestion.

**Supported formats:** PDF, Word, PowerPoint, EPUB, Excel, CSV, JSON, XML, Images, Audio, Web, YouTube, Git/GitHub

#### General Options:
- `Skip ingestion`: Skip this entire phase
- `Keep raw images`: Don't convert image formats
- `Image format`: png, jpg, webp
- `Generate metadata`: Create JSON metadata files
- `VLM describe`: Use Ollama VLM to describe images
- `Claude agent`: Use Claude for markdown cleanup

#### Git/GitHub Options:
- `Maximum repos`: Enter number or press Enter for unlimited
- `Extraction mode`:
  - `clone` - Full repository clone (all files)
  - `api` - GitHub API only (README + metadata, faster)
  - `readme` - README file only (fastest)
- Clone options: shallow/full, depth, branch, tag, commit, submodules, max-files, max-file-size, include-binary, token

#### Web/URL Options:
- `Maximum URLs`: Enter number or press Enter for unlimited
- `Deep crawling`: Enable multi-page crawling
  - `Strategy`: bfs (breadth-first), dfs (depth-first), bestfirst
  - `Max depth`: How deep to crawl (default: 2)
  - `Max pages`: Pages per site (default: 10)
  - `Include/Exclude patterns`: URL patterns to filter
  - `Domain restriction`: Stay on same domain

#### YouTube Options:
- `Maximum videos`: Enter number or press Enter for unlimited

#### Audio Options:
- `Maximum files`: Enter number or press Enter for unlimited
- `Whisper model`: tiny, base, small, medium, large, turbo

**Output:** `{output}/4_markdown/`

## Example Session

```
╔═══════════════════════════════════════════════════════════════════╗
║          Research-to-RAG Pipeline - Interactive Setup             ║
╚═══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────┐
│                    GENERAL SETTINGS                         │
└─────────────────────────────────────────────────────────────┘
Research topic: lightweight fine-tuned LLMs with reasoning
Output directory [pipeline_output]: my_research

┌─────────────────────────────────────────────────────────────┐
│         PHASE 1: DEEP RESEARCH                              │
└─────────────────────────────────────────────────────────────┘
Skip research (use existing file)? [y/N]: n
  Verbose output (show thinking steps)? [y/N]: y
  ...

┌─────────────────────────────────────────────────────────────┐
│          PHASE 2: PARSE REFERENCES                          │
└─────────────────────────────────────────────────────────────┘
Skip parsing (use existing references.json)? [y/N]: n
  Output format [json/md/both] (default: both): both

┌─────────────────────────────────────────────────────────────┐
│     PHASE 3: PAPER RETRIEVAL & BIBTEX                       │
└─────────────────────────────────────────────────────────────┘
Skip paper retrieval & BibTeX? [y/N]: n
  📄 Paper Retrieval:
    Maximum papers to download (Enter = no limit): 5
    ...

┌─────────────────────────────────────────────────────────────┐
│           PHASE 4: INGEST TO MARKDOWN                       │
└─────────────────────────────────────────────────────────────┘
Skip ingestion to markdown? [y/N]: n
  ...

═════════════════════════════════════════════════════════════════
                    CONFIGURATION SUMMARY
═════════════════════════════════════════════════════════════════
  ...

Proceed with this configuration? [Y/n]: y
```

## Resuming a Pipeline

To resume from a specific phase, skip the completed phases:

```
# Skip Phase 1 (research already done)
Skip research (use existing file)? [y/N]: y
Path to existing research file: my_research/1_research/research/research_report.md

# Skip Phase 2 (references already parsed)
Skip parsing (use existing references.json)? [y/N]: y
Path to existing references.json: my_research/2_references/references.json

# Run Phase 3
Skip paper retrieval & BibTeX? [y/N]: n
...
```

## Output Structure

```
{output}/
├── 1_research/
│   └── research/
│       └── research_report.md
├── 2_references/
│   ├── references.json
│   └── references.md
├── 3_papers/
│   └── *.pdf
├── 3_bibtex/
│   └── *.bib
├── 4_markdown/
│   ├── papers/
│   ├── github/
│   ├── youtube/
│   ├── web/
│   └── research/
└── PIPELINE_REPORT.md
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required for Phase 1 (Research)
GEMINI_API_KEY=your-gemini-api-key
# or
GOOGLE_API_KEY=your-google-api-key

# Optional for Phase 3 (Paper Retrieval)
SEMANTIC_SCHOLAR_API_KEY=your-s2-key
```

## Tips

1. **First Run**: Don't skip any phases to see the full workflow
2. **Re-run Ingestion**: Skip phases 1-3, only run phase 4 with different settings
3. **Test with Limits**: Set small limits (e.g., 2 papers, 1 repo) for testing
4. **Full Processing**: Press Enter on all limit prompts for unlimited processing
5. **Git Mode Selection**:
   - Use `readme` for quick overview
   - Use `api` for README + metadata without cloning
   - Use `clone` for full repository analysis

## Comparison with CLI Version

| Feature | `pipeline_interactive.py` | `pipeline_cli.py` |
|---------|---------------------------|-------------------|
| Configuration | Interactive prompts | Command-line args |
| Skip phases | Yes (per-phase) | Limited |
| All options exposed | Yes | Subset |
| Best for | Exploration, first-time use | Automation, scripts |

## Troubleshooting

### "GEMINI_API_KEY not set"
```bash
export GEMINI_API_KEY='your-key'
# or add to .env file
```

### Phase fails but want to continue
Re-run the pipeline and skip the completed phases.

### Too many items being processed
Set limits when prompted (e.g., enter `5` instead of pressing Enter).

### Existing files causing conflicts
The pipeline automatically cleans folders before re-running each phase (skipped phases are not cleaned).
