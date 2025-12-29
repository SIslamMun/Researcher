# Research-to-RAG Pipeline

Automated pipeline to research a topic, extract references, download papers, and convert everything to markdown for RAG.

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Research-to-RAG Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 1. Research  │───▶│ 2. Parse     │───▶│ 3. Retrieve  │      │
│  │    (Gemini)  │    │    Refs      │    │  & Verify    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  research_report.md   references.json    papers/*.pdf          │
│                       - DOIs             bibtex/*.bib          │
│                       - arXiv IDs        verified/*.bib        │
│                       - GitHub repos            │               │
│                       - URLs                    │               │
│                              │                  │               │
│                              ▼                  ▼               │
│                        ┌──────────────────────────┐            │
│                        │      4. Ingest           │            │
│                        │   Convert to Markdown    │            │
│                        └──────────────────────────┘            │
│                                    │                            │
│                                    ▼                            │
│                          4_markdown/                            │
│                          ├── papers/                            │
│                          ├── github/                            │
│                          ├── web/                               │
│                          └── research/                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

1. **Install dependencies:**
   ```bash
   uv sync --all-extras
   ```

2. **Set up API key** (required for Step 1 - Deep Research):
   
   Option A: Environment variable
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```
   
   Option B: Create `.env` file in project root
   ```bash
   # .env
   GEMINI_API_KEY=your-gemini-api-key
   ```

## Usage

### Full Pipeline

```bash
# Research a topic and process all references
uv run python scripts/pipeline_cli.py "Machine learning for drug discovery"

# With custom output directory
uv run python scripts/pipeline_cli.py "Quantum computing" -o ./my_output

# Limit downloads to save time
uv run python scripts/pipeline_cli.py "Topic" --max-papers 5 --max-repos 3 --max-urls 5
```

### Skip Research (Use Existing File)

```bash
# Use an existing research report (skips Step 1)
uv run python scripts/pipeline_cli.py "ML" --research-file path/to/research.md
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `topic` | (required) | Research topic |
| `-o, --output` | `pipeline_output` | Output directory |
| `--research-file` | None | Skip research, use existing file |
| `--max-papers` | 10 | Max papers to download |
| `--max-repos` | 5 | Max GitHub repos to clone |
| `--max-urls` | 10 | Max URLs to ingest |

## Output Structure

```
pipeline_output/
├── 1_research/              # Step 1: Deep Research
│   └── research/
│       └── research_report.md   # AI-generated research report
│
├── 2_references/            # Step 2: Parsed References
│   ├── references.json      # Structured reference data
│   └── references.md        # Human-readable summary
│
├── 3_papers/                # Step 3: Downloaded Papers
│   ├── paper1.pdf
│   └── paper2.pdf
│
├── 3_bibtex/                # Step 3: BibTeX Citations
│   ├── doi_10.1234_xxx.bib  # Individual BibTeX files
│   ├── arXiv_2301.12345.bib
│   ├── combined.bib         # All citations merged
│   └── verified/            # BibTeX verification results
│       ├── verified.bib     # Valid BibTeX entries
│       ├── failed.bib       # Failed entries (if any)
│       └── report.md        # Verification report
│
├── 4_markdown/              # Step 4: RAG-Ready Markdown
│   ├── papers/              # Converted PDFs
│   ├── github/              # Cloned repos
│   ├── web/                 # Crawled URLs
│   └── research/            # Research report
│
└── PIPELINE_REPORT.md       # Summary report
```

## Pipeline Steps

### Step 1: Deep Research

Uses Google Gemini to conduct comprehensive research on the topic.

- **Input:** Topic string
- **Output:** `1_research/research/research_report.md`
- **Requires:** `GEMINI_API_KEY`

### Step 2: Parse References

Extracts references from the research report using regex patterns:

- **DOIs:** `10.xxxx/...` patterns
- **arXiv:** `arXiv:YYMM.NNNNN` or `YYMM.NNNNN` patterns
- **GitHub:** `github.com/owner/repo` patterns
- **URLs:** General HTTP/HTTPS links

### Step 3: Retrieve & DOI2BIB

Downloads papers and generates BibTeX citations using CLI commands:

- `uv run parser retrieve` - Fetches PDFs from arXiv, Unpaywall, PMC, etc.
- `uv run parser doi2bib` - Generates BibTeX using CrossRef, Semantic Scholar
- `uv run parser verify` - Validates and verifies BibTeX entries
- Creates `combined.bib` with all citations
- Creates `verified/` directory with verification results

### Step 4: Ingest to Markdown

Converts all content to markdown using `ingestor` CLI:

- `uv run ingestor ingest` - PDFs → Markdown with extracted text, figures, tables
- `uv run ingestor clone` - GitHub repos → Markdown with code files
- `uv run ingestor ingest` - URLs → Markdown with web content
- Research report → Copied to output

## Example

```bash
$ uv run python scripts/pipeline.py "Transformer architectures in NLP" -o ./nlp_research

╔═══════════════════════════════════════════════════════════════════╗
║              Research-to-RAG Pipeline                             ║
╠═══════════════════════════════════════════════════════════════════╣
║  Topic: Transformer architectures in NLP                          ║
║  Output: nlp_research                                             ║
╚═══════════════════════════════════════════════════════════════════╝

════════════════════════════════════════════════════════════════
STEP 1: DEEP RESEARCH
════════════════════════════════════════════════════════════════
▶ Researching: Transformer architectures in NLP...
✅ Research report saved: nlp_research/1_research/research_report.md

════════════════════════════════════════════════════════════════
STEP 2: PARSE REFERENCES
════════════════════════════════════════════════════════════════
  📄 DOIs found: 12
  📄 arXiv papers: 8
  📂 GitHub repos: 5
  🔗 Other URLs: 15
✅ References saved: nlp_research/2_references/references.json

════════════════════════════════════════════════════════════════
STEP 3: RETRIEVE PAPERS & DOI2BIB
════════════════════════════════════════════════════════════════
  Processing 10 papers...
  📖 10.xxxx/nature12373
     ✓ BibTeX generated
     ✓ PDF downloaded
...
✅ Papers downloaded: 8/10
✅ BibTeX entries: 10
✅ BibTeX verified: 10/10

════════════════════════════════════════════════════════════════
STEP 4: INGEST TO MARKDOWN
════════════════════════════════════════════════════════════════
✅ Ingestion complete:
   PDFs: 8
   GitHub: 5
   URLs: 10
   Research: 1
   Total markdown files: 24

╔═══════════════════════════════════════════════════════════════════╗
║                    ✅ Pipeline Complete!                          ║
╚═══════════════════════════════════════════════════════════════════╝
```

## CLI Commands Reference

All commands require `uv run` prefix to use the correct Python environment:

```bash
# Research
uv run researcher research "topic" -o output/

# Paper retrieval
uv run parser retrieve -d DOI -o papers/
uv run parser doi2bib arXiv:XXXX.XXXXX
uv run parser verify combined.bib -o verified/

# Ingestion
uv run ingestor ingest file.pdf -o output/
uv run ingestor clone https://github.com/owner/repo -o output/
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes (for Step 1) | Google Gemini API key |
| `GOOGLE_API_KEY` | Alternative | Alternative name for Gemini API key |
| `INGESTOR_EMAIL` | Recommended | Email for CrossRef, Unpaywall, OpenAlex |
| `S2_API_KEY` | Optional | Semantic Scholar API key (higher rate limits) |

## Tips

1. **Skip research for faster iteration:** If you already have a research document, use `--research-file` to skip the AI research step.

2. **Limit downloads for testing:** Use `--max-papers 3 --max-repos 2` when testing to save time.

3. **Check references first:** After Step 2, review `2_references/references.md` to see what will be downloaded.

4. **Use the combined BibTeX:** `3_bibtex/combined.bib` contains all citations ready for LaTeX.

5. **Verify BibTeX entries:** Check `3_bibtex/verified/report.md` for verification details.
