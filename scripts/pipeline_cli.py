#!/usr/bin/env python3
"""
Research-to-RAG Pipeline (CLI Version)

Uses direct CLI commands: researcher, parser, ingestor

Workflow:
1. Deep Research → Generate research report on topic
2. Parse References → Extract links, git repos, papers from research
3. Retrieve & Convert → Download papers, generate BibTeX
4. Ingest → Convert everything to markdown for RAG
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Load .env file if it exists
def load_env():
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())
    
    # Researcher uses GOOGLE_API_KEY, so copy GEMINI_API_KEY if set
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

load_env()


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERNS FOR EXTRACTING REFERENCES
# ═══════════════════════════════════════════════════════════════════════════════

DOI_PATTERN = re.compile(r'10\.\d{4,}/[^\s\])"\'<>,]+')
ARXIV_PATTERN = re.compile(r'(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)', re.IGNORECASE)
GITHUB_PATTERN = re.compile(r'github\.com/([^/\s]+/[^/\s\])"\'<>]+)')
URL_PATTERN = re.compile(r'https?://[^\s\])"\'<>]+')


def run_cmd(cmd: list[str], desc: str, timeout: int = 300, stream: bool = False) -> tuple[bool, str]:
    """Run a shell command with live output streaming."""
    print(f"\n{'─'*60}")
    print(f"▶ {desc}")
    print(f"  $ {' '.join(cmd)}")
    print('─'*60)
    
    try:
        if stream:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            output_lines = []
            for line in process.stdout:
                print(f"  {line}", end='')
                output_lines.append(line)
            process.wait(timeout=timeout)
            return process.returncode == 0, ''.join(output_lines)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[:30]:
                    print(f"  {line}")
                if len(lines) > 30:
                    print(f"  ... ({len(lines) - 30} more lines)")
            if result.returncode != 0 and result.stderr:
                print(f"  ⚠ {result.stderr[:500]}")
            return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Timeout after {timeout}s")
        if stream:
            process.kill()
        return False, ""
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DEEP RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def step1_research(topic: str, output_dir: Path) -> Path | None:
    """Run deep research on a topic using CLI."""
    print("\n" + "═"*60)
    print("STEP 1: DEEP RESEARCH")
    print("═"*60)
    
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("⚠ GEMINI_API_KEY not set. Set with: export GEMINI_API_KEY='your-key'")
        print("  Or add to .env file: GEMINI_API_KEY=your-key")
        return None
    
    research_dir = output_dir / "1_research"
    research_dir.mkdir(parents=True, exist_ok=True)
    
    # Use CLI command with uv run
    cmd = ["uv", "run", "researcher", "research", topic, "-o", str(research_dir)]
    
    success, _ = run_cmd(cmd, f"Researching: {topic[:50]}...", timeout=900, stream=True)
    
    # Check multiple possible locations for the report
    possible_paths = [
        research_dir / "research_report.md",
        research_dir / "research" / "research_report.md",
    ]
    
    for report_file in possible_paths:
        if report_file.exists():
            print(f"✅ Research report saved: {report_file}")
            return report_file
    
    # Check for any .md file recursively
    for f in research_dir.rglob("*.md"):
        if f.name != "research_stream.md":  # Skip stream log
            print(f"✅ Research report saved: {f}")
            return f
    
    print("❌ No research report generated")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: PARSE REFERENCES
# ═══════════════════════════════════════════════════════════════════════════════

def step2_parse_refs(research_file: Path, output_dir: Path) -> dict:
    """Extract all references from research document."""
    print("\n" + "═"*60)
    print("STEP 2: PARSE REFERENCES")
    print("═"*60)
    
    refs_dir = output_dir / "2_references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    
    text = research_file.read_text()
    
    # Extract different types of references
    refs = {
        "dois": [],
        "arxiv": [],
        "github": [],
        "urls": [],
    }
    
    # DOIs
    for match in DOI_PATTERN.finditer(text):
        doi = match.group(0).rstrip('.,;:)]')
        if doi not in refs["dois"]:
            refs["dois"].append(doi)
    
    # arXiv IDs
    for match in ARXIV_PATTERN.finditer(text):
        arxiv_id = match.group(1)
        if arxiv_id not in refs["arxiv"]:
            refs["arxiv"].append(arxiv_id)
    
    # GitHub repos
    for match in GITHUB_PATTERN.finditer(text):
        repo = match.group(1).rstrip('.,;:)]/')
        parts = repo.split('/')
        if len(parts) >= 2:
            repo = f"{parts[0]}/{parts[1]}"
        if repo not in refs["github"]:
            refs["github"].append(repo)
    
    # URLs (excluding already captured)
    for match in URL_PATTERN.finditer(text):
        url = match.group(0).rstrip('.,;:)]')
        if 'doi.org' in url or 'arxiv.org' in url or 'github.com' in url:
            continue
        if url not in refs["urls"]:
            refs["urls"].append(url)
    
    # Save references
    refs_file = refs_dir / "references.json"
    with open(refs_file, 'w') as f:
        json.dump(refs, f, indent=2)
    
    # Generate markdown summary
    md_content = f"""# Extracted References

**Source:** {research_file.name}
**Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## DOIs ({len(refs['dois'])})

{chr(10).join(f'- `{d}`' for d in refs['dois']) or '_None found_'}

## arXiv Papers ({len(refs['arxiv'])})

{chr(10).join(f'- `arXiv:{a}`' for a in refs['arxiv']) or '_None found_'}

## GitHub Repositories ({len(refs['github'])})

{chr(10).join(f'- [{r}](https://github.com/{r})' for r in refs['github']) or '_None found_'}

## Other URLs ({len(refs['urls'])})

{chr(10).join(f'- <{u}>' for u in refs['urls'][:20]) or '_None found_'}
{'...' if len(refs['urls']) > 20 else ''}
"""
    
    md_file = refs_dir / "references.md"
    md_file.write_text(md_content)
    
    print(f"  📄 DOIs found: {len(refs['dois'])}")
    print(f"  📄 arXiv papers: {len(refs['arxiv'])}")
    print(f"  📂 GitHub repos: {len(refs['github'])}")
    print(f"  🔗 Other URLs: {len(refs['urls'])}")
    print(f"✅ References saved: {refs_file}")
    
    return refs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: RETRIEVE PAPERS & DOI2BIB
# ═══════════════════════════════════════════════════════════════════════════════

def step3_retrieve(refs: dict, output_dir: Path, max_papers: int = 10) -> dict:
    """Download papers and generate BibTeX using CLI."""
    print("\n" + "═"*60)
    print("STEP 3: RETRIEVE PAPERS & DOI2BIB")
    print("═"*60)
    
    papers_dir = output_dir / "3_papers"
    bibtex_dir = output_dir / "3_bibtex"
    papers_dir.mkdir(parents=True, exist_ok=True)
    bibtex_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    bibtex_entries = []
    
    # Combine DOIs and arXiv IDs
    identifiers = []
    for doi in refs.get("dois", []):
        identifiers.append(("doi", doi))
    for arxiv in refs.get("arxiv", []):
        identifiers.append(("arxiv", f"arXiv:{arxiv}"))
    
    # Limit papers
    identifiers = identifiers[:max_papers]
    
    print(f"  Processing {len(identifiers)} papers...")
    
    for id_type, identifier in identifiers:
        safe_name = identifier.replace("/", "_").replace(":", "_")[:50]
        print(f"\n  📖 {identifier}")
        
        # Generate BibTeX using parser doi2bib CLI
        bib_file = bibtex_dir / f"{safe_name}.bib"
        try:
            result = subprocess.run(
                ["uv", "run", "parser", "doi2bib", identifier],
                capture_output=True, text=True, timeout=60
            )
            if result.stdout.strip() and result.stdout.strip().startswith('@'):
                bib_file.write_text(result.stdout.strip())
                bibtex_entries.append(str(bib_file))
                print(f"     ✓ BibTeX generated")
            else:
                print(f"     ⚠ BibTeX failed")
        except Exception as e:
            print(f"     ⚠ BibTeX error: {e}")
        
        # Download paper using parser retrieve CLI
        try:
            if id_type == "doi":
                cmd = ["uv", "run", "parser", "retrieve", "-d", identifier, "-o", str(papers_dir)]
            else:
                cmd = ["uv", "run", "parser", "retrieve", identifier, "-o", str(papers_dir)]
            
            pdfs_before = set(papers_dir.glob("*.pdf"))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            pdfs_after = set(papers_dir.glob("*.pdf"))
            new_pdfs = pdfs_after - pdfs_before
            
            if new_pdfs:
                downloaded.append(identifier)
                print(f"     ✓ PDF downloaded")
            else:
                print(f"     ⚠ PDF not available")
        except Exception as e:
            print(f"     ⚠ Download error: {e}")
    
    # Combine all BibTeX
    if bibtex_entries:
        combined_bib = bibtex_dir / "combined.bib"
        all_bib = []
        for bib_file in bibtex_dir.glob("*.bib"):
            if bib_file.name not in ["combined.bib", "verified.bib", "failed.bib"]:
                content = bib_file.read_text().strip()
                if content:
                    all_bib.append(content)
        if all_bib:
            combined_bib.write_text("\n\n".join(all_bib))
            print(f"\n✅ Combined BibTeX: {combined_bib}")
        
        # Verify BibTeX using parser verify CLI
        print(f"\n  🔍 Verifying BibTeX citations...")
        try:
            verify_out = bibtex_dir / "verified"
            result = subprocess.run(
                ["uv", "run", "parser", "verify", str(combined_bib), "-o", str(verify_out)],
                capture_output=True, text=True, timeout=120
            )
            if (verify_out / "verified.bib").exists():
                print(f"     ✓ Verified BibTeX: {verify_out / 'verified.bib'}")
            if (verify_out / "failed.bib").exists() and (verify_out / "failed.bib").stat().st_size > 0:
                print(f"     ⚠ Failed entries: {verify_out / 'failed.bib'}")
            if (verify_out / "report.md").exists():
                print(f"     📄 Verification report: {verify_out / 'report.md'}")
        except Exception as e:
            print(f"     ⚠ Verification error: {e}")
    
    print(f"✅ Papers downloaded: {len(downloaded)}/{len(identifiers)}")
    print(f"✅ BibTeX entries: {len(bibtex_entries)}")
    
    return {"downloaded": downloaded, "bibtex": bibtex_entries}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: INGEST TO MARKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def step4_ingest(refs: dict, output_dir: Path, research_file: Path | None, max_repos: int = 5, max_urls: int = 10) -> dict:
    """Convert everything to markdown using ingestor CLI."""
    print("\n" + "═"*60)
    print("STEP 4: INGEST TO MARKDOWN")
    print("═"*60)
    
    ingest_dir = output_dir / "4_markdown"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    
    ingested = {"pdfs": 0, "github": 0, "urls": 0, "research": 0}
    
    # 4a. Ingest downloaded PDFs
    papers_dir = output_dir / "3_papers"
    if papers_dir.exists():
        pdfs = list(papers_dir.glob("*.pdf"))
        if pdfs:
            print(f"\n  📄 Ingesting {len(pdfs)} PDFs...")
            pdf_out = ingest_dir / "papers"
            for pdf in pdfs:
                cmd = ["uv", "run", "ingestor", "ingest", str(pdf), "-o", str(pdf_out)]
                success, _ = run_cmd(cmd, f"Converting: {pdf.name}", timeout=120)
                if success:
                    ingested["pdfs"] += 1
    
    # 4b. Ingest GitHub repos
    github_repos = refs.get("github", [])[:max_repos]
    if github_repos:
        print(f"\n  📂 Ingesting {len(github_repos)} GitHub repos...")
        github_out = ingest_dir / "github"
        for repo in github_repos:
            repo_url = f"https://github.com/{repo}"
            cmd = ["uv", "run", "ingestor", "clone", repo_url, "-o", str(github_out)]
            success, _ = run_cmd(cmd, f"Cloning: {repo}", timeout=180)
            if success:
                ingested["github"] += 1
    
    # 4c. Ingest URLs
    urls = refs.get("urls", [])[:max_urls]
    if urls:
        print(f"\n  🔗 Ingesting {len(urls)} URLs...")
        web_out = ingest_dir / "web"
        for url in urls:
            cmd = ["uv", "run", "ingestor", "ingest", url, "-o", str(web_out)]
            success, _ = run_cmd(cmd, f"Fetching: {url[:50]}...", timeout=60)
            if success:
                ingested["urls"] += 1
    
    # 4d. Copy research report
    if research_file and research_file.exists():
        print(f"\n  📝 Copying research report...")
        research_out = ingest_dir / "research"
        research_out.mkdir(parents=True, exist_ok=True)
        dest = research_out / research_file.name
        dest.write_text(research_file.read_text())
        ingested["research"] = 1
        print(f"  ✓ Copied to {dest}")
    
    # Count total markdown files
    total_md = len(list(ingest_dir.rglob("*.md")))
    
    print(f"\n✅ Ingestion complete:")
    print(f"   PDFs: {ingested['pdfs']}")
    print(f"   GitHub: {ingested['github']}")
    print(f"   URLs: {ingested['urls']}")
    print(f"   Research: {ingested['research']}")
    print(f"   Total markdown files: {total_md}")
    
    return ingested


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(topic: str, output_dir: Path, refs: dict, retrieve_stats: dict, ingest_stats: dict):
    """Generate final pipeline report."""
    report = f"""# Pipeline Report

**Topic:** {topic}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| 1. Research | Deep research on topic | `1_research/` |
| 2. Parse Refs | Extract DOIs, arXiv, GitHub, URLs | `2_references/` |
| 3. Retrieve | Download papers + BibTeX | `3_papers/`, `3_bibtex/` |
| 4. Ingest | Convert to markdown | `4_markdown/` |

## Extracted References

- **DOIs:** {len(refs.get('dois', []))}
- **arXiv papers:** {len(refs.get('arxiv', []))}
- **GitHub repos:** {len(refs.get('github', []))}
- **URLs:** {len(refs.get('urls', []))}

## Retrieved

- **Papers downloaded:** {len(retrieve_stats.get('downloaded', []))}
- **BibTeX entries:** {len(retrieve_stats.get('bibtex', []))}

## Ingested

- **PDFs converted:** {ingest_stats.get('pdfs', 0)}
- **GitHub repos:** {ingest_stats.get('github', 0)}
- **URLs:** {ingest_stats.get('urls', 0)}

## Output Structure

```
{output_dir}/
├── 1_research/          # Deep research output
│   └── research_report.md
├── 2_references/        # Extracted references
│   ├── references.json
│   └── references.md
├── 3_papers/            # Downloaded PDFs
├── 3_bibtex/            # BibTeX citations
│   └── combined.bib
└── 4_markdown/          # Final markdown for RAG
    ├── papers/
    ├── github/
    ├── web/
    └── research/
```

## Usage

All markdown files in `4_markdown/` are ready for RAG ingestion.
"""
    
    report_file = output_dir / "PIPELINE_REPORT.md"
    report_file.write_text(report)
    print(f"\n📄 Report: {report_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    topic: str,
    output_dir: Path,
    research_file: Path | None = None,
    max_papers: int = 10,
    max_repos: int = 5,
    max_urls: int = 10,
):
    """Run the full pipeline."""
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║              Research-to-RAG Pipeline (CLI)                       ║
╠═══════════════════════════════════════════════════════════════════╣
║  Topic: {topic[:55]:<55} ║
║  Output: {str(output_dir)[:55]:<55} ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Deep Research (or use provided file)
    if research_file and research_file.exists():
        print(f"\n📄 Using existing research file: {research_file}")
    else:
        research_file = step1_research(topic, output_dir)
    
    if not research_file or not research_file.exists():
        print("\n❌ No research file available. Exiting.")
        return
    
    # STEP 2: Parse References
    refs = step2_parse_refs(research_file, output_dir)
    
    # STEP 3: Retrieve Papers & DOI2BIB
    retrieve_stats = step3_retrieve(refs, output_dir, max_papers=max_papers)
    
    # STEP 4: Ingest to Markdown
    ingest_stats = step4_ingest(refs, output_dir, research_file, max_repos=max_repos, max_urls=max_urls)
    
    # Generate Report
    generate_report(topic, output_dir, refs, retrieve_stats, ingest_stats)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    ✅ Pipeline Complete!                          ║
╠═══════════════════════════════════════════════════════════════════╣
║  All markdown files ready in: {str(output_dir / '4_markdown'):<35} ║
╚═══════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Research-to-RAG Pipeline (CLI Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uses direct CLI commands: researcher, parser, ingestor

Workflow:
  1. Deep Research  → researcher research "topic"
  2. Parse Refs     → Extract DOIs, arXiv, GitHub, URLs
  3. Retrieve       → parser retrieve / parser doi2bib
  4. Ingest         → ingestor ingest / ingestor clone

Examples:
  # Full pipeline with new research
  python scripts/pipeline_cli.py "Machine learning for drug discovery"
  
  # Use existing research file
  python scripts/pipeline_cli.py "ML" --research-file path/to/report.md
  
  # Limit downloads
  python scripts/pipeline_cli.py "Topic" --max-papers 5 --max-repos 3
"""
    )
    
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("-o", "--output", type=Path, default=Path("pipeline_output"),
                        help="Output directory (default: pipeline_output)")
    parser.add_argument("--research-file", type=Path, default=None,
                        help="Use existing research file (skip Step 1)")
    parser.add_argument("--max-papers", type=int, default=10,
                        help="Max papers to download (default: 10)")
    parser.add_argument("--max-repos", type=int, default=5,
                        help="Max GitHub repos to clone (default: 5)")
    parser.add_argument("--max-urls", type=int, default=10,
                        help="Max URLs to ingest (default: 10)")
    
    args = parser.parse_args()
    
    run_pipeline(
        topic=args.topic,
        output_dir=args.output,
        research_file=args.research_file,
        max_papers=args.max_papers,
        max_repos=args.max_repos,
        max_urls=args.max_urls,
    )


if __name__ == "__main__":
    main()
