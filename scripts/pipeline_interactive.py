#!/usr/bin/env python3
"""
Research-to-RAG Pipeline (Interactive Version)

Interactive pipeline that asks for ALL configuration options before running.

Workflow:
1. Deep Research → Generate research report on topic
2. Parse References → Extract links, git repos, papers from research
3. Retrieve & Convert → Download papers, generate BibTeX
4. Ingest → Convert everything to markdown for RAG
"""

import json
import os
import re
import shutil
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
# INTERACTIVE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineConfig:
    """Configuration for the pipeline."""
    
    def __init__(self):
        # General
        self.topic: str = ""
        self.output_dir: Path = Path("pipeline_output")
        self.research_file: Path | None = None
        
        # Phase 1: Research
        self.skip_research: bool = False
        self.research_verbose: bool = False
        self.research_format: str | None = None
        self.research_no_stream: bool = False
        self.research_max_wait: int = 3600
        
        # Phase 2: Parse References
        self.skip_parse: bool = False
        self.refs_file: Path | None = None  # Existing references.json
        self.parse_format: str = "both"  # json, md, both
        
        # Phase 3: Paper Retrieval
        self.skip_retrieval: bool = False
        self.max_papers: int | None = None  # None = no limit
        self.paper_email: str | None = None
        self.paper_s2_key: str | None = None
        self.paper_skip_existing: bool = True
        self.paper_verbose: bool = False
        
        # Phase 3: BibTeX
        self.verify_bibtex: bool = True
        self.bibtex_format: str = "bibtex"  # bibtex, json, markdown
        self.bibtex_skip_keys: str | None = None
        self.bibtex_verbose: bool = False
        
        # Phase 4: General Ingestion
        self.skip_ingest: bool = False
        self.keep_raw_images: bool = False
        self.img_format: str = "png"
        self.generate_metadata: bool = False
        self.use_vlm_describe: bool = False
        self.use_claude_agent: bool = False
        self.whisper_model: str = "turbo"
        self.ollama_host: str | None = None
        self.vlm_model: str | None = None
        
        # Phase 4: PDF
        self.max_pdfs: int = 100  # practically unlimited
        self.pdf_verbose: bool = False
        
        # Phase 4: Git/GitHub
        self.max_repos: int | None = None  # None = no limit
        self.git_mode: str = "clone"  # clone, api, readme
        self.git_shallow: bool = True
        self.git_depth: int | None = None
        self.git_branch: str | None = None
        self.git_tag: str | None = None
        self.git_commit: str | None = None
        self.git_submodules: bool = False
        self.git_max_files: int | None = None
        self.git_max_file_size: int | None = None
        self.git_include_binary: bool = False
        self.git_token: str | None = None
        self.git_verbose: bool = False
        
        # Phase 4: Web/URLs
        self.max_urls: int | None = None  # None = no limit
        self.web_verbose: bool = False
        
        # Phase 4: Web Crawl (for deeper crawling)
        self.use_deep_crawl: bool = False
        self.crawl_strategy: str = "bfs"  # bfs, dfs, bestfirst
        self.crawl_max_depth: int = 2
        self.crawl_max_pages: int = 10
        self.crawl_include: str | None = None
        self.crawl_exclude: str | None = None
        self.crawl_domain_only: bool = True
        
        # Phase 4: YouTube
        self.max_youtube: int | None = None  # None = no limit
        self.youtube_verbose: bool = False
        
        # Phase 4: Audio
        self.max_audio: int | None = None  # None = no limit
        self.audio_verbose: bool = False


def get_input(prompt: str, default: str = "") -> str:
    """Get input with default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input."""
    default_str = "Y/n" if default else "y/N"
    user_input = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not user_input:
        return default
    return user_input in ('y', 'yes', '1', 'true')


def get_choice(prompt: str, choices: list[str], default: str) -> str:
    """Get choice from list."""
    choices_str = "/".join(choices)
    user_input = input(f"{prompt} [{choices_str}] (default: {default}): ").strip().lower()
    if not user_input:
        return default
    if user_input in choices:
        return user_input
    print(f"  ⚠ Invalid choice. Using default: {default}")
    return default


def get_int(prompt: str, default: int, min_val: int = 0, max_val: int = 10000) -> int:
    """Get integer input with validation."""
    user_input = input(f"{prompt} [{default}]: ").strip()
    if not user_input:
        return default
    try:
        val = int(user_input)
        if val < min_val or val > max_val:
            print(f"  ⚠ Value must be between {min_val} and {max_val}. Using default: {default}")
            return default
        return val
    except ValueError:
        print(f"  ⚠ Invalid number. Using default: {default}")
        return default


def get_optional_str(prompt: str) -> str | None:
    """Get optional string input."""
    user_input = input(f"{prompt} (press Enter to skip): ").strip()
    return user_input if user_input else None


def get_optional_int(prompt: str) -> int | None:
    """Get optional integer input."""
    user_input = input(f"{prompt} (press Enter for no limit): ").strip()
    if not user_input:
        return None
    try:
        val = int(user_input)
        if val < 0:
            print("  ⚠ Value must be positive. Using no limit.")
            return None
        return val
    except ValueError:
        print("  ⚠ Invalid number. Using no limit.")
        return None


def get_limit(prompt: str) -> int | None:
    """Get limit input - Enter for no limit, or any positive number."""
    user_input = input(f"{prompt} (Enter = no limit): ").strip()
    if not user_input:
        return None
    try:
        val = int(user_input)
        if val < 0:
            print("  ⚠ Value must be positive. Using no limit.")
            return None
        return val if val > 0 else None
    except ValueError:
        print("  ⚠ Invalid number. Using no limit.")
        return None


def interactive_config() -> PipelineConfig:
    """Interactively collect pipeline configuration."""
    config = PipelineConfig()
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║          Research-to-RAG Pipeline - Interactive Setup             ║
║                                                                   ║
║  Configure all options for the 4-phase pipeline:                  ║
║  1. Deep Research (researcher)                                    ║
║  2. Parse References (parser parse-refs)                          ║
║  3. Paper Retrieval & BibTeX (parser retrieve/doi2bib/verify)     ║
║  4. Ingest to Markdown (ingestor ingest/clone/crawl)              ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # ─────────────────────────────────────────────────────────────────
    # GENERAL SETTINGS
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│                    GENERAL SETTINGS                         │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    config.topic = get_input("Research topic")
    while not config.topic:
        print("  ⚠ Topic is required!")
        config.topic = get_input("Research topic")
    
    config.output_dir = Path(get_input("Output directory", "pipeline_output"))
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 1: RESEARCH
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│         PHASE 1: DEEP RESEARCH (researcher research)        │")
    print("│                                                             │")
    print("│  Uses Google Gemini to conduct comprehensive research.      │")
    print("│  Output: research_report.md with citations                  │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    config.skip_research = get_yes_no("Skip research (use existing file)?", default=False)
    
    if config.skip_research:
        research_path = get_input("Path to existing research file")
        if research_path:
            config.research_file = Path(research_path)
            if not config.research_file.exists():
                print(f"  ⚠ File not found: {config.research_file}")
                config.skip_research = False
                config.research_file = None
    
    if not config.skip_research:
        config.research_verbose = get_yes_no("  Verbose output (show thinking steps)?", default=False)
        config.research_format = get_optional_str("  Custom format instructions")
        config.research_no_stream = get_yes_no("  Disable streaming (use polling)?", default=False)
        config.research_max_wait = get_int("  Max wait time in seconds", 3600, 60, 7200)
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 2: PARSE REFERENCES
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│          PHASE 2: PARSE REFERENCES (parser parse-refs)      │")
    print("│                                                             │")
    print("│  Extracts from research report:                             │")
    print("│  • DOIs (10.xxxx/...)                                       │")
    print("│  • arXiv IDs (arXiv:YYMM.NNNNN)                             │")
    print("│  • GitHub repos (github.com/owner/repo)                     │")
    print("│  • YouTube URLs                                             │")
    print("│  • General URLs                                             │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    config.skip_parse = get_yes_no("Skip parsing (use existing references.json)?", default=False)
    
    if config.skip_parse:
        refs_path = get_input("Path to existing references.json")
        if refs_path:
            config.refs_file = Path(refs_path)
            if not config.refs_file.exists():
                print(f"  ⚠ File not found: {config.refs_file}")
                config.skip_parse = False
                config.refs_file = None
    
    if not config.skip_parse:
        config.parse_format = get_choice("  Output format", ["json", "md", "both"], "both")
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 3: PAPER RETRIEVAL & BIBTEX
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│     PHASE 3: PAPER RETRIEVAL & BIBTEX (parser)              │")
    print("│                                                             │")
    print("│  Commands: parser retrieve, parser doi2bib, parser verify   │")
    print("│                                                             │")
    print("│  Sources: arXiv, Unpaywall, PMC, Semantic Scholar,          │")
    print("│           OpenAlex, bioRxiv, CrossRef                       │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    config.skip_retrieval = get_yes_no("Skip paper retrieval & BibTeX?", default=False)
    
    if not config.skip_retrieval:
        print("\n  📄 Paper Retrieval (parser retrieve):")
        config.max_papers = get_limit("    Maximum papers to download")
        
        if config.max_papers is None or config.max_papers > 0:
            config.paper_email = get_optional_str("    Email for API access (CrossRef, Unpaywall)")
            config.paper_s2_key = get_optional_str("    Semantic Scholar API key")
            config.paper_skip_existing = get_yes_no("    Skip if PDF already exists?", default=True)
            config.paper_verbose = get_yes_no("    Verbose output?", default=False)
        
        print("\n  📚 BibTeX Generation (parser doi2bib):")
        config.bibtex_format = get_choice("    Output format", ["bibtex", "json", "markdown"], "bibtex")
        
        print("\n  ✓ BibTeX Verification (parser verify):")
        config.verify_bibtex = get_yes_no("    Verify BibTeX against CrossRef/arXiv?", default=True)
        if config.verify_bibtex:
            config.bibtex_skip_keys = get_optional_str("    Citation keys to skip (comma-separated)")
            config.bibtex_verbose = get_yes_no("    Verbose verification?", default=False)
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 4: INGESTION
    # ─────────────────────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│           PHASE 4: INGEST TO MARKDOWN (ingestor)            │")
    print("│                                                             │")
    print("│  Commands: ingestor ingest, ingestor clone, ingestor crawl  │")
    print("│                                                             │")
    print("│  Supported: PDF, Word, PowerPoint, EPUB, Excel, CSV, JSON,  │")
    print("│             XML, Images, Audio, Web, YouTube, Git/GitHub    │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    config.skip_ingest = get_yes_no("Skip ingestion to markdown?", default=False)
    
    if not config.skip_ingest:
        # General ingestion options
        print("\n  ⚙️  General Ingestion Options:")
        config.keep_raw_images = get_yes_no("    Keep original image formats (don't convert)?", default=False)
        if not config.keep_raw_images:
            config.img_format = get_choice("    Target image format", ["png", "jpg", "webp"], "png")
        config.generate_metadata = get_yes_no("    Generate JSON metadata files?", default=False)
        config.use_vlm_describe = get_yes_no("    Use VLM to describe images (requires Ollama)?", default=False)
        if config.use_vlm_describe:
            config.ollama_host = get_optional_str("      Ollama server URL")
            config.vlm_model = get_optional_str("      VLM model name")
        config.use_claude_agent = get_yes_no("    Use Claude agent for cleanup (requires Claude Code)?", default=False)
        
        # PDF options
        print("\n  📄 PDF Ingestion (ingestor ingest *.pdf):")
        config.pdf_verbose = get_yes_no("    Verbose PDF output?", default=False)
        
        # GitHub/Git options
        print("\n  📂 GitHub/Git Repository Ingestion:")
        config.max_repos = get_limit("    Maximum repos to process")
        
        if config.max_repos is None or config.max_repos > 0:
            print("""
    Git extraction modes:
      • clone  - Full repository clone (all files)
      • api    - GitHub API only (README + metadata, faster)
      • readme - README file only (fastest)
        """)
            config.git_mode = get_choice("    Extraction mode", ["clone", "api", "readme"], "clone")
            
            if config.git_mode == "clone":
                config.git_shallow = get_yes_no("      Use shallow clone (faster)?", default=True)
                if config.git_shallow:
                    depth = get_optional_int("      Clone depth")
                    config.git_depth = depth
                
                config.git_branch = get_optional_str("      Specific branch")
                config.git_tag = get_optional_str("      Specific tag")
                config.git_commit = get_optional_str("      Specific commit")
                config.git_submodules = get_yes_no("      Include git submodules?", default=False)
                config.git_max_files = get_optional_int("      Max files per repo")
                config.git_max_file_size = get_optional_int("      Max file size in bytes")
                config.git_include_binary = get_yes_no("      Include binary file metadata?", default=False)
                config.git_token = get_optional_str("      Git token for private repos")
            
            config.git_verbose = get_yes_no("    Verbose Git output?", default=False)
        
        # Web/URL options
        print("\n  🔗 Web/URL Ingestion (ingestor ingest/crawl URL):")
        config.max_urls = get_limit("    Maximum URLs to process")
        
        if config.max_urls is None or config.max_urls > 0:
            config.web_verbose = get_yes_no("    Verbose web output?", default=False)
            
            print("\n    Deep crawl options (for websites with multiple pages):")
            config.use_deep_crawl = get_yes_no("    Enable deep crawling?", default=False)
            
            if config.use_deep_crawl:
                config.crawl_strategy = get_choice("      Crawl strategy", ["bfs", "dfs", "bestfirst"], "bfs")
                config.crawl_max_depth = get_int("      Maximum crawl depth", 2, 1, 10)
                config.crawl_max_pages = get_int("      Maximum pages per site", 10, 1, 100)
                config.crawl_include = get_optional_str("      URL patterns to include (e.g., /docs/*)")
                config.crawl_exclude = get_optional_str("      URL patterns to exclude (e.g., /api/*)")
                config.crawl_domain_only = get_yes_no("      Restrict to same domain?", default=True)
        
        # YouTube options
        print("\n  🎬 YouTube Ingestion (ingestor ingest YouTube-URL):")
        config.max_youtube = get_limit("    Maximum YouTube videos/playlists")
        if config.max_youtube is None or config.max_youtube > 0:
            config.youtube_verbose = get_yes_no("    Verbose YouTube output?", default=False)
        
        # Audio options
        print("\n  🎵 Audio Ingestion (ingestor ingest *.mp3/*.wav):")
        config.max_audio = get_limit("    Maximum audio files to transcribe")
        if config.max_audio is None or config.max_audio > 0:
            config.whisper_model = get_choice("    Whisper model", ["tiny", "base", "small", "medium", "large", "turbo"], "turbo")
            config.audio_verbose = get_yes_no("    Verbose audio output?", default=False)
    
    # ─────────────────────────────────────────────────────────────────
    # CONFIRM CONFIGURATION
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("                    CONFIGURATION SUMMARY")
    print("═"*65)
    print(f"""
  📋 General:
     Topic: {config.topic}
     Output: {config.output_dir}
  
  🔬 Phase 1 - Research:
     Skip: {config.skip_research}
     {"Using: " + str(config.research_file) if config.skip_research else ""}
     {"Verbose: " + str(config.research_verbose) if not config.skip_research else ""}
     {"Stream: " + str(not config.research_no_stream) if not config.skip_research else ""}
  
  📑 Phase 2 - Parse References:
     Skip: {config.skip_parse}
     {"Using: " + str(config.refs_file) if config.skip_parse else ""}
     {"Output format: " + config.parse_format if not config.skip_parse else ""}
  
  📚 Phase 3 - Paper Retrieval:
     Skip: {config.skip_retrieval}
     {"Max papers: " + (str(config.max_papers) if config.max_papers is not None else "unlimited") if not config.skip_retrieval else ""}
     {"Verify BibTeX: " + str(config.verify_bibtex) if not config.skip_retrieval else ""}
  
  📁 Phase 4 - Ingestion:
     Skip: {config.skip_ingest}
     {"" if config.skip_ingest else f'''
     Git/GitHub:
       Max repos: {config.max_repos if config.max_repos is not None else 'unlimited'}
       Mode: {config.git_mode}
     
     Web/URLs:
       Max URLs: {config.max_urls if config.max_urls is not None else 'unlimited'}
       Deep crawl: {config.use_deep_crawl}
     
     YouTube:
       Max videos: {config.max_youtube if config.max_youtube is not None else 'unlimited'}
     
     Audio:
       Max files: {config.max_audio if config.max_audio is not None else 'unlimited'}
       Whisper model: {config.whisper_model}
     
     Image options:
       Keep raw: {config.keep_raw_images}
       VLM describe: {config.use_vlm_describe}
       Claude agent: {config.use_claude_agent}
'''}""")
    
    if not get_yes_no("Proceed with this configuration?", default=True):
        print("\n❌ Pipeline cancelled.")
        sys.exit(0)
    
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERNS FOR EXTRACTING REFERENCES
# ═══════════════════════════════════════════════════════════════════════════════

DOI_PATTERN = re.compile(r'10\.\d{4,}/[^\s\])"\'<>,]+')
ARXIV_PATTERN = re.compile(r'(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)', re.IGNORECASE)
GITHUB_PATTERN = re.compile(r'github\.com/([^/\s]+/[^/\s\])"\'<>]+)')
YOUTUBE_PATTERN = re.compile(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)')
URL_PATTERN = re.compile(r'https?://[^\s\])"\'<>]+')


def run_cmd(cmd: list[str], desc: str, timeout: int = 300, stream: bool = False) -> tuple[bool, str]:
    """Run a shell command with live output streaming."""
    print(f"\n{'─'*60}")
    print(f"▶ {desc}")
    print(f"  $ {' '.join(cmd)}")
    print('─'*60)
    
    try:
        if stream:
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
# HELPER: CLEAN DIRECTORY
# ═══════════════════════════════════════════════════════════════════════════════

def clean_directory(dir_path: Path, desc: str = "") -> None:
    """Remove directory if it exists to avoid duplicates/conflicts."""
    if dir_path.exists():
        try:
            shutil.rmtree(dir_path)
            if desc:
                print(f"  🗑  Cleaned existing {desc}: {dir_path.name}")
        except Exception as e:
            print(f"  ⚠ Could not clean {dir_path}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DEEP RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def step1_research(config: PipelineConfig) -> Path | None:
    """Run deep research on a topic using CLI."""
    print("\n" + "═"*65)
    print("STEP 1: DEEP RESEARCH")
    print("═"*65)
    
    if config.skip_research and config.research_file:
        print(f"📄 Using existing research file: {config.research_file}")
        return config.research_file
    
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("⚠ GEMINI_API_KEY not set. Set with: export GEMINI_API_KEY='your-key'")
        return None
    
    research_dir = config.output_dir / "1_research"
    clean_directory(research_dir, "research folder")
    research_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ["uv", "run", "researcher", "research", config.topic, "-o", str(research_dir)]
    
    if config.research_verbose:
        cmd.append("-v")
    
    if config.research_format:
        cmd.extend(["--format", config.research_format])
    
    if config.research_no_stream:
        cmd.append("--no-stream")
    
    cmd.extend(["--max-wait", str(config.research_max_wait)])
    
    success, _ = run_cmd(cmd, f"Researching: {config.topic[:50]}...", 
                         timeout=config.research_max_wait + 60, stream=True)
    
    # Check multiple possible locations for the report
    possible_paths = [
        research_dir / "research_report.md",
        research_dir / "research" / "research_report.md",
    ]
    
    for report_file in possible_paths:
        if report_file.exists():
            print(f"✅ Research report saved: {report_file}")
            return report_file
    
    for f in research_dir.rglob("*.md"):
        if f.name != "research_stream.md":
            print(f"✅ Research report saved: {f}")
            return f
    
    print("❌ No research report generated")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: PARSE REFERENCES
# ═══════════════════════════════════════════════════════════════════════════════

def step2_parse_refs(research_file: Path, config: PipelineConfig) -> dict:
    """Extract all references from research document using parser CLI."""
    print("\n" + "═"*65)
    print("STEP 2: PARSE REFERENCES")
    print("═"*65)
    
    refs_dir = config.output_dir / "2_references"
    clean_directory(refs_dir, "references folder")
    refs_dir.mkdir(parents=True, exist_ok=True)
    
    # Use parser parse-refs CLI
    cmd = ["uv", "run", "parser", "parse-refs", str(research_file), 
           "-o", str(refs_dir), "--format", config.parse_format]
    
    success, output = run_cmd(cmd, f"Parsing references from: {research_file.name}", timeout=60)
    
    # Read the generated references.json
    refs_file = refs_dir / "references.json"
    refs = {"dois": [], "arxiv": [], "github": [], "youtube": [], "urls": []}
    
    if refs_file.exists():
        try:
            with open(refs_file) as f:
                raw_refs = json.load(f)
            
            # Convert from list format to dict format
            # parser parse-refs outputs: [{"type": "github", "value": "owner/repo", ...}, ...]
            if isinstance(raw_refs, list):
                for item in raw_refs:
                    ref_type = item.get("type", "")
                    value = item.get("value", "") or item.get("url", "")
                    
                    if ref_type == "doi" and value:
                        refs["dois"].append(value)
                    elif ref_type == "arxiv" and value:
                        refs["arxiv"].append(value)
                    elif ref_type == "github" and value:
                        refs["github"].append(value)
                    elif ref_type == "youtube" and value:
                        # Store full URL for youtube
                        url = item.get("url", f"https://youtube.com/watch?v={value}")
                        refs["youtube"].append(url)
                    elif ref_type == "website" and value:
                        url = item.get("url", value)
                        refs["urls"].append(url)
            elif isinstance(raw_refs, dict):
                # Already in expected format
                refs = raw_refs
        except json.JSONDecodeError:
            print("  ⚠ Could not parse references.json, using fallback extraction")
    
    # Fallback: also extract from research file directly to catch anything missed
    text = research_file.read_text()
    
    for match in DOI_PATTERN.finditer(text):
        doi = match.group(0).rstrip('.,;:)]')
        if doi not in refs.get("dois", []):
            refs.setdefault("dois", []).append(doi)
    
    for match in ARXIV_PATTERN.finditer(text):
        arxiv_id = match.group(1)
        if arxiv_id not in refs.get("arxiv", []):
            refs.setdefault("arxiv", []).append(arxiv_id)
    
    for match in GITHUB_PATTERN.finditer(text):
        repo = match.group(1).rstrip('.,;:)]/')
        parts = repo.split('/')
        if len(parts) >= 2:
            repo = f"{parts[0]}/{parts[1]}"
        if repo not in refs.get("github", []):
            refs.setdefault("github", []).append(repo)
    
    for match in YOUTUBE_PATTERN.finditer(text):
        video_id = match.group(1)
        url = f"https://youtube.com/watch?v={video_id}"
        if url not in refs.get("youtube", []):
            refs.setdefault("youtube", []).append(url)
    
    for match in URL_PATTERN.finditer(text):
        url = match.group(0).rstrip('.,;:)]')
        if 'doi.org' in url or 'arxiv.org' in url or 'github.com' in url or 'youtube.com' in url or 'youtu.be' in url:
            continue
        if url not in refs.get("urls", []):
            refs.setdefault("urls", []).append(url)
    
    # Save updated refs
    with open(refs_file, 'w') as f:
        json.dump(refs, f, indent=2)
    
    # Print summary
    print(f"\n  📄 DOIs found: {len(refs.get('dois', []))}")
    print(f"  📄 arXiv papers: {len(refs.get('arxiv', []))}")
    print(f"  📂 GitHub repos: {len(refs.get('github', []))}")
    print(f"  🎬 YouTube videos: {len(refs.get('youtube', []))}")
    print(f"  🔗 Other URLs: {len(refs.get('urls', []))}")
    print(f"\n✅ References saved: {refs_file}")
    
    return refs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: RETRIEVE PAPERS & DOI2BIB
# ═══════════════════════════════════════════════════════════════════════════════

def step3_retrieve(refs: dict, config: PipelineConfig) -> dict:
    """Download papers and generate BibTeX using CLI."""
    print("\n" + "═"*65)
    print("STEP 3: RETRIEVE PAPERS & DOI2BIB")
    print("═"*65)
    
    papers_dir = config.output_dir / "3_papers"
    bibtex_dir = config.output_dir / "3_bibtex"
    clean_directory(papers_dir, "papers folder")
    clean_directory(bibtex_dir, "bibtex folder")
    papers_dir.mkdir(parents=True, exist_ok=True)
    bibtex_dir.mkdir(parents=True, exist_ok=True)
    
    identifiers = []
    for doi in refs.get("dois", []):
        identifiers.append(("doi", doi))
    for arxiv in refs.get("arxiv", []):
        identifiers.append(("arxiv", arxiv))
    
    if config.max_papers is not None:
        identifiers = identifiers[:config.max_papers]
    
    if not identifiers:
        print("  No papers to retrieve.")
        return {"downloaded": [], "bibtex": []}
    
    print(f"\n  Processing {len(identifiers)} papers...")
    
    downloaded = []
    bibtex_entries = []
    
    for id_type, identifier in identifiers:
        print(f"\n  📖 {identifier}")
        
        # Generate BibTeX
        bib_file = bibtex_dir / f"{id_type}_{identifier.replace('/', '_').replace(':', '_')}.bib"
        bib_cmd = ["uv", "run", "parser", "doi2bib"]
        
        if id_type == "arxiv":
            bib_cmd.append(f"arXiv:{identifier}")
        else:
            bib_cmd.append(identifier)
        
        bib_cmd.extend(["-o", str(bib_file), "--format", config.bibtex_format])
        
        if config.paper_email:
            bib_cmd.extend(["--email", config.paper_email])
        if config.paper_s2_key:
            bib_cmd.extend(["--s2-key", config.paper_s2_key])
        
        try:
            result = subprocess.run(bib_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"     ✓ BibTeX generated")
                bibtex_entries.append(identifier)
            else:
                print(f"     ⚠ BibTeX failed")
        except Exception as e:
            print(f"     ⚠ BibTeX error: {e}")
        
        # Download paper
        retrieve_cmd = ["uv", "run", "parser", "retrieve"]
        
        if id_type == "arxiv":
            retrieve_cmd.append(f"arXiv:{identifier}")
        else:
            retrieve_cmd.extend(["-d", identifier])
        
        retrieve_cmd.extend(["-o", str(papers_dir)])
        
        if config.paper_email:
            retrieve_cmd.extend(["-e", config.paper_email])
        if config.paper_s2_key:
            retrieve_cmd.extend(["--s2-key", config.paper_s2_key])
        if not config.paper_skip_existing:
            retrieve_cmd.append("--no-skip-existing")
        if config.paper_verbose:
            retrieve_cmd.append("-v")
        
        try:
            result = subprocess.run(retrieve_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"     ✓ PDF downloaded")
                downloaded.append(identifier)
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
        
        # Verify BibTeX
        if config.verify_bibtex and combined_bib.exists():
            print(f"\n  🔍 Verifying BibTeX citations...")
            verify_cmd = ["uv", "run", "parser", "verify", str(combined_bib)]
            
            verify_out = bibtex_dir / "verified"
            verify_cmd.extend(["-o", str(verify_out)])
            
            if config.paper_email:
                verify_cmd.extend(["--email", config.paper_email])
            if config.bibtex_skip_keys:
                verify_cmd.extend(["--skip-keys", config.bibtex_skip_keys])
            if config.bibtex_verbose:
                verify_cmd.append("-v")
            
            try:
                result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=120)
                if (verify_out / "verified.bib").exists():
                    print(f"     ✓ Verified BibTeX: {verify_out / 'verified.bib'}")
                if (verify_out / "failed.bib").exists() and (verify_out / "failed.bib").stat().st_size > 0:
                    print(f"     ⚠ Failed entries: {verify_out / 'failed.bib'}")
                if (verify_out / "report.md").exists():
                    print(f"     📄 Verification report: {verify_out / 'report.md'}")
            except Exception as e:
                print(f"     ⚠ Verification error: {e}")
    
    print(f"\n✅ Papers downloaded: {len(downloaded)}/{len(identifiers)}")
    print(f"✅ BibTeX entries: {len(bibtex_entries)}")
    
    return {"downloaded": downloaded, "bibtex": bibtex_entries}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: INGEST TO MARKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def build_ingest_cmd(config: PipelineConfig, input_path: str, output_dir: Path, verbose: bool = False) -> list[str]:
    """Build ingestor ingest command with common options."""
    cmd = ["uv", "run", "ingestor", "ingest", input_path, "-o", str(output_dir)]
    
    if config.keep_raw_images:
        cmd.append("--keep-raw")
    else:
        cmd.extend(["--img-to", config.img_format])
    
    if config.generate_metadata:
        cmd.append("--metadata")
    
    if config.use_vlm_describe:
        cmd.append("--describe")
        if config.ollama_host:
            cmd.extend(["--ollama-host", config.ollama_host])
        if config.vlm_model:
            cmd.extend(["--vlm-model", config.vlm_model])
    
    if config.use_claude_agent:
        cmd.append("--agent")
    
    if verbose:
        cmd.append("-v")
    
    return cmd


def build_clone_cmd(config: PipelineConfig, repo_url: str, output_dir: Path) -> list[str]:
    """Build ingestor clone command with all options."""
    cmd = ["uv", "run", "ingestor", "clone", repo_url, "-o", str(output_dir)]
    
    if config.git_shallow:
        cmd.append("--shallow")
        if config.git_depth:
            cmd.extend(["--depth", str(config.git_depth)])
    else:
        cmd.append("--full")
    
    if config.git_branch:
        cmd.extend(["--branch", config.git_branch])
    
    if config.git_tag:
        cmd.extend(["--tag", config.git_tag])
    
    if config.git_commit:
        cmd.extend(["--commit", config.git_commit])
    
    if config.git_submodules:
        cmd.append("--submodules")
    
    if config.git_max_files:
        cmd.extend(["--max-files", str(config.git_max_files)])
    
    if config.git_max_file_size:
        cmd.extend(["--max-file-size", str(config.git_max_file_size)])
    
    if config.git_include_binary:
        cmd.append("--include-binary")
    
    if config.git_token:
        cmd.extend(["--token", config.git_token])
    
    if config.generate_metadata:
        cmd.append("--metadata")
    
    if config.git_verbose:
        cmd.append("-v")
    
    return cmd


def build_crawl_cmd(config: PipelineConfig, url: str, output_dir: Path) -> list[str]:
    """Build ingestor crawl command with all options."""
    cmd = ["uv", "run", "ingestor", "crawl", url, "-o", str(output_dir)]
    
    cmd.extend(["--strategy", config.crawl_strategy])
    cmd.extend(["--max-depth", str(config.crawl_max_depth)])
    cmd.extend(["--max-pages", str(config.crawl_max_pages)])
    
    if config.crawl_include:
        cmd.extend(["--include", config.crawl_include])
    
    if config.crawl_exclude:
        cmd.extend(["--exclude", config.crawl_exclude])
    
    if config.crawl_domain_only:
        # Extract domain from URL and restrict
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        if domain:
            cmd.extend(["--domain", domain])
    
    if config.generate_metadata:
        cmd.append("--metadata")
    
    if config.web_verbose:
        cmd.append("-v")
    
    return cmd


def step4_ingest(refs: dict, config: PipelineConfig, research_file: Path | None) -> dict:
    """Convert everything to markdown using ingestor CLI."""
    print("\n" + "═"*65)
    print("STEP 4: INGEST TO MARKDOWN")
    print("═"*65)
    
    ingest_dir = config.output_dir / "4_markdown"
    clean_directory(ingest_dir, "markdown folder")
    ingest_dir.mkdir(parents=True, exist_ok=True)
    
    ingested = {"pdfs": 0, "github": 0, "urls": 0, "youtube": 0, "research": 0}
    
    # 4a. Ingest downloaded PDFs
    papers_dir = config.output_dir / "3_papers"
    if papers_dir.exists():
        pdfs = list(papers_dir.glob("*.pdf"))
        if pdfs:
            print(f"\n  📄 Ingesting {len(pdfs)} PDFs...")
            pdf_out = ingest_dir / "papers"
            for pdf in pdfs:
                cmd = build_ingest_cmd(config, str(pdf), pdf_out, config.pdf_verbose)
                success, _ = run_cmd(cmd, f"Converting: {pdf.name}", timeout=120)
                if success:
                    ingested["pdfs"] += 1
    
    # 4b. Ingest GitHub repos
    github_repos = refs.get("github", [])
    if config.max_repos is not None:
        github_repos = github_repos[:config.max_repos]
    if github_repos:
        print(f"\n  📂 Ingesting {len(github_repos)} GitHub repos (mode: {config.git_mode})...")
        github_out = ingest_dir / "github"
        
        for repo in github_repos:
            repo_url = f"https://github.com/{repo}"
            
            if config.git_mode == "readme":
                # Just fetch README via API
                cmd = build_ingest_cmd(config, f"{repo_url}/blob/main/README.md", github_out, config.git_verbose)
                desc = f"Fetching README: {repo}"
            elif config.git_mode == "api":
                # Use ingestor ingest for API access
                cmd = build_ingest_cmd(config, repo_url, github_out, config.git_verbose)
                desc = f"API fetch: {repo}"
            else:
                # Use ingestor clone for full repository
                cmd = build_clone_cmd(config, repo_url, github_out)
                desc = f"Cloning: {repo}"
            
            success, _ = run_cmd(cmd, desc, timeout=300)
            if success:
                ingested["github"] += 1
    
    # 4c. Ingest YouTube videos
    youtube_urls = refs.get("youtube", [])
    if config.max_youtube is not None:
        youtube_urls = youtube_urls[:config.max_youtube]
    if youtube_urls:
        print(f"\n  🎬 Ingesting {len(youtube_urls)} YouTube videos...")
        youtube_out = ingest_dir / "youtube"
        for url in youtube_urls:
            cmd = build_ingest_cmd(config, url, youtube_out, config.youtube_verbose)
            success, _ = run_cmd(cmd, f"Transcribing: {url[:50]}...", timeout=120)
            if success:
                ingested["youtube"] += 1
    
    # 4d. Ingest URLs
    urls = refs.get("urls", [])
    if config.max_urls is not None:
        urls = urls[:config.max_urls]
    if urls:
        print(f"\n  🔗 Ingesting {len(urls)} URLs...")
        web_out = ingest_dir / "web"
        
        for url in urls:
            # Decide whether to use crawl or simple ingest
            if config.use_deep_crawl:
                cmd = build_crawl_cmd(config, url, web_out)
                desc = f"Crawling: {url[:50]}..."
            else:
                cmd = build_ingest_cmd(config, url, web_out, config.web_verbose)
                desc = f"Fetching: {url[:50]}..."
            
            success, _ = run_cmd(cmd, desc, timeout=180)
            if success:
                ingested["urls"] += 1
    
    # 4e. Copy research report
    if research_file and research_file.exists():
        print(f"\n  📝 Copying research report...")
        research_out = ingest_dir / "research"
        research_out.mkdir(parents=True, exist_ok=True)
        dest = research_out / research_file.name
        dest.write_text(research_file.read_text())
        ingested["research"] = 1
        print(f"  ✓ Copied to {dest}")
    
    total_md = len(list(ingest_dir.rglob("*.md")))
    
    print(f"\n✅ Ingestion complete:")
    print(f"   PDFs: {ingested['pdfs']}")
    print(f"   GitHub: {ingested['github']}")
    print(f"   YouTube: {ingested['youtube']}")
    print(f"   URLs: {ingested['urls']}")
    print(f"   Research: {ingested['research']}")
    print(f"   Total markdown files: {total_md}")
    
    return ingested


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(config: PipelineConfig, refs: dict, retrieve_stats: dict, ingest_stats: dict):
    """Generate final pipeline report."""
    report = f"""# Pipeline Report

**Topic:** {config.topic}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

### Phase 1: Research
| Setting | Value |
|---------|-------|
| Skip | {config.skip_research} |
| Verbose | {config.research_verbose} |
| Stream | {not config.research_no_stream} |

### Phase 2: Parse References
| Setting | Value |
|---------|-------|
| Format | {config.parse_format} |

### Phase 3: Paper Retrieval
| Setting | Value |
|---------|-------|
| Max Papers | {config.max_papers} |
| Skip Existing | {config.paper_skip_existing} |
| Verify BibTeX | {config.verify_bibtex} |
| BibTeX Format | {config.bibtex_format} |

### Phase 4: Ingestion
| Setting | Value |
|---------|-------|
| Git Mode | {config.git_mode} |
| Shallow Clone | {config.git_shallow} |
| Branch | {config.git_branch or 'default'} |
| Max Files/Repo | {config.git_max_files or 'unlimited'} |
| Max Repos | {config.max_repos} |
| Max URLs | {config.max_urls} |
| Max YouTube | {config.max_youtube} |
| Deep Crawl | {config.use_deep_crawl} |
| Crawl Depth | {config.crawl_max_depth} |
| Crawl Pages | {config.crawl_max_pages} |
| VLM Describe | {config.use_vlm_describe} |
| Claude Agent | {config.use_claude_agent} |

## Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| 1. Research | Deep research on topic | `1_research/` |
| 2. Parse Refs | Extract DOIs, arXiv, GitHub, YouTube, URLs | `2_references/` |
| 3. Retrieve | Download papers + BibTeX | `3_papers/`, `3_bibtex/` |
| 4. Ingest | Convert to markdown | `4_markdown/` |

## Extracted References

- **DOIs:** {len(refs.get('dois', []))}
- **arXiv papers:** {len(refs.get('arxiv', []))}
- **GitHub repos:** {len(refs.get('github', []))}
- **YouTube videos:** {len(refs.get('youtube', []))}
- **URLs:** {len(refs.get('urls', []))}

## Retrieved

- **Papers downloaded:** {len(retrieve_stats.get('downloaded', []))}
- **BibTeX entries:** {len(retrieve_stats.get('bibtex', []))}

## Ingested

- **PDFs converted:** {ingest_stats.get('pdfs', 0)}
- **GitHub repos:** {ingest_stats.get('github', 0)}
- **YouTube videos:** {ingest_stats.get('youtube', 0)}
- **URLs:** {ingest_stats.get('urls', 0)}

## Output Structure

```
{config.output_dir}/
├── 1_research/          # Deep research output
│   └── research_report.md
├── 2_references/        # Extracted references
│   ├── references.json
│   └── references.md
├── 3_papers/            # Downloaded PDFs
├── 3_bibtex/            # BibTeX citations
│   ├── combined.bib
│   └── verified/
└── 4_markdown/          # Final markdown for RAG
    ├── papers/
    ├── github/
    ├── youtube/
    ├── web/
    └── research/
```

## Usage

All markdown files in `4_markdown/` are ready for RAG ingestion.
"""
    
    report_file = config.output_dir / "PIPELINE_REPORT.md"
    report_file.write_text(report)
    print(f"\n📄 Report: {report_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(config: PipelineConfig):
    """Run the full pipeline with given configuration."""
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║           Research-to-RAG Pipeline (Interactive)                  ║
╠═══════════════════════════════════════════════════════════════════╣
║  Topic: {config.topic[:55]:<55} ║
║  Output: {str(config.output_dir)[:55]:<55} ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Deep Research
    research_file = step1_research(config)
    
    if not research_file or not research_file.exists():
        print("\n❌ No research file available. Exiting.")
        return
    
    # STEP 2: Parse References
    if config.skip_parse and config.refs_file:
        print("\n" + "═"*65)
        print("STEP 2: PARSE REFERENCES (SKIPPED)")
        print("═"*65)
        print(f"  Using existing: {config.refs_file}")
        # Load refs from existing file
        refs = {"dois": [], "arxiv": [], "github": [], "youtube": [], "urls": []}
        if config.refs_file.exists():
            try:
                with open(config.refs_file) as f:
                    raw_refs = json.load(f)
                if isinstance(raw_refs, list):
                    for item in raw_refs:
                        ref_type = item.get("type", "")
                        value = item.get("value", "") or item.get("url", "")
                        if ref_type == "doi" and value:
                            refs["dois"].append(value)
                        elif ref_type == "arxiv" and value:
                            refs["arxiv"].append(value)
                        elif ref_type == "github" and value:
                            refs["github"].append(value)
                        elif ref_type == "youtube" and value:
                            url = item.get("url", f"https://youtube.com/watch?v={value}")
                            refs["youtube"].append(url)
                        elif ref_type == "website" and value:
                            url = item.get("url", value)
                            refs["urls"].append(url)
                elif isinstance(raw_refs, dict):
                    refs = raw_refs
            except json.JSONDecodeError:
                print("  ⚠ Could not parse references file")
    else:
        refs = step2_parse_refs(research_file, config)
    
    # STEP 3: Retrieve Papers & DOI2BIB
    if config.skip_retrieval:
        print("\n" + "═"*65)
        print("STEP 3: PAPER RETRIEVAL & BIBTEX (SKIPPED)")
        print("═"*65)
        retrieve_stats = {"downloaded": [], "bibtex": []}
    else:
        retrieve_stats = step3_retrieve(refs, config)
    
    # STEP 4: Ingest to Markdown
    if config.skip_ingest:
        print("\n" + "═"*65)
        print("STEP 4: INGEST TO MARKDOWN (SKIPPED)")
        print("═"*65)
        ingest_stats = {"pdfs": 0, "github": 0, "youtube": 0, "urls": 0}
    else:
        ingest_stats = step4_ingest(refs, config, research_file)
    
    # Generate Report
    generate_report(config, refs, retrieve_stats, ingest_stats)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    ✅ Pipeline Complete!                          ║
╠═══════════════════════════════════════════════════════════════════╣
║  All markdown files ready in: {str(config.output_dir / '4_markdown'):<35} ║
╚═══════════════════════════════════════════════════════════════════╝
""")


def main():
    """Main entry point - interactive configuration then run pipeline."""
    try:
        config = interactive_config()
        run_pipeline(config)
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline cancelled by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
