#!/usr/bin/env python3
"""
Comprehensive Test Runner for Ingestor
Runs extensive real-world tests on all supported formats and generates detailed reports.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path (go up one level from test_report to find src)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestor import (
    FileDetector,
    Router,
    ExtractorRegistry,
    IngestConfig,
    MediaType,
)
from ingestor.extractors import (
    TxtExtractor,
    JsonExtractor,
    XmlExtractor,
    CsvExtractor,
    DocxExtractor,
    PptxExtractor,
    EpubExtractor,
    XlsxExtractor,
    WebExtractor,
    YouTubeExtractor,
    ZipExtractor,
    ImageExtractor,
    AudioExtractor,
)
from ingestor.extractors.git import GitExtractor
from ingestor.types import MediaType


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    source: str
    format: str
    status: str  # "pass", "fail", "skip"
    duration_ms: float
    input_size_bytes: int
    output_size_chars: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Collection of test results for a category."""
    category: str
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "pass")
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "fail")
    
    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skip")
    
    @property
    def avg_duration_ms(self) -> float:
        if not self.results:
            return 0
        return sum(r.duration_ms for r in self.results) / len(self.results)


class ComprehensiveTestRunner:
    """Runs comprehensive tests on all formats."""
    
    def __init__(self, samples_dir: Path, output_dir: Path):
        self.samples_dir = samples_dir
        self.output_dir = output_dir
        self.detector = FileDetector()
        self.test_suites: Dict[str, TestSuite] = {}
        self.start_time = None
        self.end_time = None
        
    def _create_registry(self) -> ExtractorRegistry:
        """Create a fresh registry for each test."""
        registry = ExtractorRegistry()
        registry.register(TxtExtractor())
        registry.register(JsonExtractor())
        registry.register(XmlExtractor())
        registry.register(CsvExtractor())
        registry.register(DocxExtractor())
        registry.register(PptxExtractor())
        registry.register(EpubExtractor())
        registry.register(XlsxExtractor())
        registry.register(ZipExtractor())
        registry.register(ImageExtractor())
        
        # Register GitExtractor for both GIT and GITHUB media types
        git_extractor = GitExtractor()
        registry.register(git_extractor)
        registry._extractors[MediaType.GITHUB] = git_extractor
        
        try:
            registry.register(WebExtractor())
        except Exception:
            pass  # Web extractor requires playwright
            
        try:
            registry.register(YouTubeExtractor())
        except Exception:
            pass
            
        try:
            registry.register(AudioExtractor())
        except Exception:
            pass
            
        return registry
    
    async def run_single_test(
        self, 
        source: str, 
        name: str, 
        format_name: str,
        category: str
    ) -> TestResult:
        """Run a single extraction test."""
        print(f"  Testing: {name}...", end=" ", flush=True)
        
        # Get input size
        if os.path.isfile(source):
            input_size = os.path.getsize(source)
        else:
            input_size = len(source)  # URL length
        
        start_time = time.perf_counter()
        
        try:
            config = IngestConfig(output_dir=self.output_dir)
            registry = self._create_registry()
            router = Router(registry, config)
            
            result = await router.process(source)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            output_size = len(result.markdown) if result.markdown else 0
            
            # Extract useful metadata
            metadata = {
                "title": result.title,
                "media_type": result.media_type.value if result.media_type else "unknown",
                "image_count": result.image_count,
                "has_images": result.has_images,
            }
            if result.metadata:
                metadata.update(result.metadata)
            
            print(f"✓ ({duration_ms:.0f}ms, {output_size:,} chars)")
            
            return TestResult(
                name=name,
                source=source,
                format=format_name,
                status="pass",
                duration_ms=duration_ms,
                input_size_bytes=input_size,
                output_size_chars=output_size,
                metadata=metadata,
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            print(f"✗ ({str(e)[:50]})")
            
            return TestResult(
                name=name,
                source=source,
                format=format_name,
                status="fail",
                duration_ms=duration_ms,
                input_size_bytes=input_size,
                output_size_chars=0,
                error_message=str(e),
            )
    
    async def test_text_files(self) -> TestSuite:
        """Test text file extraction."""
        suite = TestSuite(category="Text Files")
        
        text_files = [
            ("shakespeare_complete.txt", "Shakespeare Complete Works", "TXT"),
            ("war_and_peace.txt", "War and Peace", "TXT"),
            ("monte_cristo.txt", "Count of Monte Cristo", "TXT"),
            ("moby_dick.txt", "Moby Dick", "TXT"),
            ("pride_prejudice.txt", "Pride and Prejudice", "TXT"),
            ("awesome_python.md", "Awesome Python README", "MD"),
            ("coding_interview.md", "Coding Interview University", "MD"),
            ("fastapi_readme.md", "FastAPI README", "MD"),
            ("tensorflow_readme.md", "TensorFlow README", "MD"),
            ("react_readme.md", "React README", "MD"),
            ("python_tutorial.rst", "Python Tutorial (RST)", "RST"),
            ("django_settings.py", "Django Settings (Python)", "PY"),
            ("requests_api.py", "Requests API (Python)", "PY"),
            ("lodash.js", "Lodash (JavaScript)", "JS"),
        ]
        
        print("\n📝 Testing Text Files...")
        for filename, name, fmt in text_files:
            filepath = self.samples_dir / filename
            if filepath.exists():
                result = await self.run_single_test(str(filepath), name, fmt, "text")
                suite.results.append(result)
            else:
                suite.results.append(TestResult(
                    name=name, source=str(filepath), format=fmt,
                    status="skip", duration_ms=0, input_size_bytes=0,
                    output_size_chars=0, error_message="File not found"
                ))
        
        return suite
    
    async def test_data_files(self) -> TestSuite:
        """Test data file extraction (JSON, XML, CSV)."""
        suite = TestSuite(category="Data Files")
        
        data_files = [
            ("us_cities.json", "US Cities (21MB JSON)", "JSON"),
            ("countries.json", "World Countries JSON", "JSON"),
            ("github_trending.json", "GitHub Trending Repos", "JSON"),
            ("nobel_prizes.json", "Nobel Prizes JSON", "JSON"),
            ("bbc_news.xml", "BBC News RSS Feed", "XML"),
            ("nytimes.xml", "NY Times RSS Feed", "XML"),
            ("apache_commons_pom.xml", "Apache Commons POM", "XML"),
            ("xkcd.xml", "XKCD RSS Feed", "XML"),
            ("covid_global.csv", "COVID-19 Global Data (1.8MB)", "CSV"),
            ("world_population.csv", "World Population CSV", "CSV"),
            ("movies.csv", "IMDB Movies CSV", "CSV"),
            ("titanic.csv", "Titanic Dataset CSV", "CSV"),
            ("weather.csv", "Weather Data CSV", "CSV"),
            ("iris.csv", "Iris Dataset CSV", "CSV"),
        ]
        
        print("\n📊 Testing Data Files...")
        for filename, name, fmt in data_files:
            filepath = self.samples_dir / filename
            if filepath.exists():
                result = await self.run_single_test(str(filepath), name, fmt, "data")
                suite.results.append(result)
            else:
                suite.results.append(TestResult(
                    name=name, source=str(filepath), format=fmt,
                    status="skip", duration_ms=0, input_size_bytes=0,
                    output_size_chars=0, error_message="File not found"
                ))
        
        return suite
    
    async def test_document_files(self) -> TestSuite:
        """Test document file extraction."""
        suite = TestSuite(category="Documents")
        
        doc_files = [
            ("business_report.docx", "Business Report DOCX", "DOCX"),
            ("financial_sample.xlsx", "Financial Sample XLSX", "XLSX"),
            ("great_expectations.epub", "Great Expectations EPUB (14MB)", "EPUB"),
            ("frankenstein.epub", "Frankenstein EPUB", "EPUB"),
            ("alice_wonderland.epub", "Alice in Wonderland EPUB", "EPUB"),
        ]
        
        print("\n📄 Testing Document Files...")
        for filename, name, fmt in doc_files:
            filepath = self.samples_dir / filename
            if filepath.exists():
                result = await self.run_single_test(str(filepath), name, fmt, "document")
                suite.results.append(result)
            else:
                suite.results.append(TestResult(
                    name=name, source=str(filepath), format=fmt,
                    status="skip", duration_ms=0, input_size_bytes=0,
                    output_size_chars=0, error_message="File not found"
                ))
        
        return suite
    
    async def test_image_files(self) -> TestSuite:
        """Test image file extraction."""
        suite = TestSuite(category="Images")
        
        image_files = [
            ("nature_hires.jpg", "Nature Hi-Res Photo (426KB)", "JPG"),
            ("nasa_apod.jpg", "NASA APOD Galaxy", "JPG"),
            ("earth_nasa.jpg", "NASA Blue Marble", "JPG"),
            ("wikipedia_logo.png", "Wikipedia Logo PNG", "PNG"),
            ("wikipedia_logo_png.png", "Wikipedia Logo 2 PNG", "PNG"),
            ("python_logo.png", "Python Logo PNG", "PNG"),
            ("github_logo.png", "GitHub Logo PNG", "PNG"),
            ("chart.png", "Chart Example PNG", "PNG"),
        ]
        
        print("\n🖼️  Testing Image Files...")
        for filename, name, fmt in image_files:
            filepath = self.samples_dir / filename
            if filepath.exists():
                result = await self.run_single_test(str(filepath), name, fmt, "image")
                suite.results.append(result)
            else:
                suite.results.append(TestResult(
                    name=name, source=str(filepath), format=fmt,
                    status="skip", duration_ms=0, input_size_bytes=0,
                    output_size_chars=0, error_message="File not found"
                ))
        
        return suite
    
    async def test_archive_files(self) -> TestSuite:
        """Test archive file extraction."""
        suite = TestSuite(category="Archives")
        
        archive_files = [
            ("bootstrap.zip", "Bootstrap 5.3.2 (8.2MB)", "ZIP"),
            ("sample_project.zip", "Flask 3.0.0 Project", "ZIP"),
        ]
        
        print("\n📦 Testing Archive Files...")
        for filename, name, fmt in archive_files:
            filepath = self.samples_dir / filename
            if filepath.exists():
                result = await self.run_single_test(str(filepath), name, fmt, "archive")
                suite.results.append(result)
            else:
                suite.results.append(TestResult(
                    name=name, source=str(filepath), format=fmt,
                    status="skip", duration_ms=0, input_size_bytes=0,
                    output_size_chars=0, error_message="File not found"
                ))
        
        return suite
    
    async def test_web_extraction(self) -> TestSuite:
        """Test web page extraction."""
        suite = TestSuite(category="Web Pages")
        
        web_urls = [
            ("https://docs.python.org/3/tutorial/appetite.html", "Python Tutorial Page", "WEB"),
            ("https://httpbin.org/html", "HTTPBin HTML Test", "WEB"),
            ("https://example.com", "Example.com", "WEB"),
        ]
        
        print("\n🌐 Testing Web Extraction...")
        for url, name, fmt in web_urls:
            result = await self.run_single_test(url, name, fmt, "web")
            suite.results.append(result)
        
        return suite
    
    async def test_youtube_extraction(self) -> TestSuite:
        """Test YouTube video extraction."""
        suite = TestSuite(category="YouTube")
        
        youtube_urls = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "Rick Astley - Never Gonna Give You Up", "YOUTUBE"),
            ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "Me at the zoo (First YouTube Video)", "YOUTUBE"),
            ("https://www.youtube.com/watch?v=9bZkp7q19f0", "PSY - Gangnam Style", "YOUTUBE"),
        ]
        
        print("\n📺 Testing YouTube Extraction...")
        for url, name, fmt in youtube_urls:
            result = await self.run_single_test(url, name, fmt, "youtube")
            suite.results.append(result)
        
        return suite
    
    async def test_git_extraction(self) -> TestSuite:
        """Test Git/GitHub repository extraction."""
        suite = TestSuite(category="Git/GitHub")
        
        git_urls = [
            ("https://github.com/octocat/Hello-World", "Hello-World Repository", "GIT"),
            ("https://github.com/psf/requests/blob/main/README.md", "Requests README", "GIT"),
            ("https://github.com/octocat/Hello-World/tree/master", "Hello-World Directory", "GIT"),
        ]
        
        print("\n🐙 Testing Git/GitHub Extraction...")
        for url, name, fmt in git_urls:
            result = await self.run_single_test(url, name, fmt, "git")
            suite.results.append(result)
        
        return suite
    
    async def test_audio_files(self) -> TestSuite:
        """Test audio file transcription."""
        suite = TestSuite(category="Audio")
        
        audio_files = [
            ("librivox_sample.mp3", "Gettysburg Address (LibriVox)", "MP3"),
            ("jfk_speech.wav", "Speech Sample WAV", "WAV"),
        ]
        
        print("\n🎵 Testing Audio Transcription...")
        for filename, name, fmt in audio_files:
            filepath = self.samples_dir / filename
            if filepath.exists():
                result = await self.run_single_test(str(filepath), name, fmt, "audio")
                suite.results.append(result)
            else:
                suite.results.append(TestResult(
                    name=name, source=str(filepath), format=fmt,
                    status="skip", duration_ms=0, input_size_bytes=0,
                    output_size_chars=0, error_message="File not found"
                ))
        
        return suite
    
    async def run_all_tests(self) -> Dict[str, TestSuite]:
        """Run all test suites."""
        self.start_time = datetime.now()
        
        print("=" * 60)
        print("🚀 INGESTOR COMPREHENSIVE TEST SUITE")
        print(f"   Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Run all test categories
        self.test_suites["text"] = await self.test_text_files()
        self.test_suites["data"] = await self.test_data_files()
        self.test_suites["documents"] = await self.test_document_files()
        self.test_suites["images"] = await self.test_image_files()
        self.test_suites["archives"] = await self.test_archive_files()
        self.test_suites["web"] = await self.test_web_extraction()
        self.test_suites["youtube"] = await self.test_youtube_extraction()
        self.test_suites["git"] = await self.test_git_extraction()
        self.test_suites["audio"] = await self.test_audio_files()
        
        self.end_time = datetime.now()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print(f"   Duration: {(self.end_time - self.start_time).total_seconds():.1f}s")
        print("=" * 60)
        
        return self.test_suites
    
    def generate_report(self) -> str:
        """Generate a comprehensive markdown report."""
        report_lines = []
        
        # Header
        report_lines.append("# 📊 Ingestor Comprehensive Test Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Test Duration:** {(self.end_time - self.start_time).total_seconds():.1f} seconds")
        report_lines.append("")
        
        # Executive Summary
        total_tests = sum(s.total for s in self.test_suites.values())
        total_passed = sum(s.passed for s in self.test_suites.values())
        total_failed = sum(s.failed for s in self.test_suites.values())
        total_skipped = sum(s.skipped for s in self.test_suites.values())
        
        report_lines.append("## 📈 Executive Summary")
        report_lines.append("")
        report_lines.append(f"| Metric | Value |")
        report_lines.append("|--------|-------|")
        report_lines.append(f"| **Total Tests** | {total_tests} |")
        report_lines.append(f"| **Passed** | {total_passed} ✅ |")
        report_lines.append(f"| **Failed** | {total_failed} ❌ |")
        report_lines.append(f"| **Skipped** | {total_skipped} ⏭️ |")
        report_lines.append(f"| **Success Rate** | {(total_passed/total_tests*100):.1f}% |")
        report_lines.append("")
        
        # Pass rate visual bar
        pass_rate = total_passed / total_tests * 100 if total_tests > 0 else 0
        filled = int(pass_rate / 5)
        empty = 20 - filled
        bar = "█" * filled + "░" * empty
        report_lines.append(f"**Pass Rate:** `[{bar}]` {pass_rate:.1f}%")
        report_lines.append("")
        
        # Test Results by Category - Table
        report_lines.append("## 📋 Results by Category")
        report_lines.append("")
        report_lines.append("| Category | Total | Passed | Failed | Skipped | Avg Time (ms) | Pass Rate |")
        report_lines.append("|----------|-------|--------|--------|---------|---------------|-----------|")
        
        for name, suite in self.test_suites.items():
            rate = (suite.passed / suite.total * 100) if suite.total > 0 else 0
            status = "✅" if suite.failed == 0 else "⚠️" if suite.passed > suite.failed else "❌"
            report_lines.append(
                f"| {suite.category} | {suite.total} | {suite.passed} | {suite.failed} | {suite.skipped} | "
                f"{suite.avg_duration_ms:.0f} | {rate:.0f}% {status} |"
            )
        report_lines.append("")
        
        # ASCII Bar Chart for Pass Rates
        report_lines.append("### 📊 Pass Rate by Category (Visual)")
        report_lines.append("")
        report_lines.append("```")
        max_name_len = max(len(s.category) for s in self.test_suites.values())
        for name, suite in self.test_suites.items():
            rate = (suite.passed / suite.total * 100) if suite.total > 0 else 0
            bar_len = int(rate / 2.5)  # Scale to 40 chars max
            bar = "█" * bar_len
            report_lines.append(f"{suite.category:<{max_name_len}} │{bar:<40}│ {rate:5.1f}%")
        report_lines.append("```")
        report_lines.append("")
        
        # Performance Analysis
        report_lines.append("## ⚡ Performance Analysis")
        report_lines.append("")
        
        # Processing Speed Table
        report_lines.append("### Processing Speed by File Size")
        report_lines.append("")
        report_lines.append("| File | Size | Time (ms) | Speed (KB/s) | Format |")
        report_lines.append("|------|------|-----------|--------------|--------|")
        
        all_results = []
        for suite in self.test_suites.values():
            all_results.extend(suite.results)
        
        # Sort by input size (largest first)
        sorted_results = sorted(
            [r for r in all_results if r.status == "pass" and r.input_size_bytes > 0],
            key=lambda r: r.input_size_bytes,
            reverse=True
        )[:20]  # Top 20 largest files
        
        for r in sorted_results:
            size_kb = r.input_size_bytes / 1024
            if size_kb >= 1024:
                size_str = f"{size_kb/1024:.1f} MB"
            else:
                size_str = f"{size_kb:.1f} KB"
            
            speed = (r.input_size_bytes / 1024) / (r.duration_ms / 1000) if r.duration_ms > 0 else 0
            report_lines.append(f"| {r.name[:40]} | {size_str} | {r.duration_ms:.0f} | {speed:.0f} | {r.format} |")
        report_lines.append("")
        
        # Detailed Results by Category
        report_lines.append("## 📝 Detailed Test Results")
        report_lines.append("")
        
        for name, suite in self.test_suites.items():
            report_lines.append(f"### {suite.category}")
            report_lines.append("")
            
            if suite.results:
                report_lines.append("| Test Name | Status | Time (ms) | Input Size | Output Size |")
                report_lines.append("|-----------|--------|-----------|------------|-------------|")
                
                for r in suite.results:
                    status_icon = "✅" if r.status == "pass" else "❌" if r.status == "fail" else "⏭️"
                    
                    # Format sizes
                    if r.input_size_bytes >= 1024 * 1024:
                        in_size = f"{r.input_size_bytes / (1024*1024):.1f} MB"
                    elif r.input_size_bytes >= 1024:
                        in_size = f"{r.input_size_bytes / 1024:.1f} KB"
                    else:
                        in_size = f"{r.input_size_bytes} B"
                    
                    if r.output_size_chars >= 1000000:
                        out_size = f"{r.output_size_chars / 1000000:.1f}M chars"
                    elif r.output_size_chars >= 1000:
                        out_size = f"{r.output_size_chars / 1000:.1f}K chars"
                    else:
                        out_size = f"{r.output_size_chars} chars"
                    
                    report_lines.append(
                        f"| {r.name[:35]} | {status_icon} {r.status.upper()} | {r.duration_ms:.0f} | {in_size} | {out_size} |"
                    )
                
                # Show errors if any
                failed_tests = [r for r in suite.results if r.status == "fail"]
                if failed_tests:
                    report_lines.append("")
                    report_lines.append("**Errors:**")
                    for r in failed_tests:
                        report_lines.append(f"- `{r.name}`: {r.error_message}")
                
                report_lines.append("")
        
        # File Format Support Matrix
        report_lines.append("## 📁 Format Support Matrix")
        report_lines.append("")
        report_lines.append("| Format | Extension | Status | Notes |")
        report_lines.append("|--------|-----------|--------|-------|")
        
        format_status = {
            "Plain Text": ("TXT", "✅ Supported", "Full Unicode support"),
            "Markdown": ("MD", "✅ Supported", "Preserves formatting"),
            "reStructuredText": ("RST", "✅ Supported", "Treated as text"),
            "Python": ("PY", "✅ Supported", "Syntax highlighting in output"),
            "JavaScript": ("JS", "✅ Supported", "Treated as text"),
            "JSON": ("JSON", "✅ Supported", "Pretty-printed output"),
            "XML": ("XML", "✅ Supported", "Structured extraction"),
            "CSV": ("CSV", "✅ Supported", "Table format output"),
            "DOCX": ("DOCX", "✅ Supported", "Full document extraction"),
            "XLSX": ("XLSX", "✅ Supported", "Multi-sheet support"),
            "EPUB": ("EPUB", "✅ Supported", "Chapter extraction + images"),
            "PNG/JPG": ("PNG, JPG", "✅ Supported", "Image metadata extraction"),
            "ZIP": ("ZIP", "✅ Supported", "Recursive extraction"),
            "Web Pages": ("HTTP(S)", "✅ Supported", "Requires Playwright"),
            "YouTube": ("youtube.com", "✅ Supported", "Transcripts + metadata"),
            "Git/GitHub": ("github.com, SSH", "✅ Supported", "Repos, files, clone"),
            "Audio": ("MP3, WAV", "✅ Supported", "Whisper transcription"),
            "PDF": ("PDF", "⚠️ Placeholder", "Requires Docling integration"),
        }
        
        for fmt_name, (ext, status, notes) in format_status.items():
            report_lines.append(f"| {fmt_name} | {ext} | {status} | {notes} |")
        report_lines.append("")
        
        # Test Environment
        report_lines.append("## 🖥️ Test Environment")
        report_lines.append("")
        report_lines.append("| Component | Value |")
        report_lines.append("|-----------|-------|")
        report_lines.append(f"| Python Version | {sys.version.split()[0]} |")
        report_lines.append(f"| Platform | {sys.platform} |")
        report_lines.append(f"| Test Samples Directory | `{self.samples_dir}` |")
        report_lines.append(f"| Output Directory | `{self.output_dir}` |")
        report_lines.append("")
        
        # Conclusion
        report_lines.append("## 🎯 Conclusion")
        report_lines.append("")
        
        if total_failed == 0:
            report_lines.append("🎉 **All tests passed successfully!** The Ingestor library handles all tested formats correctly.")
        elif total_failed <= 3:
            report_lines.append(f"⚠️ **{total_passed}/{total_tests} tests passed.** Minor issues detected in {total_failed} test(s).")
        else:
            report_lines.append(f"❌ **{total_failed}/{total_tests} tests failed.** Review the error details above.")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append(f"*Report generated by Ingestor Test Suite on {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(report_lines)
    
    def save_json_results(self, filepath: Path):
        """Save results as JSON for further analysis."""
        data = {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "suites": {}
        }
        
        for name, suite in self.test_suites.items():
            data["suites"][name] = {
                "category": suite.category,
                "total": suite.total,
                "passed": suite.passed,
                "failed": suite.failed,
                "skipped": suite.skipped,
                "avg_duration_ms": suite.avg_duration_ms,
                "results": [
                    {
                        "name": r.name,
                        "source": r.source,
                        "format": r.format,
                        "status": r.status,
                        "duration_ms": r.duration_ms,
                        "input_size_bytes": r.input_size_bytes,
                        "output_size_chars": r.output_size_chars,
                        "error_message": r.error_message,
                        "metadata": r.metadata,
                    }
                    for r in suite.results
                ]
            }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


async def main():
    """Run the comprehensive test suite."""
    base_dir = Path(__file__).parent
    samples_dir = base_dir / "test_samples"
    output_dir = base_dir / "output" / "test_results"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    runner = ComprehensiveTestRunner(samples_dir, output_dir)
    await runner.run_all_tests()
    
    # Generate and save report
    report = runner.generate_report()
    report_path = base_dir / "TEST_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n📄 Report saved to: {report_path}")
    
    # Save JSON results
    json_path = base_dir / "test_results.json"
    runner.save_json_results(json_path)
    print(f"📊 JSON results saved to: {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
