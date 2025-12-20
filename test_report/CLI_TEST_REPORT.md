# 📊 Ingestor CLI Comprehensive Test Report

**Generated:** 2025-12-20 09:41:46
**Test Duration:** 214.7 seconds

---

## 🔬 Testing Methodology

### Overview

This comprehensive test suite validates the **Ingestor** CLI tool's ability to extract and convert
various media formats to markdown. Unlike unit tests, these tests use the **actual CLI commands**
(`ingestor ingest <source> -o <output>`) with **real-world data** from public sources.

### Test Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLI TEST RUNNER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. SETUP PHASE                                                              │
│     ├── Load real test files from test_samples/ directory                   │
│     ├── Create clean output directory                                        │
│     └── Initialize test suite containers                                     │
│                                                                              │
│  2. EXECUTION PHASE (per file/URL)                                          │
│     ├── Build CLI command: `ingestor ingest <source> -o <output>`           │
│     ├── Execute via subprocess with timeout                                  │
│     ├── Capture stdout/stderr and exit code                                  │
│     └── Measure: execution time, input size                                  │
│                                                                              │
│  3. VALIDATION PHASE                                                         │
│     ├── Check exit code (0 = success)                                        │
│     ├── Verify output files were created                                     │
│     └── Record pass/fail status with metrics                                │
│                                                                              │
│  4. REPORTING PHASE                                                          │
│     ├── Aggregate results by category                                        │
│     ├── Calculate statistics (pass rate, avg time)                          │
│     └── Generate markdown report with visualizations                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How Each Test Works

For each test file or URL, the test runner:

1. **Builds a CLI command**: `uv run ingestor ingest <source> -o <output>`
2. **Executes via subprocess** with a 5-minute timeout
3. **Checks the exit code**: 0 = success, non-zero = failure
4. **Measures performance** including execution time and file sizes
5. **Records the result** as pass or fail with error details

### Test Data Sources

All test files are **real documents** downloaded from public sources:

| Source | Content Type | Examples |
|--------|--------------|----------|
| **Project Gutenberg** | Public domain books | Shakespeare, War and Peace, Moby Dick |
| **GitHub Raw** | Code & documentation | FastAPI README, Django settings, React README |
| **Public APIs** | Structured data | Nobel Prizes JSON, GitHub trending repos |
| **RSS Feeds** | XML feeds | BBC News, NY Times, XKCD |
| **Kaggle/Public Datasets** | CSV data | COVID-19 data, Titanic, IMDB movies |
| **Wikipedia/Wikimedia** | Images | NASA photos, logos, charts |
| **Archive.org/LibriVox** | Audio | Gettysburg Address speech, JFK speech |
| **Live Websites** | Web pages | Python docs, HTTPBin, Example.com |
| **YouTube** | Video metadata | Rick Astley, Gangnam Style, first YouTube video |
| **Git/GitHub** | Repository content | Hello-World, Requests README |

### Test Categories Explained

| Category | What's Tested | CLI Command Example |
|----------|---------------|---------------------|
| **Text Files** | TXT, MD, RST, PY, JS | `ingestor ingest document.txt` |
| **Data Files** | JSON, XML, CSV | `ingestor ingest data.json` |
| **Documents** | DOCX, XLSX, EPUB | `ingestor ingest report.docx` |
| **Images** | PNG, JPG, SVG | `ingestor ingest photo.jpg` |
| **Archives** | ZIP files | `ingestor ingest archive.zip` |
| **Web Pages** | HTTP(S) URLs | `ingestor ingest https://example.com` |
| **YouTube** | Video URLs | `ingestor ingest https://youtube.com/watch?v=...` |
| **Git/GitHub** | Repo/file URLs | `ingestor ingest https://github.com/owner/repo` |
| **Git Clone** | Full repo clone | `ingestor clone https://github.com/owner/repo` |
| **Audio** | MP3, WAV | `ingestor ingest audio.mp3` |

### Success Criteria

A test is considered **PASSED** when:
- ✅ The CLI command exits with code 0
- ✅ Output files are created in the output directory
- ✅ No error messages in stderr

A test is considered **FAILED** when:
- ❌ The CLI command exits with non-zero code
- ❌ Error messages are printed to stderr
- ❌ The command times out (> 5 minutes)

---

## 📈 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Tests** | 56 |
| **Passed** | 56 ✅ |
| **Failed** | 0 ❌ |
| **Skipped** | 0 ⏭️ |
| **Success Rate** | 100.0% |

**Pass Rate:** `[████████████████████]` 100.0%

## 📋 Results by Category

| Category | Total | Passed | Failed | Skipped | Avg Time (ms) | Pass Rate |
|----------|-------|--------|--------|---------|---------------|-----------|
| Text Files | 14 | 14 | 0 | 0 | 440 | 100% ✅ |
| Data Files | 14 | 14 | 0 | 0 | 460 | 100% ✅ |
| Documents | 5 | 5 | 0 | 0 | 3285 | 100% ✅ |
| Images | 8 | 8 | 0 | 0 | 574 | 100% ✅ |
| Archives | 2 | 2 | 0 | 0 | 4130 | 100% ✅ |
| Web Pages | 3 | 3 | 0 | 0 | 3636 | 100% ✅ |
| YouTube | 3 | 3 | 0 | 0 | 2508 | 100% ✅ |
| Git/GitHub | 3 | 3 | 0 | 0 | 2902 | 100% ✅ |
| Git Clone | 2 | 2 | 0 | 0 | 1629 | 100% ✅ |
| Audio | 2 | 2 | 0 | 0 | 71184 | 100% ✅ |

### 📊 Pass Rate by Category (Visual)

```
Text Files │████████████████████████████████████████│ 100.0%
Data Files │████████████████████████████████████████│ 100.0%
Documents  │████████████████████████████████████████│ 100.0%
Images     │████████████████████████████████████████│ 100.0%
Archives   │████████████████████████████████████████│ 100.0%
Web Pages  │████████████████████████████████████████│ 100.0%
YouTube    │████████████████████████████████████████│ 100.0%
Git/GitHub │████████████████████████████████████████│ 100.0%
Git Clone  │████████████████████████████████████████│ 100.0%
Audio      │████████████████████████████████████████│ 100.0%
```

## ⚡ Performance Analysis

### Processing Speed by File Size

| File | Size | Time (ms) | Speed (KB/s) | Format |
|------|------|-----------|--------------|--------|
| US Cities (21MB JSON) | 20.9 MB | 669 | 32063 | JSON |
| Great Expectations EPUB (14MB) | 13.7 MB | 13623 | 1031 | EPUB |
| Bootstrap 5.3.2 (8.2MB) | 8.2 MB | 6073 | 1381 | ZIP |
| Shakespeare Complete Works | 5.4 MB | 728 | 7568 | TXT |
| War and Peace | 3.2 MB | 706 | 4647 | TXT |
| Count of Monte Cristo | 2.7 MB | 714 | 3813 | TXT |
| COVID-19 Global Data (1.8MB) | 1.7 MB | 627 | 2837 | CSV |
| World Countries JSON | 1.3 MB | 359 | 3800 | JSON |
| Gettysburg Address (LibriVox) | 1.3 MB | 120165 | 11 | MP3 |
| Business Report DOCX | 1.3 MB | 466 | 2751 | DOCX |
| Moby Dick | 1.2 MB | 494 | 2523 | TXT |
| Flask 3.0.0 Project | 761.3 KB | 2188 | 348 | ZIP |
| Pride and Prejudice | 754.3 KB | 374 | 2015 | TXT |
| GitHub Trending Repos | 629.2 KB | 339 | 1855 | JSON |
| World Population CSV | 526.0 KB | 671 | 783 | CSV |
| Speech Sample WAV | 525.4 KB | 22204 | 24 | WAV |
| Frankenstein EPUB | 464.9 KB | 1001 | 464 | EPUB |
| Nature Hi-Res Photo (426KB) | 425.3 KB | 1538 | 277 | JPG |
| IMDB Movies CSV | 302.5 KB | 485 | 623 | CSV |
| Nobel Prizes JSON | 227.2 KB | 319 | 713 | JSON |

## 📝 Detailed Test Results

### Text Files

| Test Name | Status | Time (ms) | Input Size | Command |
|-----------|--------|-----------|------------|---------|
| Shakespeare Complete Works | ✅ PASS | 728 | 5.4 MB | `ingestor ingest "shakespeare_complete.txt"` |
| War and Peace | ✅ PASS | 706 | 3.2 MB | `ingestor ingest "war_and_peace.txt"` |
| Count of Monte Cristo | ✅ PASS | 714 | 2.7 MB | `ingestor ingest "monte_cristo.txt"` |
| Moby Dick | ✅ PASS | 494 | 1.2 MB | `ingestor ingest "moby_dick.txt"` |
| Pride and Prejudice | ✅ PASS | 374 | 754.3 KB | `ingestor ingest "pride_prejudice.txt"` |
| Awesome Python README | ✅ PASS | 336 | 77.8 KB | `ingestor ingest "awesome_python.md"` |
| Coding Interview University | ✅ PASS | 371 | 133.5 KB | `ingestor ingest "coding_interview.md"` |
| FastAPI README | ✅ PASS | 319 | 26.0 KB | `ingestor ingest "fastapi_readme.md"` |
| TensorFlow README | ✅ PASS | 340 | 11.6 KB | `ingestor ingest "tensorflow_readme.md"` |
| React README | ✅ PASS | 341 | 5.2 KB | `ingestor ingest "react_readme.md"` |
| Python Tutorial (RST) | ✅ PASS | 365 | 18.5 KB | `ingestor ingest "python_tutorial.rst"` |
| Django Settings (Python) | ✅ PASS | 388 | 22.7 KB | `ingestor ingest "django_settings.py"` |
| Requests API (Python) | ✅ PASS | 331 | 6.3 KB | `ingestor ingest "requests_api.py"` |
| Lodash (JavaScript) | ✅ PASS | 361 | 71.3 KB | `ingestor ingest "lodash.js"` |

### Data Files

| Test Name | Status | Time (ms) | Input Size | Command |
|-----------|--------|-----------|------------|---------|
| US Cities (21MB JSON) | ✅ PASS | 669 | 20.9 MB | `ingestor ingest "us_cities.json"` |
| World Countries JSON | ✅ PASS | 359 | 1.3 MB | `ingestor ingest "countries.json"` |
| GitHub Trending Repos | ✅ PASS | 339 | 629.2 KB | `ingestor ingest "github_trending.json"` |
| Nobel Prizes JSON | ✅ PASS | 319 | 227.2 KB | `ingestor ingest "nobel_prizes.json"` |
| BBC News RSS Feed | ✅ PASS | 328 | 24.7 KB | `ingestor ingest "bbc_news.xml"` |
| NY Times RSS Feed | ✅ PASS | 407 | 46.4 KB | `ingestor ingest "nytimes.xml"` |
| Apache Commons POM | ✅ PASS | 375 | 34.2 KB | `ingestor ingest "apache_commons_pom.xml"` |
| XKCD RSS Feed | ✅ PASS | 332 | 2.4 KB | `ingestor ingest "xkcd.xml"` |
| COVID-19 Global Data (1.8MB) | ✅ PASS | 627 | 1.7 MB | `ingestor ingest "covid_global.csv"` |
| World Population CSV | ✅ PASS | 671 | 526.0 KB | `ingestor ingest "world_population.csv"` |
| IMDB Movies CSV | ✅ PASS | 485 | 302.5 KB | `ingestor ingest "movies.csv"` |
| Titanic Dataset CSV | ✅ PASS | 480 | 58.9 KB | `ingestor ingest "titanic.csv"` |
| Weather Data CSV | ✅ PASS | 526 | 66.3 KB | `ingestor ingest "weather.csv"` |
| Iris Dataset CSV | ✅ PASS | 525 | 3.8 KB | `ingestor ingest "iris.csv"` |

### Documents

| Test Name | Status | Time (ms) | Input Size | Command |
|-----------|--------|-----------|------------|---------|
| Business Report DOCX | ✅ PASS | 466 | 1.3 MB | `ingestor ingest "business_report.docx"` |
| Financial Sample XLSX | ✅ PASS | 631 | 81.5 KB | `ingestor ingest "financial_sample.xlsx"` |
| Great Expectations EPUB (14MB) | ✅ PASS | 13623 | 13.7 MB | `ingestor ingest "great_expectations.epub"` |
| Frankenstein EPUB | ✅ PASS | 1001 | 464.9 KB | `ingestor ingest "frankenstein.epub"` |
| Alice in Wonderland EPUB | ✅ PASS | 702 | 184.4 KB | `ingestor ingest "alice_wonderland.epub"` |

### Images

| Test Name | Status | Time (ms) | Input Size | Command |
|-----------|--------|-----------|------------|---------|
| Nature Hi-Res Photo (426KB) | ✅ PASS | 1538 | 425.3 KB | `ingestor ingest "nature_hires.jpg"` |
| NASA APOD Galaxy | ✅ PASS | 694 | 171.1 KB | `ingestor ingest "nasa_apod.jpg"` |
| NASA Blue Marble | ✅ PASS | 483 | 167.8 KB | `ingestor ingest "earth_nasa.jpg"` |
| Wikipedia Logo PNG | ✅ PASS | 361 | 161.6 KB | `ingestor ingest "wikipedia_logo.png"` |
| Wikipedia Logo 2 PNG | ✅ PASS | 324 | 125.9 KB | `ingestor ingest "wikipedia_logo_png.png"` |
| Python Logo PNG | ✅ PASS | 431 | 15.0 KB | `ingestor ingest "python_logo.png"` |
| GitHub Logo PNG | ✅ PASS | 419 | 7.1 KB | `ingestor ingest "github_logo.png"` |
| Chart Example PNG | ✅ PASS | 340 | 20.8 KB | `ingestor ingest "chart.png"` |

### Archives

| Test Name | Status | Time (ms) | Input Size | Command |
|-----------|--------|-----------|------------|---------|
| Bootstrap 5.3.2 (8.2MB) | ✅ PASS | 6073 | 8.2 MB | `ingestor ingest "bootstrap.zip"` |
| Flask 3.0.0 Project | ✅ PASS | 2188 | 761.3 KB | `ingestor ingest "sample_project.zip"` |

### Web Pages

| Test Name | Status | Time (ms) | Input Size | Command |
|-----------|--------|-----------|------------|---------|
| Python Tutorial Page | ✅ PASS | 3372 | 48 B | `ingestor ingest "https://docs.python.org/3/tutorial/appetite.html"` |
| HTTPBin HTML Test | ✅ PASS | 4062 | 24 B | `ingestor ingest "https://httpbin.org/html"` |
| Example.com | ✅ PASS | 3474 | 19 B | `ingestor ingest "https://example.com"` |

### YouTube

| Test Name | Status | Time (ms) | Input Size | Command |
|-----------|--------|-----------|------------|---------|
| Rick Astley - Never Gonna Give You Up | ✅ PASS | 2672 | 43 B | `ingestor ingest "https://www.youtube.com/watch?v=dQw4w9WgXcQ"` |
| Me at the zoo (First YouTube Video) | ✅ PASS | 1971 | 43 B | `ingestor ingest "https://www.youtube.com/watch?v=jNQXAC9IVRw"` |
| PSY - Gangnam Style | ✅ PASS | 2882 | 43 B | `ingestor ingest "https://www.youtube.com/watch?v=9bZkp7q19f0"` |

### Git/GitHub

| Test Name | Status | Time (ms) | Input Size | Command |
|-----------|--------|-----------|------------|---------|
| Hello-World Repository | ✅ PASS | 1855 | 38 B | `ingestor ingest "https://github.com/octocat/Hello-World"` |
| Requests README | ✅ PASS | 829 | 51 B | `ingestor ingest "https://github.com/psf/requests/blob/main/README.m..."` |
| Hello-World Directory | ✅ PASS | 6020 | 50 B | `ingestor ingest "https://github.com/octocat/Hello-World/tree/master"` |

### Git Clone

| Test Name | Status | Time (ms) | Input Size | Command |
|-----------|--------|-----------|------------|---------|
| Clone Hello-World | ✅ PASS | 1676 | 0 B | `ingestor ingest "https://github.com/octocat/Hello-World"` |
| Clone with max-files | ✅ PASS | 1582 | 0 B | `ingestor ingest "https://github.com/octocat/Hello-World"` |

### Audio

| Test Name | Status | Time (ms) | Input Size | Command |
|-----------|--------|-----------|------------|---------|
| Gettysburg Address (LibriVox) | ✅ PASS | 120165 | 1.3 MB | `ingestor ingest "librivox_sample.mp3"` |
| Speech Sample WAV | ✅ PASS | 22204 | 525.4 KB | `ingestor ingest "jfk_speech.wav"` |

## 📁 Format Support Matrix

| Format | Extension | Status | Notes |
|--------|-----------|--------|-------|
| Plain Text | TXT | ✅ Supported | Full Unicode support |
| Markdown | MD | ✅ Supported | Preserves formatting |
| reStructuredText | RST | ✅ Supported | Treated as text |
| Python | PY | ✅ Supported | Code files |
| JavaScript | JS | ✅ Supported | Code files |
| JSON | JSON | ✅ Supported | Pretty-printed output |
| XML | XML | ✅ Supported | Structured extraction |
| CSV | CSV | ✅ Supported | Table format output |
| DOCX | DOCX | ✅ Supported | Full document extraction |
| XLSX | XLSX | ✅ Supported | Multi-sheet support |
| EPUB | EPUB | ✅ Supported | Chapter extraction + images |
| PNG/JPG | PNG, JPG | ✅ Supported | Image metadata extraction |
| SVG | SVG | ✅ Supported | Vector graphics (no conversion) |
| ZIP | ZIP | ✅ Supported | Recursive extraction |
| Web Pages | HTTP(S) | ✅ Supported | Requires Playwright |
| YouTube | youtube.com | ✅ Supported | Transcripts + metadata |
| GitHub | github.com | ✅ Supported | Repos, files, directories |
| Audio | MP3, WAV | ✅ Supported | Whisper transcription |

## 🖥️ Test Environment

| Component | Value |
|-----------|-------|
| Test Samples | `/home/shazzadul/Illinois_Tech/Spring26/RA/Github/ingestor/test_report/test_samples` |
| Output Directory | `/home/shazzadul/Illinois_Tech/Spring26/RA/Github/ingestor/test_report/output/cli_test_results` |
| Test Method | CLI subprocess execution |
| Timeout | 300 seconds per test |

### Running the Tests

To reproduce these tests:

```bash
# 1. Install dependencies
uv sync --extra dev --extra all-formats

# 2. Install Playwright browsers (required for web extraction)
uv run playwright install chromium

# 3. Ensure test samples exist in test_report/test_samples/
# (Download real files from public sources)

# 4. Run the CLI test suite
uv run python test_report/run_cli_tests.py

# 5. View the generated report
cat test_report/CLI_TEST_REPORT.md
```

## ✅ Source Verification

All test files are real documents from public sources (not synthetic fixtures):

| Category | Files | Total Size | Verified |
|----------|-------|------------|----------|
| Text Files | 14 | 13.6 MB | ✅ |
| Data Files | 14 | 25.9 MB | ✅ |
| Documents | 5 | 15.7 MB | ✅ |
| Images | 8 | 1.1 MB | ✅ |
| Archives | 2 | 8.9 MB | ✅ |
| Web Pages | 3 | 91 B | ✅ |
| YouTube | 3 | 129 B | ✅ |
| Git/GitHub | 3 | 139 B | ✅ |
| Git Clone | 2 | 0 B | ✅ |
| Audio | 2 | 1.8 MB | ✅ |
| **TOTAL** | **56** | **66.9 MB** | **✅** |

## 🎯 Conclusion

🎉 **All CLI tests passed successfully!**

The Ingestor CLI tool correctly handles all tested formats using real-world data.

**Summary:**
- ✅ 56/56 tests passed (100%)
- ✅ All 10 categories working
- ✅ Total processing time: 214.7 seconds

---
*Report generated by Ingestor CLI Test Suite on 2025-12-20 09:41:46*