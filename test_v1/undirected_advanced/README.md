# Test 8: Undirected Advanced - RAG Architectures (Web-Only)

## Overview
Advanced undirected research test using the same complex query as directed_advanced but with no artifacts - pure web discovery mode.

## Test Details

| Field | Value |
|-------|-------|
| **Test ID** | undirected_advanced |
| **Mode** | undirected |
| **Prompt Config** | V1 (`prompts.yaml`) |
| **Duration** | 386.7 seconds (~6.4 min) |
| **Status** | ✅ Completed |

## Query
```
Compare different RAG architectures and their implementations. Analyze the evolution 
from basic RAG to Self-RAG and advanced techniques. Include analysis of popular 
frameworks like LangChain and LlamaIndex.
```

## Artifacts
**None** - This is a pure web-discovery test. The agent conducts independent research using only web sources.

## Output Files
```
research/
├── research_report.md      # 60.3KB, 365 lines
└── research_metadata.json  # Query, timing, interaction info
```

## Command Used
```bash
uv run researcher research \
  "Compare different RAG architectures and their implementations. Analyze the evolution from basic RAG to Self-RAG and advanced techniques. Include analysis of popular frameworks like LangChain and LlamaIndex." \
  --mode undirected \
  --output test_v1/undirected_advanced
```

## Results Summary
- **Duration**: 386.7s (longer than directed test with same query)
- **Report Size**: 60.3KB (larger than directed_advanced's 53.9KB)
- **Lines**: 365 lines of detailed analysis

## Comparison with Test 7 (Directed Advanced)

| Metric | Undirected (Test 8) | Directed (Test 7) |
|--------|---------------------|-------------------|
| Duration | 386.7s | 347.4s |
| Report Size | 60.3KB | 53.9KB |
| Lines | 365 | 385 |
| Artifacts | 0 | 8 |

**Key Observations:**
- Undirected takes ~11% longer (39s more) than directed with same query
- Undirected produces larger report (6.4KB more) due to broader web search
- Directed has more structured content with user-provided materials as anchors
- Both modes successfully cover RAG architectures, Self-RAG, LangChain, and LlamaIndex

## Notes
- This test demonstrates undirected mode's ability to discover relevant information independently
- Useful for exploratory research when user has no starting materials
- Web-first approach finds diverse sources including papers, documentation, and tutorials
