# V1 Test 3: Directed - Vector Databases for RAG

## Test Configuration

| Field | Value |
|-------|-------|
| **Prompt Config** | prompts.yaml (V1 - 179 lines) |
| **Mode** | `directed` |
| **Query** | "Analyze vector databases for RAG" |
| **Artifacts** | 2 items (see below) |

## Artifacts Provided
1. **arXiv Paper:** https://arxiv.org/abs/2312.10997 (RAG Survey)
2. **GitHub Repo:** https://github.com/chroma-core/chroma (Chroma Vector DB)

## Command
```bash
uv run researcher research "Analyze vector databases for RAG" --mode directed \
  -a "https://arxiv.org/abs/2312.10997" \
  -a "https://github.com/chroma-core/chroma" \
  -o ./test_v1/directed_1
```

## Results

| Metric | Value |
|--------|-------|
| **Status** | ✅ Success |
| **Duration** | 291.4 seconds |
| **Report Size** | 33KB |
| **Output File** | research/research_report.md |

## Mode Behavior
**DIRECTED** mode prioritizes user-provided materials:
- Uses provided artifacts as PRIMARY sources
- Web search fills gaps and verifies claims
- Builds analysis primarily on the user's materials

## Expected Reference Format (V1)
```
### Publications
[1] "Title" (Author et al.). Venue, Year. DOI: xxx | URL

### Code & Tools
[N] Repository Name - Description. URL
```
