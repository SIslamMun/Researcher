# V1 Test 1: Undirected - LangChain

## Test Configuration

| Field | Value |
|-------|-------|
| **Prompt Config** | prompts.yaml (V1 - 179 lines) |
| **Mode** | `undirected` |
| **Query** | "What is LangChain?" |
| **Artifacts** | None |

## Command
```bash
uv run researcher research "What is LangChain?" --mode undirected -o ./test_v1/undirected_1
```

## Results

| Metric | Value |
|--------|-------|
| **Status** | ✅ Success |
| **Duration** | 314.3 seconds |
| **Report Size** | 46KB |
| **Output File** | research/research_report.md |

## Mode Behavior
**UNDIRECTED** mode performs autonomous web-first discovery:
- No user materials provided
- Agent searches the web extensively
- Gathers diverse perspectives and recent developments
- Prioritizes authoritative and recent sources

## Expected Reference Format (V1)
```
### Publications
[1] "Title" (Author et al.). Venue, Year. DOI: xxx | URL
```
