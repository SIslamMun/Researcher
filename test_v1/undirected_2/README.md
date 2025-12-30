# V1 Test 2: Undirected - Transformer Architecture

## Test Configuration

| Field | Value |
|-------|-------|
| **Prompt Config** | prompts.yaml (V1 - 179 lines) |
| **Mode** | `undirected` |
| **Query** | "Explain transformer architecture" |
| **Artifacts** | None |

## Command
```bash
uv run researcher research "Explain transformer architecture" --mode undirected -o ./test_v1/undirected_2
```

## Results

| Metric | Value |
|--------|-------|
| **Status** | ✅ Success |
| **Duration** | 313.3 seconds |
| **Report Size** | 49KB |
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
