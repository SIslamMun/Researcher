# V1 Test 4: Directed - BERT vs GPT Comparison

## Test Configuration

| Field | Value |
|-------|-------|
| **Prompt Config** | prompts.yaml (V1 - 179 lines) |
| **Mode** | `directed` |
| **Query** | "Compare BERT and GPT architectures" |
| **Artifacts** | 2 items (see below) |

## Artifacts Provided
1. **BERT Paper:** https://arxiv.org/abs/1810.04805 (BERT: Pre-training of Deep Bidirectional Transformers)
2. **GPT-3 Paper:** https://arxiv.org/abs/2005.14165 (Language Models are Few-Shot Learners)

## Command
```bash
uv run researcher research "Compare BERT and GPT architectures" --mode directed \
  -a "https://arxiv.org/abs/1810.04805" \
  -a "https://arxiv.org/abs/2005.14165" \
  -o ./test_v1/directed_2
```

## Results

| Metric | Value |
|--------|-------|
| **Status** | ✅ Success |
| **Duration** | 246.3 seconds |
| **Report Size** | 27KB |
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
```
