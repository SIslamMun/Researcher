# Directed Mode Test Output

## Mode Description
**DIRECTED** - Prioritize user-provided artifacts, use web search to fill gaps.

## Input

### Query
```
Analyze and explain RAG architectures
```

### Artifacts Provided
1. `https://arxiv.org/abs/2005.11401` - Original RAG paper by Lewis et al.
2. `RAG combines retrieval with generation for better factual accuracy` - Context note

### Command Used
```bash
uv run researcher research "Analyze and explain RAG architectures" \
  --mode directed \
  -a "https://arxiv.org/abs/2005.11401" \
  -a "RAG combines retrieval with generation for better factual accuracy" \
  -o ./test_output/directed
```

## Output Summary

### Duration
~292 seconds (4.9 minutes)

### Report Highlights
- **245 lines** focused architectural analysis
- Executive summary with key insights
- Covers all 3 RAG paradigms (Naive, Advanced, Modular)
- Deep dives into Self-RAG, CRAG, GraphRAG, REALM, RETRO
- Includes comparison tables (RAG vs Long Context, RAG vs Fine-Tuning)
- Implementation ecosystem section

### References Generated
| Type | Count |
|------|-------|
| Published Papers | 5 |
| Preprints (arXiv) | 2 |
| Code Repositories | 2 |
| Websites & Docs | 1 |

## Key Takeaway
Directed mode produces focused reports that build on your provided materials while filling gaps with web research. Best when you have key papers/links and want comprehensive analysis around them.
