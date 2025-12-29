# Undirected Mode Test Output

## Mode Description
**UNDIRECTED** - Web-first autonomous discovery using only the user's prompt.

## Input

### Query
```
What is RAG (Retrieval Augmented Generation)?
```

### Artifacts Provided
None - This mode relies entirely on web search.

### Command Used
```bash
uv run researcher research "What is RAG (Retrieval Augmented Generation)?" --mode undirected -o ./test_output/undirected
```

## Output Summary

### Duration
~263 seconds (4.4 minutes)

### Report Highlights
- **306 lines** comprehensive report
- Covers RAG definition, history, architecture, evolution
- Discusses Naive RAG → Advanced RAG → Modular RAG paradigms
- Deep dives into GraphRAG, Self-RAG, Hybrid Search
- Compares RAG vs Fine-tuning vs Long Context Windows

### References Generated
| Type | Count |
|------|-------|
| Published Papers | 5 |
| Preprints (arXiv) | 4 |
| Code Repositories | 3 |
| Datasets | 2 |
| Websites & Docs | 5 |
| Videos | 1 |
| Books | 1 |

## Key Takeaway
Undirected mode produces the most comprehensive reports by freely exploring the web. Best for broad topic discovery when you don't have specific materials.
