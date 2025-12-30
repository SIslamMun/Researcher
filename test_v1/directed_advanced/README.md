# Test 7: Directed Advanced - Mixed Artifact Types

## Overview
Advanced directed research test with diverse artifact types: downloaded PDFs, GitHub repositories, and blog posts.

## Test Details

| Field | Value |
|-------|-------|
| **Test ID** | directed_advanced |
| **Mode** | directed |
| **Prompt Config** | V1 (`prompts.yaml`) |
| **Duration** | 347.4 seconds (~5.8 min) |
| **Status** | ✅ Completed |

## Query
```
Compare different RAG architectures and their implementations. Analyze the evolution 
from basic RAG to Self-RAG and advanced techniques. Include analysis of popular 
frameworks like LangChain and LlamaIndex.
```

## Artifacts (8 total)

### Downloaded PDFs (4)
Located in `test_artifacts/papers/`:

| File | Source | Size |
|------|--------|------|
| `rag_original.pdf` | arXiv:2005.11401 - Original RAG paper | 865KB |
| `self_rag.pdf` | arXiv:2310.11511 - Self-RAG paper | 1.4MB |
| `rag_survey.pdf` | arXiv:2312.10997 - RAG Survey | 1.6MB |
| `attention_is_all_you_need.pdf` | arXiv:1706.03762 - Transformer paper | 2.2MB |

### GitHub Repositories (2)
- https://github.com/langchain-ai/langchain
- https://github.com/run-llama/llama_index

### Blog Posts (2)
- https://www.pinecone.io/learn/retrieval-augmented-generation/
- https://lilianweng.github.io/posts/2023-06-23-agent/

## Output Files
```
research/
├── research_report.md      # 53.9KB, 385 lines
└── research_metadata.json  # Query, timing, interaction info
```

## Command Used
```bash
uv run researcher research \
  "Compare different RAG architectures and their implementations. Analyze the evolution from basic RAG to Self-RAG and advanced techniques. Include analysis of popular frameworks like LangChain and LlamaIndex." \
  --mode directed \
  --output test_v1/directed_advanced \
  -a test_artifacts/papers/rag_original.pdf \
  -a test_artifacts/papers/self_rag.pdf \
  -a test_artifacts/papers/rag_survey.pdf \
  -a test_artifacts/papers/attention_is_all_you_need.pdf \
  -a https://github.com/langchain-ai/langchain \
  -a https://github.com/run-llama/llama_index \
  -a https://www.pinecone.io/learn/retrieval-augmented-generation/ \
  -a https://lilianweng.github.io/posts/2023-06-23-agent/
```

## Results Summary
- **Duration**: 347.4s (faster than undirected tests, despite more artifacts)
- **Report Size**: 53.9KB (comprehensive coverage)
- **Lines**: 385 lines of detailed analysis
- **Artifact Integration**: Successfully incorporated all 8 artifacts (4 PDFs + 2 GitHub + 2 blogs)

## Notes
- This test demonstrates the tool's ability to handle mixed artifact types
- PDFs were downloaded from arXiv and stored locally
- GitHub repos and blog posts were provided as URLs
- The directed mode prioritizes user materials while filling gaps from web research
