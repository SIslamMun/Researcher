# Test 9: No-Research - Jarvis HPC Framework

## Overview
No-research mode test analyzing the Jarvis framework for HPC deployment using only user-provided materials (no web search).

## Test Details

| Field | Value |
|-------|-------|
| **Test ID** | no_research_jarvis |
| **Mode** | no-research |
| **Prompt Config** | V1 (`prompts.yaml`) |
| **Duration** | 355.1 seconds (~5.9 min) |
| **Status** | ✅ Completed |

## Query
```
Analyze the Jarvis framework for HPC deployment and container orchestration. Explain 
its architecture, key components, and how it integrates with different runtime 
environments.
```

## Artifacts (5 total)

### PDFs (2)
Located in `test_artifacts/jarvis/`:

| File | Description |
|------|-------------|
| `cernuda2024jarvis.pdf` | Jarvis research paper |
| `pdsw24_wip_session2_wip1.pdf` | PDSW'24 workshop paper |

### Documentation (1)
- https://grc.iit.edu/docs/jarvis/jarvis-cd/index/

### GitHub Repositories (2)
- https://github.com/iowarp/ppi-jarvis-util
- https://github.com/iowarp/runtime-deployment

## Output Files
```
research/
├── research_report.md      # 15.8KB, 134 lines
└── research_metadata.json  # Query, timing, interaction info
```

## Command Used
```bash
uv run researcher research \
  "Analyze the Jarvis framework for HPC deployment and container orchestration. Explain its architecture, key components, and how it integrates with different runtime environments." \
  --mode no-research \
  --output test_v1/no_research_jarvis \
  -a test_artifacts/jarvis/cernuda2024jarvis.pdf \
  -a test_artifacts/jarvis/pdsw24_wip_session2_wip1.pdf \
  -a https://grc.iit.edu/docs/jarvis/jarvis-cd/index/ \
  -a https://github.com/iowarp/ppi-jarvis-util \
  -a https://github.com/iowarp/runtime-deployment
```

## Results Summary
- **Duration**: 355.1s (~5.9 min)
- **Report Size**: 15.8KB
- **Lines**: 134 lines of analysis
- **Artifact Integration**: Successfully analyzed all 5 artifacts (2 PDFs + 1 doc + 2 GitHub repos)

## Notes
- This test demonstrates no-research mode's ability to synthesize information from diverse sources
- No web search was performed - analysis based solely on provided materials
- Useful for analyzing proprietary or unpublished research materials
- Jarvis is an HPC framework for deployment and container orchestration developed at IIT
