# Comparative Analysis of RAG Architectures: From Foundations to Self-Reflective Systems

**Key Points**
*   **Foundational Shift:** Retrieval-Augmented Generation (RAG) has evolved from a static "retrieve-then-generate" pipeline (Lewis et al., 2020) into dynamic, self-correcting architectures like Self-RAG and Modular RAG.
*   **Self-RAG Innovation:** Unlike standard RAG, Self-RAG (Asai et al., 2023) trains a single language model to output special "reflection tokens" that control when to retrieve and how to critique its own output, enabling adaptive retrieval and higher factual accuracy.
*   **Framework Divergence:** LangChain and LlamaIndex represent two distinct philosophies in RAG implementation; LangChain focuses on composable, multi-step orchestration (chains/agents), while LlamaIndex prioritizes data ingestion, indexing strategies, and optimized query engines.
*   **Advanced Optimization:** Modern RAG systems employ sophisticated techniques across the pipeline, including pre-retrieval query transformation (HyDE, Multi-Query), retrieval optimization (Hybrid Search), and post-retrieval refinement (Reranking, Contextual Compression).

The field of Retrieval-Augmented Generation (RAG) has rapidly matured from a novel method to reduce hallucinations into a complex ecosystem of architectures designed for reliability and reasoning. Initially proposed to bridge the gap between parametric memory (model weights) and non-parametric memory (external databases), RAG has evolved to address limitations in precision, recall, and context integration.

This report analyzes the architectural progression of RAG, contrasting the original "Naive" implementations with sophisticated "Advanced" and "Modular" paradigms. It provides a deep technical examination of Self-RAG, a framework that integrates self-reflection directly into the generation process. Furthermore, it evaluates the practical implementation of these architectures using the two dominant open-source frameworks, LangChain and LlamaIndex, highlighting their respective strengths in orchestration versus data management.

---

## 1. Foundational RAG Architecture

The concept of Retrieval-Augmented Generation was formalized by Lewis et al. in their seminal 2020 paper, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." This work established the baseline architecture that combines pre-trained sequence-to-sequence (seq2seq) models with dense vector retrieval [cite: 1, 2].

### 1.1 The Parametric and Non-Parametric Memory Model
The core innovation of the original RAG framework was the hybrid usage of memory:
*   **Parametric Memory:** The pre-trained weights of a generator model (specifically BART in the original implementation), which stores implicit knowledge and linguistic patterns [cite: 2, 3].
*   **Non-Parametric Memory:** A dense vector index of an external corpus (e.g., Wikipedia), accessed via a pre-trained neural retriever (Dense Passage Retriever or DPR) [cite: 2, 3].

In this architecture, the input sequence $x$ is used to retrieve latent documents $z$, which are then used as additional context for generating the target sequence $y$. The probability of generating a sequence is marginalized over the retrieved documents [cite: 3].

### 1.2 RAG-Sequence vs. RAG-Token
Lewis et al. introduced two distinct formulations for how retrieved content influences generation, a distinction that remains relevant in modern implementations:

1.  **RAG-Sequence:** The model uses the *same* retrieved document to generate the *entire* output sequence. It calculates the probability of the sequence for each retrieved document and then marginalizes. This assumes a single document contains sufficient context for the full answer [cite: 2, 3].
    \[
    P_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-k}(p(\cdot|x))} p_{\eta}(z|x) \prod_{i} p_{\theta}(y_i | x, z, y_{1:i-1})
    \]
2.  **RAG-Token:** The model can attend to *different* retrieved documents for *each token* generated. This allows the generator to synthesize information from multiple sources within a single response, offering higher flexibility for complex queries requiring multi-hop reasoning [cite: 2, 3].

### 1.3 Limitations of the Foundational Model
While the original RAG set state-of-the-art benchmarks on Open Domain QA, it exhibited limitations that drove subsequent evolution:
*   **Indiscriminate Retrieval:** The model retrieves a fixed number of documents ($k$) regardless of whether the query actually requires external knowledge [cite: 4, 5].
*   **Passive Generation:** The generator is conditioned on retrieved documents but lacks a mechanism to explicitly validate their relevance or the factual support of its own output [cite: 4, 6].
*   **Flat Interaction:** The interaction between retrieval and generation is a one-way street; the generator cannot request more information or refine the query based on intermediate reasoning [cite: 7].

---

## 2. Evolutionary Taxonomy: Naive, Advanced, and Modular RAG

As RAG systems moved from research to production, the architecture evolved. Gao et al. (2023) provide a comprehensive survey categorizing this evolution into three paradigms: Naive, Advanced, and Modular RAG [cite: 8, 9].

### 2.1 Naive RAG
Naive RAG represents the earliest and simplest adoption of the technique, often characterized by a "Retrieve-Read" process.
*   **Process:** Indexing (chunking documents -> embeddings), Retrieval (top-k similarity search), and Generation (prompt stuffing with context) [cite: 9, 10].
*   **Challenges:** This approach suffers from low precision (retrieving irrelevant chunks), low recall (missing relevant chunks), and hallucination (the model generating unsupported answers despite context) [cite: 10]. It lacks pre-processing of queries or post-processing of retrieved results.

### 2.2 Advanced RAG
Advanced RAG introduces optimizations at specific stages of the pipeline to address Naive RAG's shortcomings.
*   **Pre-Retrieval:** Techniques like query rewriting, query expansion, and hypothetical document embeddings (HyDE) to improve the semantic match between query and documents [cite: 10, 11].
*   **Post-Retrieval:** Reranking retrieved documents to prioritize the most relevant chunks before they reach the context window. This often involves cross-encoders or specialized reranking models (e.g., Cohere, BAAI) [cite: 12, 13].
*   **Context Compression:** Reducing noise by summarizing retrieved documents or extracting only relevant snippets to maximize the utility of the LLM's context window [cite: 10, 14].

### 2.3 Modular RAG
Modular RAG breaks the rigid "retrieve-then-generate" structure, allowing for more flexible, dynamic workflows.
*   **Componentization:** Modules such as "Search," "Memory," "Fusion," and "Routing" can be orchestrated arbitrarily [cite: 8, 9].
*   **Dynamic Routing:** The system can decide to search a vector store, query a knowledge graph, or use a web search tool based on the query type [cite: 9].
*   **Iterative Processes:** This paradigm supports recursive retrieval and multi-hop reasoning, where the output of one step influences the retrieval of the next [cite: 9, 15].

---

## 3. Self-RAG: Architecture and Implementation

Self-Reflective Retrieval-Augmented Generation (Self-RAG), introduced by Asai et al. (2023), represents a paradigm shift from "passive" to "active" RAG. Instead of treating retrieval and generation as separate, rigid steps, Self-RAG trains a single Language Model (LM) to control the entire process via special tokens [cite: 4, 5, 16].

### 3.1 Core Concept: Reflection Tokens
The defining feature of Self-RAG is the introduction of **reflection tokens** into the model's vocabulary. These tokens allow the model to introspect and control its behavior during inference [cite: 5, 17, 18].

| Token Type | Token Name | Function | Output Values |
| :--- | :--- | :--- | :--- |
| **Retrieval** | `Retrieve` | Decides if external knowledge is needed. | `[Yes, No, Continue]` |
| **Critique** | `IsRel` | Checks if a retrieved passage is relevant to the query. | `[Relevant, Irrelevant]` |
| **Critique** | `IsSup` | Checks if the generated text is supported by the passage. | `[Fully supported, Partially supported, No support]` |
| **Critique** | `IsUse` | Checks if the generated text is useful/helpful for the query. | `[cite: 4, 5, 16, 19, 20]` |

### 3.2 Training Methodology
Self-RAG does not rely on a separate critic model at inference time. Instead, it internalizes the critic's capabilities through a two-stage training process [cite: 5, 16, 18]:

1.  **Critic Data Creation:** A separate "Critic" model (e.g., GPT-4) is used to annotate a large corpus of instruction-output pairs. It inserts reflection tokens into the text, marking where retrieval should happen and evaluating the quality of the text [cite: 5, 17].
2.  **Generator Training:** The generator model (e.g., Llama 2) is trained on this augmented corpus using a standard next-token prediction objective. This teaches the model to predict reflection tokens as part of its natural generation process [cite: 16, 18].

### 3.3 Inference Process: Adaptive Retrieval and Decoding
At inference time, Self-RAG operates dynamically:
1.  **Adaptive Retrieval:** For every segment, the model predicts a `Retrieve` token. If the probability of `Retrieve=Yes` exceeds a threshold, it triggers the retrieval module [cite: 4, 6]. This contrasts with standard RAG, which retrieves indiscriminately.
2.  **Parallel Generation & Critique:** If retrieval occurs, the model processes multiple retrieved passages in parallel. It generates a response for each passage, accompanied by `IsRel` and `IsSup` tokens [cite: 16, 18].
3.  **Tree Decoding:** The system ranks the parallel generations based on their reflection token scores (e.g., prioritizing `IsRel=Relevant` and `IsSup=Fully supported`) and selects the best segment to append to the final output [cite: 21, 22].

### 3.4 Performance and Advantages
Self-RAG has demonstrated significant improvements over standard RAG and even proprietary models like ChatGPT on tasks involving factuality and long-form generation [cite: 5, 16].
*   **Reduced Hallucination:** By explicitly checking `IsSup`, the model rejects unsupported claims [cite: 23, 24].
*   **Versatility:** The model can be tuned at inference time (by adjusting token probabilities) to prioritize factual accuracy (high retrieval) or creativity (low retrieval) [cite: 5, 16].

---

## 4. Advanced RAG Techniques

Modern RAG implementations employ a suite of techniques to optimize performance at the pre-retrieval, retrieval, and post-retrieval stages.

### 4.1 Pre-Retrieval Optimization
These techniques aim to refine the user's query to better match the document embeddings.
*   **Query Expansion / Multi-Query:** The system generates multiple variations of the user's query (e.g., using an LLM) to capture different perspectives. This increases the likelihood of retrieving relevant documents that might use different terminology [cite: 14, 25].
*   **HyDE (Hypothetical Document Embeddings):** An LLM generates a hypothetical answer to the query. This hypothetical document is then embedded and used for retrieval. This often yields better results because the hypothetical answer is semantically closer to the *actual* documents than the raw question is [cite: 11, 26, 27].

### 4.2 Retrieval Optimization
*   **Hybrid Search:** Combines dense vector search (semantic similarity) with sparse keyword search (BM25). This addresses the limitation of vector search in handling exact matches for acronyms, product IDs, or specific entities [cite: 12, 13, 28].
*   **Fine-tuned Embeddings:** Training the embedding model on domain-specific data to better represent the semantic space of the target corpus [cite: 29, 30].

### 4.3 Post-Retrieval Optimization
*   **Reranking:** A high-precision cross-encoder model re-scores the top-k documents retrieved by the vector store. Rerankers are computationally expensive but significantly improve the relevance of the context fed to the LLM [cite: 12, 13, 31].
*   **Contextual Compression:** Instead of passing full documents, an LLM or specialized module extracts only the relevant sentences or propositions from the retrieved chunks, reducing noise and token usage [cite: 14, 32].

---

## 5. Framework Implementation Analysis: LangChain vs. LlamaIndex

The implementation of these architectures relies heavily on two primary frameworks: **LangChain** and **LlamaIndex**. While they share overlapping features, their core philosophies and architectural strengths differ significantly.

### 5.1 Core Philosophies
*   **LangChain:** Positioned as a general-purpose orchestration framework. It excels at "chaining" components (prompts, models, tools) and managing complex, multi-step workflows. It is highly modular and flexible, making it ideal for Agentic RAG and complex logic [cite: 33, 34, 35].
*   **LlamaIndex (formerly GPT Index):** Positioned as a "data framework" for LLMs. Its primary focus is on efficient data ingestion, structuring, indexing, and retrieval. It treats data as a first-class citizen, offering optimized "Query Engines" that abstract away much of the retrieval complexity [cite: 34, 35, 36, 37, 38].

### 5.2 Architecture Comparison

| Feature | LangChain | LlamaIndex |
| :--- | :--- | :--- |
| **Primary Abstraction** | **Chains & Runnables (LCEL):** Composable units of logic. | **Indices & Engines:** `VectorStoreIndex`, `QueryEngine`. |
| **Retrieval Focus** | **Retriever Interface:** A generic interface for fetching documents. Requires manual composition for advanced patterns. | **Deep Indexing:** Specialized indices (Tree, Keyword, Vector) and composable retrievers optimized for hierarchy. |
| **Agent Implementation** | **LangGraph:** A stateful, graph-based orchestration tool for building complex agents with loops and memory [cite: 34, 39]. | **Agentic RAG:** Built-in agents focused on data reasoning, routing, and multi-document synthesis [cite: 40, 41]. |
| **Data Ingestion** | **Document Loaders:** Extensive library of loaders, but often requires manual chunking/splitting configuration. | **Data Connectors (LlamaHub):** Streamlined ingestion with sophisticated node parsers (e.g., hierarchical nodes) [cite: 33]. |

### 5.3 Implementation of Advanced Techniques

#### LangChain Implementation
LangChain uses a composable approach. For example, implementing **Multi-Query Retrieval** involves wrapping a base retriever:
```python
# LangChain Multi-Query Example [cite: 14, 25]
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
```
**Contextual Compression** is implemented as a pipeline step:
```python
# LangChain Contextual Compression [cite: 14, 31]
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
```

#### LlamaIndex Implementation
LlamaIndex often encapsulates these patterns into specialized classes or index structures.
**Sentence Window Retrieval:** This technique retrieves a single sentence but provides the surrounding sentences as context to the LLM. LlamaIndex handles this via metadata and node relationships [cite: 13, 40, 42].
```python
# LlamaIndex Sentence Window Logic [cite: 43]
# Uses a specific NodeParser to create window metadata during indexing
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)
```
**Auto-Merging (Parent Document) Retrieval:** Retrieves small chunks for search precision but feeds the larger "parent" chunk to the LLM if enough children are retrieved. This is natively supported via hierarchical node parsers [cite: 40, 41, 42].

### 5.4 Self-RAG Implementation
Implementing Self-RAG requires different approaches in each framework:
*   **LangChain:** Typically implemented using **LangGraph**. The "flow" of Self-RAG (Retrieve -> Grade -> Generate -> Check Support) is modeled as a state graph. Nodes represent actions (e.g., `grade_documents`), and edges represent conditional logic based on reflection tokens (or their emulated equivalents via prompting) [cite: 6, 39, 44, 45].
*   **LlamaIndex:** Can implement Self-RAG concepts using custom `QueryPipelines` or agentic flows, but LangChain's LangGraph is currently more frequently cited for explicit state-machine implementations of Self-RAG logic [cite: 39, 46].

---

## 6. Comparative Analysis: Self-RAG vs. Corrective RAG (CRAG)

While Self-RAG internalizes evaluation, **Corrective RAG (CRAG)** takes a modular approach to reliability.

### 6.1 Corrective RAG (CRAG)
CRAG introduces a lightweight **retrieval evaluator** that runs *after* retrieval but *before* generation [cite: 47, 48, 49].
*   **Mechanism:** The evaluator classifies retrieved documents into three categories:
    1.  **Correct:** Use the documents for generation.
    2.  **Incorrect:** Discard documents and fall back to web search.
    3.  **Ambiguous:** Combine internal documents with web search results [cite: 47, 48].
*   **Key Difference:** CRAG focuses on fixing the *retrieval* quality (often using web search as a corrective measure), whereas Self-RAG focuses on the *generation* quality and self-consistency [cite: 4, 47].

### 6.2 Comparison Table

| Feature | Self-RAG | Corrective RAG (CRAG) |
| :--- | :--- | :--- |
| **Core Mechanism** | Reflection Tokens (Internal) | Retrieval Evaluator (External Module) |
| **Training Requirement** | Requires fine-tuning generator on reflection data. | Requires training a lightweight evaluator (e.g., BERT-based). |
| **Retrieval Strategy** | Adaptive (can skip or retrieve multiple times). | Corrective (filters/supplements with web search). |
| **Primary Goal** | Holistic quality (relevance, support, utility). | Robustness against poor retrieval. |
| **Inference Cost** | Higher (parallel generation & tree decoding). | Moderate (evaluator overhead + potential web search). |

---

## 7. Conclusion

The evolution of RAG architectures reflects a shift from simple information fetching to complex cognitive processing.
*   **Naive RAG** served as the proof of concept but failed in complex, real-world scenarios due to noise and hallucination.
*   **Advanced RAG** patched these holes with better engineering (reranking, hybrid search), which are now standard in production pipelines using frameworks like LlamaIndex.
*   **Self-RAG** represents the future of "agentic" models that internalize the RAG process, treating retrieval as a tool to be used judiciously rather than a mandatory step.

For practitioners, the choice of framework and architecture depends on the use case. **LlamaIndex** is superior for building high-performance retrieval pipelines over complex data (Advanced RAG), while **LangChain (and LangGraph)** provides the necessary control flow for implementing custom cognitive architectures like Self-RAG and CRAG. As models become more capable, we expect a convergence where "Self-RAG" capabilities become native to foundational models, making external orchestration layers thinner but more strategic.

---

## References

### Publications
[cite: 4] "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al.). arXiv:2310.11511, 2023. https://arxiv.org/abs/2310.11511
[cite: 20] "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al.). ICLR, 2024. https://openreview.net/forum?id=hSyW5go0v8
[cite: 19] "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al.). Semantic Scholar, 2023. https://www.semanticscholar.org/paper/Self-RAG%3A-Learning-to-Retrieve%2C-Generate%2C-and-Asai-Wu/ddbd8fe782ac98e9c64dd98710687a962195dd9b
[cite: 5] "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al.). ICLR Proceedings, 2024. https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf
[cite: 16] "Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al.). arXiv HTML, 2023. https://arxiv.org/html/2310.11511v1
[cite: 10] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). Prompting Guide, 2023. https://www.promptingguide.ai/research/rag
[cite: 8] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). arXiv:2312.10997, 2023. https://arxiv.org/abs/2312.10997
[cite: 50] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). Semantic Scholar, 2023. https://www.semanticscholar.org/paper/Retrieval-Augmented-Generation-for-Large-Language-A-Gao-Xiong/46f9f7b8f88f72e12cbdb21e3311f995eb6e65c5
[cite: 26] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). BAAI, 2023. https://simg.baai.ac.cn/paperfile/25a43194-c74c-4cd3-b60f-0a1f27f8b8af.pdf
[cite: 27] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). Luke Robbins Substack, 2024. https://lukerobbins.substack.com/p/retrieval-augmented-generation-for-624
[cite: 51] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.). Leonie Monigatti Blog, 2023. https://www.leoniemonigatti.com/blog/retrieval-augmented-generation-langchain.html
[cite: 52] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.). Latenode Blog, 2025. https://latenode.com/blog/ai-frameworks-technical-infrastructure/rag-retrieval-augmented-generation/rag-lewis-2020-paper-understanding-original-retrieval-augmented-generation-research
[cite: 1] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.). Semantic Scholar, 2020. https://www.semanticscholar.org/paper/Retrieval-Augmented-Generation-for-NLP-Tasks-Lewis-Perez/659bf9ce7175e1ec266ff54359e2bd76e0b7ff31
[cite: 2] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.). arXiv:2005.11401, 2020. https://arxiv.org/abs/2005.11401
[cite: 3] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.). NeurIPS Proceedings, 2020. https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[cite: 53] "LLM Powered Autonomous Agents" (Lilian Weng). Lil'Log, 2023. https://lilianweng.github.io/posts/2023-06-23-agent/
[cite: 54] "Retrieval Augmented Generation (RAG)" (Unknown). IJSRA, 2025. https://journalijsra.com/sites/default/files/fulltext_pdf/IJSRA-2025-2156.pdf
[cite: 8] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). arXiv:2312.10997, 2023. https://arxiv.org/abs/2312.10997
[cite: 9] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). arXiv PDF, 2024. https://arxiv.org/pdf/2312.10997
[cite: 29] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). Scribd, 2024. https://www.scribd.com/document/703925361/Retrieval-Augmented-Generation-for-Large-Language-Models-A-Survey
[cite: 7] "Standard vs Self vs Agentic RAG" (Unknown). BI Group, 2025. https://www.bigroup.com.au/standard-self-agentic-rag/
[cite: 23] "Self-RAG: Retrieval Augmented Generation" (Unknown). GeeksforGeeks, 2025. https://www.geeksforgeeks.org/artificial-intelligence/self-rag-retrieval-augmented-generation/
[cite: 55] "Training AI: A Comprehensive Guide to RAG Implementations" (Unknown). The Blue Owls, 2025. https://theblueowls.com/blog/training-ai-a-comprehensive-guide-to-rag-implementations/
[cite: 6] "Self-RAG" (Unknown). Analytics Vidhya, 2025. https://www.analyticsvidhya.com/blog/2025/01/self-rag/
[cite: 24] "The 2025 Guide to Retrieval-Augmented Generation (RAG)" (Unknown). Eden AI, 2025. https://www.edenai.co/post/the-2025-guide-to-retrieval-augmented-generation-rag
[cite: 10] "Advanced RAG Techniques" (Unknown). Prompting Guide, 2025. https://www.promptingguide.ai/research/rag
[cite: 56] "Advanced RAG Techniques" (Unknown). Telus Digital, 2025. https://www.telusdigital.com/insights/data-and-ai/resource/advanced-rag-techniques
[cite: 11] "15 Advanced RAG Techniques" (Unknown). Telus Digital PDF, 2025. https://assets.ctfassets.net/3viuren4us1n/3CDjWG3R6KCvkYjQRtrVEW/f1620e4fa3b024d71f6029abbcbfcf46/15_Advanced_RAG_Techniques.pdf
[cite: 13] "Advanced RAG Techniques with LlamaIndex" (Leonie Monigatti). Blog, 2024. https://www.leoniemonigatti.com/blog/advanced-rag-techniques-llamaindex.html
[cite: 28] "Advanced RAG Techniques" (Unknown). Neo4j Blog, 2025. https://neo4j.com/blog/genai/advanced-rag-techniques/
[cite: 17] "Self-RAG: Self-Reflective Retrieval-Augmented Generation" (Samia Sahin). Medium, 2024. https://medium.com/@sahin.samia/self-rag-self-reflective-retrieval-augmented-generation-the-game-changer-in-factual-ai-dd32e59e3ff9
[cite: 57] "Self-RAG" (Asai et al.). ACL Anthology, 2024. https://aclanthology.org/2024.findings-emnlp.500.pdf
[cite: 58] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.). MDPI, 2025. https://www.mdpi.com/2504-2289/9/12/320
[cite: 16] "Self-rag: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al.). arXiv HTML, 2023. https://arxiv.org/html/2310.11511v1
[cite: 18] "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al.). arXiv PDF, 2023. https://arxiv.org/pdf/2310.11511
[cite: 15] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). Kronika, 2024. https://kronika.ac/wp-content/uploads/5-KKJ2327.pdf
[cite: 10] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). Prompting Guide, 2025. https://www.promptingguide.ai/research/rag
[cite: 59] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). IJRPR, 2025. https://ijrpr.com/uploads/V6ISSUE11/IJRPR55811.pdf
[cite: 60] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). Hugging Face, 2023. https://huggingface.co/papers/2312.10997
[cite: 8] "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al.). arXiv, 2023. https://arxiv.org/abs/2312.10997
[cite: 52] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.). Latenode Blog, 2025. https://latenode.com/blog/ai-frameworks-technical-infrastructure/rag-retrieval-augmented-generation/rag-lewis-2020-paper-understanding-original-retrieval-augmented-generation-research
[cite: 3] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.). NeurIPS PDF, 2020. https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[cite: 61] "NumByNum :: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Aria Lee). Medium, 2023. https://medium.com/@AriaLeeNotAriel/numbynum-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-lewis-et-al-df93a0f4c8f0
[cite: 62] "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Patrick Lewis). Slideshare, 2023. https://www.slideshare.net/slideshow/retrievalaugmented-generation-for-knowledgeintensive-nlp-taskspdf-258126877/258126877
[cite: 63] "A Practitioner's Guide to Retrieval" (Cameron R. Wolfe). Substack, 2024. https://cameronrwolfe.substack.com/p/a-practitioners-guide-to-retrieval
[cite: 40] "Advanced RAG Techniques: An Illustrated Overview" (Pelayo Arbues). Blog, 2025. https://www.pelayoarbues.com/literature-notes/Articles/Advanced-RAG-Techniques-An-Illustrated-Overview
[cite: 41] "Advanced RAG Techniques: An Illustrated Overview" (Unknown). Towards AI, 2023. https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6
[cite: 30] "The Ultimate Guide to Understanding Advanced Retrieval Augmented Generation Methodologies" (Unknown). PurpleScape, 2024. https://purplescape.com/the-ultimate-guide-to-understanding-advanced-retrieval-augmented-generation-methodologies/
[cite: 42] "17 Advanced RAG Techniques to Turn Your RAG App Prototype into a Production-Ready Solution" (Unknown). Medium, 2024. https://medium.com/data-science/17-advanced-rag-techniques-to-turn-your-rag-app-prototype-into-a-production-ready-solution-5a048e36cdc8
[cite: 43] "Advance Retrieval Techniques in RAG Part 03: Sentence Window Retrieval" (Unknown). GoPubby, 2024. https://ai.gopubby.com/advance-retrieval-techniques-in-rag-part-03-sentence-window-retrieval-9f246cffa07b
[cite: 14] "Advanced Retrieval Techniques in LangChain to Improve the Efficiency of RAG Systems" (Abhirag Kulkarni). Medium, 2024. https://medium.com/@abhiragkulkarni12/advanced-retrieval-techniques-in-langchain-to-improve-the-efficiency-of-rag-systems-32b88d78383d
[cite: 25] "RAG and LangChain Basics" (Priya Selvaraj). Dev.to, 2025. https://dev.to/priyaselvaraj11/rag-and-langchain-basics-3n0h
[cite: 31] "Advanced RAG: Retrieval, Reranking, Hierarchical etc. in Databricks" (Unknown). Databricks Community, 2025. https://community.databricks.com/t5/generative-ai/advanced-rag-retrieval-reranking-hierarchical-etc-in-databricks/td-p/104462
[cite: 47] "Corrective RAG (CRAG)" (Unknown). Kore.ai Blog, 2024. https://www.kore.ai/blog/corrective-rag-crag
[cite: 4] "Self-RAG" (Unknown). ProjectPro, 2025. https://www.projectpro.io/article/self-rag/1176
[cite: 48] "Advanced RAG: Comparing GraphRAG, Corrective RAG, and Self-RAG" (Unknown). JavaScript Plain English, 2025. https://javascript.plainenglish.io/advanced-rag-comparing-graphrag-corrective-rag-and-self-rag-e633cbaf5bf7
[cite: 49] "Advanced RAG Techniques" (Unknown). Pinecone, 2024. https://www.pinecone.io/learn/advanced-rag-techniques/
[cite: 64] "Understanding Different RAG Techniques" (Unknown). The Hack Weekly, 2024. https://medium.com/the-hack-weekly-ai-tech-community/understanding-different-rag-techniques-0186ea5b9a13
[cite: 6] "Self-RAG" (Unknown). Analytics Vidhya, 2025. https://www.analyticsvidhya.com/blog/2025/01/self-rag/
[cite: 39] "Agentic RAG with LangGraph" (Unknown). LangChain Blog, 2024. https://blog.langchain.com/agentic-rag-with-langgraph/
[cite: 65] "How to Build a RAG System with a Self-Querying Retriever in LangChain" (Unknown). Medium, 2024. https://medium.com/data-science/how-to-build-a-rag-system-with-a-self-querying-retriever-in-langchain-16b4fa23e9ad

### Code & Tools
[cite: 33] langchain - Framework for developing applications powered by language models. https://github.com/langchain-ai/langchain
[cite: 36] "LlamaIndex vs LangChain" - IBM comparison of RAG frameworks. https://www.ibm.com/think/topics/llamaindex-vs-langchain
[cite: 34] "LangChain vs LlamaIndex 2025" - Latenode comparison article. https://latenode.com/blog/platform-comparisons-alternatives/automation-platform-comparisons/langchain-vs-llamaindex-2025-complete-rag-framework-comparison
[cite: 66] "LlamaIndex vs LangChain" - n8n comparison article. https://blog.n8n.io/llamaindex-vs-langchain/
[cite: 37] "Top LangChain Alternatives" - Scrapfly blog post. https://scrapfly.io/blog/posts/top-langchain-alternatives
[cite: 12] "Retrieval-Augmented Generation" - Pinecone learning center article. https://www.pinecone.io/learn/retrieval-augmented-generation/
[cite: 33] "LangChain vs LlamaIndex" - Medium comparison article. https://medium.com/@tam.tamanna18/langchain-vs-llamaindex-a-comprehensive-comparison-for-retrieval-augmented-generation-rag-0adc119363fe
[cite: 67] "LangChain vs LlamaIndex RAG" - YouTube video tutorial. https://www.youtube.com/watch?v=xEgUC4bd_qI
[cite: 34] "LangChain vs LlamaIndex 2025" - Latenode comparison article. https://latenode.com/blog/platform-comparisons-alternatives/automation-platform-comparisons/langchain-vs-llamaindex-2025-complete-rag-framework-comparison
[cite: 68] "Comparing LangChain and LlamaIndex" - Blog post. https://blog.myli.page/comparing-langchain-and-llamaindex-with-4-tasks-2970140edf33
[cite: 69] "LangChain vs LlamaIndex" - Medium article. https://medium.com/innova-technology/langchain-vs-llamaindex-designing-rag-and-choosing-the-right-framework-for-your-project-e1db8c1a32be
[cite: 34] "LangChain vs LlamaIndex 2025" - Latenode comparison article. https://latenode.com/blog/platform-comparisons-alternatives/automation-platform-comparisons/langchain-vs-llamaindex-2025-complete-rag-framework-comparison
[cite: 66] "LlamaIndex vs LangChain" - n8n comparison article. https://blog.n8n.io/llamaindex-vs-langchain/
[cite: 35] "LangChain vs LlamaIndex 2025" - Sider.ai blog post. https://sider.ai/blog/ai-tools/langchain-vs-llamaindex-which-rag-framework-wins-in-2025
[cite: 70] "LlamaIndex vs LangChain RAG" - Statsig perspectives article. https://www.statsig.com/perspectives/llamaindex-vs-langchain-rag
[cite: 38] llama_index - Data framework for LLM applications. https://github.com/run-llama/llama_index
[cite: 32] "Advanced RAG Techniques" - YouTube video tutorial. https://www.youtube.com/watch?v=KQjZ68mToWo
[cite: 71] "Advanced RAG Code" - GitHub repository notebook. https://github.com/Coding-Crashkurse/Advanced-RAG/blob/main/code.ipynb
[cite: 44] Self-RAG-Systems - GitHub repository for Self-RAG implementation. https://github.com/Gihan007/Self-RAG-Systems
[cite: 45] "LangGraph Self-RAG" - LangChain tutorial. https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/
[cite: 72] self-rag - Official Self-RAG repository. https://github.com/AkariAsai/self-rag
[cite: 21] "Self-RAG Project Page" - Official project website. https://selfrag.github.io/
[cite: 22] "selfrag_llama2_7b" - Hugging Face model card. https://huggingface.co/selfrag/selfrag_llama2_7b
[cite: 73] "RAG VIII: Self-RAG" - Medium article. https://medium.com/thedeephub/rag-viii-self-rag-f07dcd3835ca
[cite: 46] "Advanced RAG with LangGraph" - YouTube video. https://www.youtube.com/watch?v=NZbgduKl9Zk

### Documentation & Websites
[cite: 74] "LangChain Repository" - GitHub. https://github.com/langchain-ai/langchain
[cite: 53] "LLM Powered Autonomous Agents" - Lil'Log. https://lilianweng.github.io/posts/2023-06-23-agent/
[cite: 12] "Retrieval-Augmented Generation" - Pinecone. https://www.pinecone.io/learn/retrieval-augmented-generation/

**Sources:**
1. [semanticscholar.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFh7GPp_SMEc37azzoMVVTazxTr556lYp0GPciS-ylePWx3b7cj7S7aC3ggGFBjZcMRjB90IxkaSzWQWIUuPvdRyiQPcAZI6vwlbR5jp-euS3JLK8V--9iBhVFb4RwIvRBOjdGECurrkU8Qm7k6kv7z8fk6bh3VgO--E0NhDuQcDztq3UN-x1U288iep2wjJ4p_iK20IXbjypKwnTp_c1z0psCVdlIEX-BpbCb5wmi_ipJPIREIvxk-fKWwGCs=)
2. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEbwMZIV702452-R9FBWiT5U01ngAKLkEC3qbWCd5pKgVr5q2ZR1X9aLCe5pyqStkOLfpwkK0J_nuvgx4HjgZSNNWEAoC4ADG4hP3Og1GpvnopOrXue)
3. [neurips.cc](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEnvArLjQ9E-7ppvVp4uSMwXCXczfy8F5dWGwKespA4fldwd3tHJN8ZMS7LiwC8Q2hj2j6VtQq3uxjwqoraXW7l4TRJfKWm2H4HRYcqSUEBYNA44pCCHEaZ3RYJERdDrBrVfYpSt_Lyxre1mrXPD-oFvYKyCsaBgNiEcKvylCfc6ac9vDDeq_Yaa9sQisDe)
4. [projectpro.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHYFrJg34uz8-DmErJzrAssqh4v7oRCxHxAGICMHVm7SCPGqlPGu6dJhVN8kzDvgdAlj1ZVPRPRJUXbzk_5oCuc_5_HTg2AbZ0vFPmZrhw1Enva9NxA6QpJy6nHqVCBbHvTSMWv)
5. [iclr.cc](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQExbfFFe5BzX19qnWanxW-O4qxNvrd2h8cJ2BaoezYJe1fgc_lzCj7_4F68z7wWE2yNpvdYr22-UXuK3TP41sb4dUaMlzdPi8LwoqGQ_IbKBcu52-Tpp04fmKYWfgTarsn-l1ISldohU5zTqE7vOo52KmqXqZI5UvNlEY53Ol4xyXmrw9Bj59hmfkh0MZTlKq9hBd8J8eHNbIXrnQkf8wRAsbs=)
6. [analyticsvidhya.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH_fHrMZEI0Bl71uuT92oW5eXc2Z8_NMb_QQeHY0lB5p0TDIktamqS6OnLvno4Ng_ChYcHITD4z1e82o5C-q2RATJ99ZUJSKqEGNC0bQ2xuwUeaegYVpMGN7c_nRpB3y1C5TxrWrNon51zyXg==)
7. [bigroup.com.au](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFUiUNdaYfX9ivPUVhu4abekgWIulFzWsQaqd20f8x05JTMUtw3924c3aO2CcAgQQS7V1wate7CZy8B1QFEEBz0gSo_0KGX1J6_8Rp6NHCGYYgG3UkSTkFHN42DAabgjCf8YFbEJ5Z7oOjh)
8. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEKwfaEBicsPjBiQ0F9gQcGm35i2k1kkL1UYlMptdmZT7DbIk96l1l-pcHqm3d-zSbH3Hq_0R3BCRSYus0e38haokhdRJjwkfr6vKK9w-VFO41PXVXc)
9. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHk8j2jObtu5CokU6Z7S4i8KmP73FErOEl_wgNS0tO64E2wpZSzC7Xtxs4Tzz1zx4c17XfbE89_T8AIMaQ1tLz2MBK31WvmSJ9_oGDbIjvE6Wld2fub)
10. [promptingguide.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH2oX_LuNGWcw025LvNiZRhN5Nas0NhPHv1wdgH-1cVpT3oq39VhzZg5VndDuEfByh4WImI8fZ6XtCqTv7gAAcpQivFStpdgjCLfIbBD62XIJinlFsnS4aL3gz491KNOQ==)
11. [ctfassets.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5nq4GOAZShpQvg2lFY1XCRLQ94KEQ6suQfKX4EXKV2U5KvXjKYurYvf1mBaR8vdtmTWdGh0nAopK-SCu0a9zD8Q6M81uB6ahugdbIpbAZEfxhXeu7ZypuUa5SBjZ_y1Tpjp86GDac1KxZ3O2LFLdLgXxX52YrsslOHpFkqfc8kEMZeygwMnE4hZr4sUOz_dKeOUSXUcYOys5Wfb4xEuo0MwAKObojuRJa2LKtexMUf6JiXvpXfw==)
12. [Link](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFNZ3VNuAK2pZ-T_WI9TeQg5LmNTDbaJaVeuv1AD3PmWVEiSwg2kreRAxXQsr2KYmpUevkNWnaWqX77CXy3Zt70Thv2juBGNBmRiKFUnT3hvpZpGfy3eDYnDtpFozr_kHeQ40JVTSXhvzjwwIau51mpUCU=)
13. [leoniemonigatti.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEgHYZ7aE3rY-1ZvcnIlb0ZtIjB7EBDc1B5l3ptMCobctDNM9jYCfv7RqPi-e_eLdLDY63qCf-qx-FAUOwTBI-_vAKZ_FDmJuFSbT7qOT5mVAoyt0uVRaaaVTtRe9s4BS_tA8gc3f8THdp3ZtX6XkDQwN4TlD3viJt8iL5kcF86j1M=)
14. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH11QzfJgO9uDqoXkPzAWkVjJfzQoE_l8tLQvZh38OqnqTXFo5e_WcqX2Rxo6ewBBJ7AsZSp-CAy8N-J4f5BTJRG1GfM4gaiX9qP6R6ktucTOAQz6ibBjYBxTQid7r6JOvxIU1nXENf76ydzTPLXA9FccbJBOYqYri3ajYqLWr5z1IlYww0Rzc8Rfp2mYYbcDVa0Znuon5kGpcG-s7ErXCd6-f02AlBarMyaolLhysYElQfeRgIipSQJj3SSg==)
15. [kronika.ac](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHIs3PuyH_V6oZfC73orDKNM02hXQn0xv59XZFVXJQ0uQSBcvZ5F912mtUlvFQeDezHfMvQWG9VNle2sHhCHH9GBjG5QFkqBLAUmponiun6TnZNutDQ2dFeudnS8seW2VgDbPUhMbza1A==)
16. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG1O8wAknw22G8te4S2sCBd_EKzlL4CZRU0BVwicw12L9Wp_wU3kPuJWQwNsQKLYLOSxDfWE7z5121d5Lv4_0EBuRo_YNzLhHH1JWXJSRstrKIoWJqfBfuR)
17. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGvZx9VBpN2QgTq1-AeM9ZkhkdufEwZjGFLGJyCZRcWqfM2gRhjBcaV-oH_Mpf5AaEEUhkyd67vip1HChFxf3U5cHwDF0XbMxVBGvxXS6gn5yumgw6w2JcmPhOkZKqG5tl4H5qiE7c26DlWXVEGeuyYUvlPd015LTDisznfkLwJl-GVzewnQ2vuVl4n11T_xdxYbUhodG9dkTTTuunKpLpROT_T29UMzmnfPGbKZSw5XazwqgX-m-vNqg==)
18. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG1BMGSPPOoJgRPkBiQ356PVAaLxOf531aLStNyiXl7isQHBRAMXm_mZHYHKvcSMi3alaDBNYV9lxLv997vupeFdwWvcmAZQrI8e8SJRotqXIuvmZ0r)
19. [semanticscholar.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG3xDVR2AGHkLGuCv5w7KF_H6OHA7fwzBj0y8WpMazG7_F60gNcxBQMhv4ROqmzPaMKJM_q5h-xq4d3DWAALF9Utv6-xoVMRlsqimhv_cw0jbJC-58sFTvhhjUbwA-j_we0DU_cZnpNfbkQDLS2CpPZDxNj-fKA_f5b354UAkphmjsAq6LHeJYgl7DUb0oK3W_v5fE7MhbmEVIwZaAVVEFY3CK9OtHvm7gZTnIwSZuxu5D9e4XHx9D8HWFXoeCwFz8=)
20. [openreview.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGSghdaBeGb9A3-pcywNsGt43vFN7faruk7WgptuPq8e-T-OBKPAj_MkHOpKlFHZFXrNe5tK4AVXSKJ1TOrTtPNU3r3il7dJLxKpXsq4sOT_TRjlVA8Z3nGfP5BqG53YQ==)
21. [github.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFcWQ-ZsZLadyKzZkKa7HQKjyfbh3BHdQmAqBDeizendf8iz_iOBofNdGpETiwn77XcVusHs11vE8kwk_D1dlpJY3ni4I72X8fPQ0jHdye8)
22. [huggingface.co](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG6-oDqeklG67tfQepxw11F_mYcWCRzZGWTNRp4A6Tcm89eVQYH4NSLHJjhuA4b9hXTbsQlawlcj4JhowV9OHSwYIekwvKzDtUNNN1nsqQcwwySduERcK3YwH2vbMkwCwjcQ-AV-A==)
23. [geeksforgeeks.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFrfY9r268a5w-uniO6O631dLl59tctsNirRXVi1KxEWJZh8Kew9dbqHKbhFT2mK1WZ2WwaCkkxkfly4irkc_Jo1mnTQ104KVoUuVUrF3OCyjHBv1OBO5MBB_tkf4iucGPcDnCNQY1HSN50F1vbxspb4AWeDaWgfiiGOp5WCQJLQO9TKbcBdQWdWYwHVPOLYI4IT6o=)
24. [edenai.co](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF1GiywiU7S44FOAnh5Wg17p1jf5TGAsNMbbexgBrShtf6Hm3T5tpvtvHmIE-ZvMC12zuC7DI6G4LJWCUwbHNJL_CfL-CY32XfapPDbc4Akd_W3pauoMvyfvbmTxIXdbVRs6wF3UX6Wn3T2CRc2LDopcyYnwfFG-SOkegRuYc5b4PzcHyo=)
25. [dev.to](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE1DD3EsBO6pOwGI6ck-V-gPWnYiDVGtkRk_qwjSa7e1SCCFOMMt1ACSk7O2RRXvGUVKnJHrES_0j-nKd9HonpY-uGc6BiAwifgM-uq11NivE-jIjWC45wlZYZQL4qovIA1jrfYZzSyMVKbNdtAzzEwjw==)
26. [baai.ac.cn](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG-h670q7h1ZzA_20zSq54H1KgNcoioBLdF2ixXXGp2uCUhSlDPgSK-K68uO4CQfoHP1rIFFHZ1yejUCOQFbYxeLBBOYoOkVZg2C60H9hY_jMd6ezjaMT3ji51L_z6R2HSbFDq7MxxSEo4pkSyAOG9uFzKHlv5zhRURpO_eODiW)
27. [substack.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHStxdnJnxCaLo3HDygW_qD90Ce-ZRie8-AZ-3d6mohREQT7DyXNEZnNClLMoSJAE47V_ghfSWDYjyA-5yZ6JrFI9FhUg2Q7pdeNK2Vy-M8TJcjf7rO1mu_DqvYKvNjzOfIsGELBOuppp_-xSNtasMrJTKU7BcVDmjeof3ndOw=)
28. [neo4j.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHi7dspsMzCCUkimh2bCFTo4daMEiutOJAcOLeXPmpgZodH-gw2DjtOPFBQe3etdx32jPSBHbtxcFijiDykHfo76pXgTakedH8hxATlwo0lpLzlOwwj5qmkDLFBZtqw1rQyJ-teXlKTXKPN)
29. [scribd.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHpPzMat7ZZ7pgToVnRaDJsfPx-bAlFPiqIjSaaLFZOCll98wP54GD9qQM7KtTu0Mw0g380MsvB8znPoEUE13Etzci0O47wSssOUwPYmFwF8_tvQEvnkU3JHaCeLCUKi_7rpTWz4N4yIjeHac7LDr2GIR-rofk7nkoHRUfeJsA_UF98BNS675ZYlLFh8fLwVRs4frwq0xw5PBX6fOfEgkLr)
30. [purplescape.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE_IIv3sP0mbiH5mPMdq91HRp_7LY9VDapbhwGR4BWaUMtiKh7FRNeqPMU23tlfA2LlKYWKVyocqf0lf6pQuTsvKbzvo69jKvRG5K4PHzRLtEiuvyDyKogazncQG2jBqbutAS5ZbXoq2moGK6binGy9Vj5QQzCocVk67RELVtHxJM7Mc-g1ry82oY8G4YLi-krXlHu6CY0HnSCDSTG4i4nqDrNZ57Lv5Q==)
31. [databricks.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFO2JYa8_15UNDkz7pde7lKM1rRBmwxhn1ThDx2OFPxpU9NNmUXLSsrDe0syEBV5rB9EdVUncOpYZA9gUDtr785N18h4gzQyKJNmM-FDwBgDdtsYN7fV3RiMSEzFAsQ9mgs_OZ2jNSCOrWZj_ek7VbWJ2iwha4oyztciANy1aZqj8qjtRlSTxiFUi6ZMOfcBufnPQQjImVTs84nvYEywep9tBpCmkbfjCCpon9huYDibBju)
32. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE8dG7GNEsFBEwReqXS_xiBYofWWSdB11kSXrO8lJ3howOebxKXeuys13qRmP8aeh8_ELTgNV0Pwx7NGSoxhEef_Nm5-cgGPfsWlgHonwU5J5XKGqD6ml_z_cqqUWAo1Vk=)
33. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGoB4LQWsV4QcSRc3LIAMud79xz7fTIEZrQ9Yd4yVhNV9Zm3itKXP5hNbA4BRJTc3Qd-4pHmHY-EU5fI1mWjUlaioNSSmc8xVpU2Nz4w3WSQVeEZGL4YcOV0i1F2yZHR7vcZAg6MtssIxoaJo4kL-OAGyjufXHzGkmDR-OjBz7k7lKbB3bp5SI7RamYSwGhW99QJKHiD3uTmKbMOp950fs3qCDGkgxk0k0gRZ7i_HS7CN9u7bhpNLMPzMxmncq5)
34. [latenode.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEXXFwBT5nG-txtg2Dekk7v_FKiGkTmlouOyHOCbETfVOdk-7oE_plveduWkSsysLHoANvtDxPFwhfanbn3nmpky6XUiQ47Qx7BAg1RcD9QsIDfnFpoPNyTuGqW32QwQ7fOH4WNRGOCsK0e1FZxPiOcOX5OTr7a6SOkY4Vo5sIQOpAj4ZPLAcoGdYy40mgTZy9g064bQP5GDydc3O5Xh_bNxOrKrW3DpNQcgSX-xEsbss-Yia6lZtcGpL5v6tUlqvtNpCDDi0O5k9Wohgp8HByJ)
35. [sider.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFsl47t3W9MMxwjJwc-ieTMxF5hm6KcooXMMTx4AThmnP-CyHavqBtCVeZzVMIBgQDipCm-9wl2FKK1uCnBxqimeKyCIf6xWukgkH8vhuhSMHokyguhLb4M-IGkaAbK3WPsKGukDqgsdNxtVJ4gua1JIOy5qQqkq3_S_m0uT-8ENShA-9TrCrMDiNzTnA==)
36. [ibm.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGUKVUHqFYwoaCTZ7pYnacxow9ntuihsvoTaRAKAQ-T0vz9nIqRu2F0wsAmkBjwnsiBKSUggoefoF-rxmX0Y81_mCPi-jFTn7CM5RSskW2hSly9WSY1Rf-4r70TEShSHLPgcj34iXQ7qffroMhg)
37. [scrapfly.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQERoJzoX5pCl5TOnnq7wPu2Uh4Y8DvcftqqXyAGPpnDYq9WTZcPhpILNIddV6Xcj0I9Ozx5wb-jbU7mu-Y9Y9DpWstQ0KyQRbXydDDYKg-Tp7irzrOuHne33_9Fqidmom7zgvx75ylL-cf7shObPg==)
38. [Link](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFjfbf51RTj9UJ7EGWbF0ljK1ndbWPFU5VCkG3hQNoXYg2sYVLyjFYDzRltXYodwenbr50QX8sNHFFvm5cKAzH4NjtErY4U8kzvXpeiaJ4BDyGKFj1nm2ETnCEMgk8=)
39. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF5v3ig6TVNuz0Xm7AmpBjCiVOqy-IxmJHyp_bc63OoT3WoWGI_QtrjFOpe7IBzg1gn5nRNoHg7y_bYGpMdXwNXJ9X_Uoo3vW8r1Wrv0cK8cnIeYQtyxp3diTk6-VHm9PC25WeVQBfZ6Ug7BQ==)
40. [pelayoarbues.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHZlpnbn-uV8BcRMuyD2YUep6pgb0S-SiJUjmL9Sbx-G5fB5jHINHdGEr8IPazuC8yHYPgoJC53RJGwBSGyNtK2-PYVhpXpS-rF3YlRhlV5WUI4AYU8cmFWBLgKI4UwbXyJuUcLdtov_vq6gHmWsU3G_dQ2Zm2YWisIhiMMRsGlrA-Jbgp6L4HnndyMtUhioHruk4PJ34hR9sfuAg==)
41. [towardsai.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCObVI-Gf8n2OdBjHJMchaQca3RW0WmernCu8C6MAeAgHINnMQNlLlNHlYuJPgkbXVhOJK1N1xLQXv-CtZ_lbbeiULRigFrl5WJSDr-7BzVCmQ4LGF-o9ZMknCefGvpxUq9r8pD5xt7QxTb9wVqjNYA1D1_LU2OoviFaYDZAE3dAvFshWYFuWHNZWH)
42. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGMcW4h73o69p1uyI6HpmSVfI8r1P8rhuZnH4DoYEpvuY8JbCMuH4MsdegI39FOxhAT668tjyR2rx9aA1c6RWK2W_p3LgqCCSxwOs_4huESGoLeGri9GSG3RtkKToktHM6HIagciNVKLrqGkLwDx6O24bn6cUDtYE71yyOVfovypyXRNMzNwhl1XKSFP9IjhO3ZlgXrGUcUwR7w2qYSBUDv6B1vo4SCVE3SQAD095cY1rF-dRB0PWbpSWHxbFk=)
43. [gopubby.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFiO2kipdnSrqwuTVyRsgMgjGhw0nLT8GSHGSyomdXfmAkVSh4NG59mnhjA9K_svt90KaQe6LC9Ls3ZQdyRENqY3m2AvDIse-ab8bd1TV5Armi65ntWWCLj4Ez0EWSpOTqC58rXv8M3cVKVZ0JcL99xIk35TPit7iy8z-zBHu1tYmzHg_tTUpM06pVCNJBD9rhgOfZDxogaApJPjnds0A==)
44. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGNHxjp-JUZp_8R-tqZy4z_ww0bMhCbmgY8HYJ-SX4ZYMxG6vD2KbJGFyuvg9SN9UfDNU54rtmimG5EeznzBngKkM9oj742UkPG60joqwIfJdOszjf72sAVFym-hzZj5ZdV)
45. [github.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQECi3XwBejh_rO-JRQYgnxoJ8W9PZuxu2Ezkk7KV_EdY-U5_yWMIR-zoq3PsP9WHsK5oswCiSRKoaS51-DRjmazjE15-T5BCfA5GUucCp-RZ2K5U4ulxisEfAuYyD4l8BlKyxX3YYkqiM-mwjjZmTi3qeMaII7SEsVDg1IEQVZS)
46. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFu6nJv8rNNN8eE6LwEP651Fp1z6TyUSo-d0JCvuvaQPfY9U2ki-mvT72d6aVHVaPjNBrVMOTw3UPX9okVb7O8l4i6Z6PPSUO1GHrU1iRafBUgdXf6vrwCpa1lQa31cTI4=)
47. [kore.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG4vsKblM1HyuK0J2Pnkh-RD9dIOz8txJ8X23oJPCEXMbdJEbyzZFw-EEVLHKCiYVqOezrJoKn5ZQxDPh3MImowTfX_56mpcHx13GZMP-thOOzAtEbkwKEvChjGSMgSN77S)
48. [plainenglish.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF7jYthyq1nN_gH9ZPbjfupn9ZkcooTOkGg9AA7XMgJSA7SkfCZnaxJyocT3LtYnPXA6ehGiimSAG36WGcit7iRubXjZSIQf6h26kkLvhBuPILxF91EFFQZYtdOKoH-9ImjfrgdqJ6CYL1jhtVAwbHg8J0PBCQpsGGF1BESO0jjHYHQMQ7OLyfj4gznT76Z7DSKLB1zX_G9Y-G_dnBrVZVJ)
49. [pinecone.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFL8Gc-DixAI7NKaTHnR7IgSKuVZO8IfivJzWpEfa6L2s5aovkqRTwrerBSoJspSoLzuTGI4nP8eqhYuu__BfIADwIx9ZmQ6MuoSYB4hwEXnFX_Q6YH_HIijfyl7TvlQNlOmF4i89UKp7ZiPA==)
50. [semanticscholar.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH7M5RJUHNqqWkjBktJkLy1o3lA2F68JkfllaVLh8U5SO0KvdJKafzf5VhlQy7M0lyIoqP_Cvw9F5ntDIwyfnUX6jJW2ADkqOOSNyOlfJKr5BXwkDNelNZI9mLZBh055QgbVNdpcR3Ib6tZblITonUPul4_uKMUaojZTS7YLMLRT02-B5CWabAzGlXBu2hUHpZRbaCzDzMbQ8j-cqptQXmzaqlbZ34rncx_ej9E1Ei49UmBx24HRaXng6QtafZ3ReB-rw==)
51. [leoniemonigatti.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF21v31wkjxh36c2FKol9_0r40aysue4Yp_vtWi4vw7aHPc1uXOgwBvfNonfwP3RG8wyw2bcr4PsmYG_QXTpxva7PTECGvXIM8Nh6WEJYCjcUvC-5u6K5Z_MAR6TwkoSTcAv16yfZBC-DFVAhsQ6CuZXyHVMV9_H3c_qYg5NUCz7-khkArelU4=)
52. [latenode.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCH1-jyCVdL_nKU9LNik9wXren-t3LIhEAaNv6MytyyTnLrEQsDhrevxYWb_PnU00DDEmjE1-2i87P_MNqO7FPQ84eIENgqN6pX1faMtQRboC8cKO4Uwvgorxjeliw-MwvJG9zIPVOdJeegwam9QkIFMPUEpjH1lH2O-XQL4wIrperNGWETw1msn2o0GEcQkebuku9FCe4R-IGw5mmmDdx8A-QvNnBFTSdH_UXRliTbUW3IuvZwAZLOgsu7ZfIDfuVtrsUfdbJsh67ndL6EcydmxMa2Py3PKPs_RbOEUB2VGOwuCYzEIBYtJKKWrs=)
53. [Link](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHltlm07JncRoXysUMGNSOvwSJ2WHx7W8LGbrcx7lGBcM9jTyI7Q7jxYPpO35qeGzlchI3pw2nkrWM2A54Aqdeuu0oxEMfDEcekHKL9rEAmnRMdzxrAMTMNEJVmv4S1QhfFR-vg-lEBo1g=)
54. [journalijsra.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFnyqBvo39HAiNxtAPO17qTkSeK0LD750I_y--fTX4awg6SwJtLo_7HWsp71DVeeBqYSoiqasw2BEqrMN65j-laPzS0KwELpTLrHRVxcObPf_3ODMS4-mV2n1hXhjnE97T_6w3OvkGmpwlRc5JuRuD-UX5dqF_v3mf5Wl689krGnhcd)
55. [theblueowls.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGeyWkTS3FiRJ-PbXClNrgVjSkOzRuYr5mMO3XKPtqF8-AZOBP0uEagoD_TVhBFM5Lq7rxn2eZHhsvbkIEIzKSKkzviIvSxcxO6xb0EkZX16Cetg3zMe2bd-VdGN73v3s43qkfVXf7XxKI2TSqCRe74bHBMgC7olgKm_dPEsEaXya6X9Y4X8-VxI3TK)
56. [telusdigital.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH3BT_PJTbOzDkg2ckdniSrms-HgCvZ5fFpeIx6tVyCPeCHBQt7C11drUV0Za8af36bhN7YuYSxmfpL0nGQrLDJI5OntAv3xUmgRa35NbCqFpWy9yX5V_-Ux3YjkflIU0gXFpfa-zK5gMbBdrUoQlgEGAL_pRXdZD5XechGyoxf2OLwtfIdsBE=)
57. [aclanthology.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQELf3GNoRFW1TikmLpqCDxWDUDcDquAgCS3VDx-6f8SIVTWOXR1U2Sn7txIISO97v2XPH8FdZr8GgXCbIGTh-zjeYCEIZLHCS990ZGrmYBkQRb4QkZrox1Lg_5a5nBi0_qvQr0KW0vexSc=)
58. [mdpi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFG2ef5OR3gTrNMYMA4bOyy9ZcKRyUSsqXdoWxXQ5xifXR-7Sruz5wSNWPun0WckjSstl_dkEyuRL3SB_Pe-mdLcG2x9q3brgSJl-YvRf8TLMtLxMyfk0WuDjcJqg==)
59. [ijrpr.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFDZw_kxlviNGQIdME8UzotroyDLE7PKirQzQRBMEVmZ1MLUEsLXRLjohnzB3W79hQfU8lfPd29a8yGz4D67u5-eepx63Z-zco9alegRu5yMokoK8eV8xo9Ard-ejA7mXcG9k6nt8xW)
60. [huggingface.co](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFs83HwMWocm-eNWE5KHDB3VTCOCkuDxpgXZL9-3NqmkmOXCjGjI3_GnmkE0kTuJmOplNVM-H76RK5RkJf7Jr2lbDMbnLlRcla0GqQ4typA_hYeiF5qS4LaddW4Xno=)
61. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGiATjPSzv0fdEMYl32gsEv5tYZ5feXYAzlDLdHmxpchhQ0qXPHCLZ54wAp5nrMJz6wh36xlB9RpKXGqQvNOKDfkYX2arP7G9nY4QWtVhP-Ovrz2zoQBnjvkYKtz8P54C4qkwQVVHLQta9p5YDiRpl-lzmPLC_y2eRCgjaFejmZdxnBiY8IWJnncKB6NK7Wh3c5p1SKFp2rmm6eN57QOwJxoZSu1rqO2ZlKhzusGvYXwn9l6J3ndbVJTMmqkQ==)
62. [slideshare.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHdrQTQ2k5v3x_D7W_ho84zqv9fUwymJN8kp8UzQan1JQ7cR-g98IWy9HM3U70KyEo1uE3OZswrfrSfqlAWhbE77Li56fz4XX9LDOSBrJNHuorN5odcGHyOlLoywy2xm44pWbfQMvU8vrAk6X5G37fy76hEN8Gsdn2XnwqKHjQ9MwuNoo_BNbkFxyvoRw5xLCCrqdrQekPnx-PCZbpeEq4beOEGQIn4UgFgibwB7XzH)
63. [substack.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGDqt9Gd9KCZo-K-BLuWT0-isSJMN7N6zMRx95pxxhyaqO9P0LxwJGrERhbooPuYvmwdP2FPWgnZ7Y6r8MIc9026LGKJe7NHx3BxmicXECL18P4bT4VBXx5unQfzgz_JNkIxZeFG5TtaTaN4ym8R8XY8SW-61RkddiTw9Np)
64. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFtuPMEqzG18rkX1n20xFb7tP97gg88vOddJRqDdK8uFtlBINgPnvA9KHzb_rlZTNq7tqPzT5yhe8N036lGbngN1xraBAAvBUL69OStjs8Tbi_dxC1LstM9m-pKYEmxBp1izF9Blscc7_gG8mFKPLRArTa93zVvx91Td9eSA1VSWCi1WnQxVxcsqgunwGxbw4nnokeLhmMdTN3ZH18r)
65. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHChlZi9Y82zy8TEmG3MEt3W3yDx_cRF3jbGqdHufW_A718FuV05U5gbmlF94EhFYKYRBPVDWmbsF1-mEY1pdjYIOpMe8hYsKrX1rX4QvkMU56yhTJxQyDPpQKk7EzqMsScGMU0XYxNiw9k5M8fnW7j05QzAzqfARac8Pu4S-mBO1AmDQ0KQPvi9D67POMDzBhbZPac_KgR0vIs7Viiigao7T63dxZ-KQ==)
66. [n8n.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHnREGoyOPoTM8EnhqH-jZ3vzkf0JUFgzE12mjNrf8g_jc7zTsgm84PJ056zUf9HVSZgGe3XtwBtO12j28iP8Ud06Q0c69Cjm6ZGPfpnhicRigtawJjYzEh00sm5DjyysRD)
67. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGhvhtDl7p8wqnKTVTtIWNr1f4kORRi5RmdejIdjV0uE_sWb2jv22v9qaidz4oNI4b9YJA92299-AYtiyDdc-g4w8-ycNnVN-MX4yY_-G3qTTyN9hEFvgqNNsLL24f0oqU=)
68. [myli.page](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG170jMf72uABTF1jt788Q68W0oWc3142t9AfHSGxMknyAI2rxiDzCewbppnlvlbUZnAJDMpm68zoufvoUsslRXuANAWSQ9sAKDYyBHwo835hYkYvaMlo9xOmIJbzOFQKd2PwAHapC8Woucrq8bx_bGluTnOFYFN05o4AVH9W672y84d_0lrWzE)
69. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHz3rz-N6N8M3PDcJ8RdxhYKdkzusoBmJVSsl8F689qr4N2l5gPNJqG3ECtClwT1BoN15uxglbxC1Y2jtQx3vgptg-OAywcqPlozoVYCvgYHl51u45O0Ntvq1JyVVD640cmXo07WZSH9y-KXyB66bZq0MGCb-bz2N_bJ0Gm5YjOSrKSlHtCwVLgb9npzIvfYJ3PWiOmnKazbj61dlU8vsDtWcc5PJbV0LbzGJ3EzhqhRkqg80xC4bAiZ0KE9tF_aA==)
70. [statsig.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE8XX8ITrklrcdJdM_4tYSylb2TbJuDX_niEaqBZ_BF1_jDvs7GUzKFiFhkOv7KuddkVJhwiE2A2J01VZEuvxoc42Zy0Qi2ZeSif6vL8PJEfKx5mKUj-Q_JWyNLZYF5Y2t32fLRsILvR_-sbtwiUlLCFUQN_fE=)
71. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFruEZQmmLll2YkleLHxpZTaH4eZsZLzyqj4M8aTm8KpXwpYBwhjkPmrzJLXx0lAJ8zdNINq6p86lZEhmxHLuYDy-EQcA2KT-0QT50qH0WxdPU60lIs-2lX8DHr67qBOiLbVgD0DK2-bqCPJgLZ6uoDXHk-Dp_Iv1Fb5dk=)
72. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEcdBrh3pbnq3SzzccMOwtm3GtqRzzXCtl9SI6C1YYPeM9huR7xo1TZmx7d5PEdYqWU6GsYAIeW76FaRk5oIH8zX1-s4XXW0Xf5N87T8Y-CSykejXfvHoBb-h0=)
73. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFa09upCSNvywCxQsj0g5X5efrKs3KrRmcRlZKK6dvbX_RZhxcJa_8AW8-7DJgy3h1KGA-kRt-peCZfiVlGfK7GdYqOxi1ZSwnYeBwM_q-xch7KcNiFf4rznKkvjRnOsrztiPgNbATghohSmscyr76fAQ==)
74. [Link](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFnFpCHo5Ve0Lt0lL_ES-IaqwYiJNHoTz18RCvNUUE4oycV4bdSKH17M_irSi22x_fJA6STfgFnxo2hFY7yE1l0KTJzdFyuSSCuuwI6ryrjPjk-4utyd4ez9mbmvQKa)
