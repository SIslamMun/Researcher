# LangChain: A Comprehensive Framework for Large Language Model Applications

## Key Points
*   **Definition:** LangChain is an open-source orchestration framework designed to simplify the development of applications powered by Large Language Models (LLMs). It provides abstractions to connect language models with external data sources, computation, and other tools.
*   **Core Philosophy:** The framework emphasizes **composability**, allowing developers to chain together distinct components (prompts, models, parsers) into complex workflows using the LangChain Expression Language (LCEL).
*   **Ecosystem Evolution:** Originally a Python library for simple chains, it has evolved into a comprehensive ecosystem including **LangGraph** (for stateful, multi-agent workflows), **LangSmith** (for observability and testing), and **LangServe** (for deployment).
*   **Commercial Status:** As of late 2025, LangChain Inc. has achieved "unicorn" status with a valuation of $1.25 billion following a Series B funding round, reflecting its position as a standard infrastructure layer in the Generative AI stack.
*   **Primary Use Cases:** It is widely used for Retrieval-Augmented Generation (RAG), chatbots, document analysis, and building autonomous agents that can reason and execute multi-step tasks.

---

**LangChain** acts as the "glue" code for the artificial intelligence era. While Large Language Models (LLMs) like GPT-4 or Claude possess immense raw intelligence, they are isolated from the real world—they cannot inherently access private data, remember past conversations, or trigger actions in software systems. LangChain solves this by providing a structured way to connect these models to the outside world.

At its simplest, LangChain allows a developer to create a sequence of actions: "Take this user question, search my company's database for the answer, format it into a prompt, send it to the AI, and then clean up the AI's response." This sequence is called a **chain**. As AI applications have become more sophisticated, LangChain has introduced **agents**, which use the LLM as a reasoning engine to decide *which* actions to take dynamically, rather than following a hard-coded script.

The framework has grown from a popular open-source project into a critical piece of enterprise infrastructure, supporting everything from simple Q&A bots to complex, multi-agent systems that can code, research, and analyze data autonomously.

---

## 1. Introduction and Definition

LangChain is defined as a software framework that facilitates the integration of Large Language Models (LLMs) into applications [cite: 1, 2]. It serves as a middleware layer that abstracts the complexities of working with raw API calls to model providers (such as OpenAI, Anthropic, or Google), offering a unified interface for model interaction, prompt management, and data retrieval [cite: 3, 4].

The framework's primary value proposition is **context-awareness** and **reasoning** [cite: 5]. It enables LLMs to be:
1.  **Data-aware:** Connecting a language model to other sources of data (e.g., private documents, SQL databases, the internet).
2.  **Agentic:** Allowing a language model to interact with its environment (e.g., executing code, calling APIs) [cite: 2].

LangChain is available as open-source libraries in both **Python** and **JavaScript/TypeScript**, ensuring broad accessibility across the two most dominant programming ecosystems for AI and web development [cite: 1, 6].

## 2. History and Commercial Evolution

### 2.1 Origins (2022)
LangChain was launched in **October 2022** by **Harrison Chase**, a machine learning engineer previously at Robust Intelligence and Kensho [cite: 1, 7]. The project began as a lightweight open-source wrapper around prompt templates to solve the repetitive "glue code" problems developers faced when building early GPT-3 prototypes [cite: 5, 8].

The timing of the launch was fortuitous, occurring just weeks before the release of ChatGPT in November 2022. As the generative AI wave exploded, LangChain became the default framework for developers seeking to build applications beyond the standard ChatGPT interface, specifically for "Chat with your Data" (RAG) use cases [cite: 5].

### 2.2 Rapid Growth and Funding (2023–2024)
Following its viral adoption on GitHub, LangChain incorporated as a company in early 2023. Its growth trajectory was steep:
*   **Seed Round:** In April 2023, the company announced a $10 million seed investment led by Benchmark [cite: 1].
*   **Series A:** Shortly thereafter, in early 2023/2024, LangChain raised roughly $20-$25 million in a Series A round led by Sequoia Capital, valuing the young startup at approximately $200 million [cite: 1, 9].

During this period, the framework expanded from simple chains to complex cognitive architectures. In late 2023, the team introduced the **LangChain Expression Language (LCEL)** to standardize how chains were constructed, and launched **LangSmith**, a platform for debugging and monitoring LLM applications [cite: 5, 10].

### 2.3 Unicorn Status and Series B (2025)
By late 2025, LangChain had solidified its position as a foundational layer of the AI stack. In **October 2025**, LangChain announced a **$125 million Series B** funding round led by **IVP**, with participation from existing investors Sequoia and Benchmark, as well as new strategic backers like CapitalG, ServiceNow Ventures, and Datadog [cite: 11, 12].

This round valued the company at **$1.25 billion**, officially granting it "unicorn" status [cite: 9, 12]. The funding was earmarked to expand its "Agent Engineering" platform, specifically focusing on **LangGraph** for stateful agent orchestration and **LangSmith** for enterprise-grade observability [cite: 13].

## 3. Core Architecture and Components

LangChain's architecture is modular, designed to allow developers to swap components (e.g., changing from OpenAI to Anthropic) with minimal code changes. The framework was significantly re-architected in version 0.1 (January 2024) to separate core abstractions from third-party integrations [cite: 14, 15].

### 3.1 Package Structure
*   **`langchain-core`:** Contains the base abstractions (e.g., `BaseLLM`, `BaseRetriever`, `BaseTool`) and the LangChain Expression Language (LCEL). This package has few dependencies and defines the standard interfaces [cite: 14, 16].
*   **`langchain-community`:** A massive collection of third-party integrations (e.g., specific vector stores like Pinecone, loaders for PDFs, wrappers for various APIs). This separates the volatile ecosystem of integrations from the stable core [cite: 14, 15].
*   **`langchain`:** The main library containing the cognitive architecture logic—chains, agents, and retrieval strategies—that ties the core and community components together [cite: 14].
*   **Partner Packages:** High-priority integrations (like `langchain-openai`, `langchain-anthropic`) were moved to their own lightweight packages to improve maintenance and dependency management [cite: 16, 17].

### 3.2 LangChain Expression Language (LCEL)
Introduced in late 2023, LCEL is a declarative way to compose chains. It uses the Unix pipe operator (`|`) to link components together, ensuring that the output of one step is automatically passed as the input to the next [cite: 18, 19].

**Key Benefits of LCEL:**
*   **Streaming:** Chains built with LCEL support streaming of tokens out-of-the-box, improving user experience (time-to-first-token) [cite: 18].
*   **Parallelism:** Steps that do not depend on each other can be executed in parallel automatically (e.g., fetching documents from two different retrievers) [cite: 20].
*   **Standard Interface:** All LCEL objects implement the `Runnable` interface, offering standard methods like `.invoke()`, `.stream()`, and `.batch()` [cite: 21].

**Example of LCEL Syntax:**
```python
chain = prompt | model | output_parser
```
In this architecture, the `prompt` formats the user input, the `model` generates a response, and the `output_parser` extracts the relevant string or structured data [cite: 22].

### 3.3 Core Modules
LangChain is organized into several key modules [cite: 4]:

1.  **Model I/O:** Interfaces for **LLMs** (text-in, text-out) and **Chat Models** (message-in, message-out). It also handles **Prompts** (templating user inputs) and **Output Parsers** (structuring model responses into JSON, lists, etc.) [cite: 2, 23].
2.  **Retrieval (RAG):** Tools to load, transform, and query data.
    *   **Document Loaders:** Read data from sources like PDFs, CSVs, Notion, or the web [cite: 24].
    *   **Text Splitters:** Break large documents into smaller chunks (e.g., `RecursiveCharacterTextSplitter`) to fit within model context windows [cite: 24].
    *   **Vector Stores:** Databases (e.g., Milvus, FAISS, Pinecone) that store semantic embeddings of text for similarity search [cite: 2, 24].
    *   **Retrievers:** Algorithms to fetch relevant documents given a query [cite: 25].
3.  **Agents:** Systems where the LLM acts as a reasoning engine to determine a sequence of actions. Agents use **Tools** (functions like "Google Search" or "Calculator") to perform tasks [cite: 3, 4].
4.  **Memory:** Components that allow chains to retain state (context) across multiple interactions, essential for chatbots [cite: 4, 26].

## 4. The LangChain Ecosystem

As the complexity of AI applications grew, LangChain expanded from a single library into a suite of complementary tools, often referred to as the "LangChain Stack" [cite: 27, 28].

### 4.1 LangGraph
Released as a response to the limitations of linear chains, **LangGraph** is an orchestration framework for building stateful, multi-agent applications [cite: 29, 30].

*   **Graph-Based:** Unlike standard LangChain (which uses Directed Acyclic Graphs or DAGs), LangGraph supports **cyclic graphs**, allowing for loops. This is critical for agentic behaviors like "retry until successful" or "ask for human feedback and loop back" [cite: 31, 32].
*   **State Management:** LangGraph makes state explicit. Each node in the graph can read and write to a shared state, enabling complex coordination between multiple agents [cite: 30].
*   **Use Case:** It is the recommended tool for building production-grade agents, whereas the core LangChain library is often sufficient for simpler, linear RAG pipelines [cite: 33].

### 4.2 LangSmith
**LangSmith** is a platform for **observability, testing, and evaluation** [cite: 27]. Because LLMs are non-deterministic "black boxes," debugging them is difficult.
*   **Tracing:** LangSmith logs every step of a chain (inputs, outputs, latency, token usage), allowing developers to pinpoint exactly where a prompt failed or why a response was hallucinated [cite: 27, 34].
*   **Evaluation:** It allows developers to run datasets of questions against their apps to measure performance (accuracy, relevance) over time, preventing regressions [cite: 27].
*   **Adoption:** It is widely used in production to monitor costs and quality [cite: 35].

### 4.3 LangServe
**LangServe** is a deployment library that automatically converts LangChain chains or runnables into production-ready **REST APIs** [cite: 10, 29]. It handles the complexities of streaming, async processing, and schema generation, allowing developers to expose their AI logic as web services with minimal boilerplate [cite: 36].

### 4.4 LangFlow
**LangFlow** is a visual, low-code interface for building LangChain applications [cite: 37]. It provides a drag-and-drop canvas where users can connect components (loaders, models, vector stores) to prototype flows without writing code. While LangChain focuses on code-first flexibility, LangFlow targets rapid prototyping and accessibility for non-technical users [cite: 37, 38].

## 5. Technical Implementation and Versioning

### 5.1 Versioning Policy
LangChain follows a semantic versioning strategy for its core packages.
*   **v0.1 (Jan 2024):** The first stable release, marking the separation of `langchain-core` and `langchain-community` to ensure backward compatibility [cite: 15, 39].
*   **v0.2 (May 2024):** Introduced versioned documentation and further modularization [cite: 40].
*   **v0.3 (Sep 2024):** A significant update that migrated the internal data validation logic from **Pydantic v1 to Pydantic v2**, offering performance improvements but requiring migration for some users. It also ended support for Python 3.8 [cite: 17, 41].

### 5.2 Security
As a widely deployed framework, LangChain is subject to security scrutiny. For example, in late 2025, a critical vulnerability (**CVE-2025-68664**) was identified in `langchain-core` related to unsafe deserialization methods (`dumps`/`dumpd`), highlighting the risks of processing untrusted data in agentic workflows [cite: 42]. The maintainers actively patch such issues, with support policies defined for current and previous minor versions [cite: 43].

## 6. Applications and Impact

### 6.1 Primary Use Cases
1.  **Retrieval-Augmented Generation (RAG):** The most common use case. LangChain manages the pipeline of ingesting documents, chunking them, embedding them into vector stores, and retrieving them to ground LLM responses in factual data [cite: 2, 44].
2.  **Chatbots:** Using the `Memory` and `Chain` components to build conversational interfaces that remember context [cite: 45].
3.  **Data Extraction:** Parsing unstructured text (like emails or invoices) into structured formats (JSON) using Output Parsers [cite: 46].
4.  **Autonomous Agents:** Building systems that can use tools (web search, calculators, APIs) to solve multi-step problems. LangGraph is increasingly the standard for this [cite: 32].

### 6.2 Academic and Industry Adoption
LangChain has become a standard reference in AI research.
*   **Research:** It is cited in papers exploring automated customer service [cite: 47] and scientific question-answering systems like **PaperQA**, which uses LangChain agents to retrieve and synthesize scientific literature [cite: 48].
*   **Industry:** By late 2025, LangChain reported over **90 million monthly downloads** and usage by major enterprises like Cisco, Rakuten, and Elastic to build internal AI platforms [cite: 12, 34].

## 7. Conclusion

LangChain has evolved from a simple productivity tool for prompt engineering into a comprehensive **infrastructure framework** for the Generative AI economy. By standardizing the interfaces for models, retrieval, and agentic behavior, it allows developers to build complex applications that are loosely coupled to specific model providers.

While the ecosystem has grown complex—spanning the core library, the graph-based orchestration of **LangGraph**, and the observability of **LangSmith**—this breadth addresses the full lifecycle of AI application development, from prototype to production. With its "unicorn" valuation and widespread adoption, LangChain represents the de facto standard for building context-aware, reasoning applications with LLMs.

---

## References

### Publications
[cite: 25] "LangChain component architecture" (LangChain). Documentation, 2025. https://docs.langchain.com/oss/python/langchain/component-architecture
[cite: 29] "Understanding the LangChain Framework" (TechLatest). Medium, 2024. https://medium.com/@techlatest.net/understanding-the-langchain-framework-8624e68fca32
[cite: 44] "I made a visual guide breaking down EVERY LangChain component" (Reddit User). Reddit, 2025. https://www.reddit.com/r/LangChain/comments/1p9fpp2/i_made_a_visual_guide_breaking_down_every/
[cite: 34] "LangChain Official Website" (LangChain). Website, 2025. https://www.langchain.com/
[cite: 3] "LangChain Overview" (LangChain). Documentation, 2025. https://docs.langchain.com/oss/python/langchain/overview
[cite: 1] "LangChain" (Wikipedia). Wikipedia, 2025. https://en.wikipedia.org/wiki/LangChain
[cite: 2] "What is LangChain?" (AWS). Amazon Web Services, 2025. https://aws.amazon.com/what-is/langchain/
[cite: 6] "LangChain" (IBM). IBM Topics, 2025. https://www.ibm.com/think/topics/langchain
[cite: 49] "LangChain Use Cases" (Google Cloud). Google Cloud, 2025. https://cloud.google.com/use-cases/langchain
[cite: 16] "LangChain Framework Explained" (DigitalOcean). DigitalOcean, 2025. https://www.digitalocean.com/community/conceptual-articles/langchain-framework-explained
[cite: 4] "Key Concepts of LangChain" (Ksolves). Ksolves Blog, 2024. https://www.ksolves.com/blog/artificial-intelligence/key-concepts-of-langchain
[cite: 50] "LangChain Intro" (Decube). Decube Blog, 2025. https://www.decube.io/post/langchain-intro
[cite: 27] "LangChain vs LangGraph vs LangSmith" (LangCopilot). LangCopilot, 2025. https://langcopilot.com/posts/2025-09-24-langchain-vs-langgraph-vs-langsmith-which
[cite: 10] "LangChain vs LangGraph vs LangSmith vs LangFlow" (DataCamp). DataCamp Tutorial, 2025. https://www.datacamp.com/tutorial/langchain-vs-langgraph-vs-langsmith-vs-langflow
[cite: 35] "LangChain vs LangGraph vs LangSmith" (Galileo). Galileo Blog, 2025. https://galileo.ai/blog/langchain-vs-langgraph-vs-langsmith
[cite: 51] "Understanding LangChain, LangGraph, and LangSmith" (Pollabd). Dev.to, 2025. https://dev.to/pollabd/understanding-langchain-langgraph-and-langsmith-5fm0
[cite: 28] "LangChain, LangGraph, LangFlow, LangSmith AI Guide" (DZone). DZone, 2025. https://dzone.com/articles/langchain-langgraph-langflow-langsmith-ai-guide
[cite: 7] "Harrison Chase LangChain" (Frederick.ai). Frederick.ai Blog, 2025. https://www.frederick.ai/blog/harrison-chase-langchain
[cite: 52] "The Story of LangChain" (Riyansh Chouhan). Medium, 2025. https://medium.com/@riyanshchouhan1223/the-story-of-langchain-how-a-small-team-built-the-framework-powering-generative-ai-e3899f7f2f93
[cite: 5] "LangChain's Origins" (Latent Space). Latent Space Newsletter, 2023. https://www.latent.space/p/langchain
[cite: 53] "Harrison Chase Interview" (Arize AI). Arize AI Resource, 2025. https://arize.com/resource/langchain/
[cite: 39] "LangChain Announces v0.1.0" (Web3Universe). Web3Universe, 2023. https://web3universe.today/langchain-announces-v0-1-0/
[cite: 14] "LangChain Release Notes Week of 12/11" (LangChain). LangChain Changelog, 2023. https://changelog.langchain.com/announcements/week-of-12-11-langchain-release-notes
[cite: 46] "LangChain 0.1.0: A New Milestone" (Pankaj Pandey). Medium, 2024. https://medium.com/@pankaj_pandey/langchain-0-1-0-a-new-milestone-in-language-model-frameworks-adbd575f2183
[cite: 15] "LangChain v0.1.0 Announcement" (LangChain). LangChain Blog, 2024. https://blog.langchain.com/langchain-v0-1-0/
[cite: 54] "LangChain Changelog" (LangChain). LangChain Changelog, 2024. https://changelog.langchain.com/?date=2024-01-01
[cite: 18] "LangChain Expression Language" (Pinecone). Pinecone Learn, 2025. https://www.pinecone.io/learn/series/langchain/langchain-expression-language/
[cite: 19] "LangChain Expression Language (LCEL)" (GeeksforGeeks). GeeksforGeeks, 2025. https://www.geeksforgeeks.org/artificial-intelligence/langchain/
[cite: 21] "LangChain Expression Language" (K21Academy). K21Academy, 2025. https://k21academy.com/ai-ml/langchain-expression-language/
[cite: 55] "What is LangChain Expression Language?" (Cobus Greyling). Medium, 2024. https://cobusgreyling.medium.com/what-is-langchain-expression-language-lcel-8a828c38b37d
[cite: 20] "LangChain Expression Language LCEL Explanation" (James Calam). YouTube, 2023. https://www.youtube.com/watch?v=O0dUOtOIrfs
[cite: 30] "LangChain vs LangGraph: A Comparative Analysis" (Tahir Balarabe). Medium, 2025. https://medium.com/@tahirbalarabe2/%EF%B8%8Flangchain-vs-langgraph-a-comparative-analysis-ce7749a80d9c
[cite: 31] "LangChain vs LangGraph" (DuploCloud). DuploCloud Blog, 2025. https://duplocloud.com/blog/langchain-vs-langgraph/
[cite: 32] "LangChain vs LangGraph" (Milvus). Milvus Blog, 2025. https://milvus.io/blog/langchain-vs-langgraph.md
[cite: 37] "LangGraph vs LangChain" (Oxylabs). Oxylabs Blog, 2025. https://oxylabs.io/blog/langgraph-vs-langchain
[cite: 33] "LangChain vs LangGraph" (GeeksforGeeks). GeeksforGeeks, 2025. https://www.geeksforgeeks.org/artificial-intelligence/langchain-vs-langgraph/
[cite: 56] "Langflow Community" (Langflow). Langflow Docs, 2025. https://docs.langflow.org/contributing-community
[cite: 57] "What is Langflow?" (Langflow). Langflow Docs, 2025. https://docs.langflow.org/
[cite: 58] "Langflow Website" (Langflow). Langflow.org, 2025. https://www.langflow.org/
[cite: 59] "Langflow GitHub" (Langflow). GitHub, 2025. https://github.com/langflow-ai/langflow
[cite: 38] "My very first hands-on experience with Langflow" (Aairom). Dev.to, 2025. https://dev.to/aairom/my-very-first-hands-on-epxerience-with-langflow-o33
[cite: 47] "Automating Customer Service using LangChain" (Pandya et al.). arXiv:2310.05421, 2023. https://arxiv.org/abs/2310.05421
[cite: 60] "LangChain Analysis" (Mavroudis). ResearchGate, 2024. https://www.researchgate.net/publication/385681151_LangChain
[cite: 61] "An Effective Query System Using LLMs and LangChain" (Ghane). IJSET, 2024. https://www.ijset.in/wp-content/uploads/IJSET_V12_issue4_659.pdf
[cite: 24] "LangChain News Research Tool" (IJRPR). IJRPR, 2024. https://ijrpr.com/uploads/V5ISSUE7/IJRPR31768.pdf
[cite: 62] "LLM Source Citation with LangChain" (ReadyTensor). ReadyTensor, 2025. https://app.readytensor.ai/publications/llm-source-citation-with-langchain-dXocR7whrOTR
[cite: 17] "Announcing LangChain v0.3" (LangChain). LangChain Blog, 2024. https://blog.langchain.com/announcing-langchain-v0-3/
[cite: 41] "Migrating to Pydantic 2 for Python" (LangChain). LangChain Changelog, 2024. https://changelog.langchain.com/announcements/langchain-v0-3-migrating-to-pydantic-2-for-python-peer-dependencies-for-javascript
[cite: 63] "LangChain JS Release Notes" (LangChain). LangChain Docs, 2025. https://docs.langchain.com/oss/javascript/releases/changelog
[cite: 64] "LangChain Elixir Changelog" (Brainlid). GitHub, 2025. https://github.com/brainlid/langchain/blob/main/CHANGELOG.md
[cite: 40] "LangChain v0.2 Release Notes" (LangChain). LangChain Changelog, 2024. https://changelog.langchain.com/announcements/the-langchain-v0-2-release-has-versioned-docs-with-clearer-structure-and-content
[cite: 1] "LangChain Wikipedia Entry" (Wikipedia). Wikipedia, 2025. https://en.wikipedia.org/wiki/LangChain
[cite: 65] "LangChain Company Profile" (Tracxn). Tracxn, 2025. https://tracxn.com/d/companies/langchain/__O9N2dOHcgRE9Nbcn5BFfkUHn-rVk6GTbq8oY-UJ0Ba4
[cite: 7] "Harrison Chase and LangChain" (Frederick.ai). Frederick.ai, 2025. https://www.frederick.ai/blog/harrison-chase-langchain
[cite: 8] "Three Years of LangChain" (Harrison Chase). LangChain Blog, 2025. https://blog.langchain.com/three-years-langchain/
[cite: 52] "The Story of LangChain" (Medium). Medium, 2025. https://medium.com/@riyanshchouhan1223/the-story-of-langchain-how-a-small-team-built-the-framework-powering-generative-ai-e3899f7f2f93
[cite: 4] "Key Concepts of LangChain" (Ksolves). Ksolves, 2024. https://www.ksolves.com/blog/artificial-intelligence/key-concepts-of-langchain
[cite: 45] "What is LangChain Complete Guide" (Metaschool). Metaschool, 2025. https://metaschool.so/articles/what-is-langchain-complete-guide-2025
[cite: 66] "What is LangChain ML Architecture" (LakeFS). LakeFS Blog, 2025. https://lakefs.io/blog/what-is-langchain-ml-architecture/
[cite: 26] "LangChain in 2025" (Medium). Medium, 2025. https://medium.com/@balu.ds1524/langchain-in-2025-the-ultimate-cheat-sheet-every-developer-actually-needs-93e201752b89
[cite: 23] "LangChain Components Guide" (Deepchecks). Deepchecks, 2023. https://www.deepchecks.com/langchain-components-a-comprehensive-beginners-guide/
[cite: 48] "PaperQA: Retrieval-Augmented Generative Agent for Scientific Research" (Lala et al.). arXiv:2312.07559, 2023. https://arxiv.org/pdf/2312.07559
[cite: 67] "LangChain Cite Sources" (Denys on Data). YouTube, 2023. https://www.youtube.com/watch?v=MOawB4k9-jk
[cite: 68] "Citation Parsing with LLMs" (Research). arXiv:2505.15948, 2025. https://arxiv.org/pdf/2505.15948
[cite: 69] "Evaluation of Source Veracity in LLMs" (PMC). PMC, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12003634/
[cite: 70] "Attribution, Citation, and Quotation in LLMs" (Schreieder et al.). arXiv:2508.15396, 2025. https://arxiv.org/html/2508.15396v1
[cite: 10] "LangChain Ecosystem Overview" (DataCamp). DataCamp, 2025. https://www.datacamp.com/tutorial/langchain-vs-langgraph-vs-langsmith-vs-langflow
[cite: 22] "LangChain, LangSmith and LangGraph" (Finxter). Finxter Academy, 2025. https://academy.finxter.com/langchain-langsmith-and-langgraph/
[cite: 51] "Understanding LangChain Ecosystem" (Pollabd). Dev.to, 2025. https://dev.to/pollabd/understanding-langchain-langgraph-and-langsmith-5fm0
[cite: 36] "LangChain Beginners Guide" (Analytics Vidhya). Analytics Vidhya, 2025. https://www.analyticsvidhya.com/blog/2025/12/langchain-beginners-guide/
[cite: 71] "LangChain vs LangGraph Decision Guide" (BinaryVerse). BinaryVerse AI, 2025. https://binaryverseai.com/langchain-vs-langgraph-decision-guide-framework/
[cite: 11] "LangChain Raises $125 Million" (Silicon Valley Investclub). Substack, 2025. https://siliconvalleyinvestclub.substack.com/p/langchain-raises-125-million-at-a
[cite: 12] "LangChain Raises $125M Series B" (WebProNews). WebProNews, 2025. https://www.webpronews.com/langchain-raises-125m-in-series-b-hits-1-25b-unicorn-valuation/
[cite: 72] "LangChain Raises $125M" (SiliconANGLE). SiliconANGLE, 2025. https://siliconangle.com/2025/10/20/ai-agent-tooling-provider-langchain-raises-125m-1-25b-valuation/
[cite: 13] "LangChain Series B Funding" (The Head and Tale). The Head and Tale, 2025. https://theheadandtale.com/ai-emerging-tech/langchain-raises-series-b-funding-at--1-25-billion-valuation-led-by-ivp/
[cite: 9] "LangChain is now a Unicorn" (Fortune). Fortune, 2025. https://fortune.com/2025/10/20/exclusive-early-ai-darling-langchain-is-now-a-unicorn-with-a-fresh-125-million-in-funding/
[cite: 73] "LangChain PyPI" (PyPI). PyPI, 2025. https://pypi.org/project/langchain/
[cite: 42] "LangGrinch: LangChain Core CVE-2025-68664" (Cyata). Cyata Blog, 2025. https://cyata.ai/blog/langgrinch-langchain-core-cve-2025-68664/
[cite: 74] "LangChain GitHub Releases" (GitHub). GitHub, 2025. https://github.com/langchain-ai/langchain/releases
[cite: 75] "LangChain Release Policy" (LangChain). LangChain Docs, 2025. https://docs.langchain.com/oss/python/release-policy
[cite: 76] "LangChain Changelog Page 5" (LangChain). LangChain Changelog, 2025. https://changelog.langchain.com/?page=5
[cite: 43] "LangSmith Release Versions" (LangChain). LangSmith Docs, 2025. https://docs.langchain.com/langsmith/release-versions
[cite: 1] "LangChain Wikipedia" (Wikipedia). Wikipedia, 2025. https://en.wikipedia.org/wiki/LangChain
[cite: 34] "LangChain Homepage" (LangChain). LangChain.com, 2025. https://www.langchain.com/

### Code & Tools
 langchain - Framework for developing applications powered by language models. https://github.com/langchain-ai/langchain
 langgraph - Library for building stateful, multi-agent applications with LLMs. https://github.com/langchain-ai/langgraph
 langsmith - Platform for debugging, testing, and monitoring LLM applications. https://smith.langchain.com/
 langflow - Visual prototyping tool for LangChain. https://github.com/langflow-ai/langflow

**Sources:**
1. [wikipedia.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEk2GB5dzmILULV0TYrpQUl-o9-Rjcs6Rqjb2NLjjqUN503bkm2tAleknP6nbRs_-RuoG5u4tueMKGbuvaZykxN4colSgGYMptlDLLHKo_wHupCTeF0V7D0NFnYbKo=)
2. [amazon.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQExAAtF1VFu16ugTiaMFYh7A7TMkHVX0u922Hd9RV_stVmvNb7JRh-5QwtH3UcywKxm5_3pQ1QAzh-Lq6taSWYcTBvRyzoQC3oAZa87TPRN4b_v1gSfP30BtxnoAsXPag==)
3. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQErqXDoS23jyrA-ue6KbS3Nfwqg3wAvACk8EhtmuZIF08c4AcQSRYfO375t1TKzd7FGCjZdjxIg0hvPBlAPsTvtloESO-wInpAWm2z3YhPSUYTwk0S7Jdwtl2YpTI0ujyDYKxChcKS1jayQizphyg==)
4. [ksolves.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFVkxcassSx4ou9WuCkMFal91XnrnmHsIsgZNLNczdnMXDX8q6UXwIRau2kR6QX8rNLFLhYcJEznttuqjhiuUW9n_YgvemTvsvQ8IA9KMQTeH_lwtqUe_VfJ6j-tqdLjHMvPCRoaDmdtdGAs7Vk8Vum6jrNmlTUTda4HqY-FOXbtIS4y_Y=)
5. [latent.space](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHuWmxN6lS1b5zruKuJ61BdJd_Ga1P6P4-uI6TM_-pLQf_YpQWTwXeb6d3vpVLf_gCAEsJ8791770YKtjQvjo9f4H_0wWUIiknhORK9VMn2tRLCzccTPHd5OOA=)
6. [ibm.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFwjU3vObP4d4UyM6qDs8hM1UHra0Pc-AHDXXIEKeD5vqqrYeBmP17ScJPc2OaY5zIaYhrKQcwLMirpv80NH2am3xfUlXYBGopWTtYw1wOOwC2AyU8gXfCHNbyJaiQQznw=)
7. [frederick.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE8G5hqfLhRG3cf2c16f4RnUC-BtQumxnVArEcAxc2i_gdKcEHjGh0M5TCBP_DLZpud7VHqEuFtqOkLa8iVXXQY3kAtLRF6zn2ox1IydF9dKn2qO837o-cE7ojEHOhzrQ0DtdAEv5VRHFD8yHM=)
8. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQECIpEYiwNZd2-oGxb0y1CCFKau_sNCNPGqXAtEIK2BoeJuDc8UQtbH6u55tzC6DHdHCzrNykLeo03Xg5_LWQZLObU8nglKilLeaJjqxQtovA23B0rvi-XMPI9L_V-UrR610YiP6jQU)
9. [fortune.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEmKGmWvJ_ViQctx696Uk9cOgs5MrOTGMn8tQwV6HxtEYXIcZUMcaqfBdlb6ntJnLO4e26SXScCSLw8fqI1gyLGsbkbNYXCBxxWBESqk05Ix9tLkk53w71yHA93UNd3ux28k-uVIWFSL_ZDIy4s76FrygQPuQP4gk4NIuPPuhM7OW3uDWchxbaBZEuMgI7byTzrcXdui3JOFdIM-LOvA8XwiGdrWDDpbogb7l6G_bjR)
10. [datacamp.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHPllSXMYf_6lsVAZk57eFQUbDknpBVcQ0Y46o7j1V8N5Jcm32K86Ijs2KqxvOTJ3Ewtds_TovKVYYrWqfFEg-FZ5Ws10rci6C0pbxLddm3nREMNMng4ViKRdH_8TN75cygCW_wRvZGqL9biQM6hoKnFdUq_113iG6oJkhSBVarktWuhLO2l-Y=)
11. [substack.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEpOhRgcZy-K7QjAtHHcG6GOWzwM5nGor9nJBEjN9qw64L7Kne-Gb6YeTdePe77z-yhm-z5v9EMMjViduiA4ew10rhmRsZvwwTe7DgR31fdwSUjxF3EN7MKv2euCZdQLleXeAkd2zFtmDTLxNrfXvqBJ9jvGtkuC4e9ztVKX6KB7A4Rceb1sw==)
12. [webpronews.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEgmF1gCqoYEZlPrzoainmCzhRNv4ffcW0HIslnrl5uFJY4GzbPQUyzgSXre2rJ_nUvTnrYpzjNjItIVEEX85SckahzTiwlMcA7pL_vq-U_NTwWhiN7VY5B9BYJ95QHMF3L_lMrs0hwrMB0Hm29jc4Y5QFXtqnFRgHRAP27orwIC9mIs6y_j95_mUCZGTvBJwc=)
13. [theheadandtale.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHANaJ8-9g9UdzQwoUmRz6bp_78ynncefBz_6x9DfMt0p1Il33Rfrl9rVLQUC8eElDoMQ20u4JI2mX30bYlp7T8Y5t8ek5NfS1vekQKXgcLGAMN8ATNeB4T06bzUz0e9BGWgZn9XGx6N3VgRTagabrCEJYocPGhjPFracMWLtuvW13SD67O0AMe7wfaLRAirAHrUeYDMbeNucSZ87IjCvBtah4lNI6jaehTKw==)
14. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHaLF56zmFMz7UcHXcKTPAb7L4o3GY5he0HZAfTDLqcMn24FLEuqY9-98MlTVEbB0n6R6tEw-A25DxZxWL8-SK5th7C1xhK31gswlHaB4uyxiBWNLFlR3YgTIKKEPMWwL3CDMyxGdfr85FLIQqLhtXJNuYTDlMQMwJ1xTWIba-zvPJYZUCBKfXX0Q==)
15. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFBNWn8UvKq9k0rUBsECnQpeFh8rmyaLEWvt7xHav-1GpCNp4FDH1lsB66-Wg0I6Z_rpt11QD1H5QrF6pkhAC0SnY4Duw7_TRPm1SjAnJBL1vDHLc79e_t0TVmeEW4AVlqjhg==)
16. [digitalocean.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGzZCz_z1RmfE5bIyJ1FDSB1KIicM-hPpxG6HJgaDyDOfuDvDOvi982A9LPevl7ypQvIt1V55FamppI2VII2bclZ-NDfMs1jX2XGEG_S6qH2IEpFS62iJDnxDTQm6qvtxiL_UY6Y3Djs0KUrLHJK9WwBlvUz84ix407U2p-3no7K1xYujN-FtC6QH7ewzIn)
17. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHD7sNgMFBFspLJXXEnxAsmjB0DNkodGcC59Z-BfkH62-Yhb3fB2ROSJ4pAhFpqMwjp9UceH0haLJ3si_WLC7NfTvBoZJGTNJimp-Hp6HBHFz0uqZ7kJrzzCwC8M6akTvCHUt4Lvr3g5xbk1g==)
18. [pinecone.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG3Ata9f_vJGNhUQislgHKBig12nzQYnZLR15rSRC3SKNCY1KDkSXwPO7SDGK97LzUVqORrlnuoPC9fh3RLAdGrDFjmXLV4ihhITr3NLzihKrsi1K5h8RcP-EbiJ1K9eaFZ7AEqleycsI4e_w6G-6PQUS-mq9ntaY0uO8gMkaEPOrWR3A==)
19. [geeksforgeeks.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH3rgpC2aUHE3qRgyS0d_1Wnm5Yt7Spn3E0Q2-lzcDBmZrxtny9lXxUCshxHHVp2vPv7ML9WnrEg01yuD1t7V618_mIBZhQd9DSvH4M8TF_W5Yaf_7-jGcg43fv8-j-iCY1nL1lQiWEvboWjOQRF7RbR6B3IsTX)
20. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF7G487LKiArsACxBUW5RBVDFNbCVdojxD3-7EDwrtDat7aqKNsMifJcnHaQv_Mgj8B0FulktnDxhTo1l8EUQl7luyoiFIIh0no_0k-YpD5YepTN_HdkUwgWOqlPCINF7ef)
21. [k21academy.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEcXD6KIhmkFUvsy4-yWr7FNkJ16u8tJRyYiG9jjnYIli9rPb31aJ5EYDpVqKxCEH6sEVTgzWP8BGDuMjDIjFRTa-DkBPd9Ee0nw4ScW7ZBPIgWSXdtYXw0O1uFlNjvULMQdH6B7GiNqsnl7jUqgT3BSw==)
22. [finxter.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGZDMSp76gbJ8zdev9BgE-mCYX6bO0iArd2vcBRQIyBvq7j5cK9L7Zw4S030c0VeRFCk-zlkhPm6__7VNhHVSvLTfdAPS2ERfo5E625GbSmRBrCIHU2a6JOccUPo39x0mG-BxXE_itjZ3hGTHC2Gh6Zkgy5_g==)
23. [deepchecks.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGpLxNUkICRNei0b8JL9C3iN4q85UzBkEkJBywrL46C6oURYGmTIasWYAJJp2mK1vsB8CNeDKPBe5QgygS3wud2t4J-InYL8WIj8ps45e07KjBCNCnWHuitUGnPU-6Bx6PjQ6LNBc-xZJ-9qmydfeFn4JY7OPx1yzmiqXwsaqx5akXIU6Fwnw==)
24. [ijrpr.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHXB_RM25trPHRWDb9Kkzidl-vqxq1MR-MW6HSkNbZPtu9vrRqTY6cYXtQmd6DTpCCGrN5BNVdB-1MF2SzYDdnNoeUSvucQ-WqBWI1K7ubiBRIv18i4kbyPLDpyYdLl-qXpdursdJ4t)
25. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFhT_dOn9o4oqo2NVDVFEhha_FygxF8XVc9hROCvJ_9JCcnQwslQAJyWCmZmQ6294ObsmpbB1-HOBenrH2GvwtZ9jQX8XbH1Gm4zf8oSGSq3mkC6vmP5n8LhGs75VBfd1InV9B0TdNrGpuq2t0B7o-n1yO39cQ-AYlhJFBz)
26. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEAIG6-PEkcrNTMuU0_Mg1cHym_aoXYHgDTlUyEYwT5CkWhyXbJ_K4CKTb9DYKW30pTXW8BgDaPf3qPTIC8ZfzETIMyOX3Gm0w44uEzLQiTHee79UxZDrwJWizoFu8cYTi8nERaAbGdzsPNgBUTBtXuSTh7t23jzlIcwMdB9XhrCjdbgkE0-6klR7ku8ZusAzmToYfybAk3oAXMZjpoIEtsQbjrDXNID2QFcP-y)
27. [langcopilot.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE92Z0loO2ZgJWcwTzaFBKmEZPG8do2DBuuVBIrFw3rrJ9BOv_eUuNj9CgUc4VHHAX9rIft72XMD86K9JQcxfJQtUfKVvh9BaNaujavMxZjRcX1btP7sE5njKG-msUNslf0RpNCvZUYWGDhQ1zrgEPBAN0lU_QSysMHKW1YYbREqdrYIZA1WhGe)
28. [dzone.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEXY7ewl_qwoj8Tl97Q7CoXbtTy5lEhNwyG6IecZeuGCUOQwSq_HW6R-OJGTo6G6xV9-iuQJPzPBmpflh4CGXQo0xRtG3MkPAs8OxLEu-YmdJj2YFFLJTSlehT3tSorfjqYTBSVyoIQW3zSfrd9RB_H8nYXMshJRV9-u9hLnKoaeg==)
29. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGEMXtRpPWxtJZLvwu6sqTuDnG0ikDLyaT8EHYfRcBNw2bntMXr7tMkuROdN0QHLnVvurwKKoFNhT9Hu5hsTl1F1QWVzE62vPat60-BulwH1kndAH7wKte8ootps-QTkalKKCGzQCisXLRUPW-iaNrtA2AUkH6pI9G51pPeTdHEfMt897I7Xy6nxYPy)
30. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEuEMCOaRT6fnrZ2yFOqqtmo_9mfgC-B59fFsuXqakGbpQFutn6viPAg9p-aJldJBR7gOmuj7mrp0KB7xdB3O-kFagI_LSqPVYZnPUVCUyPth2-aWW21qzLInwjBg8z9gEkZ6YHvbGAtc5x2wJHe_xYeDewOg1Vk12xHLh_thB4qErmPesq_CxI0sRUPXzZ8y2PKxAvYNNvoZPuO20=)
31. [duplocloud.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF1yzgjLG4IWKHVt11ZY-EXThXsqoAhDqyeNFK0NVHi5SvuBiu_CpOpVw_4eDFOeenI_BjvMuyx14dFCNio-aQGpGCzV9vav4nbO4a0Olygv641T2eDMzHpGUqnk2sOU641enfHcUBaCkw=)
32. [milvus.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEe9e38AJXbf_D9kZKX3e1hmBYkHLE4_rnAO0Wb7wEq3lJHgLxm3MwxivqSu6q5a86EvM45bcsYCPQzYjqurX4tXFH-2SZVSzWGr3qZ_3UIZ9CsVYrFGoVgafOvxkEspJXevHXlhvM=)
33. [geeksforgeeks.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHucltwyml_Y8g67vsIcxPgUO9m8tIp2wcE3BISeNL639VHcvzQ2a3NOCCdfIL9MmKeJPhu_39ZFCbttqRV4QQUPlhaJO5jO3KpoLu3RoDQvvpMgm5fKqQrB7N1NPe4wJmUut9jZ_axC2TWFajXA9i6KWhkiLdn9Z729QT8xaVrYqbaow==)
34. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGpVu19gy1T0Q0kvNLupij76UGMZXPCoGkOyfm6JGPkqCSEeCGqW0pZyW4dCgK4mz8W_1i2kWNhQPmU-Sz3f5GAZkoZ_RG4kZL2gZXzrec-pw==)
35. [galileo.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE-gcH4EKymIhxmhiob-18hIas0u2lczZbEA5yrYYzn5CppSeAQg7adkMcmJH33BPJ-fflguoe-apHpf7sweRVnZwG30hT2Vlrzoje5g-FP7xUxaR1AXYQWcy0Me6SFxHSexoxfVnvLCe1L_W7d8W5aaw==)
36. [analyticsvidhya.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGJlj4zVzsSGtBopD9unrI5b_84u5ikT52APW_Nzl-PSpHc9k-Qo3rjChzu-x1oIt5PEukOFwhl7yATmy-zWI-sQyhzuvSaspoyK97qIY7GBS_FZhlHws2Yocscg29jtqxx95UYZ5tCnaD_j4NzVUCqlY6hA6fVDB5YOZejmQ==)
37. [oxylabs.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF_cPn9sYffEkRLgrzLTiiZH0voRDGt9DnyszUUeSC7EKO6CVcYbtIIh__9a1YsHtNwKePoNTE6KG5Etjid1OVL69UaIlOVJr816WZMNR4nqmRlbs-iZixgMy0MyQ-hPAxTfB2g)
38. [dev.to](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGQFzVW4nejeRnX2JN5lKolLtEdzuDz3leS3AZXBQZChnhxJHqyR4eZedOLg5LcPrwZWZz-uV8DrbQjwMDjLM9XXUlGKJ3qWevwaaZ74hHdUnk3Av6GMMDk7TGo5dpYQKL67Q8cXEZ45Cd7DMSSi83Jynhq8ViW7R4yvxSSRJSy)
39. [web3universe.today](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFc3kuVRj4lF-hssiykG9ZLRl_QA3CSd1mJ9l9brsdfcDluohuTcIcLwTZbRgHun0owlHWZOz8dPFf8jawUxUhbBD09rMJqSCZIHytTuqttybVrqxjJXMrj_C5BuMCG5K3PXppliZjwrvDmkLE=)
40. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHnCE2TpgqohChmnK02uBsZlGzcbhzqgwNBP-BlTCAf3m4NLCDbyIypIDacH8PoSo4tK3DeLXffuz-kGM8hpZV-nxbNzVc0PPlSG1_bCqBg-EXXUNMjsuf1keUz8toKR4zOV8fgzLV_8FIOPpgMYwTyIBIDIPn_F1z1G87xUkajODsfsk8Kj3NQmUtr50zlo-EyKxPw9rIjtydTndfT1FJAYWRaTTQmBs893vmuVVdH-a9-e_4=)
41. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFb6L3xdfCXTjEZeEmmlV3j3rFhMkUEa3DZuXvDsxod8THAA6wKA-475zGQzsAmxJk4mxY_j2ha006koaTdki4F2usW_ocGmU4XE7ogjyEpS-GrdLfI-UFvXs9VZWOLpsgqcQeGzQzcBPHd9f-i02SYhAAO5yaxVlf2C6L51u5fWHR0UpAEchvtY26Vw7nxqu9mgqpNi8Z4UQ8r-HUwqMsyJr8ky3p0KikJPSCp_dvA8TPDSyIp66M=)
42. [cyata.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGbbQdLy7pTx_KJIM1MgO1ih0QZDXqSboDUokLMqbgQDteC61rbSLpklPOvHysccS3EGs0XiWdDnQQAqR-Fp3TGr7Vt4R4UIlUdiJpZR4rG7Mhd4tFR9-v3CJUjTyfwmiEaAIZuBzuRwvTww0zDjKl6xdf5cYY=)
43. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG-1x1rHIWbA5wD7VCz00XXM9b02yZHmcQKZWMrssxeGmCFPWlHUb4tWTW6_46LaUgo4ohts3WMvjJ1Lfek91B0T14hW8eI8_NuGkHYllJ55sGbPxbIQUbeeudar5Yrc0e1gWsjmA9zgYLLZQ==)
44. [reddit.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFa_OivJVFounkbMd4pUyIiZXRtbcjvXHJFhqvlX3avfieQ-jhb6sLHND6jYQMwJamfoMfqD5lg9ebVXW-HdlBt2_ECziWyErz4rAGOzNb1ZcFO4MDt689d7P6sfB2Z5Mn5RQ-tcA1wHXK5Ov6O9qCwSWDWhuMlLFwxxRvzzhB4x3f6YWQkD18mEy28vFthb6tDHVmb)
45. [metaschool.so](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHH5qbWn7sOsOOpFD4yd_nBYyi6ujH1RItMoJD7hC0Jw6NC7aKijMRkItYNLzpuLT5IQQCenyB9GqqoGvDHkZ6u3hBYyRzM2BO6yCr6TSb3ImuitYCVYACrnkPf2kVtpVL10MTknQOavvvUCc2lGesmInFctB6kLZCVhQ==)
46. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFB2jubplSfACzuFifO4yox_MyDLp5qnjNsB9ogHmazUu-zxwDLBrg1JtQk-eKiGq2QSrkI8uvgnzjR0GiY8yrRyzHjdRB9oUlJcBX19NUsdlCVHiJmp4IQr9QCaiNJ1zq3BSeISqttSwVbenQ2JWJ7hNUfTztlBy7tVoLNJ6d_HOE2WfbLfA1YJm8eYCclK3c3c6DhrxEeOf_jIwG7TcdfDg==)
47. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFffZb-ElDd7FYxACCIisnZJcWtA7PGFENugqbZ8UrvxX96oM8E5feKBRcEEir8--rrSuGzTq6BpU8kamArAqsk22EdHaIGsnH222EwsU0QJvvHM1H-ww==)
48. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH-PJImfkloymtq69P6LkoGMc5ctkC_SckWXho3jsc4zU7zDkH7SDK9oYwjr2Dnbo39gXsXZRQuS0eqSY2Tno84oGTWmRoGqnbN9qeegnZGBuIEed09jw==)
49. [google.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHTqEHZCWc2fAFQvFEZSihTLjFTSHOAOBriecVKBcoY7meeil-Qx7rJYxxKMF-u6sA42Y6X5uOn5ikQs6EtBWNhd8VkHi-2tx6R9g-ZRw_V6fDU72Fg9VIT6J-tK-fUwZMOzA==)
50. [decube.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGtBGXUUJkL4AKThIxK6XJ-qAJLe_keKUA9wq621Fav9TVNmSgzkEa3L7JL4TvaYvHSxQlNEZjAwmEav_l5uCSqmYdWNc9ung1l59twxoxlbNA7ff3Iyq5nGrgDu8iAqKA=)
51. [dev.to](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE9F3lWU5wp9McSpQsMvPm1WM_d-S_Z8SwmcNGeqmeZ1pvNeQDiIyIfUcw2t_hMi06LwI4dhA-vPPqV363tI-HGQOnbLAb4xUknBv5reeyQWbyxREJ0cEGxIp9LXpiBN2BM-jaYnROG8hJdK5vFq_mcgdeug3_EsdjW5OS5DcZOS1s=)
52. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEgy8UOYptttxZHWvRdeAz3yORcyFWMROVyAoEX_WSHVqfCkWsRT5Xc1hVfDoymqhsrGnE5Bw0Sc3WzkPC8QfMMXrsejLA5xp0AWdRvKGQN7cQFy2PP7CEDvNJgA2z0humEOaJS1R0kVGymUZ_7MwxvN9MywRdGWHNIBXEF-Y_GRjNOGDcXqe231bN3OV8SqSvt6TqvHQUJeC-rWf9LoNGcFJbsKeXgKhy7HwDofUWz6MpTPSvKC8D9x4OBrmI=)
53. [arize.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEEaz0yldnMGxiI68pu6qhi2wDB0hKz9aVLGXCbx_bu8ByUBNRYljbRk27GtJE1gAKITbOFZMIeVWyu4btpMGpvRlKI3F4eioe34Rm63MatlDmkCH5aJ-DBdw2z)
54. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEV5eHlM4dwFE1Qtj7rRmIIQu1zQjuA-BkOSeVPJEsXCdaDzWd3khG7BcSxDp5PVGyWjnyQsXs3h1Z86S1i9WDezXoDhh9WddGzu3LHzWHuPS9CYIPSTCQTBH5UTnrz9p3eJ9uLMeU=)
55. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEmp1rqq2nf4UzSzF3qrwelwgnlQfqgkWDVXwvZQjK52yQhsJ3khFl1aDySpYc8EDwbAyQFz1PPPy3LfaVTiWPo-DP7_O0n36Iel1_xraFJmdsp6LJuua93V3bBmcNLCWgN8JAuzoYr9aOvUASiiuCecGzecIxeY0Nukf_JlbjLaBd__9wbT5HTfK0QJuMt)
56. [langflow.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGf_weTknNw7hVflZL-UabkTEe2HYMarEKGDulRjNr5gdsmMvBa4Xmv93MwtvAZUpGJDjXyphOK4-II1cR4gk2JpwmodCovYf54iybkRy_dOF_bQf5R1kZBnVyeErieV0_jSn6DkiY=)
57. [langflow.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE6GlynMhKiYVRVUEmgQ0wcO_ZwjyrY0bZdT48Uec9C9wR768kfWBlxKxHpinEn80jEX6ud6NtvtPXsVNzzfXHz92RR377eOGoOq4imawsPiA==)
58. [langflow.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG8u_w-MIaVaZozrBVevVBy7Vkx3amIUdTyWVDC_CcTgrRMPIpyKjopAr9JNLPY-6XuuvH3pK1akh4ygE5sZqPeSVsJc2npHGcGl-3J7DxC)
59. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGK96AxFpks9tn6OhC5Kzo8WPgqAuvC3xbw-vH10IXeQQtJDCriQsr0ELFGPaZSeuqFKYqIPck0b6yqkCoRgngzbdRAWK-gsIjuCVcihPYI8DWLzTxZD0SVyfpqSC0=)
60. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEfiWbVbSIy45ZC3sJqnY5mQziWfY-YuuQyNfSg_RDb1W5hOx6i9ikZdEhK-ifxeqXB1g9IgsrhX7YNYoKLO0Kjky0UDUUIpk7iIY2NcakK0lHdgg1G0ZYaOYREhAGQ6A0Fhwg6lsAELkQni_6sgHwbT4Y=)
61. [ijset.in](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH7jBd6rTJSjxtpzNTe6FAGUE_-2h09h3GSJimkTYpyrzeADxk-7kvkXbObby85QXtUGWttA260YrKmilL8Nm36-Q-A8sYQJTGTFz2PEJpn_7ZqBRkOrIk5WYNakofQX_HtMEhgU2AE7JKTTlNj5bKBYMlQFkfv)
62. [readytensor.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH05WSmrtMg_lbATjKnuiTtpQ-ShTgTp6dVS0WhtBgCLA7pCxva1prFJdvUtig_hOHMHVfBR4luKT8SPnQvn7NTxltfiHzP4XHcuSof--3z7C7uRuU72dSiMR8b_ULmpruDCHTzlF9y2I50vysH7bpYCv-5E7dtxXj5CwbTdX4SR5djLMg0ieLC1AgeVUQ=)
63. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEklxHFt0-sk5YSnsEH1kZ8T12KHkABBykZQhTpMPrnK361OSjyY9MKtx1BI7ud0dz0mKFrnhLs_-C_xQSq4eY2aq1I1aD3ssdBarDKfFdRJYFfNqPRn5lqe8nx5kZMZQxAHiHpMa0OaZ38AYt4SohkXyA=)
64. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHRt0GRHJLvdSTFrpUz4ItEN3-FUmBc1RPHfWKRU9fj89FVw3YurWcG3rbGKgcm4npuHPyIi8f3Mo18eMEnh6Ct1RPg5UUfBzjYmVcmiLasA068xIv7rZyUB-eJPXDEQBFHFlmOHj8x-8SrZLTGpnHCPUQ=)
65. [tracxn.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGJZ0Wem1SZ54nFkEFs6PtD4bcc4c1H_NtjP82QJwMR0hDJn1kYdAH9t4gqcrHPiBwwWs3Mj1bTVKWN28UamBF0oAF-5nqWGVrYtK7N_untww-hB9F6O8HOzTis9BT-YArBfrmEPNYHP9mosO_fX9foy-K2Kt7BHRMXTFq47F27BrYsl_7Se4G-QuzR9Q==)
66. [lakefs.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEgJuzLHLSvG7SJCN69MNIGiVYgq7TFfnzc6QEzuTgzdsnCQMnuGoU7ZR4MyNyO3aCmc2k-aYPQ6JBlCTBBOOadWzNsFSNnIykBWCm0TfEOG5LlbGghxZo7QGJ9ZaWArYMa0b6xtFa4q0Wjz8x5ozQ=)
67. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFdy9lEKvyO7kVaEP7bciQUtrykdE5HwLv70WTrZTPVV1BLIXbuYyQqsHIDpZMOMzIOD6rkOy4L41Gf3H_bvp8y4tAKO_EoqDDzv4ThLwcOorAxNxqJgFK0sRXCdm8P4sg8)
68. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGrwS-oGtMpUFD5pVPkz6ch3Biam5Zxj1r1S47LiZnVxwp57ywQeTrEs7wetcUDRB1xq-zT3wqzGZd7nKSeuOVjBvg-gTbJ9mToNwnwHtpVcVooKzmghA==)
69. [nih.gov](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGd7oYLjCHLWMNHCV_3SnBbH_7GeMJGCU4H45PhQBoJUikgMVYBVLmpxGIE6EF4aKOW-Eryb-xcapHLhJEZ6cbEVcPKPtv4VKYXrI-ghqofNpE_RhZLB4_HfmzEzIgs0NS1io4AKkoRAw==)
70. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGlJ9-_ZUG01Dzq22gujiYTPzQ5toWdGAb9X-bk_EAc6noyxk06Y04i_s5fWuteLMSy5K8iogE7QCPGDntd6aCFtdFXNWT8JUPllAtzhS1BQzfM2Z6cetJSig==)
71. [binaryverseai.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEQ_aGzLttgUACZyyo0sn9bdJ807HfsgK4zrWT8WiUjxMOGvXBlTPAID_UPGuc8dMeU6SVAoxeRdx2zy3m4cpyyFZDzeBdHdsdGNzCTysy8oWP_crq-sf3J5wXU4qHYgSwHRRI-3mbYYRtS6gtNpwe8ZRGt--q2i1HH1PVGmMdWHQ==)
72. [siliconangle.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE84Ckks9QxrYfHswPrK9hcQ5FYzpI7v5kjnPQipA_wcv5ahEy3VXtilMclKzAouGkvUuPQy6DsamimyBIqVeF4Ab1E1CM-i0Wjd7oc_c1pe0baqlD-a4hTJ5n1G5P_rztuvuMEUEWiyMN0jSmkHms2NMFE88mtkYMmnv8OtjiJp_utcvy9XSf9tRG2snMCvFiLbEXla1FplJVe)
73. [pypi.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH6G_bGA29UvxODtH747DLdUI91Qr723oIIK51vmyJblfFKgaO9cyycvnFsaE_rYkJqOAJQzrdIIJF7lXrI20MqeUatQ_infgGh3grvY0IIrLnK5to67Hr96Q==)
74. [github.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFrdsNitYAdBr-Cnokf_URC5_cEV0KFf3jKvgd1PyTT4Q2J-GZfKHhJpJ7LXXSVaf0z584y_ct5jCmHJ6ESaD6PFpyyTxTpOVuKpyGebQqg6ziakiu7UqkTwDjS6RlWZNlk9J9askSNDg==)
75. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5FsyuPtt2gFEp1_3YzlXNaznPkwU3uUyYUTiSFIlTbVD9h68IdEY4oegF12TiaYS74IpgSroxlWIBXi-Bgt1uwGMlIknLPEL8h4DrpJPHYG7Relj5o_dRqqbq-L8_aN2sIvItOVhX9OLg)
76. [langchain.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHiE6pB5Odi151soCKFhazAd4OXfWMIPRnvFzrHAzKgAEHEezxm6Z0R6fgIRYAZZGoJTeeibdI3B7iB2FHBsf_NV8rK4ra1dxw7nEEwGerjx80qkiFcj74jiy8lvmY=)
