# Comparative Analysis of BERT and GPT Architectures

**Key Points**
*   **Fundamental Architectural Divergence:** BERT utilizes the **Transformer Encoder** architecture to process text bidirectionally, allowing it to capture context from both the left and right of a token simultaneously. In contrast, GPT (specifically GPT-3) employs the **Transformer Decoder** architecture, processing text unidirectionally (left-to-right) to predict the next token in a sequence.
*   **Pre-training Objectives:** BERT is pre-trained using **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**, enabling it to learn deep bidirectional representations. GPT is pre-trained using a standard **Autoregressive Language Modeling** objective, optimizing for the probability of the next word given the previous context.
*   **Usage Paradigm:** BERT is designed for **transfer learning via fine-tuning**, where the model's weights are updated for specific downstream tasks (e.g., classification, question answering). GPT-3 introduces a **meta-learning** paradigm (in-context learning), where the model performs tasks via text prompts (zero-shot, one-shot, few-shot) without updating its gradient weights.
*   **Strengths and Applications:** BERT excels at **discriminative tasks** requiring deep understanding of full context (e.g., sentiment analysis, named entity recognition). GPT excels at **generative tasks** and open-ended text synthesis due to its autoregressive nature.

---

## 1. Introduction

The landscape of Natural Language Processing (NLP) was fundamentally altered by the introduction of the Transformer architecture. Two of the most influential derivatives of this architecture are **BERT** (Bidirectional Encoder Representations from Transformers) and **GPT** (Generative Pre-trained Transformer). While both models share the same underlying mathematical foundation—the Transformer's self-attention mechanism—they represent divergent philosophies in how to model language.

BERT, introduced by Devlin et al. (2018), focuses on creating deep bidirectional representations by conditioning on both left and right contexts in all layers. This makes it an "Encoder" model, ideal for understanding and classifying existing text [cite: 1, 2]. Conversely, GPT-3, detailed by Brown et al. (2020), scales the "Decoder" approach to 175 billion parameters, focusing on autoregressive generation where the model predicts the next token based solely on prior context [cite: 3].

This report provides an exhaustive comparison of these two architectures, analyzing their structural differences, training methodologies, input representations, and usage paradigms based on the foundational research papers.

---

## 2. Architectural Fundamentals

The core difference between BERT and GPT lies in which component of the original Transformer architecture they utilize. The original Transformer consists of an encoder (reading input) and a decoder (generating output).

### 2.1 BERT: The Encoder Architecture
BERT stands for **Bidirectional Encoder Representations from Transformers**. As the name suggests, it utilizes the **Encoder** stack of the Transformer.

*   **Bidirectional Attention:** The defining characteristic of BERT is its "deep bidirectional" nature. In the self-attention mechanism of the encoder, every token in the input sequence can attend to every other token, regardless of its position (left or right) [cite: 1, 2]. This allows the model to build a representation of a word based on its entire context simultaneously.
*   **Structure:** BERT_BASE consists of 12 Transformer blocks (layers), a hidden size of 768, and 12 self-attention heads (Total Parameters: 110M). BERT_LARGE scales this to 24 layers, a hidden size of 1024, and 16 attention heads (Total Parameters: 340M) [cite: 2].
*   **Mechanism:** The encoder reads the entire sequence of text at once. This holistic view is crucial for tasks where the meaning of a word is heavily dependent on future context (e.g., disambiguating "bank" in "river bank" vs. "bank deposit") [cite: 4, 5].

### 2.2 GPT: The Decoder Architecture
GPT (Generative Pre-trained Transformer), and specifically GPT-3, utilizes the **Decoder** stack of the Transformer.

*   **Masked Self-Attention (Unidirectional):** GPT is an autoregressive model. In its self-attention layers, a token is only allowed to attend to positions that precede it in the sequence (left context). Future tokens are "masked" (set to negative infinity in the attention score calculation) so the model cannot "cheat" by seeing what comes next [cite: 3, 6, 7].
*   **Structure:** GPT-3 is a massive scaling of this architecture. It consists of 96 Transformer layers, a hidden size of 12,288, and 96 attention heads, totaling **175 billion parameters** [cite: 3, 7, 8]. This is significantly larger than BERT_LARGE.
*   **Mechanism:** The decoder generates text sequentially. It predicts the probability distribution of the next token given the history of previous tokens. This design aligns perfectly with text generation tasks but inherently limits the model's ability to incorporate future context during the encoding phase [cite: 4, 9].

### 2.3 Comparison of Attention Mechanisms
The critical mathematical difference lies in the attention mask:

| Feature | BERT (Encoder) | GPT (Decoder) |
| :--- | :--- | :--- |
| **Attention Type** | Bidirectional Self-Attention | Masked Self-Attention (Causal) |
| **Visibility** | Token $i$ can see tokens $1...N$ | Token $i$ can only see tokens $1...i$ |
| **Context** | Global (Left & Right) | Local History (Left only) |
| **Primary Goal** | Representation / Understanding | Generation / Prediction |

---

## 3. Pre-training Objectives

The architecture dictates the pre-training objectives. Since BERT sees the whole sentence, it cannot use standard next-word prediction (the model would simply "see" the next word). GPT, being autoregressive, naturally fits next-word prediction.

### 3.1 BERT Pre-training Tasks
BERT employs two unsupervised tasks to train its bidirectional representations:

#### 3.1.1 Masked Language Modeling (MLM)
To train a bidirectional representation without allowing the model to "see itself," BERT introduces the "Cloze" task, referred to as MLM [cite: 1, 2, 10].
*   **Procedure:** 15% of the input tokens are selected at random for prediction.
*   **Masking Strategy:** The selected tokens are not always replaced with the `[MASK]` token. To prevent a mismatch between pre-training (where `[MASK]` appears) and fine-tuning (where it doesn't), the following heuristic is used on the chosen 15%:
    *   **80% of the time:** Replace with the `[MASK]` token.
    *   **10% of the time:** Replace with a random word.
    *   **10% of the time:** Keep the original word unchanged.
*   **Objective:** The model must predict the original identity of the masked token based on its context. The loss is calculated only over the masked tokens [cite: 2, 11].

#### 3.1.2 Next Sentence Prediction (NSP)
Many downstream tasks (like Question Answering and Natural Language Inference) require understanding the relationship between two sentences. To model this, BERT uses NSP [cite: 1, 10, 12].
*   **Procedure:** The model is fed pairs of sentences (A and B).
*   **Data Generation:**
    *   50% of the time, B is the actual next sentence that follows A in the corpus (Label: `IsNext`).
    *   50% of the time, B is a random sentence from the corpus (Label: `NotNext`).
*   **Objective:** The model predicts whether B follows A using the representation of the special `[CLS]` token [cite: 2].

### 3.2 GPT-3 Pre-training Task
GPT-3 utilizes a standard **Autoregressive Language Modeling** objective, often called Causal Language Modeling (CLM) [cite: 3, 4, 13].

*   **Objective:** Maximize the likelihood of the next token in the sequence given all previous tokens.
    \[ L = \sum_{i} \log P(u_i | u_{i-k}, ..., u_{i-1}; \Theta) \]
    Where $k$ is the context window size.
*   **Simplicity:** Unlike BERT, GPT does not use NSP or complex masking schemes. It relies purely on predicting the next token. The authors of GPT-3 note that this simple objective, when scaled with sufficient data and parameters, results in emergent meta-learning capabilities [cite: 3].
*   **Data:** GPT-3 is trained on a massive corpus (Common Crawl, WebText2, Books1, Books2, Wikipedia), totaling roughly 499 billion tokens [cite: 8].

---

## 4. Input Representations

Both models require specific formatting of input text to function.

### 4.1 BERT Input Representation
BERT represents a single input sequence as a combination of three embeddings [cite: 1, 2]:
1.  **Token Embeddings:** BERT uses **WordPiece** embeddings (30,000 token vocabulary). It splits rare words into sub-word units (e.g., "playing" -> "play" + "##ing").
2.  **Segment Embeddings:** Because BERT is trained on sentence pairs (for NSP), it adds a learned embedding to indicate whether a token belongs to Sentence A or Sentence B.
3.  **Position Embeddings:** Learned embeddings to indicate the position of the token in the sequence (up to 512 tokens).

**Special Tokens:**
*   `[CLS]`: Added to the start of every sequence. The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks (like NSP).
*   `[SEP]`: Separates Sentence A and Sentence B.
*   `[MASK]`: Used during pre-training for MLM.

### 4.2 GPT-3 Input Representation
GPT-3's input representation is simpler, reflecting its single-stream nature [cite: 3, 14].
1.  **Token Embeddings:** GPT-3 uses **Byte Pair Encoding (BPE)**. This allows it to handle any text string without "out of vocabulary" errors.
2.  **Position Embeddings:** Learned embeddings to indicate token position. GPT-3 has a much larger context window of **2048 tokens** compared to BERT's 512 [cite: 7, 8].

**Differences:**
*   GPT-3 does not use Segment Embeddings because it treats input as a single continuous stream of text, even if that stream contains multiple "sentences" or "examples" (separated by delimiters in the text itself).
*   GPT-3 does not use a `[CLS]` token for classification; instead, classification is performed generatively or by looking at the probability of specific output tokens [cite: 3].

---

## 5. Usage Paradigms: Fine-tuning vs. In-Context Learning

This is perhaps the most significant divergence between the two research papers provided.

### 5.1 BERT: Transfer Learning via Fine-Tuning
The BERT paper proposes a **Fine-Tuning** approach [cite: 1, 2].
*   **Process:**
    1.  **Pre-training:** Train the model on a large corpus with MLM and NSP objectives.
    2.  **Fine-tuning:** Initialize a model with the pre-trained parameters. Add a small task-specific layer (e.g., a classification layer on top of `[CLS]`). Train the *entire* model (all parameters) on the labeled downstream dataset.
*   **Implication:** For every new task (Sentiment Analysis, QA, NER), you create a *copy* of the BERT model and update its weights. This results in a separate model for each task.
*   **Results:** BERT achieved state-of-the-art results on 11 NLP tasks (GLUE, MultiNLI, SQuAD) using this method [cite: 2].

### 5.2 GPT-3: Meta-Learning via In-Context Learning
The GPT-3 paper introduces the concept of **Few-Shot Learners** without gradient updates [cite: 3].
*   **Process:**
    1.  **Pre-training:** Train a massive autoregressive model.
    2.  **In-Context Learning (Inference):** To perform a task, you feed the model a natural language prompt that describes the task and optionally provides examples.
        *   **Zero-shot:** "Translate English to French: cheese =>"
        *   **One-shot:** "Translate English to French: sea otter => loutre de mer. cheese =>"
        *   **Few-shot:** Provide 10-100 examples in the context window.
*   **Implication:** The model weights are **never updated** for the specific task. The model "learns" the task dynamically by attending to the examples in its context window. One model serves all tasks.
*   **Philosophy:** The authors argue that large-scale language models develop a broad set of skills and pattern-recognition abilities during pre-training that can be unlocked simply by showing the model a pattern at inference time [cite: 3].

---

## 6. Comparison Summary

| Feature | BERT (Devlin et al., 2018) | GPT-3 (Brown et al., 2020) |
| :--- | :--- | :--- |
| **Core Architecture** | Transformer Encoder | Transformer Decoder |
| **Directionality** | Bidirectional (Left-to-Right & Right-to-Left) | Unidirectional (Left-to-Right) |
| **Parameters** | 110M (Base) - 340M (Large) | 175 Billion |
| **Pre-training Objectives** | Masked LM (MLM) + Next Sentence Prediction (NSP) | Autoregressive LM (Next Token Prediction) |
| **Context Window** | 512 Tokens | 2048 Tokens |
| **Input Encoding** | WordPiece | Byte Pair Encoding (BPE) |
| **Primary Usage** | Fine-tuning (Gradient updates on task data) | Few-Shot / In-Context Learning (No gradient updates) |
| **Best For** | Understanding, Classification, QA, NER | Generation, Completion, Creative Writing, Reasoning |
| **Dependency** | Requires task-specific fine-tuning data | Can operate with zero or few examples |

## 7. Conclusion

BERT and GPT represent two complementary approaches to leveraging the Transformer architecture. BERT demonstrates that **bidirectional context** is crucial for deep language understanding and discriminative tasks, achieving state-of-the-art performance by fine-tuning on specific datasets. Its architecture allows it to "read" a sentence fully before making a decision.

GPT-3, conversely, demonstrates the power of **scale and autoregression**. By training a massive decoder-only model to simply predict the next word, it acquires emergent capabilities that allow it to perform tasks without explicit fine-tuning. Its architecture mimics the human process of producing language sequentially. While BERT requires a new model for every task, GPT-3 proposes a general-purpose interface where the task is defined by the input text itself.

Together, these papers [cite: 1, 3] form the foundation of modern Large Language Models, defining the spectrum from specialized understanding (Encoder) to generalized generation (Decoder).

---

## References

### Publications
[cite: 3] "Language Models are Few-Shot Learners" (Brown et al.). arXiv:2005.14165, 2020. https://arxiv.org/abs/2005.14165
[cite: 1] "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al.). arXiv:1810.04805, 2018. https://arxiv.org/abs/1810.04805
[cite: 8] "GPT-3 Technical Overview" (Wikipedia/OpenAI). https://en.wikipedia.org/wiki/GPT-3
[cite: 15] "ChatGPT's architecture: decoder-only or encoder-decoder?" (StackExchange). https://datascience.stackexchange.com/questions/118260/chatgpts-architecture-decoder-only-or-encoder-decoder
[cite: 14] "Autoregressive Models for Natural Language Processing" (Medium). https://medium.com/@zaiinn440/autoregressive-models-for-natural-language-processing-b95e5f933e1f
[cite: 16] "GPT Architecture" (Medium). https://medium.com/@prashunjaveri/gpt-architecture-0415e7a5796d
[cite: 17] "Meet GPT: The Decoder-Only Transformer" (Towards Data Science). https://towardsdatascience.com/meet-gpt-the-decoder-only-transformer-12f4a7918b36/
[cite: 10] "BERT base model (uncased)" (Hugging Face). https://huggingface.co/google-bert/bert-base-uncased
[cite: 18] "BERT (language model)" (Wikipedia). https://en.wikipedia.org/wiki/BERT_(language_model)
[cite: 19] "Masked Language Model: All you need to know" (Medium). https://medium.com/@amit25173/masked-language-model-all-you-need-to-know-12ab35319d09
[cite: 20] "Core Tasks of BERT Pre-training" (OreateAI). https://www.oreateai.com/blog/core-tasks-of-bert-pretraining-detailed-explanation-of-masked-language-model-and-next-sentence-prediction-mechanism/b9d97c99e74bdad0fb17224f8a5e3417
[cite: 21] "NextLevelBERT: Investigating Masked Language Modeling" (Czinczoll et al.). arXiv:2402.17682, 2024. https://arxiv.org/html/2402.17682v1
[cite: 4] "BERT vs GPT pre-training objectives comparison" (arXiv). https://arxiv.org/pdf/2405.12990
[cite: 13] "GPT vs BERT: A Comprehensive Comparison" (Medium). https://medium.com/@vijayjun89/gpt-vs-bert-a-comprehensive-comparison-of-two-powerful-language-models-df27c2b45733
[cite: 22] "BERT vs GPT: Architectures, Use Cases, Limits" (Scrile). https://www.scrile.com/blog/bert-vs-gpt
[cite: 23] "Pre-trained Transformer Models: BERT, GPT, T5" (Fiveable). https://fiveable.me/deep-learning-systems/unit-10/pre-trained-transformer-models-bert-gpt-t5/study-guide/o8JLDj9oFwOSdcRt
[cite: 24] "GPT-3 vs BERT" (InvGate). https://blog.invgate.com/gpt-3-vs-bert
[cite: 9] "BERT vs GPT: The Ultimate Guide to Encoder and Decoder Models" (TipTinker). https://www.tiptinker.com/bert-vs-gpt-the-ultimate-guide-to-encoder-and-decoder-models/
[cite: 5] "Transformer Architectures: Encoder vs Decoder Only" (Medium). https://medium.com/@mandeep0405/transformer-architectures-encoder-vs-decoder-only-fea00ae1f1f2
[cite: 25] "GPT and BERT: A Comparison of Transformer Architectures" (Dev.to). https://dev.to/meetkern/gpt-and-bert-a-comparison-of-transformer-architectures-2k46
[cite: 26] "Foundation Models: Transformers, BERT and GPT" (Heidloff). https://heidloff.net/article/foundation-models-transformers-bert-and-gpt/
[cite: 27] "Architectural comparison of BERT and GPT" (ResearchGate). https://www.researchgate.net/figure/Architectural-comparison-of-BERT-Encoder-only-and-GPT-Decoder-only-models-showing_fig1_391181536
[cite: 3] "GPT-3 architecture details" (arXiv Snippet). https://arxiv.org/abs/2005.14165
[cite: 1] "BERT architecture details" (arXiv Snippet). https://arxiv.org/abs/1810.04805
[cite: 12] "Basics of Language Modeling: Transformers & BERT" (Columbia Univ). https://etc.cuit.columbia.edu/news/basics-language-modeling-transformers-bert
[cite: 28] "Everything Product People Need to Know About Transformers (Part 3: BERT)" (Towards Data Science). https://towardsdatascience.com/everything-product-people-need-to-know-about-transformers-part-3-bert-a1227cead488/
[cite: 29] "BERT base multilingual uncased" (ModelScope). https://www.modelscope.cn/models/AI-ModelScope/bert-base-multilingual-uncased
[cite: 2] "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al. - Full Paper Text). https://arxiv.org/pdf/1810.04805
[cite: 11] "BERT Paper PDF" (CSU Ohio). https://eecs.csuohio.edu/~sschung/CIS660/BERTGoogle2018.pdf
[cite: 6] "GPT Architecture: A Technical Anatomy" (Medium). https://medium.com/@DogukanUrker/gpt-architecture-a-technical-anatomy-e06c2f10f5c7
[cite: 30] "The GPT-3 Language Model: Revolution or Evolution?" (Orange). https://hellofuture.orange.com/en/the-gpt-3-language-model-revolution-or-evolution/
[cite: 8] "GPT-3 Wikipedia Entry". https://en.wikipedia.org/wiki/GPT-3
[cite: 17] "Meet GPT: The Decoder-Only Transformer" (Towards Data Science). https://towardsdatascience.com/meet-gpt-the-decoder-only-transformer-12f4a7918b36/
[cite: 7] "A Primer on Transformer Architecture" (Viks Newsletter). https://www.viksnewsletter.com/p/a-primer-on-transformer-architecture

**Sources:**
1. [Link](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHib0bkI2jyFTfZSgFxyC5pN2V06eYQDUTUZmOuagZyFePIgSeAv7XX6B5T1QSegtWNDHhZwTw692EOyC7wKb4IhWabd9AeaYAFesNQ0HK6LrmbKihNpA==)
2. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFg-2b_sQpyS4JaFlZ0xRlkBEIjk8gJSCwfQndMEvPdPBv6EXAVa_xXZ7RCmm5mtHEoQgWIZwW-Cb8MmCvO8tVAQ5uy9uBzvvgxc1cvgveh3JtJXmSZsA==)
3. [Link](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGXAqYb-ph9QEyznmXv9Lc870vlZ_nb7aGkYeM6zmgVv-q0yXJsgbB89HdhirbfLbQSoopRzPUTVO-RA_tFVAbwd0JS6znNhd2GOoo2llCHtOBuiwxaxw==)
4. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEygRNyILMpqT0sGR0qe-arHn2Pq3lSfiV9R2fPVlopC9TjbvyC7x2PqmOOgvrcyZyrIxOtMjAuqnLz2SPEXH_G04CwjBWFVbthtfdjV9Xz0LI2lSyHHg==)
5. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFOKFNKjeLwmW2L7CKKCK5yFbSLy797CylLEtK1IDhPYP5keuLCatJBIjFCUGssKhDlOcNqlFICxBJHBQyuvEQr-J-bhBQlxOXs1zGggvPYEA1xgQwCwWUk_j-PZq4Vn8CQ-x0ne7gfm5bOlIoXYMwCU2vzBuB8_iOYNjCOxCxT-J1WV9IJfW3IxsaNK3rz8pmSYFgv)
6. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGYSnjZ00ltRiIP29BpV-xvIRZEqNa7m1tRKt5Ep65CjH5tK3yegPgGTF6AEcD8DO-XLzBgL-Mkdyezhw3polgPHG8qSIK79MwF9cACr1cyEFscXm4VfkpKQRoCxtB--n-LZ7IPtbWAYT6L07lVQ8O2JwOC_gW1yVXhtYsXhs4IYLMklmGXmA7O)
7. [viksnewsletter.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHLlHeBFid66pH3RfVQwiM2ZzpThjTPZTzh6E869Rp54BQzx6hQ7QE8ujb1Qq5GIL7XRpe6WTalS4WXw3C_7QqoVi_snBJzpkWWDZ0HqEx5_q3n_p-UcYZx5dLjLbUJa3bIhMYPxS7ueES2qvz-7H-e_XjL0jb_jyB5mZw=)
8. [wikipedia.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEQt3CmpIztZmDN1ogY7WAClI-dFZ784Ya7n6Wg3zNNz0YvWDjdP5HlgSf8szO6doQReFmEy1MEGrXisDkztr4GzfjEijS4-9tk2QDNaFqbgoPpw4s7hgq5zA==)
9. [tiptinker.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEAAqWxWPdX3tqX9U9Tw9rnGjitNphePjbAYqlmQE4a1Vpe5O48uxFNxSre19wBS1A6jSuRuhAjIuT9s-iEfhxDSbe2rsYBjk_KTpnn4y3Td9pz3OIm3h6RC3qhJtA59rXYWmBxvWDcW0inweJjL2h-l9bi_CE8Xlr_66GvprdmQFKHGGkz0dHsB85Aw0w=)
10. [huggingface.co](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG_-Sdvltl5ycCMGX1d6YYAyhLQVezZ50alRe-cBwPVqGxpdbesA5GbXMxo0aOFuoia7a-4BJy7PIiCqxGAXAGfq3Z-RpZiTGhXuRDagkli3RMZaPmNztmjxPXIuwHHzUS2DZN2pFKRVoS6)
11. [csuohio.edu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHVRWRHP6Rj_v-h7NLKXBYafa9r--myQLn0vIaY2JC-Y6CLQGX8-SbP7cV-KRYENZWWAdNj0wP6u5Z0ldCe9d8x9aUPauZAnW_1TZIRkX-8DwlOtwxSJYPyRX0Wajgq-FehYx2vV9FwU-j7MdbgG1rG2A==)
12. [columbia.edu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGrpG7Nitld9FAhO59Rgjnu3Y6cmJiCfJzcwnh_sZqUX9V67KEQ3mlH4xNpEmnp6s_CjkIsFRpCApVsM2exXlFG3D9MNgp4zNxyFedOBGeYwVT8vQ4rrTjOIExSzyBP3MJQjYLnfO2ttdsrVjbsl93S57Whyccpz7M4oTZK2EmYO8_-sA==)
13. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE_0r3UMdCYJ7Yt0Glnn4ZsYhe6rlSl9nKx5aEGHTidpG5uCUMTYT1499j-tvYCNS5F349NPOzTxFpC9jEgNcc2pLWNu9eVE9ds9FB4mzsgBH70xST1bFf2_m0MI1bThdC8bNgEhXBNrp4mP0onwnuKe4a7p-TS6VQoOwnvYUwms5aaEjMRMHb3IsBRL2OZn-e-GBRZhHLA-5vVXYgp2B3aBC3mZgrZQOY=)
14. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE_Tep22OEjJfa1fWbut8fzxFlR_t6_NGquaXYXWitM_uQ7-mn5j8U_fsJTdd1Q33l9eV0uQBtFglzlCyUHaiQuImnwex0NRUJhzVGP-85p4eyhiexnno8uRwwrf20rgCTcsbWZ-1WPSYKg4iSxeTVxcQCPtQJriFMhfs8beScOjgN37Sa1xzr26OLr359nSGA4kNG-Fb0=)
15. [stackexchange.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEe6zYp5Y0obnAJ_blh8To44aUFp-QuBEaRIDV1EyVycFUQFM1qRpgtyoo5UAi0JFl-X5PaJroXNv3Z33z8osYbDK7BVoKWXnuTcUh2HH55xogg3L39WymFGcU_-B20zyjaIN0lYEQp0mb2LayHV93PGXzVuuhzcilkvC4qNJxUYCV1sEb-S-o_5gyveter39wDHg4O7URSFujH5I66qAPjVkU=)
16. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEqI9ZiiqMaUYC1QnvryAHGmO6YG5g7Cwl5sNztqxgdB4YN1Y4scLjIux1Jh3-ePuf5zmhmD9qmWD4ZO50rMNDhUzuYv9qlVCGWoXk52X0DOfkr2QiJfR0Xa9PLR6-EouVUgfjxtV07SgjyILSH7luaAEF28dE=)
17. [towardsdatascience.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEgDda1F5UQ2ZTeVmZHLaRVK7C-H1STH80kiMkx3Jl5PvDlJprgYMc76nVfvGPH6FKT2tj2ZwHukDf-pDsAlIIQfaA4G8_zDCV1Luitx1ptXiRooZ-bH5weDh9cfDmF_9xfgOaLhRbL0WRG_ztS2iLu2LIEBPcrWFV1KJylsqB2_-SFsks2eGoq)
18. [wikipedia.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHzlSZNDQx7NubJnZ6EampwHxMuDl0_XWcmwNP1YdOYxe_SMuZNl2OkLb2-5kp2zJdsqaRp4iF1o9qsbRqJWdC1cYgNPwqspLjYq-Wg8Evv91lHtjOTsPNd030gb2wEJrqON0FNcHT8b2s=)
19. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFb-qhfJYfvP6h1FjbjWVuTS4ZA79beFCmIYb8qGdr_iTpG3xZ7XikTD4UhQy37Mq9Rh0ynOqcHF_1Expt8aK1jFLPp5fMX8U1if_Dwsm0M574gL--TWiqbqOmxwkpe5SZmYrkOg2e2zJeTqEIQBR3pS45E6e3WvXP1qW4iK0LhJlJVPDRH9Ko4-RE1)
20. [oreateai.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFy1nl1gbtozgoJViNmJ_vOt5gr7taZbVj-GfrHjuXp_3deWG0Xjipdet6lo557R_qJkQHChVSeHKLYXhjei0OKy1qTSH9hAX43itqXZWEbEsCwtSndqkRl99ryUAjhP2naMgNweml-VnUIjgRjMIEFD0fKmSofAkvmYksi-X7-vz5uWnYiBlzWpNsku3_eVhbJ-BBPWwnoy-bYT3dc7jU35ylwKYowIHMN5AotetgamCe5pkSbcyaKsMWePyFp7NSZZrzyAZTRXKFJaou7vBidE4o8PvcW_BdKIg8US-lnYIZTdaLqcDnwqg==)
21. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE6qxrXRAHuLAIG4Bmt9Pn4-uZIw38tUPatJaJmsZl4LyoPdUHXhh7a00UiWQ8Vd09Zdyk0EQ2wdWn90JpSqmjU3NSvFhtxvc7pIUio4cePt0FjEBiCdeokog==)
22. [scrile.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHakYfvPqkSTgzAb3t_TdGs7fn6f3pvZ0ts1CR122JuU3RX-umCXRnaEt4qFjgel1LGJNXUIGPf78WM1lB9zDj6iWTKaIg8lzV9uhxur4pHqg9jXt6UPKgjoWNGOKA=)
23. [fiveable.me](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHvc4P4UH8PNXY5ovmuS2ZbQHt6_0TxqjlZSwJz1hs8UxNhZP2rtVV8TMhxFuag12RWcKxqYaZWF_I-ilqCTizlPFNd3gM6yRkp_IWasz3NC3jDUJIWzoLt2uVfIcQN3A5CcvHlsPOQrSwcu6oXlzrRzMnGB_YneHFrVkXXweRbAUIIx58e8vZoJjg1qwDemZiRTFZ2Jb6tzmLxzITsI05L99nXoEL0RT-mmN0VuskV)
24. [invgate.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQELiXBp7uMl6v8MDw-KZvIZoSJ0kyjdNoSesSb7I1n5mU86luN_7Z-VqABUhTqj13boKbAyV5hp-krViLlBKrNFbAPVO_RO9TFqnZOe30hLYq8QKdZiNdQqLyLKGA==)
25. [dev.to](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHLQXRnZj8d7r-pUjGbQjOhq0vNYr5PVX9EUp8QfYUxRrtg4auFLAHWkC00iTgqgV7qgDRG3Y3YhSnnTUVb0qIaqf-IsNyMJCl1J6G94f6iXkOAU1Zi_g7ESkPTGc1YmNhHpObAgpapCLnG7u7ItAotKMMgbkQIoLYeNuDf-bg7JH9T7XWsMp6C2g==)
26. [heidloff.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFpc69vEhluVYsB7SfLcgsdxiOWYLhPPrFvCqOUZafHbPxkt3S6pcUj7P4n-ZoqSGbAxwhIikGNRlbfMOjiYdlqWVW6u1vO5oOp1PYzC6-zKS8vk8dbd1adFdnuyT5wS4PxZRazpNXPGbx8sKjUEEv40xODpYzWUhO8T-p89lpi)
27. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEsinJcif0kiIdzMgoFBcw3Hwq8pBdAOJm7z8QpcpyrBBHwOIMWKY_2Z8b_1Jpv9_w5dtNSKm-GMe4s8OZRURxmb6CLzok3HzTBu6QzdOBc1PltIb8aQjT_Z0wuwJvn-wUaAYMbFO3OFl2ATyAX4EBpQsIMH9XDmckBCSC63XfirrF3tMKNKxWRDC1JbpxAa6mXSdwXwBQY7uHOpzvbfeKhWI9RC0i00__Gbl5Plfb1wTAZT7zDLeYp--VR)
28. [towardsdatascience.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH5YpekCkU_ntUF_ePds5sUwbqZNLr7Q_kPJJ26hfVw8BfDtKCTt4B0YhS2Y2fTYqAZDLrutiKNiTsCCcXNtTU4RN4Key7bbglsSML3z34OsKvAyYCvy8q8k6Joa03T8tJKHqMe2ctqUBW094ePZhf7YiUaunAHrzy_Md3sEDxU6RMIxhQQGBvyw1zc46se2Ue3dhW4M_oTJ00HyJGePRnlCJpOnaTF0PM=)
29. [modelscope.cn](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE71d9voc512D_iHk2caVJ6bOhj8JRpZINaMlSjN0sb_CN-wMiWLMJIwkBlt9hfwG_PHNzAj8b4ksx8XYWsVp8G1lBcy9GWcOxJued8zbwnlXmzdxJsNSD345ICjPsMj0W8FqtqgmLNlhZB5cDIDz1LFphmIUAMpeLhAtxHNTXfb6nBdA==)
30. [orange.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE3giIoDidxPKJjqN4jWlQH08bkqzTGIE0JeCNLsKyKcSp3d-QoeqCostpCNo1nZthhA-RcOVDG4_VVSmltE5iZ5kiTkvi8lY3Rc0eMJxnCVCPuQEt9xTAcns7ilDuXxgNw4bmFRy7iUwJdUI1nak39fSO-5AFuef2UtQ8Y3tr4rUt60i1lcGpUJw==)
