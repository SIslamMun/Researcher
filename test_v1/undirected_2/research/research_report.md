# Comprehensive Analysis of Transformer Architecture

## Key Points
*   **Foundational Shift:** The Transformer architecture, introduced in 2017, replaced Recurrent Neural Networks (RNNs) by relying entirely on **self-attention mechanisms**, enabling parallel processing of sequences and capturing long-range dependencies effectively.
*   **Core Mechanism:** The **Scaled Dot-Product Attention** allows the model to weigh the importance of different tokens in a sequence relative to each other. It uses Query (Q), Key (K), and Value (V) vectors, scaled by the square root of the dimension depth to ensure training stability.
*   **Positional Awareness:** Since Transformers process tokens in parallel, they require **Positional Encodings** to understand sequence order. Modern iterations often use **Rotary Positional Embeddings (RoPE)**, which encode relative positions via rotation matrices.
*   **Architectural Variants:** The architecture has evolved into three main families: **Encoder-only** (e.g., BERT) for understanding tasks, **Decoder-only** (e.g., GPT) for generation tasks, and **Encoder-Decoder** (e.g., T5) for sequence-to-sequence tasks like translation.
*   **Optimization:** Innovations like **FlashAttention** have significantly optimized the computational efficiency of Transformers by managing GPU memory hierarchy (IO-awareness), reducing the quadratic complexity bottleneck in practice.

---

## 1. Introduction and Historical Context

The introduction of the Transformer architecture in the seminal paper *"Attention Is All You Need"* by Vaswani et al. (2017) marked a paradigm shift in deep learning, particularly for Natural Language Processing (NLP) [cite: 1, 2]. Prior to this innovation, sequence transduction tasks—such as machine translation and text summarization—were dominated by Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Gated Recurrent Units (GRUs) [cite: 3, 4].

### 1.1 Limitations of Predecessor Architectures
RNNs and LSTMs processed data sequentially, computing hidden states step-by-step. This sequential nature precluded parallelization within training examples, creating a significant computational bottleneck for long sequence lengths [cite: 2, 4]. Furthermore, these models struggled with **long-range dependencies**; as the distance between two tokens in a sequence increased, the ability of the network to retain and relate information diminished, despite mechanisms like gating intended to mitigate this issue [cite: 5, 6].

### 1.2 The Transformer Breakthrough
The Transformer dispensed with recurrence and convolutions entirely, relying solely on **attention mechanisms** to draw global dependencies between input and output [cite: 2, 7]. This architectural choice allowed for:
1.  **Parallelization:** The entire sequence could be processed simultaneously, significantly reducing training times on modern hardware like GPUs [cite: 5, 8].
2.  **Global Context:** The path length between any two positions in the network was reduced to a constant, enabling the model to effectively capture relationships between distant words (e.g., resolving pronoun references across long paragraphs) [cite: 9, 10].

---

## 2. High-Level Architecture

The original Transformer model is an **Encoder-Decoder** structure. While many modern derivatives (like GPT or BERT) utilize only one half of this architecture, understanding the full model is essential for grasping the underlying principles.

### 2.1 The Encoder
The Encoder is responsible for processing the input sequence and compressing it into a context-aware representation. It consists of a stack of $N$ identical layers (originally $N=6$) [cite: 2]. Each layer has two main sub-layers:
1.  **Multi-Head Self-Attention Mechanism:** Allows the model to associate each word with every other word in the input sentence [cite: 8, 11].
2.  **Position-wise Feed-Forward Network (FFN):** A fully connected network applied to each position separately and identically [cite: 9, 12].

The output of the encoder is a set of continuous representations (vectors) that hold the semantic and syntactic information of the input [cite: 11, 13].

### 2.2 The Decoder
The Decoder generates the output sequence one element at a time (autoregressively). Like the encoder, it is composed of a stack of $N$ identical layers. However, each decoder layer includes a third sub-layer:
1.  **Masked Multi-Head Self-Attention:** Prevents positions from attending to subsequent positions. This ensures that the prediction for position $i$ can depend only on the known outputs at positions less than $i$ [cite: 8, 14].
2.  **Encoder-Decoder Attention (Cross-Attention):** Performs attention over the output of the encoder stack, allowing the decoder to focus on relevant parts of the input sequence while generating the output [cite: 8, 15].
3.  **Position-wise Feed-Forward Network:** Similar to the encoder's FFN [cite: 8].

---

## 3. The Attention Mechanism

The core innovation of the Transformer is the "Self-Attention" mechanism, which allows the model to weigh the importance of different tokens in a sequence when processing a specific token [cite: 9, 10].

### 3.1 Query, Key, and Value Vectors
For every token in the input, the model generates three vectors through learned linear transformations (matrix multiplications) [cite: 16, 17]:
*   **Query ($Q$):** Represents the current token looking for relevant information.
*   **Key ($K$):** Represents the "tag" or identifier of other tokens against which the query is matched.
*   **Value ($V$):** Contains the actual content or information of the token.

The attention score is calculated by measuring the compatibility (similarity) between a Query and a Key. This score then determines how much of the corresponding Value is retrieved [cite: 16, 18].

### 3.2 Scaled Dot-Product Attention
The mathematical formulation of the attention mechanism used in Transformers is called **Scaled Dot-Product Attention**. It is defined as:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Where:
*   $QK^T$ is the dot product of the Query and Key matrices, resulting in a similarity score matrix.
*   $d_k$ is the dimension of the key vectors.
*   $\sqrt{d_k}$ is the scaling factor.
*   Softmax normalizes the scores into probabilities (summing to 1) [cite: 17, 19].

#### Why Scale by $\sqrt{d_k}$?
The scaling factor is crucial for training stability. As the dimension $d_k$ increases, the dot products can grow large in magnitude. Large values push the softmax function into regions where gradients are extremely small (vanishing gradients), effectively halting learning [cite: 19, 20]. Dividing by $\sqrt{d_k}$ ensures that the dot products have a variance of approximately 1 (assuming inputs have unit variance), keeping the softmax function in a region with significant gradients [cite: 21, 22, 23].

### 3.3 Multi-Head Attention
Instead of performing a single attention function, the Transformer employs **Multi-Head Attention**. This involves running the self-attention mechanism multiple times in parallel (e.g., 8 heads in the original paper) [cite: 8, 24].
*   **Purpose:** Each head can learn to focus on different aspects of the relationships between words. For example, one head might track syntactic dependencies (subject-verb), while another tracks semantic relationships (pronoun resolution) [cite: 4, 5].
*   **Mechanism:** The outputs of the independent attention heads are concatenated and linearly transformed to produce the final output [cite: 1, 12].

---

## 4. Positional Encodings

Since the Transformer contains no recurrence and no convolution, it is invariant to the order of the sequence. To inject information about the relative or absolute position of the tokens, **Positional Encodings** are added to the input embeddings [cite: 9, 11].

### 4.1 Sinusoidal Positional Encodings
The original paper used fixed sinusoidal functions of different frequencies:
\[
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
\]
\[
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
\]
This method allows the model to potentially learn to attend by relative positions, as for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$ [cite: 24, 25].

### 4.2 Rotary Positional Embeddings (RoPE)
Modern Transformers, particularly Large Language Models (LLMs) like LLaMA and PaLM, often use **Rotary Positional Embeddings (RoPE)** [cite: 26, 27].
*   **Concept:** Instead of adding a position vector to the embedding, RoPE encodes position by **rotating** the Query and Key vectors in a high-dimensional space [cite: 28, 29].
*   **Mechanism:** It applies a rotation matrix based on the absolute position $m$ to the vectors. The dot product $q^T k$ then depends only on the relative distance $m-n$ between tokens, not their absolute positions [cite: 30, 31].
*   **Advantages:** RoPE offers better generalization to sequence lengths longer than those seen during training (extrapolation) and naturally decays the attention score as relative distance increases [cite: 27, 32].

---

## 5. Structural Components and Stability

To train deep Transformer networks effectively, several structural components are critical.

### 5.1 Layer Normalization (LayerNorm)
Layer Normalization normalizes the inputs across the feature dimension (ensuring zero mean and unit variance) for each sample independently. This stabilizes the hidden state dynamics and improves convergence [cite: 33, 34].

### 5.2 Pre-LN vs. Post-LN
The placement of LayerNorm has a significant impact on training stability.
*   **Post-LN (Original):** In the original paper, LayerNorm was placed *after* the residual connection: $x + \text{Sublayer}(x) \rightarrow \text{LayerNorm}$. This configuration often requires a "warm-up" stage for the learning rate to prevent divergence during early training [cite: 33, 35].
*   **Pre-LN (Modern Standard):** Most modern LLMs (e.g., GPT-3, LLaMA) place LayerNorm *before* the sublayer, inside the residual block: $x + \text{Sublayer}(\text{LayerNorm}(x))$. This creates a "gradient highway" that significantly improves stability and allows for training deeper networks without complex warm-up schedules [cite: 36, 37, 38]. However, some research suggests Post-LN may achieve slightly better performance if successfully trained [cite: 33, 39].

### 5.3 Feed-Forward Networks (FFN)
Each layer contains a fully connected feed-forward network, applied to each position separately and identically. It consists of two linear transformations with a non-linear activation function (originally ReLU, now often GeLU or SwiGLU) in between [cite: 8, 15].
\[
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
\]
This layer processes the information gathered by the attention mechanism and transforms it into a more complex representation [cite: 40].

---

## 6. Architectural Families and Variants

The flexibility of the Transformer architecture has led to three primary distinct families, each suited for different tasks [cite: 41, 42].

### 6.1 Encoder-Only (e.g., BERT)
*   **Architecture:** Uses only the Encoder stack.
*   **Mechanism:** Utilizes **Bidirectional Self-Attention**, allowing tokens to attend to both previous and future tokens simultaneously.
*   **Training Objective:** Masked Language Modeling (MLM)—predicting masked words based on context.
*   **Use Cases:** Understanding tasks such as Text Classification, Sentiment Analysis, Named Entity Recognition (NER), and Question Answering [cite: 43, 44].

### 6.2 Decoder-Only (e.g., GPT)
*   **Architecture:** Uses only the Decoder stack (without the cross-attention layer).
*   **Mechanism:** Utilizes **Masked (Causal) Self-Attention**, preventing the model from seeing future tokens.
*   **Training Objective:** Causal Language Modeling (CLM)—predicting the next token in the sequence.
*   **Use Cases:** Generative tasks such as Text Generation, Chatbots, and Code Generation [cite: 41, 43].

### 6.3 Encoder-Decoder (e.g., T5, BART)
*   **Architecture:** Retains the full original architecture.
*   **Mechanism:** The encoder processes the input bidirectionally, and the decoder generates output autoregressively while attending to the encoder's representation.
*   **Training Objective:** Span corruption (T5) or Denoising (BART).
*   **Use Cases:** Sequence-to-sequence tasks like Machine Translation and Text Summarization [cite: 14, 44, 45].

---

## 7. Optimization: FlashAttention

A major bottleneck in Transformers is the quadratic memory and time complexity of the attention mechanism ($O(N^2)$ with respect to sequence length). **FlashAttention**, introduced by Dao et al. (2022), is a hardware-aware algorithm that drastically improves efficiency [cite: 46, 47].

### 7.1 The IO-Awareness Principle
Standard attention implementations require repeatedly reading and writing large $N \times N$ attention matrices to the GPU's High Bandwidth Memory (HBM), which is slow. FlashAttention optimizes this by accounting for the memory hierarchy (HBM vs. fast on-chip SRAM) [cite: 48, 49].

### 7.2 Tiling and Recomputation
*   **Tiling:** FlashAttention splits the input matrices ($Q, K, V$) into small blocks that fit entirely into the fast SRAM. It computes attention for these blocks locally, avoiding the need to materialize the full $N \times N$ matrix in HBM [cite: 49, 50].
*   **Recomputation:** Instead of storing intermediate values for backpropagation (which consumes vast memory), it recomputes them on-the-fly during the backward pass. Surprisingly, this is faster because it reduces the bottleneck of HBM access [cite: 47, 51].

### 7.3 Impact
FlashAttention enables training on much longer sequences (e.g., increasing context windows from 4K to 64K+) and provides significant wall-clock speedups (2-4x) without any approximation—it yields the exact same mathematical result as standard attention [cite: 50, 52].

---

## 8. Vision Transformers (ViT)

The success of Transformers in NLP led to their adaptation for Computer Vision, challenging the dominance of Convolutional Neural Networks (CNNs).

### 8.1 Patch Embeddings
Instead of processing pixels individually, the **Vision Transformer (ViT)** divides an image into fixed-size patches (e.g., $16 \times 16$ pixels). These patches are flattened and linearly projected into embeddings, effectively treating them as "words" in a sequence [cite: 53, 54, 55].

### 8.2 Global Receptive Field
Unlike CNNs, which build context hierarchically (local features $\rightarrow$ global features), ViT allows every patch to attend to every other patch from the very first layer. This provides a global receptive field immediately, allowing the model to capture long-range dependencies in images (e.g., relating objects at opposite corners of an image) [cite: 53, 56].

### 8.3 Inductive Bias
ViTs lack the strong inductive biases of CNNs (translation invariance and locality). Consequently, they typically require larger datasets (like JFT-300M or ImageNet-21k) to perform well, as they must learn these spatial relationships from scratch. However, given sufficient data, they often outperform CNNs [cite: 53, 54].

---

## 9. Conclusion

The Transformer architecture has fundamentally reshaped the landscape of Artificial Intelligence. By decoupling sequence processing from temporal order via **Self-Attention**, it solved the parallelization and long-term dependency issues of RNNs. Its modular design has spawned a diverse ecosystem of models—from **BERT's** deep understanding capabilities to **GPT's** generative power and **ViT's** visual prowess. Continued innovations like **RoPE** and **FlashAttention** ensure the architecture remains scalable and efficient, underpinning the current era of Large Language Models and Generative AI.

---

## References

### Publications
[cite: 11] "Transformers Model Architecture Components" (MyScale). MyScale Blog, 2024. https://myscale.com/blog/transformers-model-architecture-components-deep-learning/
[cite: 9] "What Is a Transformer Model?" (AWS). Amazon Web Services. https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/
[cite: 8] "Architecture and Working of Transformers in Deep Learning" (GeeksforGeeks). GeeksforGeeks, 2025. https://www.geeksforgeeks.org/deep-learning/architecture-and-working-of-transformers-in-deep-learning/
[cite: 57] "Encoder vs Decoder in Transformers: Unpacking the Differences" (Hassaan Idrees). Medium, 2024. https://medium.com/@hassaanidrees7/encoder-vs-decoder-in-transformers-unpacking-the-differences-9e6ddb0ff3c5
[cite: 58] "Difference between Transformer Encoder and Decoder" (Hugging Face Discuss). Hugging Face, 2021. https://discuss.huggingface.co/t/difference-between-transformer-encoder-and-decoder/4127
[cite: 13] "Understanding Encoder and Decoder" (Sebastian Raschka). Sebastian Raschka Blog, 2023. https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder
[cite: 14] "Encoder vs Decoder Transformer: A Clear Comparison" (DhiWise). DhiWise, 2025. https://www.dhiwise.com/post/encoder-vs-decoder-transformer-a-clear-comparison
[cite: 15] "Transformer (deep learning)" (Wikipedia). Wikipedia. https://en.wikipedia.org/wiki/Transformer_(deep_learning)
[cite: 24] "Transformer Explainer" (Polo Club). Georgia Tech. https://poloclub.github.io/transformer-explainer/
[cite: 59] "How Transformers Work" (DataCamp). DataCamp, 2024. https://www.datacamp.com/tutorial/how-transformers-work
[cite: 3] "Transformer Neural Network" (Built In). Built In. https://builtin.com/artificial-intelligence/transformer-neural-network
[cite: 40] "Transformer Architecture Explained" (Amanatulla). Medium, 2023. https://medium.com/@amanatulla1606/transformer-architecture-explained-2c49e2257b4c
[cite: 10] "Self-Attention" (IBM). IBM. https://www.ibm.com/think/topics/self-attention
[cite: 16] "Self-Attention" (H2O.ai). H2O.ai Wiki. https://h2o.ai/wiki/self-attention/
[cite: 5] "The Detailed Explanation of Self-Attention in Simple Words" (Maninder Singh). Medium, 2025. https://medium.com/@manindersingh120996/the-detailed-explanation-of-self-attention-in-simple-words-dec917f83ef3
[cite: 6] "Self Attention in NLP" (GeeksforGeeks). GeeksforGeeks, 2025. https://www.geeksforgeeks.org/nlp/self-attention-in-nlp/
[cite: 1] "Attention Is All You Need" (Vaswani et al.). NeurIPS, 2017. https://en.wikipedia.org/wiki/Attention_Is_All_You_Need
[cite: 4] "Attention Is All You Need Paper Summary" (Kingy AI). Kingy AI, 2024. https://kingy.ai/blog/attention-is-all-you-need-paper-summary/
[cite: 60] "Attention Is All You Need Overview" (AlphaXiv). AlphaXiv. https://www.alphaxiv.org/overview/1706.03762v4
[cite: 12] "Attention Is All You Need Summary" (David Min). Medium, 2023. https://medium.com/@dminhk/attention-is-all-you-need-summary-6f0437e63a91
[cite: 61] "Attention Is All You Need" (Programming Ocean). Programming Ocean. https://www.programming-ocean.com/articles/attention-is-all-you-need.php
[cite: 48] "Understanding Flash Attention" (Alex Dremov). Towards Data Science, 2025. https://towardsdatascience.com/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton-5609f0b143ea/
[cite: 52] "Flash Attention Article" (Modal). Modal Blog, 2024. https://modal.com/blog/flash-attention-article
[cite: 62] "Flash Attention Dictionary" (Hopsworks). Hopsworks. https://www.hopsworks.ai/dictionary/flash-attention
[cite: 63] "What is Flash Attention Explained" (Reddit). Reddit r/MachineLearning, 2024. https://www.reddit.com/r/MachineLearning/comments/1e36jye/r_what_is_flash_attention_explained/
[cite: 49] "Flash Attention: A Deep Dive" (Towards Data Science). Towards Data Science, 2024. https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b/
[cite: 53] "Vision Transformer (ViT) Architecture" (GeeksforGeeks). GeeksforGeeks, 2025. https://www.geeksforgeeks.org/deep-learning/vision-transformer-vit-architecture/
[cite: 54] "Vision Transformer" (Wikipedia). Wikipedia. https://en.wikipedia.org/wiki/Vision_transformer
[cite: 56] "Vision Transformer ViT" (Viso.ai). Viso.ai. https://viso.ai/deep-learning/vision-transformer-vit/
[cite: 64] "Vision Transformers" (Roboflow). Roboflow Blog, 2025. https://blog.roboflow.com/vision-transformers/
[cite: 55] "Vision Transformer Guide" (V7 Labs). V7 Labs, 2022. https://www.v7labs.com/blog/vision-transformer-guide
[cite: 19] "Scaled Dot-Product Attention" (APXML). APXML Courses. https://apxml.com/courses/foundations-transformers-architecture/chapter-2-attention-mechanism-core-concepts/scaled-dot-product-attention
[cite: 65] "Scaled Dot-Product Self-Attention Mechanism" (Manny Maminta). Medium, 2024. https://pub.aimind.so/scaled-dot-product-self-attention-mechanism-in-transformers-870855d65475
[cite: 17] "Scaled Dot-Product Attention Explained" (iTOBOS). iTOBOS. https://itobos.eu/images/iTOBOS/Articles_Blog/NTUA/scaled_dot_attention.pdf
[cite: 18] "Demystifying Scaled Dot-Product Attention" (PyML). PyML Substack, 2023. https://pyml.substack.com/p/demystifying-scaled-dot-product-attention
[cite: 66] "Understanding Scaled Dot-Product Attention" (Bandaru Devisri). Medium, 2024. https://medium.com/@bandarudevisri.ds/understanding-scaled-dot-product-attention-in-transformers-8cd794ed8164
[cite: 21] "Why is Attention Divided by sqrt(dk)" (Srivatsa N). Medium, 2025. https://medium.com/@srivatsa.n63/why-is-attention-divided-by-d%E2%82%96-the-secret-behind-scaled-attention-in-transformers-44f36465266f
[cite: 20] "The Scaling Factor sqrt(dk) in Attention Mechanisms" (Chris Yan). Medium, 2024. https://chrisyandata.medium.com/the-scaling-factor-sqrt-dk-in-attention-mechanisms-origins-purpose-and-impact-66bce649a104
[cite: 22] "Why does multiplication of Q and K have variance dk" (Stack Exchange). AI Stack Exchange, 2020. https://ai.stackexchange.com/questions/21237/why-does-this-multiplication-of-q-and-k-have-a-variance-of-d-k-in-scaled
[cite: 23] "Why do we divide attention scores by sqrt(dk)" (Priyesh Dave). Medium, 2023. https://medium.com/@priyeshdave90/why-do-we-divide-the-attention-scores-in-self-attention-by-the-sqrt-dk-c7f505a69506
[cite: 67] "What is Scaling in Transformers Self Attention" (Useless AI). Useless AI, 2024. https://uselessai.in/what-is-scaling-in-transformers-self-attention-you-ll-not-regret-reading-this-d37121f6644e
[cite: 26] "Rotary Position Embedding RoPE" (APXML). APXML Courses. https://apxml.com/courses/how-to-build-a-large-language-model/chapter-13-positional-encoding-variations/rotary-position-embedding-rope
[cite: 28] "Rotary Positional Embeddings" (Cyril Zakka). LLM Playbook. https://cyrilzakka.github.io/llm-playbook/nested/rot-pos-embed.html
[cite: 27] "Rotary Positional Embeddings RoPE" (Emergent Mind). Emergent Mind, 2025. https://www.emergentmind.com/topics/rotary-positional-embeddings-rope
[cite: 30] "Rotary Embeddings" (EleutherAI). EleutherAI Blog, 2021. https://blog.eleuther.ai/rotary-embeddings/
[cite: 29] "RoPE Explained Video" (YouTube). YouTube, 2025. https://www.youtube.com/watch?v=V8r__fXx7tU
[cite: 41] "Pre-trained Transformer Models: BERT, GPT, T5" (Fiveable). Fiveable. https://fiveable.me/deep-learning-systems/unit-10/pre-trained-transformer-models-bert-gpt-t5/study-guide/o8JLDj9oFwOSdcRt
[cite: 43] "Transformer Models Use Case Guide" (DevsTree). DevsTree, 2025. https://www.devstree.com/transformer-models-use-case-guide-bert-gpt-t5/
[cite: 42] "Comparing BERT, GPT, and T5" (Data Science Collective). Medium, 2025. https://medium.com/data-science-collective/comparing-bert-gpt-and-t5-when-should-you-use-each-one-f9bfcfd5454c
[cite: 44] "Comparing Large Language Models" (MDM Generative AI). WordPress, 2025. https://mdmgenerativeaimodelsblog.wordpress.com/2025/03/14/comparing-large-language-models-gpt-vs-bert-vs-t5/
[cite: 45] "Comparing Large Language Models" (Automotive Visions). WordPress, 2025. https://automotivevisions.wordpress.com/2025/03/21/comparing-large-language-models-gpt-vs-bert-vs-t5/
[cite: 33] "Normalization Layer Placement" (APXML). APXML Courses. https://apxml.com/courses/how-to-build-a-large-language-model/chapter-11-scaling-transformers-architectural-choices/normalization-layer-placement
[cite: 34] "Pre-Normalization vs Post-Normalization" (VectorWorks Academy). Medium, 2025. https://medium.com/@VectorWorksAcademy/pre-normalization-vs-post-normalization-in-transformers-e84872e0a3cd
[cite: 36] "Pre-Norm vs Post-Norm" (Zao Yang). Newline, 2025. https://www.newline.co/@zaoyang/pre-norm-vs-post-norm-which-to-use--3ea6df8c
[cite: 35] "Pre-Norm vs Post-Norm" (Aussie AI). Aussie AI. https://www.aussieai.com/book/ch24-pre-norm-vs-post-norm
[cite: 37] "Pre-LayerNorm Transformer" (Emergent Mind). Emergent Mind, 2025. https://www.emergentmind.com/topics/pre-layernorm-transformer
[cite: 31] "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al.). arXiv:2104.09864, 2021. https://arxiv.org/abs/2104.09864
[cite: 68] "Context-aware Rotary Position Embedding" (arXiv). arXiv:2507.23083, 2025. https://arxiv.org/abs/2507.23083
[cite: 69] "Rotary Position Embedding for Vision Transformer" (arXiv). arXiv:2403.13298, 2024. https://arxiv.org/abs/2403.13298
[cite: 32] "RoFormer Paper PDF" (Su et al.). arXiv, 2023. https://arxiv.org/pdf/2104.09864
[cite: 70] "RoPE Explained Video" (YouTube). YouTube, 2023. https://www.youtube.com/watch?v=GQPOtyITy54
[cite: 54] "Vision Transformer ViT Architecture" (Wikipedia). Wikipedia. https://en.wikipedia.org/wiki/Vision_transformer
[cite: 56] "Vision Transformer ViT" (Viso.ai). Viso.ai. https://viso.ai/deep-learning/vision-transformer-vit/
[cite: 71] "Vision Transformers Working Architecture" (Codecademy). Codecademy. https://www.codecademy.com/article/vision-transformers-working-architecture-explained
[cite: 53] "Vision Transformer ViT Architecture" (GeeksforGeeks). GeeksforGeeks, 2025. https://www.geeksforgeeks.org/deep-learning/vision-transformer-vit-architecture/
[cite: 72] "Vision Transformers Tutorial" (DataCamp). DataCamp, 2025. https://www.datacamp.com/tutorial/vision-transformers
[cite: 38] "Pre-LN vs Post-LN Analysis" (APXML). APXML Courses. https://apxml.com/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/pre-ln-vs-post-ln
[cite: 34] "Pre-Normalization vs Post-Normalization" (VectorWorks Academy). Medium, 2025. https://medium.com/@VectorWorksAcademy/pre-normalization-vs-post-normalization-in-transformers-e84872e0a3cd
[cite: 73] "Pre-LN vs Post-LN Stability Paper" (AlphaXiv). AlphaXiv. https://www.alphaxiv.org/overview/2510.09904
[cite: 39] "Findings of ACL 2023" (ACL Anthology). ACL Anthology, 2023. https://aclanthology.org/2023.findings-acl.192/
[cite: 74] "Stability of Transformers under Layer Normalization" (ResearchGate). ResearchGate, 2025. https://www.researchgate.net/publication/396458347_Stability_of_Transformers_under_Layer_Normalization
[cite: 46] "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al.). Semantic Scholar, 2022. https://www.semanticscholar.org/paper/FlashAttention%3A-Fast-and-Memory-Efficient-Exact-Dao-Fu/87c5b281fa43e6f27191b20a8dd694eda1126336
[cite: 50] "FlashAttention Breakthrough" (Saurabh K). Medium, 2025. https://medium.com/@saurabhk1/flashattention-the-io-aware-breakthrough-powering-faster-transformers-e8728edcc7a9
[cite: 51] "FlashAttention-3" (Tri Dao). Tri Dao Blog, 2024. https://tridao.me/blog/2024/flash3/
[cite: 47] "FlashAttention arXiv" (Dao et al.). arXiv:2205.14135, 2022. https://arxiv.org/abs/2205.14135
[cite: 75] "FlashAttention ResearchGate" (ResearchGate). ResearchGate. https://www.researchgate.net/publication/360936499_FlashAttention_Fast_and_Memory-Efficient_Exact_Attention_with_IO-Awareness
[cite: 25] "Transformer Architecture Slides" (Ysu.edu). Ysu.edu. https://ysu1989.github.io/courses/au20/cse5539/Transformer.pdf
[cite: 2] "Attention Is All You Need PDF" (Vaswani et al.). NeurIPS, 2017. https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf
[cite: 1] "Attention Is All You Need" (Wikipedia). Wikipedia. https://en.wikipedia.org/wiki/Attention_Is_All_You_Need
[cite: 76] "Attention Is All You Need Semantic Scholar" (Vaswani et al.). Semantic Scholar, 2017. https://www.semanticscholar.org/paper/Attention-is-All-you-Need-Vaswani-Shazeer/204e3073870fae3d05bcbc2f6a8e263d9b72e776
[cite: 7] "Attention Is All You Need ResearchGate" (Vaswani et al.). ResearchGate. https://www.researchgate.net/publication/317558625_Attention_Is_All_You_Need

**Sources:**
1. [wikipedia.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFZyR6J9F1z2FHRYmrYyk6R37thc9UNWq2r8SzCqjSUAw9y75s3UZEfcG-U21In7Rzj1qLZetrKl3niWTitVG7xotJaUo2BW0y_ZDRYqwi3Yqm-cFXvLtlIOx1ZrI-I1lCkQrPVI5sJAVzYLDjO)
2. [neurips.cc](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGJkipKk2ZdMPa0cD4eJgDmVLskwgXYF68SCGyYfC8PgMQYlyELFw9gHKPy9JoXOKDF9BSUYl7j26PXRzP1JhkiTAMjvmEWMabyEe1dYJn5UEWw843-NnNZ18yaEH2i8wia-13ClvHJ9s6K5GZIwr9vOxKULZFwbAk=)
3. [builtin.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHpodc6bV_Rx7AdG3VZLTp9brsPLqIKeK5uPLoA0IDgp2Kj_bJ9ZqMDZTuwtHPsUu217LkG3ZTIMk3EK6NyPRkHcNdZz7WC7f7ZZmVkGrsJ8gjcgKR-AMAmIwsgeS6Ux30nIgRqZbtxi01Q1NqdEPjUWRaf9cG8EOtq9tcz)
4. [kingy.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHx3-9qXZtQGk0feVPpzgOj5ajP8p6rOScKClT5XZUOD5QzMwvMAcbo9j8v8op7WhdJGXn0Z5_onNd4Rxx3eWaWo7Mbwogm-ATnhAQP2Hc2KvCH1QQss74b3itQxC8d0LbFyD_tSxPtLFUfwWN9na1GDGDHnQ==)
5. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHY3C_0nxL9xwtdUxrw5YTWUN27sDMGFka8C3ODodSJQ20H0XSGWINfVT5FDgV0fPvmgMBqSu5ACGccHZ_hXRfprcPYSj0fpOPH_EG94KJCjHfiTOwqYWJfridBJTMMR93R4cmerjzpII3hLaGd-pSEgJ0GEUObJx4_Whovl_hMlAH2qcsMaZBNAoqy-6YoRv8HACasnYcePcpZihUQrIJ4dFL3GVA=)
6. [geeksforgeeks.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHGGZsZ6OfwcdiNtWnWdop-dqeaXSmSPNQVmXpM4bAieIJU0JdU-RgjtKTkyhP6EZrTVw7UqubEbXVNwTkymaaHIgGHCNA8Meq_hd3sGg37tG2SWWpusxYQTWe1Uv4OBkt7iznrAMyLz9RuoLCU3g==)
7. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGZvgW2xt3_218n_gH5IJGTQdao6aVhujz1kFkVSkTdJLYuIY-E8NWIqBBJgWf1kd1_4AG15VHu-EAfli_4l1tF18zMBBe-r9DHqQfgMYpCg5so9MGgeCFkEt92zTSdm7GoMyhn45LpsZpDFVFtUE-m7LZ3STlA9tmbj7FynEK53BsO)
8. [geeksforgeeks.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHooQZ-2MLgybNZH7VIkm73FMjiEvnOSubteQMghva7Rpc4uY-jK_zPJXdS1NKZDgLZJk5fuNNG1V6BKNfjazWRbVW-o5p6QoJXEXo6Fk58C2cpMDpgavvJukHGca3uFYlgoM1J6gKOLczWzLWsSG07xMOlQcULaysst-TnK1T63oDvWT3ZzfKdy99zv1qONwfy-C-7AkGTwWjJMig=)
9. [amazon.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEXRpfJC9txBAU8d2tlsB7hK2qkLYKhcu4nd0hnG07ckBAD8Z7yA8Y92PijQKUAv8XvaczDhbiPpCJ78_ot-_TbvwZmQcqdo80i8LS_StqkeGo4tfvLbjtzXEGqnLkW6JYOvxeUka8QYefg407aruk-L_VYlEPcmKLT1HDfyA==)
10. [ibm.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHWqMnxuBhqLFnxDNUmTUbAU3pwB3q9QPNtPrcQRJTDz83NuZJgZehF6OX0hIztmQZib6MjNRJpBstDkwcw9xq7Gvf7WNVaQ-tIaERrIWqPMonyCgkZbCdYCENECwrf38SV3nIURg==)
11. [myscale.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEYCsGCrgduJi68HWjfcSCKN73c8Ru5YnMFIAqALYb9t06fih-jMg1bpA21ibDzV71F-C9ZEbhR2xza4eL5KKd-1GuJAfqkjiC6nuOaKJ6lHpiONkRsO-HCocl07EwN_yrJCG2hwri5r3f0jfAFOYBABo1YoyeovYGrXPFZctEuBULC-o78XG-I)
12. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE3MLxehJkFiJrM4wQTAhqwN-wBmTVUeN2_7rnOxGGGsF__AeGbtLs7x8ilVU7Yly_lSqNCKWYqYWS2cFmCvACvxsF7amNE4ZBOwPt6DiqUndNcn6JSlab0AwWakm5c3JdFzGE_I54JJjXzCixQFmJB0M8FbltdwhRTeXdohRM8)
13. [sebastianraschka.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEaAZeRGy776au9esAEL0uuEvLDSLys4bJZ_gclNuAKuz9igHjuDKfVNaFjhusTBjzFLU_xHCC3Q97zKgvsvcmAdLrgMo_n7HucImL5J7tR7DrQb1MxPq1WnqkjussA4HGg1uxDXyA3NLgAzL5fEoaIN-MLsVKEIJZaVUYR-TPe)
14. [dhiwise.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVrHV9cdez_CADOzFmg2UIy6q9B-h2T6TKQgrXk7Bspz5FTnIrHsqiMwpaNcTt7STJZeXNz_IzXUZZIjim3uCvQBp6u9_nwFJll41Ij0qQK7dft-Zd02KNjgPEMKQtyeatxvns_YbXJ22EaLG4dQZxI3thoCBgPl4X8tUNpxEHKSYXOJ8=)
15. [wikipedia.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFpJEoMPqWqi_JxarsoSjTGOk4n6qkRge-McIcMIAQG5qkplI1rs_1ZEUItqTdp9Vkm4KqLAHSu0v5wtjGcED-rx6cN7B4INYg_eJy1Xv6ZXEh_6PfHjkVxqZjPlx0WMUudg5u0UEuJ1Bb30wEmOkQ=)
16. [h2o.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEjuSLwDjjXUq1cm5uvmak3E-cx83cjDtVxb4NUarerVtt26sOdHd1w4p-zEjI71sBirB43nqCdWf6DEcXyMaDlyjwjarhwh1_nIcD-CaQ4npVjJhkBgZOriw==)
17. [itobos.eu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFdMf2-d2tco6bs6oY3GPx_wJUqg5LMsUu9yUquu3ju295zkk0DVELKs8KJniCyHRkdsBV-dnNM0BrUzBoy1JbN_ObpLkIBoh-Ptrf7F-PI2MXg39P0hHRQAWjZddr0JHKZZC5rLSYC53bfmbXUGzlbxNcMu3O628XRHlIq-qmqM9g=)
18. [substack.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH4JdgLCAcvS7EpKbzfibTABu6rYSF7UCIHKT3vVGbPNHO5Fz3n25EvXg2_PBf8TPRXFG04XulR9fWnJzrSlEsHG1fR8hef1yzNSqfHOa9PEBrVSSZVWuY7gMWJCR1nFv5ONJOcBTFq4gTT9iUUlvQIQmhuDq9XdMW8LAM=)
19. [apxml.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFnBn_rOIDk7h7C0h8RhewC3zTEjQLnlHxOvlhzv8L47wWRBpP519VFdMO2gTVsDnakJ-C9Zx0vVaImrQKC8VtXBY4TA11Pkj2KQ6HvTV01_vk6yxMbVycfAtzRhuTeIOeZ7XJUrlnIvN0lYCDdEyPuuNo3zV70OW4qoRS9cGbS_jP8mM5WlM1dvrlEzzjmnUNm2s10L3f82gIFaCmHrHrmbZVDX4S7aNfSEuA8kC2gGexBYHNgMttt_hQde4ZHTQ==)
20. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHRpNGAWQMlsfPyIx4cBbpMxWAK0y5KExMxIG519YjqMcByFKFavCgDp-ynca1p7hqpczZN-McPZOJ2QtXZI3aPeRzX6ayuS6xxptFV7SYDDqnNg9S-KjH9jhLhr-hZHBhN8BHi2srZGjmF9NjG2MXSZ9u7O_nfWdieM46o112OhkGLF6-zifM4tk2twlDY9TrJqxgybtmRAcQq6aXvgDqCj9Rn-A3gJ_2iOR4EISV-rQ==)
21. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEtuz6SkZxNlNj7zl_iJ5Ko71zEdCfOi1q0kzyXJ2A4Vvo4EEUN77mwXOOibmdvVcvVvGU1G872COtLv-P8uDOCQvqivh8RTQNjZwONY0yHFnzFzYafpNlaaZD0v4ndNoU10K-sO0PrD5fdM1V7XDwCBUI5UWnmEnVF8ksxMPoDJathwSevD9VvGaK38XV8prhUuRLZ9zNpf9_iXjjHwadqKkXMs_wmi9uJNrgEWIsMgR4So9Jc2TiCMMAGEaWA)
22. [stackexchange.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFH-bKZi3vLJZCReGpoPnb28xK0sKguJ_A2sLmdnfc0yJOWHs1UMsZS4S33nLryp2LYHuMmtLKpZhlC3fw2SAiqMHJ3VPASuKju1JyAMF1rn3TmbAJE7d_WVK8APqK8qLlg_eBrS8-_fOk21d-C6cwOf69FOetszCqhcgQLL3Ilic04c-aqTUG_j59sa0U7Nu5kuVTDRe_NOzWS72UMRfLcls5RKqPyjTgvDBc=)
23. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEfitzAX2HdEIyFUm2mGD5EARoyNe5bNTgF6HMlzpfJdEQdfUGlzcJbvptGj3oUgFxkPvsvWP_xMAv92VPhPHno_M01QhsVzNw66KVELxheRpyoJomvTu9SakIJU3gXn1YT4ZqXeTxIjrxQnQSNvT9ZuXwoYpdT8rr0DjsLM4-bd5iu7UuWNr3OBz2DQLOTv9mEz3P37pIbQXjSmQTgbTMQCGrrlf_Jd0SzuEs=)
24. [github.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG9kF1o_HBRSN5aEttV_yZ9HDVr-wzEMPYq4-BrO56oiSGRKfKNeRhCDtWUkQk0lIaJARFcw6xq_iHu5NS6vfoKdo3I9_HXuCx4PYFM_GPsECkkME3nS4buHKoRC-6p2dWs1SDiPhZL)
25. [github.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF0PnNATM6L7wJjGnK-IqO5UHpV0k0P5ur-EBNwB26N6YKraAtGYYYbBdpdsunXUjGPAMdwjxF5K1akRJl_kAQxY641EtwcJzIFcdxHO8l4OeSYKBZavFuQPFM3YriXlOExd53xLnGUPktYjuwaX6exZMZDew==)
26. [apxml.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG6A8JQe7TlS16zLc2_7izWU1DIteOfIn3_qpUESdIQsUrBXsh9wyV9iH4_RRLULTG11GWBWsITL-nmwheilTMWcuJbAK6-ZmI6xhJzJY706vJH35IZw95Mzk1HbMrn0pEXLNo1Wvy94z60LT_ZiUS4Tl5_iIWBFPzzloTBEn-40ydEm3MIPoXIJiCRcxil99qHG5OfO6_iiHBUON83RtNmG-Fm2MCLlgrBEAWoXxq3XrUZB1vnwxdb7YpoYyE=)
27. [emergentmind.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE4xDO6tH9i59LQi4aouxcvCcjCGtlHdRe-F1KEipxaKl-eqa7RmlWkMgvSWqCcnv5ILIeyn6tgxl59tnxbj7tjToC7FG9oQqVX1U20HLxNGBxek7pDgnDQc0IBf7p3zbin5kBcAyZn4pgqtmTqc_WV-3b-6q1PAITT_Qw=)
28. [github.io](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFNM2ToJQeSa3Aoi4pE10kDE5NV1hkT_nkut5ZJ1QJbUPH8_H3zUxYOFH_d14ctaZAHOcBd8RC9gPcDom2BXewb5mvXNGE9h0JXhjbWt-6rWylgSSAmgUgPgtPJQMYENkWDjGz4audnUSvg8QgtUQRBliL3yTOZ5O55)
29. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGvhRAXnPGLBjjtO_wB23sKC9whilnx4qxRRQdHFn6j7iPM3B00_BYipeVZmVMGXdnkAG4GC4g8h0OOGCT-njVb3uWG5n2TYXlSj49S5nNlfSJ_n_Hqem7vpYpHvaGU6u2r)
30. [eleuther.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHdmNjhOiLwiHijzmU2UjCHnFIvCM8iqHg0zzcc-enrtiQFLUK2XL5t58wrHsr0JZF6RlGvrN9XgH2x9ysAitiTH2OuFFxGdQNTMyy-D8Kt5ndaEZK0xJ_wxH8oWKG-5yPk)
31. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHrdqH0UO77q7WPSwAE68UqX7_bcaXjRKFZHycfmCFIylENk673m4VDEqdEFHM6zz0SOL-p6IXfp7b1F6QkOInHyyecSox2zsIfvVfF4tLMHlwxdW82iQ==)
32. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGxHrIB_OQuoyA-SYWFJX1oo1Ln7KdKhRvAGBihh4EUcBsmZMwdANXQtLXg26ImkttmF6U3kTTfMYvVcL9ANCI6jcEyDQs2PWosN7Ja8jhgSMb0bTYoJA==)
33. [apxml.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGfvHPIvK_Pr4UHkSSB_k7-D6Nly2L_cYTHTh0ND2JNkI41i6j6h0yITHGkgOXXBY4JerP9in0XhKubDxzhfsyNrKDpbpEQdKLVPI8QEEYAZusLokvd5oWW8km57Yj2qOUfl7VI3QgzZNWQ9dUyeo4vJZizT1Jh5_9abReSL6J_JpGDB_65eGDuQUXKmJS7EviQZBc4LnJ2eO81LMDChvKTQC0oStChH1FDlIddnLd1mCo_VB3Ki5qR8n-NKaf6UYK49XXra6bmCw==)
34. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEscNyqR_RQy2QMuKXVfcN2Xe0HO4c4RfypDLcqj2Ezz_PBzuNewO7omS4X5MmtYPUR_YSwIzQPS-4wOTB43Cmn37QBrQTKBUA8TkqpPXnRos225907sqGh1Aymh2MMF-YbVcPFafi91ikcsOHs7WvO6_3JCPbzTEc604czFPJMYiYFdhQMLRdOk9g7LdGxIJDiAxTjbmexdOlmQf0MMzWv1w==)
35. [aussieai.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQES0vugHTtqj79W19EIYzkqklo18EB8gTszsLksTyCrzHHUYSjLmexZooiWIN1CBdGoJUFm4MTrVMLkgVjdGkONV0grl1qaszVtHshjJ2CbHZ2aCd_Z9BlKMm-aWkbb-VwbjBmbXwHMkXlWn4ocDw==)
36. [newline.co](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFH-mWI_MVfjsaHQrskzI66uX8NpSEzhu9InGCX25CpLREtlKNMBXFch1oDp6q5xdqydL_q1_r4M_QFFIfYbxxmcYmcM_Rxf0s8Wy9jieKCUeyWlPffPD3FyVO3YtdIzVv0c8c_m48_ZCZvdmHbgOs2er0tKTcjnngMFDh2ZcQyS0Ui)
37. [emergentmind.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFGKzZj9vY_oh0PJf8ji6DSFAqojAyZZMTXeIJfr_aEriHUAat0J1Fb6d7qwldvL5Vwmu3DUtKUwTaK856SJunuJ6ufMwU6frTYpyNnnpV9OZt6Zs91pzkLYHsPvHZ655mVOYyAgvnmTTmWPEE-9aJ3CtEq)
38. [apxml.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE1VFPbpaHQj4RtnKw0UJCWygwZ0gvB4ClzWboa7fbRj7M3YnkRWVpPHwBBu3aIpxEUD1LfVX5Pc3v1FqIEkEVzdo77dwgErq5-GrLCetbfge8VSXD9wnorWQfsfGfioz8Vswcho3IRnllsbhltzdHNCyiLFjvwfeJRh-0eT5V2T3f3chAM08lZH0D5m0GQ_7uiz_9tVQdp02jnQMLwFNdfO2VfPO5YanT7LDqr9apqxEmpkwabbSIwFDVZ)
39. [aclanthology.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGYmr5QOkeM8NSw-CiJVExYQrMBBesuS8VH9JTWTEvQOSGpSQMStvzUlXPEqDwx2DQjFnVslbDl5ddZIiH9ToyAt_h-vKULvet1nlAu_SpbiMghwWsKxi8RPPU2OtGKTWh8vEZn5g==)
40. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF8mAr5YtC9kQ14-iCFuGYPdHC9tsiPZQardqW4qVFJyMOYg5AXtIjd2T7Bd_KZtItfodPzXi94lUI5yLOFGw49c5aAzl0GTVqS5Rn1qFrli3ki05WLc56OevWlNKaGp-GkhWJE1Xwyl9lYnpGHbTm0hXF0Q-QCosiwaS_acPzXW3EmSDgV9ZUw)
41. [fiveable.me](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHgmTWIiEv-Hlc7FLU3txhqxFYDRlMQNb_fbqmFWhAV-8WhvDzZW9bo0EiA7OJ2qMdHuwG2UovmoFK-JlZLnzBQwdqHZnTrF3IA8yEj3NyFgxZzrC5R2ra7y4wmBGXrnQG_yd4ZBHdl1NdlMLANXnSFSB4Cd3rkEXGnVwfs8hxK_M15fHWl7YDMm-SQrAgV-zcbkamIOjk-IL_4kpEe7G4Gem7yZO1VygRi3rNoiH8H)
42. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHx_lP83kkByBIxMantzgBwflgr55D12KsLxfSY7JReW1aciln1CTRAkTHm6ELWTCbIPjnIpM_kbKFF07AA_4SwdwEiQnCAAdnNB4AxI7KKIZRM_2BjWnxHcnFFOPhCLH-WrNsJ-g1zz8aUvScjymoGr4wjZjhfEXqkCGUTN1gVxl4qwnXnXhNAZiq5F6p2z5Z99ItoUrj_2Bun6Ft7uoBa2z7z4Q==)
43. [devstree.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGlKNWGgIS99nQXcN_R9gCG0J3WplOVcLmnGFKZEq3PvWitPmScYB42iLKlXwwm5ohGnfa_kGKTKLRUDAs_7a-NSvNw2KN1eV525pNGOcwgRFAL74LILGpZeGEIoMarNy76ZNE3V2_HDYgiOWA3lJ0uEYGk7fJinQjyeOVCMA==)
44. [wordpress.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE_Y2VsRwy0Ed9wY2hlLL1FG0wvz_t8l1tXA9QnxtDhwtRce7-norF5-Pw3IspqqouP-gcIlve8cGJuNSSQGrY7PujxoVfCxCdrdSm2cCPNs8eboBd3MVIghd-Ig1HDi40RuJrtUezjpn4Rw-9khUN4vaDoiiBCjXSlrYtPyLbpvq7q5X_YLqyji6E4Q6HHga5-DhZauoPOiYDe7TkC9xjzFX5Z)
45. [wordpress.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE-k-ZNivq2EOghHgbuTtQy8bJc8qS1kPm88aBwnTyrinpv7ph7A0V2C9npVoSIkpQDoDFxFGfOdT6syZnNR8TektxCUtcFP81OYtYggHI5oTJE-A0p7TbCcEWbzJSivjpYksv8aiZJgZplipkjut38NybFl--Q9LLulrOcpoWlXWtbxzyCkiUbHriXuDpx7Gb8TEBdRjPwN2AI9w==)
46. [semanticscholar.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHCdB5xF7NI3loktjNSkDNvRHv29PfoXlgAO40OjoSQyWfyFqJVwSCMEJdieHDAERF5MeWxglRzqXVCqKehJH-AvBDpYwPbRmQx-Jxu-4f4OJOSYjJSsXM6k-ygomoHtn3FMks02SmYLqhI2FA59fVJ1IbUgjj0K3LejCMyUJ7cdtKNkGV-7CTGi8QnwMkU0BRfoHlZb9w88zmg_2L1h5xF9YqanSa-3T9B3tW5CjQHjlsXCCBBL18X-NADVldi)
47. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEoI1weHFLzyp9xxQgkGTYb7quiI5mc6m9UM8L8Scs8iYI-wnqFSOGyOxr3vWBSPt4sKnpDY165fYWC4nUou2by7zbqKRqncrj_z16MGXMMB4VF0ASyNw==)
48. [towardsdatascience.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEDxvGRTbVxFWaZXJl7pbJkTC6b_sfUjXBC5QMPHd35-Vs7MjtRCLLlkohw7rBMAelHfvyhVTRpJsuxtk-ecEvHOz44lOMCvlYWscg_KJaGGPlgIZA2IND8gYFvOMZ0CIcki-fS9xim9SX_pac4jNI291jgNwa9ycCtAONpKyl-z4xV-GxA-H3a4sftFrK8p4oPukywz4e4RwjaIcSg0phPiS_5dGj91eZKiqFLGw==)
49. [towardsdatascience.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHsNzrRJQHV7AXcS5D2ODDFwPWvo7A4QC9MY9PDYpDIFWHUGT_-kK7EFSNHpayICI62_eWMtwKoDmr6BwdOldPDJYSHQlv5gxK4koV0DB84vCr8M37j1KC1WEm8sdKfANJDahl-k3-LCSm_Qw7gyEqzIYAplganaKlP5wBsPHGAWv7PJExNdNRYAqyPCFnHGPDMhYvuVEKulxSqqtkpa8TVMRXpAh2cN04ySOjxuxOVVqRmniyjRpYWVVdj)
50. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHI7ZXYy3S92lhp3HoedD9ncO7zPAM3oKkho0Fymxd2vMi0H6sGjTOVcZV2JLUAL9AG-YnLGHYnAI8yhfpNOkcC6301RArFAR3spY56pL90GezjK0O2LoFr1ORONYpzdgfcponW41GSqSbEojW_KifqNIrV-P3y9n0S6PtXLaUMjURj_8UM97w_ydJ539ebZnPH4AMYwUowIIG4-632DqDKGquBCdgr)
51. [tridao.me](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF0K57M-_6gMumv4TPR8UtygKBAoU-HrhvRiTCjaOWoUe2sqXOv3nAQvrU_cAjIK7oOH6vPoyg4c0d9oGSDMjUUDIJ04REGS9o7AQ8oWZvyY0FFcb2tsxPCOg==)
52. [modal.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEYuATOpl9kL88Tb9QCEvpbZfcXIRgpb6cu0Z1uy-P6N473_RyrI5RkjnzhUDlnmzwov9Lil_5ziKvAkhCss8wYhmxeMbrwV3zYSZgDQEATO4NzxhWwjwfHyxtZzZARkiH6wXqE)
53. [geeksforgeeks.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHRdPLRuH8K9cTJT_qrJY3YUaaJ7eEv4kUxcrBdDQvvkmtbVjEg6DXbkBmuwzk0WkFtvCX6jZxrh-RyyRxZWEnY9YnkuVoVTBSf0ccGA4eD2XKFEtH5WKsl4HKH-GVx6StpnEvcI9atGAVHp5iR6PxXaaLsrD-pwgvhu8pCKIqK5-R9yzFIRQ==)
54. [wikipedia.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEVYhWSdlpeEQ9Q6JbrDwPATj0SG7A-x4QVXoSx7MGji2Le4KYPws1mO5rvbRAnROEqLjPOKns4LBdyaJMIjLQPs1ijIRgkuoO0ixU-EBwNqQE1PYL4A-_ZuDcb328oG-gZ0uO6lHg=)
55. [v7labs.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFgEgeXa7GhsdkB7EfL0LNSTuiQR_hLDPKrJT7cT0AlleZcWSe06s6IVBVA_seRNd8OL7I32w5rC-UUfz_J0VF1NI9auNhVzvv9kT_hchtrCC2jw-9EahJNMwleX97LXXqtHOK6Lw9Tu7kC)
56. [viso.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHBOtJi2PUBOiDYRsXcE_bbKPmp6_ZDxLqdjebFXkqKxsCPTX7qS_3EDAxF1RThHN5aPr4qOsA5p7EehH0r6iLW0-ZJPcolWgCZgJkzGoVWBekntGECFinCBSwWZH103pIyzlBjwadroD0fVQ==)
57. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFKVJ-PdvPLAF9cSBXXAjDniaMBVUVagoybHRVAc3F6lSzFeaFAn7xgCKPXbab-v-zlVpDaqNStmpxmSnN5M6plbG72pDP8mZh5Bwwoi1zMKJPMgK8XzAWKuzspMp4rvVQMe08Ok9yAxtgqoV8XOJOK6HQVunpFtOMLY-p5sQkgqP7pmbyr4Wlb9wErs6FOSY40V9xaSyV-nbPBo5IfsrU6BSc=)
58. [huggingface.co](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFgiEaqiD35dzEboayGJFxLbCA867rseMENllm0og6ljvoz6eklfKpnkW3nk72a3pyP94w4QLln2Wa-DDceGtedXnhIUKpzebpZhqtFQbnrM1LERMm1LXEOn1OWPlrA4M3vz9K4r2lnFGYuEMEM16Uzw6j2OOAyVN2QBAdcE-hsjXXGlR7cp8Mi5XKkchTI)
59. [datacamp.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHIlosF_f5e89uGzNoOY3omTvR1X9O_MYXwkCfXinyuGY1P7S0Ci_6LxMHFrtEV_Ox82ALlRUryp7M6rClV-1x2WR1UP0AzMDrKzRjC6oA7L_agJm_g5Hvo_mvBoNony_dWecDga56WLlc5Y5mK)
60. [alphaxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHUTTH8QflKojvCqntaaZqIEudL3kNDRLT4t1vseXzWBof1wxpbEmCGxgDZZYrwxBXk7xyob0CZbQMiXGh-H3xjmjHk7khGEYEZ65iV7zI7-qzP6yAez9psPfwRtXxuTu0TVeoh)
61. [programming-ocean.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH9M8NnqIfusM_7vEJ3vE5dRmPDm8UJD5YfE9nOps-a7K5DpLS8LpK_CqugZgRll8XZ24iC1N0aBvXyEuqmXSz454UK7jzfS4MqrGjqGFry7krZ0EnlKqv5bjAPFR7kfyMBfKSfw_3U_2qQ-XqslmIg684F-T-S4-X7jnH0GZM=)
62. [hopsworks.ai](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5EW3didq0oUm7x8zqJPfH-YbtYDiE-Afo6wUafjBkIQ8mvjjJeL8Iu5xhyDsWIY-xhYcIqirKN6wW_ltWTuM-RW0jyLeqcVlKdY9mw1HrkEKxl9Rh8tO6srhjc6nrhM6r0XYbjFbdbr0=)
63. [reddit.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHM_iVS9JmMw0mzbKGoJpM89VQFNEg3CXjW4NquUZbt-eHk9d_lk0eeAdj6qVS4vnc7BFaBk3s7gUdLgeuPa90ATIHDKR7kchM4iCZ5zZUTEwf3jhxr6BUSr5aHZIX5c-EYBIEbrTyP6VrIDfM2LoXBKkikxsbA3bDWMz4Pys2q2q44IedeQvrBopAiqOHt-FD1kr4q)
64. [roboflow.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGShw-9JT4ZoLsRDrVvirU0-zxsniIDGFTYowGJ8mYzSab73XM-XRcZcSq_deUb-Du2gtkUWuKYc5MUx0SimR_S17KITtaRVws3N0FoEfI0_qZCKymRBhJrPm6RISAgpIVKZymU)
65. [aimind.so](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGpfdpx4lfJh-NzyCU76TKqfGfPbznJCbZ9N5joVKLva47AeF9kjCAvZxQxObEi_Pg2zvry4XuVyQKzlWEIA20mStbPwsMtgABvJXN4gGK34BdYXyYFkvcSRhXKgjBtX1YYfOFQm4ib8CFhuua6EQT1cK8H0GzlaOWgvp5b-ptoGM2OPldJGVNHngfX_XcXJy8fHDsk)
66. [medium.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFcYTOVlZvVwjrPjzjacA16wQVkMFHH2bM3oSyZiVL6UL6vtfGJgPMeSUkvKLwWw-EQYGvCXV3XX93QdeLDIVFnMxh3xDA4o8u7JgTbD9yd3WoU3r4cwgQbet3qbK3lLX8EQdVML3a12dg4Rqvs4eyyHGAKwf5YwZndE6FNQDiRQxEtIpCCNM0D8VnKm7nnl0Wjux6zbtmUTq9LMbG5LSWziUML)
67. [uselessai.in](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFyNQB1oCP0EFXDBjKNvNPba6LwKZbBhpfB2-ZneuoEJVAT6J7BwlFfY6ZSQvmh1Sa_dxrfDQhhuc0qpdsQd8CSJuNUBxM2m8ZkCmqNAvTCc5u6zEs_X1JPYElLJ7Xo3gk0vLFccUPphcCy4N7s8PJ9r5Zm_VL6nR-Z_iLo3nnhVtlUan6URw-TRHFQmBpUV0qWDEhFjFOGU20fJQWpRrNQ8mcVkN0=)
68. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFR7yLIzUC8-pW7-uPb_oIA5hZwFClAcbJVC86LuYku2fdkKNHcdKvNj6SHH9srLgCgaLGOoFyIG9JhARoBry8A38vlDMUsPoTR08oQIWi_otHPA7iIdA==)
69. [arxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF6c57S0UfzxAk6KfuaiE8pHv31QuWwIJTVuLdB2-0FpuzdjaRmvxoGi5dPx3tQQ3GDIP1iscSx3GwLk7g43DQtDJpwI1vXaoe1gruYGSsrLEsPFRKslQ==)
70. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGJlsvpxUdUFqcuamyDJ2RNIZqzcE28x81dPg_mcBfPMt_s-Ft8a1DJPt57rN0VOD2UmxXtEH4j5dnHrIboeocw3P2m23LGmNBerDLRP5vQVmc-GO4mGAjzIgVzzrKu1fpQ)
71. [codecademy.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGxW0ChNgzSXZHm8UcTUWFUipgUgkdLLfH2EM6I1MlOzrzfxdYzsG_8EuBhY_0mBW0TQGHFSTWaUsKhlQX2r0h_YBboJMcD7hrPCjtcTw7TJxiyHuFzrvAQ-3C1jdq22KeTRuBzybwkK18466LKh15DMszzWHVpSCrERPTdzs0FutcgLxfIcV2CknAf)
72. [datacamp.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFBzc1Q6eczRef9GZW6VG_7-QfmJEnp1flJd-h-65d82p0fTJbhigqMFMA2o9Wkfg9hAHipG6TuLSA_MIPCGzNV5aMfgAKSzp8TxYZLargNcyhw_vqNSFEYbA40V6phAJFlq_Z3BCZ-pbNqsg==)
73. [alphaxiv.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFmXwGywSqHUircqzaqWR1iDBvqxUBCzeYa1URdPsT8wydgft-rgvSD_VZRkeMaO_B1N0W_MbfL7dcMWzc_SXNh5_WcgZCdcCB255sPWNoWbF-_MqfkfLKYqb2dirNFA-oRyw==)
74. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHhgjp1QUV1immafiGYeHgW3--4xBQ5lwF-M5qJc_kAV38I03rTMmGEhvXoXPP49Z7aizhfhTCquf2I6hmGXb3KEOOnMZAHVF-7XA3n8Of4khNnmwnxOTUAmQoRwEDWahoBboBDTObMqYI_y6puW9m8P1lz83oNRb6uUiN3wxeUSRAm-wMGfRTAzO5ZzKpeK37slEXYy5SwTfoIc9c=)
75. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHq2FfoeKQ5lvhLYGgqbVcOx03UdBWGcfKnvMAFGvEaCCbwoz60oehFjuCYfbv91oJkiorQ3iBYu6Yx8zfv11VBoS3f5B59BWlk4npE53YUQa7wZc951upb3GRkizm--ytDBEa5Fs2Knyw9krXRBqkcxQcxQQbOiIW8Hc7iZ6HnGJaQOB4IlgYR4qmg3XnJXgIYbumgkqIu2_Uo7brXGzSFRiNAu6kzk_RCffqQktu6L0Dzrw==)
76. [semanticscholar.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE7pMmdMh_HJZtQ1MmtkaUM5_gb4wS7mR9wQIbUQniQaXNTsZIAMlXdd8it-MjZEij-XJPtar8z37eZ8mffLSNi3MOhm2KPDhHxvuVWIciuxygvjEaIu4V0VvigPnfHMfO3EjoQGf_MS_zWt_XCj_C2VZ354sOvU4VZv5j_AlqGpb6jW73hGbR4fZBcUU0ksEJJmUyNGhH0TOmHAB56-es0n5F26xIgpCOZocRL8Ok=)
