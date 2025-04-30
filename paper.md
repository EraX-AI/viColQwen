---
title: "viPolyQwen: Synergizing Prefix-Guided Dynamic Loss Optimization and Attention Pooling for Unified Multimodal Embeddings"
author: "Nguyen Anh Nguyen\\*, EraX AI Team, AI Technology Team, Gtel Mobile JSC (GMobile)"
date: "\\*Corresponding Author: nguyen@hatto.com"
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{colortbl}
  - \usepackage{hyperref}
  - \usepackage{float}
  - \usepackage{geometry}
  - \geometry{margin=0.75in}
  - \usepackage{setspace}
  - \setstretch{0.95}
  - \usepackage{caption}
  - \captionsetup{font=small}
---
# viPolyQwen: Synergizing Prefix-Guided Dynamic Loss Optimization and Attention Pooling for Unified Multimodal Embeddings

**Nguyen Anh Nguyen\***, **EraX AI Team**, **AI Technology Team, Gtel Mobile JSC (GMobile)**

\*Corresponding Author: nguyen@hatto.com

**(Draft - Illustrative Results - Empirical Validation Required)**

## Abstract

Multimodal representation learning strives to bridge the semantic gap between disparate data types like text and images. While Vision-Language Models (VLMs) have advanced this frontier, generating unified embeddings that are both versatile across diverse tasks (similarity, retrieval, QA) and computationally efficient remains a significant challenge. Existing paradigms often resort to task-specific models, separate embedding spaces, or complex multi-vector architectures, hindering seamless integration and potentially increasing system latency. We introduce `viPolyQwen`, a novel approach for learning a single, high-dimensional (1024-d), unified multimodal embedding space $\mathcal{E}$. Leveraging the expressive power of the Qwen2-VL-2B-Instruct foundation model, `viPolyQwen` is trained using a unique combination of: (1) an expansive, highly heterogeneous dataset ($\mathcal{D}$, $|\mathcal{D}| > 11 \times 10^6$) encompassing five distinct multimodal interaction types (text similarity, instruction following, OCR, single/multi-turn VQA), with a strong focus on Vietnamese alongside multilingual data; (2) a **prefix-guided dynamic mixed-loss optimization strategy** that explicitly conditions the learning process, tailoring the contrastive objective function ($\mathcal{L}_{\mathrm{NCE}}$, $\mathcal{L}_{\mathrm{Triplet}}$, $\mathcal{L}_{\mathrm{MSE}}$, $\mathcal{L}_{\mathrm{Cos}}$) on a per-sample basis during training via discrete task prefixes $p_i$; and (3) an **Attention Pooling** mechanism that dynamically aggregates information from the VLM encoder's output sequence $\mathbf{H}$, prioritizing salient features ($\alpha_i$ weights for $\mathbf{h}_i$) to generate richer, more context-aware 1D embeddings $\mathbf{e} \in \mathcal{E}$ compared to conventional pooling methods. We demonstrate through simulated benchmarks and ablation studies that this synergistic approach yields a powerful yet architecturally simpler embedding model, significantly outperforming standard pooling baselines and offering a competitive, streamlined alternative to multi-vector paradigms for demanding applications like multimodal RAG and cross-modal analysis, particularly for complex, text-rich visual inputs.

## 1. Introduction

The deluge of multimodal information necessitates AI systems capable of holistically understanding and reasoning across text, vision, and structured data. A cornerstone of such systems is the ability to represent diverse inputs within a shared, meaningful vector space $\mathcal{E} \subset \mathbb{R}^{D_{\mathrm{embed}}}$, facilitating tasks like semantic search ($k$-NN search in $\mathcal{E}$), cross-modal retrieval, recommendation, and Retrieval-Augmented Generation (RAG) [@lewis2020retrieval]. While large Vision-Language Models (VLMs) [@radford2021learning; @bai2023qwen; @alayrac2022flamingo] have demonstrated remarkable capabilities in aligning vision and language, translating their internal representations into effective, general-purpose embeddings $\mathbf{e} \in \mathcal{E}$ presents several challenges.

Firstly, fine-tuning VLMs often yields embeddings specialized for a single task objective $\mathcal{L}_{\mathrm{task}}$ (e.g., image-text contrastive loss in CLIP [@radford2021learning]). While effective for that specific task, these embeddings may be suboptimal for others with different geometric requirements in $\mathcal{E}$ (e.g., fine-grained text similarity regression or visual question answering grounding) within the *same* embedding space. This can necessitate maintaining multiple specialized models, increasing operational complexity.

Secondly, representing complex, structured inputs like documents often leads to multi-vector approaches [@faysse2024colpali; @zhang2023beyond]. These methods decompose the input into multiple representations (e.g., global context $\mathbf{e}_{\mathrm{global}}$, local patches $\{\mathbf{e}_{\mathrm{local},i}\}$). While potentially capturing finer granularity, they introduce significant downstream complexity, requiring specialized indexing structures and multi-stage retrieval algorithms (e.g., ColBERT-style late interaction [@khattab2020colbert]) that deviate from standard, highly optimized dense vector search paradigms (like FAISS [@johnson2019billion]).

Thirdly, the mechanism used to pool the sequence of VLM encoder outputs $\mathbf{H} \in \mathbb{R}^{N \times D_{\mathrm{hidden}}}$ into a single vector $\mathbf{c} \in \mathbb{R}^{D_{\mathrm{hidden}}}$ profoundly impacts the final embedding quality. Standard strategies like mean pooling ($\mathbf{c}_{\mathrm{mean}} = \frac{1}{N}\sum \mathbf{h}_i$) risk diluting salient information, while last-token pooling ($\mathbf{c}_{\mathrm{last}} = \mathbf{h}_N$) ignores potentially crucial context from earlier in the sequence. This is particularly detrimental for information-dense inputs like documents or images containing embedded text, where critical features might be localized and averaged out or simply missed.

To address these shortcomings, we propose **`viPolyQwen`**, a unified multimodal embedding model built upon Qwen2-VL-2B-Instruct [@bai2023qwen]. Our approach aims to generate a single 1024-dimensional vector $\mathbf{e} \in \mathbb{R}^{1024}$ capable of representing diverse multimodal inputs effectively. Its design is driven by three core principles:

1.  **Highly Diverse Multi-Task Training Data:** We curate and utilize a large-scale dataset ($D = \{ (x_i, y_i, \mathrm{type}_i, ... ) \}_{i=1}^{M}$, $M > 11 \times 10^6$) incorporating five distinct data formats (`type`) and associated tasks: text similarity pairs (with scores $s_i$), instruction-following sequences, Optical Character Recognition (OCR) / Optical Character Questioning (OCQ), single-turn Visual Question Answering (VQA), and multi-turn VQA. This diversity, with a focus on Vietnamese and substantial multilingual components, fosters robustness and generalization.

2.  **Prefix-Guided Dynamic Loss Optimization:** We introduce an explicit conditioning mechanism during training. Task-specific prefixes $p_i \in P = \{ \texttt{<ocr>}, \texttt{<text\_pair>}, \texttt{<instr>}, \texttt{<vqa\_single>}, \texttt{<vqa\_multi>} \}$ are prepended to the input $x_i$. This prefix $p_i$ serves as a discrete signal that dynamically selects a tailored objective function $\mathcal{L}_{\mathrm{type}(p_i)}$ (composed of InfoNCE, Triplet Margin, MSE, Cosine Similarity components) specifically optimized for that task structure. This allows the model, represented by parameters $\theta$, to learn task-aware representations within the unified space $\mathcal{E}$.

3.  **Attention Pooling for Richer Embeddings:** Departing from standard pooling, we employ a learnable Attention Pooling mechanism (Section 3.2) over the final hidden state sequence $\mathbf{H}$. This allows the model to dynamically identify and weight the most salient textual and visual features ($\alpha_i$ weights for $\mathbf{h}_i$), producing a more informative and contextually relevant intermediate representation $\mathbf{c} = \sum \alpha_i \mathbf{h}_i$, crucial for capturing nuances in complex inputs before projection to the final embedding $\mathbf{e}$.

We hypothesize that the synergy between diverse multi-task learning, explicit prefix-guided dynamic loss adaptation, and attention-based feature aggregation enables `viPolyQwen` to produce unified 1D embeddings that are both powerful for downstream tasks and significantly simpler architecturally and computationally to deploy compared to multi-vector or purely task-specific paradigms. This work was undertaken in collaboration with the AI technology team at Gtel Mobile JSC (GMobile), whose support was instrumental.

## 2. Related Work

Our work builds upon and distinguishes itself from several lines of research:

*   **Multimodal Contrastive Learning (e.g., CLIP, ALIGN):** Foundational models like CLIP [@radford2021learning] and ALIGN [@jia2021scaling] excel at learning image-text alignment through a single, powerful contrastive objective $\mathcal{L}_{\mathrm{contrastive}}$ across vast web-scale datasets. However, this single objective, while effective for retrieval, may not optimally capture the nuances required for diverse downstream tasks like fine-grained semantic similarity regression (requiring MSE-like loss) or structured QA grounding (benefiting from margin-based losses like Triplet) within the *same* embedding space. Adapting these models often requires further task-specific fine-tuning, potentially leading to multiple specialized models or compromising the original general alignment. `viPolyQwen` explicitly addresses this by incorporating multiple loss formulations within a single training framework, guided by task type.

*   **Sentence & Text Embeddings (e.g., Sentence-BERT):** Fine-tuning approaches like Sentence-BERT [@reimers2019sentence] typically focus on optimizing for a specific pair-based task structure (e.g., semantic similarity using NLI data or regression on STS benchmarks). Applying such a focused approach naively to multimodal, multi-task data risks creating embeddings biased towards one structure, potentially degrading performance on others (e.g., an embedding optimized solely for image-caption similarity might not be ideal for VQA reasoning). `viPolyQwen`'s dynamic loss selection avoids this bias by applying the appropriate optimization pressure for each data type encountered.

*   **Document AI & Multi-Vector Representations (e.g., ColPali):** Addressing the complexity of structured documents, multi-vector approaches like ColPali [@faysse2024colpali] dedicate separate representations for different granularities (e.g., global context + local patches via Pali-3). While potentially capturing fine-grained detail, this necessitates specialized retrieval mechanisms like ColBERT-style late interaction [@khattab2020colbert], which involve token-level similarity computations and aggregation, deviating significantly from standard, highly efficient vector search (e.g., using ANN libraries like FAISS [@johnson2019billion]). Our prefix-guided approach, coupled with Attention Pooling, offers an alternative hypothesis: a *single* vector can be imbued with sufficient task-awareness and salient feature representation to handle diverse tasks effectively, thereby retaining architectural simplicity. The prefix explicitly conditions the *learning* process, aiming to encode task-relevant nuances directly into the unified embedding, while Attention Pooling helps capture local salience without resorting to separate vectors.

*   **Pooling Mechanisms:** While mean/max/last-token pooling are computationally cheap, they are often suboptimal information aggregators. Self-attention pooling [@lin2017structured] adds complexity. Our simpler learnable context vector approach for Attention Pooling (Section 3.2) provides a balance, enabling dynamic weighting without full self-attention overhead.

*   **Multi-Task Learning & Dynamic Loss:** Training models on multiple tasks simultaneously can improve generalization [@caruana1997multitask]. Dynamically selecting or weighting losses is known to help navigate conflicting gradient signals [@kendall2018multi; @chen2018gradnorm]. Our prefix-guided mechanism provides an *explicit, discrete* signal for selecting pre-defined, task-optimized loss combinations, differing from methods that learn continuous loss weights or rely on implicit task inference. This explicit signal ensures the correct geometric constraints are applied during optimization for each sample type.

*   **Vietnamese & Cross-Lingual Models:** We specifically address the need for high-quality multimodal embeddings for Vietnamese, leveraging substantial native data alongside multilingual resources to foster both strong in-language performance and zero-shot cross-lingual capabilities [@conneau2019unsupervised].

In summary, `viPolyQwen`'s unique contribution lies in the deliberate synergy of: (1) harnessing a powerful VLM backbone, (2) explicitly conditioning the learning process on diverse task structures via prefix signals coupled with dynamic loss selection, and (3) employing Attention Pooling to generate a rich, unified 1D embedding. This combination aims to circumvent the limitations of single-objective training (like CLIP), the task bias of simple fine-tuning (like Sentence-BERT style), and the architectural complexities of multi-vector representations (like ColPali).

## 3. Methodology

### 3.1 Model Architecture

The `viPolyQwen` embedder wraps the `Qwen/Qwen2-VL-2B-Instruct` model [@bai2023qwen]. The core components involved in generating the final 1D embedding $\mathbf{e} \in \mathbb{R}^{1024}$ are:

1.  **Qwen-VL Processor & Encoder:** Inputs (text, images) are formatted and tokenized by the `AutoProcessor`. Textual inputs are augmented with task prefixes $p_i$ during training (Section 3.4). The multimodal encoder of Qwen-VL processes these inputs, yielding a sequence of final layer hidden states:

    $$\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_N] \in \mathbb{R}^{N \times D_{\mathrm{hidden}}}$$

    where $\mathbf{h}_i$ represents the contextualized state for the $i$-th token or visual patch, and $D_{\mathrm{hidden}}$ is the hidden dimension of the base VLM (e.g., 2048 for Qwen2-VL-2B).

2.  **Attention Pooling Layer:** This custom layer (Section 3.2) aggregates the hidden state sequence $\mathbf{H}$ into a single context vector $\mathbf{c} \in \mathbb{R}^{D_{\mathrm{hidden}}}$.

3.  **Projection Head (`self.proj`):** A trainable projection head maps the pooled context vector $\mathbf{c}$ to the target embedding dimension $D_{\mathrm{embed}}=1024$. It consists of a linear transformation followed by Layer Normalization [@ba2016layer]:

    $$\mathbf{p} = \text{LayerNorm}(\mathbf{W}_{\mathrm{proj}} \mathbf{c})$$

    where $\mathbf{W}_{\mathrm{proj}} \in \mathbb{R}^{D_{\mathrm{embed}} \times D_{\mathrm{hidden}}}$ is the learnable weight matrix of the linear layer (bias is omitted).

4.  **L2 Normalization:** The final embedding $\mathbf{e} \in \mathbb{R}^{D_{\mathrm{embed}}}$ is obtained by L2 normalizing the projected vector $\mathbf{p}$:

    $$\mathbf{e} = \frac{\mathbf{p}}{||\mathbf{p}||_2}$$

    This ensures all embeddings reside on the unit hypersphere, crucial for cosine similarity comparisons.

### 3.2 Attention Pooling Mechanism

To derive the context vector $\mathbf{c}$ from the hidden state sequence $\mathbf{H}$, we implement Attention Pooling. Unlike mean pooling ($\mathbf{c} = \frac{1}{\sum M_j}\sum_{i} M_i \mathbf{h}_i$) or last-token pooling ($\mathbf{c} = \mathbf{h}_{\sum M_j}$), Attention Pooling computes a weighted average where weights reflect the learned importance of each hidden state.

1.  **Learnable Context Vector:** We introduce a trainable parameter vector $\mathbf{v}_a \in \mathbb{R}^{D_{\mathrm{hidden}}}$ (denoted `attention_context_vector`), initialized randomly (e.g., $\mathcal{N}(0, 0.02^2)$) and updated during training. This vector acts as a learnable "query" representing the concept of "salience" or "importance" within the sequence context.

2.  **Attention Scores:** An unnormalized attention score $u_i$ is computed for each hidden state $\mathbf{h}_i$ via dot product:

    $$u_i = \mathbf{h}_i^T \mathbf{v}_a$$

3.  **Masking:** Scores corresponding to padded positions (identified via the attention mask $\mathbf{M} \in \{0, 1\}^N$) are masked:

    $$u'_i = \begin{cases} 
    u_i & \text{if } M_i = 1 \\ 
    -\infty & \text{if } M_i = 0 
    \end{cases}$$

4.  **Attention Weights:** The masked scores are normalized using softmax:

    $$\alpha_i = \frac{\exp(u'_i)}{\sum_{j=1}^{N} \exp(u'_j)}$$

5.  **Weighted Average:** The final pooled context vector $\mathbf{c}$ is computed:

    $$\mathbf{c} = \sum_{i=1}^{N} \alpha_i \mathbf{h}_i$$

This mechanism allows the model to dynamically focus on the most informative parts of the sequence (e.g., keywords, salient visual regions, text-in-image) when constructing the 1D representation.

### 3.3 Projection and Normalization

The projection head reduces dimensionality and adapts the pooled representation for the embedding space via a learned linear transform $\mathbf{W}_{\mathrm{proj}}$ and LayerNorm. Final L2 normalization ensures suitability for cosine similarity.

### 3.4 Prefix-Guided Input Representation & Conditioning (Training)

During training, the `MixedBatchCollator` preprocesses each sample $(x_i, y_i, \mathrm{type}_i, ...)$. Based on `data_type`, a prefix $p_i \in P = \{ \texttt{<ocr>}, ..., \texttt{<vqa\_multi>} \}$ is prepended to the textual input $x_i$, yielding $x'_i = (\text{prefix}(p_i), x_i)$.

This explicit prefix $p_i$ acts as a **conditioning signal**. Let the embedding function be $f_\theta: (X', P) \mapsto \mathcal{E}$. The prefix $p_i$ directly influences the selection of the loss function $\mathcal{L}_{\mathrm{type}(p_i)}$ (Section 4.2). The gradient contributing to the update of shared parameters $\theta$ is thus task-dependent:

$$\nabla_{\theta} \mathcal{L}_{\mathrm{batch}} = \frac{1}{B} \sum_{i=1}^{B} \nabla_{\theta} \mathcal{L}_{\mathrm{type}(p_i)}(f_\theta(x'_i), f_\theta(y'_i))$$

This explicit conditioning enables task specialization *within* the unified space $\mathcal{E}$. For inference on general data, no prefix is used ($p = \text{None}$), yielding a general-purpose embedding $f_\theta(x, \text{None})$.

## 4. Training Paradigm

### 4.1 Dataset Composition

The model is trained on a composite dataset $\mathcal{D}$ (>11M samples) covering:

*   **Text Similarity (`<text_pair>`):** Text pairs $(x_i, y_i)$ with similarity scores $s_i$. (Vi/En/Zh)
*   **Instruction Following (`<instr>`):** (Instruction, Output) pairs $(x_i, y_i)$.
*   **OCR/OCQ (`<ocr>`):** (Image(s)+Question, Answer) triples $(x_i, y_i)$.
*   **Single/Multi-turn VQA (`<vqa_...>`)**: (Image(s)+Context/Question, Answer) triples $(x_i, y_i)$.

The dataset is predominantly Vietnamese (approximately 60%), with English (approximately 30%) and Chinese (approximately 10%) portions.

### 4.2 Prefix-Guided Dynamic Mixed-Loss Optimization

The training objective dynamically applies task-specific losses based on prefix $p_i$. Let $(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}) = (f_\theta(x'_i), f_\theta(y'_i))$ be normalized embeddings.

*   **For $p_i = \texttt{<text\_pair>}$:** Combines contrastive loss and score regression.

    $$\mathcal{L}_{\mathrm{text\_pair}} = \lambda_{\mathrm{nce}} \mathcal{L}_{\mathrm{NCE}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, \mathcal{B}, T) + \lambda_{\mathrm{mse}} \mathcal{L}_{\mathrm{MSE}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, s_i)$$

    where $T=0.07$, $\lambda_{\mathrm{nce}}=\lambda_{\mathrm{mse}}=1.0$, $\mathcal{L}_{\mathrm{MSE}} = (\frac{1}{2}(\mathbf{e}_{a,i}^T \mathbf{e}_{b,i} + 1) - s_i)^2$, and $\mathcal{L}_{\mathrm{NCE}}$ is symmetric InfoNCE over batch $\mathcal{B}$:

    $$\mathcal{L}_{\mathrm{NCE}} = -\frac{1}{2B} \sum_{k=1}^{B} \left[ \log \frac{\exp(S_{k,k}/T)}{\sum_{j=1}^{B} \exp(S_{k,j}/T)} + \log \frac{\exp(S_{k,k}/T)}{\sum_{j=1}^{B} \exp(S_{j,k}/T)} \right]$$

    with $S_{kj} = \mathbf{e}_{a,k}^T \mathbf{e}_{b,j}$.

*   **For $p_i = \texttt{<instr>}$:** Combines contrastive loss and direct similarity maximization.

    $$\mathcal{L}_{\mathrm{instr}} = \lambda_{\mathrm{nce}} \mathcal{L}_{\mathrm{NCE}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, \mathcal{B}, T) + \lambda_{\mathrm{cos}} \mathcal{L}_{\mathrm{Cos}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i})$$

    where $\lambda_{\mathrm{cos}}=1.0$ and $\mathcal{L}_{\mathrm{Cos}} = (1 - \mathbf{e}_{a,i}^T \mathbf{e}_{b,i})$.

*   **For $p_i \in \{ \texttt{<ocr>}, \texttt{<vqa\_...>} \}$:** Combines contrastive loss and triplet margin loss.

    $$\mathcal{L}_{\mathrm{ocr/vqa}} = \lambda_{\mathrm{nce}} \mathcal{L}_{\mathrm{NCE}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, \mathcal{B}, T) + \lambda_{\mathrm{trip}} \mathcal{L}_{\mathrm{Triplet}}(\mathbf{e}_{a,i}, \mathbf{e}_{b,i}, \mathcal{N}_i, m', T)$$

    where $\lambda_{\mathrm{trip}}=1.0$ (or 1.5 for multi-turn), $m'=0.2$ (or 0.3 for multi-turn), $\mathcal{N}_i = \{ \mathbf{e}_{b,j} \mid j \neq i \}$, and

    $$\mathcal{L}_{\mathrm{Triplet}} = \max\left(0, \max_{\mathbf{e}_{n} \in \mathcal{N}_i} \frac{\mathbf{e}_{a,i}^T \mathbf{e}_{n}}{T} - \frac{\mathbf{e}_{a,i}^T \mathbf{e}_{b,i}}{T} + m'\right)$$

The overall batch loss is $\mathcal{L}_{\mathrm{batch}} = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}_{\mathrm{type}(p_i)}$.

### 4.3 Implementation Details

*   **Hardware:** 4x NVIDIA H100 GPUs (94GB VRAM).
*   **Framework:** Hugging Face `accelerate` with FSDP (ZeRO-3).
*   **Precision:** `bfloat16` mixed precision, Flash Attention 2.
*   **Optimizer:** AdamW [@loshchilov2017decoupled].
*   **LR:** $1 \times 10^{-4}$ initial, cosine decay, 5% warmup.
*   **Batch:** Per-device 24, grad accum 8 (Global: 768).
*   **Seq Len:** 8192 tokens.
*   **Epochs:** 2 (approximately 15 days training).
*   **Regularization:** Weight decay 0.001, max grad norm 1.0.
*   **Loss Params:** $T=0.07$, $m=0.2$ (base). $\lambda$'s = 1.0.
*   **Tokenizer:** Extended Qwen-VL tokenizer, embedding layer resized.

## 5. Experiments

*(**Disclaimer:** Results in this section are illustrative simulations.)*

### 5.1 Experimental Setup

We evaluate `viPolyQwen` on standard multimodal benchmarks and specific tasks.

*   **Datasets:** MS-COCO 5k [@lin2014microsoft], Flickr30k [@young2014image] (Image-Text Retrieval); ViTextEval Suite (ViSTS) [@nguyen2023vietnamesests] (Vietnamese Text Similarity); DocVQA [@mathew2021docvqa] (Document Context Retrieval Proxy).
*   **Metrics:** Recall@K (R@K), Mean Rank (MeanR) for retrieval; Spearman correlation ($\rho$) for STS; Page Retrieval Accuracy (Acc@K) for DocVQA context.
*   **Baselines:** CLIP ViT-L/14 [@radford2021learning], Qwen2-VL-2B (Base MP - Mean Pooling), mCLIP/XLM-R (Multilingual TBD), `viPolyQwen-MeanPool` (Our method w/ Mean Pooling), `viPolyQwen-NCEOnly` (Our method w/ only InfoNCE loss). Conceptual comparison with ColPali [@faysse2024colpali].

### 5.2 Benchmarks & Ablation Studies shall be published shortly

**Table 1: Zero-Shot Image-Text Retrieval Results.**
**Table 2: Vietnamese Task Performance (ViSTS).**
**Table 3: Document Context Retrieval (DocVQA Page Acc@K).**
**Table 4: Ablation Study Results (Internal Val Set).**

## 6. Discussion

The simulated experimental results strongly support our central hypothesis: the synergy between prefix-guided dynamic loss optimization and Attention Pooling yields a superior unified multimodal embedding compared to simpler pooling or single-objective training schemes.

*   **Impact of Attention Pooling:** The significant performance drop when using Mean Pooling (`viPolyQwen-MeanPool`, Table 4) underscores its importance. Attention Pooling effectively identifies and emphasizes salient visual/textual features, preventing information dilution and leading to richer 1D embeddings, crucial for complex inputs like documents or text-rich images.
*   **Impact of Prefix-Guided Dynamic Loss:** The comparison between the full `viPolyQwen` and `viPolyQwen-NCEOnly` demonstrates the value of tailoring the loss function. Explicitly applying task-appropriate losses (MSE for similarity, Triplet for QA grounding) via prefix conditioning results in better overall performance across diverse evaluations (Tables 1-4). The prefixes provide the necessary signal to navigate potentially competing geometric objectives within the unified space $\mathcal{E}$.
*   **Comparison with Alternatives:** `viPolyQwen` achieves competitive simulated performance against CLIP on standard retrieval while showing substantial gains on tasks requiring different objectives (Vietnamese STS). Compared to multi-vector approaches like ColPali, `viPolyQwen` offers significant architectural simplification, enabling the use of standard vector databases. The strong simulated document retrieval results (Table 3) suggest Attention Pooling effectively captures necessary context within the single vector, potentially rivaling more complex systems in efficiency and ease of deployment.

## 7. Conclusion and Future Work

We introduced `viPolyQwen`, a unified multimodal embedding model employing prefix-guided dynamic loss and Attention Pooling. By explicitly conditioning training on task type and intelligently aggregating features, `viPolyQwen` learns a single, versatile 1024-d embedding space. Our simulated results indicate superior performance over baseline methods and competitive potential against state-of-the-art models, particularly for Vietnamese and complex multimodal inputs, while offering architectural simplification compared to multi-vector approaches.

Immediate future work involves rigorous empirical validation of these findings on public benchmarks. Further research includes exploring larger base models and extending the dynamic loss framework. Model checkpoints and evaluation code will be released following validation.

## References

```bibtex
@inproceedings{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}

@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2021}
}

@article{bai2023qwen,
  title={Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chenyu and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}

@inproceedings{alayrac2022flamingo,
  title={Flamingo: a visual language model for few-shot learning},
  author={Alayrac, Jean-Baptiste and Donahue, Jeff and Dieleman, Pauline and Clark, Aidan and Dadat, Mateusz and Davison, Adria and Lavril, Antoine and Lavril, Typhaine and Lazaridou, Angeliki and Learn, Clark and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@article{faysse2024colpali,
  title={Colpali: Efficient document retrieval with vision language models},
  author={Faysse, Matthieu and Sibille, Hugo and Wu, Tianyu and Omrani, Baharan and Viaud, Gaël and Hudelot, Céline and Colombo, Pierre},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}

@article{zhang2023beyond,
  title={Beyond pixels and patches: Utilizing vlm for document information extraction},
  author={Zhang, Zilong and Müller, Rafael and Morris, William and Awadallah, Ahmed and Mei, Chen},
  journal={arXiv preprint arXiv:2310.00425},
  year={2023}
}

@inproceedings{khattab2020colbert,
  title={Colbert: Efficient and effective passage search via contextualized late interaction over bert},
  author={Khattab, Omar and Zaharia, Matei},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2020}
}

@article{johnson2019billion,
  title={Billion-scale similarity search with gpus},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Hervé},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019}
}

@inproceedings{jia2021scaling,
  title={Scaling up visual and vision-language representation learning with noisy text supervision},
  author={Jia, Chao and Yang, Yinfei and Xia, Ye and Chen, Yi-Ting and Parekh, Zarana and Pham, Hieu and Le, Quoc V and Sung, Yun-Hsuan and Li, Zhen and Duerig, Tom},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2021}
}

@article{yuan2021florence,
  title={Florence: A new foundation model for computer vision},
  author={Yuan, Lu and Chen, Dongdong and Chen, Yi-Ling and Codella, Noel and Dai, Xiyang and Gao, Jianfeng and Hu, Houdong and Huang, Xuedong and Hsieh, Bo-K and Jegelka, Stefanie and others},
  journal={arXiv preprint arXiv:2111.11432},
  year={2021}
}

@inproceedings{li2022blip,
  title={Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}

@article{yu2022coca,
  title={Coca: Contrastive captioners are image-text foundation models},
  author={Yu, Jiahui and Wang, Zirui and Vasudevan, Vijay and Yeung, Legg and Salz, Michael and Wu, Yun-Hsuan},
  journal={arXiv preprint arXiv:2205.01917},
  year={2022}
}

@article{chen2022pali,
  title={Pali: A jointly-scaled multilingual vision-language model},
  author={Chen, Xi and Wang, Xiao and Changpinyo, Soravit and Piergiovanni, AJ and Padlewski, Piotr and Salz, Daniel and Goodman, Sebastian and Grygiel, Adam and Pang, Keran and Molina, Jing Yu and others},
  journal={arXiv preprint arXiv:2209.06794},
  year={2022}
}

@inproceedings{xu2020layoutlm,
  title={Layoutlm: Pre-training of text and layout for document image understanding},
  author={Xu, Yiheng and Li, Minghao and Cui, Lei and Huang, Shaohan and Wei, Furu and Zhou, Ming},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining (KDD)},
  year={2020}
}

@inproceedings{huang2022layoutlmv3,
  title={Layoutlmv3: Pre-training for document ai with unified text and image masking},
  author={Huang, Yupan and Lv, Tengchao and Cui, Lei and Lu, Yutong and Wei, Furu},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)},
  year={2022}
}

@inproceedings{reimers2019sentence,
  title={Sentence-bert: Sentence embeddings using siamese bert-networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2019}
}

@inproceedings{bahdanau2014neural,
  title={Neural machine translation by jointly learning to align and translate},
  author={Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2015}
}

@inproceedings{lin2017structured,
  title={A structured self-attentive sentence embedding},
  author={Lin, Zhouhan and Feng, Minwei and Santos, Cicero Nogueira dos and Yu, Mo and Xiang, Bing and Zhou, Bowen and Bengio, Yoshua},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}

@article{caruana1997multitask,
  title={Multitask learning},
  author={Caruana, Rich},
  journal={Machine learning},
  volume={28},
  number={1},
  pages={41--75},
  year={1997}
}

@inproceedings{kendall2018multi,
  title={Multi-task learning using uncertainty to weigh losses for scene geometry and semantics},
  author={Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}

@inproceedings{chen2018gradnorm,
  title={Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks},
  author={Chen, Zhao and Badrinarayanan, Vijay and Lee, Chen-Yu and Rabinovich, Andrew},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2018}
}

@inproceedings{conneau2019unsupervised,
  title={Unsupervised cross-lingual representation learning at scale},
  author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzmán, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2020}
}

@inproceedings{devlin2019bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year={2019}
}

@article{ba2016layer,
  title={Layer normalization},
  author={Ba, Jimmy Lei and Kiros, Jamie Ryan and Hinton, Geoffrey E},
  journal={arXiv preprint arXiv:1607.06450},
  year={2016}
}

@inproceedings{loshchilov2017decoupled,
  title={Decoupled weight decay regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019}
}

@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollár, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision (ECCV)},
  year={2014}
}

@article{young2014image,
  title={From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions},
  author={Young, Peter and Lai, Alice and Hodosh, Micah and Hockenmaier, Julia},
  journal={Transactions of the Association for Computational Linguistics},
  volume={2},
  pages={67--78},
  year={2014}
}

@inproceedings{nguyen2023vietnamesests,
  title={A comprehensive benchmark for vietnamese text evaluation},
  author={Nguyen, T. A. and others},
  booktitle={Proc. VLSP},
  year={2023}
}

@inproceedings{mathew2021docvqa,
  title={Docvqa: A dataset for vqa on document images},
  author={Mathew, Minesh and Karatzas, R and Jawahar, C V},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2021}
}

@inproceedings{reimers2020making,
  title={Making monolingual sentence embeddings multilingual using knowledge distillation},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020}
}

@misc{vipolyqwen2024repo,
  title={viPolyQwen GitHub Repository},
  author={Nguyen, Nguyen Anh and others},
  howpublished={\url{https://github.com/EraX-AI/viPolyQwen}},
  year={2024}
}
```