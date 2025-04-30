  <img src="https://huggingface.co/erax-ai/EraX-Translator-V1.0/resolve/main/erax-gmobile.png?download=true" alt="Logo" width="400">
<p align="left">
</p>

# viPolyQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization with Attention Pooling

[[Tiếng Việt](README.md)] | **English**

**(Model Release Pending - Stay Tuned!)**

## Abstract

Modern multimodal systems often face challenges due to the complexity of managing separate embedding spaces for diverse data types (e.g., text, images), leading to representational fragmentation and suboptimal cross-modal reasoning. We introduce **viPolyQwen**, an advanced multimodal embedding model generating **high-dimensional, unified representations** for images, text, and their combinations within a single vector space. Its full name reflects its core approach: **Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization**, built on **Qwen** architecture.

This research, including the development and training of the viPolyQwen model, was conducted with close collaboration from **the AI technology team at Gtel Mobile JSC (GMobile)**. Their technical expertise and collaborative support were crucial throughout the research process and model training.

Built upon the powerful **Qwen2-VL 2B-Instruct** architecture, viPolyQwen employs a sophisticated contrastive learning framework trained on a **large-scale, exceptionally diverse dataset (>11M samples)**. This dataset integrates text-similarity pairs, instruction-following data, and extensive multi-image OCR and VQA scenarios (including documents, charts, handwriting, and specialized medical images).

The core algorithmic innovation lies in its **prefix-guided dynamic mixed-loss optimization strategy**. Task-specific prefixes (`<text_pair>`, `<instr>`, `<ocr>`, `<vqa_multi>`, `<vqa_single>`) guide the model by signaling the data type, dynamically triggering a **tailored loss function** (InfoNCE, Triplet, MSE, Cosine Similarity) for each sample.

Crucially, instead of standard mean or last-token pooling, final 1D embeddings are extracted using **Attention Pooling**. This mechanism allows the model to **dynamically focus on the most salient visual and textual features** within the encoder's output tokens before projection. By learning to weight important features (like text regions within an image or key semantic concepts) higher, attention pooling aims to create richer, more nuanced 1D embeddings compared to simple averaging, significantly enhancing the model's ability to capture semantic content, even from images containing text.

The resulting 1024-dimensional embeddings facilitate robust downstream applications like multimodal RAG, Graph RAG, cross-modal search, and document analysis. While optimized for **Vietnamese**, its multilingual training data enables effective zero-shot capabilities.

---

## Model Details

*   **Base Architecture:** `Qwen/Qwen2-VL-2B-Instruct` - The foundational Vision-Language Model (VLM).
*   **Embedding Strategy:** Unified Embedding Space via Prefix-Guided Dynamic Contrastive Learning with **Attention Pooling**.
*   **Embedding Dimension:** `1024`.
*   **Pooling Strategy:** **Attention Pooling.** This is a key differentiator. Instead of simple averaging (mean pooling) or selecting the last token, viPolyQwen employs a *learned attention mechanism* over the final hidden states sequence (representing both text tokens and image patches).
    *   It calculates attention scores based on the relevance of each hidden state to the overall context.
    *   It assigns higher weights to more informative states (e.g., specific text regions in an image, key visual objects, important semantic tokens).
    *   It computes a *weighted average* based on these attention weights.
    *   **Benefit:** This allows the model to create a more contextually relevant and nuanced 1D representation by focusing on salient features, significantly improving the capture of core semantic and visual essence compared to uniform averaging. This is particularly beneficial for representing images containing text or complex visual structures like charts and tables in a single vector.
*   **Input Representation:** Input data (text strings, PIL Images) is processed by the Qwen-VL processor. Images are represented by the `<image>` token. Crucially, a **task-specific prefix** is prepended to the main textual input during *training* to signal the data type and guide the loss calculation:
    *   `<text_pair>`: For text similarity pairs.
    *   `<instr>`: For instruction-following data.
    *   `<ocr>`: For OCR/OCQ data.
    *   `<vqa_multi>`: For multi-turn VQA.
    *   `<vqa_single>`: For single-turn VQA.
    *(Note: For general inference/embedding, prefixes are typically omitted unless querying a specific task like OCR/VQA - see Usage Guide)*.
*   **Output:** A single `1024-d` dense, L2-normalized vector representing the input.

---

## Training Paradigm

viPolyQwen's robustness stems from its unique optimization strategy and diverse training data:

1.  **Heterogeneous and Rich Dataset (>11M Samples):** (Description of the diverse dataset components - text-similarity, instructions, OCR, VQA, medical - remains the same as previous version).
    *   **Language Distribution:** Predominantly **Vietnamese**, with substantial **English** and **Chinese** samples, fostering strong zero-shot generalization.

2.  **Prefix-Guided Dynamic Mixed-Loss Optimization:**
    *   During training, each sample's prefix signals the appropriate loss function.
    *   **Loss Function Suite Applied:**
        *   `<text_pair>`: Symmetric InfoNCE + MSE Similarity Regression.
        *   `<instr>`: Symmetric InfoNCE + Direct Cosine Similarity Maximization.
        *   `<ocr>`, `<vqa_single>`, `<vqa_multi>`: Symmetric InfoNCE + Triplet Margin Loss.
    *   The final 1D embeddings used for these loss calculations are generated via **Attention Pooling** applied to the encoder's output sequence.

This combination allows viPolyQwen to learn a highly capable unified embedding space applicable across diverse real-world scenarios.

## Training details

The training of the `viPolyQwen` model involved a significant computational effort.

*   **Hardware:** Trained on a cluster with **4x NVIDIA H100 GPUs (94GB VRAM, NVLink)** via Vast.AI.
*   **Duration:** Approx. **15 days** of continuous computation.
*   **Framework:** Distributed training via Hugging Face `accelerate` using FSDP (likely ZeRO-3).
*   **Precision & Optimizations:** **`bfloat16`** mixed precision; **Flash Attention 2**.
*   **Key Hyperparameters:**
    *   Tokenizer/Embeddings: Extended Qwen2VL tokenizer/embedding layer for new special tokens.
    *   **Base Model:** `Qwen/Qwen2-VL-2B-Instruct`
    *   **Optimizer:** AdamW
    *   **Learning Rate:** 1e-4 (cosine decay after 5% warmup)
    *   **Epochs:** 2
    *   **Batch Size (per device):** 24
    *   **Gradient Accumulation:** 8 (Effective Global Batch Size: 768)
    *   **Max Sequence Length:** 8192
    *   **Weight Decay:** 0.001
    *   **Max Grad Norm:** 1.0
    *   **Pooling Strategy:** **Attention Pooling** *(During training, loss calculated on attention-pooled embeddings)*
    *   **Loss Hyperparameters:** Temperature = 0.07, Contrastive Margin = 0.2
*   **Dataset:** 11M+ training samples, 5k evaluation samples.

---

## Key Features & Advantages

*   ✅ **Unified Multimodal Embedding:** Single vector space simplifies integration.
*   ✅ **Prefix-Guided Training:** Enables task-aware learning during training.
*   ✅ **Attention Pooling:** Creates richer, more nuanced 1D embeddings by focusing on salient visual/textual features, **enhancing capture of semantic details (including text-in-image concepts)** compared to mean pooling.
*   ✅ **Exceptional Data Diversity:** Robustness from training on similarity, instructions, complex OCR, and deep VQA (incl. medical).
*   ✅ **Simplified Multimodal RAG/Search:** Efficient retrieval from a single index.
*   ✅ **Enhanced Cross-Modal Understanding:** Joint training fosters deep correlations.
*   ✅ **High-Dimensional Nuance:** 1024-d captures fine-grained details.
*   ✅ **Multi-Image Aware:** Natively processes multiple input images.
*   ✅ **Strong Vietnamese & Zero-Shot Capabilities:** Optimized for Vietnamese with cross-lingual potential.
*   ✅ **Foundation for Advanced AI:** Ideal for next-gen multimodal systems.

---

## How to Use: [Usage Guide & Examples](USAGE.md)

*(The usage guide will explain strategy for using/omitting prefixes during inference as discussed previously: embed general data without prefix, use prefix only for specific OCR/VQA queries if desired).*

---

## Potential Applications

*   **Multimodal RAG:** Retrieve highly relevant text passages, images, tables, or document sections (including medical reports or financial statements) using unified queries.
*   **Graph RAG:** Build knowledge graphs where nodes represent diverse entities (patients, documents, procedures, visual findings) linked via unified embeddings.
*   **Cross-Modal Retrieval:** Efficiently search for medical images based on textual descriptions, find relevant documents from images of forms, etc.
*   **Document Intelligence:** Deep analysis of complex documents like insurance claims, scientific papers, or technical manuals, leveraging both visual layout and content.
*   **Contextual Visual Search:** Find visually similar images (e.g., medical scans, product photos) refined by specific textual context.

---

## Development Status & Future Work

*   Actively under development. Model checkpoints, evaluation code, benchmarks, and comprehensive usage examples will be released soon.
*   Ongoing work includes extensive benchmarking across Vietnamese, English, and cross-lingual tasks, ablation studies on data components, exploring larger base models, and potential integration of further modalities.

---

## License

*   Licensing details will be announced upon release.
*   A commercial license option will be available. For inquiries regarding commercial use, please contact: **nguyen@hatto.com**.

---

## Citation

Please cite this repository URL until a formal publication is available.

```bibtex
@misc{viPolyQwen_github_2024,
  author       = {Steve Nguyen Anh Nguyen Erax GMobile},
  title        = {viPolyQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/EraX-AI/viPolyQwen}}
}

@misc{faysse2024colpali,
      title={ColPali: Efficient Document Retrieval with Vision Language Models},
      author={Manuel Faysse and Hugues Sibille and Tony Wu and Bilel Omrani and Gautier Viaud and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2407.01449},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.01449}
}

@misc{bai2023qwen,
      title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
      author={Jinze Bai and Shuai Bai and Shusheng Yang and Shijie Wang and Sinan Tan and Peng Wang and Junyang Lin and Chang Zhou and Jingren Zhou},
      year={2023},
      eprint={2308.12966},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
