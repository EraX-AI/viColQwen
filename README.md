# viOmniQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization

**(Model Release Pending - Stay Tuned!)**

## Abstract

Modern multimodal systems often struggle with the complexity of managing separate embedding spaces for different data types (e.g., text, images), leading to representational fragmentation, intricate retrieval pipelines, and suboptimal cross-modal reasoning. We introduce **viOmniQwen**, an advanced multimodal embedding model engineered to generate **high-dimensional, unified representations** for images, texts, and their arbitrary combinations within a single vector space. Built upon the powerful **Qwen2-VL 2B** vision-language architecture, viOmniQwen employs a sophisticated contrastive learning paradigm, inspired by ColPali but significantly enhanced. The model is trained on a **large-scale, heterogeneous dataset exceeding 11 million samples**, strategically integrating challenging text-text semantic similarity pairs (with continuous scores), complex instruction-following data, multi-image Optical Character Recognition (OCR) tasks, and multi-image Visual Question Answering (VQA) scenarios. The core innovation lies in its **prefix-guided dynamic mixed-loss optimization strategy**. Task-specific prefixes (`<text_pair>`, `<instr>`, `<ocr>`, `<vqa_multi>`, `<vqa_single>`) are prepended to the input, signaling the data type and **dynamically triggering a corresponding, tailored loss function** (including InfoNCE, Triplet Loss, MSE, and direct cosine similarity maximization) for each sample. Final embeddings are extracted using **mean pooling** over the encoder's output tokens, capturing comprehensive semantic and visual information. The resulting 1024-dimensional embeddings exhibit nuanced semantic and visual understanding, significantly simplifying and enhancing downstream applications such as multimodal Retrieval-Augmented Generation (RAG), Graph RAG, cross-modal search, and complex document analysis, particularly within the Vietnamese language context, although it can certainly zero-shot for major languages likes English, Chinese and the likes. 

---

## Model Details

*   **Base Architecture:** `Qwen/Qwen2-VL-2B` - The foundational Vision-Language Model (VLM).
*   **Embedding Strategy:** Unified Embedding Space via Prefix-Guided Dynamic Contrastive Learning.
*   **Embedding Dimension:** `1024`.
*   **Pooling Strategy:** **Mean Pooling**. The final embedding vector is obtained by averaging the hidden states of all output tokens from the final layer of the Qwen2-VL encoder, followed by L2 normalization. This aggregates information across the entire input sequence (text tokens and image patch tokens).
*   **Input Representation:** Input data (text strings, PIL Images) is processed by the Qwen-VL processor. Images are represented by the `<image>` token. Crucially, a **task-specific prefix** is prepended to the main textual input to signal the data type:
    *   `<text_pair>`: For text similarity pairs with continuous scores.
    *   `<instr>`: For instruction-following data (instruction-response pairs).
    *   `<ocr>`: For OCR/OCQ data (image(s)+query -> answer).
    *   `<vqa_multi>`: For multi-turn VQA (image(s)+question -> answer).
    *   `<vqa_single>`: For single-turn VQA (image(s)+question -> answer).
*   **Output:** A single `1024-d` dense vector representing the semantic and/or visual content of the input.

---

## Training Paradigm

viOmniQwen's robustness stems from its diverse data mixture and unique optimization strategy:

1.  **Heterogeneous Dataset (Over 11 Million Samples):** Integrates four primary data types linked to the prefixes above:
    *   **Text-Text Semantic Similarity (`<text_pair>`, ~5.6M):** Pairs $(t_a, t_b)$ with similarity scores $s \in [0, 1]$.
    *   **Instruction Following (`<instr>`, ~0.6M):** Pairs (instruction $i$, response $r$).
    *   **Multi-Image OCR/OCQ (`<ocr>`, ~2.5M):** Triples $(\{\text{image(s)}\}_q, \text{query } q, \text{answer } a)$.
    *   **Multi-Image VQA (`<vqa_single>`, `<vqa_multi>`, ~2.5M):** Triples $(\{\text{image(s)}\}_q, \text{question } q, \text{answer } a)$.
    The dataset has a primary focus on Vietnamese (vi), with substantial English (en) and Chinese (zh) coverage.

2.  **Prefix-Guided Dynamic Mixed-Loss Optimization:**
    *   Each sample in a batch is identified by its corresponding task prefix.
    *   Based on the detected prefix, a specific loss function ($\mathcal{L}_{\text{prefix}}$) from a pre-defined suite is **dynamically selected and applied** to the embedding pair $(e_a, e_b)$ computed for that sample.
    *   The total batch loss ($\mathcal{L}_{\text{batch}}$) is the average of these individually computed losses across all samples in the batch.
    *   **Loss Function Suite:**
        *   **For `<text_pair>`:** Combines Symmetric InfoNCE loss with Mean Squared Error (MSE) Similarity Regression (aligning predicted similarity with ground-truth scores).
        *   **For `<instr>`:** Combines Symmetric InfoNCE loss with Direct Cosine Similarity Maximization (encouraging high similarity between instruction and response embeddings).
        *   **For `<ocr>`, `<vqa_single>`, `<vqa_multi>`:** Combines Symmetric InfoNCE loss with Triplet Margin Loss (enforcing a margin between positive pairs and the hardest negative pairs within the batch, with potentially adjusted margin for multi-turn VQA).

This dynamic, prefix-guided approach allows the model to effectively learn from diverse data structures within a single unified embedding space.

---

## Key Features & Advantages

*   ✅ **Unified Multimodal Embedding:** Single vector space for text, image(s), and combinations.
*   ✅ **Prefix-Guided Training:** Enables specialized handling of different data types (similarity, instructions, OCR, VQA) via prefixes and tailored losses.
*   ✅ **Simplified Multimodal RAG/Search:** Streamlines querying a single vector index with diverse inputs.
*   ✅ **Enhanced Cross-Modal Understanding:** Joint training on diverse tasks fosters deep visual-textual correlations.
*   ✅ **High-Dimensional Nuance:** 1024-d embeddings capture fine-grained details.
*   ✅ **Multi-Image Aware:** Natively encodes context from multiple input images.
*   ✅ **Robust Performance:** Diverse training data and loss functions yield versatile embeddings.
*   ✅ **Strong Vietnamese & Multilingual Focus:** Optimized for Vietnamese with significant en/zh capabilities.
*   ✅ **Foundation for Advanced AI:** Ideal for next-generation multimodal systems.

---

## How to Use (Conceptual Example)

```python
import torch
from PIL import Image
# Assume viOmniQwenEmbedder class is available after release
# from viOmniQwen_embedder import viOmniQwenEmbedder

# embedder = viOmniQwenEmbedder(checkpoint_path="./path/to/viOmniQwen/", device="cuda")

# --- Example: VQA Single Turn ---
# Note: The embedder's encode method should handle prefix internally,
# or you might need to prepend it manually if using the base model directly.
prefix_vqa = "<vqa_single>"
text_input = "What color is the object on the left?"
image_input = Image.open("image.jpg").convert("RGB")

# Conceptual encoding call - the prefix guides the internal processing
# mixed_embedding = embedder.encode(text=f"{prefix_vqa} {text_input}", images=[image_input])
# print(mixed_embedding.shape) # Expected: torch.Size([1, 1024])

# --- Example: Text Similarity ---
prefix_sim = "<text_pair>"
text_a = "The cat sat on the mat."
text_b = "A feline rested upon the rug."

# Conceptual encoding calls
# text_a_embedding = embedder.encode(text=f"{prefix_sim} {text_a}")
# text_b_embedding = embedder.encode(text=f"{prefix_sim} {text_b}")

# Compute similarity (e.g., cosine)
# similarity = torch.nn.functional.cosine_similarity(text_a_embedding, text_b_embedding)
# print(similarity)
```

---

## Potential Applications

*   **Multimodal RAG:** Retrieve diverse multimodal context (text passages, images, tables within documents) using unified queries for richer LLM grounding.
*   **Graph RAG:** Construct knowledge graphs with nodes representing text, images, or multimodal documents, navigable via unified embeddings.
*   **Cross-Modal Retrieval:** Robustly find images from text queries, text from image queries, or similar multimodal items within a single index.
*   **Document Intelligence:** Analyze complex documents (e.g., reports, invoices) by capturing visual layout and textual content in one representation.
*   **Contextual Visual Search:** Enhance image search results by incorporating accompanying textual context during embedding.

---

## Development Status & Future Work

*   Under active development. Model checkpoints, evaluation code, benchmarks, and detailed usage examples will be released soon.
*   Ongoing work includes comprehensive benchmarking (Vietnamese, English, cross-lingual tasks), exploring larger base models, and potential integration of other modalities.

---

## License

*   Licensing details will be announced upon release.
*   A commercial license option will be available. For inquiries, please contact: **nguyen@hatto.com**.

---

## Citation

Please cite this repository URL until a formal publication is available.

```bibtex
@misc{viOmniQwen_github_2024,
  author       = {Steve Nguyen Anh Nguyen and the EraX AI Team},
  title        = {viOmniQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/EraX-AI/viOmniQwen}} % Final URL
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
```