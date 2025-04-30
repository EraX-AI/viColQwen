<p align="left">
  <img src="https://huggingface.co/erax-ai/EraX-Translator-V1.0/resolve/main/erax-gmobile.png?download=true" alt="Logo">
</p>

# viPyloQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization

[[Tiếng Việt](README.md)] | **English**

**(Model Release Pending - Stay Tuned!)**

## Abstract

Modern multimodal systems often face challenges due to the complexity of managing separate embedding spaces for diverse data types (e.g., text, images). This can lead to representational fragmentation, cumbersome retrieval pipelines, and limitations in cross-modal reasoning. 

We introduce **viPyloQwen**, an advanced multimodal embedding model designed to generate **high-dimensional, unified representations** for images, text, and their arbitrary combinations within a single, cohesive vector space. We called it Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization.

This research, including the development and training of the viPyloQwen model, was conducted with close collaboration from **the AI technology team at Gtel Mobile JSC (GMobile)**. Their technical expertise and collaborative support were crucial throughout the research process and model training.

Built upon the powerful **Qwen2-VL 2B Base** vision-language architecture, viPyloQwen employs a sophisticated contrastive learning framework. While inspired by approaches like ColPali, viPyloQwen introduces significant enhancements, particularly through its unique training methodology. The model is trained on a **large-scale, exceptionally diverse dataset exceeding 11 million samples**. This meticulously curated dataset strategically integrates challenging text-text semantic similarity pairs (with continuous scores), complex instruction-following data, and perhaps most distinctively, a vast collection of multi-image Optical Character Recognition (OCR) and Visual Question Answering (VQA) scenarios.

The core algorithmic innovation lies in viPyloQwen's **prefix-guided dynamic mixed-loss optimization strategy**. Task-specific prefixes (`<text_pair>`, `<instr>`, `<ocr>`, `<vqa_multi>`, `<vqa_single>`) are prepended to the input, serving as cues to signal the data type. This mechanism **dynamically triggers a corresponding, tailored loss function** (including InfoNCE, Triplet Loss, MSE, and direct cosine similarity maximization) specifically designed for each sample type.

Final embeddings are extracted using **mean pooling** over the encoder's output tokens, ensuring comprehensive capture of semantic and visual information. The resulting 1024-dimensional embeddings, derived from this rich data mixture and unique training strategy, exhibit nuanced semantic and visual understanding. This significantly simplifies and enhances downstream applications such as multimodal Retrieval-Augmented Generation (RAG), Graph RAG, cross-modal search, and complex document analysis. While demonstrating particularly strong performance in **Vietnamese** due to data focus, the model's multilingual training data (including substantial English and Chinese) facilitates effective zero-shot transfer capabilities to other languages.

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

viPyloQwen's robustness and versatility stem from the synergistic combination of its unique optimization strategy and its exceptionally diverse training data:

1.  **Heterogeneous and Rich Dataset (Over 11 Million Samples):** The training corpus integrates multiple data modalities and task types, linked via the input prefixes:
    *   **Text-Text Semantic Similarity (`<text_pair>`, ~5.6M):** Pairs $(t_a, t_b)$ with similarity scores $s \in [0, 1]$, fostering nuanced textual understanding.
    *   **Instruction Following (`<instr>`, ~0.6M):** Pairs (single and multi-turns instruction $i$, response $r$), enhancing contextual reasoning and task execution representation.
    *   **Diverse Multi-Image OCR/OCQ (`<ocr>`, ~2.5M):** This category goes far beyond simple document text. It includes a wide spectrum of visual text recognition tasks on 1-5 images per sample, such as:
        *   Street scene captioning and text recognition.
        *   Mathematical document understanding (formulas, diagrams).
        *   Text and image interplay in general documents.
        *   Chart and diagram analysis.
        *   Handwriting recognition (e.g., invoices, insurance claims forms, accident reports).
        *   Recognition of common Vietnamese documents (e.g., National ID cards - CCCD, driver's licenses).
    *   **Complex Multi-Image VQA (`<vqa_single>`, `<vqa_multi>`, ~2.5M):** These tasks, single and multi-turns VQA, also using 1-5 images, demand deeper visual reasoning integrated with textual queries. The data spans:
        *   General visual question answering across various scenes.
        *   Complex table and chart interpretation requiring reasoning.
        *   **Specialized Medical Imaging Analysis (~0.5M samples):** A significant subset dedicated to radiology OCR and VQA. This involves analyzing diverse medical scans (dermatology images, X-rays, CT, MRI) for diagnostic question answering related to critical health areas including skin, bone, heart, lung, brain, and dental conditions.
    *   **Language Distribution:** While the dataset predominantly features **Vietnamese** content to ensure strong performance in this context, it strategically incorporates substantial **English** and **Chinese** samples across all categories. This multilingual foundation is crucial for enabling the model's effective **zero-shot generalization** to other unseen languages.

2.  **Prefix-Guided Dynamic Mixed-Loss Optimization:**
    *   As described previously, each sample's prefix dynamically selects a tailored loss function from a pre-defined suite.
    *   **Loss Function Suite Applied:**
        *   `<text_pair>`: Symmetric InfoNCE + MSE Similarity Regression.
        *   `<instr>`: Symmetric InfoNCE + Direct Cosine Similarity Maximization.
        *   `<ocr>`, `<vqa_single>`, `<vqa_multi>`: Symmetric InfoNCE + Triplet Margin Loss (margin potentially adjusted for multi-turn).

This combination of a rich, domain-diverse dataset and an adaptive training mechanism allows viPyloQwen to develop a truly unified and highly capable embedding space applicable across a wide range of real-world scenarios.

## Trainign details

The training of the `viPyloQwen` model involved a significant computational effort, underscoring the complexity of learning from such a large and diverse multimodal dataset.

*   **Hardware:** The model was trained on a high-performance computing cluster equipped with **4x NVIDIA H100 GPUs on Vast.AI**, each with 94GB of VRAM connected via NVLink.
*   **Duration:** The primary training phase spanned approximately **15 days** of continuous computation on this hardware setup.
*   **Framework:** Distributed training was orchestrated using the **Hugging Face `accelerate` library**, leveraging its capabilities for efficient multi-GPU scaling (likely configured with DeepSpeed ZeRO stage 3 or FSDP, as specified in the `qwen2VL2B.yaml` configuration file).
*   **Precision & Optimizations:** Training utilized **`bfloat16` mixed precision** to optimize memory usage and computational throughput. **Flash Attention 2** was enabled for further efficiency gains in the attention mechanism.
*   **Key Hyperparameters:**
    *   **Base Model:** `Qwen/Qwen2-VL-2B-Instruct`
    *   **Optimizer:** AdamW (standard with Hugging Face Trainer)
    *   **Learning Rate:** 1e-4 (with linear decay and 5% warmup)
    *   **Epochs:** 2
    *   **Batch Size (per device):** 24
    *   **Gradient Accumulation Steps:** 8
    *   **Effective Global Batch Size:** 768 (24 * 4 GPUs * 8 accumulation)
    *   **Max Sequence Length:** 8192 tokens
    *   **Weight Decay:** 0.001
    *   **Max Gradient Norm:** 1.0
    *   **Pooling Strategy:** Mean Pooling
    *   **Loss Hyperparameters:** Temperature = 0.07, Contrastive Margin = 0.2
*   **Dataset:** Trained on the described 11M+ sample dataset (`TRAIN_11M.jsonl`) and evaluated using a 5k sample split (`EVAL_5k.jsonl`).

This setup highlights the substantial resources required to train state-of-the-art multimodal embedding models capable of handling diverse, real-world data effectively.

---

## Key Features & Advantages

*   ✅ **Unified Multimodal Embedding:** A single, coherent vector space simplifies integration and downstream tasks.
*   ✅ **Prefix-Guided Training:** Enables nuanced, task-aware learning within the unified space.
*   ✅ **Exceptional Data Diversity:** Training on text similarity, instructions, complex OCR (handwriting, forms, diagrams, medical), and deep VQA (reasoning, charts, specialized radiology) ensures robustness and broad applicability.
*   ✅ **Simplified Multimodal RAG/Search:** Allows querying a single index with text, image, or mixed queries to retrieve relevant multimodal information.
*   ✅ **Enhanced Cross-Modal Understanding:** Joint training fosters embeddings sensitive to fine-grained visual-textual correlations.
*   ✅ **High-Dimensional Nuance:** 1024-d captures detailed information crucial for complex tasks.
*   ✅ **Multi-Image Aware:** Natively processes inputs containing multiple images.
*   ✅ **Strong Vietnamese & Zero-Shot Capabilities:** Optimized for Vietnamese with proven cross-lingual generalization potential due to multilingual data inclusion.
*   ✅ **Foundation for Advanced AI:** An ideal building block for sophisticated multimodal RAG, Graph RAG, semantic search, classification, and analysis systems.

---

## How to Use (Conceptual Example)

```python
import torch
from PIL import Image
# Giả sử bạn đã load model và processor vào biến `embedder` và `processor`
# embedder = ColPaLiQwenEmbedder.from_pretrained("./path/to/your/finetuned_model")
# processor = AutoProcessor.from_pretrained("./path/to/your/finetuned_model", trust_remote_code=True) # Hoặc từ model gốc
# embedder.processor = processor # Gán processor cho embedder nếu load riêng
# embedder.to("cuda") # Chuyển model lên GPU

# --- Ví dụ: VQA Single Turn (e.g., Medical Image) ---
prefix_vqa = "<vqa_single>"
text_input_vqa = "Is there evidence of fracture in the distal radius?" # Example query
image_input_vqa = Image.open("wrist_xray.png").convert("RGB") # Example image

# Gọi phương thức encode mới
# Quan trọng: Đảm bảo text có chứa prefix
mixed_embedding_vqa = embedder.encode(
    text=f"{prefix_vqa} {text_input_vqa}",
    images=image_input_vqa
)
print("VQA Embedding Shape:", mixed_embedding_vqa.shape) # Expected: torch.Size([1, 1024])

# --- Ví dụ: Text Similarity ---
prefix_sim = "<text_pair>"
text_a = "Patient reported mild discomfort."
text_b = "Subject experienced slight pain."

# Mã hóa từng câu riêng lẻ (vì chúng là 2 thực thể riêng biệt trong cặp)
text_a_embedding = embedder.encode(text=f"{prefix_sim} {text_a}")
text_b_embedding = embedder.encode(text=f"{prefix_sim} {text_b}")

# Tính độ tương đồng
similarity = torch.nn.functional.cosine_similarity(text_a_embedding, text_b_embedding)
print("Text Similarity:", similarity.item())
print("Text A Embedding Shape:", text_a_embedding.shape) # Expected: torch.Size([1, 1024])

# --- Ví dụ: OCR (e.g., Handwritten Form) ---
prefix_ocr = "<ocr>"
text_input_ocr = "What is the policy number listed?" # Example query
image_input_ocr = Image.open("handwritten_claim_form.jpg").convert("RGB") # Example image

form_embedding_ocr = embedder.encode(
    text=f"{prefix_ocr} {text_input_ocr}",
    images=image_input_ocr
)
print("OCR Embedding Shape:", form_embedding_ocr.shape) # Expected: torch.Size([1, 1024])

# --- Ví dụ: Mã hóa nhiều mẫu cùng lúc (batch) ---
batch_texts = [
    f"<vqa_single> What is shown?",
    f"<ocr> Read the title",
    f"<text_pair> First sentence.",
    f"<text_pair> Second sentence, similar to first."
]
batch_images = [
    Image.open("image1.jpg").convert("RGB"),
    Image.open("document_page.png").convert("RGB"),
    None, # text_pair không cần ảnh
    None  # text_pair không cần ảnh
]

batch_embeddings = embedder.encode(text=batch_texts, images=batch_images, batch_size=2) # Ví dụ batch_size=2
print("Batch Embedding Shape:", batch_embeddings.shape) # Expected: torch.Size([4, 1024])
```

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
@misc{viPyloQwen_github_2024,
  author       = {Steve Nguyen Anh Nguyen Erax GMobile},
  title        = {viPyloQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/EraX-AI/viPyloQwen}}
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