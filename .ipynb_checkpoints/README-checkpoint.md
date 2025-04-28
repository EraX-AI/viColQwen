# viColQwen: High-Performance Multimodal Embeddings for Cross-Domain Understanding

**[Model Release Pending - Stay Tuned!]**

## Abstract

We introduce **viColQwen**, a powerful multimodal embedding model designed to generate unified, high-dimensional representations for images, texts, and their combinations. Built upon the robust **Qwen2-VL 2B** architecture, viColQwen leverages state-of-the-art contrastive learning techniques, trained on a massive and exceptionally diverse dataset comprising 11 million samples. This dataset uniquely merges text-text similarity pairs, complex instructional data, multi-image OCR, and multi-image VQA tasks, primarily in Vietnamese with significant English and Chinese subsets. The resulting 1024-dimensional embeddings capture fine-grained semantic and visual nuances, making viColQwen particularly well-suited for demanding downstream applications such as multimodal Retrieval-Augmented Generation (RAG), Graph RAG, cross-modal search, and complex document understanding.

---

## Model Details

*   **Base Architecture:** `Qwen2-VL 2B` - A powerful foundation model known for its strong vision-language capabilities.
*   **Core Technique:** Contrastive Learning (inspired by ColPali). The model learns to map related multimodal inputs closer together in the embedding space while pushing dissimilar inputs apart.
*   **Embedding Dimension:** `1024` - A large projection dimension is employed to capture rich, detailed information from both visual and textual modalities.
*   **Output:** A unified embedding vector representing the semantic content of single or multiple images, single or multiple texts, or interleaved image-text inputs.

## Training Paradigm

viColQwen's robustness stems from its sophisticated training strategy:

1.  **Heterogeneous Data Integration:** The model is trained on a carefully curated mixture of four distinct data types, forcing it to learn diverse aspects of vision-language correlation:
    *   **Text-Text Semantic Similarity:** Pairs of texts with continuous similarity scores (0.0 to 1.0), combining hard-negative, hard-negative, multi-lingual to teaching nuanced semantic understanding. (5.6M samples)
    *   **Instruction Following:** Typical Large Language Model (LLM) instructions (single and multi-turn), enhancing contextual understanding and responsiveness. (0.6M samples)
    *   **Multi-Image OCR:** Single-turn Optical Character Recognition tasks involving 1 to 5 images, grounding textual understanding in visual text. (2.5M sampples)
    *   **Multi-Image VQA:** Single and multi-turn Visual Question Answering tasks with 1 to 5 images, fostering deep visual reasoning capabilities. (2.5M samples)
2.  **Mixed Loss Optimization:** Different data types are processed distinctly within the training loop, utilizing a combination of losses tailored to the specific task (e.g., similarity regression, instruction prediction, VQA accuracy) alongside the core contrastive objective.
3.  **Contrastive Objective:** Employs InfoNCE loss (or potentially Adaptive NCE) to effectively learn discriminative representations across modalities.
4.  **Scale:** Trained on a massive dataset of **11 million samples**, ensuring generalization and robustness.
5.  **Multilinguality:** Primarily trained on Vietnamese data, with substantial inclusion of English and Chinese examples, enabling cross-lingual transfer capabilities.

## Key Features & Capabilities

*   **Unified Multimodal Embedding:** Generates a single vector representation for images, texts, or combinations thereof.
*   **High-Dimensional Nuance:** 1024-d embeddings capture fine-grained details.
*   **Data Diversity Driven Robustness:** Training on text similarity, instructions, OCR, and VQA leads to versatile embeddings.
*   **Multi-Image Aware:** Inherently understands contexts involving multiple images (up to 5 tested).
*   **Strong Vietnamese Performance:** Optimized for Vietnamese while retaining significant multilingual capabilities (EN, ZH).
*   **Foundation for Advanced AI:** Ideal for building next-generation multimodal RAG, Graph RAG, search, and analysis systems.

## How to Use (Preliminary Example)

*(Note: The `ColPaLiEvaluator` class name is used here based on the provided snippet; the final class name might differ upon release.)*

**1. Setup & Initialization:**

```python
# Ensure necessary libraries are installed
# pip install transformers torch Pillow accelerate bitsandbytes # (Example dependencies)

import torch
from PIL import Image
import os
# Assuming the evaluator class is provided in the final package
from colpali_evaluator import ColPaLiEvaluator # Replace with actual import path

# Initialize the evaluator
# Ensure checkpoint path points to the downloaded/local model directory
CHECKPOINT_DIR = "./path/to/viColQwen/checkpoints/" # CHANGE THIS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 1024

evaluator = ColPaLiEvaluator(
    checkpoint_path=CHECKPOINT_DIR,
    embed_dim=EMBED_DIM,
    device=DEVICE
)
```

**2. Generating Embeddings for Images OR Texts:**

```python
# Example Images
img1 = Image.open("path/to/your/image1.jpg").convert("RGB")
img2 = Image.open("path/to/your/image2.png").convert("RGB")
images = [img1, img2]

# Example Text Queries
queries = [
    "Mô tả cấu trúc tổ chức của bộ phận R&D.",
    "Cung cấp bảng phân tích hiệu quả tài chính năm ngoái?"
]

# Get embeddings (ensure inputs are lists)
image_embeddings = evaluator.get_image_embeddings(images)
query_embeddings = evaluator.get_query_embeddings(queries)

print("Image Embeddings Shape:", image_embeddings.shape) # e.g., torch.Size([2, 1024])
print("Query Embeddings Shape:", query_embeddings.shape) # e.g., torch.Size([2, 1024])

# Calculate similarity scores (e.g., cosine similarity)
# evaluator.score likely computes dot products or cosine similarities
similarity_scores = evaluator.score(query_embeddings, image_embeddings)
print("Similarity Scores:\n", similarity_scores)
# Example output: tensor([[score_q1_img1, score_q1_img2],
#                         [score_q2_img1, score_q2_img2]])
```

**3. Generating Embeddings for Mixed Image(s) + Text(s):**

*(The following demonstrates the principle using methods likely available within the evaluator or model class)*

```python
# Example Mixed Input
text_input = "Dựa vào hình ảnh này, hãy tóm tắt các điểm chính."
image_input = Image.open("path/to/relevant/image.jpg").convert("RGB")
# or image_input = "path/to/relevant/image.jpg" # If path handling is supported

# The evaluator should provide a method to handle this directly.
# Assuming a method like 'get_multimodal_embedding' exists:
# (This mirrors the structure provided in the initial request)

# Option A: Using a potential high-level method (Hypothetical)
# mixed_embedding = evaluator.get_multimodal_embedding(text=text_input, image=image_input)

# Option B: Illustrating the internal logic (as provided in the prompt)
# These functions might be internal or exposed for advanced use.

def process_multimodal_input(evaluator_instance, text, image, image_base_path=""):
    """Process a combined text+image input for embedding (Illustrative)"""
    if isinstance(image, str):
        img_path = os.path.join(image_base_path, image)
        pil_image = Image.open(img_path).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Ensure text contains image tag (model specific requirement)
    if "<image>" not in text:
        # Prepend or append based on model's expected format
        text = f"<image>\n{text}"

    # Use the model's processor
    inputs = evaluator_instance.processor(
        text=[text], images=[pil_image], padding="longest", return_tensors="pt"
    )

    # Add necessary flags/keys expected by the specific model architecture
    # This part is highly model-dependent
    inputs["has_image_a"] = torch.tensor([True]) # Example flag
    if "pixel_values" in inputs:
        inputs["pixel_values_a"] = inputs.pop("pixel_values")
    if "input_ids" in inputs:
        inputs["input_ids_a"] = inputs.pop("input_ids")
    if "attention_mask" in inputs:
        inputs["attention_mask_a"] = inputs.pop("attention_mask")

    return inputs

def get_multimodal_embedding(evaluator_instance, text, image, image_base_path=""):
    """Get embedding for a text+image combination (Illustrative)"""
    inputs = process_multimodal_input(evaluator_instance, text, image, image_base_path)
    # Move inputs to the correct device
    inputs = {k: v.to(evaluator_instance.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        # The model forward pass returns the embedding
        # The exact output key ('embedding', 'pooler_output', etc.) depends on the model implementation
        outputs = evaluator_instance.model(**inputs)
        # Assuming the model directly outputs the embedding or it's under a specific key
        embedding = outputs # Or outputs.embedding, outputs.pooler_output, etc.

    return embedding.cpu() # Return embedding on CPU by default

# --- Using the illustrative functions ---
mixed_embedding = get_multimodal_embedding(evaluator, text_input, image_input)
print("Mixed Modality Embedding Shape:", mixed_embedding.shape) # e.g., torch.Size([1, 1024])

# You can then use this 'mixed_embedding' for downstream tasks like similarity search
# against other embeddings (image, text, or mixed).
```

## Potential Applications

The high-quality, unified embeddings generated by viColQwen unlock sophisticated multimodal applications:

*   **Multimodal RAG:** Retrieve relevant image(s) and/or text passages to augment LLM responses based on complex multimodal queries.
*   **Graph RAG:** Construct and query knowledge graphs where nodes represent images, text chunks, or entities, connected by learned relationships derived from viColQwen embeddings.
*   **Cross-Modal Search:** Search for images using text queries, search for text using image queries, or search for image-text pairs.
*   **Image/Document Clustering & Classification:** Group or categorize images and documents based on their combined visual and textual content.
*   **Visual Similarity Search:** Find visually similar images, potentially guided by textual context.
*   **OCR-Aware Document Analysis:** Understand documents by leveraging both the visual layout/images and the recognized text within a single embedding space.

## Development Status & Future Work

*   This repository is under active development and will be updated regularly with model checkpoints, refined code, and usage examples.
*   Further benchmarking on Vietnamese and cross-lingual multimodal tasks is planned.
*   Exploration of model variants and potential scaling is ongoing.

**Stay tuned for the official model release!** ✨🚀

## License

*(To be determined - Likely Apache 2.0 or similar permissive license upon release)*

## Citation

*(Placeholder - Please cite this repository URL for now. A paper/preprint citation will be added if available)*

```bibtex
@misc{vicolqwen_github,
  author       = {Steve Nguyen Anh Nguyen},
  title        = {viColQwen: High-Performance Multimodal Embeddings for Cross-Domain Understanding},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/EraX-AI/viColQwen}}
}
```

---