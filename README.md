# viColQwen: High-Performance Unified Embeddings for Advanced Multimodal Understanding

**[Model Release Pending - Stay Tuned!]**

## Abstract

The landscape of Retrieval-Augmented Generation (RAG) and multimodal AI is often hampered by the complexity of managing separate embedding spaces for different modalities (e.g., text via SentenceTransformers, images via ViT). This necessitates multiple vector databases, intricate query strategies, and often suboptimal cross-modal retrieval. We introduce **viColQwen**, a state-of-the-art multimodal embedding model designed to overcome these limitations by generating **unified, high-dimensional representations** for images, texts, and their arbitrary combinations within a single vector space. Built upon the powerful **Qwen2-VL 2B** architecture and trained using contrastive learning inspired by ColPali, viColQwen leverages a massive, uniquely diverse dataset of over **11 million samples**. This dataset merges challenging text-text similarity pairs, complex instructions, multi-image OCR, and multi-image VQA tasks (primarily Vietnamese, with substantial English and Chinese data). The resulting 1024-dimensional embeddings capture fine-grained semantic and visual nuances, drastically simplifying and enhancing downstream applications like multimodal RAG, Graph RAG, cross-modal search, and complex document understanding, paving the way for more coherent and powerful multimodal AI systems.

---

## Model Details

*   **Base Architecture:** `Qwen2-VL 2B` - A robust foundation model renowned for its strong vision-language capabilities.
*   **Core Technique:** Contrastive Learning (ColPali-inspired). Learns a unified embedding space by mapping related multimodal inputs closer and pushing dissimilar inputs apart.
*   **Embedding Dimension:** `1024` - Captures rich, detailed information from both visual and textual modalities in a high-dimensional space.
*   **Output:** A **single** embedding vector representing the semantic content of one or more images, one or more texts, or interleaved image-text inputs.

## Training Paradigm: The Foundation of Robustness

viColQwen's strength lies in its sophisticated training strategy and diverse data mixture:

1.  **Heterogeneous Data Integration (Over 11 Million Samples):**
    *   **Text-Text Semantic Similarity (5.6M samples):** Pairs of texts with continuous similarity scores (0.0-1.0), specifically curated to include challenging hard-negative and hard-positive examples across multiple languages, teaching nuanced semantic distinction.
    *   **Instruction Following (0.6M samples):** Standard LLM instructions (single/multi-turn) enhance contextual understanding and task adaptability.
    *   **Multi-Image OCR (2.5M samples):** Single-turn OCR tasks involving 1-5 images ground textual understanding in visually presented text.
    *   **Multi-Image VQA (2.5M samples):** Single/multi-turn VQA tasks with 1-5 images foster deep visual reasoning and question-answering capabilities within context.
    *   OCR/VQA are **very diversed** as well, including captioning, radiology MRI/CT scan prediction, very complex multi layers json-generated extraction, tables, maths, charts, hand-writing and many turns complex VQA 
2.  **Mixed Loss Optimization:** Employs a combination of losses tailored to each data type (e.g., similarity regression, instruction prediction) alongside the core contrastive objective (InfoNCE or Adaptive NCE) for multifaceted learning.
3.  **Scale and Multilinguality:** Primarily trained on Vietnamese, with substantial English and Chinese data, enabling strong performance in Vietnamese and facilitating cross-lingual transfer.

## Key Features & Advantages

*   ✅ **Unified Multimodal Embedding:** A single vector represents images, text, or combinations, eliminating the need for separate models and vector stores.
*   ✅ **Simplified Multimodal RAG/Search:** Query with text, image, or both to retrieve relevant multimodal information from a single index, streamlining complex retrieval pipelines.
*   ✅ **Enhanced Cross-Modal Understanding:** Joint training fosters embeddings that capture deeper correlations between visual and textual concepts than separate models allow.
*   ✅ **High-Dimensional Nuance:** 1024-d embeddings capture fine-grained details crucial for complex tasks.
*   ✅ **Multi-Image Aware:** Natively handles contexts involving multiple images (up to 5 tested).
*   ✅ **Robust Performance:** Data diversity (similarity, instructions, OCR, VQA) leads to versatile and robust embeddings.
*   ✅ **Strong Vietnamese & Multilingual Capabilities:** Optimized for Vietnamese (vi) with significant English (en) and Chinese (zh) understanding.
*   ✅ **Foundation for Next-Gen AI:** Ideal for building advanced multimodal RAG, Graph RAG, semantic search, classification, and analysis systems.
*   ✅ **Extensible:** The core approach can potentially be extended to incorporate other modalities like video or audio in the future.

## How to Use (Preliminary Example)

*(Note: The `ColPaLiEvaluator` class and specific method names are based on provided snippets and may evolve in the final release.)*

**1. Setup & Initialization:**

```python
# Ensure necessary libraries are installed
# pip install transformers torch Pillow accelerate bitsandbytes # Example dependencies

import torch
from PIL import Image
import os
# Replace with actual import path upon release
from colpali_evaluator import ColPaLiEvaluator

# Initialize the evaluator (load the model)
CHECKPOINT_DIR = "./path/to/viColQwen/checkpoints/" # IMPORTANT: Set this path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 1024 # Should match the trained model

evaluator = ColPaLiEvaluator(
    checkpoint_path=CHECKPOINT_DIR,
    embed_dim=EMBED_DIM,
    device=DEVICE
)
print(f"Evaluator initialized on device: {evaluator.device}")
```

**2. Generating Embeddings for Images OR Texts:**

```python
# Example Images (ensure RGB format)
try:
    img1 = Image.open("path/to/your/image1.jpg").convert("RGB")
    img2 = Image.open("path/to/your/image2.png").convert("RGB")
    images = [img1, img2]
    image_embeddings = evaluator.get_image_embeddings(images)
    print("Image Embeddings Shape:", image_embeddings.shape) # Should be [2, 1024]
except FileNotFoundError:
    print("Image file(s) not found. Skipping image embedding generation.")
    image_embeddings = None

# Example Text Queries
queries = [
    "Mô tả cấu trúc tổ chức của bộ phận R&D.",
    "Provide a breakdown of last year's financial performance."
]

# Get text embeddings
query_embeddings = evaluator.get_query_embeddings(queries)
print("Query Embeddings Shape:", query_embeddings.shape) # Should be [2, 1024]

# Calculate similarity scores (if both embeddings were generated)
# The evaluator.score method likely computes cosine similarity or dot product
if image_embeddings is not None and query_embeddings is not None:
    similarity_scores = evaluator.score(query_embeddings, image_embeddings)
    print("Similarity Scores (Query vs. Image):\n", similarity_scores)
    # Example output: tensor([[score_q1_img1, score_q1_img2],
    #                         [score_q2_img1, score_q2_img2]])
```

**3. Generating Embeddings for Mixed Image(s) + Text(s):**

*This demonstrates the principle. The final API might offer a more direct method.*

```python
# Example Mixed Input
text_input = "Dựa vào hình ảnh này, hãy tóm tắt các điểm chính."
try:
    image_input = Image.open("path/to/relevant_document_page.jpg").convert("RGB")
except FileNotFoundError:
    print("Mixed input image not found. Skipping mixed embedding generation.")
    image_input = None

if image_input:
    # Illustrative functions showing potential internal logic or helper usage
    # (These might be methods of the evaluator class in the final release)
    def process_multimodal_input(evaluator, text, image, image_base_path=""):
        """Prepares combined text+image input for the model (Illustrative)."""
        if isinstance(image, str):
            img_path = os.path.join(image_base_path, image)
            pil_image = Image.open(img_path).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB') # Ensure RGB
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Ensure text contains the image token placeholder (model-specific)
        if "<image>" not in text:
            text = f"<image>\n{text}" # Prepend or append based on model training

        # Use the model's processor
        inputs = evaluator.processor(
            text=[text], images=[pil_image], padding="longest", return_tensors="pt"
        )

        # Add/rename keys required by the specific model architecture
        inputs["has_image_a"] = torch.tensor([True]) # Example flag
        if "pixel_values" in inputs:
            inputs["pixel_values_a"] = inputs.pop("pixel_values")
        if "input_ids" in inputs:
            inputs["input_ids_a"] = inputs.pop("input_ids")
        if "attention_mask" in inputs:
            inputs["attention_mask_a"] = inputs.pop("attention_mask")
        # Add other necessary keys based on the model's forward signature

        return inputs

    def get_multimodal_embedding(evaluator, text, image, image_base_path=""):
        """Gets embedding for a text+image combination (Illustrative)."""
        inputs = process_multimodal_input(evaluator, text, image, image_base_path)
        inputs = {k: v.to(evaluator.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            # CRITICAL: The exact method call depends on the final implementation.
            # It might be get_image_embeddings, get_multimodal_embeddings, or model(**inputs)
            # This example uses get_image_embeddings based on the user prompt, assuming it's overloaded.
            # Check documentation/code upon release.
            outputs = evaluator.get_image_embeddings(**inputs) # Or appropriate method call
            # The embedding might be the direct output or accessed via a key (e.g., outputs.embedding)
            embedding = outputs

        return embedding.cpu() # Return embedding on CPU

    # --- Generate the mixed embedding ---
    mixed_embedding = get_multimodal_embedding(evaluator, text_input, image_input)
    print("Mixed Modality Embedding Shape:", mixed_embedding.shape) # Should be [1, 1024]

    # This 'mixed_embedding' can now be used in similarity searches against
    # image_embeddings, query_embeddings, or other mixed_embeddings in the SAME vector DB.
```

## Potential Applications

Leveraging viColQwen's unified embeddings fundamentally enhances multimodal tasks:

*   **Superior Multimodal RAG:** Retrieve image(s) *and/or* text or even multi-turn instruction, using a single query vector, providing richer, coherent context to LLMs than disjoint systems.
*   **Simplified Graph RAG:** Build knowledge graphs with nodes representing images, text, multi-turn instruction, or multimodal documents, queryable via unified embeddings for complex relationship discovery.
*   **Effective Cross-Modal Search:** Robustly find images from text queries, text from image queries, or similar image-text pairs using standard vector search.
*   **Advanced Document Analysis:** Understand complex documents by capturing layout, images, and OCR'd text within one representation for clustering, classification, or search.
*   **Contextual Visual Search:** Find visually similar images, refined by accompanying textual context embedded simultaneously.

## Development Status & Future Work

*   This repository is under active development. Model checkpoints, refined code, comprehensive usage examples, and benchmarks will be released soon.
*   Ongoing work includes extensive benchmarking on Vietnamese and cross-lingual multimodal tasks, further model scaling exploration, and potential video integration.
*   Community feedback and contributions will be welcomed upon release.

**Stay tuned for the official model release! We believe viColQwen represents a significant step towards more intuitive and powerful multimodal AI.** ✨🚀

## License

*   The licensing details will be announced upon release.
*   A commercial license option will be available. For inquiries regarding commercial use, please contact us at **nguyen@hatto.com**.

## Citation

*(Please cite this repository URL until a formal publication is available)*

```bibtex
@misc{vicolqwen_github_2024,
  author       = {Steve Nguyen Anh Nguyen and the EraX AI Team},
  title        = {viColQwen: High-Performance Unified Embeddings for Advanced Multimodal Understanding},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/EraX-AI/viColQwen}} % Replace with actual final URL
}
```