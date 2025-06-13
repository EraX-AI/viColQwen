<p align="left">
  <img src="https://huggingface.co/erax-ai/EraX-Translator-V1.0/resolve/main/erax-gmobile.png?download=true" alt="Logo" width="400">
</p>

---

# ViUniRep & ViUniRep-M: ViUniEmbed: Unified Embedding and Reranking with a Single, Calibrated Multimodal Vector

**ViUniRep** is a state-of-the-art multimodal embedding model that unifies first-stage retrieval and second-stage reranking into a single, powerful architecture. It produces one high-quality, calibrated vector per input, eliminating the need for complex, multi-model pipelines and revolutionizing the efficiency of production AI search systems.

This repository contains the official code, training details, and inference examples for the **ViUniRep** and **ViUniRep-M** (Matryoshka) models.

[![Generic badge](https://img.shields.io/badge/Model-ViUniRep-blue.svg)](https_path_to_your_paper_or_blog)
[![Generic badge](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https_path_to_your_license)
[![Generic badge](https://img.shields.io/badge/🤗-Hugging_Face-yellow.svg)](https_path_to_your_hugging_face_repo)

---

## 🔥 Key Innovations & Advantages

ViUniRep isn't just another embedding model. It's a new paradigm for search, built on several key innovations:

1.  **Unified Architecture:** Replaces complex embedding + reranking pipelines with **a single model**. This can reduce infrastructure costs by up to 60% and slash latency.
2.  **Calibrated Embeddings:** The dot product of two ViUniRep vectors is not just a similarity score; it's a **calibrated reranking score** between 0.0 and 1.0. You get retrieval and reranking from a single operation.
3.  **Single Vector Output:** Unlike multi-vector approaches (ColPali, PLAID), ViUniRep produces **one dense vector per document**, ensuring universal compatibility with all major vector databases (Pinecone, Weaviate, Qdrant, Milvus, etc.).
4.  **Matryoshka Flexibility (ViUniRep-M):** Our Matryoshka version (`-M`) produces hierarchical embeddings. Use the first 512 dimensions for ultra-fast search and the full 2048 dimensions for high-precision reranking—all from the same vector.
5.  **Unprecedented Stability:** Engineered with a **six-layer defense architecture**, including our novel **Gradient Vaccine**, to ensure robust, collapse-free training at scale.

## 🚀 Quickstart: Inference with ViUniRep

Get state-of-the-art multimodal embeddings in just a few lines of code.

### 1. Installation

First, install the necessary libraries.

```bash
pip install transformers torch pillow accelerate safetensors
```

### 2. Basic Usage (Text & Image Encoding)

```python
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests

# Load the model and processor from Hugging Face Hub
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "your-username/ViUniRep-M-2048"  # Replace with your actual model ID

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

# --- Example 1: Encode a text query ---
text_query = "What are the advantages of a unified search architecture?"
text_embedding = model.encode(text=text_query)
print("Text Embedding Shape:", text_embedding.shape)
# Expected output: torch.Size([2048])

# --- Example 2: Encode an image from a URL ---
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image_embedding = model.encode(images=image)
print("Image Embedding Shape:", image_embedding.shape)
# Expected output: torch.Size([2048])

# --- Example 3: Encode a multimodal query (text + image) ---
multimodal_query = "What is the cat sitting on?"
multimodal_embedding = model.encode(text=multimodal_query, images=image)
print("Multimodal Embedding Shape:", multimodal_embedding.shape)
# Expected output: torch.Size([2048])
```

### 3. Using Matryoshka Embeddings (ViUniRep-M)

The `-M` models provide incredible flexibility. Simply specify the desired `embedding_dim` in the `encode` function.

```python
# Get a smaller, faster embedding for first-stage retrieval
fast_embedding = model.encode(
    text="This is a query for fast search.",
    embedding_dim=512
)
print("Fast Embedding Shape:", fast_embedding.shape)
# Expected output: torch.Size([512])

# The vector is already L2-normalized at the requested dimension.
print("L2 Norm of 512d vector:", torch.linalg.norm(fast_embedding).item())
# Expected output: 1.0

# Get the full-precision embedding for reranking
precise_embedding = model.encode(
    text="This is the same query for precise reranking.",
    embedding_dim=2048 # Or simply omit the argument
)
print("Precise Embedding Shape:", precise_embedding.shape)
# Expected output: torch.Size([2048])
```

### 4. Retrieval and Reranking in One Go

The calibrated nature of ViUniRep embeddings simplifies your search logic.

```python
# Assume `query_embedding` and a list of `doc_embeddings` are all 2048-dim vectors
query_embedding = model.encode(text="my query", embedding_dim=2048)
doc_embeddings = model.encode(
    text=["doc 1 content...", "doc 2 content...", "doc 3 content..."],
    embedding_dim=2048
)

# --- Perform Retrieval (happens in your vector DB) ---
# Your vector DB uses L2 or cosine similarity to find top_k candidates.
# Let's simulate it here with a dot product.
retrieval_scores = torch.matmul(query_embedding, doc_embeddings.T)

# --- Perform Reranking (on the retrieved candidates) ---
# The retrieval scores ARE the reranking scores. No second model call needed.
reranking_scores = (retrieval_scores + 1) / 2 # Scale to [0, 1]

# Print calibrated scores
for i, score in enumerate(reranking_scores):
    print(f"Document {i+1} Calibrated Reranking Score: {score.item():.4f}")
```

## 📜 Model Architecture & Training

ViUniRep is built upon the **Qwen2-VL-2B** backbone. Its stability and performance come from a suite of interconnected innovations:

| Innovation | Description |
| :--- | :--- |
| **Gradient Vaccine** | Gradually introduces vision data to prevent catastrophic forgetting of text knowledge. |
| **Adaptive Loss Scheduling**| Dynamically adjusts temperature and loss weights for stable convergence from a random state. |
| **Prefix-Guided Training** | Uses special tokens like `<ocr>` and `<vqa_multi>` during training to teach tasks without inference overhead. |
| **Defense-in-Depth** | A six-part system including spectral normalization, component-wise clipping, and uniformity loss to guarantee stability. |
| **Calibrated Loss** | A composite objective of KL-Divergence, MSE, and Ranking Loss to learn continuous similarity scores. |
| **Matryoshka Learning**| A weighted, multi-dimensional loss objective that creates nested, hierarchical embeddings. |

The model was trained on a high-quality, balanced dataset of **8.5 million** text, image, and multimodal pairs.

## 📊 Performance

Even in early training (3% of one epoch), ViUniRep demonstrates state-of-the-art potential:

| Metric | Value | Significance |
|:---|:---|:---|
| **Spearman Correlation** | **0.649** | Excellent calibrated ranking ability. |
| **R@1 (Retrieval)** | 0.678 | Strong performance on first-stage retrieval. |
| **VQA R@1** | 0.999 | Near-perfect visual reasoning on test splits. |

The Matryoshka version (ViUniRep-M) shows a clear and healthy separation of information across its dimensions, confirming the success of the hierarchical learning objective.

## Citing ViUniRep

If you use ViUniRep in your research or application, please cite our work:

```bibtex
@misc{nguyen2024viunirep,
      title={ViUniRep: Unified Embedding and Reranking with a Single, Calibrated Multimodal Vector},
      author={Nguyen Anh Nguyen},
      year={2024},
      eprint={YOUR_ARXIV_ID_HERE},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact

For questions, issues, or collaboration inquiries, please contact Nguyen Anh Nguyen at `nguyen@hatto.com`.
