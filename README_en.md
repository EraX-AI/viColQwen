<p align="left">
  <img src="https://huggingface.co/erax-ai/EraX-Translator-V1.0/resolve/main/erax-gmobile.png?download=true" alt="Logo">
</p>

# viPyloQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization

[[Tiếng Việt](README.md)] | **English**

**(Model Release Pending - Stay Tuned!)**

## Abstract

Modern multimodal systems often face challenges due to the complexity of managing separate embedding spaces for diverse data types (e.g., text, images). This can lead to representational fragmentation, cumbersome retrieval pipelines, and limitations in cross-modal reasoning. 

We introduce **viPyloQwen**, an advanced multimodal embedding model designed to generate **high-dimensional, unified representations** for images, text, and their arbitrary combinations within a single, cohesive vector space. We called it Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization (hence viPyloQwen)

This research, including the development and training of the viPyloQwen model, was conducted with close collaboration from **the AI technology team at Gtel Mobile JSC (GMobile)**. Their technical expertise and collaborative support were crucial throughout the research process and model training.

Built upon the powerful **Qwen2-VL 2B** vision-language architecture, viPyloQwen employs a sophisticated contrastive learning framework. While inspired by approaches like ColPali, viPyloQwen introduces significant enhancements, particularly through its unique training methodology. The model is trained on a **large-scale, exceptionally diverse dataset exceeding 11 million samples**. This meticulously curated dataset strategically integrates challenging text-text semantic similarity pairs (with continuous scores), complex instruction-following data, and perhaps most distinctively, a vast collection of multi-image Optical Character Recognition (OCR) and Visual Question Answering (VQA) scenarios.

The core algorithmic innovation lies in viPyloQwen's **prefix-guided dynamic mixed-loss optimization strategy**. Task-specific prefixes (`<text_pair>`, `<instr>`, `<ocr>`, `<vqa_multi>`, `<vqa_single>`) are prepended to the input, serving as cues to signal the data type. This mechanism **dynamically triggers a corresponding, tailored loss function** (including InfoNCE, Triplet Loss, MSE, and direct cosine similarity maximization) specifically designed for each sample type.

Final embeddings are extracted using **mean pooling** over the encoder's output tokens, ensuring comprehensive capture of semantic and visual information. The resulting 1024-dimensional embeddings, derived from this rich data mixture and unique training strategy, exhibit nuanced semantic and visual understanding. This significantly simplifies and enhances downstream applications such as multimodal Retrieval-Augmented Generation (RAG), Graph RAG, cross-modal search, and complex document analysis. While demonstrating particularly strong performance in **Vietnamese** due to data focus, the model's multilingual training data (including substantial English and Chinese) facilitates effective zero-shot transfer capabilities to other languages.

---

## Model Details

*   **Base Architecture:** `Qwen/Qwen2-VL-2B-Instruct` - The foundational Vision-Language Model (VLM).
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
    *   **Extended Qwen2VL tokenizer with new special tokens (`<text_pair>`, `<instr>`, `<ocr>`, `<vqa_multi>`, `<vqa_single>`) and resize its embedding layer. 
    *   **Base Model:** `Qwen/Qwen2-VL-2B-Instruct`
    *   **Optimizer:** AdamW (standard with Hugging Face Trainer)
    *   **Learning Rate:** 1e-4 (with cosine decay after 5% warmup)
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
from transformers import AutoProcessor
import logging
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.auto import tqdm # Sử dụng tqdm.auto cho linh hoạt
import time # Thêm time để đo thời gian nếu cần

# --- Phần 1: Cấu hình và Khởi tạo ---

# -- Cấu hình Logging --
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("viPyloQwen_Usage")

# -- Cấu hình Model và Device --
# !!! THAY THẾ BẰNG ĐƯỜNG DẪN THỰC TẾ ĐẾN MODEL ĐÃ HUẤN LUYỆN CỦA BẠN !!!
MODEL_PATH = "./path/to/your/finetuned_viPyloQwen_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Kích thước embedding bạn đã huấn luyện (ví dụ: 1024)
EXPECTED_EMBED_DIM = 1024
# Kích thước batch cho inference (điều chỉnh tùy theo VRAM)
INFERENCE_BATCH_SIZE = 16
# Độ dài sequence tối đa (nếu cần override processor default)
MAX_LENGTH = None # Để None sẽ dùng processor default

logger.info(f"Starting viPyloQwen Usage Script on device: {DEVICE}")
logger.info(f"Attempting to load model from: {MODEL_PATH}")

# -- Load Model và Processor --
try:
    # Import lớp model từ file đã cập nhật
    from model import ColPaLiQwenEmbedder as viPyloQwenEmbedder # Đổi tên khi import cho rõ
    logger.info(f"Imported viPyloQwenEmbedder from model.py")

    # Load processor từ cùng đường dẫn model (quan trọng)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    logger.info(f"Processor loaded successfully.")

    # Load model viPyloQwen đã huấn luyện
    # Truyền embed_dim đúng vào đây nếu config của model không lưu
    embedder = viPyloQwenEmbedder.from_pretrained(
        MODEL_PATH,
        embed_dim=EXPECTED_EMBED_DIM, # Đảm bảo embed_dim đúng
        trust_remote_code=True
    )
    logger.info(f"Model loaded successfully.")

    # Gán processor cho embedder nếu __init__ không tự làm
    if not hasattr(embedder, 'processor') or embedder.processor is None:
         embedder.processor = processor
         logger.info("Processor assigned to embedder instance.")

    embedder.to(DEVICE)
    embedder.eval() # Chuyển sang chế độ inference
    logger.info(f"Model moved to {DEVICE} and set to evaluation mode.")

except FileNotFoundError:
    logger.error(f"Fatal Error: Model or processor files not found at {MODEL_PATH}. Please check the path.")
    exit()
except Exception as e:
    logger.error(f"Fatal Error: Could not load model or processor. Error: {e}")
    import traceback
    logger.error(traceback.format_exc())
    exit()

# --- Phần 2: Tạo Embeddings cho Lưu trữ (Mô phỏng Vector DB) ---

logger.info("\n--- Part 2: Generating Embeddings for Storage (Simulated DB) ---")

# -- Dữ liệu mẫu --
# !!! THAY THẾ BẰNG ĐƯỜNG DẪN VÀ DỮ LIỆU THỰC TẾ CỦA BẠN !!!
sample_data = {
    "doc1_chunk1": {"type": "text", "content": "Hợp đồng này quy định các điều khoản về việc cung cấp dịch vụ điện toán đám mây."},
    "doc1_chunk2": {"type": "text", "content": "Bên B chịu trách nhiệm bảo mật thông tin khách hàng."},
    "page_img1": {"type": "image", "path": "sample_images/page_image.png"},
    "invoice_img": {"type": "image", "path": "sample_images/invoice_handwritten.png"},
    "group1_img1": {"type": "image", "path": "sample_images/group_img1.jpg", "group": "group1"},
    "group1_img2": {"type": "image", "path": "sample_images/group_img2.jpg", "group": "group1"},
    "red_car_desc": {"type": "image_with_desc", "path": "sample_images/red_car.jpg", "text": "Ảnh chụp cận cảnh một chiếc xe hơi thể thao màu đỏ đang đỗ."},
    "invoice_ocr_q1": {"type": "ocr_task", "path": "sample_images/invoice_handwritten.png", "question": "Số tiền tổng cộng phải thanh toán là bao nhiêu?"},
    "license_ocr_q1": {"type": "ocr_task", "path": "sample_images/giay_phep_lai_xe.png", "question": "Ngày hết hạn của giấy phép này là khi nào?"},
    "scan_vqa_q1": {"type": "vqa_task", "path": "sample_images/medical_scan.jpg", "question": "Phát hiện có khối u bất thường ở thùy trên phổi trái không?"},
}

# -- Lưu trữ mô phỏng --
vector_database = {} # Lưu {id: numpy_embedding}
metadata_database = {} # Lưu {id: metadata}

# -- Quá trình Embedding --
logger.info("Starting embedding process for sample data...")
batch_texts_to_encode = []
batch_images_to_encode = []
batch_ids_to_encode = []

for item_id, data in sample_data.items():
    text_input = None
    image_input = None
    prefix_to_use = "" # Mặc định không prefix

    data_type = data["type"]

    # Chuẩn bị text và ảnh dựa trên loại dữ liệu
    if data_type == "text":
        text_input = data["content"]
        # KHÔNG prefix
    elif data_type == "image":
        try:
            image_input = Image.open(data["path"]).convert("RGB")
            # KHÔNG prefix (hàm encode sẽ tự thêm <image> placeholder)
        except FileNotFoundError:
            logger.warning(f"Image file not found for {item_id}: {data['path']}. Skipping.")
            continue
    elif data_type == "image_with_desc":
        try:
            image_input = Image.open(data["path"]).convert("RGB")
            text_input = data["text"]
            # KHÔNG prefix
        except FileNotFoundError:
            logger.warning(f"Image file not found for {item_id}: {data['path']}. Skipping.")
            continue
    elif data_type == "ocr_task":
        try:
            image_input = Image.open(data["path"]).convert("RGB")
            text_input = data["question"]
            prefix_to_use = "<ocr>" # <<<<< DÙNG PREFIX OCR >>>>>
        except FileNotFoundError:
            logger.warning(f"Image file not found for {item_id}: {data['path']}. Skipping.")
            continue
    elif data_type == "vqa_task":
        try:
            image_input = Image.open(data["path"]).convert("RGB")
            text_input = data["question"]
            # Có thể chọn <vqa_single> hoặc <vqa_multi> tùy ngữ cảnh
            prefix_to_use = "<vqa_single>" # <<<<< DÙNG PREFIX VQA >>>>>
        except FileNotFoundError:
            logger.warning(f"Image file not found for {item_id}: {data['path']}. Skipping.")
            continue
    else:
        logger.warning(f"Unknown data type '{data_type}' for item {item_id}. Skipping.")
        continue

    # Xử lý text input (thêm prefix nếu cần, hoặc tạo placeholder nếu chỉ có ảnh)
    if text_input is None and image_input is not None:
        final_text_input = "<image>" # Placeholder cho processor khi chỉ có ảnh
    elif text_input is not None and prefix_to_use:
        final_text_input = f"{prefix_to_use} {text_input}"
    else:
        final_text_input = text_input # Giữ nguyên text chunk hoặc text mô tả

    batch_texts_to_encode.append(final_text_input)
    batch_images_to_encode.append(image_input)
    batch_ids_to_encode.append(item_id)
    metadata_database[item_id] = data # Lưu metadata gốc

# Thực hiện embedding theo batch (nếu có dữ liệu)
if batch_ids_to_encode:
    logger.info(f"Encoding {len(batch_ids_to_encode)} items in batches...")
    try:
        # Gọi hàm encode đã tích hợp xử lý batch
        all_embeddings = embedder.encode(
            text=batch_texts_to_encode,
            images=batch_images_to_encode,
            batch_size=INFERENCE_BATCH_SIZE,
            max_length=MAX_LENGTH
        )
        # Lưu vào DB mô phỏng
        for i, item_id in enumerate(batch_ids_to_encode):
            vector_database[item_id] = all_embeddings[i:i+1].cpu().numpy() # Lưu từng vector [1, dim]
            # logger.info(f"Stored embedding for {item_id}")

        logger.info(f"Finished embedding {len(vector_database)} items.")

    except Exception as e:
        logger.error(f"Error during batch encoding: {e}")
        import traceback
        logger.error(traceback.format_exc())

# --- Phần 3: Truy vấn và So sánh ---

logger.info("\n--- Part 3: Querying and Comparisons ---")

# -- Chuẩn bị DB cho tìm kiếm --
db_ids = list(vector_database.keys())
if db_ids:
    db_embeddings = np.concatenate(list(vector_database.values()), axis=0)
    logger.info(f"Prepared simulated DB with {len(db_ids)} items. Embeddings matrix shape: {db_embeddings.shape}")
else:
    db_embeddings = None
    logger.warning("Simulated DB is empty. Search operations will be skipped.")

# -- Hàm tìm kiếm mô phỏng --
def search_db(query_embedding_np, top_k=3):
    """Simulates searching the vector database."""
    if db_embeddings is None or len(db_ids) == 0:
        logger.warning("DB is empty, cannot search.")
        return []
    if query_embedding_np.shape[1] != db_embeddings.shape[1]:
        logger.error(f"Query embedding dim ({query_embedding_np.shape[1]}) does not match DB embedding dim ({db_embeddings.shape[1]})!")
        return []

    try:
        similarities = cosine_similarity(query_embedding_np, db_embeddings)[0]
        # Lấy top K chỉ số (indices)
        # Sử dụng argpartition để hiệu quả hơn khi K nhỏ so với N
        # Hoặc dùng argsort đơn giản nếu N không quá lớn
        if top_k >= len(db_ids):
             top_k_indices = np.argsort(similarities)[::-1] # Lấy hết và đảo ngược
        else:
             top_k_indices = np.argpartition(similarities, -top_k)[-top_k:] # Lấy K phần tử lớn nhất (chưa sắp xếp)
             top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1] # Sắp xếp K phần tử đó

        results = []
        for idx in top_k_indices:
            results.append({
                "id": db_ids[idx],
                "score": similarities[idx],
                "metadata": metadata_database.get(db_ids[idx], {})
            })
        return results
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        return []

# -- Kịch bản Truy vấn / So sánh --

# 3a) Query bằng text thông thường
try:
    query_text_normal = "Tìm điều khoản bảo mật"
    logger.info(f"\nQuerying DB with text: '{query_text_normal}' (No prefix)...")
    # KHÔNG dùng prefix
    query_text_embedding = embedder.encode(text=query_text_normal).cpu().numpy()
    results_text = search_db(query_text_embedding)
    print("Text Query Results:")
    if results_text:
        for res in results_text:
            print(f"  - ID: {res['id']}, Score: {res['score']:.4f}, Type: {res['metadata'].get('type')}, Content: {str(res['metadata'].get('content', res['metadata'].get('path')))[:80]}...")
    else:
        print("  No results found or DB empty.")
except Exception as e:
    logger.error(f"Error during text query: {e}")

# 3b) Query bằng ảnh
try:
    query_image_path = "sample_images/query_car.jpg" # !!! THAY ĐƯỜNG DẪN !!!
    query_image = Image.open(query_image_path).convert("RGB")
    logger.info(f"\nQuerying DB with image: '{query_image_path}' (No prefix)...")
    # KHÔNG dùng prefix
    query_image_embedding = embedder.encode(images=query_image).cpu().numpy()
    results_image = search_db(query_image_embedding)
    print("Image Query Results:")
    if results_image:
        for res in results_image:
             print(f"  - ID: {res['id']}, Score: {res['score']:.4f}, Type: {res['metadata'].get('type')}, Path: {res['metadata'].get('path')}")
    else:
        print("  No results found or DB empty.")
except FileNotFoundError:
    logging.warning(f"Query image file not found: {query_image_path}. Skipping image query.")
except Exception as e:
    logging.error(f"Error during image query: {e}")

# 3c) Query bằng ảnh + text MÔ TẢ
try:
    query_img_desc_path = "sample_images/query_invoice.png" # !!! THAY ĐƯỜNG DẪN !!!
    query_img_desc = Image.open(query_img_desc_path).convert("RGB")
    query_desc = "Tìm các hóa đơn tương tự về bố cục"
    logger.info(f"\nQuerying DB with image+description: '{query_img_desc_path}' + '{query_desc}' (No prefix)...")
    # KHÔNG dùng prefix
    query_img_desc_embedding = embedder.encode(text=query_desc, images=query_img_desc).cpu().numpy()
    results_img_desc = search_db(query_img_desc_embedding)
    print("Image+Description Query Results:")
    if results_img_desc:
        for res in results_img_desc:
            print(f"  - ID: {res['id']}, Score: {res['score']:.4f}, Type: {res['metadata'].get('type')}, Content: {str(res['metadata'].get('content', res['metadata'].get('path')))[:80]}...")
    else:
        print("  No results found or DB empty.")
except FileNotFoundError:
    logging.warning(f"Query image file not found: {query_img_desc_path}. Skipping image+description query.")
except Exception as e:
    logging.error(f"Error during image+description query: {e}")


# 3d) Query bằng ảnh + text CÂU HỎI OCR/VQA
try:
    query_img_task_path = "sample_images/license_plate.jpg" # !!! THAY ĐƯỜNG DẪN !!!
    query_img_task = Image.open(query_img_task_path).convert("RGB")
    query_task_text = "Biển số xe này là gì?"
    query_prefix = "<ocr>" # <<<<< QUAN TRỌNG: DÙNG PREFIX PHÙ HỢP >>>>>
    logger.info(f"\nQuerying DB with image+task: '{query_img_task_path}' + '{query_prefix} {query_task_text}' (Using prefix)...")
    query_img_task_embedding = embedder.encode(text=f"{query_prefix} {query_task_text}", images=query_img_task).cpu().numpy()
    results_img_task = search_db(query_img_task_embedding)
    print("Image+Task (OCR/VQA) Query Results:")
    # Kết quả này sẽ khớp tốt nhất với các mục được embed bằng prefix tương ứng
    if results_img_task:
        for res in results_img_task:
            print(f"  - ID: {res['id']}, Score: {res['score']:.4f}, Type: {res['metadata'].get('type')}, Question: {res['metadata'].get('question')}")
    else:
        print("  No results found or DB empty.")
except FileNotFoundError:
    logging.warning(f"Query image file not found: {query_img_task_path}. Skipping image+task query.")
except Exception as e:
    logging.error(f"Error during image+task query: {e}")

# 3e) So sánh trực tiếp 2 danh sách (Arrays) Ảnh hoặc Text
try:
    # Ví dụ so sánh danh sách ảnh
    image_paths_array1 = ["sample_images/cat1.jpg", "sample_images/dog1.jpg"] # !!! THAY ĐƯỜNG DẪN !!!
    image_paths_array2 = ["sample_images/cat2.jpg", "sample_images/car1.jpg"] # !!! THAY ĐƯỜNG DẪN !!!
    images1 = [Image.open(p).convert("RGB") for p in image_paths_array1]
    images2 = [Image.open(p).convert("RGB") for p in image_paths_array2]
    logging.info(f"\nComparing two arrays of images (No prefix)...")

    # KHÔNG dùng prefix
    embeddings_array1_img = embedder.encode(images=images1, batch_size=INFERENCE_BATCH_SIZE).cpu().numpy()
    embeddings_array2_img = embedder.encode(images=images2, batch_size=INFERENCE_BATCH_SIZE).cpu().numpy()

    # Tính ma trận tương đồng (item i của array1 vs item j của array2)
    similarity_matrix_img = cosine_similarity(embeddings_array1_img, embeddings_array2_img)
    print("Image Array Comparison Matrix (Array1 vs Array2):")
    print(np.round(similarity_matrix_img, 4)) # Làm tròn để dễ đọc

    # Ví dụ so sánh danh sách text
    texts_array1 = ["Quả táo màu đỏ.", "Bầu trời trong xanh."]
    texts_array2 = ["Trái cây có màu đỏ.", "Màu xanh của bầu trời."]
    logging.info(f"\nComparing two arrays of texts (No prefix)...")
    # KHÔNG dùng prefix
    embeddings_array1_txt = embedder.encode(text=texts_array1, batch_size=INFERENCE_BATCH_SIZE).cpu().numpy()
    embeddings_array2_txt = embedder.encode(text=texts_array2, batch_size=INFERENCE_BATCH_SIZE).cpu().numpy()
    similarity_matrix_txt = cosine_similarity(embeddings_array1_txt, embeddings_array2_txt)
    print("Text Array Comparison Matrix (Array1 vs Array2):")
    print(np.round(similarity_matrix_txt, 4))

except FileNotFoundError:
    logging.warning("One or more image files not found for array comparison. Skipping.")
except Exception as e:
    logging.error(f"Error during array comparison: {e}")


logger.info("\n--- Full Example Execution Finished ---")
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