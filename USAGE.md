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
```

# Phần 1: Cấu hình và Khởi tạo

```python
# -- Cấu hình Logging --
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("viPyloQwen_Usage")

# -- Cấu hình Model và Device --
MODEL_PATH = "./path/to/your/finetuned_viPyloQwen_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kích thước embedding bạn đã huấn luyện (ví dụ: 1024)
EXPECTED_EMBED_DIM = 1024

# Kích thước batch cho inference (điều chỉnh tùy theo VRAM)
INFERENCE_BATCH_SIZE = 16

# Độ dài sequence tối đa (nếu cần override processor default)
MAX_LENGTH = 8192

logger.info(f"Starting viPyloQwen Usage Script on device: {DEVICE}")
logger.info(f"Attempting to load model from: {MODEL_PATH}")

# -- Load Model và Processor --
try:
    from model import ViPolyQwenEmbedder as ViPyloQwenEmbedder
    logger.info(f"Imported ViPyloQwenEmbedder from model.py")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    logger.info(f"Processor loaded successfully.")

    # Load model viPyloQwen đã huấn luyện
    embedder = ViPyloQwenEmbedder.from_pretrained(
        MODEL_PATH,
        embed_dim=EXPECTED_EMBED_DIM,
        processor = processor,
        trust_remote_code=True
    )
    logger.info(f"Model loaded successfully.")

    embedder.to(DEVICE)
    embedder.eval()
    logger.info(f"Model moved to {DEVICE} and set to evaluation mode.")

except FileNotFoundError:
    logger.error(f"Fatal Error: Model or processor files not found at {MODEL_PATH}. Please check the path.")
    exit()
except Exception as e:
    logger.error(f"Fatal Error: Could not load model or processor. Error: {e}")
    import traceback
    logger.error(traceback.format_exc())
    exit()
```

# Phần 2: Tạo Embeddings cho Lưu trữ (Mô phỏng Vector DB)

```python
# -- Dữ liệu ví dụ --
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

# -- Lưu trữ vector DB mô phỏng --
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
        # Trường hợp này KHÔNG prefix
        
    elif data_type == "image":
        try:
            image_input = Image.open(data["path"]).convert("RGB")
            # Trường hợp này KHÔNG prefix (hàm encode sẽ tự thêm <image> placeholder)
            
        except FileNotFoundError:
            logger.warning(f"Image file not found for {item_id}: {data['path']}. Skipping.")
            continue
            
    elif data_type == "image_with_desc":
        try:
            image_input = Image.open(data["path"]).convert("RGB")
            text_input = data["text"]
            # Trường hợp này KHÔNG prefix
        except FileNotFoundError:
            logger.warning(f"Image file not found for {item_id}: {data['path']}. Skipping.")
            continue
            
    elif data_type == "ocr_task":
        try:
            image_input = Image.open(data["path"]).convert("RGB")
            text_input = data["question"]
            prefix_to_use = "<ocr>" # <<<<< PHẢI DÙNG PREFIX OCR >>>>>
        except FileNotFoundError:
            logger.warning(f"Image file not found for {item_id}: {data['path']}. Skipping.")
            continue
            
    elif data_type == "vqa_task":
        try:
            image_input = Image.open(data["path"]).convert("RGB")
            text_input = data["question"]
            # Có thể chọn <vqa_single> hoặc <vqa_multi> tùy ngữ cảnh
            prefix_to_use = "<vqa_single>" # <<<<< PHẢI DÙNG PREFIX VQA >>>>>
        except FileNotFoundError:
            logger.warning(f"Image file not found for {item_id}: {data['path']}. Skipping.")
            continue
            
    else:
        logger.warning(f"Unknown data type '{data_type}' for item {item_id}. Skipping.")
        continue

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

# Thực hiện embedding theo batch
if batch_ids_to_encode:
    logger.info(f"Encoding {len(batch_ids_to_encode)} items in batches...")
    try:
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
```

# Phần 3: Truy vấn và So sánh

## Chuẩn bị DB cho tìm kiếm

```python
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
```

## Kịch bản Truy vấn / So sánh

### 3a) Query bằng text thông thường

```python
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
```

### 3b) Query bằng ảnh

```python
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
```

### 3c) Query bằng ảnh + text MÔ TẢ

```python
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
```

### 3d) Query bằng ảnh + text CÂU HỎI OCR/VQA

```python
try:
    query_img_task_path = "sample_images/license_plate.jpg" # !!! THAY ĐƯỜNG DẪN !!!
    query_img_task = Image.open(query_img_task_path).convert("RGB")
    query_task_text = "Biển số xe này là gì?"
    
    query_prefix = "<ocr>" # <<<<< QUAN TRỌNG: PHẢI DÙNG PREFIX PHÙ HỢP >>>>>
    
    logger.info(f"\nQuerying DB with image+task: '{query_img_task_path}' + '{query_prefix} {query_task_text}' (Using prefix)...")
    query_img_task_embedding = embedder.encode(text=f"{query_prefix} {query_task_text}", images=query_img_task).cpu().numpy()
    results_img_task = search_db(query_img_task_embedding)
    print("Image+Task (OCR/VQA) Query Results:")
    if results_img_task:
        for res in results_img_task:
            print(f"  - ID: {res['id']}, Score: {res['score']:.4f}, Type: {res['metadata'].get('type')}, Question: {res['metadata'].get('question')}")
    else:
        print("  No results found or DB empty.")
except FileNotFoundError:
    logging.warning(f"Query image file not found: {query_img_task_path}. Skipping image+task query.")
except Exception as e:
    logging.error(f"Error during image+task query: {e}")
```

### 3e) So sánh trực tiếp 2 danh sách (Arrays) Ảnh hoặc Text

```python
try:
    # Ví dụ so sánh danh sách ảnh
    image_paths_array1 = ["sample_images/cat1.jpg", "sample_images/dog1.jpg"]
    image_paths_array2 = ["sample_images/cat2.jpg", "sample_images/car1.jpg"]
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
```