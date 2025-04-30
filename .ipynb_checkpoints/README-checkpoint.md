<p align="left">
  <img src="https://huggingface.co/erax-ai/EraX-Translator-V1.0/resolve/main/erax-gmobile.png?download=true" alt="Logo">
</p>

# viPyloQwen: Embedding Đa phương thức Thống nhất: Tối ưu Loss Linh hoạt theo Tín hiệu Tiền tố

**(Mô hình sắp được phát hành - Vui lòng theo dõi!)**

[[English Version](README_en.md)] | **Tiếng Việt**

## Tóm tắt

Các hệ thống đa phương thức hiện đại thường đối mặt với thách thức do sự phức tạp của việc quản lý các không gian embedding riêng biệt cho nhiều loại dữ liệu khác nhau (như văn bản, hình ảnh). Điều này có thể dẫn đến sự phân mảnh trong biểu diễn, quy trình truy xuất cồng kềnh và hạn chế trong khả năng suy luận chéo phương thức. 

Chúng tôi giới thiệu **viPyloQwen**, một mô hình embedding đa phương thức tiên tiến, được thiết kế để tạo ra các **biểu diễn thống nhất, không gian đa chiều** cho hình ảnh, văn bản và các kết hợp tùy ý của chúng trong một không gian vector duy nhất, gắn kết. Chúng tôi đặt cho giải thuật này tên tiếng Anh là **Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization** (tên viPyloQwen là từ đây).

Nghiên cứu này, bao gồm việc phát triển và huấn luyện mô hình viPyloQwen, được thực hiện với sự hợp tác chặt chẽ của **đội ngũ công nghệ AI tại Công ty Cổ phần Viễn thông Di động Toàn Cầu Gtel Mobile JSC (GMobile)**. Chuyên môn kỹ thuật và sự hỗ trợ hợp tác của họ đóng vai trò vô cùng quan trọng trong suốt quá trình nghiên cứu và đào tạo mô hình.

Được xây dựng trên kiến trúc vision-language mạnh mẽ **Qwen2-VL 2B**, viPyloQwen sử dụng một framework học tương phản (contrastive learning) tinh vi. Mặc dù lấy cảm hứng từ các phương pháp như ColPali, viPyloQwen mang đến những cải tiến đáng kể, đặc biệt qua phương pháp huấn luyện độc đáo. Mô hình được huấn luyện trên một **tập dữ liệu quy mô lớn, cực kỳ đa dạng, vượt quá 11 triệu mẫu**. Tập dữ liệu được tuyển chọn tỉ mỉ này tích hợp một cách chiến lược các cặp tương đồng ngữ nghĩa văn bản-văn bản phức tạp (với điểm số tương đồng là liên tục 0.1...0.85), dữ liệu hướng dẫn phức tạp, và có lẽ đặc biệt nhất, một bộ sưu tập lớn các tình huống Nhận dạng Ký tự Quang học (OCR) và Trả lời Câu hỏi Trực quan (VQA) đa hình ảnh.

Đổi mới thuật toán cốt lõi nằm ở **chiến lược tối ưu hóa tổn thất hỗn hợp động được dẫn hướng bằng tiền tố** của viPyloQwen. Các tiền tố nhiệm vụ cụ thể (`<text_pair>`, `<instr>`, `<ocr>`, `<vqa_multi>`, `<vqa_single>`) được thêm vào đầu vào, đóng vai trò như tín hiệu để báo hiệu loại dữ liệu. Cơ chế này **kích hoạt động một hàm loss tương ứng, được thiết kế riêng** (bao gồm InfoNCE, Triplet Loss, MSE, và tối đa hóa độ tương đồng cosine) đặc thù cho từng loại mẫu.

Các embedding cuối cùng được trích xuất bằng phương pháp **pooling trung bình (mean pooling)** trên các token đầu ra của bộ mã hóa, đảm bảo thu giữ toàn diện thông tin ngữ nghĩa và thị giác. Kết quả là các embedding 1024 chiều, được tạo ra từ hỗn hợp dữ liệu phong phú và chiến lược huấn luyện độc đáo này, thể hiện sự hiểu biết ngữ nghĩa và hình ảnh sâu sắc, tinh tế. Điều này giúp đơn giản hóa và nâng cao đáng kể các ứng dụng đầu cuối như Sinh Tăng cường Truy xuất (RAG) đa phương thức, Graph RAG, tìm kiếm chéo phương thức và phân tích tài liệu phức tạp. Mặc dù mô hình thể hiện hiệu suất đặc biệt mạnh mẽ bằng **tiếng Việt** do trọng tâm dữ liệu, dữ liệu huấn luyện đa ngôn ngữ (bao gồm lượng đáng kể tiếng Anh và tiếng Trung) tạo điều kiện cho khả năng **chuyển giao zero-shot** hiệu quả sang các ngôn ngữ khác.

---

## Chi tiết Mô hình

*   **Kiến trúc Nền tảng:** `Qwen/Qwen2-VL-2B-Instruct` - Mô hình Ngôn ngữ-Thị giác (VLM) làm nền tảng.
*   **Chiến lược Embedding:** Không gian Embedding Thống nhất qua Học Tương phản Động được Dẫn hướng bằng Tiền tố.
*   **Chiều Embedding:** `1024`.
*   **Chiến lược Pooling:** **Pooling Trung bình (Mean Pooling)**. Vector embedding cuối cùng được lấy bằng cách tính trung bình các trạng thái ẩn của tất cả các token đầu ra từ lớp cuối cùng của bộ mã hóa Qwen2-VL, sau đó chuẩn hóa L2. Điều này tổng hợp thông tin trên toàn bộ chuỗi đầu vào (token văn bản và token patch hình ảnh).
*   **Biểu diễn Đầu vào:** Dữ liệu đầu vào (chuỗi văn bản, ảnh PIL) được xử lý bởi bộ xử lý của Qwen-VL. Hình ảnh được biểu diễn bằng token `<image>`. Điểm quan trọng: một **tiền tố nhiệm vụ cụ thể** được thêm vào *trước* nội dung văn bản chính để báo hiệu loại dữ liệu:
    *   `<text_pair>`: Cho cặp văn bản với điểm tương đồng liên tục.
    *   `<instr>`: Cho dữ liệu hướng dẫn (cặp instruction-response).
    *   `<ocr>`: Cho dữ liệu OCR/OCQ (ảnh + câu hỏi -> câu trả lời).
    *   `<vqa_multi>`: Cho VQA đa lượt (ảnh + câu hỏi -> câu trả lời).
    *   `<vqa_single>`: Cho VQA đơn lượt (ảnh + câu hỏi -> câu trả lời).
*   **Đầu ra:** Một vector dày `1024-d` duy nhất biểu diễn nội dung ngữ nghĩa và/hoặc thị giác của đầu vào.

---

## Huấn luyện

Sự mạnh mẽ và linh hoạt của viPyloQwen bắt nguồn từ sự kết hợp cộng hưởng giữa chiến lược tối ưu hóa độc đáo và dữ liệu huấn luyện cực kỳ đa dạng:

1.  **Tập Dữ liệu Phong phú và Không đồng nhất (Hơn 11 Triệu Mẫu):** Kho dữ liệu huấn luyện tích hợp nhiều loại dữ liệu và nhiệm vụ, được liên kết thông qua các tiền tố đầu vào:
    *   **Tương đồng Ngữ nghĩa Văn bản-Văn bản (`<text_pair>`, ~5.6M):** Các cặp $(t_a, t_b)$ với điểm tương đồng $s \in [0, 1]$, thúc đẩy hiểu biết văn bản tinh tế.
    *   **Thực hiện Hướng dẫn (`<instr>`, ~0.6M):** Các cặp (hướng dẫn đơn và đa lượt $i$, phản hồi $r$), tăng cường khả năng suy luận theo ngữ cảnh và biểu diễn việc thực thi nhiệm vụ.
    *   **OCR/OCQ Đa hình ảnh Đa dạng (`<ocr>`, ~2.5M):** Hạng mục này vượt xa việc nhận dạng văn bản tài liệu đơn giản. Nó bao gồm một phổ rộng các tác vụ nhận dạng văn bản trực quan trên 1-5 ảnh mỗi mẫu, chẳng hạn như:
        *   Chú thích cảnh đường phố và nhận dạng văn bản.
        *   Hiểu tài liệu toán học (công thức, sơ đồ).
        *   Sự tương tác giữa văn bản và hình ảnh trong tài liệu nói chung.
        *   Phân tích biểu đồ và sơ đồ.
        *   Nhận dạng chữ viết tay (ví dụ: hóa đơn, mẫu yêu cầu bảo hiểm, báo cáo tai nạn).
        *   Nhận dạng các giấy tờ thông dụng của Việt Nam (ví dụ: Căn cước công dân - CCCD, giấy phép lái xe).
    *   **VQA Đa hình ảnh Phức tạp (`<vqa_single>`, `<vqa_multi>`, ~2.5M):** Các tác vụ này (VQA đơn và đa lượt), cũng sử dụng 1-5 ảnh, đòi hỏi khả năng suy luận trực quan sâu sắc hơn được tích hợp với các truy vấn văn bản. Dữ liệu bao gồm:
        *   Trả lời câu hỏi trực quan tổng quát trên nhiều cảnh khác nhau.
        *   Diễn giải bảng và biểu đồ phức tạp đòi hỏi suy luận.
        *   **Phân tích Hình ảnh Y tế Chuyên sâu (~0.5M mẫu):** Một tập hợp con đáng kể dành riêng cho OCR và VQA trong lĩnh vực X quang. Điều này bao gồm việc phân tích các bản quét y tế đa dạng (hình ảnh da liễu, X-quang, CT, MRI) để trả lời các câu hỏi chẩn đoán liên quan đến các lĩnh vực sức khỏe quan trọng bao gồm da, xương, tim, phổi, não và răng.
    *   **Phân bổ Ngôn ngữ:** Mặc dù tập dữ liệu chủ yếu bao gồm nội dung **tiếng Việt** để đảm bảo hiệu suất mạnh mẽ trong bối cảnh này, nó tích hợp một cách chiến lược lượng đáng kể các mẫu **tiếng Anh** và **tiếng Trung** trong tất cả các danh mục. Nền tảng đa ngôn ngữ này rất quan trọng để cho phép mô hình **khái quát hóa zero-shot** hiệu quả sang các ngôn ngữ khác chưa được thấy.

2.  **Tối ưu hóa Tổn thất Hỗn hợp Động được Dẫn hướng bằng Tiền tố:**
    *   Như đã mô tả trước đây, tiền tố của mỗi mẫu sẽ động chọn một hàm loss phù hợp từ một bộ được định nghĩa trước.
    *   **Bộ Hàm Loss được Áp dụng:**
        *   `<text_pair>`: InfoNCE Đối xứng + Hồi quy Tương đồng MSE.
        *   `<instr>`: InfoNCE Đối xứng + Tối đa hóa Tương đồng Cosine Trực tiếp.
        *   `<ocr>`, `<vqa_single>`, `<vqa_multi>`: InfoNCE Đối xứng + Tổn thất Lề Triplet (Triplet Margin Loss) (lề có thể được điều chỉnh cho đa lượt).

Sự kết hợp giữa tập dữ liệu phong phú, đa dạng về lĩnh vực và cơ chế huấn luyện thích ứng này cho phép viPyloQwen phát triển một không gian embedding thực sự thống nhất và có năng lực cao, áp dụng được trong nhiều tình huống thực tế.

## Chi tiết Huấn luyện

Việc huấn luyện mô hình `viPyloQwen` đòi hỏi yêu cầu tài nguyên tính toán lớn, nhấn mạnh sự phức tạp của việc học từ một tập dữ liệu đa phương thức lớn và đa dạng như vậy.

*   **Phần cứng:** Mô hình được huấn luyện trên một cụm máy tính hiệu năng cao bao gồm **4x GPU NVIDIA H100 trên Vast.AI**, mỗi GPU có 94GB VRAM được kết nối qua NVLink.
*   **Thời gian:** Giai đoạn huấn luyện chính kéo dài khoảng **15 ngày** tính toán liên tục trên cấu hình phần cứng này.
*   **Framework:** Quá trình huấn luyện phân tán được điều phối sử dụng thư viện **`accelerate` của Hugging Face**, khai thác khả năng mở rộng đa GPU hiệu quả của nó với FSDP ZwRO-3.
*   **Độ chính xác & Tối ưu hóa:** Quá trình huấn luyện sử dụng độ chính xác hỗn hợp **`bfloat16`** để tối ưu hóa việc sử dụng bộ nhớ và thông lượng tính toán. **Flash Attention 2** đã được kích hoạt nhằm tăng cường hiệu quả cho cơ chế attention.
*   **Các siêu tham số chính (Key Hyperparameters):**
    *   **Chèn vào Qwen2VL tokenizer với (`<text_pair>`, `<instr>`, `<ocr>`, `<vqa_multi>`, `<vqa_single>`) và **mở rộng lớp embedding** tương ứng 
    *   **Mô hình cơ sở (Base Model):** `Qwen/Qwen2-VL-2B-Instruct`
    *   **Bộ tối ưu hóa (Optimizer):** AdamW (tiêu chuẩn của Hugging Face Trainer)
    *   **Tốc độ học (Learning Rate):** 1e-4 (với giảm theo hàm cosine sau 5% warmup)
    *   **Số Epochs:** 2
    *   **Kích thước lô trên mỗi thiết bị (Batch Size per device):** 24
    *   **Số bước Tích lũy Gradient (Gradient Accumulation Steps):** 8
    *   **Kích thước lô hiệu dụng toàn cục (Effective Global Batch Size):** 768 (24 * 4 GPUs * 8 accumulation)
    *   **Độ dài chuỗi tối đa (Max Sequence Length):** 8192 tokens
    *   **Suy giảm trọng số (Weight Decay):** 0.001
    *   **Chuẩn Gradient Tối đa (Max Gradient Norm):** 1.0
    *   **Chiến lược Pooling (Pooling Strategy):** Mean Pooling
    *   **Siêu tham số Hàm Loss:** Temperature = 0.07, Contrastive Margin = 0.2
*   **Dataset:** Huấn luyện trên tập dữ liệu hơn 11 triệu mẫu đã mô tả (`TRAIN_11M.jsonl`) và đánh giá bằng tập con 5 nghìn mẫu (`EVAL_5k.jsonl`).

Cấu hình này cho thấy yêu cầu tài nguyên đáng kể cần thiết để huấn luyện các mô hình embedding đa phương thức tiên tiến, có khả năng xử lý hiệu quả dữ liệu đa dạng trong thực tế.

---

## Tính năng & Ưu điểm Chính

*   ✅ **Embedding Đa phương thức Thống nhất:** Không gian vector đơn nhất, gắn kết giúp đơn giản hóa việc tích hợp và các tác vụ đầu cuối.
*   ✅ **Huấn luyện Dẫn hướng bằng Tiền tố:** Cho phép học các sắc thái, nhận biết nhiệm vụ trong không gian thống nhất.
*   ✅ **Dữ liệu Cực kỳ Đa dạng:** Huấn luyện trên tương đồng văn bản, hướng dẫn, OCR phức tạp (chữ viết tay, biểu mẫu, sơ đồ, y tế) và VQA sâu (suy luận, biểu đồ, X quang chuyên ngành) đảm bảo tính mạnh mẽ và khả năng ứng dụng rộng rãi.
*   ✅ **RAG/Tìm kiếm Đa phương thức Đơn giản hóa:** Cho phép truy vấn một chỉ mục duy nhất với các truy vấn văn bản, hình ảnh hoặc hỗn hợp để truy xuất thông tin đa phương thức liên quan.
*   ✅ **Tăng cường Hiểu biết Chéo phương thức:** Huấn luyện chung thúc đẩy các embedding nhạy cảm với các mối tương quan hình ảnh-văn bản tinh tế.
*   ✅ **Nắm bắt Chi tiết ở Chiều cao:** Embedding 1024-d thu giữ thông tin chi tiết quan trọng cho các tác vụ phức tạp.
*   ✅ **Nhận biết Đa hình ảnh:** Xử lý tự nhiên các đầu vào chứa nhiều hình ảnh.
*   ✅ **Mạnh mẽ Tiếng Việt & Zero-Shot Tốt:** Tối ưu hóa cho tiếng Việt với khả năng khái quát hóa chéo ngôn ngữ đã được chứng minh nhờ bao gồm dữ liệu đa ngôn ngữ.
*   ✅ **Nền tảng cho AI Tiên tiến:** Một khối xây dựng lý tưởng cho các hệ thống RAG đa phương thức, Graph RAG, tìm kiếm ngữ nghĩa, phân loại và phân tích phức tạp.

---

## Cách Sử dụng (Ví dụ Khái niệm)

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
    from model import ViPolyQwenEmbedder as ViPyloQwenEmbedder # Đổi tên khi import cho rõ
    logger.info(f"Imported ViPyloQwenEmbedder from model.py")

    # Load processor từ cùng đường dẫn model (quan trọng)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    logger.info(f"Processor loaded successfully.")

    # Load model viPyloQwen đã huấn luyện
    # Truyền embed_dim đúng vào đây nếu config của model không lưu
    embedder = ViPyloQwenEmbedder.from_pretrained(
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

## Ứng dụng Tiềm năng

*   **RAG Đa phương thức:** Truy xuất các đoạn văn bản, hình ảnh, bảng biểu hoặc các phần tài liệu có liên quan cao (bao gồm báo cáo y tế hoặc báo cáo tài chính) bằng các truy vấn thống nhất.
*   **Graph RAG:** Xây dựng đồ thị tri thức nơi các nút đại diện cho các thực thể đa dạng (bệnh nhân, tài liệu, quy trình, phát hiện hình ảnh) được liên kết qua các embedding thống nhất.
*   **Truy xuất Chéo phương thức:** Tìm kiếm hiệu quả hình ảnh y tế dựa trên mô tả văn bản, tìm tài liệu liên quan từ hình ảnh biểu mẫu, v.v.
*   **Trí tuệ Tài liệu (Document Intelligence):** Phân tích sâu các tài liệu phức tạp như yêu cầu bảo hiểm, bài báo khoa học hoặc hướng dẫn kỹ thuật, tận dụng cả bố cục trực quan và nội dung.
*   **Tìm kiếm Hình ảnh theo Ngữ cảnh:** Tìm các hình ảnh tương tự về mặt trực quan (ví dụ: ảnh quét y tế, ảnh sản phẩm) được tinh chỉnh bởi ngữ cảnh văn bản cụ thể đi kèm.

---

## Tình trạng Phát triển & Công việc Tương lai

*   Đang được phát triển tích cực. Các điểm kiểm tra (checkpoints) mô hình, mã đánh giá, benchmarks và ví dụ sử dụng toàn diện sẽ sớm được phát hành.
*   Công việc đang diễn ra bao gồm benchmarking sâu rộng trên các tác vụ tiếng Việt, tiếng Anh và chéo ngôn ngữ, các nghiên cứu cắt lớp (ablation studies) về các thành phần dữ liệu, khám phá các mô hình cơ sở lớn hơn và tích hợp tiềm năng các phương thức khác.

---

## Giấy phép

*   Chi tiết giấy phép sẽ được công bố khi phát hành.
*   Sẽ có tùy chọn giấy phép thương mại. Đối với các yêu cầu liên quan đến việc sử dụng thương mại, vui lòng liên hệ: **nguyen@hatto.com**.

---

## Trích dẫn

Vui lòng trích dẫn URL của kho lưu trữ này cho đến khi có ấn phẩm chính thức.

```bibtex
@misc{viPyloQwen_github_2025,
  author       = {Steve Nguyen Anh Nguyen EraX GMobile},
  title        = {viPyloQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/EraX-AI/viPyloQwen}}
}

@misc{faysse2024ColPali,
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