# viPyloQwen: Embedding Đa phương thức Thống nhất: Tối ưu Loss Linh hoạt theo Tín hiệu Tiền tố
(Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization)

**(Mô hình sắp được phát hành - Vui lòng theo dõi!)**

[[English Version](README.en)] | **Tiếng Việt**

## Tóm tắt

Các hệ thống đa phương thức hiện đại thường đối mặt với thách thức do sự phức tạp của việc quản lý các không gian embedding riêng biệt cho nhiều loại dữ liệu khác nhau (như văn bản, hình ảnh). Điều này có thể dẫn đến sự phân mảnh trong biểu diễn, quy trình truy xuất cồng kềnh và hạn chế trong khả năng suy luận chéo phương thức. Chúng tôi giới thiệu **viPyloQwen**, một mô hình embedding đa phương thức tiên tiến, được thiết kế để tạo ra các **biểu diễn thống nhất, chiều cao** cho hình ảnh, văn bản và các kết hợp tùy ý của chúng trong một không gian vector duy nhất, gắn kết.

Được xây dựng trên kiến trúc vision-language mạnh mẽ **Qwen2-VL 2B**, viPyloQwen sử dụng một framework học tương phản (contrastive learning) tinh vi. Mặc dù lấy cảm hứng từ các phương pháp như ColPali, viPyloQwen mang đến những cải tiến đáng kể, đặc biệt qua phương pháp huấn luyện độc đáo. Mô hình được huấn luyện trên một **tập dữ liệu quy mô lớn, cực kỳ đa dạng, vượt quá 11 triệu mẫu**. Tập dữ liệu được tuyển chọn tỉ mỉ này tích hợp một cách chiến lược các cặp tương đồng ngữ nghĩa văn bản-văn bản phức tạp (với điểm số liên tục), dữ liệu hướng dẫn phức tạp, và có lẽ đặc biệt nhất, một bộ sưu tập lớn các tình huống Nhận dạng Ký tự Quang học (OCR) và Trả lời Câu hỏi Trực quan (VQA) đa hình ảnh.

Đổi mới thuật toán cốt lõi nằm ở **chiến lược tối ưu hóa tổn thất hỗn hợp động được dẫn hướng bằng tiền tố** của viPyloQwen. Các tiền tố nhiệm vụ cụ thể (`<text_pair>`, `<instr>`, `<ocr>`, `<vqa_multi>`, `<vqa_single>`) được thêm vào đầu vào, đóng vai trò như tín hiệu để báo hiệu loại dữ liệu. Cơ chế này **kích hoạt động một hàm loss tương ứng, được thiết kế riêng** (bao gồm InfoNCE, Triplet Loss, MSE, và tối đa hóa độ tương đồng cosine) đặc thù cho từng loại mẫu.

Các embedding cuối cùng được trích xuất bằng phương pháp **pooling trung bình (mean pooling)** trên các token đầu ra của bộ mã hóa, đảm bảo thu giữ toàn diện thông tin ngữ nghĩa và thị giác. Kết quả là các embedding 1024 chiều, được tạo ra từ hỗn hợp dữ liệu phong phú và chiến lược huấn luyện độc đáo này, thể hiện sự hiểu biết ngữ nghĩa và hình ảnh sâu sắc, tinh tế. Điều này giúp đơn giản hóa và nâng cao đáng kể các ứng dụng đầu cuối như Sinh Tăng cường Truy xuất (RAG) đa phương thức, Graph RAG, tìm kiếm chéo phương thức và phân tích tài liệu phức tạp. Mặc dù mô hình thể hiện hiệu suất đặc biệt mạnh mẽ bằng **tiếng Việt** do trọng tâm dữ liệu, dữ liệu huấn luyện đa ngôn ngữ (bao gồm lượng đáng kể tiếng Anh và tiếng Trung) tạo điều kiện cho khả năng **chuyển giao zero-shot** hiệu quả sang các ngôn ngữ khác.

---

## Chi tiết Mô hình

*   **Kiến trúc Nền tảng:** `Qwen/Qwen2-VL-2B` - Mô hình Ngôn ngữ-Thị giác (VLM) làm nền tảng.
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

## Παράδειγμα Huấn luyện

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
# Giả sử lớp viPyloQwenEmbedder có sẵn sau khi phát hành
# from viPyloQwen_embedder import viPyloQwenEmbedder

# embedder = viPyloQwenEmbedder(checkpoint_path="./duong/dan/toi/viPyloQwen/", device="cuda")

# --- Ví dụ: VQA Đơn lượt (ví dụ: Ảnh Y tế) ---
prefix_vqa = "<vqa_single>"
van_ban_dau_vao = "Có bằng chứng về gãy xương ở đầu dưới xương quay không?" # Truy vấn ví dụ
anh_dau_vao = Image.open("xquang_cotay.png").convert("RGB") # Ảnh ví dụ

# Lệnh mã hóa khái niệm - tiền tố hướng dẫn xử lý nội bộ
# embedding_hon_hop = embedder.encode(text=f"{prefix_vqa} {van_ban_dau_vao}", images=[anh_dau_vao])
# print(embedding_hon_hop.shape) # Dự kiến: torch.Size([1, 1024])

# --- Ví dụ: Tương đồng Văn bản ---
prefix_sim = "<text_pair>"
van_ban_a = "Bệnh nhân báo cáo khó chịu nhẹ."
van_ban_b = "Đối tượng trải qua cơn đau nhẹ."

# Lệnh mã hóa khái niệm
# embedding_a = embedder.encode(text=f"{prefix_sim} {van_ban_a}")
# embedding_b = embedder.encode(text=f"{prefix_sim} {van_ban_b}")

# Tính toán độ tương đồng (ví dụ: cosine)
# do_tuong_dong = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
# print(do_tuong_dong)

# --- Ví dụ: OCR (ví dụ: Biểu mẫu Viết tay) ---
prefix_ocr = "<ocr>"
van_ban_dau_vao = "Số hợp đồng được liệt kê là gì?" # Truy vấn ví dụ
anh_dau_vao = Image.open("mau_yeucau_viet_tay.jpg").convert("RGB") # Ảnh ví dụ

# Lệnh mã hóa khái niệm
# embedding_mau = embedder.encode(text=f"{prefix_ocr} {van_ban_dau_vao}", images=[anh_dau_vao])
# print(embedding_mau.shape) # Dự kiến: torch.Size([1, 1024])
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
  author       = {Steve Nguyen Anh Nguyen},
  title        = {viPyloQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization},
  year         = {2025},
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