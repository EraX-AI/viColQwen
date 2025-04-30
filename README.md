<p align="left">
  <img src="https://huggingface.co/erax-ai/EraX-Translator-V1.0/resolve/main/erax-gmobile.png?download=true" alt="Logo" width="400">
</p>

# viPolyQwen: Embedding Đa phương thức với Tối ưu Loss Linh hoạt theo Tín hiệu Tiền tố với Attention Pooling

[English](README_en.md) | **Tiếng Việt**

**(Mô hình sắp được phát hành - Vui lòng theo dõi!)**

## Tóm tắt

Các hệ thống đa phương thức hiện đại thường đối mặt với thách thức do sự phức tạp của việc quản lý các không gian embedding riêng biệt cho nhiều loại dữ liệu khác nhau (ví dụ: văn bản, hình ảnh). Điều này có thể dẫn đến sự phân mảnh trong biểu diễn, quy trình truy xuất cồng kềnh và hạn chế trong khả năng suy luận chéo phương thức.

Chúng tôi giới thiệu **viPolyQwen**, một mô hình embedding đa phương thức tiên tiến, được thiết kế để tạo ra các **biểu diễn thống nhất, chiều cao** cho hình ảnh, văn bản và các kết hợp tùy ý của chúng trong một không gian vector duy nhất, gắn kết. Tên gọi của mô hình phản ánh phương pháp cốt lõi: **Embedding Đa phương thức Thống nhất qua Tối ưu Loss Linh hoạt theo Tín hiệu Tiền tố (Prefix-Guided Dynamic Loss Optimization)**, xây dựng trên kiến trúc **Qwen 2 Visual Language**.

Nghiên cứu này, bao gồm việc phát triển và huấn luyện mô hình viPolyQwen, được thực hiện với sự hợp tác chặt chẽ của **đội ngũ công nghệ AI tại Công ty Cổ phần Viễn thông Di động Toàn Cầu Gtel Mobile JSC (GMobile)**. Chuyên môn kỹ thuật và sự hỗ trợ hợp tác của họ đóng vai trò vô cùng quan trọng trong suốt quá trình nghiên cứu và đào tạo mô hình.

Được xây dựng trên kiến trúc vision-language mạnh mẽ **Qwen2-VL 2B-Instruct**, viPolyQwen sử dụng một framework học tương phản (contrastive learning) tinh vi, được huấn luyện trên một **tập dữ liệu quy mô lớn, cực kỳ đa dạng, vượt quá 11 triệu mẫu**. Tập dữ liệu được tuyển chọn tỉ mỉ này tích hợp một cách chiến lược các cặp tương đồng ngữ nghĩa văn bản-văn bản phức tạp (với điểm số liên tục), dữ liệu thực hiện hướng dẫn phức tạp, và có lẽ đặc biệt nhất, một bộ sưu tập lớn các tình huống Nhận dạng Ký tự Quang học (OCR) và Trả lời Câu hỏi Trực quan (VQA) đa hình ảnh (bao gồm tài liệu, biểu đồ, chữ viết tay và hình ảnh y tế chuyên ngành).

Đổi mới thuật toán cốt lõi nằm ở **chiến lược tối ưu hóa tổn thất hỗn hợp động được dẫn hướng bằng tiền tố** của viPolyQwen. Các tiền tố nhiệm vụ cụ thể (`<text_pair>`, `<instr>`, `<ocr>`, `<vqa_multi>`, `<vqa_single>`) dẫn hướng mô hình bằng cách báo hiệu loại dữ liệu, **kích hoạt động một hàm loss tương ứng, được thiết kế riêng** (bao gồm InfoNCE, Triplet Loss, MSE, và Tối đa hóa Tương đồng Cosine) cho từng loại mẫu.

Quan trọng hơn, thay vì sử dụng các phương pháp pooling thông thường như mean pooling hay last-token pooling, vector embedding 1D cuối cùng được trích xuất bằng **Attention Pooling**. Cơ chế này cho phép mô hình **tập trung động vào các đặc trưng thị giác và văn bản nổi bật nhất** trong chuỗi token đầu ra của bộ mã hóa trước khi chiếu (projection). Bằng cách học cách gán trọng số cao hơn cho các đặc trưng quan trọng (như vùng văn bản trong ảnh hoặc các khái niệm ngữ nghĩa chính), attention pooling hướng tới việc tạo ra các embedding 1D **giàu thông tin và tinh tế hơn** so với việc lấy trung bình đơn giản, giúp tăng cường đáng kể khả năng nắm bắt nội dung ngữ nghĩa, ngay cả từ các hình ảnh chứa văn bản.

Kết quả là các embedding 1024 chiều tạo điều kiện cho các ứng dụng đầu cuối mạnh mẽ như Sinh Tăng cường Truy xuất (RAG) đa phương thức, Graph RAG, tìm kiếm chéo phương thức và phân tích tài liệu phức tạp. Mặc dù được tối ưu hóa cho **tiếng Việt**, dữ liệu huấn luyện đa ngôn ngữ cho phép mô hình có khả năng **chuyển giao zero-shot** hiệu quả.

---

## Chi tiết Mô hình

*   **Kiến trúc Nền tảng:** `Qwen/Qwen2-VL-2B-Instruct` - Mô hình Ngôn ngữ-Thị giác (VLM) làm nền tảng.
*   **Chiến lược Embedding:** Không gian Embedding Thống nhất qua Học Tương phản Động được Dẫn hướng bằng Tiền tố với **Attention Pooling**.
*   **Chiều Embedding:** `1024`.
*   **Chiến lược Pooling:** **Attention Pooling.** Đây là một điểm khác biệt chính. Thay vì lấy trung bình đơn giản (mean pooling) hoặc chọn token cuối cùng, viPolyQwen sử dụng một *cơ chế chú ý học được (learned attention mechanism)* trên chuỗi trạng thái ẩn cuối cùng (đại diện cho cả token văn bản và patch hình ảnh).
    *   Nó tính toán điểm chú ý (attention scores) dựa trên mức độ liên quan của mỗi trạng thái ẩn đối với ngữ cảnh tổng thể.
    *   Nó gán trọng số cao hơn cho các trạng thái chứa nhiều thông tin hơn (ví dụ: các vùng văn bản cụ thể trong ảnh, các đối tượng thị giác chính, các token ngữ nghĩa quan trọng).
    *   Nó tính toán một *trung bình có trọng số (weighted average)* dựa trên các trọng số chú ý này.
    *   **Lợi ích:** Điều này cho phép mô hình tạo ra một biểu diễn 1D phù hợp hơn với ngữ cảnh và tinh tế hơn bằng cách tập trung vào các đặc trưng nổi bật, cải thiện đáng kể việc nắm bắt bản chất ngữ nghĩa và thị giác cốt lõi so với việc lấy trung bình đồng nhất. Điều này đặc biệt có lợi cho việc biểu diễn các hình ảnh chứa văn bản hoặc các cấu trúc trực quan phức tạp như biểu đồ và bảng biểu trong một vector duy nhất.
*   **Biểu diễn Đầu vào:** Dữ liệu đầu vào (chuỗi văn bản, ảnh PIL) được xử lý bởi bộ xử lý của Qwen-VL. Hình ảnh được biểu diễn bằng token `<image>`. Điểm quan trọng: một **tiền tố nhiệm vụ cụ thể** được thêm vào *trước* nội dung văn bản chính trong quá trình *huấn luyện* để báo hiệu loại dữ liệu và dẫn hướng việc tính toán loss:
    *   `<text_pair>`: Cho cặp tương đồng văn bản.
    *   `<instr>`: Cho dữ liệu thực hiện hướng dẫn.
    *   `<ocr>`: Cho dữ liệu OCR/OCQ.
    *   `<vqa_multi>`: Cho VQA đa lượt.
    *   `<vqa_single>`: Cho VQA đơn lượt.
    *(Lưu ý: Đối với inference/embedding thông thường, tiền tố thường được bỏ qua trừ khi truy vấn một nhiệm vụ cụ thể như OCR/VQA - xem Hướng dẫn Sử dụng)*.
*   **Đầu ra:** Một vector dày `1024-d` duy nhất, đã được chuẩn hóa L2, đại diện cho đầu vào.

---

## Παράδειγμα Huấn luyện

Sự mạnh mẽ và linh hoạt của viPolyQwen bắt nguồn từ sự kết hợp cộng hưởng giữa chiến lược tối ưu hóa độc đáo và dữ liệu huấn luyện cực kỳ đa dạng:

1.  **Tập Dữ liệu Phong phú và Không đồng nhất (>11M Mẫu):** (Mô tả chi tiết các thành phần dữ liệu - tương đồng văn bản, hướng dẫn, OCR, VQA, y tế - giữ nguyên như bản trước).
    *   **Phân bổ Ngôn ngữ:** Chủ yếu là **tiếng Việt**, với lượng đáng kể mẫu **tiếng Anh** và **tiếng Trung**, thúc đẩy khả năng khái quát hóa zero-shot mạnh mẽ.

2.  **Tối ưu hóa Tổn thất Hỗn hợp Động được Dẫn hướng bằng Tiền tố:**
    *   Trong quá trình huấn luyện, tiền tố của mỗi mẫu báo hiệu hàm loss phù hợp.
    *   **Bộ Hàm Loss được Áp dụng:**
        *   `<text_pair>`: InfoNCE Đối xứng + Hồi quy Tương đồng MSE.
        *   `<instr>`: InfoNCE Đối xứng + Tối đa hóa Tương đồng Cosine Trực tiếp.
        *   `<ocr>`, `<vqa_single>`, `<vqa_multi>`: InfoNCE Đối xứng + Tổn thất Lề Triplet.
    *   Vector embedding 1D cuối cùng được sử dụng để tính toán các hàm loss này được tạo ra thông qua **Attention Pooling** áp dụng trên chuỗi đầu ra của bộ mã hóa.

Sự kết hợp này cho phép viPolyQwen học được một không gian embedding thống nhất có năng lực cao, áp dụng được trong nhiều tình huống thực tế đa dạng.

## Chi tiết Huấn luyện

Việc huấn luyện mô hình `viPolyQwen` đòi hỏi yêu cầu tài nguyên tính toán lớn.

*   **Phần cứng:** Huấn luyện trên cụm máy tính với **4x GPU NVIDIA H100 (94GB VRAM, NVLink)** qua Vast.AI.
*   **Thời gian:** Khoảng **15 ngày** tính toán liên tục.
*   **Framework:** Huấn luyện phân tán qua Hugging Face `accelerate` sử dụng FSDP (có thể là ZeRO-3).
*   **Độ chính xác & Tối ưu hóa:** Độ chính xác hỗn hợp **`bfloat16`**; **Flash Attention 2**.
*   **Các siêu tham số chính (Key Hyperparameters):**
    *   Tokenizer/Embeddings: Mở rộng tokenizer/lớp embedding của Qwen2VL cho các token đặc biệt mới.
    *   **Mô hình cơ sở:** `Qwen/Qwen2-VL-2B-Instruct`
    *   **Bộ tối ưu hóa:** AdamW
    *   **Tốc độ học:** 1e-4 (giảm cosine sau 5% warmup)
    *   **Số Epochs:** 2
    *   **Kích thước lô (mỗi device):** 24
    *   **Tích lũy Gradient:** 8 (Kích thước lô hiệu dụng toàn cục: 768)
    *   **Độ dài chuỗi tối đa:** 8192
    *   **Suy giảm trọng số:** 0.001
    *   **Chuẩn Gradient Tối đa:** 1.0
    *   **Chiến lược Pooling:** **Attention Pooling** *(Trong huấn luyện, loss được tính trên embedding đã qua attention pooling)*
    *   **Siêu tham số Loss:** Temperature = 0.07, Contrastive Margin = 0.2
*   **Dataset:** Hơn 11 triệu mẫu huấn luyện, 5 nghìn mẫu đánh giá.

---

## Tính năng & Ưu điểm Chính

*   ✅ **Embedding Đa phương thức Thống nhất:** Không gian vector đơn nhất giúp đơn giản hóa tích hợp.
*   ✅ **Huấn luyện Dẫn hướng bằng Tiền tố:** Cho phép học các sắc thái, nhận biết nhiệm vụ trong quá trình huấn luyện.
*   ✅ **Attention Pooling:** Tạo ra embedding 1D **giàu thông tin và tinh tế hơn** bằng cách tập trung vào các đặc trưng thị giác/văn bản nổi bật, **nâng cao khả năng nắm bắt chi tiết ngữ nghĩa (bao gồm khái niệm text-trong-ảnh)** so với mean pooling.
*   ✅ **Dữ liệu Cực kỳ Đa dạng:** Mạnh mẽ nhờ huấn luyện trên tương đồng văn bản, hướng dẫn, OCR phức tạp, và VQA sâu (bao gồm y tế).
*   ✅ **RAG/Tìm kiếm Đa phương thức Đơn giản hóa:** Truy xuất hiệu quả từ một chỉ mục duy nhất.
*   ✅ **Tăng cường Hiểu biết Chéo phương thức:** Huấn luyện chung thúc đẩy các mối tương quan sâu sắc.
*   ✅ **Nắm bắt Chi tiết ở Chiều cao:** Embedding 1024-d thu giữ thông tin tinh vi.
*   ✅ **Nhận biết Đa hình ảnh:** Xử lý tự nhiên các đầu vào chứa nhiều hình ảnh.
*   ✅ **Mạnh mẽ Tiếng Việt & Zero-Shot Tốt:** Tối ưu hóa cho tiếng Việt với tiềm năng chéo ngôn ngữ.
*   ✅ **Nền tảng cho AI Tiên tiến:** Khối xây dựng lý tưởng cho các hệ thống đa phương thức thế hệ mới.

---

## Hướng dẫn Sử dụng: [Hướng dẫn Sử dụng & Ví dụ](USAGE_vi.md)

*(Hướng dẫn sử dụng sẽ giải thích chiến lược sử dụng/bỏ qua tiền tố trong quá trình inference như đã thảo luận: embed dữ liệu chung không cần tiền tố, chỉ dùng tiền tố cho các truy vấn OCR/VQA cụ thể nếu muốn).*

---

## Ứng dụng Tiềm năng

*   **RAG Đa phương thức:** Truy xuất các đoạn văn bản, hình ảnh, bảng biểu, hoặc các phần tài liệu có liên quan cao (bao gồm báo cáo y tế hoặc báo cáo tài chính) bằng các truy vấn thống nhất.
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
@misc{viPolyQwen_github_2024,
  author       = {Steve Nguyen Anh Nguyen and EraX AI and GMobile AI Team}, # Cập nhật tác giả
  title        = {viPolyQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization with Attention Pooling}, # Cập nhật tiêu đề
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/EraX-AI/viPolyQwen}} # Thay bằng URL repo cuối cùng
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