# Introduce viOmniQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization

## Abstract

Các hệ thống đa phương thức hiện đại thường gặp trở ngại bởi sự phức tạp của việc quản lý không gian embedding riêng biệt cho từng loại dữ liệu (văn bản, hình ảnh), dẫn đến sự phân mảnh trong biểu diễn, quy trình truy xuất phức tạp và hạn chế trong khả năng suy luận chéo phương thức. Chúng tôi giới thiệu **viOmniQwen**, một mô hình embedding đa phương thức tiên tiến, được thiết kế để tạo ra các biểu diễn **thống nhất, chiều cao** cho hình ảnh, văn bản và các kết hợp tùy ý của chúng trong một không gian vector duy nhất. Dựa trên kiến trúc vision-language mạnh mẽ **Qwen2-VL 2B**, viOmniQwen áp dụng một phương pháp học tương phản (contrastive learning) tinh vi, lấy cảm hứng từ ColPali nhưng được cải tiến đáng kể. Mô hình được huấn luyện trên một tập dữ liệu **đa dạng quy mô lớn (hơn 11 triệu mẫu)**, tích hợp một cách chiến lược các cặp tương đồng ngữ nghĩa văn bản-văn bản phức tạp (với điểm số liên tục), dữ liệu hướng dẫn phức tạp, tác vụ OCR đa hình ảnh và VQA đa hình ảnh. Điểm độc đáo cốt lõi nằm ở **chiến lược tối ưu hóa tổn thất hỗn hợp động (dynamic mixed-loss optimization)**, được dẫn hướng bởi các **tiền tố nhiệm vụ cụ thể (task-specific prefixes)**. Các tiền tố này (`<text_pair>`, `<instr>`, `<ocr>`, `<vqa_multi>`, `<vqa_single>`) được thêm vào đầu vào để báo hiệu loại dữ liệu và kích hoạt một **hàm loss tương ứng** (bao gồm InfoNCE, Triplet Loss, MSE, và tối đa hóa độ tương đồng cosine) được thiết kế riêng cho từng loại mẫu. Embedding cuối cùng được trích xuất bằng phương pháp **mean pooling**, thu giữ thông tin ngữ nghĩa và thị giác một cách toàn diện. Kết quả là các embedding 1024 chiều thể hiện sự hiểu biết ngữ nghĩa và hình ảnh sâu sắc, giúp đơn giản hóa và nâng cao đáng kể các ứng dụng như RAG đa phương thức, Graph RAG, tìm kiếm chéo phương thức và phân tích tài liệu phức tạp, đặc biệt trong bối cảnh ngôn ngữ Việt.

---

## Model Details

*   **Base Architecture:** `Qwen/Qwen2-VL-2B` - Vision-Language Model (VLM) nền tảng.
*   **Embedding Strategy:** Không gian Embedding Thống nhất qua Học Tương phản Động được Dẫn hướng bởi Tiền tố (Prefix-Guided Dynamic Contrastive Learning).
*   **Embedding Dimension:** `1024`.
*   **Pooling Strategy:** **Mean Pooling**. Embedding cuối cùng $`e \in \mathbb{R}^{1024}`$ được tính bằng cách lấy trung bình các trạng thái ẩn $`H = [h_1, h_2, ..., h_N] \in \mathbb{R}^{N \times d}`$ từ lớp cuối cùng, sau đó chuẩn hóa L2:
    ```math
    \bar{h} = \frac{1}{N} \sum_{i=1}^{N} h_i
    ```
    ```math
    e = \frac{\bar{h}}{\|\bar{h}\|_2}
    ```
    (Here, $`h_i`$ represents the hidden state of the $`i`$-th token, $`N`$ is the sequence length, $`\bar{h}`$ is the mean pooled vector, and $`\|\cdot\|_2`$ denotes the L2 norm).
*   **Input Representation:** Dữ liệu đầu vào (văn bản, hình ảnh PIL) được xử lý bởi bộ xử lý của Qwen-VL. Hình ảnh được biểu diễn bằng token `<image>`. Quan trọng hơn, *trước* phần nội dung văn bản chính, một **tiền tố nhiệm vụ cụ thể** được thêm vào để báo hiệu loại dữ liệu:
    *   `<text_pair>`: Cho cặp văn bản với điểm tương đồng.
    *   `<instr>`: Cho dữ liệu hướng dẫn (instruction-response).
    *   `<ocr>`: Cho dữ liệu OCR/OCQ.
    *   `<vqa_multi>`: Cho VQA đa lượt.
    *   `<vqa_single>`: Cho VQA đơn lượt.
*   **Output:** Một vector dày `1024-d` duy nhất $`e`$ biểu diễn nội dung ngữ nghĩa và/hoặc thị giác của đầu vào.

---

## Training Paradigm

Sức mạnh của viOmniQwen đến từ sự kết hợp giữa tập dữ liệu đa dạng và chiến lược tối ưu hóa độc đáo:

1.  **Heterogeneous Dataset (Hơn 11 Triệu Mẫu):** Tích hợp 4 loại dữ liệu chính, liên kết với các tiền tố:
    *   **Text-Text Semantic Similarity (`<text_pair>`, ~5.6M):** Cặp $(t_a, t_b)$ với điểm số $`s \in [0, 1]`$.
    *   **Instruction Following (`<instr>`, ~0.6M):** Cặp (instruction $`i`$, response $`r`$).
    *   **Multi-Image OCR/OCQ (`<ocr>`, ~2.5M):** Bộ ba $(\{\text{image(s)}\}_q, \text{query } q, \text{answer } a)$.
    *   **Multi-Image VQA (`<vqa_single>`, `<vqa_multi>`, ~2.5M):** Bộ ba $(\{\text{image(s)}\}_q, \text{question } q, \text{answer } a)$.
    Tập trung vào tiếng Việt (vi), cùng với tiếng Anh (en) và Trung (zh).

2.  **Prefix-Guided Dynamic Mixed-Loss Optimization:**
    *   Mỗi mẫu trong batch được gắn tiền tố nhiệm vụ tương ứng.
    *   Dựa trên tiền tố, một hàm loss cụ thể $\mathcal{L}_{\text{prefix}}$ được **kích hoạt và áp dụng** cho cặp embedding $(e_a, e_b)$ của mẫu đó.
    *   Tổn thất tổng của batch $\mathcal{L}_{\text{batch}}$ là trung bình của các tổn thất riêng lẻ cho từng mẫu $`i`$ trong batch $`B`$:
        ```math
        \mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}_{\text{prefix}(i)}(e_{a,i}, e_{b,i}, \text{params}_i)
        ```
    *   **Các hàm loss được sử dụng:**
        *   **Cho `<text_pair>`:** Kết hợp InfoNCE đối xứng và MSE Regression (so khớp điểm tương đồng dự đoán $`\hat{s}`$ với điểm thật $`s_{\text{true}}`$).
        *   **Cho `<instr>`:** Kết hợp InfoNCE đối xứng và Direct Cosine Similarity Maximization (khuyến khích $`e_a \cdot e_b`$ tiến tới 1).
        *   **Cho `<ocr>`, `<vqa_single>`, `<vqa_multi>`:** Kết hợp InfoNCE đối xứng và Triplet Margin Loss (đảm bảo khoảng cách $`m`$ giữa cặp dương và âm khó nhất, với margin có thể điều chỉnh cho multi-turn).

---

## Key Features & Advantages

*   ✅ **Unified Multimodal Embedding:** Không gian vector đơn nhất cho mọi loại đầu vào.
*   ✅ **Prefix-Guided Training:** Cho phép mô hình chuyên biệt hóa xử lý từng loại dữ liệu.
*   ✅ **Simplified Multimodal RAG/Search:** Truy vấn đơn giản trên một chỉ mục vector duy nhất.
*   ✅ **Enhanced Cross-Modal Understanding:** Huấn luyện phối hợp thúc đẩy sự hiểu biết sâu sắc.
*   ✅ **High-Dimensional Nuance:** Embedding 1024-d nắm bắt chi tiết tinh vi.
*   ✅ **Multi-Image Aware:** Xử lý tự nhiên ngữ cảnh nhiều hình ảnh.
*   ✅ **Robust Performance:** Dữ liệu và loss đa dạng tạo ra embedding linh hoạt.
*   ✅ **Strong Vietnamese & Multilingual Focus:** Tối ưu cho tiếng Việt, hỗ trợ tốt tiếng Anh/Trung.
*   ✅ **Foundation for Advanced AI:** Nền tảng lý tưởng cho AI đa phương thức.

---

## How to Use (Conceptual Example)

```python
import torch
from PIL import Image
# Assume viOmniQwenEmbedder class available after release
# from viOmniQwen_embedder import viOmniQwenEmbedder

# embedder = viOmniQwenEmbedder(checkpoint_path="./path/to/viOmniQwen/", device="cuda")

# --- Example: VQA Single Turn ---
# Note: The embedder's encode method should handle prefix internally,
# or you might need to prepend it manually if using the base model directly.
prefix_vqa = "<vqa_single>"
text_input = "What color is the object on the left?"
image_input = Image.open("image.jpg").convert("RGB")

# Conceptual encoding call
# mixed_embedding = embedder.encode(text=f"{prefix_vqa} {text_input}", images=[image_input])
# print(mixed_embedding.shape) # torch.Size([1, 1024])

# --- Example: Text Similarity ---
prefix_sim = "<text_pair>"
text_a = "The cat sat on the mat."
text_b = "A feline rested upon the rug."

# text_a_embedding = embedder.encode(text=f"{prefix_sim} {text_a}")
# text_b_embedding = embedder.encode(text=f"{prefix_sim} {text_b}")

# similarity = torch.nn.functional.cosine_similarity(text_a_embedding, text_b_embedding)
# print(similarity)
```

---

## Potential Applications

*   **Multimodal RAG:** Truy xuất ngữ cảnh đa phương thức.
*   **Graph RAG:** Xây dựng đồ thị tri thức đa phương thức.
*   **Cross-Modal Retrieval:** Tìm kiếm linh hoạt giữa các phương thức.
*   **Document Intelligence:** Phân tích tài liệu phức tạp.
*   **Contextual Visual Search:** Tìm kiếm hình ảnh theo ngữ cảnh.

---

## Development Status & Future Work

*   Đang trong quá trình phát triển tích cực. Checkpoints, code đánh giá, benchmarks, ví dụ sử dụng chi tiết sẽ sớm được phát hành.
*   Công việc đang diễn ra: Benchmarking toàn diện, khám phá mô hình lớn hơn, tích hợp phương thức khác.

---

## License

*   Chi tiết giấy phép sẽ được công bố khi phát hành. Có tùy chọn giấy phép thương mại. Liên hệ: **nguyen@hatto.com**.

---

## Citation

```bibtex
@misc{viOmniQwen_github_2024,
  author       = {Steve Nguyen Anh Nguyen and the EraX AI Team},
  title        = {viOmniQwen: Unified Multimodal Embeddings via Prefix-Guided Dynamic Loss Optimization},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/EraX-AI/viOmniQwen}} % Replace with final URL
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