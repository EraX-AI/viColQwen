# Introduce viColQwen
    - Base on Qwen2-VL 2B
    - Use constrastive learning to train with massive and diversed dataset
    - Create embedding for multi-modal image(s), text(s) or combined images + text(s)

# Dataset:
    4 kinds of datasets have been merged
    1. **Text-Text** pair with **simiality scores** (0.0 --> 1.0)
    2. Typical **LLM instruction** single-turn and multi-turn
    3. **OCR** single turn with 1 or many images (up to 5)
    4. **VQA** single turn and multi-turn with 1 or many images (up to 5)

    Massive **11 millions samples**, of which 5.6M are text pairs are, the rest are OCR/VQA in Vietnamese, some English and Chinese.

# Training strategy
    1. Large projected dimension (embedding) 1024
    2. Process different data type differently with mixed losses
    2. use InfoNCE or Adaptive NCE Loss (optional)

# How to use

- For image(s) **or** text(s) samples:
```
# 1. Import and initialize the evaluator
from colpali_evaluator import ColPaLiEvaluator

# Initialize the evaluator
evaluator = ColPaLiEvaluator(
    checkpoint_path="./qwen2vl2b_colpali_checkpoints/final/",
    embed_dim=1024,
    device="cuda"
)

# Use it as shown in the ColPali example
from PIL import Image

# Your inputs
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year's financial performance?",
]

# Get embeddings
image_embeddings = evaluator.get_image_embeddings(images)
query_embeddings = evaluator.get_query_embeddings(queries)

# Calculate scores
scores = evaluator.score(query_embeddings, image_embeddings)
```

- For **mixed images & texts** together:
```
def process_multimodal_input(self, text, image, image_base_path=""):
    """Process a combined text+image input for embedding"""
    # Load image if path is provided
    if isinstance(image, str):
        img_path = os.path.join(image_base_path, image)
        pil_image = Image.open(img_path).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Ensure text contains image tag
    if "<image>" not in text:
        text = f"<image>\n{text}"
    
    # Process with the processor
    inputs = self.processor(
        text=[text],  # Make sure text is a list
        images=[pil_image],  # Make sure image is a list
        padding="longest",
        return_tensors="pt"
    )
    
    # Add has_image flag
    inputs["has_image_a"] = torch.tensor([True])
    
    # Rename keys for model compatibility
    if "pixel_values" in inputs:
        inputs["pixel_values_a"] = inputs.pop("pixel_values")
    if "input_ids" in inputs:
        inputs["input_ids_a"] = inputs.pop("input_ids")
    if "attention_mask" in inputs:
        inputs["attention_mask_a"] = inputs.pop("attention_mask")
    
    return inputs

def get_multimodal_embedding(self, text, image, image_base_path=""):
    """Get embedding for a text+image combination"""
    inputs = self.process_multimodal_input(text, image, image_base_path)
    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
            for k, v in inputs.items()}
    
    with torch.no_grad():
        embedding = self.model(**inputs)
    
    return embedding
```
# This repo is being updated regularly, stay tuned!
