import torch
from PIL import Image
import os
from transformers import AutoProcessor
from model import ColPaLiQwenEmbedder

import warnings
import transformers

# Suppress the specific warning about weights not being initialized
warnings.filterwarnings("ignore", message="Some weights of .* were not initialized")

# Alternatively, you can silence just the transformers warnings
transformers.logging.set_verbosity_error()  # This will show only errors, not warnings

class ColPaLiEvaluator:
    def __init__(self, checkpoint_path, embed_dim=1024, device=None):
        """Initialize the evaluator"""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        
        original_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
    
        try:
            # Load the model
            self.model = ColPaLiQwenEmbedder(checkpoint_path, embed_dim=embed_dim)
        finally:
            # Restore original verbosity
            transformers.logging.set_verbosity(original_verbosity)
        
        # Load weights from checkpoint
        self._load_weights(checkpoint_path)
        
        # Set model to eval mode
        self.model.to(self.device)
        self.model.eval()
    
    def _load_weights(self, checkpoint_path):
        
        """Load weights from checkpoint"""
        # Find model file
        model_files = []
        for file in os.listdir(checkpoint_path):
            if file.endswith(".safetensors") and file.startswith("model-"):
                model_files.append(os.path.join(checkpoint_path, file))
        
        if not model_files:
            raise ValueError(f"No model files found in {checkpoint_path}")
        
        # Use safetensors to load
        from safetensors.torch import load_file
        
        # Load and merge weights
        state_dict = {}
        for file in model_files:
            print(f"Loading weights from {file}")
            weights = load_file(file)
            state_dict.update(weights)
        
        # Load weights into model
        self.model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully")
    
    def process_images(self, images, image_base_path=""):
        """Process images for embedding"""
        # Convert paths to PIL images if needed
        pil_images = []
        for img in images:
            if isinstance(img, str):
                img_path = os.path.join(image_base_path, img)
                pil_images.append(Image.open(img_path).convert('RGB'))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        # For Qwen VL we need both images and text with <image> tags
        text_inputs = ["<image>"] * len(pil_images)
        
        # Process with the processor
        inputs = self.processor(text=text_inputs, images=pil_images, return_tensors="pt")
        
        # Add has_image flag
        inputs["has_image_a"] = torch.tensor([True] * len(pil_images))
        
        # Rename keys for model compatibility
        if "pixel_values" in inputs:
            inputs["pixel_values_a"] = inputs.pop("pixel_values")
        if "input_ids" in inputs:
            inputs["input_ids_a"] = inputs.pop("input_ids")
        if "attention_mask" in inputs:
            inputs["attention_mask_a"] = inputs.pop("attention_mask")
        
        return inputs
    
    def process_queries(self, queries):
        """Process text queries for embedding"""
        # Process with the processor's tokenizer directly for text-only
        inputs = self.processor.tokenizer(
            queries,
            padding="longest",  # Add padding to handle different lengths
            truncation=True,
            return_tensors="pt"
        )
        
        # Add has_image flag
        inputs["has_image_a"] = torch.tensor([False] * len(queries))
        
        # Rename keys for model compatibility
        if "input_ids" in inputs:
            inputs["input_ids_a"] = inputs.pop("input_ids")
        if "attention_mask" in inputs:
            inputs["attention_mask_a"] = inputs.pop("attention_mask")
        
        return inputs
    
    def get_image_embeddings(self, images, image_base_path="", batch_size=16):
        """Get embeddings for images"""
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            inputs = self.process_images(batch_images, image_base_path)
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in inputs.items()}
            
            with torch.no_grad():
                embeddings = self.model(**inputs)
                
            all_embeddings.append(embeddings.cpu())
            
        return torch.cat(all_embeddings, dim=0)
    
    def get_query_embeddings(self, queries, batch_size=16):
        """Get embeddings for text queries"""
        all_embeddings = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            inputs = self.process_queries(batch_queries)
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in inputs.items()}
            
            with torch.no_grad():
                embeddings = self.model(**inputs)
                
            all_embeddings.append(embeddings.cpu())
            
        return torch.cat(all_embeddings, dim=0)
    
    def score(self, query_embeddings, image_embeddings):
        """Calculate similarity scores between queries and images"""
        return torch.matmul(query_embeddings, image_embeddings.t())