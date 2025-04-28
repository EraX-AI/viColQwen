import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    AutoModel,
)
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from transformers import PreTrainedModel, AutoConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColPaLiQwenEmbedder(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "qwen"

    def __init__(self, model_name_or_path, embed_dim=512, config=None):
        # Load config from the base model
        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # Save embed_dim in config for later restoration
        config.embed_dim = embed_dim
        
        # Initialize with the config
        super().__init__(config)
        
        # Set model dtype for consistency
        self.model_dtype = torch.bfloat16
        
        # Load base model
        self.qwen = AutoModel.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True, 
            torch_dtype=self.model_dtype, 
            attn_implementation="flash_attention_2",
            config=config
        )
        
        # Projection head - ensure same dtype as base model
        hidden_size = self.qwen.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, embed_dim, bias=False),
            nn.LayerNorm(embed_dim)
        )
        
        # Convert projection head to same dtype as base model
        self.proj = self.proj.to(dtype=self.model_dtype)
        
        # Store model type for Qwen-specific processing
        self.is_qwen_vl = "qwen" in model_name_or_path.lower() and "vl" in model_name_or_path.lower()
    
    def get_input_embeddings(self):
        """Required method for PreTrainedModel"""
        return self.qwen.get_input_embeddings()
    
    def set_input_embeddings(self, embeddings):
        """Required method for PreTrainedModel"""
        self.qwen.set_input_embeddings(embeddings)
    
    def _init_weights(self, module):
        """Initialize the weights - required for PreTrainedModel"""
        if isinstance(module, nn.Linear):
            # Initialize linear modules with small random values
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm modules
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model"""
        if hasattr(self.qwen, "gradient_checkpointing_enable"):
            # If the base model supports it, pass through the kwargs
            if gradient_checkpointing_kwargs is not None:
                self.qwen.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
            else:
                self.qwen.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for the base model")
        else:
            logger.warning("Base model does not support gradient checkpointing")
    
    def normalize_embeddings(self, embeddings):
        """L2 normalize embeddings"""
        return F.normalize(embeddings, p=2, dim=-1)
    
    def _qwen_vl_process(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        """Special processing for Qwen VL models"""
        # For Qwen VL, we only pass input_ids and attention_mask, not the pixel_values
        # The image data is already encoded into the input_ids by the processor
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the CLS token embedding and project
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        projected = self.proj(cls_embeddings)
        
        # Normalize and return
        return self.normalize_embeddings(projected)
    
    def _get_embeddings(self, pixel_values=None, input_ids=None, attention_mask=None, has_image=None):
        """Helper to process either text-only, image-only, or image+text inputs"""
        # Use special Qwen VL processing if applicable
        if self.is_qwen_vl:
            return self._qwen_vl_process(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # Standard processing for non-Qwen VL models
        batch_size = input_ids.shape[0]
        
        if has_image is None:
            # Default to assuming all examples have images if pixel_values is provided
            has_image = [True] * batch_size if pixel_values is not None else [False] * batch_size
        
        # Convert list to tensor for proper indexing with FSDP
        if isinstance(has_image, list):
            has_image = torch.tensor(has_image, device=input_ids.device)
        
        # Handle mixed batch (some examples with images, some without)
        outputs_list = []
        
        # Process examples with images
        image_indices = torch.where(has_image)[0]
        if len(image_indices) > 0 and pixel_values is not None:
            # Select relevant inputs for image examples
            image_pixels = pixel_values[image_indices]
            image_input_ids = input_ids[image_indices]
            image_attention_mask = attention_mask[image_indices]
            
            # Run image+text examples through the model
            image_outputs = self.qwen(
                pixel_values=image_pixels,
                input_ids=image_input_ids,
                attention_mask=image_attention_mask,
                return_dict=True
            )
            
            # Store outputs for these examples
            for idx, orig_idx in enumerate(image_indices.tolist()):
                outputs_list.append((orig_idx, image_outputs.last_hidden_state[idx, 0, :]))
        
        # Process text-only examples
        text_indices = torch.where(~has_image)[0]
        if len(text_indices) > 0:
            # Select relevant inputs for text-only examples
            text_input_ids = input_ids[text_indices]
            text_attention_mask = attention_mask[text_indices]
            
            # Run text-only examples through the model
            text_outputs = self.qwen(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                return_dict=True
            )
            
            # Store outputs for these examples
            for idx, orig_idx in enumerate(text_indices.tolist()):
                outputs_list.append((orig_idx, text_outputs.last_hidden_state[idx, 0, :]))
        
        # Re-order outputs to match input batch order
        outputs_list.sort(key=lambda x: x[0])
        cls_embeddings = torch.stack([emb for _, emb in outputs_list])
        
        # Project and normalize
        return self.normalize_embeddings(self.proj(cls_embeddings))
        
    def forward(self, 
                pixel_values_a=None, input_ids_a=None, attention_mask_a=None, has_image_a=None,
                pixel_values_b=None, input_ids_b=None, attention_mask_b=None, has_image_b=None,
                pixel_values=None, input_ids=None, attention_mask=None,  # Support standard args for eval
                **kwargs):  # Add **kwargs to catch any extra arguments like 'data_type'
        """
        Forward pass handling all input combinations:
        - For contrastive training: process both modalities (a and b)
        - For inference/eval: process single modality inputs
        - For single-modality input during eval: handled via standard args
        
        All inputs can be either text-only, image-only, or image+text
        """
        # During evaluation, the trainer might pass standard args instead of a/b modality format
        if (input_ids_a is None and input_ids is not None) or (pixel_values_a is None and pixel_values is not None):
            # Map standard args to modality a
            input_ids_a = input_ids
            attention_mask_a = attention_mask
            pixel_values_a = pixel_values
            has_image_a = kwargs.get('has_image', None) if 'has_image' in kwargs else None
        
        # Check if we're in contrastive training mode (both a and b provided)
        is_contrastive = input_ids_b is not None
        
        # Get embeddings for first modality (a)
        embeds_a = self._get_embeddings(
            pixel_values=pixel_values_a,
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            has_image=has_image_a
        )
        
        if is_contrastive:
            # Get embeddings for second modality (b)
            embeds_b = self._get_embeddings(
                pixel_values=pixel_values_b,
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                has_image=has_image_b
            )
            return embeds_a, embeds_b
        else:
            # Inference/eval mode - return single embeddings
            return embeds_a