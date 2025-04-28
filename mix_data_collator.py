import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    AutoModel,
)
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import random
from PIL import Image
import copy
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_image_tags(text):
    """Count the number of <image> tags in the text"""
    if not text:
        return 0
    return len(re.findall(r'<image>', text))

def add_missing_image_tags(text, num_images):
    """Add necessary image tags if they don't already exist"""
    if text is None:
        text = ""
    existing_tags = count_image_tags(text)
    if existing_tags >= num_images:
        return text  # Already has enough tags
    
    # Add missing tags at the beginning
    missing_tags = num_images - existing_tags
    img_references = "\n".join(["<image>"] * missing_tags)
    return f"{img_references}\n{text}"

def load_images_from_example(example, prefix=""):
    """Extract and load images from an example in various formats"""
    loaded_images = []
    
    # Check for images list - THE PRIMARY WAY IMAGES SHOULD BE STORED
    if "images" in example and example["images"]:
        for img_path in example["images"]:
            try:
                full_path = os.path.join(prefix, img_path) if isinstance(img_path, str) else img_path
                if isinstance(img_path, str) and os.path.exists(full_path):
                    img = Image.open(full_path).convert('RGB')
                    loaded_images.append(img)
                elif isinstance(img_path, Image.Image):
                    loaded_images.append(img_path)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
    
    # Fallback: Check for single image (not recommended, but supported for compatibility)
    elif "image" in example and example["image"] is not None:
        img_path = example["image"]
        try:
            full_path = os.path.join(prefix, img_path) if isinstance(img_path, str) else img_path
            if isinstance(img_path, str) and os.path.exists(full_path):
                img = Image.open(full_path).convert('RGB')
                loaded_images.append(img)
            elif isinstance(img_path, Image.Image):
                loaded_images.append(img_path)
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            
    return loaded_images

def format_chat_template(processor, messages):
    """
    Format conversation messages using the model's chat template
    """
    if not messages or not isinstance(messages, list):
        return ""
        
    if hasattr(processor.tokenizer, "apply_chat_template"):
        try:
            # Ensure all messages have required fields
            valid_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    valid_messages.append(msg)
            
            if not valid_messages:
                return ""
                
            return processor.tokenizer.apply_chat_template(
                valid_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"Error applying chat template: {e}")
            # Fallback to simple concatenation
            return "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in messages])
    else:
        # Fallback for tokenizers without chat template
        return "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in messages])

def is_qwen_vl_processor(processor):
    """Check if this is a Qwen VL processor"""
    return (hasattr(processor, "__class__") and 
            "qwen" in processor.__class__.__name__.lower() and 
            "vl" in processor.__class__.__name__.lower())

def process_qwen_vl_images(examples, processor, image_base_path=""):
    """Process examples using Qwen-specific format"""
    try:
        # Import the process_vision_info function
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            logger.error("Failed to import process_vision_info from qwen_vl_utils. Make sure the file is available.")
            return None
        
        # Initialize batch components
        batch = {
            "data_type": [],
            "similarity_scores": []
        }
        
        # Process each example into Qwen's conversation format
        conversations_a = []
        conversations_b = []
        
        for example in examples:
            if example is None:
                continue
                
            # Get data type and similarity score
            data_type = example.get("data_type", "default")
            similarity_score = example.get("similarity_score", None)
            
            batch["data_type"].append(data_type)
            batch["similarity_scores"].append(similarity_score)
            
            # Load images
            loaded_images = load_images_from_example(example, prefix=image_base_path)
            
            # Format modality A (input side)
            conversation_a = []
            user_a = {"role": "user", "content": []}
            
            # Add images if available
            if loaded_images:
                for img in loaded_images:
                    user_a["content"].append({
                        "type": "image_url",
                        "image": img
                    })
            
            # Add text content
            input_text = example.get("input", example.get("text1", example.get("question", "")))
            user_a["content"].append({
                "type": "text",
                "text": input_text
            })
            
            conversation_a.append(user_a)
            conversations_a.append(conversation_a)
            
            # Format modality B (output side) - always text only
            conversation_b = []
            user_b = {"role": "user", "content": []}
            
            output_text = example.get("output", example.get("text2", example.get("answer", "")))
            user_b["content"].append({
                "type": "text",
                "text": output_text
            })
            
            conversation_b.append(user_b)
            conversations_b.append(conversation_b)
            
        # Process both modalities using Qwen's processor
        # Process modality A
        image_a, video_a = process_vision_info(conversations_a)
        texts_a = [processor.apply_chat_template(c, tokenize=False) for c in conversations_a]
        
        inputs_a = processor(
            text=texts_a,
            images=image_a,
            videos=video_a,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )
        
        # Process modality B
        texts_b = [processor.apply_chat_template(c, tokenize=False) for c in conversations_b]
        
        inputs_b = processor(
            text=texts_b,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )
        
        # Determine which examples have images
        has_image_a = []
        for conv in conversations_a:
            has_img = False
            for content in conv[0]["content"]:
                if content["type"] == "image_url":
                    has_img = True
                    break
            has_image_a.append(has_img)
        
        # Fill batch with results
        batch["input_ids_a"] = inputs_a["input_ids"]
        batch["attention_mask_a"] = inputs_a["attention_mask"]
        batch["has_image_a"] = torch.tensor(has_image_a, dtype=torch.bool)
        if "pixel_values" in inputs_a:
            batch["pixel_values_a"] = inputs_a["pixel_values"]
            
        batch["input_ids_b"] = inputs_b["input_ids"]
        batch["attention_mask_b"] = inputs_b["attention_mask"]
        batch["has_image_b"] = torch.tensor([False] * len(conversations_b), dtype=torch.bool)
        
        # Handle similarity scores
        if any(batch["similarity_scores"]):
            valid_scores = [s for s in batch["similarity_scores"] if s is not None]
            if valid_scores:
                scores_tensor = torch.zeros(len(batch["similarity_scores"]), dtype=torch.float)
                for i, score in enumerate(batch["similarity_scores"]):
                    if score is not None:
                        scores_tensor[i] = score
                batch["similarity_scores"] = scores_tensor
        
        # Convert data types to strings
        if "data_type" in batch:
            batch["data_type"] = [str(dt) if dt is not None else "unknown" for dt in batch["data_type"]]
        
        return batch
    
    except Exception as e:
        logger.error(f"Error in process_qwen_vl_images: {e}")
        return None

def process_modality(processor, examples_with_images, examples_without_images, images, texts, batch_data_types, max_length=8192):
    """Process a single modality (either A or B) with mixed image and text examples"""
    # Initialize result containers
    batch = {}
    has_image = [False] * len(examples_with_images + examples_without_images)
    
    # Set has_image flags
    for idx in examples_with_images:
        has_image[idx] = True
    
    # Process texts according to their data type
    processed_texts = []
    for i, text in enumerate(texts):
        data_type = batch_data_types[i] if i < len(batch_data_types) else "unknown"
        
        if text is None:
            processed_texts.append("")
            continue
            
        if isinstance(text, list) and data_type in ["instruction", "vqa_multi", "vqa_single"]:
            # For instruction and VQA types that may contain conversation format
            processed_texts.append(format_chat_template(processor, text))
        elif isinstance(text, list):
            # Other list types (not chat format)
            processed_texts.append(" ".join(str(item) for item in text if item is not None))
        else:
            # String or other types
            processed_texts.append(str(text))
    
    # Handle text-only case
    if not examples_with_images:
        try:
            text_inputs = processor.tokenizer(
                processed_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            return {
                "input_ids": text_inputs.input_ids,
                "attention_mask": text_inputs.attention_mask,
                "has_image": torch.tensor(has_image, dtype=torch.bool),
                "pixel_values": None
            }
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            # Create empty tensors as fallback
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return {
                "input_ids": torch.zeros((len(processed_texts), 10), dtype=torch.long, device=device),
                "attention_mask": torch.zeros((len(processed_texts), 10), dtype=torch.long, device=device),
                "has_image": torch.tensor(has_image, dtype=torch.bool, device=device),
                "pixel_values": None
            }
    
    try:
        # Flatten image inputs
        flat_images = []
        image_to_example_map = []  # Maps flattened image index to original example index
        
        for i in examples_with_images:
            image_list = images[i]
            if isinstance(image_list, list) and image_list:
                for img in image_list:
                    if img is not None:
                        flat_images.append(img)
                        image_to_example_map.append(i)
            elif image_list is not None:
                flat_images.append(image_list)
                image_to_example_map.append(i)
        
        # Process text-only examples
        all_examples_inputs = []
        for i in examples_without_images:
            try:
                if i >= len(processed_texts):
                    # Handle out of range indices
                    text_inp = processor.tokenizer(
                        "",
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt"
                    )
                else:
                    text_inp = processor.tokenizer(
                        processed_texts[i],
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt"
                    )
                all_examples_inputs.append({
                    "input_ids": text_inp.input_ids.squeeze(0),
                    "attention_mask": text_inp.attention_mask.squeeze(0)
                })
            except Exception as e:
                logger.warning(f"Error tokenizing text at index {i}: {e}")
                # Create empty tensors for this example
                device = "cuda" if torch.cuda.is_available() else "cpu"
                all_examples_inputs.append({
                    "input_ids": torch.zeros((10), dtype=torch.long, device=device),
                    "attention_mask": torch.zeros((10), dtype=torch.long, device=device)
                })
        
        # Process images with their texts
        if flat_images:
            try:
                image_texts = []
                for i in range(len(flat_images)):
                    if image_to_example_map[i] < len(processed_texts):
                        image_texts.append(processed_texts[image_to_example_map[i]])
                    else:
                        image_texts.append("")
                
                # Process with processor
                image_inputs = processor(
                    images=flat_images,
                    text=image_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                
                # Get maximum sequence length
                max_seq_length = max(
                    image_inputs.input_ids.shape[1] if hasattr(image_inputs, 'input_ids') else 0,
                    max([inp["input_ids"].shape[0] for inp in all_examples_inputs]) if all_examples_inputs else 0
                )
                
                # Create tensors for all examples
                total_examples = len(examples_with_images) + len(examples_without_images)
                device = image_inputs.pixel_values.device
                all_input_ids = torch.zeros((total_examples, max_seq_length), 
                                           dtype=torch.long, device=device)
                all_attention_mask = torch.zeros((total_examples, max_seq_length), 
                                                dtype=torch.long, device=device)
                
                # Fill in text-only examples
                for i, text_example_idx in enumerate(examples_without_images):
                    if i < len(all_examples_inputs):
                        inp = all_examples_inputs[i]
                        length = min(inp["input_ids"].shape[0], max_seq_length)
                        all_input_ids[text_example_idx, :length] = inp["input_ids"][:length]
                        all_attention_mask[text_example_idx, :length] = inp["attention_mask"][:length]
                
                # Map flat image inputs back to their examples
                image_idx_map = {}
                for img_idx, example_idx in enumerate(image_to_example_map):
                    if example_idx not in image_idx_map:
                        image_idx_map[example_idx] = img_idx
                
                # Fill in image examples (use the first image for each example if multiple)
                for example_idx, img_idx in image_idx_map.items():
                    if hasattr(image_inputs, 'input_ids'):
                        length = min(image_inputs.input_ids.shape[1], max_seq_length)
                        all_input_ids[example_idx, :length] = image_inputs.input_ids[img_idx, :length]
                        all_attention_mask[example_idx, :length] = image_inputs.attention_mask[img_idx, :length]
                
                return {
                    "input_ids": all_input_ids,
                    "attention_mask": all_attention_mask,
                    "has_image": torch.tensor(has_image, dtype=torch.bool, device=device),
                    "pixel_values": image_inputs.pixel_values
                }
            except Exception as e:
                logger.warning(f"Error processing images: {e}")
                # Fall through to text-only fallback
    except Exception as e:
        logger.warning(f"Failed in process_modality: {e}")
    
    # Fallback to text-only processing
    try:
        text_inputs = processor.tokenizer(
            processed_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "has_image": torch.tensor([False] * len(processed_texts), dtype=torch.bool),
            "pixel_values": None
        }
    except Exception as e:
        logger.error(f"Final fallback error: {e}")
        # Create empty tensors as absolute last resort
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = len(texts)
        return {
            "input_ids": torch.zeros((batch_size, 10), dtype=torch.long, device=device),
            "attention_mask": torch.zeros((batch_size, 10), dtype=torch.long, device=device),
            "has_image": torch.tensor([False] * batch_size, dtype=torch.bool, device=device),
            "pixel_values": None
        }
        
@dataclass
class MixedBatchCollator:
    """
    Collator that handles mixed data types within the same batch
    - Handles text-text pairs, instruction tuning data, OCQ/OCR, and VQA multi-turn
    - Properly manages different modalities and data types
    - Supports multiple images per sample
    """
    processor: Any  # Processor
    max_length: int = 8192
    image_base_path: str = ""  # Path to image files
    
    def __call__(self, examples):
        # Check if we're dealing with a Qwen VL processor
        if is_qwen_vl_processor(self.processor):
            #logger.info("Using Qwen VL-specific processing")
            qwen_results = process_qwen_vl_images(examples, self.processor, self.image_base_path)
            if qwen_results is not None:
                return qwen_results
            logger.warning("Qwen VL-specific processing failed, falling back to standard processing")
        
        # Initialize batch components
        batch = {
            "data_type": [],
            "similarity_scores": []
        }
        
        # First part of pairs (could be image, text, or image+text)
        images_a = []
        texts_a = []
        has_image_a_indices = []  # Indices of examples with images in modality A
        
        # Second part of pairs (could be image, text, or image+text)
        images_b = []
        texts_b = []
        has_image_b_indices = []  # Indices of examples with images in modality B
        
        # Process each example based on its data type
        for example_idx, example in enumerate(examples):
            # Skip None examples
            if example is None:
                logger.warning(f"Skipping None example at index {example_idx}")
                # Add placeholders
                batch["data_type"].append("none")
                batch["similarity_scores"].append(None)
                images_a.append(None)
                texts_a.append("")
                images_b.append(None)
                texts_b.append("")
                continue
                
            # Get data type and similarity score
            data_type = example.get("data_type", "default")
            similarity_score = example.get("similarity_score", None)
            
            batch["data_type"].append(data_type)
            batch["similarity_scores"].append(similarity_score)
            
            # Process by data type
            if data_type == "contrastive_with_score":
                # KEY TYPE 1: Simple text-text pairs with similarity score
                input_text = example.get("input", example.get("text1", example.get("question", "")))
                output_text = example.get("output", example.get("text2", example.get("answer", "")))
                
                images_a.append(None)
                texts_a.append(input_text)
                
                images_b.append(None)
                texts_b.append(output_text)
                
            elif data_type == "instruction":
                # KEY TYPE 3: Instruction data in ChatML format
                # For instruction tuning, we need to handle the full conversation history
                input_data = example.get("input", [])
                
                # Keep the original conversation format for proper chat template formatting
                if isinstance(input_data, list) and all(isinstance(msg, dict) for msg in input_data):
                    # This is already in the proper format - a list of message dicts
                    conversation = input_data
                elif isinstance(input_data, list):
                    # Convert list items to user messages
                    conversation = [{"role": "user", "content": str(item)} for item in input_data]
                elif isinstance(input_data, dict) and "role" in input_data and "content" in input_data:
                    # Single message dict
                    conversation = [input_data]
                else:
                    # Convert simple string or other type to a user message
                    conversation = [{"role": "user", "content": str(input_data) if input_data is not None else ""}]
                
                # Get the output/response
                output_text = example.get("output", "")
                if output_text:
                    # Add assistant response to the conversation if present
                    conversation.append({"role": "assistant", "content": str(output_text)})
                
                images_a.append(None)
                texts_a.append(conversation)  # Pass the full conversation
                
                images_b.append(None)
                texts_b.append("")  # No separate text for output (already in conversation)
                
            elif data_type == "ocr" or data_type == "ocq":
                # KEY TYPE 2: OCR with plain text question/answer
                # Load images - support multiple images
                loaded_images = load_images_from_example(example, prefix=self.image_base_path)
                
                # Get question and answer
                question = example.get("question", "")
                answer = example.get("answer", "")
                
                # Add image tags to question if needed
                if loaded_images:
                    question = add_missing_image_tags(question, len(loaded_images))
                    has_image_a_indices.append(example_idx)
                
                images_a.append(loaded_images if loaded_images else None)
                texts_a.append(question)
                
                images_b.append(None)
                texts_b.append(answer)
                
            elif data_type == "vqa_multi" or data_type == "vqa_single":
                # KEY TYPE 3: VQA in ChatML format with images
                # Load images - support multiple images
                loaded_images = load_images_from_example(example, prefix=self.image_base_path)
                
                # Process input data as conversation
                input_data = example.get("input", [])
                
                # Keep original conversation format for chat template
                if isinstance(input_data, list) and all(isinstance(msg, dict) for msg in input_data):
                    # This is already in the proper format - a list of message dicts
                    conversation = input_data
                elif isinstance(input_data, list):
                    # Convert list items to user messages
                    conversation = [{"role": "user", "content": str(item)} for item in input_data]
                elif isinstance(input_data, dict) and "role" in input_data and "content" in input_data:
                    # Single message dict
                    conversation = [input_data]
                else:
                    # Convert simple string to a user message
                    conversation = [{"role": "user", "content": str(input_data) if input_data is not None else ""}]
                
                # Get the output/response
                output_text = example.get("output", example.get("answer", ""))
                if output_text:
                    # Add assistant response to the conversation
                    conversation.append({"role": "assistant", "content": str(output_text)})
                
                # Add image tags to the first user message if needed
                if loaded_images and conversation:
                    for i, msg in enumerate(conversation):
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            msg["content"] = add_missing_image_tags(content, len(loaded_images))
                            break
                
                if loaded_images:
                    has_image_a_indices.append(example_idx)
                
                images_a.append(loaded_images if loaded_images else None)
                texts_a.append(conversation)  # Pass the full conversation
                
                images_b.append(None)
                texts_b.append("")  # No separate text for output (already in conversation)
                
            else:
                # Default/unknown data type - try to handle based on available fields
                logger.warning(f"Unknown data type '{data_type}' at index {example_idx}")
                
                # Load any images
                loaded_images = load_images_from_example(example, prefix=self.image_base_path)
                
                # Check for question/answer fields
                has_question = "question" in example and example["question"] is not None
                has_answer = "answer" in example and example["answer"] is not None
                
                if loaded_images and has_question:
                    # Image + question -> answer format
                    question = example["question"]
                    question = add_missing_image_tags(question, len(loaded_images))
                    has_image_a_indices.append(example_idx)
                    
                    images_a.append(loaded_images)
                    texts_a.append(question)
                    
                    if has_answer:
                        images_b.append(None)
                        texts_b.append(example["answer"])
                    else:
                        images_b.append(None)
                        texts_b.append("")
                elif has_question:
                    # Text-only question -> answer
                    images_a.append(None)
                    texts_a.append(example["question"])
                    
                    if has_answer:
                        images_b.append(None)
                        texts_b.append(example["answer"])
                    else:
                        images_b.append(None)
                        texts_b.append("")
                else:
                    # No clear structure - use input/output if available
                    input_text = example.get("input", "")
                    output_text = example.get("output", "")
                    
                    if loaded_images:
                        input_text = add_missing_image_tags(str(input_text) if input_text is not None else "", 
                                                       len(loaded_images))
                        has_image_a_indices.append(example_idx)
                        images_a.append(loaded_images)
                    else:
                        images_a.append(None)
                        
                    texts_a.append(input_text)
                    images_b.append(None)
                    texts_b.append(output_text)
        
        # Create list of examples without images
        no_image_a_indices = [i for i in range(len(examples)) if i not in has_image_a_indices]
        no_image_b_indices = [i for i in range(len(examples)) if i not in has_image_b_indices]
        
        # Process first modality (A)
        modality_a_results = process_modality(
            self.processor, 
            has_image_a_indices, 
            no_image_a_indices, 
            images_a, 
            texts_a,
            batch["data_type"],
            self.max_length
        )
        
        # Process second modality (B)
        modality_b_results = process_modality(
            self.processor, 
            has_image_b_indices, 
            no_image_b_indices, 
            images_b, 
            texts_b,
            batch["data_type"],
            self.max_length
        )
        
        # Fill batch with results
        batch["input_ids_a"] = modality_a_results["input_ids"]
        batch["attention_mask_a"] = modality_a_results["attention_mask"]
        batch["has_image_a"] = modality_a_results["has_image"]
        if modality_a_results["pixel_values"] is not None:
            batch["pixel_values_a"] = modality_a_results["pixel_values"]
            
        batch["input_ids_b"] = modality_b_results["input_ids"]
        batch["attention_mask_b"] = modality_b_results["attention_mask"]
        batch["has_image_b"] = modality_b_results["has_image"]
        if modality_b_results["pixel_values"] is not None:
            batch["pixel_values_b"] = modality_b_results["pixel_values"]
        
        # Convert similarity scores to tensor if needed
        if "similarity_scores" in batch and any(batch["similarity_scores"]):
            # Filter out None values
            valid_scores = [s for s in batch["similarity_scores"] if s is not None]
            if valid_scores:
                # Create a tensor with zeros, then fill in the valid scores
                scores_tensor = torch.zeros(len(batch["similarity_scores"]), dtype=torch.float)
                for i, score in enumerate(batch["similarity_scores"]):
                    if score is not None:
                        scores_tensor[i] = score
                batch["similarity_scores"] = scores_tensor
            else:
                # All scores are None, remove from batch
                batch.pop("similarity_scores")
                
        # Convert data_type to a list of strings
        if "data_type" in batch:
            batch["data_type"] = [str(dt) if dt is not None else "unknown" for dt in batch["data_type"]]
        
        return batch