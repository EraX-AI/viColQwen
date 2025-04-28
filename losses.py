import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def adaptive_infonce_loss(
    embeds_a,
    embeds_b,
    base_temp=0.07,
    alpha=0.5,
    data_type=None,
    similarity_scores=None,
    margin=0.2,
):
    """
    Adaptive InfoNCE loss that dynamically adjusts temperature based on example difficulty.
    Fixed to avoid in-place operations that cause backward pass issues.
    """
    # Calculate raw similarity matrix 
    raw_similarity = torch.matmul(embeds_a, embeds_b.t())
    batch_size = embeds_a.shape[0]
    device = embeds_a.device
    
    # Extract positive pair similarities (diagonal)
    pos_sim = torch.diag(raw_similarity)
    
    # Find hardest negative for each example
    hardest_neg_sim = raw_similarity.clone()
    # Create a mask for the diagonal (without in-place operation)
    diag_mask = torch.ones_like(hardest_neg_sim, device=device)
    diag_mask.diagonal().fill_(-float('inf'))
    # Apply mask instead of in-place modification
    hardest_neg_sim = hardest_neg_sim + diag_mask
    hardest_neg_sim = torch.max(hardest_neg_sim, dim=1)[0]
    
    # Calculate gap between positive and hardest negative
    sim_gap = pos_sim - hardest_neg_sim
    
    # Calculate adaptive temperature (smaller gap → lower temperature)
    adaptive_temp = base_temp * (1.0 - alpha * torch.sigmoid(-sim_gap))
    
    # Apply adaptive temperature without in-place operations
    # Create a new tensor for the scaled similarity matrix
    similarity_matrix = torch.zeros_like(raw_similarity, device=device)
    for i in range(batch_size):
        # Broadcast division instead of in-place modification
        similarity_matrix[i] = raw_similarity[i] / adaptive_temp[i]
    
    # Calculate standard InfoNCE loss with adaptive temperature
    labels = torch.arange(batch_size, device=device)
    loss_a2b = F.cross_entropy(similarity_matrix, labels)
    loss_b2a = F.cross_entropy(similarity_matrix.t(), labels)
    
    return (loss_a2b + loss_b2a) / 2.0
    
def adaptive_triplet_loss(
    embeds_a,
    embeds_b, 
    base_temp=0.07,
    alpha=0.5,
    margin=0.2,
    margin_multiplier=1.0,
    device=None
):
    """
    Adaptive triplet loss with InfoNCE, combining adaptive temperature with triplet margin.
    Fixed to avoid in-place operations.
    """
    batch_size = embeds_a.shape[0]
    device = embeds_a.device if device is None else device
    
    # Calculate raw similarity matrix
    raw_similarity = torch.matmul(embeds_a, embeds_b.t())
    
    # Get positive similarities (diagonal)
    pos_sim = torch.diag(raw_similarity)
    
    # Find hardest negative for each example (without in-place operations)
    hardest_neg_sim = raw_similarity.clone()
    diag_mask = torch.ones_like(hardest_neg_sim, device=device)
    diag_mask.diagonal().fill_(-float('inf'))
    hardest_neg_sim = hardest_neg_sim + diag_mask
    hardest_neg_sim = torch.max(hardest_neg_sim, dim=1)[0]
    
    # Calculate gap between positive and hardest negative
    sim_gap = pos_sim - hardest_neg_sim
    
    # Calculate adaptive temperature (smaller gap → lower temperature)
    adaptive_temp = base_temp * (1.0 - alpha * torch.sigmoid(-sim_gap))
    
    # Apply adaptive temperature without in-place operations
    similarity_matrix = torch.zeros_like(raw_similarity, device=device)
    for i in range(batch_size):
        similarity_matrix[i] = raw_similarity[i] / adaptive_temp[i]
    
    # InfoNCE part with adaptive temperature
    labels = torch.arange(batch_size, device=device)
    loss_a2b = F.cross_entropy(similarity_matrix, labels)
    loss_b2a = F.cross_entropy(similarity_matrix.t(), labels)
    infonce_loss = (loss_a2b + loss_b2a) / 2.0
    
    # Triplet margin part (without in-place operations)
    diagonal_mask = 1 - torch.eye(batch_size, device=device)
    masked_sim = similarity_matrix * diagonal_mask
    # Create a new mask for valid values
    valid_mask = torch.ones_like(masked_sim, device=device)
    valid_mask[masked_sim == 0] = -float('inf')
    masked_sim = masked_sim + valid_mask
    hardest_negative_sim = torch.max(masked_sim, dim=1)[0]
    
    triplet_loss = F.relu(hardest_negative_sim - torch.diag(similarity_matrix) + margin).mean()
    
    # Combined loss with balance
    return infonce_loss + triplet_loss * margin_multiplier
    
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def multi_purpose_contrastive_loss(
    embeds_a,
    embeds_b,
    data_type=None,
    similarity_scores=None,
    temperature=0.07,
    margin=0.2,
    alpha=0.5,
    use_adaptive=False,
):
    batch_size = embeds_a.shape[0]
    device = embeds_a.device

    if similarity_scores is not None and isinstance(similarity_scores, torch.Tensor):
        similarity_scores = similarity_scores.to(device)

    if not isinstance(data_type, (list, torch.Tensor)) or len(data_type) != batch_size:
        logger.warning("data_type not provided per example; applying uniform loss based on use_adaptive flag.")
        if use_adaptive:
            return adaptive_infonce_loss(embeds_a.to(device), embeds_b.to(device), base_temp=temperature, alpha=alpha)
        else:
            similarity_matrix = torch.matmul(embeds_a.to(device), embeds_b.to(device).t()) / temperature
            labels = torch.arange(batch_size, device=device)
            loss_a2b = F.cross_entropy(similarity_matrix, labels)
            loss_b2a = F.cross_entropy(similarity_matrix.t(), labels)
            return (loss_a2b + loss_b2a) / 2.0

    if isinstance(data_type, torch.Tensor):
        data_type = data_type.tolist()

    total_loss = 0.0
    count = 0

    data_type_indices = {}
    for i, dt in enumerate(data_type):
        if dt not in data_type_indices:
            data_type_indices[dt] = []
        data_type_indices[dt].append(i)

    for dt, indices in data_type_indices.items():
        if not indices:
            continue

        embeds_a_dt = embeds_a[indices].to(device)
        embeds_b_dt = embeds_b[indices].to(device)
        sub_batch_size = len(indices)

        sim_scores_dt = None
        if similarity_scores is not None:
            if isinstance(similarity_scores, torch.Tensor):
                # Use the tensor for indexing and move to the device
                sim_scores_dt = similarity_scores[torch.tensor(indices, device=device)].to(device)
            else:
                # Handle list input - convert non-None values to a tensor
                valid_scores = [s for s in [similarity_scores[i] for i in indices] if s is not None]
                if valid_scores:
                    sim_scores_dt = torch.tensor(valid_scores, device=device, dtype=torch.float)
                else:
                    sim_scores_dt = torch.zeros(sub_batch_size, device=device, dtype=torch.float)
        else:
            sim_scores_dt = torch.zeros(sub_batch_size, device=device, dtype=torch.float)
        
        if dt == "contrastive_with_score":
            submatrix_sim = torch.matmul(embeds_a_dt, embeds_b_dt.t()) / temperature
            
            # Ensure that sim_scores_dt is properly defined
            if sim_scores_dt is None:
                sim_scores_dt = torch.ones(sub_batch_size, device=device, dtype=torch.float)  # Default to perfect similarity if scores are missing

            targets = torch.zeros_like(submatrix_sim, device=device)  # Ensure targets are on the same device

            for i in range(sub_batch_size):
                targets[i, i] = sim_scores_dt[i]

            log_probs = F.log_softmax(submatrix_sim, dim=1)
            targets = targets + 1e-8  # Add a small constant (epsilon)
            targets_probs = F.softmax(targets * temperature, dim=1) # Remove * 10 and keep temperature
            sub_loss = F.kl_div(log_probs, targets_probs, reduction='batchmean', log_target=False)


        elif dt == "instruction":
            if use_adaptive:
                sub_loss = adaptive_infonce_loss(
                    embeds_a_dt,
                    embeds_b_dt,
                    base_temp=temperature,
                    alpha=alpha
                )
            else:
                direct_sim = torch.sum(embeds_a_dt * embeds_b_dt, dim=1)
                target = torch.ones_like(direct_sim, device=device)
                sub_loss = F.mse_loss(direct_sim, target)

        elif dt == "ocr" or dt == "vqa_single":
            if use_adaptive:
                sub_loss = adaptive_triplet_loss(
                    embeds_a_dt,
                    embeds_b_dt,
                    base_temp=temperature,
                    alpha=alpha,
                    margin=margin,
                    margin_multiplier=1.0,
                    device=device
                )
            else:
                submatrix_sim = torch.matmul(embeds_a_dt, embeds_b_dt.t()) / temperature
                labels = torch.arange(sub_batch_size, device=device)
                sub_loss_a2b = F.cross_entropy(submatrix_sim, labels)
                sub_loss_b2a = F.cross_entropy(submatrix_sim.t(), labels)

                positive_sim = torch.diag(submatrix_sim)
                diagonal_mask = 1 - torch.eye(sub_batch_size, device=device).to(device) # MOVE TO DEVICE
                masked_sim = submatrix_sim * diagonal_mask
                masked_sim = masked_sim.masked_fill(masked_sim == 0, -float('inf')) # USE -float('inf')
                hardest_negative_sim = torch.max(masked_sim, dim=1)[0]
                triplet_loss = F.relu(hardest_negative_sim - positive_sim + margin).mean()

                sub_loss = (sub_loss_a2b + sub_loss_b2a) / 2.0 + triplet_loss

        elif dt == "vqa_multi":
            if use_adaptive:
                sub_loss = adaptive_triplet_loss(
                    embeds_a_dt,
                    embeds_b_dt,
                    base_temp=temperature,
                    alpha=alpha,
                    margin=margin * 1.5,
                    margin_multiplier=1.5,
                    device=device
                )
            else:
                submatrix_sim = torch.matmul(embeds_a_dt, embeds_b_dt.t()) / temperature
                labels = torch.arange(sub_batch_size, device=device)
                sub_loss_a2b = F.cross_entropy(submatrix_sim, labels)
                sub_loss_b2a = F.cross_entropy(submatrix_sim.t(), labels)

                positive_sim = torch.diag(submatrix_sim)
                diagonal_mask = 1 - torch.eye(sub_batch_size, device=device).to(device) # MOVE TO DEVICE
                masked_sim = submatrix_sim * diagonal_mask
                masked_sim = masked_sim.masked_fill(masked_sim == 0, -float('inf')) # USE -float('inf')
                hardest_negative_sim = torch.max(masked_sim, dim=1)[0]
                triplet_loss = F.relu(hardest_negative_sim - positive_sim + margin * 1.5).mean()

                sub_loss = (sub_loss_a2b + sub_loss_b2a) / 2.0 + triplet_loss * 1.5

        elif dt == "adaptive":
            sub_loss = adaptive_infonce_loss(
                embeds_a_dt,
                embeds_b_dt,
                base_temp=temperature,
                alpha=alpha
            )

        else:  # Default case (treat as standard contrastive)
            logger.warning(f"Unknown or default data_type '{dt}' encountered. Applying InfoNCE loss (adaptive={use_adaptive}).")
            if use_adaptive:
                sub_loss = adaptive_infonce_loss(
                    embeds_a_dt,
                    embeds_b_dt,
                    base_temp=temperature,
                    alpha=alpha
                )
            else:
                submatrix_sim = torch.matmul(embeds_a_dt, embeds_b_dt.t()) / temperature
                labels = torch.arange(sub_batch_size, device=device)
                sub_loss_a2b = F.cross_entropy(submatrix_sim, labels)
                sub_loss_b2a = F.cross_entropy(submatrix_sim.t(), labels)
                sub_loss = (sub_loss_a2b + sub_loss_b2a) / 2.0

        total_loss += sub_loss * sub_batch_size
        count += sub_batch_size

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / count