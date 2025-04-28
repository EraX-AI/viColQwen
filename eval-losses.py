import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_retrieval_metrics(emb_a, emb_b):
    """
    Calculate retrieval metrics (Recall@K) for a batch of embeddings
    
    Args:
        emb_a: Embeddings from the first modality (e.g., image)
        emb_b: Embeddings from the second modality (e.g., text)
        
    Returns:
        Dictionary with retrieval metrics (R@1, R@5, R@10)
    """
    batch_size = emb_a.shape[0]
    
    # Calculate similarity matrix
    similarity_matrix = torch.matmul(emb_a, emb_b.t())
    
    # For each row, get the indices of the sorted similarity scores (descending)
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    # Calculate Recall@K
    targets = torch.arange(batch_size, device=emb_a.device)
    
    # Find rank of correct pairs (position of target index in sorted indices)
    ranks = torch.zeros(batch_size, dtype=torch.long, device=emb_a.device)
    for i in range(batch_size):
        ranks[i] = torch.where(sorted_indices[i] == i)[0]
    
    # Calculate metrics
    r1 = (ranks < 1).float().mean().item()
    r5 = (ranks < 5).float().mean().item()
    r10 = (ranks < 10).float().mean().item()
    
    metrics = {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "MedR": torch.median(ranks.float()).item() + 1,  # +1 because ranks are 0-indexed
        "MeanR": torch.mean(ranks.float()).item() + 1
    }
    
    return metrics

def text_pair_loss_eval(
    embeds_a,
    embeds_b,
    similarity_scores=None,
    temperature=0.07
):
    """
    Evaluate text-text pair embeddings with similarity scores
    
    Args:
        embeds_a: Embeddings from the first text
        embeds_b: Embeddings from the second text
        similarity_scores: Tensor of similarity scores (0-1) for each pair
        temperature: Temperature parameter
        
    Returns:
        Dictionary with loss and metrics
    """
    batch_size = embeds_a.shape[0]
    device = embeds_a.device

    # Calculate similarity matrix
    similarity_matrix = torch.matmul(embeds_a, embeds_b.t()) / temperature
    
    # InfoNCE loss
    labels = torch.arange(batch_size, device=device)
    loss_a2b = F.cross_entropy(similarity_matrix, labels)
    loss_b2a = F.cross_entropy(similarity_matrix.t(), labels)
    infonce_loss = (loss_a2b + loss_b2a) / 2.0
    
    # Score-based loss if similarity scores provided
    score_loss = 0.0
    if similarity_scores is not None:
        # Direct prediction loss - predict the similarity score directly
        diagonal = torch.diag(similarity_matrix)
        softmax_diag = F.softmax(diagonal, dim=0)
        score_loss = F.mse_loss(softmax_diag, similarity_scores.to(device))
    
    # Spearman correlation between predicted similarities and ground truth
    correlation = 0.0
    if similarity_scores is not None:
        # Get actual similarity scores (cosine similarity)
        actual_sim = torch.diag(torch.matmul(embeds_a, embeds_b.t())).detach().cpu().numpy()
        gt_sim = similarity_scores.detach().cpu().numpy()
        
        # Calculate Spearman correlation
        try:
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(actual_sim, gt_sim)
        except:
            correlation = 0.0
    
    # Retrieval metrics
    retrieval_metrics = calculate_retrieval_metrics(embeds_a, embeds_b)
    
    return {
        "infonce_loss": infonce_loss.item(),
        "score_loss": score_loss if isinstance(score_loss, float) else score_loss.item(),
        "total_loss": (infonce_loss + score_loss) if isinstance(score_loss, float) else (infonce_loss + score_loss).item(),
        "correlation": correlation,
        **retrieval_metrics
    }

def instruction_loss_eval(
    embeds_a,
    embeds_b,
    temperature=0.07
):
    """
    Evaluate instruction tuning embeddings
    
    Args:
        embeds_a: Embeddings from the instruction input
        embeds_b: Embeddings from the instruction output
        temperature: Temperature parameter
        
    Returns:
        Dictionary with loss and metrics
    """
    batch_size = embeds_a.shape[0]
    device = embeds_a.device
    
    # Calculate similarity matrix
    similarity_matrix = torch.matmul(embeds_a, embeds_b.t()) / temperature
    
    # InfoNCE loss
    labels = torch.arange(batch_size, device=device)
    loss_a2b = F.cross_entropy(similarity_matrix, labels)
    loss_b2a = F.cross_entropy(similarity_matrix.t(), labels)
    infonce_loss = (loss_a2b + loss_b2a) / 2.0
    
    # Direct similarity loss - for instructions, we want to maximize direct similarity
    direct_sim = torch.sum(embeds_a * embeds_b, dim=1)
    direct_loss = 1.0 - direct_sim.mean()
    
    # Retrieval metrics
    retrieval_metrics = calculate_retrieval_metrics(embeds_a, embeds_b)
    
    return {
        "infonce_loss": infonce_loss.item(),
        "direct_loss": direct_loss.item(),
        "total_loss": (infonce_loss + direct_loss).item(),
        **retrieval_metrics
    }

def ocr_loss_eval(
    embeds_a,
    embeds_b,
    temperature=0.07,
    margin=0.2
):
    """
    Evaluate OCR/OCQ embeddings
    
    Args:
        embeds_a: Embeddings from the image+text input
        embeds_b: Embeddings from the text output
        temperature: Temperature parameter
        margin: Margin for triplet loss
        
    Returns:
        Dictionary with loss and metrics
    """
    batch_size = embeds_a.shape[0]
    device = embeds_a.device
    
    # Calculate similarity matrix
    similarity_matrix = torch.matmul(embeds_a, embeds_b.t()) / temperature
    
    # InfoNCE loss
    labels = torch.arange(batch_size, device=device)
    loss_a2b = F.cross_entropy(similarity_matrix, labels)
    loss_b2a = F.cross_entropy(similarity_matrix.t(), labels)
    infonce_loss = (loss_a2b + loss_b2a) / 2.0
    
    # Triplet loss
    positive_sim = torch.diag(similarity_matrix)
    diagonal_mask = 1 - torch.eye(batch_size, device=device)
    masked_sim = similarity_matrix * diagonal_mask
    masked_sim = masked_sim.masked_fill(masked_sim == 0, -1e9)
    hardest_negative_sim = torch.max(masked_sim, dim=1)[0]
    triplet_loss = F.relu(hardest_negative_sim - positive_sim + margin).mean()
    
    # Retrieval metrics
    retrieval_metrics = calculate_retrieval_metrics(embeds_a, embeds_b)
    
    return {
        "infonce_loss": infonce_loss.item(),
        "triplet_loss": triplet_loss.item(),
        "total_loss": (infonce_loss + triplet_loss).item(),
        **retrieval_metrics
    }

def vqa_loss_eval(
    embeds_a,
    embeds_b,
    temperature=0.07,
    margin=0.2,
    is_multi_turn=False
):
    """
    Evaluate VQA embeddings
    
    Args:
        embeds_a: Embeddings from the image+question input
        embeds_b: Embeddings from the answer output
        temperature: Temperature parameter
        margin: Margin for triplet loss
        is_multi_turn: Whether this is multi-turn VQA
        
    Returns:
        Dictionary with loss and metrics
    """
    batch_size = embeds_a.shape[0]
    device = embeds_a.device
    
    # Calculate similarity matrix
    similarity_matrix = torch.matmul(embeds_a, embeds_b.t()) / temperature
    
    # InfoNCE loss
    labels = torch.arange(batch_size, device=device)
    loss_a2b = F.cross_entropy(similarity_matrix, labels)
    loss_b2a = F.cross_entropy(similarity_matrix.t(), labels)
    infonce_loss = (loss_a2b + loss_b2a) / 2.0
    
    # Triplet loss - use higher margin for multi-turn
    effective_margin = margin * 1.5 if is_multi_turn else margin
    positive_sim = torch.diag(similarity_matrix)
    diagonal_mask = 1 - torch.eye(batch_size, device=device)
    masked_sim = similarity_matrix * diagonal_mask
    masked_sim = masked_sim.masked_fill(masked_sim == 0, -1e9)
    hardest_negative_sim = torch.max(masked_sim, dim=1)[0]
    triplet_loss = F.relu(hardest_negative_sim - positive_sim + effective_margin).mean()
    
    # Weight triplet loss higher for multi-turn
    triplet_weight = 1.5 if is_multi_turn else 1.0
    
    # Retrieval metrics
    retrieval_metrics = calculate_retrieval_metrics(embeds_a, embeds_b)
    
    return {
        "infonce_loss": infonce_loss.item(),
        "triplet_loss": triplet_loss.item(),
        "total_loss": (infonce_loss + triplet_loss * triplet_weight).item(),
        **retrieval_metrics
    }

def multi_purpose_eval_losses(
    embeds_a,
    embeds_b,
    data_type=None,  # List/tensor of data types
    similarity_scores=None,  # Optional scores between 0-1 for contrastive pairs
    temperature=0.07,  # Temperature parameter for scaling similarity
    margin=0.2,  # Margin for triplet loss variants
):
    """
    A unified loss evaluation function that handles multiple types of data
    and provides detailed metrics for each type.
    
    Args:
        embeds_a: Embeddings from the first modality/input
        embeds_b: Embeddings from the second modality/output
        data_type: List or tensor indicating the type of each data sample in the batch
        similarity_scores: Optional tensor of similarity scores (0-1) for each pair
        temperature: Temperature parameter for scaling cosine similarity
        margin: Margin parameter for triplet loss variants
        
    Returns:
        Dictionary with separate metrics for each data type
    """
    batch_size = embeds_a.shape[0]
    device = embeds_a.device
    
    # Ensure data_type is a list
    if isinstance(data_type, torch.Tensor):
        data_type = data_type.cpu().tolist()
    elif data_type is None:
        # Default to a single, standard type if not provided
        data_type = ["standard"] * batch_size
    
    # Group examples by data type
    data_type_indices = defaultdict(list)
    for i, dt in enumerate(data_type):
        data_type_indices[dt].append(i)
    
    # Initialize results
    results = {
        "overall": {
            "loss": 0.0,
            "count": 0
        }
    }
    
    # Process each data type separately
    for dt, indices in data_type_indices.items():
        if not indices:  # Skip if no examples of this type
            continue
            
        # Convert indices to tensor for selection
        idx_tensor = torch.tensor(indices, device=device)
        
        # Select embeddings for this data type
        embeds_a_dt = embeds_a.index_select(0, idx_tensor)
        embeds_b_dt = embeds_b.index_select(0, idx_tensor)
        
        # Select similarity scores if available
        sim_scores_dt = None
        if similarity_scores is not None:
            if isinstance(similarity_scores, torch.Tensor) and similarity_scores.shape[0] == batch_size:
                sim_scores_dt = similarity_scores.index_select(0, idx_tensor)
        
        # Evaluate based on data type
        dt_metrics = {}
        
        if dt == "contrastive_with_score":
            dt_metrics = text_pair_loss_eval(
                embeds_a_dt, embeds_b_dt,
                similarity_scores=sim_scores_dt,
                temperature=temperature
            )
        elif dt == "instruction":
            dt_metrics = instruction_loss_eval(
                embeds_a_dt, embeds_b_dt,
                temperature=temperature
            )
        elif dt == "ocr" or dt == "ocq":
            dt_metrics = ocr_loss_eval(
                embeds_a_dt, embeds_b_dt,
                temperature=temperature,
                margin=margin
            )
        elif dt == "vqa_single":
            dt_metrics = vqa_loss_eval(
                embeds_a_dt, embeds_b_dt,
                temperature=temperature,
                margin=margin,
                is_multi_turn=False
            )
        elif dt == "vqa_multi":
            dt_metrics = vqa_loss_eval(
                embeds_a_dt, embeds_b_dt,
                temperature=temperature,
                margin=margin,
                is_multi_turn=True
            )
        else:
            # Default evaluation
            default_metrics = calculate_retrieval_metrics(embeds_a_dt, embeds_b_dt)
            similarity_matrix = torch.matmul(embeds_a_dt, embeds_b_dt.t()) / temperature
            
            # Standard InfoNCE loss
            sub_batch_size = len(indices)
            sub_labels = torch.arange(sub_batch_size, device=device)
            sub_loss_a2b = F.cross_entropy(similarity_matrix, sub_labels)
            sub_loss_b2a = F.cross_entropy(similarity_matrix.t(), sub_labels)
            sub_loss = (sub_loss_a2b + sub_loss_b2a) / 2.0
            
            dt_metrics = {
                "infonce_loss": sub_loss.item(),
                "total_loss": sub_loss.item(),
                **default_metrics
            }
        
        # Store metrics for this data type
        results[dt] = dt_metrics
        
        # Update overall loss
        results["overall"]["loss"] += dt_metrics["total_loss"] * len(indices)
        results["overall"]["count"] += len(indices)
    
    # Calculate overall average loss
    if results["overall"]["count"] > 0:
        results["overall"]["loss"] = results["overall"]["loss"] / results["overall"]["count"]
    
    return results

class ModularEvalCollector:
    """
    Helper class to collect and aggregate evaluation metrics over multiple steps
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.counts = defaultdict(int)
    
    def update(self, results):
        """Update metrics with new results"""
        for data_type, metrics in results.items():
            if data_type == "overall":
                # Special handling for overall metrics
                count = metrics.get("count", 0)
                if count > 0:
                    self.counts[data_type] += count
                    # Only add loss to overall
                    self.metrics[data_type]["loss"].append(metrics["loss"] * count)
            else:
                # For specific data types
                self.counts[data_type] += 1
                for metric_name, value in metrics.items():
                    self.metrics[data_type][metric_name].append(value)
    
    def compute(self):
        """Compute aggregated metrics"""
        results = {}
        
        for data_type, count in self.counts.items():
            if count == 0:
                continue
                
            results[data_type] = {}
            
            if data_type == "overall":
                # Special handling for overall metrics
                total_samples = sum(self.counts.values())
                if total_samples > 0:
                    results[data_type]["loss"] = sum(self.metrics[data_type]["loss"]) / total_samples
            else:
                # For specific data types
                for metric_name, values in self.metrics[data_type].items():
                    if values:
                        results[data_type][metric_name] = sum(values) / len(values)
                
        return results
