import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
import logging
from eval_losses import multi_purpose_eval_losses, ModularEvalCollector
import os
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColPaLiTrainerWithEvalLosses(Trainer):
    """
    Extended trainer with specialized evaluation loss handling for different data types
    """
    def __init__(self, 
                 *args, 
                 use_adaptive_loss=False, 
                 temperature=0.07, 
                 contrastive_margin=0.2, 
                 adaptive_alpha=0.5,
                 processor=None,
                 **kwargs):
        # Initialize parent Trainer with standard args/kwargs
        super().__init__(*args, **kwargs)
        
        # Store custom loss parameters as instance variables
        self.use_adaptive_loss = use_adaptive_loss
        self.temperature = temperature
        self.contrastive_margin = contrastive_margin
        self.adaptive_alpha = adaptive_alpha
        
        # Initialize evaluation metrics trackers
        self.eval_loss_collector = ModularEvalCollector()
        self.best_metrics = {}
        self.best_epoch = 0
        self.current_epoch = 0
        self.processor=processor
        
        # Add flag to track evaluation state
        self.is_in_eval = False
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the training loss with proper device and tensor handling for FSDP.
        """
        # Filter the inputs to include only what the model expects
        model_inputs = {k: v for k, v in inputs.items() if k in [
            'pixel_values_a', 'input_ids_a', 'attention_mask_a', 'has_image_a',
            'pixel_values_b', 'input_ids_b', 'attention_mask_b', 'has_image_b'
        ]}
        
        # Pass filtered inputs to the model
        embeds_a, embeds_b = model(**model_inputs)
    
        # Get data type and similarity scores
        data_type = inputs.get("data_type", None)
        similarity_scores = inputs.get("similarity_scores", None)
    
        # Use instance variables for loss parameters
        temperature = self.temperature
        margin = self.contrastive_margin
        
        # Check if we're in evaluation mode
        if self.state.is_local_process_zero and self.args.eval_strategy != "no" and self.is_in_eval:
            try:
                # We're in evaluation mode - compute detailed metrics per data type
                eval_results = multi_purpose_eval_losses(
                    embeds_a,
                    embeds_b,
                    data_type=data_type,
                    similarity_scores=similarity_scores,
                    temperature=temperature,
                    margin=margin
                )
                
                # Store metrics for later aggregation
                self.eval_loss_collector.update(eval_results)
                
                # Use the overall loss for the trainer
                loss_value = eval_results["overall"]["loss"]
                loss = torch.tensor(loss_value, device=embeds_a.device, requires_grad=True)
            except Exception as e:
                logger.warning(f"Error in evaluation loss calculation: {e}")
                # Fall back to standard loss calculation
                from losses import multi_purpose_contrastive_loss
                loss = multi_purpose_contrastive_loss(
                    embeds_a,
                    embeds_b,
                    data_type=data_type,
                    similarity_scores=similarity_scores,
                    temperature=temperature,
                    margin=margin,
                    alpha=self.adaptive_alpha,
                    use_adaptive=self.use_adaptive_loss
                )
        else:
            # We're in training mode - use standard loss calculation
            # Import here to avoid circular imports
            from losses import multi_purpose_contrastive_loss
            
            # Use the instance variables
            loss = multi_purpose_contrastive_loss(
                embeds_a,
                embeds_b,
                data_type=data_type,
                similarity_scores=similarity_scores,
                temperature=temperature,
                margin=margin,
                alpha=self.adaptive_alpha,
                use_adaptive=self.use_adaptive_loss
            )
        
        outputs = (embeds_a, embeds_b)
        return (loss, outputs) if return_outputs else loss
        
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluation loop to collect and report data type specific metrics
        """
        # Reset the eval metric collector
        self.eval_loss_collector.reset()
        
        # Set evaluation flag to True
        self.is_in_eval = True
        
        try:
            # Call the parent evaluation loop
            eval_output = super().evaluation_loop(
                dataloader, 
                description, 
                prediction_loss_only, 
                ignore_keys, 
                metric_key_prefix
            )
            
            # Get detailed metrics per data type
            if self.state.is_local_process_zero:
                try:
                    detailed_metrics = self.eval_loss_collector.compute()
                    
                    # Log metrics
                    self._log_eval_metrics(detailed_metrics)
                    
                    # Save metrics to file
                    metrics_file = os.path.join(self.args.output_dir, f"eval_metrics_epoch_{self.state.epoch}.json")
                    with open(metrics_file, "w") as f:
                        json.dump(detailed_metrics, f, indent=2)
                    
                    # Check for best metrics and save best model if improved
                    self._check_best_metrics(detailed_metrics)
                except Exception as e:
                    logger.error(f"Error processing evaluation metrics: {e}")
                    # Continue with training despite metrics error
            
            return eval_output
        finally:
            # Always reset the evaluation flag when done
            self.is_in_eval = False
            
    def _log_eval_metrics(self, detailed_metrics):
        """Log the detailed metrics in a readable format"""
        logger.info(f"===== Evaluation Metrics for Epoch {self.state.epoch} =====")
        
        # First log overall metrics
        if "overall" in detailed_metrics:
            logger.info(f"Overall Loss: {detailed_metrics['overall']['loss']:.4f}")
        
        # Log metrics for each data type
        for data_type, metrics in detailed_metrics.items():
            if data_type == "overall":
                continue
                
            logger.info(f"\n----- {data_type} Metrics -----")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
    
    def _check_best_metrics(self, detailed_metrics):
        """Check if current metrics are better than previous best, with FSDP-safe operations"""
        # Initialize best metrics if not already done
        if not self.best_metrics:
            self.best_metrics = detailed_metrics
            self.best_epoch = self.state.epoch
            
            # Save initial best model
            self._save_best_model()
            logger.info(f"Initial best metrics saved at epoch {self.best_epoch}")
            return
        
        # Define a simple priority order for different data types
        priority_order = ["vqa_multi", "vqa_single", "ocr", "instruction", "contrastive_with_score"]
        
        # Track if improvements were found
        improved = False
        
        # Check for improvements in priority order
        for data_type in priority_order:
            if data_type in detailed_metrics and data_type in self.best_metrics:
                if "R@1" in detailed_metrics[data_type] and "R@1" in self.best_metrics[data_type]:
                    # For retrieval-based metrics, higher R@1 is better
                    if detailed_metrics[data_type]["R@1"] > self.best_metrics[data_type]["R@1"] * 1.01:  # 1% improvement threshold
                        improved = True
                        logger.info(f"Improved {data_type} R@1: {self.best_metrics[data_type]['R@1']:.4f} -> {detailed_metrics[data_type]['R@1']:.4f}")
                        break
                elif "total_loss" in detailed_metrics[data_type] and "total_loss" in self.best_metrics[data_type]:
                    # For loss-based metrics, lower is better
                    if detailed_metrics[data_type]["total_loss"] < self.best_metrics[data_type]["total_loss"] * 0.99:  # 1% improvement threshold
                        improved = True
                        logger.info(f"Improved {data_type} loss: {self.best_metrics[data_type]['total_loss']:.4f} -> {detailed_metrics[data_type]['total_loss']:.4f}")
                        break
        
        # If overall metrics improved significantly, consider it an improvement
        if "overall" in detailed_metrics and "overall" in self.best_metrics:
            if detailed_metrics["overall"]["loss"] < self.best_metrics["overall"]["loss"] * 0.99:  # 1% improvement threshold
                improved = True
                logger.info(f"Improved overall loss: {self.best_metrics['overall']['loss']:.4f} -> {detailed_metrics['overall']['loss']:.4f}")
        
        # Save new best model if improved
        if improved:
            self.best_metrics = detailed_metrics
            self.best_epoch = self.state.epoch
            self._save_best_model()
            logger.info(f"New best model saved at epoch {self.best_epoch}")
            
    def _save_best_model(self):
        """
        Save the current model as the best model, with FSDP-aware operations.
        For FSDP, we'll just save the metrics and indicate this is the best checkpoint.
        """
        best_model_dir = os.path.join(self.args.output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        # Create a marker file to identify this as the best model directory
        with open(os.path.join(best_model_dir, "best_checkpoint_marker.txt"), "w") as f:
            f.write(f"Best checkpoint from epoch: {self.best_epoch}\n")
        
        # Save the metrics
        with open(os.path.join(best_model_dir, "best_metrics.json"), "w") as f:
            json.dump({
                "best_metrics": self.best_metrics,
                "epoch": self.best_epoch
            }, f, indent=2)
        
        # For non-FSDP modes, we can save the model directly
        if not getattr(self.args, "fsdp", None):
            logger.info("Saving best model weights directly")
            self.model.save_pretrained(best_model_dir)
            if self.processor is not None:
                self.processor.save_pretrained(best_model_dir)
        else:
            # For FSDP, just log that we would save at this point 
            # The full model will be saved at the regular save steps
            logger.info(f"FSDP detected: Best model at epoch {self.best_epoch}, saving metrics only")
    
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        self.current_epoch = state.epoch
        return super().on_epoch_end(args, state, control, **kwargs)