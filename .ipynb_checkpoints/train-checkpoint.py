import torch, json, jsonlines
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    TrainingArguments,
)
import logging
import os
import random
import argparse

from model import ColPaLiQwenEmbedder
from mix_data_collator import MixedBatchCollator
from colpali_trainer_with_eval import ColPaLiTrainerWithEvalLosses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["ACCELERATE_LOG_LEVEL"]="debug"

def main():
    parser = argparse.ArgumentParser(description="Train ColPaLi model with evaluation")
    # Model configuration
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Base model")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Embedding dimension")
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    # Dataset configuration
    parser.add_argument("--train_dataset", type=str, required=True, help="Pre-prepared train dataset path")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Pre-prepared eval dataset path")
    parser.add_argument("--image_base_path", type=str, default="", help="Base path for loading images")
    # Evaluation configuration
    parser.add_argument("--eval_steps", type=int, default=500, help="Steps between evaluations")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], 
                        help="Evaluation strategy")
    # Loss configuration
    parser.add_argument("--use_adaptive_loss", action="store_true", help="Use adaptive loss")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    parser.add_argument("--contrastive_margin", type=float, default=0.2, help="Margin for contrastive loss")
    parser.add_argument("--adaptive_alpha", type=float, default=0.5, help="Alpha for adaptive loss")
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./qwen2p5_colpali_checkpoints", help="Output directory")
    # FSDP configuration
    parser.add_argument("--use_fsdp", action="store_true", help="Use FSDP for distributed training")
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default="Qwen2VLDecoderLayer", 
                      help="Transformer layer class to wrap for FSDP")
    parser.add_argument("--force_ddp", action="store_true", help="Force using DDP instead of FSDP if there are issues")
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Detect if we're using a Qwen VL model
    is_qwen_vl = "qwen" in args.model.lower() and "vl" in args.model.lower()
    if is_qwen_vl:
        logger.info(f"Detected Qwen VL model: {args.model}")
    
    # Load processor
    logger.info(f"Loading processor from {args.model}")
    processor = AutoProcessor.from_pretrained(
        args.model, 
        model_max_length=args.max_length,
        trust_remote_code=True
    )

    def LoadJsonL(file):
        data = []
        with open(file, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line in {file}: {e}")
        return data
        
    # Load pre-prepared dataset
    logger.info(f"Loading dataset from {args.train_dataset}")
    logger.info(f"Loading dataset from {args.eval_dataset}")
    try:
        train_dataset = LoadJsonL(args.train_dataset)
        eval_dataset = LoadJsonL(args.eval_dataset)        
        logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval examples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Set up data collator for mixed batches
    data_collator = MixedBatchCollator(
        processor=processor, 
        max_length=args.max_length,
        image_base_path=args.image_base_path
    )
    
    # Initialize model with Qwen backbone
    logger.info(f"Initializing model from {args.model}")
    model = ColPaLiQwenEmbedder(args.model, embed_dim=args.embed_dim)
    
    # Enable gradient checkpointing for memory efficiency
    model.qwen.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")
    
    # Print trainable parameters count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    # Training arguments - only standard parameters that HF TrainingArguments accepts
    fsdp_config = None
    if args.use_fsdp and not args.force_ddp:
        fsdp_config = "full_shard auto_wrap"
        logger.info("Using FSDP for distributed training")
    elif args.force_ddp:
        logger.info("Forcing DDP instead of FSDP as requested")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=1,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        save_steps=args.eval_steps,
        save_total_limit=10,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        bf16=True,
        tf32=True,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        # FSDP specific settings
        fsdp=fsdp_config,
        fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap if args.use_fsdp else None,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Initialize trainer with custom loss parameters passed directly
    trainer = ColPaLiTrainerWithEvalLosses(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # Custom loss parameters
        use_adaptive_loss=args.use_adaptive_loss,
        temperature=args.temperature,
        contrastive_margin=args.contrastive_margin,
        adaptive_alpha=args.adaptive_alpha,
        processor=processor
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_model_path)
    logger.info(f"Model saved to {final_model_path}")
    
    # Final evaluation
    logger.info("Running final evaluation")
    eval_metrics = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_metrics}")

if __name__ == "__main__":
    main()