# src/train.py
# Main training script for LLaVA-ORPO

import os
import sys
import logging
import argparse
from pathlib import Path
import yaml
import wandb
import torch
from transformers import TrainingArguments, set_seed
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model_utils import LLaVAModelManager
from data_loader import RLAIFVDataLoader, collate_fn
from orpo_trainer import ORPOTrainer, ORPOCollator
from constants import (
    TRAINING_HYPERPARAMS,
    WANDB_CONFIGS,
    OUTPUT_CONFIGS,
    DATA_PROCESSING_CONFIGS,
    MEMORY_CONFIGS,
    PROJECT_ROOT
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "outputs" / "training.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train LLaVA with ORPO")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf",
                       help="Model ID to load")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for parameter-efficient training")
    parser.add_argument("--lora_r", type=int, default=4,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=8,
                       help="LoRA alpha")
    
    # Data arguments
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use (for testing)")
    parser.add_argument("--streaming", action="store_true", default=True,
                       help="Use streaming dataset loading")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, 
                       default=str(OUTPUT_CONFIGS["output_dir"]),
                       help="Output directory for checkpoints")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    
    # ORPO arguments
    parser.add_argument("--orpo_beta", type=float, default=0.1,
                       help="ORPO beta parameter")
    parser.add_argument("--orpo_alpha", type=float, default=1.0,
                       help="ORPO alpha parameter")
    
    # Logging arguments
    parser.add_argument("--wandb_project", type=str, default="llava-orpo",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--cpu_offload", action="store_true", default=True,
                       help="Enable CPU offloading for memory efficiency")
    
    return parser.parse_args()

def setup_wandb(args):
    """Initialize Weights & Biases logging"""
    if args.no_wandb:
        logger.info("Weights & Biases logging disabled")
        return
    
    # Create run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.wandb_run_name or f"llava_orpo_{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_id": args.model_id,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "orpo_beta": args.orpo_beta,
            "orpo_alpha": args.orpo_alpha,
            "max_samples": args.max_samples,
        },
        tags=WANDB_CONFIGS["tags"],
        notes=WANDB_CONFIGS["notes"]
    )
    
    logger.info(f"Initialized Weights & Biases: {wandb.run.url}")

def create_training_arguments(args):
    """Create TrainingArguments for the trainer"""
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        # Output
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        
        # Training schedule
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Optimization
        optim="adamw_torch",
        weight_decay=0.01,
        warmup_steps=TRAINING_HYPERPARAMS["warmup_steps"],
        lr_scheduler_type="cosine",
        
        # Precision
        fp16=False,
        bf16=TRAINING_HYPERPARAMS["bf16"],
        
        # Memory optimization
        gradient_checkpointing=TRAINING_HYPERPARAMS["gradient_checkpointing"],
        dataloader_num_workers=TRAINING_HYPERPARAMS["dataloader_num_workers"],
        dataloader_pin_memory=TRAINING_HYPERPARAMS["dataloader_pin_memory"],
        remove_unused_columns=TRAINING_HYPERPARAMS["remove_unused_columns"],
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=TRAINING_HYPERPARAMS["eval_steps"],
        load_best_model_at_end=OUTPUT_CONFIGS["load_best_model_at_end"],
        metric_for_best_model=OUTPUT_CONFIGS["metric_for_best_model"],
        greater_is_better=OUTPUT_CONFIGS["greater_is_better"],
        
        # Checkpointing
        save_strategy="steps",
        save_steps=TRAINING_HYPERPARAMS["save_steps"],
        save_total_limit=OUTPUT_CONFIGS["save_total_limit"],
        
        # Logging
        logging_strategy="steps",
        logging_steps=TRAINING_HYPERPARAMS["logging_steps"],
        report_to=["wandb"] if not args.no_wandb else [],
        
        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
        
        # Other
        push_to_hub=False,
    )
    
    return training_args

def main():
    """Main training function"""
    logger.info("🚀 Starting LLaVA-ORPO Training")
    
    # Parse arguments
    args = parse_args()
    logger.info(f"Arguments: {vars(args)}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup Weights & Biases
    setup_wandb(args)
    
    try:
        # 1. Initialize model manager and load model
        logger.info("📦 Loading model and processor...")
        model_config = {
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha
            },
            "memory": {
                "use_cpu_offload": args.cpu_offload
            }
        }
        
        model_manager = LLaVAModelManager(config=model_config)
        model, processor = model_manager.load_model_and_processor(
            use_quantization=True,
            use_lora=args.use_lora
        )
        
        logger.info("✅ Model and processor loaded successfully")
        
        # 2. Load and prepare dataset
        logger.info("📚 Loading and preparing dataset...")
        data_loader = RLAIFVDataLoader(
            processor=processor,
            streaming=args.streaming,
            max_samples=args.max_samples
        )
        
        # Load raw dataset
        raw_dataset = data_loader.load_raw_dataset(split="train")
        
        # Analyze dataset
        analysis = data_loader.analyze_dataset(raw_dataset)
        logger.info("Dataset analysis completed")
        
        # Create training datasets
        train_dataset, eval_dataset, test_dataset = data_loader.create_training_dataset(raw_dataset)
        
        logger.info(f"✅ Dataset prepared - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")
        
        # 3. Create training arguments
        training_args = create_training_arguments(args)
        
        # 4. Create data collator
        data_collator = ORPOCollator(
            processor=processor,
            max_length=TRAINING_HYPERPARAMS["max_length"]
        )
        
        # 5. Initialize trainer
        logger.info("🏋️ Initializing ORPO trainer...")
        trainer = ORPOTrainer(
            model=model,
            processor=processor,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # 6. Run initial evaluation
        logger.info("📊 Running initial evaluation...")
        try:
            initial_metrics = trainer.evaluate()
            logger.info(f"Initial evaluation metrics: {initial_metrics}")
            
            if wandb.run:
                wandb.log({"initial_eval": initial_metrics})
        except Exception as e:
            logger.warning(f"Initial evaluation failed: {e}")
        
        # 7. Start training
        logger.info("🚀 Starting training...")
        
        # Log training start
        if wandb.run:
            wandb.log({
                "training_started": True,
                "total_train_samples": len(train_dataset),
                "total_eval_samples": len(eval_dataset),
                "effective_batch_size": args.batch_size * args.gradient_accumulation_steps
            })
        
        # Train the model
        train_result = trainer.train()
        
        logger.info("✅ Training completed!")
        logger.info(f"Training metrics: {train_result.metrics}")
        
        # 8. Final evaluation
        logger.info("📊 Running final evaluation...")
        final_metrics = trainer.evaluate()
        logger.info(f"Final evaluation metrics: {final_metrics}")
        
        # 9. Save model
        logger.info("💾 Saving trained model...")
        trainer.save_model()
        trainer.save_state()
        
        # Save additional artifacts
        model_manager.save_model(
            output_dir=str(Path(args.output_dir) / "final_model")
        )
        
        # Save training info
        training_info = {
            "args": vars(args),
            "train_metrics": train_result.metrics,
            "eval_metrics": final_metrics,
            "dataset_analysis": analysis
        }
        
        info_path = Path(args.output_dir) / "training_info.yaml"
        with open(info_path, 'w') as f:
            yaml.dump(training_info, f, default_flow_style=False)
        
        logger.info(f"Training info saved to {info_path}")
        
        # 10. Log final results
        if wandb.run:
            wandb.log({
                "training_completed": True,
                "final_metrics": final_metrics,
                "total_steps": train_result.global_step
            })
            
            # Save model to wandb
            wandb.save(str(Path(args.output_dir) / "final_model" / "*"))
        
        logger.info("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if wandb.run:
            wandb.log({"training_failed": True, "error": str(e)})
        raise
    
    finally:
        # Cleanup
        if 'model_manager' in locals():
            model_manager.cleanup()
        
        if wandb.run:
            wandb.finish()
        
        logger.info("🧹 Cleanup completed")

if __name__ == "__main__":
    main()