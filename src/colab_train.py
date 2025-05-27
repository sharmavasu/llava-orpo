#!/usr/bin/env python3
"""
Memory-optimized training script for Google Colab
FIXES: 
- Index out of bounds errors in ORPO trainer
- Memory optimization for limited GPU memory
- Conservative training settings
"""

import os
import sys
import torch
import gc

# Memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Add src to path
sys.path.append('src')

def optimize_memory():
    """Aggressive memory optimization"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    print("🚀 Starting LLaVA-ORPO Training on Colab (Memory Optimized)")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        try:
            torch.cuda.set_per_process_memory_fraction(0.9)
        except Exception as e:
            print(f"Warning: Could not set per-process memory fraction: {e}")
    else:
        print("❌ No GPU available!")
        return
    
    model_manager = None # Initialize for finally block
    trainer = None # Initialize for finally block

    # Import training components
    try:
        from model_utils import LLaVAModelManager
        from data_loader import RLAIFVDataLoader
        from orpo_trainer_colab import ORPOTrainer, ORPOCollator # Use fixed version
        from transformers import TrainingArguments, set_seed
        import logging
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        set_seed(42)
        
        print("✅ Starting training with ultra-conservative settings...")
        
        logger.info("📦 Loading model with aggressive memory optimization...")
        model_config = {
            "lora": {"r": 8, "alpha": 16, "dropout": 0.1},
            "memory": {"use_cpu_offload": True, "max_memory_gb": 12.0}
        }
        
        model_manager = LLaVAModelManager(config=model_config)
        model, processor = model_manager.load_model_and_processor(
            use_quantization=True, use_lora=True
        )
        
        optimize_memory()
        logger.info("✅ Model loaded successfully")
        
        logger.info("📚 Loading minimal dataset...")
        data_loader = RLAIFVDataLoader(processor=processor, streaming=True, max_samples=1000)
        raw_dataset = data_loader.load_raw_dataset(split="train")
        logger.info("Creating minimal training dataset...")
        train_dataset, _, _ = data_loader.create_training_dataset(raw_dataset)
        optimize_memory()
        logger.info(f"✅ Dataset prepared - Train: {len(train_dataset)}")
        
        training_args = TrainingArguments(
            output_dir="./outputs/colab_test",
            max_steps=100, learning_rate=1e-6,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            fp16=False, bf16=True, gradient_checkpointing=True,
            dataloader_num_workers=0, dataloader_pin_memory=False,
            remove_unused_columns=False, # Important for ORPO
            logging_steps=1, save_steps=5, eval_steps=2,
            eval_strategy="no", save_strategy="steps", report_to=["wandb"],
            seed=42, warmup_steps=5, weight_decay=0.01,
            optim="adamw_torch",
        )
        
        data_collator = ORPOCollator(processor=processor, max_length=3072)
        
        logger.info("🏋️ Initializing ORPO trainer...")
        trainer = ORPOTrainer(
            model=model,
            processor=processor, # Pass processor
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
        
        # --- Explicit Optimizer Creation Test ---
        logger.info("🛠️ Attempting to explicitly create optimizer...")
        try:
            trainer.create_optimizer() # Call the method that should create self.optimizer
            if trainer.optimizer is not None:
                logger.info(f"✅ Optimizer created successfully: {type(trainer.optimizer)}")
                logger.info(f"Optimizer param groups: {len(trainer.optimizer.param_groups)}")
                for i, group in enumerate(trainer.optimizer.param_groups):
                    logger.info(f"  Group {i}: {len(group['params'])} params, lr: {group['lr']}")
            else:
                logger.error("❌ trainer.optimizer is still None after calling create_optimizer().")
        except Exception as opt_e:
            logger.error(f"❌ Error during explicit optimizer creation: {opt_e}")
            import traceback
            traceback.print_exc()
        # --- End Explicit Optimizer Creation Test ---

        optimize_memory()
        
        logger.info("🚀 Testing training with single step...")
        try:
            # Check if optimizer exists before training
            if trainer.optimizer is None and trainer.args.max_steps > 0 : # trainer.args.max_steps > 0 check is from HF trainer
                logger.warning("Trainer.optimizer is None before trainer.train() but training steps are requested. This will likely fail.")
                # This indicates create_optimizer() in the Trainer init or our explicit call failed.
                # The trainer.train() method will call create_optimizer_and_scheduler if optimizer is None.
            
            train_result = trainer.train()
            
            logger.info("✅ Training step completed successfully!")
            logger.info(f"Final loss: {train_result.training_loss if hasattr(train_result, 'training_loss') else 'N/A'}")
            
            logger.info("💾 Attempting to save model...")
            trainer.save_model("./outputs/test_model")
            logger.info("✅ Model saved successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training step failed: {e}")
            import traceback
            traceback.print_exc()
            
            logger.info("🔄 Trying emergency fallback with basic forward pass...")
            try:
                model.train()
                train_dataloader = trainer.get_train_dataloader()
                batch = next(iter(train_dataloader))
                batch_on_device = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
                batch = batch_on_device
                
                autocast_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
                with torch.amp.autocast(device_type=model.device.type, dtype=autocast_dtype, enabled=training_args.bf16 or training_args.fp16):
                    input_ids = batch.get("input_ids")
                    attention_mask = batch.get("attention_mask")  
                    pixel_values = batch.get("pixel_values")
                    image_sizes = batch.get("image_sizes")

                    if input_ids is None or pixel_values is None:
                        raise ValueError("Missing input_ids or pixel_values in the batch for emergency fallback.")

                    model_inputs = {
                        "input_ids": input_ids, "attention_mask": attention_mask,
                        "pixel_values": pixel_values
                    }
                    if image_sizes is not None: model_inputs["image_sizes"] = image_sizes
                    
                    labels = batch.get("labels", input_ids.clone()) # Use input_ids as labels if 'labels' not present
                    model_inputs["labels"] = labels

                    logger.info(f"Emergency forward pass inputs: { {k: (v.shape, v.dtype if isinstance(v, torch.Tensor) else type(v)) for k,v in model_inputs.items()} }")
                    outputs = model(**model_inputs)
                    loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else torch.tensor(0.0, device=model.device)
                
                if loss.requires_grad:
                    logger.info("🚀 Attempting backward pass for emergency test...")
                    loss.backward()
                    logger.info("✅ Backward pass successful.")
                else:
                    logger.warning("Computed loss does not require gradients. Skipping backward pass.")
                logger.info(f"✅ Emergency test successful! Loss: {loss.item()}")
                
            except Exception as e2:
                logger.error(f"❌ Emergency test also failed: {e2}")
                import traceback
                traceback.print_exc()
        
        logger.info("🎉 Training test completed!")
        
    except Exception as e:
        print(f"❌ Training failed at a higher level: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            if model_manager is not None and hasattr(model_manager, 'cleanup'):
                model_manager.cleanup()
            # if trainer is not None and hasattr(trainer, 'model') and trainer.model is not None:
            #     del trainer.model # Try to help GC
            # if 'model' in locals() and model is not None:
            #     del model
            # if 'processor' in locals() and processor is not None:
            #     del processor
            optimize_memory()
            print("🧹 Cleanup completed")
        except Exception as final_cleanup_e:
            print(f"🧹 Error during final cleanup: {final_cleanup_e}")
            pass

if __name__ == "__main__":
    main()