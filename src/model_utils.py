# # src/model_utils.py

# import torch
# from transformers import BitsAndBytesConfig # Now we will definitely use this
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# from llava.model.builder import load_pretrained_model 
# from llava.mm_utils import get_model_name_from_path

# def load_llava_qlora_model(
#     model_path: str = "liuhaotian/llava-v1.6-mistral-7b",
#     use_qlora: bool = True,
#     lora_r: int = 16,
#     lora_alpha: int = 32,
#     lora_dropout: float = 0.05,
#     lora_target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj", 
#                                  "gate_proj", "up_proj", "down_proj"],
#     torch_dtype_for_training=torch.bfloat16
# ):
#     print(f"Loading LLaVA model from: {model_path}")

#     model_name = get_model_name_from_path(model_path)
    
#     # Prepare kwargs for load_pretrained_model
#     kwargs_for_llava_loader = {
#         "device_map": "cuda:0",
#         "torch_dtype": torch.float32,
#     }

#     if use_qlora:
#         print(f"Attempting QLoRA: r={lora_r}, alpha={lora_alpha}, target_modules={lora_target_modules}")
#         print(f"Compute dtype for QLoRA (and LoRA weights) will be: {torch_dtype_for_training}")
        
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True, # This flag within bnb_config is fine
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch_dtype_for_training,
#             bnb_4bit_use_double_quant=True,
#         )
#         # Pass the bnb_config object via quantization_config
#         kwargs_for_llava_loader["quantization_config"] = bnb_config
#         # Ensure load_4bit/load_8bit flags are NOT passed if quantization_config is present
#         # LLaVA's script might not explicitly look for quantization_config for the LLM,
#         # but Hugging Face's from_pretrained (which LLaVA calls) does.
#         # We must ensure load_4bit is False or not present in kwargs if quantization_config is.
#         kwargs_for_llava_loader["load_4bit"] = False 
#         kwargs_for_llava_loader["load_8bit"] = False

#         print("Using explicit BitsAndBytesConfig for QLoRA.")

#     else: # Not using QLoRA
#         kwargs_for_llava_loader["load_4bit"] = False
#         kwargs_for_llava_loader["load_8bit"] = False
#         print(f"Loading base LLaVA model with dtype: {torch_dtype_for_training} (no QLoRA)")

#     try:
#         tokenizer, model, image_processor, context_len = load_pretrained_model(
#             model_path=model_path,
#             model_base=None,
#             model_name=model_name,
#             **kwargs_for_llava_loader # Pass all prepared arguments
#         )
#         if use_qlora:
#             print("Base LLaVA model loaded with explicit quantization_config for QLoRA.")
#         else:
#             print("Base LLaVA model loaded.")

#     except Exception as e:
#         print(f"Error during LLaVA's load_pretrained_model call: {e}")
#         raise

#     model_config = model.config

#     if use_qlora:
#         # Prepare model for k-bit training
#         model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
#         print("Model prepared for k-bit training (QLoRA with gradient checkpointing).")

#         lora_config = LoraConfig(
#             r=lora_r,
#             lora_alpha=lora_alpha,
#             target_modules=lora_target_modules,
#             lora_dropout=lora_dropout,
#             bias="none",
#             task_type="CAUSAL_LM"
#         )
#         model = get_peft_model(model, lora_config)
#         print("LoRA applied to the model.")
#         model.print_trainable_parameters()

#     return model, tokenizer, image_processor, model_config, context_len

# # --- __main__ test block ---
# if __name__ == '__main__':
#     print("--- Testing LLaVA Model Loading Utility ---")
    
#     current_torch_dtype = torch.bfloat16
#     if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
#         print("BF16 not supported or CUDA not available, falling back to FP16 for compute_dtype.")
#         current_torch_dtype = torch.float16
#     elif not torch.cuda.is_available():
#         print("CUDA not available, QLoRA test will likely fail or run on CPU very slowly.")
#         # QLoRA fundamentally relies on GPU.

#     try:
#         print(f"\n--- Test 1: Loading with QLoRA (compute dtype: {current_torch_dtype}) ---")
#         model, tokenizer, image_processor, config, context_len = load_llava_qlora_model(
#             use_qlora=True,
#             torch_dtype_for_training=current_torch_dtype
#         )
#         print("QLoRA Model loaded successfully.")
#         # ... (rest of the print statements and cleanup) ...
#         print(f"  Model type: {type(model)}")
#         print(f"  Tokenizer type: {type(tokenizer)}")
#         print(f"  Image processor type: {type(image_processor)}")
#         print(f"  Context length: {context_len}")
#         if hasattr(model, 'device'): # For PeftModel, device is on base_model
#              print(f"  PEFT Model overall device (might be 'auto' or a specific device): {model.device}")
#         if hasattr(model, 'base_model') and hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'model') and hasattr(model.base_model.model.model, 'device'):
#              print(f"  PEFT Model - Underlying LLM device: {model.base_model.model.model.device}")


#         del model, tokenizer, image_processor, config
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         import gc
#         gc.collect()

#     except Exception as e:
#         print(f"Error during QLoRA loading test: {e}")
#         import traceback
#         traceback.print_exc()

#     print("\n--- Model Loading Utility Test Complete ---")

# src/model_utils.py
# LLaVA model loading, QLoRA setup, and memory optimization utilities

import torch
import gc
import os
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from typing import Tuple, Optional, Dict, Any

from constants import (
    MODEL_CONFIGS,
    TRAINING_CONFIGS, 
    MEMORY_CONFIGS,
    DEVICE_CONFIGS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set memory allocation strategy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = MEMORY_CONFIGS["pytorch_cuda_alloc_conf"]

class LLaVAModelManager:
    """
    Manages LLaVA model loading, configuration, and memory optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model = None
        self.processor = None
        
        # Merge with default configs
        self.model_config = {**MODEL_CONFIGS["llava_1_6_7b"], **self.config.get("model", {})}
        self.lora_config = {**TRAINING_CONFIGS["lora"], **self.config.get("lora", {})}
        self.memory_config = {**MEMORY_CONFIGS, **self.config.get("memory", {})}
        
        logger.info("Initialized LLaVAModelManager")
    
    def determine_optimal_dtype(self) -> torch.dtype:
        """Determine optimal dtype based on hardware capabilities"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU with float32")
            return torch.float32
        
        if torch.cuda.is_bf16_supported():
            logger.info("Using bfloat16 (recommended)")
            return torch.bfloat16
        else:
            logger.info("BF16 not supported, using float16")
            return torch.float16
    
    def create_quantization_config(self, compute_dtype: torch.dtype) -> BitsAndBytesConfig:
        """Create BitsAndBytesConfig for 4-bit quantization"""
        quant_config = TRAINING_CONFIGS["quantization"]
        
        return BitsAndBytesConfig(
            load_in_4bit=quant_config["load_in_4bit"],
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
        )
    
    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration for parameter-efficient fine-tuning"""
        return LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["alpha"],
            target_modules=self.lora_config["target_modules"],
            lora_dropout=self.lora_config["dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def load_model_and_processor(
        self, 
        use_quantization: bool = True,
        use_lora: bool = True,
        device_map: str = "auto"
    ) -> Tuple[Any, Any]:
        """
        Load LLaVA model and processor with optimizations
        
        Args:
            use_quantization: Whether to use 4-bit quantization
            use_lora: Whether to apply LoRA adapters
            device_map: Device mapping strategy
            
        Returns:
            Tuple of (model, processor)
        """
        model_id = self.model_config["model_id"]
        logger.info(f"Loading LLaVA model: {model_id}")
        
        # Determine optimal settings
        compute_dtype = self.determine_optimal_dtype()
        
        # Clear memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._log_memory_usage("Before model loading")
        
        # Load processor first (lightweight)
        logger.info("Loading processor...")
        processor = LlavaNextProcessor.from_pretrained(model_id)
        
        # Configure model loading arguments
        model_kwargs = {
            "torch_dtype": compute_dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        
        if use_quantization:
            logger.info(f"Setting up 4-bit quantization with dtype: {compute_dtype}")
            quantization_config = self.create_quantization_config(compute_dtype)
            model_kwargs["quantization_config"] = quantization_config
        
        # Load model
        logger.info("Loading model...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, **model_kwargs
        )
        
        if torch.cuda.is_available():
            self._log_memory_usage("After model loading")
        
        # Apply quantization optimizations
        if use_quantization:
            logger.info("Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=True
            )
        
        # Apply LoRA if requested
        if use_lora:
            logger.info(f"Applying LoRA with r={self.lora_config['r']}, alpha={self.lora_config['alpha']}")
            lora_config = self.create_lora_config()
            model = get_peft_model(model, lora_config)
            
            # Log trainable parameters
            model.print_trainable_parameters()
        
        if torch.cuda.is_available():
            self._log_memory_usage("After LoRA application")
        
        self.model = model
        self.processor = processor
        
        logger.info("✅ Model and processor loaded successfully")
        return model, processor
    
    def enable_cpu_offloading(self) -> None:
        """Enable CPU offloading for memory-constrained inference"""
        if not self.model:
            raise ValueError("Model must be loaded before enabling CPU offloading")
        
        logger.info("CPU offloading enabled - model will be moved during inference")
        self.model.cpu_offload_enabled = True
    
    def cpu_offload_inference(self, inputs: Dict, generation_kwargs: Dict = None):
        """
        Perform inference with CPU offloading to save GPU memory
        
        Args:
            inputs: Model inputs (will be moved to appropriate device)
            generation_kwargs: Generation parameters
            
        Returns:
            Model outputs
        """
        if not self.model or not self.processor:
            raise ValueError("Model and processor must be loaded first")
        
        generation_kwargs = generation_kwargs or {}
        
        # Get original device
        original_device = next(self.model.parameters()).device
        
        try:
            # Step 1: Move model to CPU for preprocessing
            logger.debug("Moving model to CPU for preprocessing...")
            self.model.cpu()
            torch.cuda.empty_cache()
            
            # Step 2: Move model back to GPU for generation
            logger.debug("Moving model back to GPU for generation...")
            self.model.to(original_device)
            
            # Move inputs to same device as model
            device_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    device_inputs[key] = value.to(original_device)
                else:
                    device_inputs[key] = value
            
            # Step 3: Generate with memory optimizations
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **device_inputs,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    **generation_kwargs
                )
            
            # Step 4: Cleanup
            del device_inputs
            torch.cuda.empty_cache()
            
            return outputs
            
        except Exception as e:
            logger.error(f"CPU offload inference failed: {e}")
            # Ensure model is back on original device
            self.model.to(original_device)
            raise
    
    def test_inference(self, test_image_url: str = None, test_question: str = None):
        """Test model inference with a sample image and question"""
        if not self.model or not self.processor:
            raise ValueError("Model and processor must be loaded first")
        
        # Default test inputs
        if test_image_url is None:
            test_image_url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        
        if test_question is None:
            test_question = "Describe this image briefly."
        
        try:
            from PIL import Image
            import requests
            
            logger.info("Running inference test...")
            
            # Load test image
            image = Image.open(requests.get(test_image_url, stream=True).raw) 
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for memory efficiency
            image = image.resize((224, 224))
            
            # Create conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": test_question},
                        {"type": "image"},
                    ],
                },
            ]
            
            # Process inputs
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(image, prompt, return_tensors="pt")
            
            # Generate response
            if self.memory_config["use_cpu_offload"]:
                outputs = self.cpu_offload_inference(
                    inputs,
                    {
                        "max_new_tokens": 50,
                        "temperature": 0.7,
                        "do_sample": True,
                        "use_cache": False
                    }
                )
            else:
                # Standard inference
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=False
                    )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            logger.info("✅ Inference test successful!")
            logger.info(f"Question: {test_question}")
            logger.info(f"Response: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Inference test failed: {e}")
            raise
    
    def save_model(self, output_dir: str, safe_serialization: bool = True):
        """Save the trained model and processor"""
        if not self.model or not self.processor:
            raise ValueError("Model and processor must be loaded first")
        
        logger.info(f"Saving model to {output_dir}")
        
        # Save model
        self.model.save_pretrained(
            output_dir,
            safe_serialization=safe_serialization
        )
        
        # Save processor
        self.processor.save_pretrained(output_dir)
        
        logger.info("✅ Model and processor saved successfully")
    
    def cleanup(self):
        """Clean up model and free memory"""
        if self.model:
            del self.model
            self.model = None
        
        if self.processor:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("✅ Model cleanup completed")
    
    def _log_memory_usage(self, stage: str = ""):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"GPU Memory {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")

def find_target_modules(model) -> list:
    """
    Utility function to find linear layer names for LoRA targeting
    
    Args:
        model: The model to analyze
        
    Returns:
        List of module names suitable for LoRA targeting
    """
    linear_cls = torch.nn.Linear
    lora_module_names = set()
    
    # Get the base model (unwrap PEFT wrapper if present)
    base_model = model
    if hasattr(model, 'base_model'):
        base_model = model.base_model
    if hasattr(base_model, 'model'):
        base_model = base_model.model
    
    for name, module in base_model.named_modules():
        # Skip vision encoder and multimodal projector
        if any(keyword in name.lower() for keyword in ['vision_tower', 'multi_modal_projector', 'vision_model']):
            continue
        
        if isinstance(module, linear_cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    
    # Remove output layer if present
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

# Testing and demo
if __name__ == "__main__":
    logger.info("Testing LLaVA Model Manager")
    
    try:
        # Initialize model manager
        model_manager = LLaVAModelManager()
        
        # Load model with conservative settings for testing
        model, processor = model_manager.load_model_and_processor(
            use_quantization=True,
            use_lora=True
        )
        
        logger.info("✅ Model loading test successful")
        
        # Test inference
        model_manager.test_inference()
        
        logger.info("✅ All tests completed successfully")
        
        # Cleanup
        model_manager.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Model manager test failed: {e}")
        raise