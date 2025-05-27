# # src/model_utils_modern.py
# # Modern approach using Hugging Face transformers + PEFT

# import torch
# from transformers import (
#     LlavaNextProcessor, 
#     LlavaNextForConditionalGeneration,
#     BitsAndBytesConfig
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# def load_llava_qlora_model_modern(
#     model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
#     use_qlora: bool = True,
#     lora_r: int = 16,
#     lora_alpha: int = 32,
#     lora_dropout: float = 0.05,
#     lora_target_modules: list = None,
#     torch_dtype_for_training=torch.bfloat16
# ):
#     """
#     Modern approach using Hugging Face transformers + PEFT
#     More stable and better documented than original LLaVA repo
#     """
#     print(f"Loading LLaVA model from: {model_id}")
    
#     # Default target modules for LLaVA (language model part)
#     if lora_target_modules is None:
#         lora_target_modules = [
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj"
#         ]
    
#     # Processor (handles both text and images)
#     processor = LlavaNextProcessor.from_pretrained(model_id)
    
#     if use_qlora:
#         print(f"Setting up QLoRA: r={lora_r}, alpha={lora_alpha}")
#         print(f"Compute dtype: {torch_dtype_for_training}")
        
#         # Configure 4-bit quantization
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch_dtype_for_training,
#             bnb_4bit_use_double_quant=True,
#         )
        
#         # Load model with quantization
#         model = LlavaNextForConditionalGeneration.from_pretrained(
#             model_id,
#             quantization_config=bnb_config,
#             torch_dtype=torch_dtype_for_training,
#             device_map="auto",
#             low_cpu_mem_usage=True
#         )
        
#         print("Model loaded with 4-bit quantization")
        
#         # Prepare for k-bit training
#         model = prepare_model_for_kbit_training(
#             model, 
#             use_gradient_checkpointing=True
#         )
#         print("Model prepared for k-bit training")
        
#         # Configure LoRA
#         lora_config = LoraConfig(
#             r=lora_r,
#             lora_alpha=lora_alpha,
#             target_modules=lora_target_modules,
#             lora_dropout=lora_dropout,
#             bias="none",
#             task_type="CAUSAL_LM"
#         )
        
#         # Apply LoRA
#         model = get_peft_model(model, lora_config)
#         print("LoRA applied to model")
#         model.print_trainable_parameters()
        
#     else:
#         # Load without quantization
#         model = LlavaNextForConditionalGeneration.from_pretrained(
#             model_id,
#             torch_dtype=torch_dtype_for_training,
#             device_map="auto",
#             low_cpu_mem_usage=True
#         )
#         print("Model loaded without quantization")
    
#     return model, processor

# def test_model_inference(model, processor):
#     """Test basic inference to verify model works"""
#     try:
#         from PIL import Image
#         import requests
        
#         # Load test image
#         url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
#         image = Image.open(requests.get(url, stream=True).raw)
        
#         # Create conversation
#         conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "What is shown in this image?"},
#                     {"type": "image"},
#                 ],
#             },
#         ]
        
#         # Process
#         prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#         inputs = processor(image, prompt, return_tensors="pt").to(model.device)
        
#         # Generate (short response for testing)
#         with torch.no_grad():
#             output = model.generate(
#                 **inputs, 
#                 max_new_tokens=50,
#                 temperature=0.7,
#                 do_sample=True,
#                 pad_token_id=processor.tokenizer.eos_token_id
#             )
        
#         # Decode
#         response = processor.decode(output[0], skip_special_tokens=True)
#         print(f"Test inference successful!")
#         print(f"Response: {response[-100:]}")  # Last 100 chars
        
#         return True
        
#     except Exception as e:
#         print(f"Test inference failed: {e}")
#         return False

# def find_target_modules(model):
#     """
#     Utility to find all linear layer names in the model
#     Useful for determining LoRA target modules
#     """
#     linear_cls = torch.nn.Linear
#     lora_module_names = set()
    
#     for name, module in model.named_modules():
#         # Skip vision encoder and multimodal projector
#         if any(keyword in name for keyword in ['vision_tower', 'multi_modal_projector']):
#             continue
            
#         if isinstance(module, linear_cls):
#             names = name.split('.')
#             lora_module_names.add(names[-1])
    
#     # Remove output layer if present
#     if 'lm_head' in lora_module_names:
#         lora_module_names.remove('lm_head')
    
#     return list(lora_module_names)

# # --- Test/Demo ---
# if __name__ == '__main__':
#     print("--- Testing Modern LLaVA QLoRA Setup ---")
    
#     # Check compute capabilities
#     current_torch_dtype = torch.bfloat16
#     if not torch.cuda.is_available():
#         print("CUDA not available! QLoRA requires GPU.")
#         current_torch_dtype = torch.float16
#     elif not torch.cuda.is_bf16_supported():
#         print("BF16 not supported, using FP16")
#         current_torch_dtype = torch.float16
    
#     try:
#         print(f"\n--- Loading LLaVA with QLoRA (dtype: {current_torch_dtype}) ---")
        
#         model, processor = load_llava_qlora_model_modern(
#             use_qlora=True,
#             torch_dtype_for_training=current_torch_dtype,
#             lora_r=8,  # Smaller rank for testing
#             lora_alpha=16
#         )
        
#         print(f"✅ Model loaded successfully!")
#         print(f"   Model type: {type(model)}")
#         print(f"   Processor type: {type(processor)}")
        
#         # Optional: Test inference
#         print("\n--- Testing Inference ---")
#         test_model_inference(model, processor)
        
#         # Optional: Show target modules
#         print(f"\n--- Available Target Modules ---")
#         if hasattr(model, 'base_model'):
#             target_modules = find_target_modules(model.base_model)
#             print(f"   Found modules: {target_modules}")
        
#         # Cleanup
#         del model, processor
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
        
#         print("✅ Test completed successfully!")
        
#     except Exception as e:
#         print(f"❌ Error during test: {e}")
#         import traceback
#         traceback.print_exc()

#     print("\n--- Test Complete ---").


# src/model_utils_modern.py
# Modern approach using Hugging Face transformers + PEFT

# src/model_utils_modern.py
# Modern approach using Hugging Face transformers + PEFT

# import torch
# from transformers import (
#     LlavaNextProcessor, 
#     LlavaNextForConditionalGeneration,
#     BitsAndBytesConfig
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# def load_llava_qlora_model_modern(
#     model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
#     use_qlora: bool = True,
#     lora_r: int = 16,
#     lora_alpha: int = 32,
#     lora_dropout: float = 0.05,
#     lora_target_modules: list = None,
#     torch_dtype_for_training=torch.bfloat16
# ):
#     """
#     Modern approach using Hugging Face transformers + PEFT
#     More stable and better documented than original LLaVA repo
#     """
#     print(f"Loading LLaVA model from: {model_id}")
    
#     # Default target modules for LLaVA (language model part)
#     if lora_target_modules is None:
#         lora_target_modules = [
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj"
#         ]
    
#     # Processor (handles both text and images)
#     processor = LlavaNextProcessor.from_pretrained(model_id)
    
#     if use_qlora:
#         print(f"Setting up QLoRA: r={lora_r}, alpha={lora_alpha}")
#         print(f"Compute dtype: {torch_dtype_for_training}")
        
#         # Configure 4-bit quantization
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch_dtype_for_training,
#             bnb_4bit_use_double_quant=True,
#         )
        
#         # Load model with quantization - removed invalid parameters
#         model = LlavaNextForConditionalGeneration.from_pretrained(
#             model_id,
#             quantization_config=bnb_config,
#             torch_dtype=torch_dtype_for_training,
#             device_map="auto",
#             low_cpu_mem_usage=True,
#             # Removed: llm_int8_enable_fp32_cpu_offload=True  # This parameter is not valid
#             # Removed: max_memory parameter as it can cause issues
#         )
        
#         print("Model loaded with 4-bit quantization")
        
#         # Prepare for k-bit training
#         model = prepare_model_for_kbit_training(
#             model, 
#             use_gradient_checkpointing=True
#         )
#         print("Model prepared for k-bit training")
        
#         # Configure LoRA
#         lora_config = LoraConfig(
#             r=lora_r,
#             lora_alpha=lora_alpha,
#             target_modules=lora_target_modules,
#             lora_dropout=lora_dropout,
#             bias="none",
#             task_type="CAUSAL_LM"
#         )
        
#         # Apply LoRA
#         model = get_peft_model(model, lora_config)
#         print("LoRA applied to model")
#         model.print_trainable_parameters()
        
#     else:
#         # Load without quantization
#         model = LlavaNextForConditionalGeneration.from_pretrained(
#             model_id,
#             torch_dtype=torch_dtype_for_training,
#             device_map="auto",
#             low_cpu_mem_usage=True,
#         )
#         print("Model loaded without quantization")
    
#     return model, processor

# def test_model_inference(model, processor):
#     """Test basic inference to verify model works"""
#     try:
#         from PIL import Image
#         import requests
        
#         # Load test image
#         url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
#         image = Image.open(requests.get(url, stream=True).raw)
        
#         # Create conversation
#         conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "What is shown in this image?"},
#                     {"type": "image"},
#                 ],
#             },
#         ]
        
#         # Process
#         prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#         inputs = processor(image, prompt, return_tensors="pt").to(model.device)
        
#         # Generate (short response for testing)
#         with torch.no_grad():
#             output = model.generate(
#                 **inputs, 
#                 max_new_tokens=50,
#                 temperature=0.7,
#                 do_sample=True,
#                 pad_token_id=processor.tokenizer.eos_token_id
#             )
        
#         # Decode
#         response = processor.decode(output[0], skip_special_tokens=True)
#         print(f"Test inference successful!")
#         print(f"Response: {response[-100:]}")  # Last 100 chars
        
#         return True
        
#     except Exception as e:
#         print(f"Test inference failed: {e}")
#         return False

# def find_target_modules(model):
#     """
#     Utility to find all linear layer names in the model
#     Useful for determining LoRA target modules
#     """
#     linear_cls = torch.nn.Linear
#     lora_module_names = set()
    
#     for name, module in model.named_modules():
#         # Skip vision encoder and multimodal projector
#         if any(keyword in name for keyword in ['vision_tower', 'multi_modal_projector']):
#             continue
            
#         if isinstance(module, linear_cls):
#             names = name.split('.')
#             lora_module_names.add(names[-1])
    
#     # Remove output layer if present
#     if 'lm_head' in lora_module_names:
#         lora_module_names.remove('lm_head')
    
#     return list(lora_module_names)

# # --- Test/Demo ---
# if __name__ == '__main__':
#     print("--- Testing Modern LLaVA QLoRA Setup ---")
    
#     # Check compute capabilities
#     current_torch_dtype = torch.bfloat16
#     if not torch.cuda.is_available():
#         print("CUDA not available! QLoRA requires GPU.")
#         current_torch_dtype = torch.float16
#     elif not torch.cuda.is_bf16_supported():
#         print("BF16 not supported, using FP16")
#         current_torch_dtype = torch.float16
    
#     try:
#         print(f"\n--- Loading LLaVA with QLoRA (dtype: {current_torch_dtype}) ---")
        
#         model, processor = load_llava_qlora_model_modern(
#             use_qlora=True,
#             torch_dtype_for_training=current_torch_dtype,
#             lora_r=8,  # Smaller rank for testing
#             lora_alpha=16
#         )
        
#         print(f"✅ Model loaded successfully!")
#         print(f"   Model type: {type(model)}")
#         print(f"   Processor type: {type(processor)}")
        
#         # Optional: Test inference
#         print("\n--- Testing Inference ---")
#         test_model_inference(model, processor)
        
#         # Optional: Show target modules
#         print(f"\n--- Available Target Modules ---")
#         if hasattr(model, 'base_model'):
#             target_modules = find_target_modules(model.base_model)
#             print(f"   Found modules: {target_modules}")
        
#         # Cleanup
#         del model, processor
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
        
#         print("✅ Test completed successfully!")
        
#     except Exception as e:
#         print(f"❌ Error during test: {e}")
#         import traceback
#         traceback.print_exc()

#     print("\n--- Test Complete ---")
# src/model_utils_modern.py
# Modern approach using Hugging Face transformers + PEFT with memory optimizations

# src/model_utils_modern.py
# Modern approach using Hugging Face transformers + PEFT with memory optimizations
# src/model_utils_modern.py
# Modern approach using Hugging Face transformers + PEFT with aggressive memory optimizations

import torch
import gc
import os
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Set memory allocation strategy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def load_llava_qlora_model_modern(
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    use_qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list = None,
    torch_dtype_for_training=torch.bfloat16
):
    """
    Modern approach using Hugging Face transformers + PEFT
    More stable and better documented than original LLaVA repo
    """
    print(f"Loading LLaVA model from: {model_id}")
    
    # Default target modules for LLaVA (language model part)
    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    # Processor (handles both text and images)
    processor = LlavaNextProcessor.from_pretrained(model_id)
    
    if use_qlora:
        print(f"Setting up QLoRA: r={lora_r}, alpha={lora_alpha}")
        print(f"Compute dtype: {torch_dtype_for_training}")
        
        # Configure 4-bit quantization with memory optimizations
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype_for_training,
            bnb_4bit_use_double_quant=True,
        )
        
        # Clear cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model with quantization and memory optimizations
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype_for_training,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        print("Model loaded with 4-bit quantization")
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=True
        )
        print("Model prepared for k-bit training")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        print("LoRA applied to model")
        model.print_trainable_parameters()
        
    else:
        # Load without quantization
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype_for_training,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("Model loaded without quantization")
    
    return model, processor

def test_model_inference_with_cpu_offload(model, processor):
    """Test basic inference with aggressive CPU offloading"""
    try:
        from PIL import Image
        import requests
        
        print("Loading test image...")
        # Load test image
        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        image = Image.open(requests.get(url, stream=True).raw)
        
        # Resize image to reduce memory usage
        image = image.resize((224, 224))  # Even smaller resolution
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {"type": "image"},
                ],
            },
        ]
        
        print("Processing inputs...")
        
        # STEP 1: Move model to CPU for preprocessing
        print("Moving model to CPU for preprocessing...")
        original_device = next(model.parameters()).device
        model.cpu()
        torch.cuda.empty_cache()
        
        # Process inputs
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(image, prompt, return_tensors="pt")
        
        print("Moving model back to GPU for generation...")
        # STEP 2: Move model back to GPU for generation
        model.to(original_device)
        
        # Clear cache and move inputs to GPU
        torch.cuda.empty_cache()
        inputs = {k: v.to(original_device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        print("Generating response...")
        # STEP 3: Generate with very conservative settings
        with torch.no_grad():
            # One more cache clear before generation
            torch.cuda.empty_cache()
            
            output = model.generate(
                **inputs, 
                max_new_tokens=20,  # Very small output
                temperature=0.0,    # Greedy decoding (no sampling)
                do_sample=False,    # No sampling to save memory
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=False,    # Disable KV cache to save memory
            )
        
        # Decode
        response = processor.decode(output[0], skip_special_tokens=True)
        print(f"✅ Test inference successful!")
        
        # Extract just the assistant's response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        print(f"Response: {response}")
        
        # STEP 4: Aggressive cleanup
        del inputs, output
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Test inference failed: {e}")
        # Emergency cleanup
        torch.cuda.empty_cache()
        gc.collect()
        return False

def test_minimal_inference(model, processor):
    """Ultra-minimal inference test without image"""
    try:
        print("Testing text-only inference...")
        
        # Text-only test to check if model works
        text_prompt = "Hello, how are you?"
        inputs = processor.tokenizer(text_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        response = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"✅ Text-only inference successful!")
        print(f"Response: {response}")
        
        del inputs, output
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"❌ Text-only inference failed: {e}")
        return False

def find_target_modules(model):
    """
    Utility to find all linear layer names in the model
    Useful for determining LoRA target modules
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

def get_memory_usage():
    """Helper function to print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

# --- Test/Demo ---
if __name__ == '__main__':
    print("--- Testing Modern LLaVA QLoRA Setup with CPU Offloading ---")
    
    # Check compute capabilities
    current_torch_dtype = torch.bfloat16
    if not torch.cuda.is_available():
        print("CUDA not available! QLoRA requires GPU.")
        current_torch_dtype = torch.float16
    elif not torch.cuda.is_bf16_supported():
        print("BF16 not supported, using FP16")
        current_torch_dtype = torch.float16
    
    # Print initial memory usage
    if torch.cuda.is_available():
        print(f"Initial GPU memory usage:")
        get_memory_usage()
    
    try:
        print(f"\n--- Loading LLaVA with QLoRA (dtype: {current_torch_dtype}) ---")
        
        model, processor = load_llava_qlora_model_modern(
            use_qlora=True,
            torch_dtype_for_training=current_torch_dtype,
            lora_r=4,  # Keep small rank
            lora_alpha=8
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Processor type: {type(processor)}")
        
        # Check memory after loading
        if torch.cuda.is_available():
            print("Memory usage after model loading:")
            get_memory_usage()
        
        # Test 1: Minimal text-only inference
        print("\n--- Testing Text-Only Inference ---")
        test_minimal_inference(model, processor)
        
        # Test 2: Full inference with CPU offloading
        print("\n--- Testing CPU Offloading Inference ---")
        test_model_inference_with_cpu_offload(model, processor)
        
        # Optional: Show target modules
        print(f"\n--- Available Target Modules ---")
        try:
            target_modules = find_target_modules(model)
            print(f"   Found modules: {target_modules}")
        except Exception as e:
            print(f"   Could not determine target modules: {e}")
        
        # Final memory check
        if torch.cuda.is_available():
            print("Final memory usage:")
            get_memory_usage()
        
        # Cleanup
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Test Complete ---")