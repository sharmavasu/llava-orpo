# src/model_utils.py

import torch
from transformers import BitsAndBytesConfig # Now we will definitely use this
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from llava.model.builder import load_pretrained_model 
from llava.mm_utils import get_model_name_from_path

def load_llava_qlora_model(
    model_path: str = "liuhaotian/llava-v1.6-mistral-7b",
    use_qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                 "gate_proj", "up_proj", "down_proj"],
    torch_dtype_for_training=torch.bfloat16
):
    print(f"Loading LLaVA model from: {model_path}")

    model_name = get_model_name_from_path(model_path)
    
    # Prepare kwargs for load_pretrained_model
    kwargs_for_llava_loader = {
        "device_map": "cuda:0",
        "torch_dtype": torch.float32,
    }

    if use_qlora:
        print(f"Attempting QLoRA: r={lora_r}, alpha={lora_alpha}, target_modules={lora_target_modules}")
        print(f"Compute dtype for QLoRA (and LoRA weights) will be: {torch_dtype_for_training}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # This flag within bnb_config is fine
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype_for_training,
            bnb_4bit_use_double_quant=True,
        )
        # Pass the bnb_config object via quantization_config
        kwargs_for_llava_loader["quantization_config"] = bnb_config
        # Ensure load_4bit/load_8bit flags are NOT passed if quantization_config is present
        # LLaVA's script might not explicitly look for quantization_config for the LLM,
        # but Hugging Face's from_pretrained (which LLaVA calls) does.
        # We must ensure load_4bit is False or not present in kwargs if quantization_config is.
        kwargs_for_llava_loader["load_4bit"] = False 
        kwargs_for_llava_loader["load_8bit"] = False

        print("Using explicit BitsAndBytesConfig for QLoRA.")

    else: # Not using QLoRA
        kwargs_for_llava_loader["load_4bit"] = False
        kwargs_for_llava_loader["load_8bit"] = False
        print(f"Loading base LLaVA model with dtype: {torch_dtype_for_training} (no QLoRA)")

    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            **kwargs_for_llava_loader # Pass all prepared arguments
        )
        if use_qlora:
            print("Base LLaVA model loaded with explicit quantization_config for QLoRA.")
        else:
            print("Base LLaVA model loaded.")

    except Exception as e:
        print(f"Error during LLaVA's load_pretrained_model call: {e}")
        raise

    model_config = model.config

    if use_qlora:
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("Model prepared for k-bit training (QLoRA with gradient checkpointing).")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("LoRA applied to the model.")
        model.print_trainable_parameters()

    return model, tokenizer, image_processor, model_config, context_len

# --- __main__ test block ---
if __name__ == '__main__':
    print("--- Testing LLaVA Model Loading Utility ---")
    
    current_torch_dtype = torch.bfloat16
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        print("BF16 not supported or CUDA not available, falling back to FP16 for compute_dtype.")
        current_torch_dtype = torch.float16
    elif not torch.cuda.is_available():
        print("CUDA not available, QLoRA test will likely fail or run on CPU very slowly.")
        # QLoRA fundamentally relies on GPU.

    try:
        print(f"\n--- Test 1: Loading with QLoRA (compute dtype: {current_torch_dtype}) ---")
        model, tokenizer, image_processor, config, context_len = load_llava_qlora_model(
            use_qlora=True,
            torch_dtype_for_training=current_torch_dtype
        )
        print("QLoRA Model loaded successfully.")
        # ... (rest of the print statements and cleanup) ...
        print(f"  Model type: {type(model)}")
        print(f"  Tokenizer type: {type(tokenizer)}")
        print(f"  Image processor type: {type(image_processor)}")
        print(f"  Context length: {context_len}")
        if hasattr(model, 'device'): # For PeftModel, device is on base_model
             print(f"  PEFT Model overall device (might be 'auto' or a specific device): {model.device}")
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'model') and hasattr(model.base_model.model.model, 'device'):
             print(f"  PEFT Model - Underlying LLM device: {model.base_model.model.model.device}")


        del model, tokenizer, image_processor, config
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    except Exception as e:
        print(f"Error during QLoRA loading test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Model Loading Utility Test Complete ---")