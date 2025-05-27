# src/model_utils_cpu.py
# CPU-only LLaVA for limited GPU memory situations

import torch
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

def load_llava_cpu_mode(
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    use_qlora: bool = True,
    torch_dtype=torch.float32  # CPU needs float32, not float16
):
    """
    Load LLaVA in CPU-only mode for systems with limited GPU memory
    Will be slower but works with any amount of RAM
    """
    print(f"Loading LLaVA model: {model_id}")
    print("🚨 CPU-ONLY MODE - This will be slow but should work!")
    
    # Load processor
    processor = LlavaNextProcessor.from_pretrained(model_id)
    print("✅ Processor loaded")
    
    if use_qlora:
        print("⚠️  QLoRA training requires GPU - loading without QLoRA for CPU mode")
        use_qlora = False
    
    # Load model on CPU
    print("Loading model on CPU... (this may take a while)")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",  # Force CPU
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    print("✅ Model loaded on CPU")
    
    return model, processor

def load_llava_minimal_gpu(
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    max_gpu_memory: str = "500MB",  # Very conservative
    torch_dtype=torch.float16
):
    """
    Try to use minimal GPU memory with aggressive CPU offloading
    """
    print(f"Loading LLaVA with minimal GPU usage: {max_gpu_memory}")
    
    processor = LlavaNextProcessor.from_pretrained(model_id)
    
    # Very aggressive memory management
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        max_memory={0: max_gpu_memory, "cpu": "16GB"},
        offload_folder="./offload_weights"  # Disk offloading
    )
    
    return model, processor

def test_simple_inference(model, processor, device="cpu"):
    """Simple test that works on CPU"""
    try:
        print(f"\n🧪 Testing inference on {device}...")
        
        # Simple text-only test first
        test_text = "Describe the sky:"
        inputs = processor.tokenizer(
            test_text,
            return_tensors="pt"
        )
        
        # Ensure inputs are on the right device and dtype
        if device == "cpu":
            # For CPU, ensure float32
            inputs = {k: v.to(device).float() if v.dtype == torch.float16 else v.to(device) 
                     for k, v in inputs.items()}
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print("Generating response... (may take 30-60 seconds on CPU)")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Inference successful!")
        print(f"   Input: {test_text}")
        print(f"   Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

if __name__ == '__main__':
    print("--- CPU-Only LLaVA Test ---")
    print("⚠️  This will be SLOW but should work with limited GPU memory")
    
    # Force CPU mode for systems with limited GPU memory
    print("🚨 Forcing CPU-only mode due to limited GPU memory")
    use_cpu = True
    
    try:
        if use_cpu:
            print("\n🔄 Loading in CPU-only mode...")
            model, processor = load_llava_cpu_mode(torch_dtype=torch.float32)  # Use float32 for CPU
            test_device = "cpu"
        else:
            print("\n🔄 Trying minimal GPU mode...")
            model, processor = load_llava_minimal_gpu(max_gpu_memory="800MB")
            test_device = "cuda"
        
        print(f"\n✅ Model loaded successfully!")
        print(f"   Model device: {next(model.parameters()).device}")
        
        # Test inference
        test_simple_inference(model, processor, test_device)
        
        print(f"\n🧹 Cleaning up...")
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()