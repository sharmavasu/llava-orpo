# final_fix.py
# Fix the truncation issue causing image token mismatch

import sys
sys.path.append("src")

import torch
from datasets import Dataset
from PIL import Image
import numpy as np
from transformers import TrainingArguments, Trainer

def create_proper_training_sample(processor):
    """Create a training sample with proper image token handling - NO TRUNCATION"""
    
    # Create dummy image
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Create conversation with proper format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image"},
            ],
        },
    ]
    
    # Generate proper prompt with <image> token
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print(f"Generated prompt (first 100 chars): {prompt[:100]}...")
    print(f"Full prompt length: {len(prompt)} chars")
    
    # THE KEY FIX: Don't truncate! Let the processor handle the full sequence
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        truncation=False,  # ← This is the key fix!
        padding=False      # Don't pad either
    )
    
    print(f"✅ Input IDs shape: {inputs['input_ids'].shape}")
    print(f"✅ Pixel values shape: {inputs['pixel_values'].shape}")
    
    # Create sample for training
    sample = {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "pixel_values": inputs["pixel_values"].squeeze(0),
        "chosen": "This is a clear image showing various objects and details.",
        "rejected": "This image is unclear.",
        "question": "What do you see in this image?",
        "ds_name": "test",
        "origin_dataset": "test"
    }
    
    return sample

def test_no_truncation_inference():
    """Test model inference without truncation"""
    print("🧪 Testing model inference WITHOUT truncation...")
    
    try:
        from model_utils import LLaVAModelManager
        
        # Load model
        model_manager = LLaVAModelManager()
        model, processor = model_manager.load_model_and_processor()
        
        # Create proper sample WITHOUT truncation
        sample = create_proper_training_sample(processor)
        
        # Test inference
        device = next(model.parameters()).device
        inputs = {
            "input_ids": sample["input_ids"].unsqueeze(0).to(device),
            "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
            "pixel_values": sample["pixel_values"].unsqueeze(0).to(device)
        }
        
        print(f"Input sequence length: {inputs['input_ids'].shape[1]}")
        
        print("Running inference...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Inference successful!")
        print(f"Response: {response[-100:]}")  # Last 100 chars
        
        model_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_smart_truncation():
    """Test with smart truncation that preserves image tokens"""
    print("🧪 Testing with smart truncation...")
    
    try:
        from transformers import LlavaNextProcessor
        
        # Load processor
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        
        # Create image
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Shorter question to avoid truncation issues
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},  # Much shorter text
                    {"type": "image"},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        print(f"Shorter prompt length: {len(prompt)} chars")
        
        # Now we can use reasonable truncation
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Much larger limit
            padding=False
        )
        
        print(f"✅ Smart truncation successful: {inputs['input_ids'].shape}")
        return True
        
    except Exception as e:
        print(f"❌ Smart truncation failed: {e}")
        return False

def test_alternative_approach():
    """Test alternative approach: process text and image separately"""
    print("🧪 Testing alternative approach...")
    
    try:
        from transformers import LlavaNextProcessor
        
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        
        # Create image
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Process image separately
        image_inputs = processor.image_processor(image, return_tensors="pt")
        
        # Process text with <image> token
        text = "<image>\nDescribe this image."
        text_inputs = processor.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        
        print(f"✅ Alternative approach successful:")
        print(f"  Text shape: {text_inputs['input_ids'].shape}")
        print(f"  Image shape: {image_inputs['pixel_values'].shape}")
        
        # Combine inputs
        combined_inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": image_inputs["pixel_values"]
        }
        
        return combined_inputs
        
    except Exception as e:
        print(f"❌ Alternative approach failed: {e}")
        return None

def main():
    """Test all approaches to fix the truncation issue"""
    print("🚀 Testing Truncation Fixes")
    
    tests = [
        ("No Truncation", test_no_truncation_inference),
        ("Smart Truncation", test_smart_truncation),
        ("Alternative Approach", lambda: test_alternative_approach() is not None)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print('='*60)
        
        if test_func():
            passed += 1
            print(f"✅ {name} PASSED")
        else:
            print(f"❌ {name} FAILED")
            
        # Try next approach if this one fails
        
    print(f"\n🎯 RESULTS: {passed}/{len(tests)} approaches worked")
    
    if passed > 0:
        print("\n🎉 At least one approach works!")
        print("Apply the working approach to your training code.")
    else:
        print("\n⚠️  All approaches failed - need deeper investigation")
        
    print("\n📋 RECOMMENDATIONS:")
    print("1. Use truncation=False for training (no length limit)")
    print("2. Or use max_length=2048+ with shorter questions") 
    print("3. Or process text and images separately")

if __name__ == "__main__":
    main()