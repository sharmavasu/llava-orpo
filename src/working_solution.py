# definitive_fix.py
# The REAL working solution - manually add <image> token

import sys
sys.path.append("src")

def test_with_manual_image_token():
    """Test with manually added <image> token - this WILL work"""
    print("🧪 Testing with MANUAL <image> token...")
    
    try:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        from PIL import Image
        import numpy as np
        import torch
        
        # Load processor and model
        print("Loading processor...")
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        
        print("Loading model...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Create test image
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # THE KEY FIX: Manually add <image> token to text
        text_with_image_token = "<image>\nWhat do you see?"
        
        print(f"Text with manual <image> token: {text_with_image_token}")
        
        # Process with the manual <image> token
        inputs = processor(
            text=text_with_image_token,
            images=image,
            return_tensors="pt"
        )
        
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
        
        # Check if we have image tokens now
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        has_image_token = (inputs['input_ids'] == image_token_id).any()
        print(f"Contains <image> token: {has_image_token}")
        
        # Generate
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print("Generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Manual <image> token inference successful!")
        print(f"Response: {response}")
        
        return True, text_with_image_token
        
    except Exception as e:
        print(f"❌ Manual image token test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_conversation_template_fixed():
    """Test conversation template approach - should also work"""
    print("🧪 Testing FIXED conversation template...")
    
    try:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        from PIL import Image
        import numpy as np
        import torch
        
        # Load model components
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Create image
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Use conversation template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image"},
                ],
            },
        ]
        
        # Generate prompt with <image> token
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        print(f"Generated prompt: {prompt[:100]}...")
        print(f"Contains <image>: {'<image>' in prompt}")
        
        # Process
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        
        # Generate
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Conversation template inference successful!")
        print(f"Response: {response}")
        
        return True, prompt
        
    except Exception as e:
        print(f"❌ Conversation template test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def create_working_data_loader_fix():
    """Create the definitive working data loader fix"""
    
    print("\n" + "="*60)
    print("📝 DEFINITIVE DATA LOADER FIX")
    print("="*60)
    
    fix_code = '''
# DEFINITIVE FIX for src/data_loader.py format_sample_for_training method

def format_sample_for_training(self, sample: Dict) -> Dict:
    """
    Format a single sample for ORPO training - DEFINITIVE WORKING VERSION
    """
    if not self.processor:
        raise ValueError("Processor must be provided for formatting")
    
    try:
        question = sample["question"]
        
        # APPROACH 1: Manual <image> token (simplest, most reliable)
        text_with_image = f"<image>\\n{question}"
        
        inputs = self.processor(
            text=text_with_image,
            images=sample["image"],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048
        )
        
        # OR APPROACH 2: Conversation template (more complex but proper)
        # conversation = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": question},
        #             {"type": "image"},
        #         ],
        #     },
        # ]
        # prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # inputs = self.processor(text=prompt, images=sample["image"], return_tensors="pt", truncation=True, max_length=2048)
        
        formatted_sample = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
            "question": question,
            "ds_name": sample.get("ds_name", "unknown"),
            "origin_dataset": sample.get("origin_dataset", "unknown")
        }
        
        return formatted_sample
        
    except Exception as e:
        logger.error(f"Error formatting sample: {e}")
        raise
'''
    
    print(fix_code)
    return fix_code

def main():
    """Test both working approaches"""
    print("🚀 Testing DEFINITIVE Working Solutions")
    
    print("\n" + "="*60)
    print("TEST 1: Manual <image> Token")
    print("="*60)
    
    success1, text1 = test_with_manual_image_token()
    
    if success1:
        print(f"✅ SUCCESS! Manual approach works with: {text1}")
    
    print("\n" + "="*60)
    print("TEST 2: Conversation Template")
    print("="*60)
    
    success2, text2 = test_conversation_template_fixed()
    
    if success2:
        print(f"✅ SUCCESS! Conversation template works!")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if success1 or success2:
        print("🎉 WE HAVE WORKING SOLUTION(S)!")
        
        if success1:
            print("✅ Manual <image> token approach works")
        if success2:
            print("✅ Conversation template approach works")
        
        print("\n📋 NEXT STEPS:")
        print("1. Apply the working fix to your src/data_loader.py")
        print("2. Use the definitive fix code shown below")
        print("3. Test training: python src/train.py --max_samples 3 --max_steps 1 --no_wandb")
        
        create_working_data_loader_fix()
        
    else:
        print("❌ Both approaches failed - deeper investigation needed")

if __name__ == "__main__":
    main()