# debug_data.py
# Debug data processing issues

import sys
sys.path.append("src")

def debug_single_sample():
    """Debug processing of a single sample"""
    print("🔍 Debugging single sample processing...")
    
    try:
        from data_loader import RLAIFVDataLoader
        from model_utils import LLaVAModelManager
        
        # Load processor only (no model needed for this test)
        print("Loading processor...")
        model_manager = LLaVAModelManager()
        compute_dtype = model_manager.determine_optimal_dtype()
        
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        print("✅ Processor loaded")
        
        # Load single sample
        print("Loading sample from dataset...")
        data_loader = RLAIFVDataLoader(processor=processor, max_samples=1)
        raw_dataset = data_loader.load_raw_dataset()
        
        # Get first sample
        sample = list(raw_dataset)[0]
        print("✅ Sample loaded")
        
        print("Sample structure:")
        for key, value in sample.items():
            if key == "image":
                print(f"  {key}: {type(value)} - {value.size if hasattr(value, 'size') else 'Unknown'}")
            else:
                print(f"  {key}: {str(value)[:100]}...")
        
        # Test validation
        print("\nTesting sample validation...")
        is_valid = data_loader.validate_sample(sample)
        print(f"Sample valid: {is_valid}")
        
        if not is_valid:
            print("❌ Sample validation failed - checking why...")
            required_fields = ["image", "question", "chosen", "rejected"]
            for field in required_fields:
                if not sample.get(field):
                    print(f"  Missing field: {field}")
            
            # Check text lengths
            question = sample.get("question", "")
            chosen = sample.get("chosen", "")
            rejected = sample.get("rejected", "")
            
            print(f"  Question length: {len(question)}")
            print(f"  Chosen length: {len(chosen)}")
            print(f"  Rejected length: {len(rejected)}")
            
            return False
        
        # Test formatting
        print("\nTesting sample formatting...")
        try:
            formatted = data_loader.format_sample_for_training(sample)
            print("✅ Sample formatting successful!")
            
            print("Formatted sample structure:")
            for key, value in formatted.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {type(value)} - shape {value.shape}")
                else:
                    print(f"  {key}: {type(value)} - {str(value)[:50]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Sample formatting failed: {e}")
            
            # Try simple processor test
            print("\nTrying simple processor test...")
            try:
                simple_inputs = processor(
                    text="What is in this image?",
                    images=sample["image"],
                    return_tensors="pt"
                )
                print("✅ Simple processor test passed")
                print(f"Input IDs shape: {simple_inputs['input_ids'].shape}")
                print(f"Pixel values shape: {simple_inputs['pixel_values'].shape}")
                
            except Exception as simple_e:
                print(f"❌ Simple processor test failed: {simple_e}")
            
            return False
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processor_modes():
    """Test different processor usage modes"""
    print("🔍 Testing different processor modes...")
    
    try:
        from transformers import LlavaNextProcessor
        from datasets import load_dataset
        from PIL import Image
        
        # Load processor
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        
        # Load sample
        dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train", streaming=True)
        sample = next(iter(dataset))
        
        image = sample["image"]
        question = sample["question"]
        
        print(f"Image type: {type(image)}")
        print(f"Question: {question[:100]}...")
        
        # Test Mode 1: Simple text + image
        print("\n--- Mode 1: Simple text + image ---")
        try:
            inputs1 = processor(
                text=question,
                images=image,
                return_tensors="pt"
            )
            print("✅ Mode 1 successful")
            print(f"Input IDs shape: {inputs1['input_ids'].shape}")
            
        except Exception as e:
            print(f"❌ Mode 1 failed: {e}")
        
        # Test Mode 2: With conversation template
        print("\n--- Mode 2: Conversation template ---")
        try:
            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"}
                    ]
                }
            ]
            
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            print(f"Generated prompt: {prompt[:200]}...")
            
            inputs2 = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            print("✅ Mode 2 successful")
            print(f"Input IDs shape: {inputs2['input_ids'].shape}")
            
        except Exception as e:
            print(f"❌ Mode 2 failed: {e}")
        
        # Test Mode 3: Process image and text separately
        print("\n--- Mode 3: Separate processing ---")
        try:
            # Process image
            image_inputs = processor.image_processor(image, return_tensors="pt")
            
            # Process text
            text_inputs = processor.tokenizer(question, return_tensors="pt")
            
            print("✅ Mode 3 successful")
            print(f"Image shape: {image_inputs['pixel_values'].shape}")
            print(f"Text shape: {text_inputs['input_ids'].shape}")
            
        except Exception as e:
            print(f"❌ Mode 3 failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Processor mode test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Debugging Data Processing Issues")
    
    print("\n" + "="*60)
    print("TEST 1: Single Sample Processing")
    print("="*60)
    success1 = debug_single_sample()
    
    print("\n" + "="*60)
    print("TEST 2: Processor Modes")
    print("="*60)
    success2 = test_processor_modes()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if success1 and success2:
        print("✅ All tests passed! Data processing should work now.")
    else:
        print("❌ Some tests failed. Check the errors above.")
        
    print("\nNext steps:")
    print("1. If tests pass, try training again: python src/train.py --max_samples 1 --max_steps 1 --no_wandb")
    print("2. If tests fail, we need to fix the processor usage based on the error messages.")