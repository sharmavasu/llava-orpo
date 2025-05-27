# fixed_train_test.py
# Complete working training test with all fixes applied

import sys
sys.path.append("src")

import torch
from datasets import Dataset
from PIL import Image
import numpy as np
from transformers import TrainingArguments, Trainer

def create_proper_training_sample(processor):
    """Create a training sample with proper image token handling"""
    
    # Create dummy image
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Create conversation with proper format (this is the key fix!)
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
    
    # Process inputs - this now includes <image> token properly
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        truncation=True,
        max_length=512
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

def test_model_inference_fixed():
    """Test model inference with proper image token handling"""
    print("🧪 Testing model inference with fixes...")
    
    try:
        from model_utils import LLaVAModelManager
        
        # Load model
        model_manager = LLaVAModelManager()
        model, processor = model_manager.load_model_and_processor()
        
        # Create proper sample
        sample = create_proper_training_sample(processor)
        
        # Test inference
        device = next(model.parameters()).device
        inputs = {
            "input_ids": sample["input_ids"].unsqueeze(0).to(device),
            "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
            "pixel_values": sample["pixel_values"].unsqueeze(0).to(device)
        }
        
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

def test_training_step_fixed():
    """Test a single training step with all fixes"""
    print("🧪 Testing training step with fixes...")
    
    try:
        from model_utils import LLaVAModelManager 
        from orpo_trainer import ORPOCollator
        
        # Load model
        model_manager = LLaVAModelManager()
        model, processor = model_manager.load_model_and_processor()
        
        # Create proper training data with fixes
        print("Creating training samples...")
        samples = []
        for i in range(3):
            sample = create_proper_training_sample(processor)
            # Make each sample slightly different
            sample["chosen"] = f"This is a good response {i} about what's shown."
            sample["rejected"] = f"Bad response {i}."
            samples.append(sample)
        
        train_dataset = Dataset.from_list(samples)
        print(f"✅ Created dataset with {len(train_dataset)} samples")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir="./test_output",
            max_steps=1,
            per_device_train_batch_size=1,
            learning_rate=5e-6,
            logging_steps=1,
            save_steps=1000,  # Don't save
            remove_unused_columns=False,
            report_to=[],  # No logging
            dataloader_num_workers=0,  # Avoid multiprocessing issues
        )
        
        # Create collator
        collator = ORPOCollator(processor=processor, max_length=512)
        
        # Use regular Trainer first to test basic functionality
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        
        print("Running one training step...")
        trainer.train()
        print("✅ Training step successful!")
        
        model_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orpo_training_fixed():
    """Test ORPO training with all fixes"""
    print("🧪 Testing ORPO training with fixes...")
    
    try:
        from model_utils import LLaVAModelManager
        from orpo_trainer import ORPOTrainer, ORPOCollator
        
        # Load model
        model_manager = LLaVAModelManager()
        model, processor = model_manager.load_model_and_processor()
        
        # Create training data
        samples = []
        for i in range(3):
            sample = create_proper_training_sample(processor)
            sample["chosen"] = f"Excellent detailed description {i} of the image contents."
            sample["rejected"] = f"Poor vague response {i}."
            samples.append(sample)
        
        train_dataset = Dataset.from_list(samples)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir="./test_output",
            max_steps=1,
            per_device_train_batch_size=1,
            learning_rate=5e-6,
            logging_steps=1,
            save_steps=1000,
            remove_unused_columns=False,
            report_to=[],
            dataloader_num_workers=0,
        )
        
        # Create ORPO trainer (with optimizer fix)
        trainer = ORPOTrainer(
            model=model,
            processor=processor,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=ORPOCollator(processor=processor),
        )
        
        print("Running ORPO training step...")
        trainer.train()
        print("✅ ORPO training successful!")
        
        model_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ ORPO training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all fixed tests"""
    print("🚀 Running Fixed Training Tests")
    
    tests = [
        ("Model Inference", test_model_inference_fixed),
        ("Training Step", test_training_step_fixed),
        ("ORPO Training", test_orpo_training_fixed)
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
            # Don't continue if basic tests fail
            if name == "Model Inference":
                print("⚠️  Basic inference failed - need to fix this first")
                break
    
    print(f"\n🎯 RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed >= 2:
        print("🎉 Core functionality working! You can now run:")
        print("python src/train.py --max_samples 3 --max_steps 1 --no_wandb")
    else:
        print("⚠️  Need to apply the fixes first")

if __name__ == "__main__":
    main()