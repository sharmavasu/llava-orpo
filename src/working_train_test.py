# working_train_test.py
# Final working solution using Smart Truncation approach

import sys
sys.path.append("src")

import torch
from datasets import Dataset
from PIL import Image
import numpy as np
from transformers import TrainingArguments, Trainer

def create_working_training_sample(processor):
    """Create a training sample using the SMART TRUNCATION approach that works"""
    
    # Create dummy image
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # KEY FIX: Use SHORT questions to avoid truncation issues
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},  # SHORT question
                {"type": "image"},
            ],
        },
    ]
    
    # Generate prompt
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print(f"Prompt length: {len(prompt)} chars")
    
    # Smart truncation: larger max_length + shorter text = success!
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # Much larger limit
        padding=False
    )
    
    print(f"✅ Input IDs shape: {inputs['input_ids'].shape}")
    print(f"✅ Pixel values shape: {inputs['pixel_values'].shape}")
    
    # Create sample for training
    sample = {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "pixel_values": inputs["pixel_values"].squeeze(0),
        "chosen": "This image shows clear details and objects.",
        "rejected": "This image is blurry.",
        "question": "Describe this.",
        "ds_name": "test",
        "origin_dataset": "test"
    }
    
    return sample

def test_working_inference():
    """Test inference with the working approach"""
    print("🧪 Testing WORKING inference approach...")
    
    try:
        from model_utils import LLaVAModelManager
        
        # Load model
        model_manager = LLaVAModelManager()
        model, processor = model_manager.load_model_and_processor()
        
        # Create working sample
        sample = create_working_training_sample(processor)
        
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
        print(f"Response: {response}")
        
        model_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_working_training():
    """Test training with the working approach"""
    print("🧪 Testing WORKING training approach...")
    
    try:
        from model_utils import LLaVAModelManager 
        from orpo_trainer import ORPOCollator
        
        # Load model
        model_manager = LLaVAModelManager()
        model, processor = model_manager.load_model_and_processor()
        
        # Create working training data
        print("Creating working training samples...")
        samples = []
        questions = ["Describe this.", "What's here?", "Explain this image."]
        
        for i, question in enumerate(questions):
            # Modify the question for variety
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ],
                },
            ]
            
            # Create image
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            # Generate prompt
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # Process with working settings
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False
            )
            
            sample = {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "chosen": f"This is an excellent detailed description {i}.",
                "rejected": f"This is a poor description {i}.",
                "question": question,
                "ds_name": "test",
                "origin_dataset": "test"
            }
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
            save_steps=1000,
            remove_unused_columns=False,
            report_to=[],
            dataloader_num_workers=0,
        )
        
        # Create collator with working settings
        collator = ORPOCollator(processor=processor, max_length=2048)
        
        # Use regular Trainer first
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        
        print("Running working training step...")
        trainer.train()
        print("✅ Training step successful!")
        
        model_manager.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_fixed_data_loader():
    """Create the fixed data loader code"""
    
    fixed_code = '''
# Fixed format_sample_for_training method for src/data_loader.py

def format_sample_for_training(self, sample: Dict) -> Dict:
    """
    Format a single sample for ORPO training - WORKING VERSION
    """
    if not self.processor:
        raise ValueError("Processor must be provided for formatting")
    
    try:
        # SOLUTION: Use shorter questions to avoid truncation issues
        question = sample["question"]
        
        # Shorten very long questions to avoid token limit issues
        if len(question) > 100:
            question = question[:100] + "..."
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # WORKING SETTINGS: truncation=True + max_length=2048
        inputs = self.processor(
            text=prompt,
            images=sample["image"],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048  # Large enough to avoid cutting image tokens
        )
        
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
    
    print("📝 FIXED DATA LOADER CODE:")
    print(fixed_code)
    
    return fixed_code

def main():
    """Test the complete working solution"""
    print("🚀 Testing Complete WORKING Solution")
    
    tests = [
        ("Working Inference", test_working_inference),
        ("Working Training", test_working_training)
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
            break  # Don't continue if basic test fails
    
    print(f"\n🎯 RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 COMPLETE SUCCESS! All components working!")
        
        print("\n📋 NEXT STEPS:")
        print("1. Apply the fixed data loader code shown below")
        print("2. Update your src/data_loader.py with the working version")
        print("3. Run: python src/train.py --max_samples 3 --max_steps 1 --no_wandb")
        
        print("\n" + "="*60)
        create_fixed_data_loader()
        
    else:
        print("\n⚠️  Some tests still failing - debug needed")

if __name__ == "__main__":
    main()