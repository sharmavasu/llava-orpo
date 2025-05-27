# simple_train_test.py  
# Simplified training test to bypass the data processing issue

import sys
sys.path.append("src")

def test_basic_orpo_setup():
    """Test basic ORPO setup without complex data processing"""
    print("🧪 Testing basic ORPO setup...")
    
    try:
        from model_utils import LLaVAModelManager
        from orpo_trainer import ORPOLoss, ORPOTrainer
        import torch
        
        # Test ORPO loss function
        print("Testing ORPO loss...")
        orpo_loss = ORPOLoss(beta=0.1, alpha=1.0)
        
        # Create dummy data for testing
        batch_size, seq_len, vocab_size = 2, 10, 1000
        
        chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
        rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
        chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        loss, loss_dict = orpo_loss.compute_loss(
            model=None,
            chosen_logits=chosen_logits,
            rejected_logits=rejected_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            attention_mask_chosen=attention_mask,
            attention_mask_rejected=attention_mask
        )
        
        print(f"✅ ORPO loss computation successful: {loss.item():.4f}")
        print(f"Loss components: {loss_dict}")
        
        return True
        
    except Exception as e:
        print(f"❌ ORPO setup test failed: {e}")
        return False

def test_model_with_dummy_data():
    """Test model loading and inference with dummy data"""
    print("🧪 Testing model with dummy data...")
    
    try:
        from model_utils import LLaVAModelManager
        import torch
        from PIL import Image
        import numpy as np
        
        # Load model
        model_manager = LLaVAModelManager()
        model, processor = model_manager.load_model_and_processor(
            use_quantization=True,
            use_lora=True
        )
        print("✅ Model loaded successfully")
        
        # Create dummy image and text
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        dummy_question = "What do you see in this image?"
        
        # Test simple processing
        print("Testing simple processor usage...")
        inputs = processor(
            text=dummy_question,
            images=dummy_image,
            return_tensors="pt"
        )
        print(f"✅ Processing successful - Input IDs shape: {inputs['input_ids'].shape}")
        
        # Test inference
        print("Testing inference...")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Inference successful!")
        print(f"Response: {response}")
        
        # Cleanup
        model_manager.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_minimal_training_data():
    """Create minimal training data for testing"""
    print("🧪 Creating minimal training data...")
    
    try:
        import torch
        from datasets import Dataset
        from PIL import Image
        import numpy as np
        
        # Create dummy samples
        samples = []
        for i in range(3):
            # Create dummy image
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            sample = {
                "input_ids": torch.randint(1, 1000, (50,)),
                "attention_mask": torch.ones(50),
                "pixel_values": torch.randn(3, 336, 336),
                "chosen": f"This is a good response {i}.",
                "rejected": f"This is a bad response {i}.",
                "question": f"What is in image {i}?",
                "ds_name": "test",
                "origin_dataset": "test"
            }
            samples.append(sample)
        
        # Create dataset
        dataset = Dataset.from_list(samples)
        print(f"✅ Created dataset with {len(dataset)} samples")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Failed to create training data: {e}")
        return None

def test_minimal_training():
    """Test training with minimal setup"""
    print("🧪 Testing minimal training setup...")
    
    try:
        from model_utils import LLaVAModelManager
        from orpo_trainer import ORPOTrainer, ORPOCollator
        from transformers import TrainingArguments
        
        # Load model
        model_manager = LLaVAModelManager()
        model, processor = model_manager.load_model_and_processor(
            use_quantization=True,
            use_lora=True
        )
        print("✅ Model loaded")
        
        # Create minimal dataset
        train_dataset = create_minimal_training_data()
        if not train_dataset:
            return False
        
        print("✅ Training data created")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir="./test_output",
            max_steps=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_steps=1000,  # Don't save during test
            learning_rate=5e-6,
            remove_unused_columns=False,
            report_to=[],  # No logging
        )
        
        # Create collator
        data_collator = ORPOCollator(processor=processor)
        
        # Create trainer
        trainer = ORPOTrainer(
            model=model,
            processor=processor,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        print("✅ Trainer created")
        
        # Try one training step
        print("Running one training step...")
        trainer.train()
        print("✅ Training step completed!")
        
        # Cleanup
        model_manager.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Simplified Training Setup")
    
    tests = [
        ("ORPO Setup", test_basic_orpo_setup),
        ("Model + Dummy Data", test_model_with_dummy_data),
        ("Minimal Training", test_minimal_training)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print('='*60)
        
        if test_func():
            passed += 1
            print(f"✅ {name} test PASSED")
        else:
            print(f"❌ {name} test FAILED")
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The core training setup works.")
        print("The issue is likely in the data processing. Let's fix that next.")
    else:
        print("⚠️  Some core components failed. Fix these first.")