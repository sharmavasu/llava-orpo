# quick_fix_data_loader.py
# Simplified data loader that should work immediately

import sys
sys.path.append("src")

from data_loader import RLAIFVDataLoader
import torch
from PIL import Image

class SimplifiedRLAIFVDataLoader(RLAIFVDataLoader):
    """Simplified version that bypasses complex processor usage"""
    
    def format_sample_for_training(self, sample):
        """Simplified formatting that avoids processor issues"""
        
        # Create very basic inputs
        question = sample["question"]
        
        # Use tokenizer directly instead of full processor
        text_inputs = self.processor.tokenizer(
            question,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=256
        )
        
        # Process image separately
        image_inputs = self.processor.image_processor(
            sample["image"],
            return_tensors="pt"
        )
        
        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0), 
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
            "question": question,
            "ds_name": sample.get("ds_name", "unknown"),
            "origin_dataset": sample.get("origin_dataset", "unknown")
        }

def test_simplified_loader():
    """Test the simplified loader"""
    print("🧪 Testing simplified data loader...")
    
    try:
        from model_utils import LLaVAModelManager
        
        # Load processor
        model_manager = LLaVAModelManager()
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        
        # Create simplified loader
        data_loader = SimplifiedRLAIFVDataLoader(
            processor=processor,
            max_samples=1
        )
        
        # Load data
        raw_dataset = data_loader.load_raw_dataset()
        train_dataset, eval_dataset, test_dataset = data_loader.create_training_dataset(raw_dataset)
        
        print(f"✅ Simplified loader works! Created {len(train_dataset)} train samples")
        
        # Test sample structure
        sample = train_dataset[0]
        print("Sample structure:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {str(value)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Simplified loader failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Quick Fix Data Loader")
    
    if test_simplified_loader():
        print("\n✅ Quick fix works!")
        print("\nTo use this fix:")
        print("1. Copy the SimplifiedRLAIFVDataLoader class into your src/data_loader.py")
        print("2. Or use this file directly in your training")
        print("3. This bypasses the complex processor usage that's causing issues")
    else:
        print("\n❌ Even the simplified version failed")
        print("We need to debug further with the debug scripts")