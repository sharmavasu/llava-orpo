# # src/data_loader.py

# import torch
# from torch.utils.data import Dataset # Sticking with Dataset for now, can switch to Iterable if needed
# from datasets import load_dataset, Dataset as HFDataset # For converting list to HF Dataset object
# from PIL import Image

# from llava.conversation import conv_templates
# # Assuming constants.py will be in the same src directory or accessible in python path
# from .constants import DEFAULT_IMAGE_TOKEN 

# # Note: IMAGE_TOKEN_INDEX and process_images will be handled by the LLaVA utils
# # when called from __getitem__ or the main training script.

# class RLAIFVPreferenceDataset(Dataset):
#     def __init__(self, 
#                  dataset_id="openbmb/RLAIF-V-Dataset", 
#                  split="train", 
#                  tokenizer=None, # Will be passed from train.py
#                  image_processor=None, # Will be passed from train.py
#                  model_config=None, # Will be passed from train.py
#                  image_token=DEFAULT_IMAGE_TOKEN,
#                  # ds_name_filter=None, # If filtering by 'ds_name' directly
#                  origin_dataset_filter=None, # New: For filtering by 'origin_dataset'
#                  max_samples=None, # For loading a subset for debugging/testing
#                  slice_for_loading=None, # e.g. "train[:1%]" or "train[:1000]" for initial load
#                  streaming=False # Whether to use streaming for initial load
#                  ):
#         super().__init__()
        
#         if tokenizer is None or image_processor is None or model_config is None:
#             raise ValueError("Tokenizer, image_processor, and model_config must be provided.")

#         self.dataset_id = dataset_id
#         self.split_to_load = slice_for_loading if slice_for_loading else split # Use slice if provided
#         self.tokenizer = tokenizer
#         self.image_processor = image_processor
#         self.model_config = model_config 
#         self.image_token = image_token
#         self.origin_dataset_filter = origin_dataset_filter

#         print(f"Initializing RLAIFVPreferenceDataset with split: {self.split_to_load}, streaming: {streaming}")

#         if streaming:
#             print("Loading dataset in streaming mode...")
#             streamed_dataset = load_dataset(self.dataset_id, split=self.split_to_load, streaming=True)
#             # If streaming, we might need to process a certain number of samples into a list
#             # or this class would need to be an IterableDataset.
#             # For now, let's assume if streaming is True for init, we take 'max_samples' if provided.
#             temp_data_list = []
#             if max_samples:
#                 print(f"Streaming and collecting up to {max_samples} samples...")
#                 count = 0
#                 for example in streamed_dataset:
#                     temp_data_list.append(example)
#                     count += 1
#                     if count >= max_samples:
#                         break
#                 self.processed_data = HFDataset.from_list(temp_data_list)
#             else:
#                 # Streaming all data into memory like this for a Dataset is not truly streaming.
#                 # If the goal is to process the whole dataset via streaming, RLAIFVPreferenceDataset
#                 # should inherit from IterableDataset. For now, this 'streaming' flag is mostly for
#                 # initial loading of a subset if direct slicing fails.
#                 print("Warning: Streaming all data into memory for map-style Dataset. Consider IterableDataset for true streaming.")
#                 self.processed_data = HFDataset.from_list(list(streamed_dataset)) # Potentially very large!
#         else:
#             print(f"Loading dataset with direct slice: {self.split_to_load}")
#             full_dataset = load_dataset(self.dataset_id, split=self.split_to_load)
#             self.processed_data = full_dataset # Start with the loaded slice

#         # Filter by origin_dataset if a filter is provided
#         if self.origin_dataset_filter:
#             if isinstance(self.origin_dataset_filter, str):
#                 self.origin_dataset_filter = [self.origin_dataset_filter]
            
#             print(f"Filtering dataset for origin_dataset(s): {self.origin_dataset_filter}...")
#             self.processed_data = self.processed_data.filter(
#                 lambda example: example['origin_dataset'] in self.origin_dataset_filter
#             )
#             print(f"Filtered dataset size (by origin_dataset): {len(self.processed_data)}")
        
#         # Apply max_samples if not already applied by streaming logic and if it's smaller
#         if not streaming and max_samples is not None and max_samples < len(self.processed_data):
#             print(f"Taking a subset of {max_samples} samples from loaded/filtered data.")
#             self.processed_data = self.processed_data.select(range(max_samples))
            
#         if len(self.processed_data) == 0:
#             print("Warning: Dataset is empty after loading and filtering. Check filters and slice.")
#         else:
#             print(f"Final dataset size for this instance: {len(self.processed_data)}")

#         # Pre-fetch LLaVA utilities (or ensure they are available globally, but this is safer)
#         from llava.mm_utils import tokenizer_image_token, process_images, IMAGE_TOKEN_INDEX as LLaVA_IMAGE_TOKEN_INDEX
#         self.tokenizer_image_token_func = tokenizer_image_token
#         self.process_images_func = process_images
#         self.LLAVA_IMAGE_TOKEN_INDEX = LLaVA_IMAGE_TOKEN_INDEX


#     def __len__(self):
#         return len(self.processed_data)

#     def _get_formatted_texts_and_prompt_len(self, question, chosen_response, rejected_response):
#         # Use the conversation template for LLaVA v1.6 Mistral
#         # Needs self.image_token
        
#         # Prompt part for calculating length (for masking labels)
#         conv_prompt_structure = conv_templates["mistral_instruct"].copy()
#         conv_prompt_structure.append_message(conv_prompt_structure.roles[0], f"{self.image_token}\n{question}")
#         conv_prompt_structure.append_message(conv_prompt_structure.roles[1], None) # Assistant's turn starts here
#         prompt_text_for_masking = conv_prompt_structure.get_prompt()
        
#         # Tokenize prompt part to find its length
#         # Note: tokenizer_image_token will replace self.image_token with the actual image placeholder ID
#         # Using the passed LLaVA tokenizer
#         prompt_input_ids = self.tokenizer_image_token_func(
#             prompt_text_for_masking, self.tokenizer, self.LLAVA_IMAGE_TOKEN_INDEX, return_tensors='pt'
#         )
#         # Remove batch dimension added by return_tensors='pt' if present and get length
#         prompt_length = prompt_input_ids.squeeze().shape[0]

#         # Chosen full text
#         conv_chosen = conv_templates["mistral_instruct"].copy()
#         conv_chosen.append_message(conv_chosen.roles[0], f"{self.image_token}\n{question}")
#         conv_chosen.append_message(conv_chosen.roles[1], chosen_response)
#         chosen_full_text = conv_chosen.get_prompt()
        
#         # Rejected full text
#         conv_rejected = conv_templates["mistral_instruct"].copy()
#         conv_rejected.append_message(conv_rejected.roles[0], f"{self.image_token}\n{question}")
#         conv_rejected.append_message(conv_rejected.roles[1], rejected_response)
#         rejected_full_text = conv_rejected.get_prompt()
        
#         return chosen_full_text, rejected_full_text, prompt_length

#     def __getitem__(self, idx):
#         item = self.processed_data[idx]
        
#         pil_image = item['image']
#         if not isinstance(pil_image, Image.Image):
#             # This case should ideally not happen if datasets lib decodes correctly
#             print(f"Warning: Item at index {idx} does not contain a PIL image. Got {type(pil_image)}")
#             # Attempt to load if it's bytes, or handle error
#             if isinstance(pil_image, dict) and 'bytes' in pil_image and pil_image['bytes']:
#                 try:
#                     pil_image = Image.open(io.BytesIO(pil_image['bytes']))
#                 except Exception as e:
#                     raise ValueError(f"Could not load image from bytes at index {idx}: {e}")
#             else:
#                 raise ValueError(f"Unsupported image format or missing image at index {idx}")


#         if pil_image.mode != 'RGB':
#             pil_image = pil_image.convert('RGB')
            
#         question = item['question']
#         chosen_response = item['chosen']
#         rejected_response = item['rejected']
#         image_original_size = pil_image.size # (width, height)

#         # Process image using LLaVA's utility
#         image_tensor = self.process_images_func([pil_image], self.image_processor, self.model_config)[0]

#         # Get formatted texts and the length of the prompt part for masking labels
#         chosen_full_text, rejected_full_text, prompt_length = \
#             self._get_formatted_texts_and_prompt_len(question, chosen_response, rejected_response)

#         # Tokenize chosen
#         chosen_input_ids = self.tokenizer_image_token_func(
#             chosen_full_text, self.tokenizer, self.LLAVA_IMAGE_TOKEN_INDEX, return_tensors='pt'
#         ).squeeze(0) # Remove batch dimension
        
#         # Tokenize rejected
#         rejected_input_ids = self.tokenizer_image_token_func(
#             rejected_full_text, self.tokenizer, self.LLAVA_IMAGE_TOKEN_INDEX, return_tensors='pt'
#         ).squeeze(0) # Remove batch dimension

#         # Create labels: -100 for prompt tokens, actual token_id for response tokens
#         chosen_labels = chosen_input_ids.clone()
#         # Ensure prompt_length doesn't exceed sequence length (it shouldn't for valid data)
#         chosen_labels[:min(prompt_length, chosen_labels.shape[0])] = -100 
        
#         rejected_labels = rejected_input_ids.clone()
#         rejected_labels[:min(prompt_length, rejected_labels.shape[0])] = -100
        
#         chosen_attention_mask = torch.ones_like(chosen_input_ids)
#         rejected_attention_mask = torch.ones_like(rejected_input_ids)

#         return {
#             "image_tensor": image_tensor,
#             "image_sizes": torch.tensor([image_original_size], dtype=torch.long), # Needs to be tensor for collator
            
#             "chosen_input_ids": chosen_input_ids,
#             "chosen_attention_mask": chosen_attention_mask,
#             "chosen_labels": chosen_labels,
            
#             "rejected_input_ids": rejected_input_ids,
#             "rejected_attention_mask": rejected_attention_mask,
#             "rejected_labels": rejected_labels,
#         }

# # You would not typically run __main__ from src/data_loader.py directly in a project.
# # This is for initial testing. You'd import and use RLAIFVPreferenceDataset in train.py.
# # To test this file standalone, you'd need to:
# # 1. Ensure LLaVA is installed and its paths are correct for imports.
# # 2. Create mock or load real tokenizer, image_processor, and model_config.
# # This is non-trivial for LLaVA due to its integrated components.
# # The best place to test the Dataset instance is after loading these in your train.py.

# src/data_loader.py
# RLAIF-V dataset loading and preprocessing for ORPO training

# import torch
# from datasets import load_dataset, Dataset
# from PIL import Image
# import numpy as np
# from typing import Dict, List, Tuple, Optional, Union
# import logging
# from pathlib import Path
# import json

# from constants import (
#     DATASET_CONFIGS, 
#     DATA_PROCESSING_CONFIGS, 
#     VALIDATION_CONFIGS,
#     TRAINING_HYPERPARAMS
# )

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class RLAIFVDataLoader:
#     """
#     Handles loading and preprocessing of RLAIF-V dataset for ORPO training
#     """
    
#     def __init__(
#         self, 
#         processor=None,
#         config: Optional[Dict] = None,
#         streaming: bool = True,
#         max_samples: Optional[int] = None
#     ):
#         self.processor = processor
#         self.config = config or DATA_PROCESSING_CONFIGS
#         self.streaming = streaming
#         self.max_samples = max_samples
        
#         # Dataset info
#         self.dataset_config = DATASET_CONFIGS["rlaif_v"]
#         self.validation_config = VALIDATION_CONFIGS
        
#         logger.info(f"Initialized RLAIFVDataLoader with streaming={streaming}")
        
#     def load_raw_dataset(self, split: str = "train") -> Dataset:
#         """
#         Load raw RLAIF-V dataset
        
#         Args:
#             split: Dataset split to load
            
#         Returns:
#             Dataset object
#         """
#         dataset_id = self.dataset_config["dataset_id"]
#         logger.info(f"Loading {dataset_id} split='{split}'")
        
#         try:
#             if self.streaming:
#                 # Use streaming for memory efficiency
#                 dataset = load_dataset(dataset_id, split=split, streaming=True)
                
#                 # Convert to regular dataset if max_samples specified
#                 if self.max_samples:
#                     logger.info(f"Loading first {self.max_samples} samples via streaming")
#                     samples = []
#                     for i, example in enumerate(dataset):
#                         if i >= self.max_samples:
#                             break
#                         samples.append(example)
#                     dataset = Dataset.from_list(samples)
#                     logger.info(f"Created dataset with {len(dataset)} samples")
#                 else:
#                     logger.info("Using streaming dataset (full dataset)")
#             else:
#                 # Load full dataset into memory
#                 dataset = load_dataset(dataset_id, split=split)
#                 if self.max_samples:
#                     dataset = dataset.select(range(min(self.max_samples, len(dataset))))
#                 logger.info(f"Loaded {len(dataset)} samples into memory")
                
#             return dataset
            
#         except Exception as e:
#             logger.error(f"Failed to load dataset: {e}")
#             raise
    
#     def analyze_dataset(self, dataset: Dataset, num_samples: int = 100) -> Dict:
#         """
#         Analyze dataset structure and quality
        
#         Args:
#             dataset: Dataset to analyze
#             num_samples: Number of samples to analyze
            
#         Returns:
#             Analysis results dictionary
#         """
#         logger.info("Analyzing dataset structure and quality")
        
#         analysis = {
#             "total_samples": 0,
#             "missing_data": {},
#             "text_lengths": {},
#             "unique_sources": set(),
#             "sample_fields": []
#         }
        
#         # Handle streaming vs regular dataset
#         if hasattr(dataset, '__len__'):
#             analysis["total_samples"] = len(dataset)
#             sample_size = min(num_samples, len(dataset))
#             samples_to_analyze = dataset.select(range(sample_size))
#         else:
#             # Streaming dataset - collect samples for analysis
#             samples_list = []
#             for i, example in enumerate(dataset):
#                 if i >= num_samples:
#                     break
#                 samples_list.append(example)
#             samples_to_analyze = Dataset.from_list(samples_list)
#             analysis["total_samples"] = f"streaming (analyzed {len(samples_to_analyze)})"
        
#         # Get field names from first sample
#         if len(samples_to_analyze) > 0:
#             first_sample = samples_to_analyze[0]
#             analysis["sample_fields"] = list(first_sample.keys())
            
#             # Initialize missing data counters
#             for field in ["image", "question", "chosen", "rejected", "ds_name"]:
#                 analysis["missing_data"][field] = 0
        
#         # Analyze samples
#         text_lengths = {"question": [], "chosen": [], "rejected": []}
        
#         for sample in samples_to_analyze:
#             # Check for missing data
#             for field in analysis["missing_data"].keys():
#                 if not sample.get(field):
#                     analysis["missing_data"][field] += 1
            
#             # Collect text lengths
#             for text_field in text_lengths.keys():
#                 text = sample.get(text_field, "")
#                 if text:
#                     text_lengths[text_field].append(len(text))
            
#             # Collect unique sources
#             ds_name = sample.get("ds_name")
#             if ds_name:
#                 analysis["unique_sources"].add(ds_name)
        
#         # Calculate statistics
#         for field, lengths in text_lengths.items():
#             if lengths:
#                 analysis["text_lengths"][field] = {
#                     "min": np.min(lengths),
#                     "max": np.max(lengths),  
#                     "mean": np.mean(lengths),
#                     "median": np.median(lengths)
#                 }
        
#         # Convert set to list for JSON serialization
#         analysis["unique_sources"] = list(analysis["unique_sources"])
        
#         # Log key findings
#         logger.info(f"Dataset analysis complete:")
#         logger.info(f"  - Total samples: {analysis['total_samples']}")
#         logger.info(f"  - Sample fields: {analysis['sample_fields']}")
#         logger.info(f"  - Unique sources: {len(analysis['unique_sources'])}")
        
#         for field, count in analysis["missing_data"].items():
#             if count > 0:
#                 logger.warning(f"  - Missing {field}: {count} samples")
        
#         return analysis
    
#     def validate_sample(self, sample: Dict) -> bool:
#         """
#         Validate a single sample for training
        
#         Args:
#             sample: Sample to validate
            
#         Returns:
#             True if sample is valid, False otherwise
#         """
#         required_fields = ["image", "question", "chosen", "rejected"]
        
#         # Check required fields exist
#         for field in required_fields:
#             if not sample.get(field):
#                 return False
        
#         # Validate image
#         image = sample.get("image")
#         if not isinstance(image, Image.Image):
#             return False
        
#         # Validate text lengths
#         question = sample.get("question", "")
#         chosen = sample.get("chosen", "")
#         rejected = sample.get("rejected", "")
        
#         min_len = self.validation_config["min_response_length"]
#         max_len = self.validation_config["max_response_length"]
        
#         if (len(chosen) < min_len or len(chosen) > max_len or
#             len(rejected) < min_len or len(rejected) > max_len or
#             len(question) < 5):  # Minimum question length
#             return False
        
#         return True
    
#     def format_sample_for_training(self, sample: Dict) -> Dict:
#         """
#         Format a single sample for ORPO training
        
#         Args:
#             sample: Raw sample from dataset
            
#         Returns:
#             Formatted sample ready for training
#         """
#         if not self.processor:
#             raise ValueError("Processor must be provided for formatting")
        
#         # Create conversation format for LLaVA
#         conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": sample["question"]},
#                     {"type": "image"},
#                 ],
#             },
#         ]
        
#         # Generate prompt using processor
#         prompt = self.processor.apply_chat_template(
#             conversation, 
#             add_generation_prompt=True
#         )
        
#         # Process image and text
#         inputs = self.processor(
#             sample["image"],
#             prompt,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=TRAINING_HYPERPARAMS["max_length"]
#         )
        
#         # Format for ORPO training
#         formatted_sample = {
#             # Model inputs
#             "input_ids": inputs["input_ids"].squeeze(0),
#             "attention_mask": inputs["attention_mask"].squeeze(0),
#             "pixel_values": inputs["pixel_values"].squeeze(0),
            
#             # ORPO targets
#             "chosen": sample["chosen"],
#             "rejected": sample["rejected"],
            
#             # Metadata for analysis
#             "question": sample["question"],
#             "ds_name": sample.get("ds_name", "unknown"),
#             "origin_dataset": sample.get("origin_dataset", "unknown")
#         }
        
#         return formatted_sample
    
#     def create_training_dataset(
#         self, 
#         raw_dataset: Dataset,
#         split_ratios: Optional[Tuple[float, float, float]] = None
#     ) -> Tuple[Dataset, Dataset, Dataset]:
#         """
#         Create formatted training, validation, and test datasets
        
#         Args:
#             raw_dataset: Raw RLAIF-V dataset
#             split_ratios: (train, val, test) ratios
            
#         Returns:
#             Tuple of (train_dataset, val_dataset, test_dataset)
#         """
#         if not self.processor:
#             raise ValueError("Processor must be provided for dataset creation")
        
#         logger.info("Creating formatted training dataset")
        
#         # Default split ratios
#         if split_ratios is None:
#             split_ratios = (
#                 self.config["train_ratio"],
#                 self.config["val_ratio"], 
#                 self.config["test_ratio"]
#             )
        
#         formatted_samples = []
#         valid_count = 0
#         invalid_count = 0
        
#         # Process samples
#         total_samples = len(raw_dataset) if hasattr(raw_dataset, '__len__') else "unknown"
#         logger.info(f"Processing {total_samples} samples")
        
#         for i, sample in enumerate(raw_dataset):
#             try:
#                 # Validate sample
#                 if not self.validate_sample(sample):
#                     invalid_count += 1
#                     if self.validation_config["log_invalid_samples"] and invalid_count <= 5:
#                         logger.warning(f"Invalid sample {i}: missing or invalid data")
#                     continue
                
#                 # Format sample
#                 formatted = self.format_sample_for_training(sample)
#                 formatted_samples.append(formatted)
#                 valid_count += 1
                
#                 # Progress logging
#                 if (i + 1) % 100 == 0:
#                     logger.info(f"Processed {i + 1} samples, {valid_count} valid, {invalid_count} invalid")
                
#                 # Respect max_samples limit
#                 if self.max_samples and valid_count >= self.max_samples:
#                     logger.info(f"Reached max_samples limit: {self.max_samples}")
#                     break
                    
#             except Exception as e:
#                 invalid_count += 1
#                 if invalid_count <= 5:  # Log first few errors
#                     logger.error(f"Error processing sample {i}: {e}")
        
#         logger.info(f"Dataset processing complete: {valid_count} valid, {invalid_count} invalid samples")
        
#         if not formatted_samples:
#             raise ValueError("No valid samples found in dataset")
        
#         # Create HuggingFace dataset
#         full_dataset = Dataset.from_list(formatted_samples)
        
#         # Split dataset
#         total_size = len(full_dataset)
#         train_size = int(total_size * split_ratios[0])
#         val_size = int(total_size * split_ratios[1])
        
#         train_dataset = full_dataset.select(range(train_size))
#         val_dataset = full_dataset.select(range(train_size, train_size + val_size))
#         test_dataset = full_dataset.select(range(train_size + val_size, total_size))
        
#         logger.info(f"Dataset splits created:")
#         logger.info(f"  - Train: {len(train_dataset)} samples ({len(train_dataset)/total_size*100:.1f}%)")
#         logger.info(f"  - Validation: {len(val_dataset)} samples ({len(val_dataset)/total_size*100:.1f}%)")
#         logger.info(f"  - Test: {len(test_dataset)} samples ({len(test_dataset)/total_size*100:.1f}%)")
        
#         return train_dataset, val_dataset, test_dataset
    
#     def save_dataset_info(self, dataset_info: Dict, output_path: Path):
#         """Save dataset analysis to file"""
#         with open(output_path, 'w') as f:
#             json.dump(dataset_info, f, indent=2, default=str)
#         logger.info(f"Dataset info saved to {output_path}")

# # Utility functions
# def collate_fn(batch):
#     """Custom collate function for ORPO training"""
#     # Handle variable length sequences
#     input_ids = [item["input_ids"] for item in batch]
#     attention_masks = [item["attention_mask"] for item in batch]
#     pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
#     # Pad sequences
#     max_length = max(len(ids) for ids in input_ids)
    
#     padded_input_ids = []
#     padded_attention_masks = []
    
#     for ids, mask in zip(input_ids, attention_masks):
#         pad_length = max_length - len(ids)
#         padded_ids = torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)])
#         padded_mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
        
#         padded_input_ids.append(padded_ids)
#         padded_attention_masks.append(padded_mask)
    
#     return {
#         "input_ids": torch.stack(padded_input_ids),
#         "attention_mask": torch.stack(padded_attention_masks),
#         "pixel_values": pixel_values,
#         "chosen": [item["chosen"] for item in batch],
#         "rejected": [item["rejected"] for item in batch],
#         "metadata": [
#             {
#                 "question": item["question"],
#                 "ds_name": item["ds_name"],
#                 "origin_dataset": item["origin_dataset"]
#             }
#             for item in batch
#         ]
#     }

# # Testing and demo
# if __name__ == "__main__":
#     logger.info("Testing RLAIF-V Data Loader")
    
#     # Test basic loading
#     data_loader = RLAIFVDataLoader(streaming=True, max_samples=10)
    
#     try:
#         # Load raw dataset
#         raw_dataset = data_loader.load_raw_dataset()
#         logger.info("✅ Raw dataset loaded successfully")
        
#         # Analyze dataset
#         analysis = data_loader.analyze_dataset(raw_dataset, num_samples=10)
#         logger.info("✅ Dataset analysis completed")
        
#         # Print sample structure
#         if hasattr(raw_dataset, '__len__') and len(raw_dataset) > 0:
#             sample = raw_dataset[0]
#         else:
#             sample = next(iter(raw_dataset))
        
#         logger.info(f"Sample structure: {list(sample.keys())}")
#         logger.info(f"Question: {sample['question'][:100]}...")
#         logger.info(f"Chosen length: {len(sample['chosen'])}")
#         logger.info(f"Rejected length: {len(sample['rejected'])}")
        
#         logger.info("✅ Data loader test completed successfully")
        
#     except Exception as e:
#         logger.error(f"❌ Data loader test failed: {e}")
#         raise

# src/data_loader.py
# RLAIF-V dataset loading and preprocessing for ORPO training

# import torch
# from datasets import load_dataset, Dataset
# from PIL import Image
# import numpy as np
# from typing import Dict, List, Tuple, Optional, Union
# import logging
# from pathlib import Path
# import json

# from constants import (
#     DATASET_CONFIGS, 
#     DATA_PROCESSING_CONFIGS, 
#     VALIDATION_CONFIGS,
#     TRAINING_HYPERPARAMS
# )

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class RLAIFVDataLoader:
#     """
#     Handles loading and preprocessing of RLAIF-V dataset for ORPO training
#     """
    
#     def __init__(
#         self, 
#         processor=None,
#         config: Optional[Dict] = None,
#         streaming: bool = True,
#         max_samples: Optional[int] = None
#     ):
#         self.processor = processor
#         self.config = config or DATA_PROCESSING_CONFIGS
#         self.streaming = streaming
#         self.max_samples = max_samples
        
#         # Dataset info
#         self.dataset_config = DATASET_CONFIGS["rlaif_v"]
#         self.validation_config = VALIDATION_CONFIGS
        
#         logger.info(f"Initialized RLAIFVDataLoader with streaming={streaming}")
        
#     def load_raw_dataset(self, split: str = "train") -> Dataset:
#         """
#         Load raw RLAIF-V dataset
        
#         Args:
#             split: Dataset split to load
            
#         Returns:
#             Dataset object
#         """
#         dataset_id = self.dataset_config["dataset_id"]
#         logger.info(f"Loading {dataset_id} split='{split}'")
        
#         try:
#             if self.streaming:
#                 # Use streaming for memory efficiency
#                 dataset = load_dataset(dataset_id, split=split, streaming=True)
                
#                 # Convert to regular dataset if max_samples specified
#                 if self.max_samples:
#                     logger.info(f"Loading first {self.max_samples} samples via streaming")
#                     samples = []
#                     for i, example in enumerate(dataset):
#                         if i >= self.max_samples:
#                             break
#                         samples.append(example)
#                     dataset = Dataset.from_list(samples)
#                     logger.info(f"Created dataset with {len(dataset)} samples")
#                 else:
#                     logger.info("Using streaming dataset (full dataset)")
#             else:
#                 # Load full dataset into memory
#                 dataset = load_dataset(dataset_id, split=split)
#                 if self.max_samples:
#                     dataset = dataset.select(range(min(self.max_samples, len(dataset))))
#                 logger.info(f"Loaded {len(dataset)} samples into memory")
                
#             return dataset
            
#         except Exception as e:
#             logger.error(f"Failed to load dataset: {e}")
#             raise
    
#     def analyze_dataset(self, dataset: Dataset, num_samples: int = 100) -> Dict:
#         """
#         Analyze dataset structure and quality
        
#         Args:
#             dataset: Dataset to analyze
#             num_samples: Number of samples to analyze
            
#         Returns:
#             Analysis results dictionary
#         """
#         logger.info("Analyzing dataset structure and quality")
        
#         analysis = {
#             "total_samples": 0,
#             "missing_data": {},
#             "text_lengths": {},
#             "unique_sources": set(),
#             "sample_fields": []
#         }
        
#         # Handle streaming vs regular dataset
#         if hasattr(dataset, '__len__'):
#             analysis["total_samples"] = len(dataset)
#             sample_size = min(num_samples, len(dataset))
#             samples_to_analyze = dataset.select(range(sample_size))
#         else:
#             # Streaming dataset - collect samples for analysis
#             samples_list = []
#             for i, example in enumerate(dataset):
#                 if i >= num_samples:
#                     break
#                 samples_list.append(example)
#             samples_to_analyze = Dataset.from_list(samples_list)
#             analysis["total_samples"] = f"streaming (analyzed {len(samples_to_analyze)})"
        
#         # Get field names from first sample
#         if len(samples_to_analyze) > 0:
#             first_sample = samples_to_analyze[0]
#             analysis["sample_fields"] = list(first_sample.keys())
            
#             # Initialize missing data counters
#             for field in ["image", "question", "chosen", "rejected", "ds_name"]:
#                 analysis["missing_data"][field] = 0
        
#         # Analyze samples
#         text_lengths = {"question": [], "chosen": [], "rejected": []}
        
#         for sample in samples_to_analyze:
#             # Check for missing data
#             for field in analysis["missing_data"].keys():
#                 if not sample.get(field):
#                     analysis["missing_data"][field] += 1
            
#             # Collect text lengths
#             for text_field in text_lengths.keys():
#                 text = sample.get(text_field, "")
#                 if text:
#                     text_lengths[text_field].append(len(text))
            
#             # Collect unique sources
#             ds_name = sample.get("ds_name")
#             if ds_name:
#                 analysis["unique_sources"].add(ds_name)
        
#         # Calculate statistics
#         for field, lengths in text_lengths.items():
#             if lengths:
#                 analysis["text_lengths"][field] = {
#                     "min": np.min(lengths),
#                     "max": np.max(lengths),  
#                     "mean": np.mean(lengths),
#                     "median": np.median(lengths)
#                 }
        
#         # Convert set to list for JSON serialization
#         analysis["unique_sources"] = list(analysis["unique_sources"])
        
#         # Log key findings
#         logger.info(f"Dataset analysis complete:")
#         logger.info(f"  - Total samples: {analysis['total_samples']}")
#         logger.info(f"  - Sample fields: {analysis['sample_fields']}")
#         logger.info(f"  - Unique sources: {len(analysis['unique_sources'])}")
        
#         for field, count in analysis["missing_data"].items():
#             if count > 0:
#                 logger.warning(f"  - Missing {field}: {count} samples")
        
#         return analysis
    
#     def validate_sample(self, sample: Dict) -> bool:
#         """
#         Validate a single sample for training
        
#         Args:
#             sample: Sample to validate
            
#         Returns:
#             True if sample is valid, False otherwise
#         """
#         required_fields = ["image", "question", "chosen", "rejected"]
        
#         # Check required fields exist
#         for field in required_fields:
#             if not sample.get(field):
#                 return False
        
#         # Validate image
#         image = sample.get("image")
#         if not isinstance(image, Image.Image):
#             return False
        
#         # Validate text lengths
#         question = sample.get("question", "")
#         chosen = sample.get("chosen", "")
#         rejected = sample.get("rejected", "")
        
#         min_len = self.validation_config["min_response_length"]
#         max_len = self.validation_config["max_response_length"]
        
#         if (len(chosen) < min_len or len(chosen) > max_len or
#             len(rejected) < min_len or len(rejected) > max_len or
#             len(question) < 5):  # Minimum question length
#             return False
        
#         return True
    
#     def format_sample_for_training(self, sample: Dict) -> Dict:
#         """
#         Format a single sample for ORPO training
        
#         Args:
#             sample: Raw sample from dataset
            
#         Returns:
#             Formatted sample ready for training
#         """
#         if not self.processor:
#             raise ValueError("Processor must be provided for formatting")
        
#         try:
#             # Simple approach: just use the question as text input
#             # The processor will handle image tokens automatically
#             question_text = sample["question"]
            
#             # Process image and text together
#             # inputs = self.processor(
#             #     text=question_text,
#             #     images=sample["image"],
#             #     return_tensors="pt",
#             #     padding=False,  # Don't pad individual samples
#             #     truncation=False,  # Don't truncate to avoid token mismatch
#             # )
            
#             inputs = self.processor(
#                 text=prompt,
#                 images=sample["image"],
#                 return_tensors="pt",
#                 padding=False,
#                 truncation=False,  # ← No truncation!
#             )
                
            
#             # Format for ORPO training
#             formatted_sample = {
#                 # Model inputs
#                 "input_ids": inputs["input_ids"].squeeze(0),
#                 "attention_mask": inputs["attention_mask"].squeeze(0),
#                 "pixel_values": inputs["pixel_values"].squeeze(0),
                
#                 # ORPO targets
#                 "chosen": sample["chosen"],
#                 "rejected": sample["rejected"],
                
#                 # Metadata for analysis
#                 "question": sample["question"],
#                 "ds_name": sample.get("ds_name", "unknown"),
#                 "origin_dataset": sample.get("origin_dataset", "unknown")
#             }
            
#             return formatted_sample
            
#         except Exception as e:
#             # Log the specific error for debugging
#             logger.error(f"Error formatting sample: {e}")
#             logger.error(f"Sample keys: {list(sample.keys())}")
#             logger.error(f"Question: {sample.get('question', 'N/A')[:100]}...")
#             raise
    
#     def create_training_dataset(
#         self, 
#         raw_dataset: Dataset,
#         split_ratios: Optional[Tuple[float, float, float]] = None
#     ) -> Tuple[Dataset, Dataset, Dataset]:
#         """
#         Create formatted training, validation, and test datasets
        
#         Args:
#             raw_dataset: Raw RLAIF-V dataset
#             split_ratios: (train, val, test) ratios
            
#         Returns:
#             Tuple of (train_dataset, val_dataset, test_dataset)
#         """
#         if not self.processor:
#             raise ValueError("Processor must be provided for dataset creation")
        
#         logger.info("Creating formatted training dataset")
        
#         # Default split ratios
#         if split_ratios is None:
#             split_ratios = (
#                 self.config["train_ratio"],
#                 self.config["val_ratio"], 
#                 self.config["test_ratio"]
#             )
        
#         formatted_samples = []
#         valid_count = 0
#         invalid_count = 0
        
#         # Process samples
#         total_samples = len(raw_dataset) if hasattr(raw_dataset, '__len__') else "unknown"
#         logger.info(f"Processing {total_samples} samples")
        
#         for i, sample in enumerate(raw_dataset):
#             try:
#                 # Validate sample
#                 if not self.validate_sample(sample):
#                     invalid_count += 1
#                     if self.validation_config["log_invalid_samples"] and invalid_count <= 5:
#                         logger.warning(f"Invalid sample {i}: missing or invalid data")
#                     continue
                
#                 # Format sample
#                 formatted = self.format_sample_for_training(sample)
#                 formatted_samples.append(formatted)
#                 valid_count += 1
                
#                 # Progress logging
#                 if (i + 1) % 100 == 0:
#                     logger.info(f"Processed {i + 1} samples, {valid_count} valid, {invalid_count} invalid")
                
#                 # Respect max_samples limit
#                 if self.max_samples and valid_count >= self.max_samples:
#                     logger.info(f"Reached max_samples limit: {self.max_samples}")
#                     break
                    
#             except Exception as e:
#                 invalid_count += 1
#                 if invalid_count <= 5:  # Log first few errors
#                     logger.error(f"Error processing sample {i}: {e}")
        
#         logger.info(f"Dataset processing complete: {valid_count} valid, {invalid_count} invalid samples")
        
#         if not formatted_samples:
#             raise ValueError("No valid samples found in dataset")
        
#         # Create HuggingFace dataset
#         full_dataset = Dataset.from_list(formatted_samples)
        
#         # Split dataset
#         total_size = len(full_dataset)
#         train_size = int(total_size * split_ratios[0])
#         val_size = int(total_size * split_ratios[1])
        
#         train_dataset = full_dataset.select(range(train_size))
#         val_dataset = full_dataset.select(range(train_size, train_size + val_size))
#         test_dataset = full_dataset.select(range(train_size + val_size, total_size))
        
#         logger.info(f"Dataset splits created:")
#         logger.info(f"  - Train: {len(train_dataset)} samples ({len(train_dataset)/total_size*100:.1f}%)")
#         logger.info(f"  - Validation: {len(val_dataset)} samples ({len(val_dataset)/total_size*100:.1f}%)")
#         logger.info(f"  - Test: {len(test_dataset)} samples ({len(test_dataset)/total_size*100:.1f}%)")
        
#         return train_dataset, val_dataset, test_dataset
    
#     def save_dataset_info(self, dataset_info: Dict, output_path: Path):
#         """Save dataset analysis to file"""
#         with open(output_path, 'w') as f:
#             json.dump(dataset_info, f, indent=2, default=str)
#         logger.info(f"Dataset info saved to {output_path}")

# # Utility functions
# def collate_fn(batch):
#     """Custom collate function for ORPO training"""
#     # Handle variable length sequences
#     input_ids = [item["input_ids"] for item in batch]
#     attention_masks = [item["attention_mask"] for item in batch]
#     pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
#     # Pad sequences
#     max_length = max(len(ids) for ids in input_ids)
    
#     padded_input_ids = []
#     padded_attention_masks = []
    
#     for ids, mask in zip(input_ids, attention_masks):
#         pad_length = max_length - len(ids)
#         padded_ids = torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)])
#         padded_mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
        
#         padded_input_ids.append(padded_ids)
#         padded_attention_masks.append(padded_mask)
    
#     return {
#         "input_ids": torch.stack(padded_input_ids),
#         "attention_mask": torch.stack(padded_attention_masks),
#         "pixel_values": pixel_values,
#         "chosen": [item["chosen"] for item in batch],
#         "rejected": [item["rejected"] for item in batch],
#         "metadata": [
#             {
#                 "question": item["question"],
#                 "ds_name": item["ds_name"],
#                 "origin_dataset": item["origin_dataset"]
#             }
#             for item in batch
#         ]
#     }

# # Testing and demo
# if __name__ == "__main__":
#     logger.info("Testing RLAIF-V Data Loader")
    
#     # Test basic loading
#     data_loader = RLAIFVDataLoader(streaming=True, max_samples=10)
    
#     try:
#         # Load raw dataset
#         raw_dataset = data_loader.load_raw_dataset()
#         logger.info("✅ Raw dataset loaded successfully")
        
#         # Analyze dataset
#         analysis = data_loader.analyze_dataset(raw_dataset, num_samples=10)
#         logger.info("✅ Dataset analysis completed")
        
#         # Print sample structure
#         if hasattr(raw_dataset, '__len__') and len(raw_dataset) > 0:
#             sample = raw_dataset[0]
#         else:
#             sample = next(iter(raw_dataset))
        
#         logger.info(f"Sample structure: {list(sample.keys())}")
#         logger.info(f"Question: {sample['question'][:100]}...")
#         logger.info(f"Chosen length: {len(sample['chosen'])}")
#         logger.info(f"Rejected length: {len(sample['rejected'])}")
        
#         logger.info("✅ Data loader test completed successfully")
        
#     except Exception as e:
#         logger.error(f"❌ Data loader test failed: {e}")
#         raise
# src/data_loader.py
# RLAIF-V dataset loading and preprocessing for ORPO training
# COMPLETE VERSION WITH ALL FIXES APPLIED

# import torch
# from datasets import load_dataset, Dataset
# from PIL import Image
# import numpy as np
# from typing import Dict, List, Tuple, Optional, Union
# import logging
# from pathlib import Path
# import json

# from constants import (
#     DATASET_CONFIGS, 
#     DATA_PROCESSING_CONFIGS, 
#     VALIDATION_CONFIGS,
#     TRAINING_HYPERPARAMS
# )

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class RLAIFVDataLoader:
#     """
#     Handles loading and preprocessing of RLAIF-V dataset for ORPO training
#     """
    
#     def __init__(
#         self, 
#         processor=None,
#         config: Optional[Dict] = None,
#         streaming: bool = True,
#         max_samples: Optional[int] = None
#     ):
#         self.processor = processor
#         self.config = config or DATA_PROCESSING_CONFIGS
#         self.streaming = streaming
#         self.max_samples = max_samples
        
#         # Dataset info
#         self.dataset_config = DATASET_CONFIGS["rlaif_v"]
#         self.validation_config = VALIDATION_CONFIGS
        
#         logger.info(f"Initialized RLAIFVDataLoader with streaming={streaming}")
        
#     def load_raw_dataset(self, split: str = "train") -> Dataset:
#         """
#         Load raw RLAIF-V dataset
        
#         Args:
#             split: Dataset split to load
            
#         Returns:
#             Dataset object
#         """
#         dataset_id = self.dataset_config["dataset_id"]
#         logger.info(f"Loading {dataset_id} split='{split}'")
        
#         try:
#             if self.streaming:
#                 # Use streaming for memory efficiency
#                 dataset = load_dataset(dataset_id, split=split, streaming=True)
                
#                 # Convert to regular dataset if max_samples specified
#                 if self.max_samples:
#                     logger.info(f"Loading first {self.max_samples} samples via streaming")
#                     samples = []
#                     for i, example in enumerate(dataset):
#                         if i >= self.max_samples:
#                             break
#                         samples.append(example)
#                     dataset = Dataset.from_list(samples)
#                     logger.info(f"Created dataset with {len(dataset)} samples")
#                 else:
#                     logger.info("Using streaming dataset (full dataset)")
#             else:
#                 # Load full dataset into memory
#                 dataset = load_dataset(dataset_id, split=split)
#                 if self.max_samples:
#                     dataset = dataset.select(range(min(self.max_samples, len(dataset))))
#                 logger.info(f"Loaded {len(dataset)} samples into memory")
                
#             return dataset
            
#         except Exception as e:
#             logger.error(f"Failed to load dataset: {e}")
#             raise
    
#     def analyze_dataset(self, dataset: Dataset, num_samples: int = 100) -> Dict:
#         """
#         Analyze dataset structure and quality
        
#         Args:
#             dataset: Dataset to analyze
#             num_samples: Number of samples to analyze
            
#         Returns:
#             Analysis results dictionary
#         """
#         logger.info("Analyzing dataset structure and quality")
        
#         analysis = {
#             "total_samples": 0,
#             "missing_data": {},
#             "text_lengths": {},
#             "unique_sources": set(),
#             "sample_fields": []
#         }
        
#         # Handle streaming vs regular dataset
#         if hasattr(dataset, '__len__'):
#             analysis["total_samples"] = len(dataset)
#             sample_size = min(num_samples, len(dataset))
#             samples_to_analyze = dataset.select(range(sample_size))
#         else:
#             # Streaming dataset - collect samples for analysis
#             samples_list = []
#             for i, example in enumerate(dataset):
#                 if i >= num_samples:
#                     break
#                 samples_list.append(example)
#             samples_to_analyze = Dataset.from_list(samples_list)
#             analysis["total_samples"] = f"streaming (analyzed {len(samples_to_analyze)})"
        
#         # Get field names from first sample
#         if len(samples_to_analyze) > 0:
#             first_sample = samples_to_analyze[0]
#             analysis["sample_fields"] = list(first_sample.keys())
            
#             # Initialize missing data counters
#             for field in ["image", "question", "chosen", "rejected", "ds_name"]:
#                 analysis["missing_data"][field] = 0
        
#         # Analyze samples
#         text_lengths = {"question": [], "chosen": [], "rejected": []}
        
#         for sample in samples_to_analyze:
#             # Check for missing data
#             for field in analysis["missing_data"].keys():
#                 if not sample.get(field):
#                     analysis["missing_data"][field] += 1
            
#             # Collect text lengths
#             for text_field in text_lengths.keys():
#                 text = sample.get(text_field, "")
#                 if text:
#                     text_lengths[text_field].append(len(text))
            
#             # Collect unique sources
#             ds_name = sample.get("ds_name")
#             if ds_name:
#                 analysis["unique_sources"].add(ds_name)
        
#         # Calculate statistics
#         for field, lengths in text_lengths.items():
#             if lengths:
#                 analysis["text_lengths"][field] = {
#                     "min": np.min(lengths),
#                     "max": np.max(lengths),  
#                     "mean": np.mean(lengths),
#                     "median": np.median(lengths)
#                 }
        
#         # Convert set to list for JSON serialization
#         analysis["unique_sources"] = list(analysis["unique_sources"])
        
#         # Log key findings
#         logger.info(f"Dataset analysis complete:")
#         logger.info(f"  - Total samples: {analysis['total_samples']}")
#         logger.info(f"  - Sample fields: {analysis['sample_fields']}")
#         logger.info(f"  - Unique sources: {len(analysis['unique_sources'])}")
        
#         for field, count in analysis["missing_data"].items():
#             if count > 0:
#                 logger.warning(f"  - Missing {field}: {count} samples")
        
#         return analysis
    
#     def validate_sample(self, sample: Dict) -> bool:
#         """
#         Validate a single sample for training
        
#         Args:
#             sample: Sample to validate
            
#         Returns:
#             True if sample is valid, False otherwise
#         """
#         required_fields = ["image", "question", "chosen", "rejected"]
        
#         # Check required fields exist
#         for field in required_fields:
#             if not sample.get(field):
#                 return False
        
#         # Validate image
#         image = sample.get("image")
#         if not isinstance(image, Image.Image):
#             return False
        
#         # Validate text lengths
#         question = sample.get("question", "")
#         chosen = sample.get("chosen", "")
#         rejected = sample.get("rejected", "")
        
#         min_len = self.validation_config["min_response_length"]
#         max_len = self.validation_config["max_response_length"]
        
#         if (len(chosen) < min_len or len(chosen) > max_len or
#             len(rejected) < min_len or len(rejected) > max_len or
#             len(question) < 5):  # Minimum question length
#             return False
        
#         return True
    
#     def format_sample_for_training(self, sample: Dict) -> Dict:
#         """
#         Format a single sample for ORPO training - DEFINITIVE WORKING VERSION
#         Uses manual <image> token approach that passed all tests
        
#         Args:
#             sample: Raw sample from dataset
            
#         Returns:
#             Formatted sample ready for training
#         """
#         if not self.processor:
#             raise ValueError("Processor must be provided for formatting")
        
#         try:
#             question = sample["question"]
            
#             # WORKING SOLUTION: Manual <image> token (tested and confirmed working)
#             # This approach ensures the text contains the required <image> token
#             text_with_image = f"<image>\n{question}"
            
#             inputs = self.processor(
#                 text=text_with_image,
#                 images=sample["image"],
#                 return_tensors="pt",
#                 padding=False,
#                 truncation=True,
#                 max_length=2048  # Tested working value that accommodates image tokens
#             )
            
#             # Format for ORPO training
#             formatted_sample = {
#                 # Model inputs
#                 "input_ids": inputs["input_ids"].squeeze(0),
#                 "attention_mask": inputs["attention_mask"].squeeze(0),
#                 "pixel_values": inputs["pixel_values"].squeeze(0),
                
#                 # ORPO targets
#                 "chosen": sample["chosen"],
#                 "rejected": sample["rejected"],
                
#                 # Metadata for analysis
#                 "question": question,
#                 "ds_name": sample.get("ds_name", "unknown"),
#                 "origin_dataset": sample.get("origin_dataset", "unknown")
#             }
            
#             logger.debug(f"Formatted sample with {inputs['input_ids'].shape[1]} tokens")
#             return formatted_sample
            
#         except Exception as e:
#             # Log the specific error for debugging
#             logger.error(f"Error formatting sample: {e}")
#             logger.error(f"Sample keys: {list(sample.keys())}")
#             logger.error(f"Question: {sample.get('question', 'N/A')[:100]}...")
#             raise
    
#     def create_training_dataset(
#         self, 
#         raw_dataset: Dataset,
#         split_ratios: Optional[Tuple[float, float, float]] = None
#     ) -> Tuple[Dataset, Dataset, Dataset]:
#         """
#         Create formatted training, validation, and test datasets
        
#         Args:
#             raw_dataset: Raw RLAIF-V dataset
#             split_ratios: (train, val, test) ratios
            
#         Returns:
#             Tuple of (train_dataset, val_dataset, test_dataset)
#         """
#         if not self.processor:
#             raise ValueError("Processor must be provided for dataset creation")
        
#         logger.info("Creating formatted training dataset")
        
#         # Default split ratios
#         if split_ratios is None:
#             split_ratios = (
#                 self.config["train_ratio"],
#                 self.config["val_ratio"], 
#                 self.config["test_ratio"]
#             )
        
#         formatted_samples = []
#         valid_count = 0
#         invalid_count = 0
        
#         # Process samples
#         total_samples = len(raw_dataset) if hasattr(raw_dataset, '__len__') else "unknown"
#         logger.info(f"Processing {total_samples} samples")
        
#         for i, sample in enumerate(raw_dataset):
#             try:
#                 # Validate sample
#                 if not self.validate_sample(sample):
#                     invalid_count += 1
#                     if self.validation_config["log_invalid_samples"] and invalid_count <= 5:
#                         logger.warning(f"Invalid sample {i}: missing or invalid data")
#                     continue
                
#                 # Format sample
#                 formatted = self.format_sample_for_training(sample)
#                 formatted_samples.append(formatted)
#                 valid_count += 1
                
#                 # Progress logging
#                 if (i + 1) % 100 == 0:
#                     logger.info(f"Processed {i + 1} samples, {valid_count} valid, {invalid_count} invalid")
                
#                 # Respect max_samples limit
#                 if self.max_samples and valid_count >= self.max_samples:
#                     logger.info(f"Reached max_samples limit: {self.max_samples}")
#                     break
                    
#             except Exception as e:
#                 invalid_count += 1
#                 if invalid_count <= 5:  # Log first few errors
#                     logger.error(f"Error processing sample {i}: {e}")
        
#         logger.info(f"Dataset processing complete: {valid_count} valid, {invalid_count} invalid samples")
        
#         if not formatted_samples:
#             raise ValueError("No valid samples found in dataset")
        
#         # Create HuggingFace dataset
#         full_dataset = Dataset.from_list(formatted_samples)
        
#         # Split dataset
#         train_dataset, val_dataset, test_dataset = self.split_dataset(
#             full_dataset, split_ratios[0], split_ratios[1], split_ratios[2]
#         )
        
#         return train_dataset, val_dataset, test_dataset
    
#     def split_dataset(self, dataset, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
#         """Split dataset for training/validation/testing with handling for small datasets"""
        
#         total_size = len(dataset)
        
#         # Special handling for very small datasets
#         if total_size <= 3:
#             logger.info(f"Small dataset ({total_size} samples) - using all samples for each split")
#             # For very small datasets, use all samples for each split to avoid empty splits
#             return dataset, dataset, dataset
        
#         # Normal splitting for larger datasets
#         train_size = max(1, int(total_size * train_ratio))  # Ensure at least 1 sample
#         val_size = max(1, int(total_size * val_ratio))
        
#         # Adjust if sizes would exceed total
#         if train_size + val_size >= total_size:
#             val_size = max(1, total_size - train_size - 1)  # Leave at least 1 for test
#             if train_size + val_size >= total_size:
#                 train_size = total_size - 2  # Leave 1 each for val and test
#                 val_size = 1
        
#         # Create splits
#         train_dataset = dataset.select(range(train_size))
#         val_dataset = dataset.select(range(train_size, train_size + val_size))
#         test_dataset = dataset.select(range(train_size + val_size, total_size))
        
#         logger.info(f"Dataset splits:")
#         logger.info(f"  Train: {len(train_dataset)} samples ({len(train_dataset)/total_size*100:.1f}%)")
#         logger.info(f"  Validation: {len(val_dataset)} samples ({len(val_dataset)/total_size*100:.1f}%)")  
#         logger.info(f"  Test: {len(test_dataset)} samples ({len(test_dataset)/total_size*100:.1f}%)")
        
#         return train_dataset, val_dataset, test_dataset
    
#     def save_dataset_info(self, dataset_info: Dict, output_path: Path):
#         """Save dataset analysis to file"""
#         with open(output_path, 'w') as f:
#             json.dump(dataset_info, f, indent=2, default=str)
#         logger.info(f"Dataset info saved to {output_path}")

# # Utility functions
# def collate_fn(batch):
#     """Custom collate function for ORPO training"""
#     # Handle variable length sequences
#     input_ids = [item["input_ids"] for item in batch]
#     attention_masks = [item["attention_mask"] for item in batch]
#     pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
#     # Pad sequences
#     max_length = max(len(ids) for ids in input_ids)
    
#     padded_input_ids = []
#     padded_attention_masks = []
    
#     for ids, mask in zip(input_ids, attention_masks):
#         pad_length = max_length - len(ids)
#         padded_ids = torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)])
#         padded_mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
        
#         padded_input_ids.append(padded_ids)
#         padded_attention_masks.append(padded_mask)
    
#     return {
#         "input_ids": torch.stack(padded_input_ids),
#         "attention_mask": torch.stack(padded_attention_masks),
#         "pixel_values": pixel_values,
#         "chosen": [item["chosen"] for item in batch],
#         "rejected": [item["rejected"] for item in batch],
#         "metadata": [
#             {
#                 "question": item["question"],
#                 "ds_name": item["ds_name"],
#                 "origin_dataset": item["origin_dataset"]
#             }
#             for item in batch
#         ]
#     }

# # Testing and demo
# if __name__ == "__main__":
#     logger.info("Testing RLAIF-V Data Loader")
    
#     # Test basic loading
#     data_loader = RLAIFVDataLoader(streaming=True, max_samples=10)
    
#     try:
#         # Load raw dataset
#         raw_dataset = data_loader.load_raw_dataset()
#         logger.info("✅ Raw dataset loaded successfully")
        
#         # Analyze dataset
#         analysis = data_loader.analyze_dataset(raw_dataset, num_samples=10)
#         logger.info("✅ Dataset analysis completed")
        
#         # Print sample structure
#         if hasattr(raw_dataset, '__len__') and len(raw_dataset) > 0:
#             sample = raw_dataset[0]
#         else:
#             sample = next(iter(raw_dataset))
        
#         logger.info(f"Sample structure: {list(sample.keys())}")
#         logger.info(f"Question: {sample['question'][:100]}...")
#         logger.info(f"Chosen length: {len(sample['chosen'])}")
#         logger.info(f"Rejected length: {len(sample['rejected'])}")
        
#         logger.info("✅ Data loader test completed successfully")
        
#     except Exception as e:
#         logger.error(f"❌ Data loader test failed: {e}")
#         raise

# src/data_loader.py
# RLAIF-V dataset loading and preprocessing for ORPO training
# COMPLETE VERSION WITH ALL FIXES APPLIED

import torch
from datasets import load_dataset, Dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json

from constants import (
    DATASET_CONFIGS, 
    DATA_PROCESSING_CONFIGS, 
    VALIDATION_CONFIGS,
    TRAINING_HYPERPARAMS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLAIFVDataLoader:
    """
    Handles loading and preprocessing of RLAIF-V dataset for ORPO training
    """
    
    def __init__(
        self, 
        processor=None,
        config: Optional[Dict] = None,
        streaming: bool = True,
        max_samples: Optional[int] = None
    ):
        self.processor = processor
        self.config = config or DATA_PROCESSING_CONFIGS
        self.streaming = streaming
        self.max_samples = max_samples
        
        # Dataset info
        self.dataset_config = DATASET_CONFIGS["rlaif_v"]
        self.validation_config = VALIDATION_CONFIGS
        
        logger.info(f"Initialized RLAIFVDataLoader with streaming={streaming}")
        
    def load_raw_dataset(self, split: str = "train") -> Dataset:
        """
        Load raw RLAIF-V dataset
        
        Args:
            split: Dataset split to load
            
        Returns:
            Dataset object
        """
        dataset_id = self.dataset_config["dataset_id"]
        logger.info(f"Loading {dataset_id} split='{split}'")
        
        try:
            if self.streaming:
                # Use streaming for memory efficiency
                dataset = load_dataset(dataset_id, split=split, streaming=True)
                
                # Convert to regular dataset if max_samples specified
                if self.max_samples:
                    logger.info(f"Loading first {self.max_samples} samples via streaming")
                    samples = []
                    for i, example in enumerate(dataset):
                        if i >= self.max_samples:
                            break
                        samples.append(example)
                    dataset = Dataset.from_list(samples)
                    logger.info(f"Created dataset with {len(dataset)} samples")
                else:
                    logger.info("Using streaming dataset (full dataset)")
            else:
                # Load full dataset into memory
                dataset = load_dataset(dataset_id, split=split)
                if self.max_samples:
                    dataset = dataset.select(range(min(self.max_samples, len(dataset))))
                logger.info(f"Loaded {len(dataset)} samples into memory")
                
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def analyze_dataset(self, dataset: Dataset, num_samples: int = 100) -> Dict:
        """
        Analyze dataset structure and quality
        
        Args:
            dataset: Dataset to analyze
            num_samples: Number of samples to analyze
            
        Returns:
            Analysis results dictionary
        """
        logger.info("Analyzing dataset structure and quality")
        
        analysis = {
            "total_samples": 0,
            "missing_data": {},
            "text_lengths": {},
            "unique_sources": set(),
            "sample_fields": []
        }
        
        # Handle streaming vs regular dataset
        if hasattr(dataset, '__len__'):
            analysis["total_samples"] = len(dataset)
            sample_size = min(num_samples, len(dataset))
            samples_to_analyze = dataset.select(range(sample_size))
        else:
            # Streaming dataset - collect samples for analysis
            samples_list = []
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                samples_list.append(example)
            samples_to_analyze = Dataset.from_list(samples_list)
            analysis["total_samples"] = f"streaming (analyzed {len(samples_to_analyze)})"
        
        # Get field names from first sample
        if len(samples_to_analyze) > 0:
            first_sample = samples_to_analyze[0]
            analysis["sample_fields"] = list(first_sample.keys())
            
            # Initialize missing data counters
            for field in ["image", "question", "chosen", "rejected", "ds_name"]:
                analysis["missing_data"][field] = 0
        
        # Analyze samples
        text_lengths = {"question": [], "chosen": [], "rejected": []}
        
        for sample in samples_to_analyze:
            # Check for missing data
            for field in analysis["missing_data"].keys():
                if not sample.get(field):
                    analysis["missing_data"][field] += 1
            
            # Collect text lengths
            for text_field in text_lengths.keys():
                text = sample.get(text_field, "")
                if text:
                    text_lengths[text_field].append(len(text))
            
            # Collect unique sources
            ds_name = sample.get("ds_name")
            if ds_name:
                analysis["unique_sources"].add(ds_name)
        
        # Calculate statistics
        for field, lengths in text_lengths.items():
            if lengths:
                analysis["text_lengths"][field] = {
                    "min": np.min(lengths),
                    "max": np.max(lengths),  
                    "mean": np.mean(lengths),
                    "median": np.median(lengths)
                }
        
        # Convert set to list for JSON serialization
        analysis["unique_sources"] = list(analysis["unique_sources"])
        
        # Log key findings
        logger.info(f"Dataset analysis complete:")
        logger.info(f"  - Total samples: {analysis['total_samples']}")
        logger.info(f"  - Sample fields: {analysis['sample_fields']}")
        logger.info(f"  - Unique sources: {len(analysis['unique_sources'])}")
        
        for field, count in analysis["missing_data"].items():
            if count > 0:
                logger.warning(f"  - Missing {field}: {count} samples")
        
        return analysis
    
    def validate_sample(self, sample: Dict) -> bool:
        """
        Validate a single sample for training
        
        Args:
            sample: Sample to validate
            
        Returns:
            True if sample is valid, False otherwise
        """
        required_fields = ["image", "question", "chosen", "rejected"]
        
        # Check required fields exist
        for field in required_fields:
            if not sample.get(field):
                return False
        
        # Validate image
        image = sample.get("image")
        if not isinstance(image, Image.Image):
            return False
        
        # Validate text lengths
        question = sample.get("question", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        
        min_len = self.validation_config["min_response_length"]
        max_len = self.validation_config["max_response_length"]
        
        if (len(chosen) < min_len or len(chosen) > max_len or
            len(rejected) < min_len or len(rejected) > max_len or
            len(question) < 5):  # Minimum question length
            return False
        
        return True
    
    # def format_sample_for_training(self, sample: Dict) -> Dict:
    #     """
    #     Format a single sample for ORPO training - IMMEDIATE FIX VERSION
    #     Uses higher max_length to avoid truncating image tokens
        
    #     Args:
    #         sample: Raw sample from dataset
            
    #     Returns:
    #         Formatted sample ready for training
    #     """
    #     if not self.processor:
    #         raise ValueError("Processor must be provided for formatting")
        
    #     try:
    #         question = sample["question"]
            
    #         # WORKING SOLUTION: Manual <image> token with higher max_length
    #         text_with_image = f"<image>\n{question}"
            
    #         inputs = self.processor(
    #             text=text_with_image,
    #             images=sample["image"],
    #             return_tensors="pt",
    #             padding=False,
    #             truncation=True,
    #             max_length=4096  # INCREASED: Much higher to accommodate all image tokens
    #         )
            
    #         # Check if sequence is still too long after processing
    #         seq_length = inputs["input_ids"].shape[1]
    #         if seq_length > 4000:  # Safety check
    #             logger.warning(f"Very long sequence: {seq_length} tokens")
            
    #         # Format for ORPO training
    #         formatted_sample = {
    #             # Model inputs
    #             "input_ids": inputs["input_ids"].squeeze(0),
    #             "attention_mask": inputs["attention_mask"].squeeze(0),
    #             "pixel_values": inputs["pixel_values"].squeeze(0),
                
    #             # ORPO targets
    #             "chosen": sample["chosen"],
    #             "rejected": sample["rejected"],
                
    #             # Metadata for analysis
    #             "question": question,
    #             "ds_name": sample.get("ds_name", "unknown"),
    #             "origin_dataset": sample.get("origin_dataset", "unknown")
    #         }
            
    #         logger.debug(f"Formatted sample with {seq_length} tokens")
    #         return formatted_sample
            
    #     except Exception as e:
    #         # Log the specific error for debugging
    #         logger.error(f"Error formatting sample: {e}")
    #         logger.error(f"Sample keys: {list(sample.keys())}")
    #         logger.error(f"Question: {sample.get('question', 'N/A')[:100]}...")
    #         raise
    
    def format_sample_for_training(self, sample: Dict) -> Dict:
        """
        Format a single sample for ORPO training - FIXED with image_sizes
        """
        if not self.processor:
            raise ValueError("Processor must be provided for formatting")
        
        try:
            question = sample["question"]
            
            # WORKING SOLUTION: Manual <image> token
            text_with_image = f"<image>\n{question}"
            
            inputs = self.processor(
                text=text_with_image,
                images=sample["image"],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=2560
            )
            
            # Check if sequence is too long
            seq_length = inputs["input_ids"].shape[1]
            if seq_length > 4000:
                logger.warning(f"Very long sequence: {seq_length} tokens")
            
            # CRITICAL FIX: Include image_sizes that LLaVA needs
            formatted_sample = {
                # Model inputs
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "image_sizes": inputs["image_sizes"].squeeze(0),  # ← ADDED: Required by LLaVA
                
                # ORPO targets
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
                
                # Metadata for analysis
                "question": question,
                "ds_name": sample.get("ds_name", "unknown"),
                "origin_dataset": sample.get("origin_dataset", "unknown")
            }
            
            logger.debug(f"Formatted sample with {seq_length} tokens")
            return formatted_sample
            
        except Exception as e:
            logger.error(f"Error formatting sample: {e}")
            logger.error(f"Sample keys: {list(sample.keys())}")
            logger.error(f"Question: {sample.get('question', 'N/A')[:100]}...")
            raise
    
    def create_training_dataset(
        self, 
        raw_dataset: Dataset,
        split_ratios: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create formatted training, validation, and test datasets
        
        Args:
            raw_dataset: Raw RLAIF-V dataset
            split_ratios: (train, val, test) ratios
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if not self.processor:
            raise ValueError("Processor must be provided for dataset creation")
        
        logger.info("Creating formatted training dataset")
        
        # Default split ratios
        if split_ratios is None:
            split_ratios = (
                self.config["train_ratio"],
                self.config["val_ratio"], 
                self.config["test_ratio"]
            )
        
        formatted_samples = []
        valid_count = 0
        invalid_count = 0
        
        # Process samples
        total_samples = len(raw_dataset) if hasattr(raw_dataset, '__len__') else "unknown"
        logger.info(f"Processing {total_samples} samples")
        
        for i, sample in enumerate(raw_dataset):
            try:
                # Validate sample
                if not self.validate_sample(sample):
                    invalid_count += 1
                    if self.validation_config["log_invalid_samples"] and invalid_count <= 5:
                        logger.warning(f"Invalid sample {i}: missing or invalid data")
                    continue
                
                # Format sample
                formatted = self.format_sample_for_training(sample)
                formatted_samples.append(formatted)
                valid_count += 1
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} samples, {valid_count} valid, {invalid_count} invalid")
                
                # Respect max_samples limit
                if self.max_samples and valid_count >= self.max_samples:
                    logger.info(f"Reached max_samples limit: {self.max_samples}")
                    break
                    
            except Exception as e:
                invalid_count += 1
                if invalid_count <= 5:  # Log first few errors
                    logger.error(f"Error processing sample {i}: {e}")
        
        logger.info(f"Dataset processing complete: {valid_count} valid, {invalid_count} invalid samples")
        
        if not formatted_samples:
            raise ValueError("No valid samples found in dataset")
        
        # Create HuggingFace dataset
        full_dataset = Dataset.from_list(formatted_samples)
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = self.split_dataset(
            full_dataset, split_ratios[0], split_ratios[1], split_ratios[2]
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def split_dataset(self, dataset, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
        """Split dataset for training/validation/testing with handling for small datasets"""
        
        total_size = len(dataset)
        
        # Special handling for very small datasets
        if total_size <= 3:
            logger.info(f"Small dataset ({total_size} samples) - using all samples for each split")
            # For very small datasets, use all samples for each split to avoid empty splits
            return dataset, dataset, dataset
        
        # Normal splitting for larger datasets
        train_size = max(1, int(total_size * train_ratio))  # Ensure at least 1 sample
        val_size = max(1, int(total_size * val_ratio))
        
        # Adjust if sizes would exceed total
        if train_size + val_size >= total_size:
            val_size = max(1, total_size - train_size - 1)  # Leave at least 1 for test
            if train_size + val_size >= total_size:
                train_size = total_size - 2  # Leave 1 each for val and test
                val_size = 1
        
        # Create splits
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))
        
        logger.info(f"Dataset splits:")
        logger.info(f"  Train: {len(train_dataset)} samples ({len(train_dataset)/total_size*100:.1f}%)")
        logger.info(f"  Validation: {len(val_dataset)} samples ({len(val_dataset)/total_size*100:.1f}%)")  
        logger.info(f"  Test: {len(test_dataset)} samples ({len(test_dataset)/total_size*100:.1f}%)")
        
        return train_dataset, val_dataset, test_dataset
    
    def save_dataset_info(self, dataset_info: Dict, output_path: Path):
        """Save dataset analysis to file"""
        with open(output_path, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
        logger.info(f"Dataset info saved to {output_path}")

# Utility functions
def collate_fn(batch):
    """Custom collate function for ORPO training"""
    # Handle variable length sequences
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    # Pad sequences
    max_length = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        pad_length = max_length - len(ids)
        padded_ids = torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)])
        padded_mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "pixel_values": pixel_values,
        "chosen": [item["chosen"] for item in batch],
        "rejected": [item["rejected"] for item in batch],
        "metadata": [
            {
                "question": item["question"],
                "ds_name": item["ds_name"],
                "origin_dataset": item["origin_dataset"]
            }
            for item in batch
        ]
    }

# Testing and demo
if __name__ == "__main__":
    logger.info("Testing RLAIF-V Data Loader")
    
    # Test basic loading
    data_loader = RLAIFVDataLoader(streaming=True, max_samples=10)
    
    try:
        # Load raw dataset
        raw_dataset = data_loader.load_raw_dataset()
        logger.info("✅ Raw dataset loaded successfully")
        
        # Analyze dataset
        analysis = data_loader.analyze_dataset(raw_dataset, num_samples=10)
        logger.info("✅ Dataset analysis completed")
        
        # Print sample structure
        if hasattr(raw_dataset, '__len__') and len(raw_dataset) > 0:
            sample = raw_dataset[0]
        else:
            sample = next(iter(raw_dataset))
        
        logger.info(f"Sample structure: {list(sample.keys())}")
        logger.info(f"Question: {sample['question'][:100]}...")
        logger.info(f"Chosen length: {len(sample['chosen'])}")
        logger.info(f"Rejected length: {len(sample['rejected'])}")
        
        logger.info("✅ Data loader test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Data loader test failed: {e}")
        raise