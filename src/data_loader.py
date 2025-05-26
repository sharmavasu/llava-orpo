# src/data_loader.py

import torch
from torch.utils.data import Dataset # Sticking with Dataset for now, can switch to Iterable if needed
from datasets import load_dataset, Dataset as HFDataset # For converting list to HF Dataset object
from PIL import Image

from llava.conversation import conv_templates
# Assuming constants.py will be in the same src directory or accessible in python path
from .constants import DEFAULT_IMAGE_TOKEN 

# Note: IMAGE_TOKEN_INDEX and process_images will be handled by the LLaVA utils
# when called from __getitem__ or the main training script.

class RLAIFVPreferenceDataset(Dataset):
    def __init__(self, 
                 dataset_id="openbmb/RLAIF-V-Dataset", 
                 split="train", 
                 tokenizer=None, # Will be passed from train.py
                 image_processor=None, # Will be passed from train.py
                 model_config=None, # Will be passed from train.py
                 image_token=DEFAULT_IMAGE_TOKEN,
                 # ds_name_filter=None, # If filtering by 'ds_name' directly
                 origin_dataset_filter=None, # New: For filtering by 'origin_dataset'
                 max_samples=None, # For loading a subset for debugging/testing
                 slice_for_loading=None, # e.g. "train[:1%]" or "train[:1000]" for initial load
                 streaming=False # Whether to use streaming for initial load
                 ):
        super().__init__()
        
        if tokenizer is None or image_processor is None or model_config is None:
            raise ValueError("Tokenizer, image_processor, and model_config must be provided.")

        self.dataset_id = dataset_id
        self.split_to_load = slice_for_loading if slice_for_loading else split # Use slice if provided
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config 
        self.image_token = image_token
        self.origin_dataset_filter = origin_dataset_filter

        print(f"Initializing RLAIFVPreferenceDataset with split: {self.split_to_load}, streaming: {streaming}")

        if streaming:
            print("Loading dataset in streaming mode...")
            streamed_dataset = load_dataset(self.dataset_id, split=self.split_to_load, streaming=True)
            # If streaming, we might need to process a certain number of samples into a list
            # or this class would need to be an IterableDataset.
            # For now, let's assume if streaming is True for init, we take 'max_samples' if provided.
            temp_data_list = []
            if max_samples:
                print(f"Streaming and collecting up to {max_samples} samples...")
                count = 0
                for example in streamed_dataset:
                    temp_data_list.append(example)
                    count += 1
                    if count >= max_samples:
                        break
                self.processed_data = HFDataset.from_list(temp_data_list)
            else:
                # Streaming all data into memory like this for a Dataset is not truly streaming.
                # If the goal is to process the whole dataset via streaming, RLAIFVPreferenceDataset
                # should inherit from IterableDataset. For now, this 'streaming' flag is mostly for
                # initial loading of a subset if direct slicing fails.
                print("Warning: Streaming all data into memory for map-style Dataset. Consider IterableDataset for true streaming.")
                self.processed_data = HFDataset.from_list(list(streamed_dataset)) # Potentially very large!
        else:
            print(f"Loading dataset with direct slice: {self.split_to_load}")
            full_dataset = load_dataset(self.dataset_id, split=self.split_to_load)
            self.processed_data = full_dataset # Start with the loaded slice

        # Filter by origin_dataset if a filter is provided
        if self.origin_dataset_filter:
            if isinstance(self.origin_dataset_filter, str):
                self.origin_dataset_filter = [self.origin_dataset_filter]
            
            print(f"Filtering dataset for origin_dataset(s): {self.origin_dataset_filter}...")
            self.processed_data = self.processed_data.filter(
                lambda example: example['origin_dataset'] in self.origin_dataset_filter
            )
            print(f"Filtered dataset size (by origin_dataset): {len(self.processed_data)}")
        
        # Apply max_samples if not already applied by streaming logic and if it's smaller
        if not streaming and max_samples is not None and max_samples < len(self.processed_data):
            print(f"Taking a subset of {max_samples} samples from loaded/filtered data.")
            self.processed_data = self.processed_data.select(range(max_samples))
            
        if len(self.processed_data) == 0:
            print("Warning: Dataset is empty after loading and filtering. Check filters and slice.")
        else:
            print(f"Final dataset size for this instance: {len(self.processed_data)}")

        # Pre-fetch LLaVA utilities (or ensure they are available globally, but this is safer)
        from llava.mm_utils import tokenizer_image_token, process_images, IMAGE_TOKEN_INDEX as LLaVA_IMAGE_TOKEN_INDEX
        self.tokenizer_image_token_func = tokenizer_image_token
        self.process_images_func = process_images
        self.LLAVA_IMAGE_TOKEN_INDEX = LLaVA_IMAGE_TOKEN_INDEX


    def __len__(self):
        return len(self.processed_data)

    def _get_formatted_texts_and_prompt_len(self, question, chosen_response, rejected_response):
        # Use the conversation template for LLaVA v1.6 Mistral
        # Needs self.image_token
        
        # Prompt part for calculating length (for masking labels)
        conv_prompt_structure = conv_templates["mistral_instruct"].copy()
        conv_prompt_structure.append_message(conv_prompt_structure.roles[0], f"{self.image_token}\n{question}")
        conv_prompt_structure.append_message(conv_prompt_structure.roles[1], None) # Assistant's turn starts here
        prompt_text_for_masking = conv_prompt_structure.get_prompt()
        
        # Tokenize prompt part to find its length
        # Note: tokenizer_image_token will replace self.image_token with the actual image placeholder ID
        # Using the passed LLaVA tokenizer
        prompt_input_ids = self.tokenizer_image_token_func(
            prompt_text_for_masking, self.tokenizer, self.LLAVA_IMAGE_TOKEN_INDEX, return_tensors='pt'
        )
        # Remove batch dimension added by return_tensors='pt' if present and get length
        prompt_length = prompt_input_ids.squeeze().shape[0]

        # Chosen full text
        conv_chosen = conv_templates["mistral_instruct"].copy()
        conv_chosen.append_message(conv_chosen.roles[0], f"{self.image_token}\n{question}")
        conv_chosen.append_message(conv_chosen.roles[1], chosen_response)
        chosen_full_text = conv_chosen.get_prompt()
        
        # Rejected full text
        conv_rejected = conv_templates["mistral_instruct"].copy()
        conv_rejected.append_message(conv_rejected.roles[0], f"{self.image_token}\n{question}")
        conv_rejected.append_message(conv_rejected.roles[1], rejected_response)
        rejected_full_text = conv_rejected.get_prompt()
        
        return chosen_full_text, rejected_full_text, prompt_length

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        pil_image = item['image']
        if not isinstance(pil_image, Image.Image):
            # This case should ideally not happen if datasets lib decodes correctly
            print(f"Warning: Item at index {idx} does not contain a PIL image. Got {type(pil_image)}")
            # Attempt to load if it's bytes, or handle error
            if isinstance(pil_image, dict) and 'bytes' in pil_image and pil_image['bytes']:
                try:
                    pil_image = Image.open(io.BytesIO(pil_image['bytes']))
                except Exception as e:
                    raise ValueError(f"Could not load image from bytes at index {idx}: {e}")
            else:
                raise ValueError(f"Unsupported image format or missing image at index {idx}")


        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        question = item['question']
        chosen_response = item['chosen']
        rejected_response = item['rejected']
        image_original_size = pil_image.size # (width, height)

        # Process image using LLaVA's utility
        image_tensor = self.process_images_func([pil_image], self.image_processor, self.model_config)[0]

        # Get formatted texts and the length of the prompt part for masking labels
        chosen_full_text, rejected_full_text, prompt_length = \
            self._get_formatted_texts_and_prompt_len(question, chosen_response, rejected_response)

        # Tokenize chosen
        chosen_input_ids = self.tokenizer_image_token_func(
            chosen_full_text, self.tokenizer, self.LLAVA_IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).squeeze(0) # Remove batch dimension
        
        # Tokenize rejected
        rejected_input_ids = self.tokenizer_image_token_func(
            rejected_full_text, self.tokenizer, self.LLAVA_IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).squeeze(0) # Remove batch dimension

        # Create labels: -100 for prompt tokens, actual token_id for response tokens
        chosen_labels = chosen_input_ids.clone()
        # Ensure prompt_length doesn't exceed sequence length (it shouldn't for valid data)
        chosen_labels[:min(prompt_length, chosen_labels.shape[0])] = -100 
        
        rejected_labels = rejected_input_ids.clone()
        rejected_labels[:min(prompt_length, rejected_labels.shape[0])] = -100
        
        chosen_attention_mask = torch.ones_like(chosen_input_ids)
        rejected_attention_mask = torch.ones_like(rejected_input_ids)

        return {
            "image_tensor": image_tensor,
            "image_sizes": torch.tensor([image_original_size], dtype=torch.long), # Needs to be tensor for collator
            
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
        }

# You would not typically run __main__ from src/data_loader.py directly in a project.
# This is for initial testing. You'd import and use RLAIFVPreferenceDataset in train.py.
# To test this file standalone, you'd need to:
# 1. Ensure LLaVA is installed and its paths are correct for imports.
# 2. Create mock or load real tokenizer, image_processor, and model_config.
# This is non-trivial for LLaVA due to its integrated components.
# The best place to test the Dataset instance is after loading these in your train.py.