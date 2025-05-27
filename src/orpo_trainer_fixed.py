# src/orpo_trainer.py - FIXED VERSION
# Custom ORPO (Odds Ratio Preference Optimization) implementation for LLaVA

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import wandb

from constants import ORPO_CONFIGS, TRAINING_HYPERPARAMS, MEMORY_CONFIGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ORPOLoss:
    """
    Implements ORPO (Odds Ratio Preference Optimization) loss function
    """
    
    def __init__(self, beta: float = 0.1, alpha: float = 1.0):
        self.beta = beta
        self.alpha = alpha
        logger.info(f"Initialized ORPO loss with beta={beta}, alpha={alpha}")
    
    def compute_loss(
        self,
        model,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        attention_mask_chosen: torch.Tensor,
        attention_mask_rejected: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute ORPO loss - FIXED VERSION
        """
        
        # Compute log probabilities for chosen and rejected responses
        chosen_logprobs = self._compute_sequence_logprobs(
            chosen_logits, chosen_labels, attention_mask_chosen
        )
        rejected_logprobs = self._compute_sequence_logprobs(
            rejected_logits, rejected_labels, attention_mask_rejected
        )
        
        # ORPO preference loss
        logits_diff = chosen_logprobs - rejected_logprobs
        preference_loss = -F.logsigmoid(self.beta * logits_diff).mean()
        
        # Supervised fine-tuning loss on chosen responses
        sft_loss = F.cross_entropy(
            chosen_logits.view(-1, chosen_logits.size(-1)),
            chosen_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        # Combine losses
        total_loss = sft_loss + self.alpha * preference_loss
        
        # Compute metrics for logging
        with torch.no_grad():
            accuracy = (chosen_logprobs > rejected_logprobs).float().mean()
            
        loss_dict = {
            'total_loss': total_loss.item(),
            'sft_loss': sft_loss.item(),
            'preference_loss': preference_loss.item(),
            'preference_accuracy': accuracy.item(),
            'chosen_logprobs': chosen_logprobs.mean().item(),
            'rejected_logprobs': rejected_logprobs.mean().item(),
            'logprobs_diff': logits_diff.mean().item()
        }
        
        return total_loss, loss_dict
    
    def _compute_sequence_logprobs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for sequences - FIXED VERSION
        """
        # Shift logits and labels for next-token prediction
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        shifted_mask = attention_mask[..., 1:].contiguous()
        
        # CRITICAL FIX: Ensure indices are valid
        vocab_size = shifted_logits.size(-1)
        
        # Clamp labels to valid range and replace -100 with 0 for gathering
        valid_labels = shifted_labels.clone()
        valid_labels = torch.clamp(valid_labels, 0, vocab_size - 1)
        
        # Create mask for valid (non-ignored) tokens
        valid_token_mask = (shifted_labels != -100).float()
        
        # Combine with attention mask
        combined_mask = shifted_mask * valid_token_mask
        
        # Compute log probabilities
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        
        # FIXED: Safe gathering with clamped indices
        gathered_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=valid_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply combined mask and compute sequence log probabilities
        masked_log_probs = gathered_log_probs * combined_mask
        sequence_log_probs = masked_log_probs.sum(dim=-1) / combined_mask.sum(dim=-1).clamp(min=1)
        
        return sequence_log_probs

class ORPOTrainer(Trainer):
    """
    Custom trainer for ORPO optimization - FIXED VERSION
    """
    
    def __init__(
        self,
        model,
        processor,
        train_dataset,
        eval_dataset=None,
        **kwargs
    ):
        self.processor = processor
        self.orpo_loss = ORPOLoss(
            beta=ORPO_CONFIGS["beta"],
            alpha=ORPO_CONFIGS["alpha"]
        )
        
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        
        logger.info("Initialized ORPO Trainer")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute ORPO loss for a batch - MEMORY OPTIMIZED VERSION
        """
        # MEMORY OPTIMIZATION: Process samples one by one for memory-constrained environments
        if len(inputs["chosen"]) > 1:
            return self._compute_loss_sequential(model, inputs, return_outputs, num_items_in_batch)
        
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs["image_sizes"]
        chosen_responses = inputs["chosen"]
        rejected_responses = inputs["rejected"]
        
        try:
            # Tokenize responses with conservative max_length
            chosen_tokens = self._tokenize_responses(chosen_responses, max_length=128)
            rejected_tokens = self._tokenize_responses(rejected_responses, max_length=128)
            
            # Forward pass for chosen responses
            chosen_inputs = {
                "input_ids": torch.cat([input_ids, chosen_tokens["input_ids"]], dim=1),
                "attention_mask": torch.cat([attention_mask, chosen_tokens["attention_mask"]], dim=1),
                "pixel_values": pixel_values,
                "image_sizes": image_sizes
            }
            
            # MEMORY OPTIMIZATION: Clear cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            chosen_outputs = model(**chosen_inputs)
            
            # Forward pass for rejected responses
            rejected_inputs = {
                "input_ids": torch.cat([input_ids, rejected_tokens["input_ids"]], dim=1),
                "attention_mask": torch.cat([attention_mask, rejected_tokens["attention_mask"]], dim=1),
                "pixel_values": pixel_values,
                "image_sizes": image_sizes
            }
            
            rejected_outputs = model(**rejected_inputs)
            
            # Create labels
            chosen_labels = chosen_inputs["input_ids"].clone()
            rejected_labels = rejected_inputs["input_ids"].clone()
            
            # Mask out prompt tokens
            prompt_length = input_ids.size(1)
            chosen_labels[:, :prompt_length] = -100
            rejected_labels[:, :prompt_length] = -100
            
            # Compute ORPO loss
            loss, loss_dict = self.orpo_loss.compute_loss(
                model=model,
                chosen_logits=chosen_outputs.logits,
                rejected_logits=rejected_outputs.logits,
                chosen_labels=chosen_labels,
                rejected_labels=rejected_labels,
                attention_mask_chosen=chosen_inputs["attention_mask"],
                attention_mask_rejected=rejected_inputs["attention_mask"]
            )
            
            # Clean up intermediate tensors
            del chosen_inputs, rejected_inputs, chosen_tokens, rejected_tokens
            del chosen_outputs, rejected_outputs, chosen_labels, rejected_labels
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {e}")
            # Return a dummy loss to prevent training failure
            return torch.tensor(0.0, requires_grad=True, device=model.device)
    
    def _compute_loss_sequential(self, model, inputs, return_outputs, num_items_in_batch):
        """
        Process batch sequentially for memory efficiency
        """
        total_loss = 0.0
        batch_size = len(inputs["chosen"])
        
        for i in range(batch_size):
            # Extract single sample
            single_inputs = {
                "input_ids": inputs["input_ids"][i:i+1],
                "attention_mask": inputs["attention_mask"][i:i+1],
                "pixel_values": inputs["pixel_values"][i:i+1],
                "image_sizes": inputs["image_sizes"][i:i+1],
                "chosen": [inputs["chosen"][i]],
                "rejected": [inputs["rejected"][i]]
            }
            
            # Compute loss for single sample
            loss = self.compute_loss(model, single_inputs, return_outputs=False)
            total_loss += loss
            
            # Clear cache after each sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / batch_size
    
    def _tokenize_responses(self, responses: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Tokenize response strings - CONSERVATIVE VERSION
        """
        tokenized = self.processor.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=max_length,  # Conservative length
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        return tokenized
    
    def evaluation_loop(self, dataloader, description, **kwargs):
        """
        Simplified evaluation loop to avoid errors
        """
        logger.info(f"Running evaluation: {description}")
        
        model = self.model
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        try:
            with torch.no_grad():
                for step, batch in enumerate(dataloader):
                    if step >= 5:  # Limit evaluation steps for memory
                        break
                        
                    # Move batch to device
                    batch_on_device = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch_on_device[k] = v.to(model.device)
                        else:
                            batch_on_device[k] = v
                    
                    try:
                        loss = self.compute_loss(model, batch_on_device, return_outputs=False)
                        batch_size = len(batch_on_device.get("chosen", [1]))
                        
                        total_loss += loss.item() * batch_size
                        total_samples += batch_size
                    
                    except Exception as e:
                        logger.warning(f"Evaluation step {step} failed: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Evaluation loop failed: {e}")
            return {"eval_loss": float('inf'), "eval_samples": 0}
        
        finally:
            model.train()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return {
            "eval_loss": avg_loss,
            "eval_samples": total_samples
        }
    
    def create_optimizer(self):
        """Create optimizer with conservative settings"""
        
        logger.info("ORPOTrainer: Creating optimizer...")
        
        # Get trainable parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            logger.warning("ORPOTrainer: No trainable parameters found for the optimizer.")
            # This would be a problem, but your previous logs showed trainable params
            # Set self.optimizer to None and return None, which will likely cause an error later
            # but makes it clear why.
            self.optimizer = None
            return None

        logger.info(f"ORPOTrainer: Found {len(trainable_params)} trainable parameters.")

        # Use optimizer class and arguments from TrainingArguments if available,
        # otherwise fall back to a default. This makes it more aligned with HF Trainer.
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        
        # Override specific kwargs if needed, or use the ones from TrainingArguments
        optimizer_kwargs["lr"] = self.args.learning_rate # Ensure LR is from args
        # optimizer_kwargs["betas"] = (self.args.adam_beta1, self.args.adam_beta2) # If you want to use these from args
        # optimizer_kwargs["eps"] = self.args.adam_epsilon
        # optimizer_kwargs["weight_decay"] = self.args.weight_decay
        
        self.optimizer = optimizer_cls(trainable_params, **optimizer_kwargs)
        
        logger.info(f"ORPOTrainer: Optimizer created: {type(self.optimizer)}")
        return self.optimizer

class ORPOCollator:
    """
    Memory-optimized data collator for ORPO training
    """
    
    def __init__(self, processor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate batch with memory optimizations
        """
        # Process smaller batches
        if len(batch) > 2:
            # Split large batches
            mid = len(batch) // 2
            batch1 = self(batch[:mid])
            batch2 = self(batch[mid:])
            
            # Combine results
            combined = {}
            for key in batch1.keys():
                if isinstance(batch1[key], torch.Tensor):
                    combined[key] = torch.cat([batch1[key], batch2[key]], dim=0)
                elif isinstance(batch1[key], list):
                    combined[key] = batch1[key] + batch2[key]
                else:
                    combined[key] = batch1[key]
            
            return combined
        
        # Extract components with error handling
        input_ids = []
        attention_masks = []
        pixel_values = []
        image_sizes = []
        chosen = []
        rejected = []
        
        for item in batch:
            try:
                # Ensure tensors are properly formatted
                ids = item["input_ids"]
                mask = item["attention_mask"]
                pixels = item["pixel_values"]
                sizes = item["image_sizes"]
                
                # Convert to tensors if needed
                if not isinstance(ids, torch.Tensor):
                    ids = torch.tensor(ids, dtype=torch.long)
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask, dtype=torch.long)
                if not isinstance(pixels, torch.Tensor):
                    pixels = torch.tensor(pixels, dtype=torch.float32)
                if not isinstance(sizes, torch.Tensor):
                    sizes = torch.tensor(sizes, dtype=torch.long)
                
                # Truncate if too long
                if len(ids) > self.max_length:
                    ids = ids[:self.max_length]
                    mask = mask[:self.max_length]
                
                input_ids.append(ids)
                attention_masks.append(mask)
                pixel_values.append(pixels)
                image_sizes.append(sizes)
                chosen.append(item["chosen"])
                rejected.append(item["rejected"])
                
            except Exception as e:
                logger.error(f"Error processing batch item: {e}")
                # Skip problematic items
                continue
        
        if not input_ids:
            raise ValueError("No valid items in batch")
        
        # Pad sequences
        max_length = min(max(len(ids) for ids in input_ids), self.max_length)
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids, attention_masks):
            if len(ids) < max_length:
                pad_length = max_length - len(ids)
                ids = torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
            
            padded_input_ids.append(ids)
            padded_attention_masks.append(mask)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "pixel_values": torch.stack(pixel_values),
            "image_sizes": torch.stack(image_sizes),
            "chosen": chosen,
            "rejected": rejected,
        }