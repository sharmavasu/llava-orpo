# # src/orpo_trainer.py
# # Custom ORPO (Odds Ratio Preference Optimization) implementation for LLaVA

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from transformers import Trainer, TrainingArguments
# from transformers.trainer_utils import EvalPrediction
# import numpy as np
# import logging
# from typing import Dict, List, Tuple, Optional, Any, Union
# import wandb

# from constants import ORPO_CONFIGS, TRAINING_HYPERPARAMS, MEMORY_CONFIGS

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ORPOLoss:
#     """
#     Implements ORPO (Odds Ratio Preference Optimization) loss function
    
#     ORPO combines supervised fine-tuning with preference optimization without
#     requiring a reference model, making it more memory efficient.
#     """
    
#     def __init__(self, beta: float = 0.1, alpha: float = 1.0):
#         """
#         Initialize ORPO loss
        
#         Args:
#             beta: Regularization strength for preference optimization
#             alpha: Weight for preference loss component
#         """
#         self.beta = beta
#         self.alpha = alpha
#         logger.info(f"Initialized ORPO loss with beta={beta}, alpha={alpha}")
    
#     def compute_loss(
#         self,
#         model,
#         chosen_logits: torch.Tensor,
#         rejected_logits: torch.Tensor,
#         chosen_labels: torch.Tensor,
#         rejected_labels: torch.Tensor,
#         attention_mask_chosen: torch.Tensor,
#         attention_mask_rejected: torch.Tensor
#     ) -> Tuple[torch.Tensor, Dict[str, float]]:
#         """
#         Compute ORPO loss
        
#         Args:
#             model: The model being trained
#             chosen_logits: Logits for chosen responses
#             rejected_logits: Logits for rejected responses
#             chosen_labels: Labels for chosen responses
#             rejected_labels: Labels for rejected responses  
#             attention_mask_chosen: Attention mask for chosen responses
#             attention_mask_rejected: Attention mask for rejected responses
            
#         Returns:
#             Tuple of (total_loss, loss_dict)
#         """
        
#         # Compute log probabilities for chosen and rejected responses
#         chosen_logprobs = self._compute_sequence_logprobs(
#             chosen_logits, chosen_labels, attention_mask_chosen
#         )
#         rejected_logprobs = self._compute_sequence_logprobs(
#             rejected_logits, rejected_labels, attention_mask_rejected
#         )
        
#         # ORPO preference loss: log(sigmoid(beta * (log_p_chosen - log_p_rejected)))
#         # This encourages chosen responses and discourages rejected ones
#         logits_diff = chosen_logprobs - rejected_logprobs
#         preference_loss = -F.logsigmoid(self.beta * logits_diff).mean()
        
#         # Supervised fine-tuning loss on chosen responses
#         sft_loss = F.cross_entropy(
#             chosen_logits.view(-1, chosen_logits.size(-1)),
#             chosen_labels.view(-1),
#             ignore_index=-100,
#             reduction='mean'
#         )
        
#         # Combine losses
#         total_loss = sft_loss + self.alpha * preference_loss
        
#         # Compute metrics for logging
#         with torch.no_grad():
#             accuracy = (chosen_logprobs > rejected_logprobs).float().mean()
            
#         loss_dict = {
#             'total_loss': total_loss.item(),
#             'sft_loss': sft_loss.item(),
#             'preference_loss': preference_loss.item(),
#             'preference_accuracy': accuracy.item(),
#             'chosen_logprobs': chosen_logprobs.mean().item(),
#             'rejected_logprobs': rejected_logprobs.mean().item(),
#             'logprobs_diff': logits_diff.mean().item()
#         }
        
#         return total_loss, loss_dict
    
#     def _compute_sequence_logprobs(
#         self,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#         attention_mask: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Compute log probabilities for sequences
        
#         Args:
#             logits: Model logits [batch_size, seq_len, vocab_size]
#             labels: Target labels [batch_size, seq_len]
#             attention_mask: Attention mask [batch_size, seq_len]
            
#         Returns:
#             Log probabilities for each sequence in the batch
#         """
#         # Shift logits and labels for next-token prediction
#         shifted_logits = logits[..., :-1, :].contiguous()
#         shifted_labels = labels[..., 1:].contiguous()
#         shifted_mask = attention_mask[..., 1:].contiguous()
        
#         # Compute log probabilities
#         log_probs = F.log_softmax(shifted_logits, dim=-1)
        
#         # Gather log probabilities for target tokens
#         gathered_log_probs = torch.gather(
#             log_probs, 
#             dim=-1, 
#             index=shifted_labels.unsqueeze(-1)
#         ).squeeze(-1)
        
#         # Mask out padding tokens and sum over sequence length
#         masked_log_probs = gathered_log_probs * shifted_mask
#         sequence_log_probs = masked_log_probs.sum(dim=-1) / shifted_mask.sum(dim=-1).clamp(min=1)
        
#         return sequence_log_probs

# class ORPOTrainer(Trainer):
#     """
#     Custom trainer for ORPO optimization
    
#     Extends Hugging Face Trainer to implement ORPO-specific training logic
#     """
    
#     def __init__(
#         self,
#         model,
#         processor,
#         train_dataset,
#         eval_dataset=None,
#         **kwargs
#     ):
#         """
#         Initialize ORPO trainer
        
#         Args:
#             model: The model to train
#             processor: Text/image processor
#             train_dataset: Training dataset
#             eval_dataset: Evaluation dataset
#             **kwargs: Additional arguments for Trainer
#         """
#         self.processor = processor
#         self.orpo_loss = ORPOLoss(
#             beta=ORPO_CONFIGS["beta"],
#             alpha=ORPO_CONFIGS["alpha"]
#         )
        
#         # Initialize base trainer
#         super().__init__(
#             model=model,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             **kwargs
#         )
        
#         logger.info("Initialized ORPO Trainer")
    
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         Compute ORPO loss for a batch
        
#         Args:
#             model: The model
#             inputs: Batch inputs
#             return_outputs: Whether to return model outputs
            
#         Returns:
#             Loss tensor (and optionally outputs)
#         """
#         # Extract inputs for chosen and rejected responses
#         input_ids = inputs["input_ids"]
#         attention_mask = inputs["attention_mask"]
#         pixel_values = inputs["pixel_values"]
#         chosen_responses = inputs["chosen"]
#         rejected_responses = inputs["rejected"]
        
#         batch_size = len(chosen_responses)
        
#         # Tokenize chosen and rejected responses
#         chosen_tokens = self._tokenize_responses(chosen_responses)
#         rejected_tokens = self._tokenize_responses(rejected_responses)
        
#         # Forward pass for chosen responses
#         chosen_inputs = {
#             "input_ids": torch.cat([input_ids, chosen_tokens["input_ids"]], dim=1),
#             "attention_mask": torch.cat([attention_mask, chosen_tokens["attention_mask"]], dim=1),
#             "pixel_values": pixel_values
#         }
        
#         chosen_outputs = model(**chosen_inputs)
        
#         # Forward pass for rejected responses
#         rejected_inputs = {
#             "input_ids": torch.cat([input_ids, rejected_tokens["input_ids"]], dim=1),
#             "attention_mask": torch.cat([attention_mask, rejected_tokens["attention_mask"]], dim=1),
#             "pixel_values": pixel_values
#         }
        
#         rejected_outputs = model(**rejected_inputs)
        
#         # Create labels (only compute loss on response tokens)
#         chosen_labels = chosen_inputs["input_ids"].clone()
#         rejected_labels = rejected_inputs["input_ids"].clone()
        
#         # Mask out prompt tokens (set to -100 to ignore in loss)
#         prompt_length = input_ids.size(1)
#         chosen_labels[:, :prompt_length] = -100
#         rejected_labels[:, :prompt_length] = -100
        
#         # Compute ORPO loss
#         loss, loss_dict = self.orpo_loss.compute_loss(
#             model=model,
#             chosen_logits=chosen_outputs.logits,
#             rejected_logits=rejected_outputs.logits,
#             chosen_labels=chosen_labels,
#             rejected_labels=rejected_labels,
#             attention_mask_chosen=chosen_inputs["attention_mask"],
#             attention_mask_rejected=rejected_inputs["attention_mask"]
#         )
        
#         # Log metrics
#         if self.state.global_step % TRAINING_HYPERPARAMS["logging_steps"] == 0:
#             for key, value in loss_dict.items():
#                 if wandb.run:
#                     wandb.log({f"train/{key}": value}, step=self.state.global_step)
        
#         # Memory cleanup
#         if MEMORY_CONFIGS["clear_cache_steps"] > 0 and \
#            self.state.global_step % MEMORY_CONFIGS["clear_cache_steps"] == 0:
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
        
#         if return_outputs:
#             return loss, {"chosen_outputs": chosen_outputs, "rejected_outputs": rejected_outputs}
        
#         return loss
    
#     def _tokenize_responses(self, responses: List[str]) -> Dict[str, torch.Tensor]:
#         """
#         Tokenize response strings
        
#         Args:
#             responses: List of response strings
            
#         Returns:
#             Dictionary with tokenized responses
#         """
#         tokenized = self.processor.tokenizer(
#             responses,
#             padding=True,
#             truncation=True,
#             max_length=TRAINING_HYPERPARAMS["max_length"] // 2,  # Reserve half for responses
#             return_tensors="pt"
#         )
        
#         # Move to same device as model
#         device = next(self.model.parameters()).device
#         tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
#         return tokenized
    
#     def evaluation_loop(
#         self,
#         dataloader: DataLoader,
#         description: str,
#         prediction_loss_only: Optional[bool] = None,
#         ignore_keys: Optional[List[str]] = None,
#         metric_key_prefix: str = "eval",
#     ):
#         """
#         Custom evaluation loop for ORPO metrics
#         """
#         logger.info(f"Running evaluation: {description}")
        
#         model = self.model
#         model.eval()
        
#         total_loss = 0.0
#         total_samples = 0
#         preference_correct = 0
#         all_metrics = {}
        
#         with torch.no_grad():
#             for batch in dataloader:
#                 # Move batch to device
#                 batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
#                         for k, v in batch.items()}
                
#                 # Compute loss and metrics
#                 loss, loss_dict = self.compute_loss(model, batch, return_outputs=False)
                
#                 batch_size = len(batch["chosen"])
#                 total_loss += loss.item() * batch_size
#                 total_samples += batch_size
                
#                 # Accumulate preference accuracy
#                 if "preference_accuracy" in loss_dict:
#                     preference_correct += loss_dict["preference_accuracy"] * batch_size
                
#                 # Accumulate other metrics
#                 for key, value in loss_dict.items():
#                     if key not in all_metrics:
#                         all_metrics[key] = []
#                     all_metrics[key].append(value)
        
#         model.train()
        
#         # Compute average metrics
#         avg_loss = total_loss / total_samples
#         avg_preference_accuracy = preference_correct / total_samples
        
#         eval_metrics = {
#             f"{metric_key_prefix}_loss": avg_loss,
#             f"{metric_key_prefix}_preference_accuracy": avg_preference_accuracy
#         }
        
#         # Add other averaged metrics
#         for key, values in all_metrics.items():
#             if key != "preference_accuracy":  # Already computed above
#                 eval_metrics[f"{metric_key_prefix}_{key}"] = np.mean(values)
        
#         logger.info(f"Evaluation complete - Loss: {avg_loss:.4f}, Preference Accuracy: {avg_preference_accuracy:.4f}")
        
#         return eval_metrics
    
#     def create_optimizer(self):
#         """Create optimizer with custom settings for ORPO"""
#         # Use AdamW with conservative settings for stability
#         return torch.optim.AdamW(
#             self.model.parameters(),
#             lr=self.args.learning_rate,
#             betas=(0.9, 0.999),
#             eps=1e-8,
#             weight_decay=self.args.weight_decay
#         )

# class ORPOCollator:
#     """
#     Custom data collator for ORPO training
#     """
    
#     def __init__(self, processor, max_length: int = 512):
#         self.processor = processor
#         self.max_length = max_length
    
#     def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
#         """
#         Collate batch for ORPO training
        
#         Args:
#             batch: List of samples
            
#         Returns:
#             Collated batch
#         """
#         # Extract components
#         input_ids = [item["input_ids"] for item in batch]
#         attention_masks = [item["attention_mask"] for item in batch]
#         pixel_values = [item["pixel_values"] for item in batch]
#         chosen = [item["chosen"] for item in batch]
#         rejected = [item["rejected"] for item in batch]
        
#         # Pad sequences to same length
#         max_length = max(len(ids) for ids in input_ids)
#         max_length = min(max_length, self.max_length)
        
#         padded_input_ids = []
#         padded_attention_masks = []
        
#         for ids, mask in zip(input_ids, attention_masks):
#             # Truncate if too long
#             if len(ids) > max_length:
#                 ids = ids[:max_length]
#                 mask = mask[:max_length]
            
#             # Pad if too short
#             pad_length = max_length - len(ids)
#             if pad_length > 0:
#                 padded_ids = torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)])
#                 padded_mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
#             else:
#                 padded_ids = ids
#                 padded_mask = mask
            
#             padded_input_ids.append(padded_ids)
#             padded_attention_masks.append(padded_mask)
        
#         return {
#             "input_ids": torch.stack(padded_input_ids),
#             "attention_mask": torch.stack(padded_attention_masks),
#             "pixel_values": torch.stack(pixel_values),
#             "chosen": chosen,
#             "rejected": rejected,
#         }

# # Utility functions
# def compute_orpo_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
#     """Compute evaluation metrics for ORPO"""
#     predictions, labels = eval_pred
    
#     # This would be implemented based on your specific evaluation needs
#     # For now, return placeholder metrics
#     return {
#         "accuracy": 0.0,
#         "preference_accuracy": 0.0
#     }

# # Testing and demo
# if __name__ == "__main__":
#     logger.info("Testing ORPO Trainer components")
    
#     # Test ORPO loss
#     orpo_loss = ORPOLoss(beta=0.1, alpha=1.0)
#     logger.info("✅ ORPO Loss initialized")
    
#     # Test with dummy data
#     batch_size, seq_len, vocab_size = 2, 10, 1000
    
#     chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
#     rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
#     chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
#     rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
#     attention_mask = torch.ones(batch_size, seq_len)
    
#     loss, loss_dict = orpo_loss.compute_loss(
#         model=None,  # Not used in this test
#         chosen_logits=chosen_logits,
#         rejected_logits=rejected_logits,
#         chosen_labels=chosen_labels,
#         rejected_labels=rejected_labels,
#         attention_mask_chosen=attention_mask,
#         attention_mask_rejected=attention_mask
#     )
    
#     logger.info(f"✅ ORPO Loss computation successful: {loss.item():.4f}")
#     logger.info(f"Loss components: {loss_dict}")
    
#     logger.info("✅ ORPO Trainer components test completed")

# src/orpo_trainer.py
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
    
    ORPO combines supervised fine-tuning with preference optimization without
    requiring a reference model, making it more memory efficient.
    """
    
    def __init__(self, beta: float = 0.1, alpha: float = 1.0):
        """
        Initialize ORPO loss
        
        Args:
            beta: Regularization strength for preference optimization
            alpha: Weight for preference loss component
        """
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
        Compute ORPO loss
        
        Args:
            model: The model being trained
            chosen_logits: Logits for chosen responses
            rejected_logits: Logits for rejected responses
            chosen_labels: Labels for chosen responses
            rejected_labels: Labels for rejected responses  
            attention_mask_chosen: Attention mask for chosen responses
            attention_mask_rejected: Attention mask for rejected responses
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        
        # Compute log probabilities for chosen and rejected responses
        chosen_logprobs = self._compute_sequence_logprobs(
            chosen_logits, chosen_labels, attention_mask_chosen
        )
        rejected_logprobs = self._compute_sequence_logprobs(
            rejected_logits, rejected_labels, attention_mask_rejected
        )
        
        # ORPO preference loss: log(sigmoid(beta * (log_p_chosen - log_p_rejected)))
        # This encourages chosen responses and discourages rejected ones
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
        Compute log probabilities for sequences
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Log probabilities for each sequence in the batch
        """
        # Shift logits and labels for next-token prediction
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        shifted_mask = attention_mask[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        
        # Gather log probabilities for target tokens
        gathered_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding tokens and sum over sequence length
        masked_log_probs = gathered_log_probs * shifted_mask
        sequence_log_probs = masked_log_probs.sum(dim=-1) / shifted_mask.sum(dim=-1).clamp(min=1)
        
        return sequence_log_probs

class ORPOTrainer(Trainer):
    """
    Custom trainer for ORPO optimization
    
    Extends Hugging Face Trainer to implement ORPO-specific training logic
    """
    
    def __init__(
        self,
        model,
        processor,
        train_dataset,
        eval_dataset=None,
        **kwargs
    ):
        """
        Initialize ORPO trainer
        
        Args:
            model: The model to train
            processor: Text/image processor
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            **kwargs: Additional arguments for Trainer
        """
        self.processor = processor
        self.orpo_loss = ORPOLoss(
            beta=ORPO_CONFIGS["beta"],
            alpha=ORPO_CONFIGS["alpha"]
        )
        
        # Initialize base trainer
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        
        logger.info("Initialized ORPO Trainer")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute ORPO loss for a batch - FIXED with image_sizes
        
        Args:
            model: The model
            inputs: Batch inputs
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (new parameter in newer transformers)
            
        Returns:
            Loss tensor (and optionally outputs)
        """
        # Extract inputs for chosen and rejected responses
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs["image_sizes"]  # ← ADDED: Extract image_sizes
        chosen_responses = inputs["chosen"]
        rejected_responses = inputs["rejected"]
        
        batch_size = len(chosen_responses)
        
        # Tokenize chosen and rejected responses
        chosen_tokens = self._tokenize_responses(chosen_responses)
        rejected_tokens = self._tokenize_responses(rejected_responses)
        
        # Forward pass for chosen responses
        chosen_inputs = {
            "input_ids": torch.cat([input_ids, chosen_tokens["input_ids"]], dim=1),
            "attention_mask": torch.cat([attention_mask, chosen_tokens["attention_mask"]], dim=1),
            "pixel_values": pixel_values,
            "image_sizes": image_sizes  # ← ADDED: Pass image_sizes
        }
        
        chosen_outputs = model(**chosen_inputs)
        
        # Forward pass for rejected responses
        rejected_inputs = {
            "input_ids": torch.cat([input_ids, rejected_tokens["input_ids"]], dim=1),
            "attention_mask": torch.cat([attention_mask, rejected_tokens["attention_mask"]], dim=1),
            "pixel_values": pixel_values,
            "image_sizes": image_sizes  # ← ADDED: Pass image_sizes
        }
        
        rejected_outputs = model(**rejected_inputs)
        
        # Create labels (only compute loss on response tokens)
        chosen_labels = chosen_inputs["input_ids"].clone()
        rejected_labels = rejected_inputs["input_ids"].clone()
        
        # Mask out prompt tokens (set to -100 to ignore in loss)
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
        
        # Log metrics
        if self.state.global_step % TRAINING_HYPERPARAMS["logging_steps"] == 0:
            for key, value in loss_dict.items():
                if wandb.run:
                    wandb.log({f"train/{key}": value}, step=self.state.global_step)
        
        # Memory cleanup
        if MEMORY_CONFIGS["clear_cache_steps"] > 0 and \
           self.state.global_step % MEMORY_CONFIGS["clear_cache_steps"] == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if return_outputs:
            return loss, {"chosen_outputs": chosen_outputs, "rejected_outputs": rejected_outputs}
        
        return loss
    
    def _tokenize_responses(self, responses: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize response strings
        
        Args:
            responses: List of response strings
            
        Returns:
            Dictionary with tokenized responses
        """
        tokenized = self.processor.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=256,  # Reserve space for responses (half of total 4096)
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        return tokenized
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        ):
        """
        Custom evaluation loop for ORPO metrics - FIXED VERSION
        """
        logger.info(f"Running evaluation: {description}")
        
        model = self.model
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        preference_correct = 0
        all_metrics = {}
        
        try:
            with torch.no_grad():
                for step, batch in enumerate(dataloader):
                    # Move batch to device - handle both tensor and non-tensor items
                    batch_on_device = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch_on_device[k] = v.to(model.device)
                        else:
                            batch_on_device[k] = v
                    
                    # Compute loss and metrics
                    try:
                        loss = self.compute_loss(model, batch_on_device, return_outputs=False)
                        
                        # For now, use simplified metrics since full ORPO loss might be complex in eval
                        batch_size = len(batch_on_device.get("chosen", []))
                        if batch_size == 0:
                            batch_size = batch_on_device["input_ids"].size(0)
                        
                        total_loss += loss.item() * batch_size
                        total_samples += batch_size
                        
                        # Log progress
                        if step % 10 == 0:
                            logger.info(f"Evaluation step {step}: loss = {loss.item():.4f}")
                    
                    except Exception as e:
                        logger.warning(f"Evaluation step {step} failed: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Evaluation loop failed: {e}")
            # Return minimal metrics to avoid None return
            return {
                f"{metric_key_prefix}_loss": float('inf'),
                f"{metric_key_prefix}_samples": 0
            }
        
        finally:
            model.train()
        
        # Compute average metrics
        if total_samples > 0:
            avg_loss = total_loss / total_samples
        else:
            avg_loss = float('inf')
            total_samples = 0
        
        eval_metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_samples": total_samples
        }
        
        logger.info(f"Evaluation complete - Loss: {avg_loss:.4f}, Samples: {total_samples}")
        
        return eval_metrics
    
    def create_optimizer(self):
        """Create optimizer with custom settings for ORPO - FIXED VERSION"""
        
        # Get decay parameter names using the parent class method
        decay_parameters = self.get_decay_parameter_names(self.model)
        
        # Group parameters for weight decay
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() 
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() 
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        # Create the optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # CRITICAL: Set self.optimizer AND return it
        self.optimizer = optimizer
        return optimizer

class ORPOCollator:
    """
    Custom data collator for ORPO training - FIXED with image_sizes
    """
    
    def __init__(self, processor, max_length: int = 2560):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate batch for ORPO training - FIXED to handle image_sizes
        
        Args:
            batch: List of samples
            
        Returns:
            Collated batch
        """
        # Extract components
        input_ids = []
        attention_masks = []
        pixel_values = []
        image_sizes = []
        
        for item in batch:
            # Convert to tensors if they aren't already
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pixels = item["pixel_values"]
            sizes = item["image_sizes"]  # ← ADDED: Handle image_sizes
            
            # Ensure they are tensors
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids)
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask)
            if not isinstance(pixels, torch.Tensor):
                pixels = torch.tensor(pixels)
            if not isinstance(sizes, torch.Tensor):
                sizes = torch.tensor(sizes)
                
            input_ids.append(ids)
            attention_masks.append(mask)
            pixel_values.append(pixels)
            image_sizes.append(sizes)
        
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]
        
        # Pad sequences to same length
        max_length = max(len(ids) for ids in input_ids)
        max_length = min(max_length, self.max_length)
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids, attention_masks):
            # Truncate if too long
            if len(ids) > max_length:
                ids = ids[:max_length]
                mask = mask[:max_length]
            
            # Pad if too short
            pad_length = max_length - len(ids)
            if pad_length > 0:
                # Create padding tensors with correct dtype
                pad_ids = torch.zeros(pad_length, dtype=ids.dtype)
                pad_mask = torch.zeros(pad_length, dtype=mask.dtype)
                
                padded_ids = torch.cat([ids, pad_ids])
                padded_mask = torch.cat([mask, pad_mask])
            else:
                padded_ids = ids
                padded_mask = mask
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
        
        # Stack tensors
        try:
            stacked_input_ids = torch.stack(padded_input_ids)
            stacked_attention_masks = torch.stack(padded_attention_masks)
            stacked_pixel_values = torch.stack(pixel_values)
            stacked_image_sizes = torch.stack(image_sizes)  # ← ADDED: Stack image_sizes
        except Exception as e:
            # Debug information if stacking still fails
            print(f"Error stacking tensors: {e}")
            print(f"input_ids types: {[type(x) for x in padded_input_ids]}")
            print(f"input_ids shapes: {[x.shape if hasattr(x, 'shape') else len(x) for x in padded_input_ids]}")
            print(f"attention_masks types: {[type(x) for x in padded_attention_masks]}")
            print(f"pixel_values types: {[type(x) for x in pixel_values]}")
            print(f"image_sizes types: {[type(x) for x in image_sizes]}")
            raise
        
        return {
            "input_ids": stacked_input_ids,
            "attention_mask": stacked_attention_masks,
            "pixel_values": stacked_pixel_values,
            "image_sizes": stacked_image_sizes,  # ← ADDED: Include image_sizes
            "chosen": chosen,
            "rejected": rejected,
        }
# Utility functions
def compute_orpo_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute evaluation metrics for ORPO"""
    predictions, labels = eval_pred
    
    # This would be implemented based on your specific evaluation needs
    # For now, return placeholder metrics
    return {
        "accuracy": 0.0,
        "preference_accuracy": 0.0
    }

# Testing and demo
if __name__ == "__main__":
    logger.info("Testing ORPO Trainer components")
    
    # Test ORPO loss
    orpo_loss = ORPOLoss(beta=0.1, alpha=1.0)
    logger.info("✅ ORPO Loss initialized")
    
    # Test with dummy data
    batch_size, seq_len, vocab_size = 2, 10, 1000
    
    chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
    rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
    chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    loss, loss_dict = orpo_loss.compute_loss(
        model=None,  # Not used in this test
        chosen_logits=chosen_logits,
        rejected_logits=rejected_logits,
        chosen_labels=chosen_labels,
        rejected_labels=rejected_labels,
        attention_mask_chosen=attention_mask,
        attention_mask_rejected=attention_mask
    )
    
    logger.info(f"✅ ORPO Loss computation successful: {loss.item():.4f}")
    logger.info(f"Loss components: {loss_dict}")
    
    logger.info("✅ ORPO Trainer components test completed")