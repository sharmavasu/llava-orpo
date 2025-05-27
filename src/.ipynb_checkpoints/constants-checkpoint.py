# # src/constants.py
# DEFAULT_IMAGE_TOKEN = "<image>"
# # You might find LLaVA uses a specific index like -200 for images internally,
# # or it relies on tokenizer_image_token to handle it.
# # For now, the string placeholder is fine for tokenizer_image_token.

# src/constants.py
# Shared constants and configurations for LLaVA-ORPO training

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "llava_1_6_7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_name": "llava-1.6-mistral-7b",
        "base_model": "mistralai/Mistral-7B-Instruct-v0.1"
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    "rlaif_v": {
        "dataset_id": "openbmb/RLAIF-V-Dataset",
        "name": "RLAIF-V-Dataset",
        "splits": ["train"]
    }
}

# Training configurations
TRAINING_CONFIGS = {
    "lora": {
        "r": 2,  # Start small for memory efficiency
        "alpha": 4,
        "dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True
    }
}

# ORPO specific configurations
ORPO_CONFIGS = {
    "beta": 0.1,  # ORPO regularization strength
    "alpha": 1.0,  # Preference loss weight
    "reference_free": True,  # ORPO doesn't need reference model
}

# Training hyperparameters
TRAINING_HYPERPARAMS = {
    "learning_rate": 5e-6,  # Conservative for vision-language models
    "batch_size": 1,  # Small batch size for memory constraints
    "gradient_accumulation_steps": 8,  # Effective batch size = 8
    "max_steps": 1000,  # Start with fewer steps for testing
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 250,
    "eval_steps": 250,
    "max_length": 2560,  # Token length limit
    "dataloader_num_workers": 0,  # Avoid multiprocessing issues
    "dataloader_pin_memory": False,
    "fp16": False,  # Use bf16 instead if supported
    "bf16": True,
    "remove_unused_columns": False,
    "gradient_checkpointing": True,
}

# Evaluation configurations
EVAL_CONFIGS = {
    "metrics": ["accuracy", "preference_accuracy", "response_quality"],
    "num_eval_samples": 100,
    "generation_config": {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    }
}

# Weights & Biases configurations
WANDB_CONFIGS = {
    "project": "llava-orpo-training",
    "entity": None,  # Set this to your wandb username
    "name": None,  # Will be set automatically
    "tags": ["llava", "orpo", "vision-language", "preference-learning"],
    "notes": "LLaVA-1.6 fine-tuning with ORPO on RLAIF-V dataset"
}

# Memory optimization settings
MEMORY_CONFIGS = {
    "use_cpu_offload": True,
    "max_memory_gb": 5.5,  # Conservative for 6GB GPU
    "clear_cache_steps": 50,  # Clear cache every N steps
    "pytorch_cuda_alloc_conf": "max_split_size_mb:128"
}

# Checkpoint and output configurations
OUTPUT_CONFIGS = {
    "output_dir": OUTPUTS_DIR / "checkpoints",
    "logging_dir": OUTPUTS_DIR / "logs",
    "save_total_limit": 3,  # Keep only 3 checkpoints
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_preference_accuracy",
    "greater_is_better": True
}

# Data processing configurations
DATA_PROCESSING_CONFIGS = {
    "image_size": (336, 336),  # Standard LLaVA input size
    "max_samples_train": None,  # None = use all data
    "max_samples_eval": 500,  # Limit eval set for faster evaluation
    "train_ratio": 0.8,
    "val_ratio": 0.15,
    "test_ratio": 0.05,
    "streaming": True,  # Use streaming for large datasets
    "num_proc": 1,  # Single process for stability
}

# Environment variables
ENV_CONFIGS = {
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_VERBOSITY": "info",
    "DATASETS_VERBOSITY": "info"
}

# Set environment variables
for key, value in ENV_CONFIGS.items():
    os.environ[key] = str(value)

# GPU and device configurations
DEVICE_CONFIGS = {
    "device_map": "auto",
    "torch_dtype": "auto",  # Will be determined based on hardware
    "low_cpu_mem_usage": True,
    "use_cache": False  # Disable for training
}

# Validation configurations
VALIDATION_CONFIGS = {
    "validate_data_format": True,
    "min_response_length": 10,
    "max_response_length": 1000,
    "skip_invalid_samples": True,
    "log_invalid_samples": True
}