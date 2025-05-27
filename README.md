# LLaVA-ORPO: Aligning LLaVA-1.6 with Human Preferences using ORPO

This project aims to align the LLaVA-1.6 7B Vision Language Model (VLM) using the ORPO (Monolithic Preference Optimization without Reference Model) method on the RLAIF-V human preference dataset.

## Project Goal

The primary goal is to improve the alignment of LLaVA-1.6 with human preferences as expressed in the RLAIF-V dataset. This involves fine-tuning the model to prefer "chosen" responses over "rejected" ones for given image-prompt pairs, using the ORPO algorithm.

## Table of Contents

- [Project Goal](#project-goal)
- [Methodology](#methodology)
  - [Model](#model)
  - [Dataset](#dataset)
  - [Alignment Algorithm](#alignment-algorithm)
  - [Fine-tuning Technique](#fine-tuning-technique)
- [Codebase Structure](#codebase-structure)
- [Setup and Installation](#setup-and-installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Configuration](#configuration)
  - [Launching Training](#launching-training)
  - [Monitoring (WandB)](#monitoring-wandb)
- [Evaluation](#evaluation)
  - [Launching Evaluation](#launching-evaluation)
  - [Metrics](#metrics)
- [Results](#results)
  - [Quantitative](#quantitative)
  - [Qualitative Examples](#qualitative-examples)
  - [Analysis](#analysis)
- [Bonus: Deployment & Advanced Quantization](#bonus-deployment--advanced-quantization)
- [Future Work & Possible Improvements](#future-work--possible-improvements)
- [Acknowledgements](#acknowledgements)

## Methodology

### Model
- **Base VLM:** `liuhaotian/llava-v1.6-mistral-7b` from Hugging Face. This is a LLaVA-1.6 model built upon the Mistral-7B language model.

### Dataset
- **Preference Data:** `openbmb/RLAIF-V-Dataset` from Hugging Face. This dataset contains (image, prompt, chosen_response, rejected_response) tuples.
- We primarily focus on [Specify subset, e.g., "general_preference" or "all combined"] for initial experiments.

### Alignment Algorithm
- **ORPO (Monolithic Preference Optimization without Reference Model):** As described in [https://arxiv.org/abs/2403.07691](https://arxiv.org/abs/2403.07691). This method combines preference pair optimization with a standard supervised fine-tuning (SFT) loss on the chosen responses, without requiring a separate SFT or reference model.

### Fine-tuning Technique
- **Initial Approach:** QLoRA (Quantized Low-Rank Adaptation) is used to efficiently fine-tune the large LLaVA model on available hardware.
    - 4-bit quantization with `bitsandbytes`.
    - LoRA adapters applied to [Specify target modules, e.g., "attention and MLP layers of the LLM, and the multimodal projector"].

## Codebase Structure

The project is organized as follows:

```bash
/llava-orpo-project/
├── src/                      # Core source code
│   ├── data_loader.py        # RLAIF-V dataset loading and preprocessing
│   ├── model_utils.py        # LLaVA model loading, QLoRA setup
│   ├── orpo_trainer.py       # Custom ORPO logic / TRL ORPOTrainer integration
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation script
│   └── constants.py          # Shared constants
│
├── scripts/                  # Utility scripts
│   └── run_training.sh       # Example shell script to launch training
│
├── configs/                  # Configuration files
│   └── training_config.yaml  # Hyperparameters, paths, WandB settings
│
├── data/                     # (Placeholder for local data if not streamed)
│
├── notebooks/                # Jupyter notebooks for exploration and testing
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_testing_base.ipynb
│   ├── 03_interactive_finetuned_test.ipynb
│   └── 04_results_analysis.ipynb
│
├── outputs/                  # Saved models, logs, evaluation results
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```


## Setup and Installation

1.  **Clone the repository (if applicable) or set up the project directory.**
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate 
    # Or using conda
    # conda create -n llava_orpo python=3.10
    # conda activate llava_orpo
    ```
3.  **Install dependencies:**
    Ensure you have a compatible PyTorch version installed for your CUDA version.
    ```bash
    # Example for PyTorch (adjust for your CUDA version from pytorch.org)
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    pip install -r requirements.txt
    
    # The LLaVA codebase is included as a submodule or cloned separately.
    # Ensure it's installed, e.g., if it's in a ./LLaVA subdirectory:
    # cd LLaVA
    # pip install -e . --no-dependencies # Install editable without overriding other deps
    # cd ..
    ```
    *Note: The `requirements.txt` should be generated based on the final working environment.*

## Dataset Preparation

- The `src/data_loader.py` script handles loading the RLAIF-V dataset.
- It processes images using LLaVA's image processor and tokenizes text prompts and chosen/rejected responses according to LLaVA's expected conversational format.
- The data loader prepares batches with `image_tensor`, `chosen_input_ids`, `chosen_attention_mask`, `chosen_labels`, `rejected_input_ids`, `rejected_attention_mask`, `rejected_labels`, and `image_sizes`.

## Training

### Configuration
- Training hyperparameters, model paths, LoRA settings, and WandB details are managed in `configs/training_config.yaml`.
- Key ORPO parameters (e.g., lambda for weighting the SFT loss term) are also set here.

### Launching Training
- The main training script is `src/train.py`.
- It can be launched using the example shell script:
  ```bash
  bash scripts/run_training.sh

  ```markdown
- Or directly:
  ```bash
  python src/train.py --config_path configs/training_config.yaml
  ```

### Monitoring (WandB)
- Training progress, loss curves (overall ORPO loss, preference term, SFT term), learning rate, and evaluation metrics are logged to Weights & Biases (WandB).
- Ensure you are logged into WandB (`wandb login`) before starting training.
- The project name and run name can be set in the `training_config.yaml`.

## Evaluation

### Launching Evaluation
- The `src/evaluate.py` script is used to evaluate a trained model checkpoint.
  ```bash
  python src/evaluate.py \
      --model_checkpoint_path path/to/your/checkpoint \
      --config_path configs/training_config.yaml \
      --output_eval_path path/to/save/eval_results
  ```

### Metrics
- **Preference Accuracy:** Percentage of times the model prefers `y_w` over `y_l`.
- **Log Probabilities:** Average log probability assigned to chosen and rejected responses.
- **SFT Loss on Chosen Responses:** Measures how well the model can generate the preferred responses.
- **Qualitative Examples:** Side-by-side comparisons of (prompt, image, chosen, rejected, model_generation).

## Results
*(This section will be filled in as experiments complete)*

### Quantitative
- [Table of preference accuracy, SFT loss, etc., on the validation set]
- [Link to WandB report/dashboard]

### Qualitative Examples
- [Include a few compelling examples demonstrating successful alignment]
- [Include examples of failure cases or interesting behaviors]

### Analysis
- [Discuss what the results imply about the effectiveness of ORPO on LLaVA for this dataset]
- [Impact of QLoRA or other fine-tuning choices]

## Deployment & Advanced Quantization
- **Gradio Demo:** [Link or instructions if a demo is created]
- **Advanced Quantization (AWQ, GPTQ, etc.):** [Details on methods used and results if implemented]

## Future Work & Possible Improvements

- Experiment with different fine-tuning strategies (e.g., full fine-tuning vs. different PEFT methods, varying LoRA target modules).
- Tune ORPO hyperparameters more extensively.
- Explore the impact of training on different subsets of RLAIF-V (e.g., reasoning vs. general conversation).
- Further pre-training or post-training on external datasets.
- Architectural modifications to LLaVA.
- More sophisticated evaluation, potentially including human evaluation or LLM-as-a-judge.

## Acknowledgements

- The LLaVA team for their foundational model and research.
- The creators of the ORPO algorithm.
- The providers of the RLAIF-V dataset (OpenBMB).
- Hugging Face for their libraries and model hub.
```
