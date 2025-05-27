#!/bin/bash

# scripts/run_training.sh
# Shell script to launch LLaVA-ORPO training with various configurations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_status "Project root: $PROJECT_ROOT"

# Default configuration
DEFAULT_CONFIG="training_config.yaml"
CONFIG_FILE="$PROJECT_ROOT/configs/$DEFAULT_CONFIG"

# Parse command line arguments
EXPERIMENT_TYPE="default"
MAX_SAMPLES=""
WANDB_PROJECT="llava-orpo"
WANDB_RUN_NAME=""
NO_WANDB=false
DRY_RUN=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --experiment EXPERIMENT  Experiment type: default, quick_test, low_memory, high_performance"
    echo "  -s, --samples MAX_SAMPLES    Maximum number of samples to use (for testing)"
    echo "  -p, --project PROJECT        Weights & Biases project name"
    echo "  -n, --name RUN_NAME          Weights & Biases run name"
    echo "  --no-wandb                   Disable Weights & Biases logging"
    echo "  --dry-run                    Show command without running"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --experiment quick_test --samples 100"
    echo "  $0 --experiment low_memory --no-wandb"
    echo "  $0 --project my-llava-project --name experiment-1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiment)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        -s|--samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        -p|--project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        -n|--name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --no-wandb)
            NO_WANDB=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate experiment type
case $EXPERIMENT_TYPE in
    default|quick_test|low_memory|high_performance)
        print_status "Using experiment type: $EXPERIMENT_TYPE"
        ;;
    *)
        print_error "Invalid experiment type: $EXPERIMENT_TYPE"
        print_error "Valid options: default, quick_test, low_memory, high_performance"
        exit 1
        ;;
esac

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

print_status "Using configuration file: $CONFIG_FILE"

# Set up environment
print_status "Setting up environment..."

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "No virtual environment detected. Consider using a virtual environment."
fi

# Check Python and required packages
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python."
    exit 1
fi

# Check CUDA availability
print_status "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    print_warning "CUDA not available. Training will be very slow on CPU."
fi

# Set memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Create output directories
print_status "Creating output directories..."
mkdir -p "$PROJECT_ROOT/outputs/checkpoints"
mkdir -p "$PROJECT_ROOT/outputs/logs"
mkdir -p "$PROJECT_ROOT/outputs/final_model"

# Build training command
PYTHON_CMD="python $PROJECT_ROOT/src/train.py"

# Add experiment-specific arguments
case $EXPERIMENT_TYPE in
    quick_test)
        PYTHON_CMD="$PYTHON_CMD --max_steps 100"
        if [[ -z "$MAX_SAMPLES" ]]; then
            MAX_SAMPLES="100"
        fi
        ;;
    low_memory)
        PYTHON_CMD="$PYTHON_CMD --lora_r 2 --lora_alpha 4 --batch_size 1 --gradient_accumulation_steps 4"
        ;;
    high_performance)
        PYTHON_CMD="$PYTHON_CMD --lora_r 8 --lora_alpha 16 --max_steps 2000"
        ;;
esac

# Add common arguments
if [[ -n "$MAX_SAMPLES" ]]; then
    PYTHON_CMD="$PYTHON_CMD --max_samples $MAX_SAMPLES"
    print_status "Limiting to $MAX_SAMPLES samples"
fi

if [[ -n "$WANDB_PROJECT" ]]; then
    PYTHON_CMD="$PYTHON_CMD --wandb_project $WANDB_PROJECT"
fi

if [[ -n "$WANDB_RUN_NAME" ]]; then
    PYTHON_CMD="$PYTHON_CMD --wandb_run_name $WANDB_RUN_NAME"
fi

if [[ "$NO_WANDB" == true ]]; then
    PYTHON_CMD="$PYTHON_CMD --no_wandb"
    print_status "Weights & Biases logging disabled"
fi

# Always enable CPU offloading for memory efficiency
PYTHON_CMD="$PYTHON_CMD --cpu_offload"

# Set output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$PROJECT_ROOT/outputs/checkpoints/llava_orpo_${EXPERIMENT_TYPE}_${TIMESTAMP}"
PYTHON_CMD="$PYTHON_CMD --output_dir $OUTPUT_DIR"

print_status "Output directory: $OUTPUT_DIR"

# Show final command
print_status "Training command:"
echo "$PYTHON_CMD"
echo ""

# Run or show command
if [[ "$DRY_RUN" == true ]]; then
    print_status "Dry run - command not executed"
    exit 0
fi

# Check disk space
AVAILABLE_SPACE=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
REQUIRED_SPACE=5000000  # 5GB in KB

if [[ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]]; then
    print_warning "Low disk space detected. Consider freeing up space before training."
fi

# Pre-flight checks
print_status "Running pre-flight checks..."

# Test imports
print_status "Testing Python imports..."
python -c "
import torch
import transformers
import datasets
import peft
print('✅ All required packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'PEFT version: {peft.__version__}')
"

if [[ $? -ne 0 ]]; then
    print_error "Import test failed. Please install required packages."
    print_error "Run: pip install -r requirements.txt"
    exit 1
fi

# Confirm before starting
if [[ "$EXPERIMENT_TYPE" != "quick_test" ]] && [[ -z "$MAX_SAMPLES" ]]; then
    print_warning "This will run a full training session which may take several hours."
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Training cancelled by user"
        exit 0
    fi
fi

# Start training
print_success "Starting LLaVA-ORPO training..."
print_status "Experiment: $EXPERIMENT_TYPE"
print_status "Output: $OUTPUT_DIR"
print_status "Logs will be saved to $OUTPUT_DIR/logs/"

# Change to project directory
cd "$PROJECT_ROOT"

# Run training with error handling
set +e  # Don't exit on error to allow cleanup
eval "$PYTHON_CMD"
TRAINING_EXIT_CODE=$?
set -e

# Check training result
if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    print_success "Training completed successfully!"
    print_success "Model saved to: $OUTPUT_DIR/final_model/"
    print_success "Logs saved to: $OUTPUT_DIR/logs/"
    
    # Show output directory contents
    print_status "Output directory contents:"
    ls -la "$OUTPUT_DIR" 2>/dev/null || true
    
else
    print_error "Training failed with exit code: $TRAINING_EXIT_CODE"
    print_error "Check logs for details: $OUTPUT_DIR/logs/"
    exit $TRAINING_EXIT_CODE
fi

print_success "LLaVA-ORPO training pipeline completed!"