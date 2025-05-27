# src/evaluate.py
# Evaluation script for LLaVA-ORPO trained models

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model_utils import LLaVAModelManager
from data_loader import RLAIFVDataLoader
from constants import PROJECT_ROOT, EVAL_CONFIGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLaVAEvaluator:
    """
    Comprehensive evaluation suite for LLaVA-ORPO models
    """
    
    def __init__(self, model_path: str, config: Dict = None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            config: Evaluation configuration
        """
        self.model_path = model_path
        self.config = config or EVAL_CONFIGS
        self.model = None
        self.processor = None
        self.results = {}
        
        logger.info(f"Initialized evaluator for model: {model_path}")
    
    def load_model(self):
        """Load the trained model and processor"""
        logger.info("Loading trained model...")
        
        try:
            # Load the trained model
            model_manager = LLaVAModelManager()
            
            # Check if it's a PEFT model
            if (Path(self.model_path) / "adapter_config.json").exists():
                # Load base model first
                base_model, processor = model_manager.load_model_and_processor(
                    use_quantization=True,
                    use_lora=False  # We'll load the adapter separately
                )
                
                # Load PEFT adapter
                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, self.model_path)
                
                logger.info("✅ Loaded PEFT model with adapter")
            else:
                # Load full fine-tuned model
                from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
                
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                processor = LlavaNextProcessor.from_pretrained(self.model_path)
                
                logger.info("✅ Loaded full fine-tuned model")
            
            self.model = model
            self.processor = processor
            
            # Set to evaluation mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_response(
        self, 
        image: Image.Image, 
        question: str,
        generation_config: Dict = None
    ) -> str:
        """
        Generate response for image-question pair
        
        Args:
            image: PIL Image
            question: Question text
            generation_config: Generation parameters
            
        Returns:
            Generated response text
        """
        if not self.model or not self.processor:
            raise ValueError("Model must be loaded first")
        
        generation_config = generation_config or self.config["generation_config"]
        
        try:
            # Prepare conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ],
                },
            ]
            
            # Process inputs
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(image, prompt, return_tensors="pt")
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_config["max_new_tokens"],
                    temperature=generation_config["temperature"],
                    do_sample=generation_config["do_sample"],
                    top_p=generation_config["top_p"],
                    repetition_penalty=generation_config["repetition_penalty"],
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Generation Error: {str(e)}]"
    
    def evaluate_preference_accuracy(self, test_dataset) -> Dict[str, float]:
        """
        Evaluate preference accuracy on test dataset
        
        Args:
            test_dataset: Test dataset with chosen/rejected pairs
            
        Returns:
            Dictionary with preference accuracy metrics
        """
        logger.info("Evaluating preference accuracy...")
        
        correct_preferences = 0
        total_samples = 0
        preference_scores = []
        
        for sample in tqdm(test_dataset, desc="Evaluating preferences"):
            try:
                image = sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(sample["image"])
                question = sample["question"]
                chosen_response = sample["chosen"]
                rejected_response = sample["rejected"]
                
                # Generate model response
                generated = self.generate_response(image, question)
                
                # Simple preference scoring: check similarity to chosen vs rejected
                chosen_similarity = self._compute_response_similarity(generated, chosen_response)
                rejected_similarity = self._compute_response_similarity(generated, rejected_response)
                
                if chosen_similarity > rejected_similarity:
                    correct_preferences += 1
                
                preference_scores.append({
                    "chosen_similarity": chosen_similarity,
                    "rejected_similarity": rejected_similarity,
                    "preference_correct": chosen_similarity > rejected_similarity
                })
                
                total_samples += 1
                
                # Limit evaluation for faster testing
                if total_samples >= self.config["num_eval_samples"]:
                    break
                    
            except Exception as e:
                logger.warning(f"Error evaluating sample: {e}")
                continue
        
        accuracy = correct_preferences / total_samples if total_samples > 0 else 0.0
        
        results = {
            "preference_accuracy": accuracy,
            "total_samples": total_samples,
            "correct_preferences": correct_preferences,
            "average_chosen_similarity": np.mean([s["chosen_similarity"] for s in preference_scores]),
            "average_rejected_similarity": np.mean([s["rejected_similarity"] for s in preference_scores])
        }
        
        logger.info(f"Preference accuracy: {accuracy:.4f} ({correct_preferences}/{total_samples})")
        
        return results
    
    def evaluate_response_quality(self, test_samples: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate response quality metrics
        
        Args:
            test_samples: List of test samples
            
        Returns:
            Quality metrics dictionary
        """
        logger.info("Evaluating response quality...")
        
        metrics = {
            "response_lengths": [],
            "response_diversity": [],
            "error_count": 0,
            "sample_responses": []
        }
        
        generated_responses = set()
        
        for i, sample in enumerate(tqdm(test_samples[:self.config["num_eval_samples"]], 
                                      desc="Evaluating quality")):
            try:
                if isinstance(sample, dict):
                    # RLAIF-V format
                    image = sample["image"]
                    question = sample["question"]
                else:
                    # Custom format
                    image, question = sample
                
                # Generate response
                response = self.generate_response(image, question)
                
                # Collect metrics
                metrics["response_lengths"].append(len(response))
                generated_responses.add(response)
                
                # Save sample responses for manual review
                if i < 10:  # Save first 10 for review
                    metrics["sample_responses"].append({
                        "question": question,
                        "response": response,
                        "image_size": image.size if hasattr(image, 'size') else None
                    })
                    
            except Exception as e:
                metrics["error_count"] += 1
                logger.warning(f"Error generating response: {e}")
        
        # Compute aggregate metrics
        metrics["average_response_length"] = np.mean(metrics["response_lengths"]) if metrics["response_lengths"] else 0
        metrics["response_diversity"] = len(generated_responses) / len(metrics["response_lengths"]) if metrics["response_lengths"] else 0
        metrics["error_rate"] = metrics["error_count"] / len(test_samples)
        
        logger.info(f"Average response length: {metrics['average_response_length']:.1f}")
        logger.info(f"Response diversity: {metrics['response_diversity']:.3f}")
        logger.info(f"Error rate: {metrics['error_rate']:.3f}")
        
        return metrics
    
    def evaluate_visual_understanding(self) -> Dict[str, Any]:
        """
        Evaluate visual understanding with curated test cases
        
        Returns:
            Visual understanding metrics
        """
        logger.info("Evaluating visual understanding...")
        
        # Curated test cases for visual understanding
        test_cases = [
            {
                "url": "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true",
                "question": "What type of chart is shown in this image?",
                "expected_keywords": ["radar", "chart", "diagram", "spider", "polar"]
            },
            {
                "url": "https://images.unsplash.com/photo-1518791841217-8f162f1e1131",
                "question": "What animal is in this image?",
                "expected_keywords": ["cat", "kitten", "feline"]
            }
        ]
        
        results = {"test_cases": [], "keyword_accuracy": 0.0}
        correct_cases = 0
        
        for case in test_cases:
            try:
                # Load image
                image = Image.open(requests.get(case["url"], stream=True).raw)
                
                # Generate response
                response = self.generate_response(image, case["question"])
                
                # Check if expected keywords are present
                response_lower = response.lower()
                keywords_found = [kw for kw in case["expected_keywords"] 
                                if kw.lower() in response_lower]
                
                case_correct = len(keywords_found) > 0
                if case_correct:
                    correct_cases += 1
                
                results["test_cases"].append({
                    "question": case["question"],
                    "response": response,
                    "expected_keywords": case["expected_keywords"],
                    "keywords_found": keywords_found,
                    "correct": case_correct
                })
                
            except Exception as e:
                logger.warning(f"Error in visual understanding test: {e}")
                results["test_cases"].append({
                    "question": case["question"],
                    "error": str(e),
                    "correct": False
                })
        
        results["keyword_accuracy"] = correct_cases / len(test_cases)
        
        logger.info(f"Visual understanding accuracy: {results['keyword_accuracy']:.3f}")
        
        return results
    
    def _compute_response_similarity(self, response1: str, response2: str) -> float:
        """
        Compute similarity between two responses
        Simple implementation using word overlap
        
        Args:
            response1: First response
            response2: Second response
            
        Returns:
            Similarity score (0-1)
        """
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def run_comprehensive_evaluation(self, test_dataset=None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation suite
        
        Args:
            test_dataset: Test dataset (optional)
            
        Returns:
            Complete evaluation results
        """
        logger.info("🚀 Running comprehensive evaluation...")
        
        if not self.model:
            self.load_model()
        
        results = {"model_path": self.model_path, "evaluations": {}}
        
        # 1. Preference accuracy evaluation
        if test_dataset:
            logger.info("1️⃣ Evaluating preference accuracy...")
            preference_results = self.evaluate_preference_accuracy(test_dataset)
            results["evaluations"]["preference_accuracy"] = preference_results
        
        # 2. Response quality evaluation
        logger.info("2️⃣ Evaluating response quality...")
        if test_dataset:
            quality_samples = list(test_dataset)[:50]  # Use subset for quality eval
        else:
            # Create some test samples
            quality_samples = self._create_test_samples()
        
        quality_results = self.evaluate_response_quality(quality_samples)
        results["evaluations"]["response_quality"] = quality_results
        
        # 3. Visual understanding evaluation
        logger.info("3️⃣ Evaluating visual understanding...")
        visual_results = self.evaluate_visual_understanding()
        results["evaluations"]["visual_understanding"] = visual_results
        
        # 4. Compute overall scores
        overall_score = self._compute_overall_score(results["evaluations"])
        results["overall_score"] = overall_score
        
        logger.info("✅ Comprehensive evaluation completed!")
        logger.info(f"Overall Score: {overall_score:.3f}")
        
        return results
    
    def _create_test_samples(self) -> List[Tuple[Image.Image, str]]:
        """Create test samples for evaluation"""
        # Simple test images and questions
        test_urls = [
            "https://images.unsplash.com/photo-1518791841217-8f162f1e1131",  # Cat
            "https://images.unsplash.com/photo-1549298916-b41d501d3772",   # Food
        ]
        
        questions = [
            "What do you see in this image?",
            "Describe this image in detail.",
            "What is the main subject of this image?"
        ]
        
        samples = []
        for url in test_urls:
            try:
                image = Image.open(requests.get(url, stream=True).raw)
                for question in questions:
                    samples.append((image, question))
            except:
                continue
        
        return samples
    
    def _compute_overall_score(self, evaluations: Dict) -> float:
        """Compute overall evaluation score"""
        scores = []
        
        if "preference_accuracy" in evaluations:
            scores.append(evaluations["preference_accuracy"]["preference_accuracy"])
        
        if "response_quality" in evaluations:
            # Quality score based on diversity and low error rate
            diversity = evaluations["response_quality"]["response_diversity"]
            error_rate = evaluations["response_quality"]["error_rate"]
            quality_score = diversity * (1 - error_rate)
            scores.append(quality_score)
        
        if "visual_understanding" in evaluations:
            scores.append(evaluations["visual_understanding"]["keyword_accuracy"])
        
        return np.mean(scores) if scores else 0.0
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-ORPO model")
    
    parser.add_argument("model_path", type=str, help="Path to trained model")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--output", type=str, default="outputs/evaluation_results.json",
                       help="Output path for results")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--eval_type", type=str, default="comprehensive",
                       choices=["preference", "quality", "visual", "comprehensive"],
                       help="Type of evaluation to run")
    
    return parser.parse_args()

def main():
    """Main evaluation function"""
    args = parse_args()
    
    logger.info(f"🔍 Starting evaluation of model: {args.model_path}")
    
    # Initialize evaluator
    evaluator = LLaVAEvaluator(args.model_path)
    
    # Load test dataset if provided
    test_dataset = None
    if args.test_data:
        logger.info(f"Loading test dataset: {args.test_data}")
        # This would load your specific test dataset
        # For now, we'll use RLAIF-V test split
        data_loader = RLAIFVDataLoader(max_samples=args.num_samples)
        raw_dataset = data_loader.load_raw_dataset()
        # Use a subset as test data
        test_dataset = list(raw_dataset)[:args.num_samples]
    
    # Run evaluation based on type
    if args.eval_type == "comprehensive":
        results = evaluator.run_comprehensive_evaluation(test_dataset)
    elif args.eval_type == "preference":
        evaluator.load_model()
        results = {"preference_accuracy": evaluator.evaluate_preference_accuracy(test_dataset)}
    elif args.eval_type == "quality":
        evaluator.load_model()
        test_samples = evaluator._create_test_samples()
        results = {"response_quality": evaluator.evaluate_response_quality(test_samples)}
    elif args.eval_type == "visual":
        evaluator.load_model()
        results = {"visual_understanding": evaluator.evaluate_visual_understanding()}
    
    # Save results
    evaluator.save_results(results, args.output)
    
    logger.info("🎉 Evaluation completed successfully!")

if __name__ == "__main__":
    main()