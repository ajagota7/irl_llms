#!/usr/bin/env python
"""
Optimized simple reward analysis script.
Takes a dataset pair and model, creates a CSV with detailed scoring breakdown.
Significantly faster than the original version through batched processing and optimizations.
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.irl_utilities import RewardModel


class OptimizedRewardAnalyzer:
    """Optimized analyzer for detailed reward model scoring with significant performance improvements."""
    
    def __init__(self, device: str = None, seed: int = 42):
        """Initialize the analyzer with performance optimizations."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.set_seeds()
        
        # Enable optimizations for faster inference
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
    def set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def load_reward_model(self, model_path: str) -> Tuple[RewardModel, AutoTokenizer]:
        """
        Load a trained reward model from a local path or HuggingFace Hub.
        
        Args:
            model_path: Path to the model directory or HuggingFace model ID
            device: Device to load the model on ('cuda', 'cpu', or None for auto-detection)
            seed: Random seed for deterministic behavior
            
        Returns:
            Tuple of (reward_model, tokenizer)
        """
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        print(f"Loading reward model from {model_path} on {self.device}...")
        
        # Check if this is a local path or HuggingFace model ID
        is_local_path = os.path.exists(model_path)
        
        if is_local_path:
            # Load from local path
            model_file = os.path.join(model_path, "model.pt")
            if not os.path.exists(model_file):
                # Try to find model.pt in subdirectories
                for root, dirs, files in os.walk(model_path):
                    if "model.pt" in files:
                        model_file = os.path.join(root, "model.pt")
                        break
            
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Could not find model.pt in {model_path}")
            
            # Load the model state dict
            state_dict = torch.load(model_file, map_location=self.device)
            
            # Get the base model name from the config
            if 'config' in state_dict and 'model_name' in state_dict['config']:
                base_model_name = state_dict['config']['model_name']
            else:
                # Try to find training_info.json
                info_file = os.path.join(model_path, "training_info.json")
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                        base_model_name = info.get('model_name')
                else:
                    raise ValueError("Could not determine base model name from model files")
            
            # Create the reward model
            reward_model = RewardModel(
                model_name=base_model_name,
                use_half_precision=state_dict.get('config', {}).get('use_half_precision', False),
                device=self.device,
                num_unfrozen_layers=0  # For inference, keep all layers frozen
            )
            
            # Load the value head weights
            reward_model.v_head.load_state_dict(state_dict['v_head'])
            
            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path))
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            
        else:
            # Assume it's a HuggingFace model ID
            try:
                # First, try to download and load the model files from HuggingFace
                from huggingface_hub import hf_hub_download
                import tempfile
                
                print(f"Attempting to download model files from HuggingFace: {model_path}")
                
                # Create a temporary directory to download files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Try to download the v_head.pt file first
                    try:
                        v_head_file = hf_hub_download(
                            repo_id=model_path,
                            filename="v_head.pt",
                            cache_dir=temp_dir
                        )
                        print(f"Downloaded v_head.pt to: {v_head_file}")
                        
                        # Load the value head weights
                        v_head_state_dict = torch.load(v_head_file, map_location=self.device)
                        
                        # Try to download reward_model_config.json for base model info
                        try:
                            config_file = hf_hub_download(
                                repo_id=model_path,
                                filename="reward_model_config.json",
                                cache_dir=temp_dir
                            )
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                                base_model_name = config.get('base_model')
                        except:
                            # Fallback: extract from model path
                            if 'pythia-70m' in model_path:
                                base_model_name = "EleutherAI/pythia-70m"
                            elif 'pythia-410m' in model_path:
                                base_model_name = "EleutherAI/pythia-410m"
                            elif 'pythia-1b' in model_path:
                                base_model_name = "EleutherAI/pythia-1b"
                            elif 'llama-3.2-1b' in model_path:
                                base_model_name = "meta-llama/Llama-3.2-1B"
                            else:
                                raise ValueError(f"Could not determine base model name for {model_path}")
                        
                        print(f"Using base model: {base_model_name}")
                        
                        # Create the reward model
                        reward_model = RewardModel(
                            model_name=base_model_name,
                            device=self.device,
                            num_unfrozen_layers=0  # For inference, keep all layers frozen
                        )
                        
                        # Load the value head weights
                        reward_model.v_head.load_state_dict(v_head_state_dict)
                        print(f"Successfully loaded value head weights for {model_path}")
                        
                        # Load the tokenizer
                        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                        tokenizer.pad_token = tokenizer.eos_token
                        tokenizer.padding_side = 'left'
                        
                    except Exception as e:
                        print(f"Failed to download v_head.pt: {e}")
                        raise
                        
            except Exception as e:
                print(f"Failed to load as a reward model from HuggingFace: {e}")
                print("Attempting to load as a regular HuggingFace model...")
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = 'left'
                
                reward_model = RewardModel(
                    model_name=model_path,
                    device=self.device,
                    num_unfrozen_layers=0  # For inference, keep all layers frozen
                )
        
        print(f"Successfully loaded reward model based on {reward_model.model_name}")
        
        # Test the model on a simple input to see if it's working
        print("Testing model on sample input...")
        test_text = "This is a test sentence."
        test_inputs = tokenizer([test_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        test_inputs = {k: v.to(reward_model.device) for k, v in test_inputs.items()}
        
        with torch.no_grad():
            test_score = reward_model(**test_inputs)
            print(f"Test score for '{test_text}': {test_score.item():.6f}")
        
        return reward_model, tokenizer
    
    def load_dataset(self, dataset_id: str) -> List[Dict]:
        """Load a dataset from HuggingFace."""
        print(f"Loading dataset: {dataset_id}")
        
        try:
            from datasets import load_dataset
            
            # Try multiple loading strategies
            try:
                dataset = load_dataset(dataset_id)
                if isinstance(dataset, dict) and 'train' in dataset:
                    data = dataset['train']
                else:
                    data = dataset
                
                # Convert to list of dictionaries
                if hasattr(data, 'to_pandas'):
                    data = data.to_pandas().to_dict('records')
                else:
                    data = [item for item in data]
                    
            except Exception as e:
                print(f"Error with default loading: {e}")
                # Try with streaming
                dataset = load_dataset(dataset_id, streaming=True)
                if isinstance(dataset, dict) and 'train' in dataset:
                    data = list(dataset['train'].take(2000))
                else:
                    data = list(dataset.take(2000))
            
            return data
            
        except Exception as e:
            print(f"Error loading dataset {dataset_id}: {e}")
            # Return dummy data for testing
            return [{"output": f"Dummy text {i}"} for i in range(2000)]
    
    def score_texts_optimized(self, text_combinations: Dict[str, List[str]], reward_model: RewardModel, 
                             tokenizer: AutoTokenizer, batch_size: int = 64, 
                             max_length: int = 512) -> Dict[str, List[float]]:
        """
        Optimized scoring that processes all text combinations in a single pass.
        
        Args:
            text_combinations: Dict mapping combination names to lists of texts
            reward_model: The reward model to use for scoring
            tokenizer: The tokenizer
            batch_size: Batch size for processing (increased default for better performance)
            max_length: Maximum token length
            
        Returns:
            Dict mapping combination names to lists of scores
        """
        reward_model.eval()
        
        # Use mixed precision for faster inference if available
        use_amp = torch.cuda.is_available() and hasattr(torch, 'autocast')
        if use_amp:
            print("Using mixed precision for faster inference...")
        
        # Flatten all texts into a single list for batch processing
        all_texts = []
        text_mapping = {}  # Maps flat index to (combination_name, original_index)
        
        for combo_name, texts in text_combinations.items():
            for i, text in enumerate(texts):
                all_texts.append(text)
                text_mapping[len(all_texts) - 1] = (combo_name, i)
        
        # Initialize results
        results = {name: [0.0] * len(texts) for name, texts in text_combinations.items()}
        
        print(f"Processing {len(all_texts)} total texts in batches of {batch_size}...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(all_texts), batch_size), desc="Scoring batches"):
                batch_texts = all_texts[i:i+batch_size]
                
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                
                # Move to device
                inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
                
                # Get rewards with mixed precision if available
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        rewards = reward_model(**inputs)
                else:
                    rewards = reward_model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.squeeze().cpu().tolist()
                if not isinstance(rewards_list, list):
                    rewards_list = [rewards_list]
                
                # Map results back to original structure
                for j, score in enumerate(rewards_list):
                    flat_idx = i + j
                    if flat_idx < len(all_texts):
                        combo_name, orig_idx = text_mapping[flat_idx]
                        results[combo_name][orig_idx] = score
        
        return results
    
    def analyze_dataset_pair(self, original_dataset_id: str, detoxified_dataset_id: str,
                           reward_model_id: str, batch_size: int = 64, 
                           max_length: int = 512) -> pd.DataFrame:
        """Analyze a dataset pair with detailed scoring breakdown (optimized version)."""
        print(f"\nAnalyzing dataset pair:")
        print(f"Original: {original_dataset_id}")
        print(f"Detoxified: {detoxified_dataset_id}")
        print(f"Model: {reward_model_id}")
        
        # Load model and datasets
        reward_model, tokenizer = self.load_reward_model(reward_model_id)
        original_data = self.load_dataset(original_dataset_id)
        detoxified_data = self.load_dataset(detoxified_dataset_id)
        
        # Ensure datasets have same length
        min_len = min(len(original_data), len(detoxified_data))
        original_data = original_data[:min_len]
        detoxified_data = detoxified_data[:min_len]
        
        print(f"Analyzing {min_len} samples")
        
        # Extract texts
        original_outputs = [item.get('output', item.get('text', '')) for item in original_data]
        detoxified_outputs = [item.get('output', item.get('text', '')) for item in detoxified_data]
        
        # Extract prompts if available
        prompts = []
        for item in original_data:
            if 'prompt' in item:
                prompts.append(item['prompt'])
            elif 'input' in item:
                prompts.append(item['input'])
            else:
                prompts.append("")  # No prompt available
        
        # Create different text combinations for scoring
        prompt_only_texts = prompts
        prompt_original_texts = [f"{prompt} {output}" if prompt else output 
                               for prompt, output in zip(prompts, original_outputs)]
        prompt_detoxified_texts = [f"{prompt} {output}" if prompt else output 
                                 for prompt, output in zip(prompts, detoxified_outputs)]
        original_output_only_texts = original_outputs
        detoxified_output_only_texts = detoxified_outputs
        
        # Use optimized scoring that processes all combinations in a single pass
        print("Scoring all text combinations in a single optimized pass...")
        text_combinations = {
            'prompt_only': prompt_only_texts,
            'prompt_original': prompt_original_texts,
            'prompt_detoxified': prompt_detoxified_texts,
            'original_output': original_output_only_texts,
            'detoxified_output': detoxified_output_only_texts
        }
        
        all_scores = self.score_texts_optimized(text_combinations, reward_model, tokenizer, batch_size, max_length)
        
        # Extract individual score lists
        prompt_scores = all_scores['prompt_only']
        prompt_original_scores = all_scores['prompt_original']
        prompt_detoxified_scores = all_scores['prompt_detoxified']
        original_output_scores = all_scores['original_output']
        detoxified_output_scores = all_scores['detoxified_output']
        
        # Create results dataframe
        results_data = []
        for i in range(min_len):
            results_data.append({
                'sample_index': i,
                'prompt': prompts[i],
                'original_output': original_outputs[i],
                'detoxified_output': detoxified_outputs[i],
                'prompt_score': prompt_scores[i],
                'prompt_original_output_score': prompt_original_scores[i],
                'prompt_detoxified_output_score': prompt_detoxified_scores[i],
                'original_output_score': original_output_scores[i],
                'detoxified_output_score': detoxified_output_scores[i],
                'output_improvement': detoxified_output_scores[i] - original_output_scores[i],
                'prompt_output_improvement': prompt_detoxified_scores[i] - prompt_original_scores[i],
                'prompt_contribution_original': prompt_original_scores[i] - original_output_scores[i],
                'prompt_contribution_detoxified': prompt_detoxified_scores[i] - detoxified_output_scores[i]
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Add summary statistics
        print("\nSummary Statistics:")
        print(f"Mean output improvement: {results_df['output_improvement'].mean():.4f}")
        print(f"Mean prompt+output improvement: {results_df['prompt_output_improvement'].mean():.4f}")
        print(f"Mean prompt contribution (original): {results_df['prompt_contribution_original'].mean():.4f}")
        print(f"Mean prompt contribution (detoxified): {results_df['prompt_contribution_detoxified'].mean():.4f}")
        
        # Debug: Show some examples of how prompts affect scoring
        print("\nDebug: Sample scoring differences:")
        sample_indices = [0, 100, 500, 1000, 1500]  # Look at a few samples
        for idx in sample_indices:
            if idx < len(results_df):
                row = results_df.iloc[idx]
                print(f"\nSample {idx}:")
                print(f"  Prompt: '{row['prompt'][:50]}...'")
                print(f"  Original output: '{row['original_output'][:50]}...'")
                print(f"  Detoxified output: '{row['detoxified_output'][:50]}...'")
                print(f"  Output-only scores: {row['original_output_score']:.4f} -> {row['detoxified_output_score']:.4f} (diff: {row['output_improvement']:.4f})")
                print(f"  Prompt+output scores: {row['prompt_original_output_score']:.4f} -> {row['prompt_detoxified_output_score']:.4f} (diff: {row['prompt_output_improvement']:.4f})")
                print(f"  Prompt contribution: {row['prompt_contribution_original']:.4f} (orig), {row['prompt_contribution_detoxified']:.4f} (detox)")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description="Optimized simple reward analysis for dataset pairs")
    
    parser.add_argument("--original_dataset", type=str, required=True,
                        help="HuggingFace dataset ID for original data")
    parser.add_argument("--detoxified_dataset", type=str, required=True,
                        help="HuggingFace dataset ID for detoxified data")
    parser.add_argument("--reward_model", type=str, required=True,
                        help="HuggingFace model ID for reward model")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output CSV file path (default: auto-generated)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run models on (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for processing (increased default for better performance)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = OptimizedRewardAnalyzer(device=args.device, seed=args.seed)
    
    # Analyze dataset pair
    results_df = analyzer.analyze_dataset_pair(
        args.original_dataset,
        args.detoxified_dataset,
        args.reward_model,
        args.batch_size,
        args.max_length
    )
    
    # Generate output filename if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.reward_model.split('/')[-1]
        args.output_file = f"reward_analysis_optimized_{model_name}_{timestamp}.csv"
    
    # Save results
    results_df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to: {args.output_file}")
    
    # Print top and bottom improvers
    print("\nTop 10 Output Improvers:")
    top_improvers = results_df.nlargest(10, 'output_improvement')
    for _, row in top_improvers.iterrows():
        print(f"Sample {row['sample_index']}: {row['output_improvement']:.4f}")
    
    print("\nBottom 10 Output Improvers:")
    bottom_improvers = results_df.nsmallest(10, 'output_improvement')
    for _, row in bottom_improvers.iterrows():
        print(f"Sample {row['sample_index']}: {row['output_improvement']:.4f}")


if __name__ == "__main__":
    main() 