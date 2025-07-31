#!/usr/bin/env python
"""
Simple reward analysis script.
Takes a dataset pair and model, creates a CSV with detailed scoring breakdown.
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
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.irl_utilities import RewardModel


class SimpleRewardAnalyzer:
    """Simple analyzer for detailed reward model scoring."""
    
    def __init__(self, device: str = None, seed: int = 42):
        """Initialize the analyzer."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.set_seeds()
        
    def set_seeds(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def load_reward_model(self, model_id: str) -> Tuple[RewardModel, AutoTokenizer]:
        """Load a reward model and tokenizer."""
        print(f"Loading reward model: {model_id}")
        
        try:
            # Create reward model
            reward_model = RewardModel(
                model_name=model_id,
                device=self.device,
                num_unfrozen_layers=0  # For inference
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            
            return reward_model, tokenizer
            
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            raise
    
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
    
    def score_texts(self, texts: List[str], reward_model: RewardModel, 
                   tokenizer: AutoTokenizer, batch_size: int = 8, 
                   max_length: int = 512) -> List[float]:
        """Score a list of texts using the reward model."""
        reward_model.eval()
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                
                # Move to device
                inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
                
                # Get rewards
                rewards = reward_model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.squeeze().cpu().tolist()
                if not isinstance(rewards_list, list):
                    rewards_list = [rewards_list]
                
                all_scores.extend(rewards_list)
        
        return all_scores
    
    def analyze_dataset_pair(self, original_dataset_id: str, detoxified_dataset_id: str,
                           reward_model_id: str, batch_size: int = 8, 
                           max_length: int = 512) -> pd.DataFrame:
        """Analyze a dataset pair with detailed scoring breakdown."""
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
        
        # Score all combinations
        print("Scoring prompt only...")
        prompt_scores = self.score_texts(prompt_only_texts, reward_model, tokenizer, batch_size, max_length)
        
        print("Scoring prompt + original output...")
        prompt_original_scores = self.score_texts(prompt_original_texts, reward_model, tokenizer, batch_size, max_length)
        
        print("Scoring prompt + detoxified output...")
        prompt_detoxified_scores = self.score_texts(prompt_detoxified_texts, reward_model, tokenizer, batch_size, max_length)
        
        print("Scoring original output only...")
        original_output_scores = self.score_texts(original_output_only_texts, reward_model, tokenizer, batch_size, max_length)
        
        print("Scoring detoxified output only...")
        detoxified_output_scores = self.score_texts(detoxified_output_only_texts, reward_model, tokenizer, batch_size, max_length)
        
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
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description="Simple reward analysis for dataset pairs")
    
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
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SimpleRewardAnalyzer(device=args.device, seed=args.seed)
    
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
        args.output_file = f"reward_analysis_{model_name}_{timestamp}.csv"
    
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