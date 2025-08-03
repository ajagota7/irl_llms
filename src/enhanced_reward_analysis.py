#!/usr/bin/env python
"""
Enhanced reward analysis script with epoch-specific model support and HuggingFace dataset upload.
Supports analyzing models at specific epochs and pushing results to HuggingFace as datasets.
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import json
import random
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.irl_utilities import RewardModel


class EnhancedRewardAnalyzer:
    """Enhanced analyzer for detailed reward model scoring with epoch support and HuggingFace upload."""
    
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
    
    def extract_epoch_from_model_name(self, model_path: str) -> Optional[int]:
        """
        Extract epoch number from model name if it's embedded in the name.
        
        Args:
            model_path: Model path that may contain epoch information
            
        Returns:
            Epoch number if found, None otherwise
        """
        # Look for patterns like "checkpoint-40", "epoch-40", "-40", etc.
        patterns = [
            r'checkpoint-(\d+)',
            r'epoch-(\d+)',
            r'-(\d+)$',  # Number at the end
            r'-(\d+)-',  # Number in the middle
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_path)
            if match:
                return int(match.group(1))
        
        return None
    
    def get_epoch_model_name(self, base_model_path: str, epoch: int) -> str:
        """
        Get the full model name for a specific epoch by appending checkpoint suffix.
        
        Args:
            base_model_path: Base model path without epoch suffix
            epoch: Epoch number to append
            
        Returns:
            Full model name with epoch suffix
        """
        return f"{base_model_path}-checkpoint-{epoch}"
    
    def extract_base_model_name(self, model_path: str) -> str:
        """
        Extract the base model name by removing epoch suffixes.
        
        Args:
            model_path: Model path that may contain epoch information
            
        Returns:
            Base model name without epoch suffix
        """
        # Remove epoch patterns from the end
        patterns = [
            r'-checkpoint-\d+$',
            r'-epoch-\d+$',
            r'-\d+$',  # Number at the end
        ]
        
        base_name = model_path
        for pattern in patterns:
            base_name = re.sub(pattern, '', base_name)
        
        return base_name
    
    def load_reward_model_with_epoch(self, model_path: str, epoch: Optional[int] = None) -> Tuple[RewardModel, AutoTokenizer]:
        """
        Load a trained reward model from a local path or HuggingFace Hub with epoch support.
        
        Args:
            model_path: Path to the model directory or HuggingFace model ID
            epoch: Specific epoch to load (if None, loads final model)
            
        Returns:
            Tuple of (reward_model, tokenizer)
        """
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Handle epoch-specific model loading
        if epoch is not None:
            print(f"Loading reward model from {model_path} at epoch {epoch} on {self.device}...")
        else:
            print(f"Loading final reward model from {model_path} on {self.device}...")
        
        # Check if this is a local path or HuggingFace model ID
        is_local_path = os.path.exists(model_path)
        
        if is_local_path:
            # Load from local path
            if epoch is not None:
                # Look for checkpoint directory
                checkpoint_dir = os.path.join(model_path, f"checkpoint-{epoch}")
                if not os.path.exists(checkpoint_dir):
                    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
                model_file = os.path.join(checkpoint_dir, "model.pt")
            else:
                # Load final model
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
                if epoch is not None:
                    info_file = os.path.join(checkpoint_dir, "training_info.json")
                else:
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
            if epoch is not None:
                tokenizer_path = checkpoint_dir
            else:
                tokenizer_path = model_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            
        else:
            # Assume it's a HuggingFace model ID
            try:
                # Handle epoch-specific HuggingFace models
                if epoch is not None:
                    # Extract base model name and create epoch-specific model name
                    base_model_name = self.extract_base_model_name(model_path)
                    epoch_model_path = self.get_epoch_model_name(base_model_name, epoch)
                    print(f"Using base model: {base_model_name}")
                    print(f"Loading epoch {epoch} model: {epoch_model_path}")
                    epoch_model_paths = [epoch_model_path]
                else:
                    epoch_model_paths = [model_path]
                
                # Try each possible path
                reward_model = None
                tokenizer = None
                
                for try_path in epoch_model_paths:
                    try:
                        print(f"Trying to load from: {try_path}")
                        
                        # First, try to download and load the model files from HuggingFace
                        from huggingface_hub import hf_hub_download
                        import tempfile
                        
                        # Create a temporary directory to download files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Try to download the v_head.pt file first
                            try:
                                v_head_file = hf_hub_download(
                                    repo_id=try_path,
                                    filename="v_head.pt",
                                    cache_dir=temp_dir
                                )
                                print(f"Downloaded v_head.pt to: {v_head_file}")
                                
                                # Load the value head weights
                                v_head_state_dict = torch.load(v_head_file, map_location=self.device)
                                
                                # Try to download reward_model_config.json for base model info
                                try:
                                    config_file = hf_hub_download(
                                        repo_id=try_path,
                                        filename="reward_model_config.json",
                                        cache_dir=temp_dir
                                    )
                                    with open(config_file, 'r') as f:
                                        config = json.load(f)
                                        base_model_name = config.get('base_model')
                                except:
                                    # Fallback: extract from model path
                                    if 'pythia-70m' in try_path:
                                        base_model_name = "EleutherAI/pythia-70m"
                                    elif 'pythia-410m' in try_path:
                                        base_model_name = "EleutherAI/pythia-410m"
                                    elif 'pythia-1b' in try_path:
                                        base_model_name = "EleutherAI/pythia-1b"
                                    elif 'llama-3.2-1b' in try_path:
                                        base_model_name = "meta-llama/Llama-3.2-1B"
                                    else:
                                        continue  # Try next path
                                
                                print(f"Using base model: {base_model_name}")
                                
                                # Create the reward model
                                reward_model = RewardModel(
                                    model_name=base_model_name,
                                    device=self.device,
                                    num_unfrozen_layers=0  # For inference, keep all layers frozen
                                )
                                
                                # Load the value head weights
                                reward_model.v_head.load_state_dict(v_head_state_dict)
                                print(f"Successfully loaded value head weights for {try_path}")
                                
                                # Load the tokenizer
                                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                                tokenizer.pad_token = tokenizer.eos_token
                                tokenizer.padding_side = 'left'
                                
                                # Success! Break out of the loop
                                break
                                
                            except Exception as e:
                                print(f"Failed to download v_head.pt from {try_path}: {e}")
                                continue
                                
                    except Exception as e:
                        print(f"Failed to load from {try_path}: {e}")
                        continue
                
                if reward_model is None:
                    raise ValueError(f"Could not load model from any of the attempted paths: {epoch_model_paths}")
                        
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
    
    def analyze_dataset_pair_with_epoch(self, original_dataset_id: str, detoxified_dataset_id: str,
                                       reward_model_id: str, epoch: Optional[int] = None,
                                       batch_size: int = 64, max_length: int = 512) -> pd.DataFrame:
        """Analyze a dataset pair with detailed scoring breakdown (enhanced with epoch support)."""
        epoch_str = f"epoch-{epoch}" if epoch is not None else "final"
        print(f"\nAnalyzing dataset pair with {epoch_str} model:")
        print(f"Original: {original_dataset_id}")
        print(f"Detoxified: {detoxified_dataset_id}")
        print(f"Model: {reward_model_id}")
        
        # Load model and datasets
        reward_model, tokenizer = self.load_reward_model_with_epoch(reward_model_id, epoch)
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
        
        # Add metadata columns
        # Extract base model name for cleaner metadata
        base_model_name = self.extract_base_model_name(reward_model_id)
        results_df['base_model_id'] = base_model_name
        results_df['full_model_id'] = reward_model_id
        results_df['epoch'] = epoch if epoch is not None else -1  # -1 indicates final model
        results_df['analysis_timestamp'] = datetime.now().isoformat()
        
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
    
    def push_results_to_huggingface(self, results_df: pd.DataFrame, dataset_name: str, 
                                   hub_org: str = None, private: bool = False) -> str:
        """
        Push analysis results to HuggingFace as a dataset.
        
        Args:
            results_df: DataFrame with analysis results
            dataset_name: Name for the dataset
            hub_org: HuggingFace organization/username (if None, uses default)
            private: Whether to make the dataset private
            
        Returns:
            Dataset ID on HuggingFace
        """
        try:
            from huggingface_hub import HfApi, create_repo
            from datasets import Dataset
            
            # Set default organization if not provided
            if hub_org is None:
                hub_org = "ajagota71"  # Default username
            
            # Create dataset ID
            dataset_id = f"{hub_org}/{dataset_name}"
            
            print(f"Pushing results to HuggingFace dataset: {dataset_id}")
            
            # Convert DataFrame to HuggingFace Dataset
            hf_dataset = Dataset.from_pandas(results_df)
            
            # Create repository
            api = HfApi()
            try:
                create_repo(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    private=private,
                    exist_ok=True
                )
                print(f"Created dataset repository: {dataset_id}")
            except Exception as e:
                print(f"Repository creation note: {e}")
            
            # Push dataset to Hub
            hf_dataset.push_to_hub(dataset_id, private=private)
            print(f"Successfully pushed dataset to: {dataset_id}")
            
            # Create a README for the dataset
            readme_content = f"""# {dataset_name}

This dataset contains reward model analysis results for IRL training.

## Dataset Information
- **Base Model ID**: {results_df['base_model_id'].iloc[0] if len(results_df) > 0 else 'Unknown'}
- **Full Model ID**: {results_df['full_model_id'].iloc[0] if len(results_df) > 0 else 'Unknown'}
- **Epoch**: {results_df['epoch'].iloc[0] if len(results_df) > 0 else 'Unknown'}
- **Analysis Timestamp**: {results_df['analysis_timestamp'].iloc[0] if len(results_df) > 0 else 'Unknown'}
- **Number of Samples**: {len(results_df)}

## Columns
- `sample_index`: Index of the sample
- `prompt`: Input prompt (if available)
- `original_output`: Original model output
- `detoxified_output`: Detoxified model output
- `prompt_score`: Reward score for prompt only
- `prompt_original_output_score`: Reward score for prompt + original output
- `prompt_detoxified_output_score`: Reward score for prompt + detoxified output
- `original_output_score`: Reward score for original output only
- `detoxified_output_score`: Reward score for detoxified output only
- `output_improvement`: Improvement in output-only scores
- `prompt_output_improvement`: Improvement in prompt+output scores
- `prompt_contribution_original`: Prompt contribution to original scores
- `prompt_contribution_detoxified`: Prompt contribution to detoxified scores
- `base_model_id`: Base model ID without epoch suffix
- `full_model_id`: Full model ID with epoch suffix
- `epoch`: Training epoch of the model (-1 for final model)
- `analysis_timestamp`: When the analysis was performed

## Usage
```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{dataset_id}")
```
"""
            
            # Upload README
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=dataset_id,
                repo_type="dataset"
            )
            
            return dataset_id
            
        except Exception as e:
            print(f"Error pushing to HuggingFace: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def combine_epoch_results(self, results_list: List[pd.DataFrame], epochs: List[int]) -> pd.DataFrame:
        """
        Combine results from multiple epochs into a single DataFrame.
        
        Args:
            results_list: List of DataFrames from different epochs
            epochs: List of epoch numbers corresponding to the results
            
        Returns:
            Combined DataFrame with all epoch results
        """
        combined_results = []
        
        for i, (df, epoch) in enumerate(zip(results_list, epochs)):
            # Create a copy to avoid modifying the original
            epoch_df = df.copy()
            
            # Update epoch column to the correct epoch number
            epoch_df['epoch'] = epoch
            
            # Add epoch index for sorting
            epoch_df['epoch_index'] = i
            
            combined_results.append(epoch_df)
        
        # Combine all DataFrames
        combined_df = pd.concat(combined_results, ignore_index=True)
        
        # Sort by epoch and sample index
        combined_df = combined_df.sort_values(['epoch', 'sample_index']).reset_index(drop=True)
        
        return combined_df


def main():
    parser = argparse.ArgumentParser(description="Enhanced reward analysis for dataset pairs with epoch support and HuggingFace upload")
    
    parser.add_argument("--original_dataset", type=str, required=True,
                        help="HuggingFace dataset ID for original data")
    parser.add_argument("--detoxified_dataset", type=str, required=True,
                        help="HuggingFace dataset ID for detoxified data")
    parser.add_argument("--reward_model", type=str, required=True,
                        help="HuggingFace model ID for reward model")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch to analyze (if None, analyzes final model)")
    parser.add_argument("--epochs", type=str, default=None,
                        help="Comma-separated list of epochs to analyze (e.g., '5,10,15,20')")
    parser.add_argument("--combine_epochs", action="store_true",
                        help="Combine results from multiple epochs into a single dataset")
    parser.add_argument("--save_individual_epochs", action="store_true",
                        help="Save individual epoch results as separate CSV files")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output CSV file path (default: auto-generated)")
    parser.add_argument("--push_to_hf", action="store_true",
                        help="Push results to HuggingFace as a dataset")
    parser.add_argument("--hf_dataset_name", type=str, default=None,
                        help="Name for HuggingFace dataset (default: auto-generated)")
    parser.add_argument("--hf_org", type=str, default=None,
                        help="HuggingFace organization/username (default: ajagota71)")
    parser.add_argument("--hf_private", action="store_true",
                        help="Make HuggingFace dataset private")
    parser.add_argument("--manual_dataset_name", type=str, default=None,
                        help="Manual short name for HuggingFace dataset (overrides auto-generation)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run models on (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EnhancedRewardAnalyzer(device=args.device, seed=args.seed)
    
    # Handle multiple epochs if specified
    if args.epochs is not None:
        # Parse epochs list
        epoch_list = [int(e.strip()) for e in args.epochs.split(',')]
        print(f"Analyzing multiple epochs: {epoch_list}")
        
        # Analyze each epoch
        all_results = []
        for epoch in epoch_list:
            print(f"\n{'='*50}")
            print(f"Analyzing epoch {epoch}")
            print(f"{'='*50}")
            
            epoch_results = analyzer.analyze_dataset_pair_with_epoch(
                args.original_dataset,
                args.detoxified_dataset,
                args.reward_model,
                epoch,
                args.batch_size,
                args.max_length
            )
            all_results.append(epoch_results)
        
        # Save individual epoch results if requested
        if args.save_individual_epochs:
            print(f"\nSaving individual epoch results...")
            for i, (epoch_results, epoch) in enumerate(zip(all_results, epoch_list)):
                epoch_filename = f"reward_analysis_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                epoch_results.to_csv(epoch_filename, index=False)
                print(f"Saved epoch {epoch} results to: {epoch_filename}")
        
        # Combine results if requested
        if args.combine_epochs:
            print(f"\nCombining results from {len(epoch_list)} epochs...")
            results_df = analyzer.combine_epoch_results(all_results, epoch_list)
        else:
            # Use the last epoch's results (or you could save each separately)
            results_df = all_results[-1]
            print(f"Using results from last epoch ({epoch_list[-1]}) only")
    else:
        # Single epoch analysis
        results_df = analyzer.analyze_dataset_pair_with_epoch(
            args.original_dataset,
            args.detoxified_dataset,
            args.reward_model,
            args.epoch,
            args.batch_size,
            args.max_length
        )
    
    # Generate output filename if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract base model name for cleaner filenames
        base_model_name = analyzer.extract_base_model_name(args.reward_model)
        base_model_short = base_model_name.split('/')[-1]
        
        # Clean up the model name for filename
        base_model_short = re.sub(r'[^a-zA-Z0-9_-]', '', base_model_short)
        
        if args.epochs is not None and args.combine_epochs:
            # Multiple epochs combined
            epoch_list = [int(e.strip()) for e in args.epochs.split(',')]
            epoch_str = f"epochs_{min(epoch_list)}-{max(epoch_list)}"
        elif args.epochs is not None:
            # Multiple epochs, use the last one
            epoch_list = [int(e.strip()) for e in args.epochs.split(',')]
            epoch_str = f"epoch_{epoch_list[-1]}"
        else:
            # Single epoch
            epoch_str = f"epoch_{args.epoch}" if args.epoch is not None else "final"
        
        args.output_file = f"reward_analysis_{base_model_short}_{epoch_str}_{timestamp}.csv"
    
    # Save results locally
    results_df.to_csv(args.output_file, index=False)
    print(f"\nResults saved locally to: {args.output_file}")
    
    # Push to HuggingFace if requested
    if args.push_to_hf:
        # Generate dataset name if not provided
        if args.manual_dataset_name is not None:
            # Use manual dataset name
            args.hf_dataset_name = args.manual_dataset_name
            print(f"Using manual dataset name: {args.hf_dataset_name}")
        elif args.hf_dataset_name is None:
            # Generate a shorter, cleaner dataset name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Extract base model name for cleaner dataset names
            base_model_name = analyzer.extract_base_model_name(args.reward_model)
            base_model_short = base_model_name.split('/')[-1]
            
            # Clean up the model name to make it shorter and valid
            base_model_short = re.sub(r'[^a-zA-Z0-9_-]', '', base_model_short)
            base_model_short = base_model_short.replace('_', '-')
            
            if args.epochs is not None and args.combine_epochs:
                # Multiple epochs combined
                epoch_list = [int(e.strip()) for e in args.epochs.split(',')]
                epoch_str = f"e{min(epoch_list)}-{max(epoch_list)}"
            elif args.epochs is not None:
                # Multiple epochs, use the last one
                epoch_list = [int(e.strip()) for e in args.epochs.split(',')]
                epoch_str = f"e{epoch_list[-1]}"
            else:
                # Single epoch
                epoch_str = f"e{args.epoch}" if args.epoch is not None else "final"
            
            # Create a shorter name that fits within HuggingFace limits
            args.hf_dataset_name = f"reward-{base_model_short}-{epoch_str}-{timestamp}"
            
            # Ensure it's not too long (HuggingFace limit is 96 chars)
            if len(args.hf_dataset_name) > 90:  # Leave some room for org name
                args.hf_dataset_name = f"reward-{epoch_str}-{timestamp}"
        
        print(f"Dataset name: {args.hf_dataset_name}")
        
        dataset_id = analyzer.push_results_to_huggingface(
            results_df, 
            args.hf_dataset_name, 
            args.hf_org, 
            args.hf_private
        )
        
        if dataset_id:
            print(f"Results pushed to HuggingFace: {dataset_id}")
        else:
            print("Failed to push results to HuggingFace")
    
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