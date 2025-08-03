#!/usr/bin/env python
"""
Enhanced simple reward analysis script with fixed classification model scoring.
Takes a dataset pair and model, creates a CSV with detailed scoring breakdown.
Supports multiple model types: reward models, RoBERTa, BERT, and other HuggingFace models.
Uses index 0 logit for classification models instead of averaging.
Now includes HuggingFace dataset upload functionality.
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
from typing import Dict, List, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.irl_utilities import RewardModel


class GenericModelWrapper:
    """Wrapper for different types of models to provide a unified interface."""
    
    def __init__(self, model, tokenizer, model_type: str, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def __call__(self, **inputs):
        """Forward pass that works for different model types."""
        with torch.no_grad():
            if self.model_type == "reward_model":
                return self.model(**inputs)
            elif self.model_type in ["roberta", "bert", "sequence_classification"]:
                # For classification models, use the logits
                outputs = self.model(**inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    if len(logits.shape) == 3:  # [batch, seq_len, num_classes]
                        # For sequence classification, take the first class (index 0) logit
                        # This is typically the "toxic" or "positive" class for toxicity classifiers
                        return logits[:, 0, 0]  # [batch, seq_len, num_classes] -> [batch]
                    else:  # [batch, num_classes]
                        # Take the first class logit
                        return logits[:, 0]  # [batch, num_classes] -> [batch]
                else:
                    # For other models, try to get the last hidden state
                    if hasattr(outputs, 'last_hidden_state'):
                        return outputs.last_hidden_state.mean(dim=1).mean(dim=1)
                    else:
                        # Fallback: try to get any tensor output
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor):
                                return value.mean()
                        raise ValueError("Could not extract scores from model output")
            else:
                # Generic fallback
                outputs = self.model(**inputs)
                if isinstance(outputs, torch.Tensor):
                    return outputs
                elif hasattr(outputs, 'logits'):
                    # For generic models, also use index 0
                    logits = outputs.logits
                    if len(logits.shape) == 3:
                        return logits[:, 0, 0]
                    else:
                        return logits[:, 0]
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")


class EnhancedRewardAnalyzer:
    """Enhanced analyzer for detailed model scoring with support for multiple model types."""
    
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
    
    def detect_model_type(self, model_path: str) -> str:
        """Detect the type of model based on the path or name."""
        model_path_lower = model_path.lower()
        
        # Check for specific model types
        if any(name in model_path_lower for name in ['roberta', 'roberta-base', 'roberta-large']):
            return "roberta"
        elif any(name in model_path_lower for name in ['bert', 'bert-base', 'bert-large']):
            return "bert"
        elif any(name in model_path_lower for name in ['distilbert', 'distilbert-base']):
            return "bert"
        elif any(name in model_path_lower for name in ['albert', 'albert-base']):
            return "bert"
        elif any(name in model_path_lower for name in ['xlnet', 'xlnet-base']):
            return "xlnet"
        elif any(name in model_path_lower for name in ['electra', 'electra-base']):
            return "electra"
        elif any(name in model_path_lower for name in ['deberta', 'deberta-base']):
            return "deberta"
        elif any(name in model_path_lower for name in ['pythia', 'llama', 'gpt', 'neo']):
            return "reward_model"
        else:
            # Default to sequence classification for unknown models
            return "sequence_classification"
    
    def load_model(self, model_path: str, model_type: str = None) -> Tuple[GenericModelWrapper, AutoTokenizer]:
        """
        Load a model from a local path or HuggingFace Hub.
        
        Args:
            model_path: Path to the model directory or HuggingFace model ID
            model_type: Type of model ('reward_model', 'roberta', 'bert', 'sequence_classification', or None for auto-detect)
            
        Returns:
            Tuple of (model_wrapper, tokenizer)
        """
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Auto-detect model type if not provided
        if model_type is None:
            model_type = self.detect_model_type(model_path)
        
        print(f"Loading {model_type} model from {model_path} on {self.device}...")
        
        # Check if this is a local path or HuggingFace model ID
        is_local_path = os.path.exists(model_path)
        
        if model_type == "reward_model":
            # Load reward model using existing logic
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
                from src.irl_utilities import RewardModel
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
                            from src.irl_utilities import RewardModel
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
                    
                    from src.irl_utilities import RewardModel
                    reward_model = RewardModel(
                        model_name=model_path,
                        device=self.device,
                        num_unfrozen_layers=0  # For inference, keep all layers frozen
                    )
            
            model_wrapper = GenericModelWrapper(reward_model, tokenizer, "reward_model", self.device)
            
        else:
            # Load standard HuggingFace models
            try:
                # Try to load as sequence classification model first
                if model_type == "sequence_classification":
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                else:
                    # Load as generic model
                    model = AutoModel.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Set padding token if not set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model_wrapper = GenericModelWrapper(model, tokenizer, model_type, self.device)
                
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")
                raise
        
        print(f"Successfully loaded {model_type} model: {model_path}")
        
        # Test the model on a simple input to see if it's working
        print("Testing model on sample input...")
        test_text = "This is a test sentence."
        test_inputs = tokenizer([test_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        test_inputs = {k: v.to(self.device) for k, v in test_inputs.items()}
        
        with torch.no_grad():
            test_score = model_wrapper(**test_inputs)
            print(f"Test score for '{test_text}': {test_score.item():.6f}")
        
        return model_wrapper, tokenizer
    
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
    
    def score_texts_optimized(self, text_combinations: Dict[str, List[str]], model_wrapper: GenericModelWrapper, 
                             tokenizer: AutoTokenizer, batch_size: int = 64, 
                             max_length: int = 512) -> Dict[str, List[float]]:
        """
        Optimized scoring that processes all text combinations in a single pass.
        
        Args:
            text_combinations: Dict mapping combination names to lists of texts
            model_wrapper: The model wrapper to use for scoring
            tokenizer: The tokenizer
            batch_size: Batch size for processing (increased default for better performance)
            max_length: Maximum token length
            
        Returns:
            Dict mapping combination names to lists of scores
        """
        model_wrapper.model.eval()
        
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
                inputs = {k: v.to(model_wrapper.device) for k, v in inputs.items()}
                
                # Get scores with mixed precision if available
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        scores = model_wrapper(**inputs)
                else:
                    scores = model_wrapper(**inputs)
                
                # Convert to list of floats
                scores_list = scores.squeeze().cpu().tolist()
                if not isinstance(scores_list, list):
                    scores_list = [scores_list]
                
                # Map results back to original structure
                for j, score in enumerate(scores_list):
                    flat_idx = i + j
                    if flat_idx < len(all_texts):
                        combo_name, orig_idx = text_mapping[flat_idx]
                        results[combo_name][orig_idx] = score
        
        return results
    
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

This dataset contains ground truth classification results for model evaluation.

## Dataset Information
- **Model ID**: {results_df.get('model_id', ['Unknown'])[0] if len(results_df) > 0 else 'Unknown'}
- **Model Type**: {results_df.get('model_type', ['Unknown'])[0] if len(results_df) > 0 else 'Unknown'}
- **Analysis Timestamp**: {results_df['analysis_timestamp'].iloc[0] if len(results_df) > 0 else 'Unknown'}
- **Number of Samples**: {len(results_df)}

## Columns
- `sample_index`: Index of the sample
- `prompt`: Input prompt (if available)
- `original_output`: Original model output
- `detoxified_output`: Detoxified model output
- `prompt_score`: Classification score for prompt only
- `prompt_original_output_score`: Classification score for prompt + original output
- `prompt_detoxified_output_score`: Classification score for prompt + detoxified output
- `original_output_score`: Classification score for original output only
- `detoxified_output_score`: Classification score for detoxified output only
- `output_improvement`: Improvement in output-only scores
- `prompt_output_improvement`: Improvement in prompt+output scores
- `prompt_contribution_original`: Prompt contribution to original scores
- `prompt_contribution_detoxified`: Prompt contribution to detoxified scores
- `model_id`: Model ID used for classification
- `model_type`: Type of model used for classification
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
    
    def analyze_dataset_pair(self, original_dataset_id: str, detoxified_dataset_id: str,
                           model_id: str, model_type: str = None, batch_size: int = 64, 
                           max_length: int = 512) -> pd.DataFrame:
        """Analyze a dataset pair with detailed scoring breakdown (enhanced version)."""
        print(f"\nAnalyzing dataset pair:")
        print(f"Original: {original_dataset_id}")
        print(f"Detoxified: {detoxified_dataset_id}")
        print(f"Model: {model_id}")
        print(f"Model type: {model_type or 'auto-detect'}")
        
        # Load model and datasets
        model_wrapper, tokenizer = self.load_model(model_id, model_type)
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
        
        all_scores = self.score_texts_optimized(text_combinations, model_wrapper, tokenizer, batch_size, max_length)
        
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
        results_df['model_id'] = model_id
        results_df['model_type'] = model_type or self.detect_model_type(model_id)
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


def main():
    parser = argparse.ArgumentParser(description="Enhanced simple reward analysis for dataset pairs with multi-model support, fixed classification scoring, and HuggingFace upload")
    
    parser.add_argument("--original_dataset", type=str, required=True,
                        help="HuggingFace dataset ID for original data")
    parser.add_argument("--detoxified_dataset", type=str, required=True,
                        help="HuggingFace dataset ID for detoxified data")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--model_type", type=str, default=None,
                        choices=['reward_model', 'roberta', 'bert', 'sequence_classification'],
                        help="Type of model (auto-detected if not specified)")
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
                        help="Batch size for processing (increased default for better performance)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EnhancedRewardAnalyzer(device=args.device, seed=args.seed)
    
    # Analyze dataset pair
    results_df = analyzer.analyze_dataset_pair(
        args.original_dataset,
        args.detoxified_dataset,
        args.model,
        args.model_type,
        args.batch_size,
        args.max_length
    )
    
    # Generate output filename if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.split('/')[-1]
        detoxified_name = args.detoxified_dataset.split('/')[-1]
        
        args.output_file = f"reward_analysis_{model_name}_{detoxified_name}_{timestamp}.csv"
    
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
            # Extract model name for cleaner dataset names
            model_name = args.model.split('/')[-1]
            
            # Clean up the model name to make it shorter and valid
            model_name = re.sub(r'[^a-zA-Z0-9_-]', '', model_name)
            model_name = model_name.replace('_', '-')
            
            # Create a shorter name that fits within HuggingFace limits
            args.hf_dataset_name = f"ground-truth-{model_name}-{timestamp}"
            
            # Ensure it's not too long (HuggingFace limit is 96 chars)
            if len(args.hf_dataset_name) > 90:  # Leave some room for org name
                args.hf_dataset_name = f"ground-truth-{timestamp}"
        
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