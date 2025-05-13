"""
Script to evaluate text using trained IRL reward models.
This allows for consistent evaluation of text toxicity using models trained with irl_train.py.
"""

import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from typing import List, Dict, Union, Optional
import json

from src.irl_utilities import RewardModel


def load_reward_model(model_path: str, device: str = None) -> tuple:
    """
    Load a trained reward model from a local path or HuggingFace Hub.
    
    Args:
        model_path: Path to the model directory or HuggingFace model ID
        device: Device to load the model on ('cuda', 'cpu', or None for auto-detection)
        
    Returns:
        Tuple of (reward_model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading reward model from {model_path} on {device}...")
    
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
        state_dict = torch.load(model_file, map_location=device)
        
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
            device=device,
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
            # Load the model directly using the class method
            reward_model = RewardModel.load(model_path, device=device)
            
            # Get the base model name
            base_model_name = reward_model.model_name
            
            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            
        except Exception as e:
            # If that fails, try loading as a regular HuggingFace model
            print(f"Failed to load as a reward model: {e}")
            print("Attempting to load as a regular HuggingFace model...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            
            reward_model = RewardModel(
                model_name=model_path,
                device=device,
                num_unfrozen_layers=0  # For inference, keep all layers frozen
            )
    
    print(f"Successfully loaded reward model based on {reward_model.model_name}")
    return reward_model, tokenizer


def score_texts(texts: List[str], reward_model: RewardModel, tokenizer, batch_size: int = 8, 
                max_length: int = 512) -> List[float]:
    """
    Score a list of texts using the reward model.
    
    Args:
        texts: List of text strings to score
        reward_model: The reward model to use
        tokenizer: The tokenizer to use
        batch_size: Batch size for processing
        max_length: Maximum token length
        
    Returns:
        List of reward scores for each text
    """
    reward_model.eval()
    all_scores = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
        
        # Get rewards
        with torch.no_grad():
            rewards = reward_model(**inputs)
        
        # Convert to list of floats
        rewards_list = rewards.squeeze().cpu().tolist()
        
        # Handle single item case
        if not isinstance(rewards_list, list):
            rewards_list = [rewards_list]
        
        all_scores.extend(rewards_list)
    
    return all_scores


def compare_texts(original_texts: List[str], modified_texts: List[str], 
                 reward_model: RewardModel, tokenizer, batch_size: int = 8,
                 max_length: int = 512) -> Dict:
    """
    Compare two sets of texts using the reward model.
    
    Args:
        original_texts: List of original text strings
        modified_texts: List of modified text strings (same length as original_texts)
        reward_model: The reward model to use
        tokenizer: The tokenizer to use
        batch_size: Batch size for processing
        max_length: Maximum token length
        
    Returns:
        Dictionary of comparison metrics
    """
    assert len(original_texts) == len(modified_texts), "Both text lists must have the same length"
    
    # Score both sets of texts
    original_scores = score_texts(original_texts, reward_model, tokenizer, batch_size, max_length)
    modified_scores = score_texts(modified_texts, reward_model, tokenizer, batch_size, max_length)
    
    # Calculate metrics
    metrics = {
        "original_mean": np.mean(original_scores),
        "modified_mean": np.mean(modified_scores),
        "difference_mean": np.mean(np.array(modified_scores) - np.array(original_scores)),
        "original_std": np.std(original_scores),
        "modified_std": np.std(modified_scores),
        "win_rate": np.mean(np.array(modified_scores) > np.array(original_scores)),
        "original_scores": original_scores,
        "modified_scores": modified_scores
    }
    
    return metrics


def plot_comparison(metrics: Dict, title: str = "Reward Model Comparison", 
                   output_path: Optional[str] = None):
    """
    Plot a comparison of original vs modified text scores.
    
    Args:
        metrics: Dictionary of comparison metrics from compare_texts
        title: Title for the plot
        output_path: Path to save the plot (if None, plot is displayed)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot histograms
    ax1.hist(metrics["original_scores"], alpha=0.5, bins=20, label='Original', color='red')
    ax1.hist(metrics["modified_scores"], alpha=0.5, bins=20, label='Modified', color='green')
    
    # Plot means as vertical lines
    ax1.axvline(metrics["original_mean"], color='red', linestyle='--',
               label=f'Mean Original: {metrics["original_mean"]:.4f}')
    ax1.axvline(metrics["modified_mean"], color='green', linestyle='--',
               label=f'Mean Modified: {metrics["modified_mean"]:.4f}')
    
    # Add labels and title
    ax1.set_xlabel('Reward Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Reward Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text with summary statistics
    text = (f"Mean Difference: {metrics['difference_mean']:.4f}\n"
           f"Original Std: {metrics['original_std']:.4f}\n"
           f"Modified Std: {metrics['modified_std']:.4f}\n"
           f"Win Rate: {metrics['win_rate']:.2%}")
    ax1.text(0.02, 0.95, text, transform=ax1.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot paired comparison
    ax2.scatter(metrics["original_scores"], metrics["modified_scores"], alpha=0.5)
    
    # Add diagonal line
    min_val = min(min(metrics["original_scores"]), min(metrics["modified_scores"]))
    max_val = max(max(metrics["original_scores"]), max(metrics["modified_scores"]))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add labels and title
    ax2.set_xlabel('Original Score')
    ax2.set_ylabel('Modified Score')
    ax2.set_title('Paired Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    return fig


def load_texts_from_file(file_path: str) -> List[str]:
    """
    Load texts from a file. Supports txt (one text per line) or jsonl/json formats.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of text strings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    elif extension in ['.json', '.jsonl']:
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            if extension == '.json':
                # Try to load as a JSON array
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                texts.append(item)
                            elif isinstance(item, dict) and 'text' in item:
                                texts.append(item['text'])
                            elif isinstance(item, dict) and 'output' in item:
                                texts.append(item['output'])
                    elif isinstance(data, dict) and 'texts' in data:
                        texts = data['texts']
                except json.JSONDecodeError:
                    # Fall back to JSONL
                    f.seek(0)
                    for line in f:
                        try:
                            item = json.loads(line)
                            if isinstance(item, str):
                                texts.append(item)
                            elif isinstance(item, dict) and 'text' in item:
                                texts.append(item['text'])
                            elif isinstance(item, dict) and 'output' in item:
                                texts.append(item['output'])
                        except json.JSONDecodeError:
                            continue
            else:  # JSONL
                for line in f:
                    try:
                        item = json.loads(line)
                        if isinstance(item, str):
                            texts.append(item)
                        elif isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                        elif isinstance(item, dict) and 'output' in item:
                            texts.append(item['output'])
                    except json.JSONDecodeError:
                        continue
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
    
    return texts


def main():
    parser = argparse.ArgumentParser(description="Evaluate texts using trained IRL reward models")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the reward model directory or HuggingFace model ID")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run the model on (cuda or cpu)")
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, 
                           help="Single text to evaluate")
    input_group.add_argument("--file", type=str,
                           help="File containing texts to evaluate (txt, json, or jsonl)")
    input_group.add_argument("--compare", action="store_true",
                           help="Compare two sets of texts")
    
    # Comparison arguments
    parser.add_argument("--original_file", type=str,
                      help="File containing original texts for comparison")
    parser.add_argument("--modified_file", type=str,
                      help="File containing modified texts for comparison")
    
    # Output arguments
    parser.add_argument("--output", type=str, default=None,
                      help="Output file for results (csv or json)")
    parser.add_argument("--plot", type=str, default=None,
                      help="Output file for plot (png, jpg, pdf)")
    
    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum token length")
    
    args = parser.parse_args()
    
    # Load the model
    reward_model, tokenizer = load_reward_model(args.model_path, args.device)
    
    # Process based on input type
    if args.text:
        # Score a single text
        score = score_texts([args.text], reward_model, tokenizer, 
                           args.batch_size, args.max_length)[0]
        print(f"Score: {score:.6f}")
        
        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"text": args.text, "score": score}, f, indent=2)
    
    elif args.file:
        # Score texts from a file
        texts = load_texts_from_file(args.file)
        print(f"Loaded {len(texts)} texts from {args.file}")
        
        scores = score_texts(texts, reward_model, tokenizer, 
                            args.batch_size, args.max_length)
        
        # Print summary statistics
        print(f"Mean score: {np.mean(scores):.6f}")
        print(f"Std dev: {np.std(scores):.6f}")
        print(f"Min score: {min(scores):.6f}")
        print(f"Max score: {max(scores):.6f}")
        
        # Save output if requested
        if args.output:
            extension = os.path.splitext(args.output)[1].lower()
            
            if extension == '.csv':
                df = pd.DataFrame({'text': texts, 'score': scores})
                df.to_csv(args.output, index=False)
            else:  # Default to JSON
                results = [{"text": text, "score": score} for text, score in zip(texts, scores)]
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            
            print(f"Results saved to {args.output}")
        
        # Create plot if requested
        if args.plot:
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=20)
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Reward Scores')
            plt.grid(True, alpha=0.3)
            plt.savefig(args.plot, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {args.plot}")
    
    elif args.compare:
        # Compare two sets of texts
        if not args.original_file or not args.modified_file:
            parser.error("--compare requires --original_file and --modified_file")
        
        original_texts = load_texts_from_file(args.original_file)
        modified_texts = load_texts_from_file(args.modified_file)
        
        print(f"Loaded {len(original_texts)} original texts and {len(modified_texts)} modified texts")
        
        # Ensure same length
        if len(original_texts) != len(modified_texts):
            print(f"Warning: Different number of texts. Using the first {min(len(original_texts), len(modified_texts))} texts.")
            min_len = min(len(original_texts), len(modified_texts))
            original_texts = original_texts[:min_len]
            modified_texts = modified_texts[:min_len]
        
        # Compare texts
        metrics = compare_texts(
            original_texts, modified_texts, 
            reward_model, tokenizer,
            args.batch_size, args.max_length
        )
        
        # Print summary
        print(f"Original mean score: {metrics['original_mean']:.6f}")
        print(f"Modified mean score: {metrics['modified_mean']:.6f}")
        print(f"Mean difference: {metrics['difference_mean']:.6f}")
        print(f"Win rate (modified > original): {metrics['win_rate']:.2%}")
        
        # Save output if requested
        if args.output:
            extension = os.path.splitext(args.output)[1].lower()
            
            if extension == '.csv':
                df = pd.DataFrame({
                    'original_text': original_texts,
                    'modified_text': modified_texts,
                    'original_score': metrics['original_scores'],
                    'modified_score': metrics['modified_scores'],
                    'difference': np.array(metrics['modified_scores']) - np.array(metrics['original_scores'])
                })
                df.to_csv(args.output, index=False)
            else:  # Default to JSON
                # Remove the raw scores from the metrics for the summary
                summary_metrics = {k: v for k, v in metrics.items() 
                                 if k not in ['original_scores', 'modified_scores']}
                
                results = {
                    "metrics": summary_metrics,
                    "pairs": [
                        {
                            "original_text": orig,
                            "modified_text": mod,
                            "original_score": orig_score,
                            "modified_score": mod_score,
                            "difference": mod_score - orig_score
                        }
                        for orig, mod, orig_score, mod_score in zip(
                            original_texts, modified_texts, 
                            metrics['original_scores'], metrics['modified_scores']
                        )
                    ]
                }
                
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
            
            print(f"Results saved to {args.output}")
        
        # Create plot if requested
        if args.plot:
            plot_comparison(metrics, 
                           title=f"Reward Model Comparison ({os.path.basename(args.model_path)})",
                           output_path=args.plot)


if __name__ == "__main__":
    main() 