"""
Script to apply an ensemble of reward models to new data.
This allows using the ensemble for toxicity detection or for RLHF.
"""

import os
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional, Union

from reward_model_ensemble import RewardModelEnsembleAnalyzer, weighted_ensemble_predictions


def load_ensemble_models(
    model_size: str,
    seeds: List[int],
    checkpoint: int,
    device: str = "cuda"
):
    """
    Load an ensemble of reward models.
    
    Args:
        model_size: Size of the model (70m, 160m, 410m, 1b)
        seeds: List of seeds to use in the ensemble
        checkpoint: Checkpoint number to use
        device: Device to load models on
        
    Returns:
        Dictionary of models and tokenizers
    """
    print(f"Loading ensemble models for pythia-{model_size}...")
    
    reward_models = {}
    reward_tokenizers = {}
    
    for seed in tqdm(seeds, desc="Loading models"):
        model_id = f"ajagota71/toxicity-reward-model-max-margin-seed-{seed}-pythia-{model_size}-checkpoint-{checkpoint}"
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            model.to(device)
            model.eval()
            
            reward_models[seed] = model
            reward_tokenizers[seed] = tokenizer
            
            print(f"Successfully loaded model for seed {seed}")
        except Exception as e:
            print(f"Error loading model for seed {seed}: {e}")
            continue
            
    print(f"Loaded {len(reward_models)} reward models")
    
    return {
        "models": reward_models,
        "tokenizers": reward_tokenizers
    }


def apply_ensemble_to_dataset(
    model_size: str,
    seeds: List[int],
    checkpoint: int,
    dataset_path: str,
    weights: Optional[Dict[int, float]] = None,
    output_path: Optional[str] = None,
    batch_size: int = 16,
    max_length: int = 512,
    text_column: str = "text",
    device: str = "cuda"
):
    """
    Apply an ensemble of reward models to a dataset.
    
    Args:
        model_size: Size of the model (70m, 160m, 410m, 1b)
        seeds: List of seeds to use in the ensemble
        checkpoint: Checkpoint number to use
        dataset_path: Path to the dataset
        weights: Optional dictionary mapping seeds to weights (defaults to equal weights)
        output_path: Path to save the results
        batch_size: Batch size for inference
        max_length: Max sequence length for tokenization
        text_column: Column name containing the text
        device: Device to run inference on
    """
    # Load models
    ensemble = load_ensemble_models(model_size, seeds, checkpoint, device)
    reward_models = ensemble["models"]
    reward_tokenizers = ensemble["tokenizers"]
    
    # Use equal weights if not provided
    if weights is None:
        weights = {seed: 1.0 / len(seeds) for seed in seeds}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {seed: weight / total_weight for seed, weight in weights.items()}
    
    # Load dataset
    try:
        print(f"Loading dataset: {dataset_path}")
        try:
            ds = load_dataset(dataset_path)
            if isinstance(ds, dict) and 'train' in ds:
                ds = ds['train']
        except NotImplementedError as e:
            # Handle LocalFileSystem error by downloading the dataset first
            print(f"Encountered error: {e}")
            print("Attempting to download dataset from HuggingFace first...")
            ds = load_dataset(dataset_path, download_mode="force_redownload")
            if isinstance(ds, dict) and 'train' in ds:
                ds = ds['train']
        
        # Get texts from dataset
        if text_column in ds.column_names:
            texts = ds[text_column]
        else:
            raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {ds.column_names}")
            
        print(f"Loaded {len(texts)} texts from dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Get predictions from each model
    all_predictions = {}
    
    for seed, model in reward_models.items():
        print(f"Getting predictions for seed {seed}...")
        predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Seed {seed}"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = reward_tokenizers[seed](
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                
            # For reward models, we expect a single score
            if hasattr(outputs, "rewards"):
                scores = outputs.rewards.squeeze().cpu().numpy()
            # For classification models
            elif hasattr(outputs, "logits"):
                # Use the first logit (non-toxic) as the score
                scores = outputs.logits[:, 0].cpu().numpy()
            else:
                raise ValueError(f"Unexpected model output format: {type(outputs)}")
                
            # Handle single item case
            if not isinstance(scores, np.ndarray):
                scores = np.array([scores])
                
            predictions.extend(scores)
            
        all_predictions[seed] = np.array(predictions)
    
    # Calculate weighted ensemble predictions
    ensemble_preds = np.zeros(len(texts))
    for seed, preds in all_predictions.items():
        ensemble_preds += weights[seed] * preds
    
    # Create results dictionary
    results = {
        "model_size": model_size,
        "seeds": seeds,
        "weights": weights,
        "dataset": dataset_path,
        "num_examples": len(texts),
        "individual_predictions": {seed: preds.tolist() for seed, preds in all_predictions.items()},
        "ensemble_predictions": ensemble_preds.tolist()
    }
    
    # Save results if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


def main():
    """Main function to apply ensemble models to data."""
    parser = argparse.ArgumentParser(description="Apply ensemble reward models to data")
    parser.add_argument("--model_size", default="70m",
                        help="Model size to use (70m, 160m, 410m, 1b)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 200, 300, 400],
                        help="Seeds to use in the ensemble")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="Checkpoint number to use (defaults to model-specific value)")
    parser.add_argument("--dataset", required=True,
                        help="Path to the dataset")
    parser.add_argument("--text_column", default="text",
                        help="Column name containing the text")
    parser.add_argument("--output", default=None,
                        help="Path to save the results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--weights_file", default=None,
                        help="Path to JSON file containing weights for each seed")
    
    args = parser.parse_args()
    
    # Define checkpoints for each model size if not provided
    default_checkpoints = {
        "70m": 30,
        "160m": 50,
        "410m": 70,
        "1b": 70
    }
    
    checkpoint = args.checkpoint or default_checkpoints.get(args.model_size, 30)
    
    # Load weights if provided
    weights = None
    if args.weights_file:
        try:
            with open(args.weights_file, 'r') as f:
                weights = json.load(f)
                
            # Convert string keys to integers if needed
            if all(isinstance(k, str) for k in weights.keys()):
                weights = {int(k): v for k, v in weights.items()}
                
        except Exception as e:
            print(f"Error loading weights file: {e}")
            print("Using equal weights instead")
    
    # Apply ensemble to dataset
    apply_ensemble_to_dataset(
        model_size=args.model_size,
        seeds=args.seeds,
        checkpoint=checkpoint,
        dataset_path=args.dataset,
        weights=weights,
        output_path=args.output,
        batch_size=args.batch_size,
        text_column=args.text_column
    )


if __name__ == "__main__":
    main() 