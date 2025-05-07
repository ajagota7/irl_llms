"""
Utility functions for RLHF training.
"""

import os
import random
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    set_seed
)
from datasets import load_dataset
from omegaconf import OmegaConf


class LengthSampler:
    """Samples a length within a specified range."""
    
    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length
    
    def __call__(self) -> int:
        return random.randint(self.min_length, self.max_length)


def build_dataset(config: Dict) -> Tuple[Any, Any, AutoTokenizer]:
    """Build dataset for RLHF training."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    ds = load_dataset(config.dataset.name, split="train")
    
    # Filter for toxic prompts
    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > config.dataset.toxicity_threshold
    
    ds = ds.filter(filter_fn, batched=False)
    
    # Setup random length sampling
    input_size = LengthSampler(
        config.dataset.input_min_text_length, 
        config.dataset.input_max_text_length
    )
    
    def tokenize(sample):
        # Combine prompt and continuation
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]
        
        # Tokenize and limit to sampled length
        sample["input_ids"] = tokenizer.encode(prompt + continuation)[:input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    
    # Split into train/test
    ds = ds.train_test_split(test_size=config.dataset.test_size, seed=config.training.seed)
    
    train_ds = ds["train"]
    test_ds = ds["test"]
    
    return train_ds, test_ds, tokenizer


def collator(data: List[Dict]) -> Dict:
    """Custom collator function for PPO training."""
    return {key: [d[key] for d in data] for key in data[0]}


def setup_wandb(config: Dict) -> Any:
    """Initialize Weights & Biases logging."""
    try:
        import wandb
        
        if config.wandb.name is None:
            # Get model name safely
            model_name = "unknown-model"
            if config.model.name is not None:
                model_name = config.model.name.split('/')[-1]
            elif hasattr(config.rlhf, 'model') and config.rlhf.model.name is not None:
                model_name = config.rlhf.model.name.split('/')[-1]
                
            config.wandb.name = f"{model_name}-{config.now}"
        
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            config=OmegaConf.to_container(config, resolve=True),
            reinit=True
        )
        
        return run
    except ImportError:
        print("wandb not installed. Skipping wandb initialization.")
        return None
    except Exception as e:
        print(f"Error initializing wandb: {str(e)}")
        return None


def load_reward_model(model_id: str, device: str) -> Tuple[Any, Any]:
    """Load reward model for RLHF training."""
    
    # Check if this is a RoBERTa model (original implementation)
    if "roberta" in model_id.lower():
        from transformers import RobertaForSequenceClassification, RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_id)
        model = RobertaForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    else:
        # Use Auto classes for your custom reward models
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Set padding token for GPTNeoX models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Setting pad_token to eos_token ({tokenizer.eos_token})")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        # Make sure the model knows about the padding token
        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
            print(f"Setting model's pad_token_id to {model.config.pad_token_id}")
    
    return model, tokenizer


def evaluate_toxicity(
    model, 
    ppo_trainer, 
    tokenizer, 
    reward_model, 
    reward_tokenizer, 
    dataset, 
    config, 
    epoch
) -> Tuple[float, List[Dict]]:
    """Evaluate model toxicity on a dataset."""
    
    # Create evaluation directory
    output_dir = os.path.join(os.getcwd(), f"outputs/{config.now}")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    device = ppo_trainer.accelerator.device
    
    # Sample a subset of the dataset for evaluation
    eval_size = min(100, len(dataset))
    eval_indices = random.sample(range(len(dataset)), eval_size)
    eval_samples = [dataset[i] for i in eval_indices]
    
    # Setup generation parameters
    gen_kwargs = {
        "min_length": config.model.generation.min_length,
        "top_k": config.model.generation.top_k,
        "top_p": config.model.generation.top_p,
        "do_sample": config.model.generation.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": config.model.generation.output_max_length,
    }
    
    # Generate responses and calculate toxicity
    toxicity_scores = []
    generations = []
    
    for sample in tqdm(eval_samples, desc=f"Evaluating (epoch {epoch})"):
        query = sample["query"]
        
        # Tokenize the query - PPO trainer expects a list of input_ids, not a tensor
        query_tensor = tokenizer(query, return_tensors="pt")
        query_input_ids = query_tensor.input_ids.squeeze().to(device)  # Convert to 1D tensor
        
        # Generate response
        response_tensor = ppo_trainer.generate(query_input_ids, **gen_kwargs)
        response = tokenizer.decode(response_tensor[0], skip_special_tokens=True)
        
        # Calculate toxicity
        inputs = reward_tokenizer(response, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = reward_model(**inputs)
            toxicity = torch.sigmoid(outputs.logits)[0][0].item()
        
        toxicity_scores.append(toxicity)
        generations.append({
            "query": query,
            "response": response,
            "toxicity": toxicity
        })
    
    # Calculate average toxicity
    avg_toxicity = sum(toxicity_scores) / len(toxicity_scores)
    
    # Save generations to file
    output_file = os.path.join(eval_dir, f"generations_epoch_{epoch}.json")
    with open(output_file, "w") as f:
        json.dump(generations, f, indent=2)
    
    # Create toxicity distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(toxicity_scores, bins=20, alpha=0.7)
    plt.xlabel("Toxicity Score")
    plt.ylabel("Frequency")
    plt.title(f"Toxicity Distribution (Epoch {epoch})")
    plt.axvline(avg_toxicity, color='r', linestyle='dashed', linewidth=2, label=f"Mean: {avg_toxicity:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_file = os.path.join(eval_dir, f"toxicity_dist_epoch_{epoch}.png")
    plt.savefig(plot_file)
    plt.close()
    
    return avg_toxicity, generations


def analyze_prompt_tracking(config: Dict) -> None:
    """Analyze the tracked prompts across epochs to show toxicity trends."""
    
    output_dir = os.path.join(os.getcwd(), f"outputs/{config.now}")
    eval_dir = os.path.join(output_dir, "evaluation")
    tracking_dir = os.path.join(eval_dir, "prompt_tracking")
    
    tracking_csv = os.path.join(tracking_dir, "prompt_tracking.csv")
    
    if not os.path.exists(tracking_csv):
        print("No tracking data found.")
        return
    
    # Load the tracking data
    df = pd.read_csv(tracking_csv)
    
    # Convert epoch to numeric where possible for proper sorting
    def convert_epoch(epoch):
        try:
            return int(epoch)
        except ValueError:
            return epoch
    
    # Only sort numeric epochs
    numeric_epochs = df[df['epoch'].apply(lambda x: str(x).isdigit())]
    if not numeric_epochs.empty:
        numeric_epochs['epoch_num'] = numeric_epochs['epoch'].apply(convert_epoch)
        numeric_epochs = numeric_epochs.sort_values('epoch_num')
        
        # Group by epoch and calculate stats
        epoch_stats = numeric_epochs.groupby('epoch').agg({
            'raw_toxicity': ['mean', 'std', 'min', 'max'],
            'softmax_toxicity': ['mean', 'std', 'min', 'max']
        })
        
        # Save stats to CSV
        epoch_stats.to_csv(os.path.join(output_dir, "epoch_toxicity_stats.csv"))
        
        # Plot average toxicity by epoch
        plt.figure(figsize=(12, 8))
        
        # Plot for each individual prompt
        prompt_ids = numeric_epochs['prompt_idx'].unique()
        for prompt_id in prompt_ids:
            prompt_data = numeric_epochs[numeric_epochs['prompt_idx'] == prompt_id]
            prompt_data = prompt_data.sort_values('epoch_num')
            plt.plot(prompt_data['epoch_num'], prompt_data['raw_toxicity'],
                    alpha=0.3, linewidth=0.5, color='blue')
        
        # Plot the average
        avg_by_epoch = numeric_epochs.groupby('epoch_num')['raw_toxicity'].mean()
        plt.plot(avg_by_epoch.index, avg_by_epoch.values,
                linewidth=3, color='red', label='Average Toxicity')
        
        plt.xlabel('Epoch')
        plt.ylabel('Raw Toxicity Score')
        plt.title('Toxicity Score Evolution Across Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, "toxicity_evolution.png"), dpi=300)