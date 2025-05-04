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
            model_name = config.model.name.split('/')[-1]
            config.wandb.name = f"{model_name}-{config.now}"
        
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            config=config,
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
    """Load toxicity model for reward calculation."""
    
    tokenizer = RobertaTokenizer.from_pretrained(model_id)
    model = RobertaForSequenceClassification.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    return model, tokenizer


def evaluate_toxicity(
    model: Any,
    ppo_trainer: Any,
    tokenizer: Any,
    reward_model: Any,
    reward_tokenizer: Any,
    dataset: Any,
    config: Dict,
    fixed_prompts: Optional[List[int]] = None,
    epoch: Any = None
) -> Tuple[float, List[float]]:
    """Evaluate toxicity level of generated responses."""
    
    # Create output directories
    eval_dir = os.path.join(config.hydra.run.dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Track fixed prompts across epochs for consistency
    prompts_tracking_dir = os.path.join(eval_dir, "prompt_tracking")
    os.makedirs(prompts_tracking_dir, exist_ok=True)
    
    persistent_prompts_file = os.path.join(prompts_tracking_dir, "persistent_prompt_indices.json")
    
    if fixed_prompts is None:
        if os.path.exists(persistent_prompts_file):
            # Load previously selected prompts
            with open(persistent_prompts_file, 'r') as f:
                fixed_prompts = json.load(f)
        else:
            # Select random prompts for tracking
            dataset_size = min(len(dataset), 500)
            fixed_prompts = random.sample(range(dataset_size), min(30, dataset_size))
            with open(persistent_prompts_file, 'w') as f:
                json.dump(fixed_prompts, f)
    
    # Setup generation parameters
    generation_kwargs = {
        "min_length": config.model.generation.min_length,
        "top_k": config.model.generation.top_k,
        "top_p": config.model.generation.top_p,
        "do_sample": config.model.generation.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    output_length_sampler = LengthSampler(
        config.model.generation.output_min_length,
        config.model.generation.output_max_length
    )
    
    # Results tracking
    toxicity_scores = []
    fixed_prompt_results = {}
    
    # Number of samples to evaluate
    n_samples = 100  # Configurable
    eval_indices = range(min(n_samples, len(dataset)))
    
    for idx in tqdm(eval_indices, desc="Evaluating toxicity"):
        # Get the input ids
        item = dataset[idx]
        query = item["input_ids"].to(ppo_trainer.accelerator.device)
        query_text = item["query"]
        
        # PPO trainer expects 1D tensor for query in generate
        query = query.squeeze()
        
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        
        with torch.no_grad():
            # Generate response
            response = ppo_trainer.generate(query, **generation_kwargs)
            
            # Decode the response
            response_text = tokenizer.decode(response.squeeze()[-gen_len:])
            
            # Get toxicity score
            toxicity_inputs = reward_tokenizer(
                response_text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(ppo_trainer.accelerator.device)
            
            # Get both raw logits and softmax probabilities
            logits = reward_model(**toxicity_inputs).logits.float()
            raw_score = logits[0][0].item()
            softmax_score = torch.softmax(logits, dim=1)[0][0].item()
            
            toxicity_scores.append(raw_score)
            
            # Track fixed prompts separately
            if idx in fixed_prompts:
                fixed_prompt_results[idx] = {
                    "prompt": query_text,
                    "response": response_text,
                    "raw_toxicity": raw_score,
                    "softmax_toxicity": softmax_score
                }
    
    # Calculate average toxicity
    avg_toxicity = sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0
    
    # Track fixed prompts across epochs
    if epoch is not None and fixed_prompt_results:
        epoch_identifier = str(epoch) if isinstance(epoch, (int, float)) else epoch
        
        # Save detailed results
        with open(os.path.join(prompts_tracking_dir, f"fixed_prompts_epoch_{epoch_identifier}.json"), 'w') as f:
            json.dump(fixed_prompt_results, f, indent=2)
        
        # Update tracking CSV
        tracking_csv = os.path.join(prompts_tracking_dir, "prompt_tracking.csv")
        
        # Check if file exists
        file_exists = os.path.isfile(tracking_csv)
        
        # Create DataFrame for new data
        rows = []
        for idx, data in fixed_prompt_results.items():
            rows.append({
                'epoch': epoch_identifier,
                'prompt_idx': idx,
                'raw_toxicity': data['raw_toxicity'],
                'softmax_toxicity': data['softmax_toxicity']
            })
        
        new_df = pd.DataFrame(rows)
        
        if file_exists:
            # Append to existing CSV
            existing_df = pd.read_csv(tracking_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(tracking_csv, index=False)
        else:
            # Create new CSV
            new_df.to_csv(tracking_csv, index=False)
    
    return avg_toxicity, toxicity_scores


def analyze_prompt_tracking(tracking_dir: str, output_dir: str) -> None:
    """Analyze the tracked prompts across epochs to show toxicity trends."""
    
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