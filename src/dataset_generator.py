"""
Dataset generator for IRL detoxification.
Generates and processes datasets from toxic prompts.
"""

import os
import random
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import wandb
from omegaconf import DictConfig, OmegaConf


class LengthSampler:
    """Samples a length within a specified range."""
    
    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length
    
    def __call__(self) -> int:
        return random.randint(self.min_length, self.max_length)


class DatasetGenerator:
    """Class for generating datasets for IRL experiments with a single model."""

    def __init__(self, config: DictConfig):
        """Initialize the dataset generator."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract the model name for dataset identification
        model_name = config.dataset.original_model_name if hasattr(config.dataset, 'original_model_name') else config.dataset.model_name
        model_name = model_name.replace('/', '_')
        
        # Create a unique identifier for this dataset generation
        self.dataset_id = f"{model_name}_{config.dataset.num_samples}_samples"
        
        # Set random seed
        torch.manual_seed(config.dataset.seed)
        np.random.seed(config.dataset.seed)
        random.seed(config.dataset.seed)

        # Initialize paths
        os.makedirs(config.dataset.cache_dir, exist_ok=True)
        self.output_path = os.path.join(config.dataset.cache_dir, f"{self.dataset_id}.json")
        self.csv_path = os.path.join(config.dataset.cache_dir, f"{self.dataset_id}.csv")
        
        # Initialize data containers
        self.prompts = []
        self.generated_data = []

    def load_prompts(self):
        """Load toxic prompts from the RealToxicityPrompts dataset."""
        print("Loading RealToxicityPrompts dataset...")
        ds = load_dataset("allenai/real-toxicity-prompts", split="train")

        # Filter for prompts with toxicity > threshold
        def filter_fn(sample):
            toxicity = sample["prompt"]["toxicity"]
            return toxicity is not None and toxicity > self.config.dataset.toxicity_threshold

        ds = ds.filter(filter_fn, batched=False)

        # Select the required number of samples
        num_samples = min(self.config.dataset.num_samples, len(ds))
        ds = ds.select(range(num_samples))

        # Extract prompts
        self.prompts = [example["prompt"]["text"] for example in ds]

        print(f"Loaded {len(self.prompts)} prompts")
        return self.prompts

    def generate_completions(self):
        """Generate completions from the specified model."""
        model_name = self.config.dataset.original_model_name if hasattr(self.config.dataset, 'original_model_name') else self.config.dataset.model_name
        print(f"Loading model: {model_name}")

        # Check if we should use half precision
        use_half_precision = self.config.dataset.use_half_precision
        if use_half_precision is None:
            # Automatically detect based on model size
            large_model = any(size in model_name.lower() for size in ["2.7b", "6b", "7b", "12b", "70b", "large"])
            use_half_precision = large_model

        # Load model with appropriate settings
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16 if use_half_precision else None,
                trust_remote_code=True  # Needed for some models
            )
        except Exception as e:
            print(f"Error loading model with bfloat16: {e}")
            print("Trying again with float16...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if use_half_precision else None,
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"Error loading model with float16: {e2}")
                print("Falling back to full precision...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    trust_remote_code=True
                )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # Fix for decoder-only models

        print(f"Generating completions from the model...")
        completions = []

        for i in tqdm(range(0, len(self.prompts), self.config.dataset.batch_size)):
            batch_prompts = self.prompts[i:i+self.config.dataset.batch_size]

            # Tokenize the batch
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1000)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate completions with specified parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.dataset.max_new_tokens,
                    do_sample=(self.config.dataset.temperature > 0),
                    temperature=self.config.dataset.temperature,
                    top_p=self.config.dataset.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode the outputs
            batch_completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Extract only the new tokens (remove the prompt)
            for j, completion in enumerate(batch_completions):
                # Replace the prompt to get only the generated part
                new_text = completion.replace(batch_prompts[j], "", 1).strip()
                completions.append(new_text)

        # Free up memory
        del model
        torch.cuda.empty_cache()

        return completions

    def create_dataset(self):
        """Generate dataset from the model."""
        # Check if we should use cached dataset
        if self.config.dataset.use_cached and os.path.exists(self.output_path):
            print("Using cached dataset")
            return self.load_dataset()

        # Load prompts if not already loaded
        if not self.prompts:
            self.load_prompts()

        # Generate completions
        completions = self.generate_completions()

        # Create dataset
        print("Creating dataset...")
        self.generated_data = []

        # Get model name for metadata
        model_name = self.config.dataset.original_model_name if hasattr(self.config.dataset, 'original_model_name') else self.config.dataset.model_name

        for i in range(len(self.prompts)):
            self.generated_data.append({
                "prompt": self.prompts[i],
                "output": completions[i],
                "model_name": model_name,
                "temperature": self.config.dataset.temperature,
                "top_p": self.config.dataset.top_p,
                "generation_timestamp": datetime.now().isoformat()
            })

        # Save dataset
        self.save_dataset()

        return self.generated_data

    def save_dataset(self):
        """Save the generated dataset to disk and optionally to HuggingFace."""
        print(f"Saving dataset...")

        # Save as JSON
        with open(self.output_path, "w") as f:
            json.dump(self.generated_data, f, indent=2)

        # Save as CSV for easier inspection
        pd.DataFrame(self.generated_data).to_csv(self.csv_path, index=False)

        # Log to wandb if configured
        if self.config.logging.use_wandb and wandb.run is not None:
            # Log dataset metadata
            wandb.log({
                "dataset": {
                    "samples": len(self.generated_data),
                    "model_name": self.config.dataset.original_model_name if hasattr(self.config.dataset, 'original_model_name') else self.config.dataset.model_name,
                    "dataset_id": self.dataset_id
                }
            })

            # Log sample dataset to wandb
            sample_table = wandb.Table(dataframe=pd.DataFrame(self.generated_data[:10]))
            wandb.log({"dataset_samples": sample_table})

        # Push to HuggingFace if configured
        if self.config.dataset.push_to_hub:
            print(f"Pushing dataset to HuggingFace Hub...")
            
            # Create the dataset
            hf_dataset = Dataset.from_pandas(pd.DataFrame(self.generated_data))
            
            # Push to hub
            repo_id = f"{self.config.dataset.hub_org}/{self.dataset_id}" if self.config.dataset.hub_org else self.dataset_id
            
            hf_dataset.push_to_hub(
                repo_id,
                private=self.config.dataset.private,
                token=self.config.dataset.hub_token
            )
            
            print(f"Dataset pushed to HuggingFace: {repo_id}")

    def load_dataset(self):
        """Load dataset from disk."""
        if not os.path.exists(self.output_path):
            print("Cached dataset not found, generating new one...")
            return self.create_dataset()

        print(f"Loading dataset from {self.output_path}...")

        with open(self.output_path, "r") as f:
            self.generated_data = json.load(f)

        # Extract prompts for reference
        self.prompts = [item["prompt"] for item in self.generated_data]

        print(f"Loaded {len(self.generated_data)} samples")

        return self.generated_data

    def analyze_dataset(self):
        """Analyze the generated dataset."""
        if not self.generated_data:
            print("No dataset loaded. Loading from disk...")
            self.load_dataset()

        print("\n--- Dataset Analysis ---")
        print(f"Total samples: {len(self.generated_data)}")

        # Check for empty generations
        empty_count = sum(1 for item in self.generated_data if not item['output'])
        empty_pct = empty_count/len(self.generated_data)*100

        print(f"\nEmpty generations: {empty_count} ({empty_pct:.2f}%)")

        # Length statistics
        lengths = [len(item['output']) for item in self.generated_data]

        print(f"\nOutput length statistics:")
        print(f"  Average: {np.mean(lengths):.2f}")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  Std Dev: {np.std(lengths):.2f}")

        # Create analysis report
        report = {
            "total_samples": len(self.generated_data),
            "empty_count": empty_count,
            "empty_pct": float(empty_pct),
            "length_avg": float(np.mean(lengths)),
            "length_min": min(lengths),
            "length_max": max(lengths),
            "length_std": float(np.std(lengths))
        }

        # Save the analysis
        analysis_path = os.path.join(self.config.dataset.cache_dir, f"{self.dataset_id}_analysis.json")
        with open(analysis_path, "w") as f:
            json.dump(report, f, indent=2)

        # Log to wandb if configured
        if self.config.logging.use_wandb and wandb.run is not None:
            wandb.log({
                "dataset_analysis": report,
                "output_lengths_hist": wandb.Histogram(lengths)
            })

        print("\nAnalysis complete!")
        return report


def generate_dataset(config: DictConfig) -> str:
    """Generate a dataset using the specified configuration."""
    model_name = config.dataset.original_model_name if hasattr(config.dataset, 'original_model_name') else config.dataset.model_name
    print(f"Generating dataset for model: {model_name}")
    
    # Initialize wandb if enabled
    if config.logging.use_wandb:
        model_name_safe = model_name.split('/')[-1]
        run_name = f"dataset_gen_{model_name_safe}_{config.dataset.num_samples}"
        wandb.init(
            project=config.logging.project_name,
            name=run_name,
            config=OmegaConf.to_container(config, resolve=True),
            mode=config.logging.wandb_mode
        )
    
    # Generate dataset
    generator = DatasetGenerator(config)
    data = generator.create_dataset()
    
    # Analyze dataset
    analysis = generator.analyze_dataset()
    
    # Finish wandb run
    if config.logging.use_wandb and wandb.run is not None:
        wandb.finish()
    
    # Return the dataset ID for reference
    return generator.dataset_id