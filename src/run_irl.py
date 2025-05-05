#!/usr/bin/env python3
"""
Main entry point for IRL detoxification.
"""

import os
import sys
import hydra
import datetime
from omegaconf import DictConfig, OmegaConf
import wandb
import json

# Add the current directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_generator import generate_dataset, DatasetGenerator
from irl_train import train_irl


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for IRL detoxification."""
    # Get the IRL config
    irl_cfg = cfg.irl if hasattr(cfg, 'irl') else cfg
    
    # Get the mode from the top-level config
    mode = cfg.mode if hasattr(cfg, 'mode') else "train"
    
    print(f"Running IRL detoxification with mode: {mode}")
    print(f"Using configuration for model: {irl_cfg.model.reward_model_base}")
    
    # Update the mode in the irl config
    if hasattr(irl_cfg, 'mode'):
        irl_cfg.mode = mode
    else:
        OmegaConf.update(irl_cfg, "mode", mode)
    
    # Save the full config for reference
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(irl_cfg.output.base_dir, "configs"), exist_ok=True)
    with open(os.path.join(irl_cfg.output.base_dir, "configs", f"config_{timestamp}.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(irl_cfg))
    
    # For "generate_dataset" or "all" mode, generate both datasets if both models are specified
    if irl_cfg.mode in ["generate_dataset", "all"]:
        print("Generating datasets...")
        
        # Check if both models are specified
        has_both_models = (irl_cfg.dataset.original_model_name is not None and 
                          irl_cfg.dataset.detoxified_model_name is not None and
                          irl_cfg.dataset.original_model_name != irl_cfg.dataset.detoxified_model_name)
        
        # Load prompts once to ensure both datasets use the same prompts
        base_cfg = OmegaConf.create(OmegaConf.to_container(irl_cfg, resolve=True))
        base_cfg.mode = "generate_dataset"
        # Set a default model_name for the base generator
        base_cfg.dataset.model_name = base_cfg.dataset.original_model_name
        base_generator = DatasetGenerator(base_cfg)
        prompts = base_generator.load_prompts()
        
        # Generate original dataset
        print(f"Generating dataset for original model: {irl_cfg.dataset.original_model_name}")
        
        # Configure for original model dataset generation
        dataset_cfg = OmegaConf.create(OmegaConf.to_container(irl_cfg, resolve=True))
        dataset_cfg.mode = "generate_dataset"
        dataset_cfg.dataset.model_name = dataset_cfg.dataset.original_model_name
        
        # Set up wandb if enabled
        if dataset_cfg.logging.use_wandb:
            model_name = dataset_cfg.dataset.original_model_name.split('/')[-1]
            run_name = f"dataset_gen_original_{model_name}_{timestamp}"
            wandb.init(
                project=dataset_cfg.logging.project_name,
                name=run_name,
                config=OmegaConf.to_container(dataset_cfg, resolve=True),
                mode=dataset_cfg.logging.wandb_mode
            )
        
        # Generate original dataset
        generator = DatasetGenerator(dataset_cfg)
        generator.prompts = prompts  # Use the same prompts
        original_data = generator.create_dataset()
        original_dataset_path = generator.output_path
        
        # Update the config with the actual path
        irl_cfg.dataset.original_dataset_path = original_dataset_path
        
        # Analyze dataset
        generator.analyze_dataset()
        
        # Finish wandb run if active
        if dataset_cfg.logging.use_wandb and wandb.run is not None:
            wandb.finish()
        
        # Generate detoxified dataset if needed
        if has_both_models:
            print(f"Generating dataset for detoxified model: {irl_cfg.dataset.detoxified_model_name}")
            
            # Configure for detoxified model dataset generation
            dataset_cfg = OmegaConf.create(OmegaConf.to_container(irl_cfg, resolve=True))
            dataset_cfg.mode = "generate_dataset"
            dataset_cfg.dataset.model_name = dataset_cfg.dataset.detoxified_model_name
            
            # Set up wandb if enabled
            if dataset_cfg.logging.use_wandb:
                model_name = dataset_cfg.dataset.detoxified_model_name.split('/')[-1]
                run_name = f"dataset_gen_detoxified_{model_name}_{timestamp}"
                wandb.init(
                    project=dataset_cfg.logging.project_name,
                    name=run_name,
                    config=OmegaConf.to_container(dataset_cfg, resolve=True),
                    mode=dataset_cfg.logging.wandb_mode
                )
            
            # Generate detoxified dataset
            generator = DatasetGenerator(dataset_cfg)
            generator.prompts = prompts  # Use the same prompts
            detoxified_data = generator.create_dataset()
            detoxified_dataset_path = generator.output_path
            
            # Update the config with the actual path
            irl_cfg.dataset.detoxified_dataset_path = detoxified_dataset_path
            
            # Analyze dataset
            generator.analyze_dataset()
            
            # Finish wandb run if active
            if dataset_cfg.logging.use_wandb and wandb.run is not None:
                wandb.finish()
        else:
            # If using the same model, just use a different temperature
            print("Using same model for original and detoxified datasets")
            irl_cfg.dataset.detoxified_dataset_path = irl_cfg.dataset.original_dataset_path
        
        # Compare datasets if both were generated
        if has_both_models:
            print("\nComparing original and detoxified datasets:")
            with open(irl_cfg.dataset.original_dataset_path, 'r') as f:
                original_data = json.load(f)
            with open(irl_cfg.dataset.detoxified_dataset_path, 'r') as f:
                detoxified_data = json.load(f)
            
            # Check if they're identical
            identical_count = sum(1 for i in range(min(len(original_data), len(detoxified_data))) 
                                if original_data[i]['output'] == detoxified_data[i]['output'])
            
            print(f"Total samples: {len(original_data)}")
            print(f"Identical outputs: {identical_count} ({identical_count/len(original_data)*100:.2f}%)")
            
            # Sample comparison
            print("\nSample comparison (first 3 examples):")
            for i in range(min(3, len(original_data))):
                print(f"\nPrompt: {original_data[i]['prompt'][:100]}...")
                print(f"Original: {original_data[i]['output'][:100]}...")
                print(f"Detoxified: {detoxified_data[i]['output'][:100]}...")
        
        # If we're in "all" mode, continue to training
        if irl_cfg.mode == "all":
            print("Dataset generation complete. Moving to training phase...")
            irl_cfg.mode = "train"
            
            # Set up wandb for training
            if irl_cfg.logging.use_wandb:
                original_model = irl_cfg.dataset.original_model_name.split('/')[-1]
                detoxified_model = irl_cfg.dataset.detoxified_model_name.split('/')[-1]
                run_name = f"irl_{irl_cfg.training.irl_method}_{original_model}_to_{detoxified_model}_{timestamp}"
                
                wandb.init(
                    project=irl_cfg.logging.project_name,
                    name=run_name,
                    config=OmegaConf.to_container(irl_cfg, resolve=True),
                    mode=irl_cfg.logging.wandb_mode
                )
            
            # Train the model
            results = train_irl(irl_cfg)
            
            # Finish wandb run if active
            if irl_cfg.logging.use_wandb and wandb.run is not None:
                wandb.finish()
            
            print(f"All mode complete with results: {results}")
        
        return
    
    # Handle train mode
    if irl_cfg.mode == "train":
        # Train model
        print(f"Training reward model using {irl_cfg.training.irl_method} IRL...")
        
        # Check if dataset paths are set
        if not irl_cfg.dataset.original_dataset_path or not irl_cfg.dataset.detoxified_dataset_path:
            print("Error: Dataset paths are not set. Please generate datasets first or provide paths.")
            return
        
        # Set up wandb if enabled
        if irl_cfg.logging.use_wandb:
            original_model = irl_cfg.dataset.original_model_name.split('/')[-1]
            detoxified_model = irl_cfg.dataset.detoxified_model_name.split('/')[-1]
            run_name = f"irl_{irl_cfg.training.irl_method}_{original_model}_to_{detoxified_model}_{timestamp}"
            
            wandb.init(
                project=irl_cfg.logging.project_name,
                name=run_name,
                config=OmegaConf.to_container(irl_cfg, resolve=True),
                mode=irl_cfg.logging.wandb_mode
            )
        
        # Train the model
        results = train_irl(irl_cfg)
        
        # Finish wandb run if active
        if irl_cfg.logging.use_wandb and wandb.run is not None:
            wandb.finish()
        
        print(f"Training complete with results: {results}")


if __name__ == "__main__":
    main()