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

# Add the current directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .dataset_generator import generate_dataset, DatasetGenerator
from .irl_train import train_irl


@hydra.main(config_path="src/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for IRL detoxification."""
    # Get the IRL config
    irl_cfg = cfg.irl if hasattr(cfg, 'irl') else cfg
    
    print(f"Running IRL detoxification with mode: {irl_cfg.mode}")
    print(f"Using configuration for model: {irl_cfg.model.reward_model_base}")
    
    # Save the full config for reference
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(irl_cfg.output.base_dir, "configs"), exist_ok=True)
    with open(os.path.join(irl_cfg.output.base_dir, "configs", f"config_{timestamp}.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(irl_cfg))
    
    if irl_cfg.mode == "generate_dataset" or irl_cfg.mode == "all":
        # Generate original dataset
        print(f"Generating dataset for original model: {irl_cfg.dataset.original_model_name}")
        
        # Configure for original model dataset generation
        dataset_cfg = OmegaConf.create(OmegaConf.to_container(irl_cfg, resolve=True))
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
        if irl_cfg.dataset.original_model_name != irl_cfg.dataset.detoxified_model_name:
            print(f"Generating dataset for detoxified model: {irl_cfg.dataset.detoxified_model_name}")
            
            # Configure for detoxified model dataset generation
            dataset_cfg = OmegaConf.create(OmegaConf.to_container(irl_cfg, resolve=True))
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
            print("Using same model for original and detoxified datasets with different temperature")
            irl_cfg.dataset.detoxified_dataset_path = irl_cfg.dataset.original_dataset_path
    
    if irl_cfg.mode == "train" or irl_cfg.mode == "all":
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