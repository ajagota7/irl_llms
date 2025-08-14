#!/usr/bin/env python3
"""
Test script to verify configuration loading.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(config_path="src/configs", config_name="config", version_base=None)
def test_config(cfg: DictConfig) -> None:
    """Test configuration loading."""
    
    print("Configuration loaded successfully!")
    print(f"Configuration keys: {list(cfg.keys())}")
    
    if hasattr(cfg, 'model'):
        print(f"Model name: {cfg.model.name}")
        print(f"Model learning rate: {cfg.model.learning_rate}")
    else:
        print("No model configuration found!")
    
    if hasattr(cfg, 'dataset'):
        print(f"Dataset name: {cfg.dataset.name}")
    else:
        print("No dataset configuration found!")
    
    if hasattr(cfg, 'training'):
        print(f"Training epochs: {cfg.training.num_train_epochs}")
    else:
        print("No training configuration found!")
    
    print("\nFull configuration:")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    test_config() 