#!/usr/bin/env python3
"""
Test script to verify checkpoint configuration is working correctly.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

@hydra.main(config_path="src/configs", config_name="config", version_base=None)
def test_checkpoint_config(cfg: DictConfig) -> None:
    """Test that checkpoint configuration is properly loaded."""
    
    # Get the IRL config
    config = cfg.irl if hasattr(cfg, 'irl') else cfg
    
    print("Testing checkpoint configuration...")
    print(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Check if checkpoint settings are available
    output_config = config.output
    
    print(f"\nCheckpoint settings:")
    print(f"  save_checkpoints: {output_config.save_checkpoints}")
    print(f"  checkpoint_interval: {getattr(output_config, 'checkpoint_interval', 'NOT SET')}")
    print(f"  push_checkpoints_to_hub: {getattr(output_config, 'push_checkpoints_to_hub', 'NOT SET')}")
    print(f"  push_to_hub: {output_config.push_to_hub}")
    
    # Test the logic that would be used in training
    checkpoint_interval = getattr(output_config, 'checkpoint_interval', 5)
    push_checkpoints_to_hub = getattr(output_config, 'push_checkpoints_to_hub', True)
    
    print(f"\nComputed values:")
    print(f"  checkpoint_interval: {checkpoint_interval}")
    print(f"  push_checkpoints_to_hub: {push_checkpoints_to_hub}")
    
    # Test which epochs would save checkpoints
    epochs = list(range(21))  # 0 to 20
    checkpoint_epochs = [epoch for epoch in epochs if epoch % checkpoint_interval == 0]
    
    print(f"\nCheckpoints would be saved at epochs: {checkpoint_epochs}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_checkpoint_config() 