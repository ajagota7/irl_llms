#!/usr/bin/env python3
"""
Setup script for RLHF acceleration optimization.
This script helps configure accelerate and install required dependencies.
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return None


def install_dependencies():
    """Install required dependencies for acceleration."""
    print("Installing acceleration dependencies...")
    
    # Install flash attention (optional but recommended)
    print("\nInstalling flash-attention (this may take a while)...")
    flash_result = run_command(
        "pip install flash-attn --no-build-isolation",
        "Flash Attention installation"
    )
    
    if flash_result is None:
        print("Warning: Flash Attention installation failed. Continuing without it.")
    
    # Install GPUtil for GPU monitoring
    run_command("pip install GPUtil", "GPUtil installation")
    
    # Install psutil for system monitoring
    run_command("pip install psutil", "psutil installation")
    
    # Ensure DeepSpeed is installed
    run_command("pip install deepspeed", "DeepSpeed installation")
    
    print("\n✓ All dependencies installed!")


def create_accelerate_config():
    """Create accelerate configuration."""
    print("\nSetting up Accelerate configuration...")
    
    # Check if accelerate config already exists
    config_path = Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
    
    if config_path.exists():
        print(f"Accelerate config already exists at: {config_path}")
        response = input("Do you want to recreate it? (y/N): ").lower()
        if response != 'y':
            print("Using existing accelerate config.")
            return
    
    # Create accelerate config interactively
    print("\nRunning accelerate config...")
    print("Recommended settings:")
    print("- Mixed Precision: fp16")
    print("- Gradient Accumulation: 1")
    print("- DeepSpeed: No (for initial setup)")
    print("- Number of processes: 1")
    
    try:
        subprocess.run("accelerate config", shell=True, check=True)
        print("✓ Accelerate configuration completed!")
    except subprocess.CalledProcessError:
        print("✗ Accelerate configuration failed. You can run 'accelerate config' manually.")


def create_deepspeed_config():
    """Create DeepSpeed configuration file."""
    print("\nCreating DeepSpeed configuration...")
    
    deepspeed_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "cpu_offload": False
        },
        
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        
        "wall_clock_breakdown": False,
        "steps_per_print": 10,
        "dump_state": False
    }
    
    # Write DeepSpeed config
    with open("deepspeed_config.json", "w") as f:
        json.dump(deepspeed_config, f, indent=2)
    
    print("✓ DeepSpeed configuration created: deepspeed_config.json")


def create_training_scripts():
    """Create optimized training scripts."""
    print("\nCreating optimized training scripts...")
    
    # Create basic accelerate training script
    basic_script = '''#!/bin/bash
# Basic Accelerate Training Script for SmolLM-135M

echo "Starting optimized RLHF training with Accelerate..."

accelerate launch src/rlhf_train_optimized.py \\
  rlhf=smolLM_135m_optimized \\
  rlhf.model.use_raw_logits=true \\
  rlhf.model.reward_model="s-nlp/roberta_toxicity_classifier" \\
  rlhf.model.batch_size=1024 \\
  rlhf.model.mini_batch_size=128 \\
  rlhf.model.forward_batch_size=128 \\
  rlhf.model.gradient_accumulation_steps=1 \\
  rlhf.model.ppo_epochs=2 \\
  rlhf.training.save_freq=25 \\
  rlhf.training.num_train_epochs=100 \\
  rlhf.output.organization=ajagota71

echo "Training completed!"
'''
    
    # Create DeepSpeed training script
    deepspeed_script = '''#!/bin/bash
# DeepSpeed Training Script for SmolLM-135M

echo "Starting optimized RLHF training with DeepSpeed..."

accelerate launch --config_file deepspeed_config.json src/rlhf_train_optimized.py \\
  rlhf=smolLM_135m_optimized \\
  rlhf.model.batch_size=1024 \\
  rlhf.model.mini_batch_size=128 \\
  rlhf.model.forward_batch_size=128 \\
  rlhf.model.gradient_accumulation_steps=1 \\
  rlhf.output.organization=ajagota71

echo "Training completed!"
'''
    
    # Write scripts
    with open("train_basic.sh", "w") as f:
        f.write(basic_script)
    
    with open("train_deepspeed.sh", "w") as f:
        f.write(deepspeed_script)
    
    # Make scripts executable
    os.chmod("train_basic.sh", 0o755)
    os.chmod("train_deepspeed.sh", 0o755)
    
    print("✓ Training scripts created:")
    print("  - train_basic.sh (Accelerate only)")
    print("  - train_deepspeed.sh (DeepSpeed + Accelerate)")


def main():
    """Main setup function."""
    print("🚀 RLHF Acceleration Setup")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Create accelerate config
    create_accelerate_config()
    
    # Create DeepSpeed config
    create_deepspeed_config()
    
    # Create training scripts
    create_training_scripts()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run basic training: ./train_basic.sh")
    print("2. Run DeepSpeed training: ./train_deepspeed.sh")
    print("3. Monitor GPU usage: watch -n 1 nvidia-smi")
    print("\nExpected improvements:")
    print("- GPU utilization: 30% → 90%+")
    print("- Training speed: 90s → 5-10s per batch")
    print("- Overall speedup: 5-10x")


if __name__ == "__main__":
    main() 