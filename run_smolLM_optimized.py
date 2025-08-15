#!/usr/bin/env python3
"""
Script to run SmolLM-1.7B RLHF training with different memory optimization levels.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and handle errors."""
    print(f"Running: {command}")
    print("-" * 80)
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"Command completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False

def main():
    print("SmolLM-1.7B RLHF Training with Memory Optimization")
    print("=" * 60)
    print()
    print("Choose a configuration:")
    print("1. Standard optimized (recommended first try)")
    print("2. Conservative memory settings")
    print("3. Advanced memory optimization")
    print("4. Custom configuration")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    base_command = "python src/rlhf_train.py"
    common_args = [
        "model.use_raw_logits=true",
        "model.reward_model=s-nlp/roberta_toxicity_classifier",
        "training.save_freq=50",
        "training.num_train_epochs=100",
        "output.checkpoint_push_freq=20",
        "output.push_to_hub=true",
        "output.organization=ajagota71"
    ]
    
    if choice == "1":
        # Standard optimized
        config = "rlhf=smolLM_1p7b"
        print("\nUsing standard optimized configuration...")
        
    elif choice == "2":
        # Conservative
        config = "rlhf=smolLM_1p7b_conservative"
        print("\nUsing conservative memory settings...")
        
    elif choice == "3":
        # Advanced memory optimization
        config = "rlhf=smolLM_1p7b_memory_optimized"
        print("\nUsing advanced memory optimization...")
        
    elif choice == "4":
        # Custom
        print("\nCustom configuration options:")
        print("Available configs:")
        print("- smolLM_1p7b (standard optimized)")
        print("- smolLM_1p7b_conservative (very conservative)")
        print("- smolLM_1p7b_memory_optimized (advanced optimization)")
        
        config_name = input("Enter config name: ").strip()
        if not config_name:
            config_name = "smolLM_1p7b"
        config = f"rlhf={config_name}"
        
        # Allow custom batch size
        custom_batch = input("Custom batch size (press Enter to use default): ").strip()
        if custom_batch:
            try:
                batch_size = int(custom_batch)
                common_args.append(f"model.batch_size={batch_size}")
                print(f"Using custom batch size: {batch_size}")
            except ValueError:
                print("Invalid batch size, using default")
    else:
        print("Invalid choice. Using standard optimized configuration.")
        config = "rlhf=smolLM_1p7b"
    
    # Build the full command
    command_parts = [base_command, config] + common_args
    full_command = " ".join(command_parts)
    
    print(f"\nFull command:")
    print(f"{full_command}")
    print()
    
    # Ask for confirmation
    confirm = input("Proceed with training? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        success = run_command(full_command)
        if success:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining failed. Consider trying a more conservative configuration.")
    else:
        print("Training cancelled.")

if __name__ == "__main__":
    main() 