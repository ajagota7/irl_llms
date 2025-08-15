#!/usr/bin/env python3
"""
Test script for ultra-conservative SmolLM-1.7B configuration.
"""

import subprocess
import sys

def test_ultra_conservative():
    """Test the ultra-conservative configuration with a small number of epochs."""
    
    print("Testing Ultra-Conservative SmolLM-1.7B Configuration")
    print("=" * 60)
    
    # Test command with minimal epochs
    command = [
        "python", "src/rlhf_train.py",
        "rlhf=smolLM_1p7b_ultra_conservative",
        "model.use_raw_logits=true",
        "model.reward_model=s-nlp/roberta_toxicity_classifier",
        "training.save_freq=5",
        "training.num_train_epochs=5",  # Only 5 epochs for testing
        "output.push_to_hub=false",  # Don't push during test
        "wandb.mode=disabled"  # Disable wandb for testing
    ]
    
    print(f"Running: {' '.join(command)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print("\n✓ Ultra-conservative configuration test PASSED!")
        print("You can now run the full training with this configuration.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Ultra-conservative configuration test FAILED!")
        print(f"Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return False

def main():
    """Main function."""
    print("This script will test the ultra-conservative configuration")
    print("with only 5 epochs to verify it works without OOM errors.")
    print()
    
    confirm = input("Proceed with test? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        success = test_ultra_conservative()
        if success:
            print("\nSUCCESS: Ultra-conservative configuration works!")
            print("You can now run the full training with:")
            print("python src/rlhf_train.py rlhf=smolLM_1p7b_ultra_conservative [your_args]")
        else:
            print("\nFAILED: Ultra-conservative configuration still has issues.")
            print("Consider:")
            print("1. Restarting the Colab runtime")
            print("2. Running the memory diagnostic script")
            print("3. Using an even smaller model")
    else:
        print("Test cancelled.")

if __name__ == "__main__":
    main() 