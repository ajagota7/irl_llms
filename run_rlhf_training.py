#!/usr/bin/env python3
"""
Simple script to run RLHF training with optimized configuration.
"""

import subprocess
import sys
import os

def main():
    """Run RLHF training with the specified configuration."""
    
    # Change to the src directory
    os.chdir("src")
    
    # Command to run RLHF training
    cmd = [
        "python", "rlhf_train.py",
        "--config-name=rlhf/smolLM_135m"
    ]
    
    print("Running RLHF training with optimized configuration...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main() 