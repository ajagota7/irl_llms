#!/usr/bin/env python3
"""
Wrapper script to run Gemma3 RLHF training with TorchDynamo completely disabled.
"""

import os
import sys
import subprocess

def main():
    """Set environment variables and run the RLHF training script."""
    
    # Set environment variables to disable TorchDynamo completely
    env = os.environ.copy()
    env.update({
        'TORCHDYNAMO_DISABLE': '1',
        'TORCH_COMPILE_DISABLE': '1',
        'PYTORCH_DISABLE_TORCH_COMPILE': '1',
        'TORCH_LOGS': 'off',
        'TORCHDYNAMO_VERBOSE': '0'
    })
    
    # Get the command line arguments (skip the script name)
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Default to conservative configuration if no args provided
    if not args:
        args = ['rlhf=gemma_270m_conservative', 'model.use_raw_logits=true', 
                'model.reward_model=s-nlp/roberta_toxicity_classifier', 
                'output.push_to_hub=true', 'output.organization=ajagota71']
    
    # Construct the command
    cmd = [sys.executable, 'src/rlhf_train.py'] + args
    
    print("Running Gemma3 RLHF training with TorchDynamo disabled...")
    print(f"Command: {' '.join(cmd)}")
    print("Environment variables set:")
    for key, value in env.items():
        if key.startswith('TORCH'):
            print(f"  {key}={value}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print("Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return 1

if __name__ == "__main__":
    exit(main()) 