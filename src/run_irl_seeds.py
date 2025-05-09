#!/usr/bin/env python3
"""
Script to run IRL training over multiple seeds.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Add the current directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_irl_with_seeds(seeds, base_command=None):
    """
    Run IRL training with multiple seeds.
    
    Args:
        seeds: List of seeds to run
        base_command: Base command to run (if None, use default)
    """
    if base_command is None:
        base_command = [
            "python", "src/run_irl.py",
            "mode=train",
            "irl.dataset.original_dataset_path=ajagota71/EleutherAI_pythia-70M_2000_samples_original",
            "irl.dataset.detoxified_dataset_path=ajagota71/ajagota71_pythia-70m-detox-epoch-100_2000_samples_detoxified",
            "irl.model.reward_model_base=EleutherAI/pythia-70M",
            "irl.training.irl_method=max_margin",
            "irl.training.learning_rate=1e-5",
            "irl.training.epochs=10",
            "irl.training.batch_size=8",
            "irl.model.num_unfrozen_layers=0",
            "irl.output.push_to_hub=false",
            "irl.output.hub_org=ajagota71"
        ]
    
    # Get timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create a log directory
    log_dir = os.path.join(os.getcwd(), "seed_logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting IRL training with {len(seeds)} seeds: {seeds}")
    print(f"Logs will be saved to: {log_dir}")
    
    # Run for each seed
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running with seed {seed}")
        
        # Create command with seed
        cmd = base_command.copy()
        
        # Add seed to command
        cmd.append(f"irl.training.seed={seed}")
        
        # Add seed to repo name prefix
        repo_name_prefix = f"toxicity-reward-model-max-margin-seed-{seed}"
        cmd.append(f"irl.output.repo_name_prefix={repo_name_prefix}")
        
        # Create log file
        log_file = os.path.join(log_dir, f"seed_{seed}.log")
        
        # Print command
        print(f"Running command: {' '.join(cmd)}")
        print(f"Logging to: {log_file}")
        
        # Run command and capture output
        start_time = time.time()
        
        with open(log_file, 'w') as f:
            # Write command to log
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.flush()
            
            # Run process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)
                f.flush()
            
            # Wait for process to complete
            process.wait()
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Log completion
        with open(log_file, 'a') as f:
            f.write(f"\nProcess completed with return code: {process.returncode}\n")
            f.write(f"Runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)\n")
        
        print(f"Completed seed {seed} in {runtime/60:.2f} minutes with return code {process.returncode}")
    
    print(f"\nAll seeds completed. Logs saved to {log_dir}")


if __name__ == "__main__":
    # Define seeds to run
    seeds = [42, 100, 200, 300, 400]
    
    # You can customize the command here if needed
    # base_command = ["python", "src/run_irl.py", ...]
    
    # Run with default command
    run_irl_with_seeds(seeds) 