#!/usr/bin/env python
"""
Multi-device training script for running concurrent IRL training jobs.
"""

import os
import sys
import subprocess
import time
import json
from typing import List, Dict
import argparse
from pathlib import Path

def get_device_configs():
    """Get configurations for different devices and experiments."""
    configs = []
    
    # Device configurations
    devices = [
        {"device_id": 0, "name": "cuda:0"},
        {"device_id": 1, "name": "cuda:1"}
    ]
    
    # Experiment configurations
    experiments = [
        {
            "name": "pythia-70m-prompt-output-tox0.3",
            "config_overrides": {
                "model.reward_model_base": "EleutherAI/pythia-70m",
                "training.include_prompt": True,
                "dataset.original_dataset_path": "ajagota71/EleutherAI_pythia-70M_2000_samples_original",
                "dataset.detoxified_dataset_path": "ajagota71/ajagota71_pythia-70m-detox-epoch-100_2000_samples_detoxified",
                "training.irl_method": "max_margin",
                "training.epochs": 30,
                "training.batch_size": 16,
                "training.gradient_accumulation_steps": 2,
                "training.use_amp": True,
                "training.use_gradient_checkpointing": False
            }
        },
        {
            "name": "pythia-70m-output-only-tox0.3",
            "config_overrides": {
                "model.reward_model_base": "EleutherAI/pythia-70m",
                "training.include_prompt": False,
                "dataset.original_dataset_path": "ajagota71/EleutherAI_pythia-70M_2000_samples_original",
                "dataset.detoxified_dataset_path": "ajagota71/ajagota71_pythia-70m-detox-epoch-100_2000_samples_detoxified",
                "training.irl_method": "max_margin",
                "training.epochs": 30,
                "training.batch_size": 16,
                "training.gradient_accumulation_steps": 2,
                "training.use_amp": True,
                "training.use_gradient_checkpointing": False
            }
        },
        {
            "name": "pythia-70m-prompt-output-tox0.8",
            "config_overrides": {
                "model.reward_model_base": "EleutherAI/pythia-70m",
                "training.include_prompt": True,
                "dataset.original_dataset_path": "ajagota71/EleutherAI_pythia-70m_2000_samples_temp0p7_tox0p8_original",
                "dataset.detoxified_dataset_path": "ajagota71/ajagota71_pythia-70m-s-nlp-detox-checkpoint-epoch-100_2000_samples_temp0p7_tox0p8_detoxified",
                "training.irl_method": "max_margin",
                "training.epochs": 30,
                "training.batch_size": 16,
                "training.gradient_accumulation_steps": 2,
                "training.use_amp": True,
                "training.use_gradient_checkpointing": False
            }
        },
        {
            "name": "pythia-70m-output-only-tox0.8",
            "config_overrides": {
                "model.reward_model_base": "EleutherAI/pythia-70m",
                "training.include_prompt": False,
                "dataset.original_dataset_path": "ajagota71/EleutherAI_pythia-70m_2000_samples_temp0p7_tox0p8_original",
                "dataset.detoxified_dataset_path": "ajagota71/ajagota71_pythia-70m-s-nlp-detox-checkpoint-epoch-100_2000_samples_temp0p7_tox0p8_detoxified",
                "training.irl_method": "max_margin",
                "training.epochs": 30,
                "training.batch_size": 16,
                "training.gradient_accumulation_steps": 2,
                "training.use_amp": True,
                "training.use_gradient_checkpointing": False
            }
        }
    ]
    
    # Assign experiments to devices
    for i, experiment in enumerate(experiments):
        device = devices[i % len(devices)]
        configs.append({
            "experiment": experiment,
            "device": device
        })
    
    return configs

def create_config_file(config_overrides: Dict, output_path: str):
    """Create a temporary config file with overrides."""
    config_content = {
        "model": {
            "reward_model_base": config_overrides.get("model.reward_model_base", "EleutherAI/pythia-70m"),
            "use_half_precision": True,
            "num_unfrozen_layers": 1
        },
        "training": {
            "irl_method": config_overrides.get("training.irl_method", "max_margin"),
            "epochs": config_overrides.get("training.epochs", 30),
            "batch_size": config_overrides.get("training.batch_size", 16),
            "gradient_accumulation_steps": config_overrides.get("training.gradient_accumulation_steps", 2),
            "use_amp": config_overrides.get("training.use_amp", True),
            "use_gradient_checkpointing": config_overrides.get("training.use_gradient_checkpointing", False),
            "include_prompt": config_overrides.get("training.include_prompt", False),
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "adam_epsilon": 1e-8,
            "max_length": 512,
            "margin": 0.1,
            "temperature": 0.1,
            "save_every": 5,
            "train_test_split": 0.8,
            "seed": 42
        },
        "dataset": {
            "original_dataset_path": config_overrides.get("dataset.original_dataset_path"),
            "detoxified_dataset_path": config_overrides.get("dataset.detoxified_dataset_path"),
            "original_model_name": "EleutherAI/pythia-70m",
            "detoxified_model_name": "EleutherAI/pythia-70m"
        },
        "evaluation": {
            "true_reward_model": "unitary/toxic-bert"
        },
        "output": {
            "base_dir": "outputs",
            "push_to_hub": True,
            "private": False
        },
        "logging": {
            "use_wandb": False
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_content, f, indent=2)

def run_training_job(experiment_name: str, device: Dict, config_overrides: Dict, log_dir: str):
    """Run a single training job on a specific device."""
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create temporary config file
    config_path = os.path.join(log_dir, f"{experiment_name}_config.json")
    create_config_file(config_overrides, config_path)
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device["device_id"])
    
    # Create log file
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    # Build command
    cmd = [
        sys.executable, "src/irl_train_optimized.py",
        "--config-path", "src/configs",
        "--config-name", "config",
        "hydra.run.dir=outputs/multi_device_runs",
        f"hydra.sweep.dir=outputs/multi_device_runs/{experiment_name}",
        f"hydra.sweep.subdir=.",
        f"output.base_dir=outputs/multi_device_runs/{experiment_name}",
        f"model.reward_model_base={config_overrides['model.reward_model_base']}",
        f"training.include_prompt={config_overrides['training.include_prompt']}",
        f"training.irl_method={config_overrides['training.irl_method']}",
        f"training.epochs={config_overrides['training.epochs']}",
        f"training.batch_size={config_overrides['training.batch_size']}",
        f"training.gradient_accumulation_steps={config_overrides['training.gradient_accumulation_steps']}",
        f"training.use_amp={config_overrides['training.use_amp']}",
        f"training.use_gradient_checkpointing={config_overrides['training.use_gradient_checkpointing']}",
        f"dataset.original_dataset_path={config_overrides['dataset.original_dataset_path']}",
        f"dataset.detoxified_dataset_path={config_overrides['dataset.detoxified_dataset_path']}"
    ]
    
    print(f"Starting {experiment_name} on {device['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    
    # Run the command
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    return process, log_file

def main():
    parser = argparse.ArgumentParser(description="Run multiple IRL training jobs concurrently")
    parser.add_argument("--max_concurrent", type=int, default=2,
                        help="Maximum number of concurrent jobs")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for log files")
    parser.add_argument("--experiments", type=str, nargs="+",
                        help="Specific experiments to run (default: all)")
    
    args = parser.parse_args()
    
    # Get device configurations
    configs = get_device_configs()
    
    # Filter experiments if specified
    if args.experiments:
        configs = [c for c in configs if c["experiment"]["name"] in args.experiments]
    
    print(f"Found {len(configs)} experiments to run")
    print(f"Maximum concurrent jobs: {args.max_concurrent}")
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Track running processes
    running_processes = []
    completed_processes = []
    
    # Start initial batch of jobs
    for i in range(min(args.max_concurrent, len(configs))):
        config = configs[i]
        experiment = config["experiment"]
        device = config["device"]
        
        process, log_file = run_training_job(
            experiment["name"],
            device,
            experiment["config_overrides"],
            args.log_dir
        )
        
        running_processes.append({
            "process": process,
            "config": config,
            "log_file": log_file,
            "start_time": time.time()
        })
    
    # Monitor and manage jobs
    config_index = args.max_concurrent
    
    while running_processes:
        # Check for completed processes
        for i, job in enumerate(running_processes):
            if job["process"].poll() is not None:
                # Process completed
                return_code = job["process"].returncode
                experiment_name = job["config"]["experiment"]["name"]
                device_name = job["config"]["device"]["name"]
                
                if return_code == 0:
                    print(f"✅ {experiment_name} on {device_name} completed successfully")
                else:
                    print(f"❌ {experiment_name} on {device_name} failed with return code {return_code}")
                
                completed_processes.append(job)
                running_processes.pop(i)
                
                # Start next job if available
                if config_index < len(configs):
                    config = configs[config_index]
                    experiment = config["experiment"]
                    device = config["device"]
                    
                    process, log_file = run_training_job(
                        experiment["name"],
                        device,
                        experiment["config_overrides"],
                        args.log_dir
                    )
                    
                    running_processes.append({
                        "process": process,
                        "config": config,
                        "log_file": log_file,
                        "start_time": time.time()
                    })
                    
                    config_index += 1
                
                break
        
        # Wait a bit before checking again
        time.sleep(10)
    
    print(f"All {len(completed_processes)} jobs completed!")
    
    # Print summary
    successful = sum(1 for job in completed_processes if job["process"].returncode == 0)
    failed = len(completed_processes) - successful
    
    print(f"Summary: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main() 