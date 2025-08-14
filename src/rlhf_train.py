"""
Main RLHF training script for detoxifying language models.
"""

import os
import time
import torch
import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from torch.optim import Adam
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from huggingface_hub import HfApi

from rlhf_utilities import (
    build_dataset,
    collator,
    setup_wandb,
    load_reward_model,
    evaluate_toxicity,
    analyze_prompt_tracking,
    LengthSampler,
    safe_reward_computation
)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train_rlhf(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Add current timestamp
    cfg.now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # OPTIMIZATION 5: Parameter Validation - Validate all parameters upfront
    print("Validating configuration parameters...")
    
    # Validate model configuration
    if cfg.model.name is None:
        raise ValueError("Model name must be specified in config")
    
    # Validate batch size configuration
    if cfg.model.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if cfg.model.mini_batch_size <= 0:
        raise ValueError("Mini batch size must be positive")
    if cfg.model.gradient_accumulation_steps <= 0:
        raise ValueError("Gradient accumulation steps must be positive")
    
    # Validate generation parameters
    if cfg.model.generation.output_min_length > cfg.model.generation.output_max_length:
        raise ValueError("output_min_length cannot be greater than output_max_length")
    
    # Validate dataset configuration
    if cfg.dataset.name is None:
        raise ValueError("Dataset name must be specified in config")
    if cfg.dataset.toxicity_threshold < 0 or cfg.dataset.toxicity_threshold > 1:
        raise ValueError("Toxicity threshold must be between 0 and 1")
    
    # Validate reward model configuration
    if cfg.model.reward_model is None:
        raise ValueError("Reward model must be specified in config")
    
    print("Configuration validation passed!")
    
    # Print configuration
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directories
    output_dir = os.path.join(os.getcwd(), f"outputs/{cfg.now}")
    os.makedirs(output_dir, exist_ok=True)
    eval_dir = os.path.join(output_dir, "evaluation")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # OPTIMIZATION 6: Async I/O Setup - Initialize async operations
    checkpoint_queue = queue.Queue()
    evaluation_queue = queue.Queue()
    
    def async_checkpoint_saver():
        """Background thread for saving checkpoints asynchronously."""
        while True:
            try:
                checkpoint_data = checkpoint_queue.get(timeout=1)
                if checkpoint_data is None:  # Shutdown signal
                    break
                
                epoch, ppo_trainer, reward_stats, output_dir = checkpoint_data
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}")
                
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(checkpoint_path)
                    
                    # Save reward stats
                    reward_df = pd.DataFrame(reward_stats)
                    reward_df.to_csv(os.path.join(output_dir, "reward_stats.csv"), index=False)
                
                print(f"Async checkpoint saved: {checkpoint_path}")
                checkpoint_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async checkpoint saving: {e}")
    
    # Start async checkpoint saver thread
    checkpoint_thread = threading.Thread(target=async_checkpoint_saver, daemon=True)
    checkpoint_thread.start()
    
    def async_evaluator():
        """Background thread for running evaluation asynchronously."""
        while True:
            try:
                eval_data = evaluation_queue.get(timeout=1)
                if eval_data is None:  # Shutdown signal
                    break
                
                epoch, model, ppo_trainer, tokenizer, reward_model, reward_tokenizer, test_dataset, config = eval_data
                
                print(f"\nRunning async evaluation at epoch {epoch}...")
                avg_toxicity, _ = evaluate_toxicity(
                    model=model,
                    ppo_trainer=ppo_trainer,
                    tokenizer=tokenizer,
                    reward_model=reward_model,
                    reward_tokenizer=reward_tokenizer,
                    dataset=test_dataset,
                    config=config,
                    epoch=epoch
                )
                
                print(f"Async evaluation epoch {epoch}: Average toxicity = {avg_toxicity:.4f}")
                
                # Save evaluation results
                with open(os.path.join(eval_dir, "evaluation_results.txt"), "a") as f:
                    f.write(f"Epoch {epoch}: Average toxicity = {avg_toxicity:.4f}\n")
                
                # Log evaluation metrics
                if wandb_run:
                    wandb_run.log({"eval/toxicity": avg_toxicity, "eval/epoch": epoch})
                
                evaluation_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async evaluation: {e}")
    
    # Start async evaluator thread
    evaluation_thread = threading.Thread(target=async_evaluator, daemon=True)
    evaluation_thread.start()
    
    # Set random seed
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)
    
    # Setup WandB logging
    wandb_run = setup_wandb(cfg)
    
    # Build dataset and tokenizer
    print("Building dataset...")
    
    # Use the model name from the rlhf config if the main model name is null
    if cfg.model.name is None and hasattr(cfg.rlhf, 'model') and cfg.rlhf.model.name is not None:
        cfg.model.name = cfg.rlhf.model.name
    
    # Use the dataset name from the rlhf config if the main dataset name is null
    if cfg.dataset.name is None and hasattr(cfg.rlhf, 'dataset') and cfg.rlhf.dataset.name is not None:
        cfg.dataset.name = cfg.rlhf.dataset.name
    
    # Use the reward model from the rlhf config if the main reward model is null
    if cfg.model.reward_model is None and hasattr(cfg.rlhf, 'model') and cfg.rlhf.model.reward_model is not None:
        cfg.model.reward_model = cfg.rlhf.model.reward_model
    
    train_dataset, test_dataset, tokenizer = build_dataset(cfg)
    print(f"Train set: {len(train_dataset)} examples")
    print(f"Test set: {len(test_dataset)} examples")
    
    # OPTIMIZATION 7: Optimized DataLoader Setup
    # Create optimized DataLoader with proper num_workers and prefetching
    num_workers = min(4, os.cpu_count() or 1)  # Use up to 4 workers
    print(f"Using {num_workers} data loading workers")
    
    # Create optimized data loader for better performance
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # PPO trainer handles batching
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collator
    )
    
    # Load model and add value head
    print(f"Loading model {cfg.model.name}...")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    
    # OPTIMIZATION 4: Memory Optimizations
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing for memory efficiency")
    
    # Enable mixed precision training if supported
    if torch.cuda.is_available():
        # Use bfloat16 if available (better numerical stability than fp16)
        if hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported():
            model = model.to(torch.bfloat16)
            print("Using bfloat16 mixed precision training")
        else:
            model = model.to(torch.float16)
            print("Using fp16 mixed precision training")
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    # Create reference model
    ref_model = create_reference_model(model)
    
    # Create optimizer
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.model.learning_rate
    )
    
    # Create learning rate scheduler
    total_steps = cfg.training.num_train_epochs * (len(train_dataset) // cfg.model.batch_size + 1)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Get PPO parameters from RLHF config if they exist
    ppo_params = {
        "model_name": cfg.model.name,
        "learning_rate": cfg.model.learning_rate,
        "log_with": "wandb" if wandb_run else None,
    }
    
    # OPTIMIZATION 3: Pre-calculated Batch Sizes - Remove runtime adjustments
    # Calculate optimal batch sizes based on available memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        # Conservative memory allocation: use 70% of available GPU memory
        available_memory = gpu_memory * 0.7
        
        # Estimate memory per sample (rough approximation)
        # This can be tuned based on model size and sequence length
        estimated_memory_per_sample = 0.1  # GB per sample (adjust based on model)
        
        # Calculate optimal batch size
        optimal_batch_size = max(32, min(256, int(available_memory / estimated_memory_per_sample)))
        
        # Ensure batch_size is a power of 2 for better GPU utilization
        optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))
        
        print(f"GPU Memory: {gpu_memory:.1f}GB, Available: {available_memory:.1f}GB")
        print(f"Calculated optimal batch size: {optimal_batch_size}")
        
        # Override config batch size with calculated optimal size
        batch_size = optimal_batch_size
        mini_batch_size = min(32, batch_size)  # Keep mini_batch_size reasonable
        gradient_accumulation_steps = max(1, batch_size // mini_batch_size)
    else:
        # CPU fallback - use smaller batch sizes
        batch_size = cfg.model.batch_size
        mini_batch_size = cfg.model.mini_batch_size
        gradient_accumulation_steps = cfg.model.gradient_accumulation_steps
    
    # Validate batch size configuration
    if batch_size % (mini_batch_size * gradient_accumulation_steps) != 0:
        # Adjust mini_batch_size to make it work
        mini_batch_size = batch_size // gradient_accumulation_steps
        print(f"Adjusted mini_batch_size to {mini_batch_size} for compatibility")
    
    # Set validated batch parameters
    ppo_params["batch_size"] = batch_size
    ppo_params["mini_batch_size"] = mini_batch_size
    ppo_params["gradient_accumulation_steps"] = gradient_accumulation_steps

    # Add PPO-specific parameters from RLHF config if available
    if hasattr(cfg.rlhf, 'model'):
        rlhf_model = cfg.rlhf.model
        if hasattr(rlhf_model, 'ppo_epochs'):
            ppo_params["ppo_epochs"] = rlhf_model.ppo_epochs
        if hasattr(rlhf_model, 'init_kl_coef'):
            ppo_params["init_kl_coef"] = rlhf_model.init_kl_coef
        if hasattr(rlhf_model, 'target'):
            ppo_params["target"] = rlhf_model.target
        if hasattr(rlhf_model, 'cliprange'):
            ppo_params["cliprange"] = rlhf_model.cliprange
        if hasattr(rlhf_model, 'cliprange_value'):
            ppo_params["cliprange_value"] = rlhf_model.cliprange_value
        if hasattr(rlhf_model, 'vf_coef'):
            ppo_params["vf_coef"] = rlhf_model.vf_coef
        if hasattr(rlhf_model, 'adap_kl_ctrl'):
            ppo_params["adap_kl_ctrl"] = rlhf_model.adap_kl_ctrl
        if hasattr(rlhf_model, 'use_score_norm'):
            ppo_params["use_score_norm"] = rlhf_model.use_score_norm
        if hasattr(rlhf_model, 'ratio_threshold'):
            ppo_params["ratio_threshold"] = rlhf_model.ratio_threshold
    
    # Create PPO config
    ppo_config = PPOConfig(**ppo_params)
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    
    # OPTIMIZATION 7: Use optimized DataLoader
    # Replace the default dataloader with our optimized version
    ppo_trainer.dataloader = train_dataloader
    
    # Load toxicity model
    print(f"Loading toxicity model {cfg.model.reward_model}...")
    reward_model, reward_tokenizer = load_reward_model(
        cfg.model.reward_model,
        ppo_trainer.accelerator.device
    )
    
    # Setup generation parameters
    output_length_sampler = LengthSampler(
        cfg.model.generation.output_min_length,
        cfg.model.generation.output_max_length
    )
    
    # Initial evaluation
    print("Performing initial evaluation...")
    initial_toxicity, _ = evaluate_toxicity(
        model=model,
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        dataset=test_dataset,
        config=cfg,
        epoch="initial"
    )
    
    print(f"Initial average toxicity: {initial_toxicity:.4f}")
    
    # Save evaluation results
    with open(os.path.join(eval_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"Epoch 0: Average toxicity = {initial_toxicity:.4f}\n")
    
    # Log initial metrics
    if wandb_run:
        wandb_run.log({"eval/initial_toxicity": initial_toxicity})
    
    # Create a dictionary to store reward stats across epochs
    reward_stats = {
        'epoch': [],
        'raw_rewards_mean': [],
        'raw_rewards_std': [],
        'nan_inf_count': [],
    }
    
    # Training loop
    print("Starting training loop...")
    training_start_time = time.time()
    
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=cfg.training.num_train_epochs):
        if epoch >= cfg.training.num_train_epochs:
            break
        
        # Process batch
        query_tensors = batch["input_ids"]
        
        # OPTIMIZATION 1: Batched Generation - Replace sequential loop with batch processing
        # Sample generation length for the entire batch (use average for efficiency)
        avg_gen_len = (cfg.model.generation.output_min_length + cfg.model.generation.output_max_length) // 2
        generation_kwargs = {
            "min_length": cfg.model.generation.min_length,
            "top_k": cfg.model.generation.top_k,
            "top_p": cfg.model.generation.top_p,
            "do_sample": cfg.model.generation.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": avg_gen_len
        }
        
        # Stack all queries into a single batch tensor
        # Ensure all queries have the same length by padding
        max_query_len = max(len(q) for q in query_tensors)
        padded_queries = []
        for query in query_tensors:
            query = query.squeeze()
            if len(query) < max_query_len:
                padding = torch.full((max_query_len - len(query),), 
                                    tokenizer.pad_token_id, 
                                    device=query.device)
                padded_query = torch.cat([query, padding], dim=0)
            else:
                padded_query = query[:max_query_len]
            padded_queries.append(padded_query)
        
        # Stack into batch tensor
        query_batch = torch.stack(padded_queries)
        
        # Generate responses for entire batch at once
        response_batch = ppo_trainer.generate(query_batch, **generation_kwargs)
        
        # Extract generated parts (last avg_gen_len tokens from each response)
        response_tensors = []
        for i, response in enumerate(response_batch):
            if response.size(0) >= avg_gen_len:
                response_tensors.append(response[-avg_gen_len:])
            else:
                # If response is shorter than expected, pad it
                padding = torch.full((avg_gen_len - response.size(0),), 
                                    tokenizer.pad_token_id, 
                                    device=response.device)
                padded_response = torch.cat([response, padding], dim=0)
                response_tensors.append(padded_response)
        
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # OPTIMIZATION 2: Optimized Reward Model Pipeline - Batch reward computation
        texts = batch["response"]
        
        # Pre-tokenize all texts at once for better efficiency
        toxicity_inputs = reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(ppo_trainer.accelerator.device)
        
        # Batch reward computation - compute all rewards at once
        with torch.no_grad():  # Ensure no gradients are computed for reward model
            raw_values = safe_reward_computation(
                reward_model, 
                toxicity_inputs, 
                ppo_trainer.accelerator.device
            )
        
        # Calculate rewards based on configuration
        if cfg.model.use_raw_logits:
            raw_toxicity_labels = raw_values.tolist()
            # Check for NaN or inf values and replace them
            raw_toxicity_labels = [
                0.0 if (not isinstance(x, (int, float)) or np.isnan(x) or np.isinf(x)) 
                else x for x in raw_toxicity_labels
            ]
            rewards = [torch.tensor(output) for output in raw_toxicity_labels]
        else:
            # Apply softmax for probability scores
            softmax_values = torch.nn.functional.softmax(raw_values.view(-1, 1), dim=1)[:, 0]
            softmax_toxicity_labels = softmax_values.tolist()
            # Check for NaN or inf values and replace them
            softmax_toxicity_labels = [
                0.0 if (not isinstance(x, (int, float)) or np.isnan(x) or np.isinf(x)) 
                else x for x in softmax_toxicity_labels
            ]
            rewards = [torch.tensor(output) for output in softmax_toxicity_labels]
        
        # Calculate statistics for logging
        rewards_tensor = torch.tensor([r.item() for r in rewards])
        raw_mean = rewards_tensor.mean().item()
        raw_std = rewards_tensor.std().item()
        
        # Store statistics in tracking
        reward_stats['epoch'].append(epoch)
        reward_stats['raw_rewards_mean'].append(raw_mean)
        reward_stats['raw_rewards_std'].append(raw_std)
        
        # Count NaN/Inf values
        nan_inf_count = sum(1 for x in raw_toxicity_labels if not isinstance(x, (int, float)) or np.isnan(x) or np.isinf(x))
        reward_stats['nan_inf_count'].append(nan_inf_count)
        
        # Print reward stats periodically
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch} reward stats:")
            print(f"  Rewards - Mean: {raw_mean:.4f}, Std: {raw_std:.4f}")
            print(f"  NaN/Inf values replaced: {nan_inf_count}/{len(raw_toxicity_labels)} ({nan_inf_count/len(raw_toxicity_labels)*100:.1f}%)")
        
        # Run PPO update with minimal overhead
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Augment stats dictionary with reward metrics
        stats["rewards/mean"] = raw_mean
        stats["rewards/std"] = raw_std
        stats["current_epoch"] = epoch
        
        # Log stats efficiently
        ppo_trainer.log_stats(stats, batch, rewards)
        
        # Save model checkpoint
        if (epoch + 1) % cfg.training.save_freq == 0:
            checkpoint_data = (epoch + 1, ppo_trainer, reward_stats, output_dir)
            checkpoint_queue.put(checkpoint_data)
        
        # Push checkpoint to Hub if enabled (separate from local saving)
        if cfg.output.push_to_hub and cfg.output.push_checkpoints_to_hub and (epoch + 1) % cfg.output.checkpoint_push_freq == 0:
            try:
                # Create a temporary checkpoint path if we didn't just save one
                if (epoch + 1) % cfg.training.save_freq != 0:
                    temp_checkpoint_path = os.path.join(checkpoint_dir, f"temp-checkpoint-epoch-{epoch+1}")
                    print(f"Creating temporary checkpoint for Hub push at {temp_checkpoint_path}")
                    
                    if ppo_trainer.accelerator.is_main_process:
                        # Save the model to the temporary path
                        ppo_trainer.save_pretrained(temp_checkpoint_path)
                        checkpoint_path = temp_checkpoint_path
                else:
                    # Use the already saved checkpoint
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch+1}")
                
                # Determine repository name
                if cfg.output.repository_name:
                    repo_name = cfg.output.repository_name
                else:
                    model_short_name = cfg.model.name.split('/')[-1]
                    repo_name = f"{model_short_name}-detox"
                
                # Prepare repository ID
                repo_id = f"{cfg.output.organization}/{repo_name}" if cfg.output.organization else repo_name
                
                # Add epoch information to the checkpoint folder
                checkpoint_repo_name = f"{repo_name}-checkpoint-epoch-{epoch+1}"
                checkpoint_repo_id = f"{cfg.output.organization}/{checkpoint_repo_name}" if cfg.output.organization else checkpoint_repo_name
                
                print(f"Pushing checkpoint to Hugging Face Hub: {checkpoint_repo_id}")
                
                # Save model and tokenizer to the checkpoint path
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                # Save config file
                with open(os.path.join(checkpoint_path, "rlhf_config.yaml"), "w") as f:
                    f.write(OmegaConf.to_yaml(cfg))
                
                # Push to Hub
                api = HfApi()
                
                # Check if the repository exists, create it if it doesn't
                if not api.repo_exists(repo_id=checkpoint_repo_id):
                    api.create_repo(repo_id=checkpoint_repo_id, private=cfg.output.private)
                
                # Upload the folder
                api.upload_folder(
                    folder_path=checkpoint_path,
                    repo_id=checkpoint_repo_id,
                    commit_message=f"Checkpoint after epoch {epoch+1}"
                )
                print(f"Successfully pushed checkpoint to {checkpoint_repo_id}")
                
                # Clean up temporary checkpoint if created
                if (epoch + 1) % cfg.training.save_freq != 0 and os.path.exists(temp_checkpoint_path):
                    import shutil
                    shutil.rmtree(temp_checkpoint_path)
                    print(f"Removed temporary checkpoint directory {temp_checkpoint_path}")
                    
            except Exception as e:
                print(f"Error pushing checkpoint to Hugging Face Hub: {str(e)}")
                print("Continuing training without pushing checkpoint.")
        
        # Run evaluation asynchronously
        if (epoch + 1) % cfg.training.eval_freq == 0:
            eval_data = (epoch + 1, model, ppo_trainer, tokenizer, reward_model, reward_tokenizer, test_dataset, cfg)
            evaluation_queue.put(eval_data)
    
    # Save final model
    final_path = os.path.join(output_dir, "final-model")
    print(f"Saving final model to {final_path}")
    
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.save_pretrained(final_path)
        
        # Save final reward stats
        reward_df = pd.DataFrame(reward_stats)
        reward_df.to_csv(os.path.join(output_dir, "final_reward_stats.csv"), index=False)
        
        # Push to Hugging Face Hub if enabled
        if cfg.output.push_to_hub:
            # Determine repository name
            if cfg.output.repository_name:
                repo_name = cfg.output.repository_name
            else:
                model_short_name = cfg.model.name.split('/')[-1]
                repo_name = f"{model_short_name}-detox"
            
            # Prepare repository ID
            repo_id = f"{cfg.output.organization}/{repo_name}" if cfg.output.organization else repo_name
            
            print(f"Pushing final model to Hugging Face Hub: {repo_id}")
            
            # Save model and tokenizer
            model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            
            # Save config file
            with open(os.path.join(final_path, "rlhf_config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
            
            # Push to Hub
            try:
                api = HfApi()
                
                # Check if the repository exists, create it if it doesn't
                if not api.repo_exists(repo_id=repo_id):
                    api.create_repo(repo_id=repo_id, private=False)
                
                # Upload the folder
                api.upload_folder(
                    folder_path=final_path,
                    repo_id=repo_id,
                    commit_message="Final model after RLHF training"
                )
                print(f"Successfully pushed model to {repo_id}")
            except Exception as e:
                print(f"Error pushing to Hugging Face Hub: {str(e)}")
                print("Continuing without pushing to Hub.")
    
    # Final evaluation
    final_toxicity, _ = evaluate_toxicity(
        model=model,
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        dataset=test_dataset,
        config=cfg,
        epoch="final"
    )
    
    print(f"Final evaluation: Average toxicity = {final_toxicity:.4f}")
    
    # Calculate total training time
    total_time = time.time() - training_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Training complete! Models and results saved to: {output_dir}")
    
    # Cleanup async threads
    print("Cleaning up async threads...")
    checkpoint_queue.put(None)  # Shutdown signal
    evaluation_queue.put(None)  # Shutdown signal
    
    # Wait for threads to finish
    checkpoint_thread.join(timeout=10)
    evaluation_thread.join(timeout=10)
    
    print("Async cleanup complete!")
    
    # Return final toxicity for potential programmatic use
    return final_toxicity


if __name__ == "__main__":
    # Let Hydra handle all command-line arguments
    train_rlhf()