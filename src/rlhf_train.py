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
    LengthSampler
)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train_rlhf(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Add current timestamp
    cfg.now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Print configuration
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directories
    output_dir = os.path.join(os.getcwd(), f"outputs/{cfg.now}")
    os.makedirs(output_dir, exist_ok=True)
    eval_dir = os.path.join(output_dir, "evaluation")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
    
    # Load model and add value head
    print(f"Loading model {cfg.model.name}...")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
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
    
    # Handle batch size parameters to ensure they're compatible
    batch_size = cfg.model.batch_size
    mini_batch_size = cfg.model.mini_batch_size
    gradient_accumulation_steps = cfg.model.gradient_accumulation_steps

    # Ensure batch_size is a multiple of mini_batch_size * gradient_accumulation_steps
    if batch_size % (mini_batch_size * gradient_accumulation_steps) != 0:
        # Option 1: Adjust mini_batch_size to make it work
        if batch_size >= gradient_accumulation_steps:
            new_mini_batch_size = batch_size // gradient_accumulation_steps
            print(f"Warning: Adjusting mini_batch_size from {mini_batch_size} to {new_mini_batch_size} to ensure compatibility with batch_size={batch_size}")
            mini_batch_size = new_mini_batch_size
        # Option 2: If that's not possible, adjust gradient_accumulation_steps
        else:
            new_gradient_accumulation_steps = 1
            new_mini_batch_size = batch_size
            print(f"Warning: Adjusting gradient_accumulation_steps from {gradient_accumulation_steps} to {new_gradient_accumulation_steps} and mini_batch_size from {mini_batch_size} to {new_mini_batch_size} to ensure compatibility with batch_size={batch_size}")
            gradient_accumulation_steps = new_gradient_accumulation_steps
            mini_batch_size = new_mini_batch_size

    # Add the adjusted batch parameters
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
    }
    
    # Training loop
    print("Starting training loop...")
    training_start_time = time.time()
    
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=cfg.training.num_train_epochs):
        if epoch >= cfg.training.num_train_epochs:
            break
        
        # Process batch
        query_tensors = batch["input_ids"]
        
        # Get response from policy model
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs = {
                "min_length": cfg.model.generation.min_length,
                "top_k": cfg.model.generation.top_k,
                "top_p": cfg.model.generation.top_p,
                "do_sample": cfg.model.generation.do_sample,
                "pad_token_id": tokenizer.eos_token_id,
                "max_new_tokens": gen_len
            }
            
            # Make sure query is 1D
            query = query.squeeze()
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # Compute toxicity scores as rewards
        texts = batch["response"]
        toxicity_inputs = reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(ppo_trainer.accelerator.device)
        
        logits = reward_model(**toxicity_inputs).logits.float()
        
        # Calculate rewards based on configuration
        if cfg.model.use_raw_logits:
            raw_toxicity_labels = (logits[:, 0]).tolist()
            rewards = [torch.tensor(output) for output in raw_toxicity_labels]
        else:
            softmax_toxicity_labels = reward_model(**toxicity_inputs).logits.softmax(dim=1)[:, 0].float().tolist()
            rewards = [torch.tensor(output) for output in softmax_toxicity_labels]
        
        # Calculate statistics for logging
        rewards_tensor = torch.tensor([r.item() for r in rewards])
        raw_mean = rewards_tensor.mean().item()
        raw_std = rewards_tensor.std().item()
        
        # Store statistics in tracking
        reward_stats['epoch'].append(epoch)
        reward_stats['raw_rewards_mean'].append(raw_mean)
        reward_stats['raw_rewards_std'].append(raw_std)
        
        # Print reward stats periodically
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch} reward stats:")
            print(f"  Rewards - Mean: {raw_mean:.4f}, Std: {raw_std:.4f}")
        
        # Run PPO update
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Augment stats dictionary with reward metrics
        stats["rewards/mean"] = raw_mean
        stats["rewards/std"] = raw_std
        stats["current_epoch"] = epoch
        
        # Log stats
        ppo_trainer.log_stats(stats, batch, rewards)
        
        # Save model checkpoint
        if (epoch + 1) % cfg.training.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch+1}")
            print(f"Saving model checkpoint to {checkpoint_path}")
            
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(checkpoint_path)
                
                # Save reward stats
                reward_df = pd.DataFrame(reward_stats)
                reward_df.to_csv(os.path.join(output_dir, "reward_stats.csv"), index=False)
        
        # Run evaluation
        if (epoch + 1) % cfg.training.eval_freq == 0:
            print(f"\nEvaluating at epoch {epoch+1}...")
            
            avg_toxicity, _ = evaluate_toxicity(
                model=model,
                ppo_trainer=ppo_trainer,
                tokenizer=tokenizer,
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                dataset=test_dataset,
                config=cfg,
                epoch=epoch+1
            )
            
            print(f"Epoch {epoch+1}: Average toxicity = {avg_toxicity:.4f}")
            
            # Save evaluation results
            with open(os.path.join(eval_dir, "evaluation_results.txt"), "a") as f:
                f.write(f"Epoch {epoch+1}: Average toxicity = {avg_toxicity:.4f}\n")
            
            # Log evaluation metrics
            if wandb_run:
                wandb_run.log({"eval/toxicity": avg_toxicity, "eval/epoch": epoch+1})
    
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
    
    # Return final toxicity for potential programmatic use
    return final_toxicity


if __name__ == "__main__":
    # Let Hydra handle all command-line arguments
    train_rlhf()