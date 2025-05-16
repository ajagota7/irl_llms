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
    LengthSampler,
    safe_reward_computation
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
            
            # Use safe generation instead of direct generation
            response = safe_generate(ppo_trainer, query, generation_kwargs)
            
            # Extract the generated part (last gen_len tokens)
            if response.size(1) >= gen_len:
                response_tensors.append(response.squeeze()[-gen_len:])
            else:
                # If response is shorter than expected, pad it
                padding = torch.full((gen_len - response.size(1),), 
                                    tokenizer.pad_token_id, 
                                    device=response.device)
                padded_response = torch.cat([response.squeeze(), padding], dim=0)
                response_tensors.append(padded_response[-gen_len:])
        
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # Compute toxicity scores as rewards
        texts = batch["response"]
        toxicity_inputs = reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(ppo_trainer.accelerator.device)
        
        # Use safe reward computation
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
        
        # Run PPO update safely
        stats = safe_ppo_step(ppo_trainer, query_tensors, response_tensors, rewards)
        
        # Augment stats dictionary with reward metrics
        stats["rewards/mean"] = raw_mean
        stats["rewards/std"] = raw_std
        stats["current_epoch"] = epoch
        
        # Log stats safely
        safe_log_stats(ppo_trainer, stats, batch, rewards)
        
        # Save model checkpoint
        if (epoch + 1) % cfg.training.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch+1}")
            print(f"Saving model checkpoint to {checkpoint_path}")
            
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(checkpoint_path)
                
                # Save reward stats
                reward_df = pd.DataFrame(reward_stats)
                reward_df.to_csv(os.path.join(output_dir, "reward_stats.csv"), index=False)
                
                # Push checkpoint to Hub if enabled
                if cfg.output.push_to_hub and cfg.output.push_checkpoints_to_hub:
                    try:
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
                    except Exception as e:
                        print(f"Error pushing checkpoint to Hugging Face Hub: {str(e)}")
                        print("Continuing training without pushing checkpoint.")
        
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


def safe_generate(ppo_trainer, query, generation_kwargs):
    """Safely generate text, handling potential CUDA errors."""
    try:
        # Standard generation
        response = ppo_trainer.generate(query, **generation_kwargs)
        return response
    except RuntimeError as e:
        if "CUDA error" in str(e) or "device-side assert triggered" in str(e):
            print(f"CUDA error during generation: {e}")
            print("Attempting fallback generation with safer parameters...")
            
            # Create safer generation parameters
            safe_kwargs = generation_kwargs.copy()
            # Disable sampling which can cause probability issues
            safe_kwargs["do_sample"] = False
            # Use greedy decoding instead
            safe_kwargs["num_beams"] = 1
            
            try:
                # Try again with safer parameters
                response = ppo_trainer.generate(query, **safe_kwargs)
                return response
            except Exception as e2:
                print(f"Fallback generation also failed: {e2}")
                print("Creating empty response as last resort")
                
                # Create a minimal valid response as last resort
                # Just return the input with a simple completion
                device = ppo_trainer.accelerator.device
                if hasattr(ppo_trainer.model, 'pretrained_model'):
                    vocab_size = ppo_trainer.model.pretrained_model.config.vocab_size
                else:
                    vocab_size = ppo_trainer.model.config.vocab_size
                
                # Get the token IDs for a simple completion like " is"
                simple_tokens = ppo_trainer.tokenizer(" is", add_special_tokens=False).input_ids
                
                # Create a response that's just the query plus this simple completion
                min_length = generation_kwargs.get("min_length", 5)
                response_length = max(min_length, len(simple_tokens))
                
                # Create a tensor with the right shape
                response = torch.cat([
                    query.unsqueeze(0),  # Add batch dimension
                    torch.tensor([simple_tokens], device=device)
                ], dim=1)
                
                return response
        else:
            # If it's not a CUDA error, re-raise
            raise


def safe_log_stats(ppo_trainer, stats, batch, rewards):
    """Safely log stats, handling NaN values."""
    # Clean up stats dictionary to remove NaN/inf values
    clean_stats = {}
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            if np.isnan(v) or np.isinf(v):
                print(f"Warning: {k} has invalid value {v}, replacing with 0")
                clean_stats[k] = 0.0
            else:
                clean_stats[k] = v
        else:
            clean_stats[k] = v
    
    # Handle histograms specially
    if 'ppo/advantages' in clean_stats:
        advantages = clean_stats['ppo/advantages']
        if isinstance(advantages, list):
            # Filter out NaN and inf values
            filtered_advantages = [x for x in advantages if isinstance(x, (int, float)) and not np.isnan(x) and not np.isinf(x)]
            if not filtered_advantages:  # If all values were invalid
                filtered_advantages = [0.0]
            clean_stats['ppo/advantages'] = filtered_advantages
    
    # Same for other potential histogram values
    for key in ['ppo/ratio', 'ppo/policy_loss', 'ppo/value_loss']:
        if key in clean_stats and isinstance(clean_stats[key], list):
            clean_stats[key] = [x for x in clean_stats[key] if isinstance(x, (int, float)) and not np.isnan(x) and not np.isinf(x)]
            if not clean_stats[key]:  # If all values were invalid
                clean_stats[key] = [0.0]
    
    try:
        ppo_trainer.log_stats(clean_stats, batch, rewards)
    except Exception as e:
        print(f"Error in logging stats: {e}")
        # Try a minimal logging approach
        try:
            minimal_stats = {
                'rewards/mean': clean_stats.get('rewards/mean', 0.0),
                'current_epoch': clean_stats.get('current_epoch', 0)
            }
            ppo_trainer.accelerator.log(minimal_stats)
            print(f"Logged minimal stats: {minimal_stats}")
        except Exception as e2:
            print(f"Even minimal logging failed: {e2}")


def safe_ppo_step(ppo_trainer, query_tensors, response_tensors, rewards):
    """Safely perform PPO step with error handling."""
    try:
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        return stats
    except RuntimeError as e:
        if "CUDA error" in str(e) or "device-side assert triggered" in str(e):
            print(f"CUDA error during PPO step: {e}")
            print("Returning empty stats dictionary")
            return {"error": str(e)}
        else:
            raise


if __name__ == "__main__":
    # Let Hydra handle all command-line arguments
    train_rlhf()