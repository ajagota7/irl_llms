"""
RLHF training script that uses IRL-trained reward models instead of ground truth models.
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
    evaluate_toxicity,
    analyze_prompt_tracking,
    LengthSampler
)
from irl_utilities import RewardModel


def load_irl_reward_model(model_id: str, device: str) -> tuple:
    """Load IRL-trained reward model for RLHF training."""
    
    print(f"Loading IRL reward model from: {model_id}")
    
    try:
        # First, try to load as a standard HuggingFace model
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Set padding token for GPTNeoX models if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Setting pad_token to eos_token ({tokenizer.eos_token})")
        
        # Try to load using our RewardModel class
        try:
            # Load the reward model using our custom class
            reward_model = RewardModel.load_from_hf_model(model_id, device=device)
            print(f"Successfully loaded IRL reward model using RewardModel.load_from_hf_model")
            
        except Exception as e:
            print(f"Error loading with RewardModel.load_from_hf_model: {e}")
            print("Trying alternative loading method...")
            
            # Alternative: Load base model and try to reconstruct
            from transformers import AutoModelForCausalLM
            import json
            
            # Load the base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            # Try to load reward model config
            try:
                from huggingface_hub import hf_hub_download
                config_path = hf_hub_download(repo_id=model_id, filename="reward_model_config.json")
                with open(config_path, 'r') as f:
                    reward_config = json.load(f)
                
                # Get the original base model name
                original_base_model = reward_config.get('base_model', model_id)
                print(f"Found original base model: {original_base_model}")
                
                # Create reward model with the original base model name
                reward_model = RewardModel(
                    model_name=original_base_model,
                    use_half_precision=(device == "cuda"),
                    device=device,
                    num_unfrozen_layers=0  # For inference
                )
                
                # Replace the base model with the loaded one
                reward_model.model = base_model.to(device)
                
                # Try to load the value head
                try:
                    v_head_path = hf_hub_download(repo_id=model_id, filename="v_head.pt")
                    v_head_state = torch.load(v_head_path, map_location=device)
                    reward_model.v_head.load_state_dict(v_head_state)
                    print("Successfully loaded value head weights")
                except Exception as e:
                    print(f"Warning: Could not load value head weights: {e}")
                    print("Using randomly initialized value head")
                
            except Exception as e:
                print(f"Could not load reward model config: {e}")
                print("Creating reward model with default settings")
                
                # Create a basic reward model
                reward_model = RewardModel(
                    model_name=model_id,
                    use_half_precision=(device == "cuda"),
                    device=device,
                    num_unfrozen_layers=0
                )
                
                # Replace with the loaded model
                reward_model.model = base_model.to(device)
        
        # Set model to evaluation mode
        reward_model.eval()
        print(f"IRL reward model loaded and set to evaluation mode")
        
        return reward_model, tokenizer
        
    except Exception as e:
        print(f"Error loading IRL reward model: {e}")
        raise


def compute_irl_rewards(
    reward_model, 
    reward_tokenizer, 
    texts: list, 
    device: str,
    normalize_rewards: bool = True
) -> torch.Tensor:
    """Compute rewards using IRL-trained reward model."""
    
    # Tokenize the texts
    inputs = reward_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        try:
            # Get rewards from the IRL model
            rewards = reward_model(**inputs)
            
            # Handle different output formats
            if isinstance(rewards, torch.Tensor):
                raw_values = rewards.squeeze(-1).float()
            else:
                # If it's a more complex output, try to extract the values
                raw_values = rewards.squeeze(-1).float()
            
            # Check for NaN or inf values
            if torch.isnan(raw_values).any() or torch.isinf(raw_values).any():
                print("Warning: NaN or Inf values in IRL reward model output, replacing with zeros")
                raw_values = torch.where(
                    torch.isnan(raw_values) | torch.isinf(raw_values),
                    torch.zeros_like(raw_values),
                    raw_values
                )
            
            # Normalize rewards if requested
            if normalize_rewards and len(raw_values) > 1:
                # Z-score normalization
                mean_reward = raw_values.mean()
                std_reward = raw_values.std()
                
                # Avoid division by zero
                if std_reward > 1e-8:
                    normalized_rewards = (raw_values - mean_reward) / std_reward
                else:
                    normalized_rewards = raw_values - mean_reward
                
                return normalized_rewards
            else:
                return raw_values
                
        except Exception as e:
            print(f"Error in IRL reward computation: {e}")
            # Return neutral values as fallback
            batch_size = inputs['input_ids'].size(0)
            return torch.zeros(batch_size, device=device)


def safe_irl_reward_computation(reward_model, reward_inputs, device, normalize_rewards=True):
    """Safely compute IRL rewards, handling potential errors."""
    with torch.no_grad():
        try:
            # Try standard computation
            rewards = reward_model(**reward_inputs)
            
            # Handle different model output formats
            if isinstance(rewards, torch.Tensor):
                raw_values = rewards.squeeze(-1).float()
            else:
                raw_values = rewards.squeeze(-1).float()
            
            # Check for NaN or inf values
            if torch.isnan(raw_values).any() or torch.isinf(raw_values).any():
                print("Warning: NaN or Inf values in IRL reward model output, replacing with zeros")
                raw_values = torch.where(
                    torch.isnan(raw_values) | torch.isinf(raw_values),
                    torch.zeros_like(raw_values),
                    raw_values
                )
            
            # Normalize rewards if requested
            if normalize_rewards and len(raw_values) > 1:
                # Z-score normalization
                mean_reward = raw_values.mean()
                std_reward = raw_values.std()
                
                # Avoid division by zero
                if std_reward > 1e-8:
                    normalized_rewards = (raw_values - mean_reward) / std_reward
                else:
                    normalized_rewards = raw_values - mean_reward
                
                return normalized_rewards
            else:
                return raw_values
            
        except Exception as e:
            print(f"Error in IRL reward computation: {e}")
            # Return neutral values as fallback
            batch_size = reward_inputs['input_ids'].size(0)
            return torch.zeros(batch_size, device=device)


def evaluate_irl_toxicity(
    model, 
    ppo_trainer, 
    tokenizer, 
    reward_model, 
    reward_tokenizer, 
    dataset, 
    config, 
    epoch
) -> tuple:
    """Evaluate model toxicity using IRL reward model."""
    
    # Create evaluation directory
    output_dir = os.path.join(os.getcwd(), f"outputs/{config.now}")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    device = ppo_trainer.accelerator.device
    
    # Sample a subset of the dataset for evaluation
    eval_size = min(100, len(dataset))
    eval_indices = np.random.choice(len(dataset), eval_size, replace=False)
    eval_samples = [dataset[i] for i in eval_indices]
    
    # Setup generation parameters
    gen_kwargs = {
        "min_length": config.model.generation.min_length,
        "top_k": config.model.generation.top_k,
        "top_p": config.model.generation.top_p,
        "do_sample": config.model.generation.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": config.model.generation.output_max_length,
    }
    
    # Generate responses and calculate toxicity using IRL model
    toxicity_scores = []
    generations = []
    
    for sample in tqdm(eval_samples, desc=f"Evaluating with IRL model (epoch {epoch})"):
        query = sample["query"]
        
        # Tokenize the query
        query_tensor = tokenizer(query, return_tensors="pt")
        query_input_ids = query_tensor.input_ids.squeeze().to(device)
        
        # Generate response
        try:
            response_tensor = ppo_trainer.generate(query_input_ids, **gen_kwargs)
            response = tokenizer.decode(response_tensor[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error generating response: {e}")
            response = query  # Fallback to original query
        
        # Calculate toxicity using IRL reward model
        try:
            toxicity_score = compute_irl_rewards(
                reward_model, 
                reward_tokenizer, 
                [response], 
                device,
                normalize_rewards=False  # Don't normalize single scores
            )[0].item()
            
            # Convert to a toxicity score (higher = more toxic)
            # IRL models typically output higher values for less toxic content
            # So we invert the score
            toxicity = -toxicity_score
            
        except Exception as e:
            print(f"Error calculating IRL toxicity: {e}")
            toxicity = 0.0  # Default value
        
        toxicity_scores.append(toxicity)
        generations.append({
            "query": query,
            "response": response,
            "irl_toxicity": toxicity
        })
    
    # Calculate average toxicity
    avg_toxicity = sum(toxicity_scores) / len(toxicity_scores)
    
    # Save generations to file
    output_file = os.path.join(eval_dir, f"irl_generations_epoch_{epoch}.json")
    with open(output_file, "w") as f:
        import json
        json.dump(generations, f, indent=2)
    
    # Create toxicity distribution plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(toxicity_scores, bins=20, alpha=0.7)
    plt.xlabel("IRL Toxicity Score")
    plt.ylabel("Frequency")
    plt.title(f"IRL Toxicity Distribution (Epoch {epoch})")
    plt.axvline(avg_toxicity, color='r', linestyle='dashed', linewidth=2, label=f"Mean: {avg_toxicity:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_file = os.path.join(eval_dir, f"irl_toxicity_dist_epoch_{epoch}.png")
    plt.savefig(plot_file)
    plt.close()
    
    return avg_toxicity, generations


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train_irl_rlhf(cfg: DictConfig) -> None:
    """Main RLHF training function using IRL reward models."""
    
    # Add current timestamp
    cfg.now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Print configuration
    print(f"IRL-RLHF Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directories
    output_dir = os.path.join(os.getcwd(), f"outputs/irl-rlhf/{cfg.now}")
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
    
    # Use the IRL reward model from the rlhf config
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
    
    # Get PPO parameters
    ppo_params = {
        "model_name": cfg.model.name,
        "learning_rate": cfg.model.learning_rate,
        "log_with": "wandb" if wandb_run else None,
    }
    
    # Handle batch size parameters
    batch_size = cfg.model.batch_size
    mini_batch_size = cfg.model.mini_batch_size
    gradient_accumulation_steps = cfg.model.gradient_accumulation_steps

    # Ensure batch_size is compatible
    if batch_size % (mini_batch_size * gradient_accumulation_steps) != 0:
        if batch_size >= gradient_accumulation_steps:
            new_mini_batch_size = batch_size // gradient_accumulation_steps
            print(f"Warning: Adjusting mini_batch_size from {mini_batch_size} to {new_mini_batch_size}")
            mini_batch_size = new_mini_batch_size
        else:
            new_gradient_accumulation_steps = 1
            new_mini_batch_size = batch_size
            print(f"Warning: Adjusting gradient_accumulation_steps to {new_gradient_accumulation_steps} and mini_batch_size to {new_mini_batch_size}")
            gradient_accumulation_steps = new_gradient_accumulation_steps
            mini_batch_size = new_mini_batch_size

    ppo_params["batch_size"] = batch_size
    ppo_params["mini_batch_size"] = mini_batch_size
    ppo_params["gradient_accumulation_steps"] = gradient_accumulation_steps

    # Add PPO-specific parameters
    if hasattr(cfg.rlhf, 'model'):
        rlhf_model = cfg.rlhf.model
        for param in ['ppo_epochs', 'init_kl_coef', 'target', 'cliprange', 'cliprange_value', 
                      'vf_coef', 'adap_kl_ctrl', 'use_score_norm', 'ratio_threshold']:
            if hasattr(rlhf_model, param):
                ppo_params[param] = getattr(rlhf_model, param)
    
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
    
    # Load IRL reward model
    print(f"Loading IRL reward model {cfg.model.reward_model}...")
    reward_model, reward_tokenizer = load_irl_reward_model(
        cfg.model.reward_model,
        ppo_trainer.accelerator.device
    )
    
    # Setup generation parameters
    output_length_sampler = LengthSampler(
        cfg.model.generation.output_min_length,
        cfg.model.generation.output_max_length
    )
    
    # Initial evaluation
    print("Performing initial evaluation with IRL model...")
    initial_toxicity, _ = evaluate_irl_toxicity(
        model=model,
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        dataset=test_dataset,
        config=cfg,
        epoch="initial"
    )
    
    print(f"Initial average IRL toxicity: {initial_toxicity:.4f}")
    
    # Save evaluation results
    with open(os.path.join(eval_dir, "irl_evaluation_results.txt"), "w") as f:
        f.write(f"Epoch 0: Average IRL toxicity = {initial_toxicity:.4f}\n")
    
    # Log initial metrics
    if wandb_run:
        wandb_run.log({"eval/initial_irl_toxicity": initial_toxicity})
    
    # Create a dictionary to store reward stats
    reward_stats = {
        'epoch': [],
        'irl_rewards_mean': [],
        'irl_rewards_std': [],
        'nan_inf_count': [],
    }
    
    # Training loop
    print("Starting IRL-RLHF training loop...")
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
            
            # Generate response safely
            try:
                response = ppo_trainer.generate(query, **generation_kwargs)
                
                # Extract the generated part
                if response.size(1) >= gen_len:
                    response_tensors.append(response.squeeze()[-gen_len:])
                else:
                    # Pad if necessary
                    padding = torch.full((gen_len - response.size(1),), 
                                        tokenizer.pad_token_id, 
                                        device=response.device)
                    padded_response = torch.cat([response.squeeze(), padding], dim=0)
                    response_tensors.append(padded_response[-gen_len:])
            except Exception as e:
                print(f"Error in generation: {e}")
                # Create a fallback response
                fallback_response = torch.full((gen_len,), tokenizer.pad_token_id, device=query.device)
                response_tensors.append(fallback_response)
        
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # Compute IRL rewards
        texts = batch["response"]
        reward_inputs = reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(ppo_trainer.accelerator.device)
        
        # Use safe IRL reward computation with normalization
        raw_values = safe_irl_reward_computation(
            reward_model, 
            reward_inputs, 
            ppo_trainer.accelerator.device,
            normalize_rewards=cfg.model.get('use_score_norm', True)
        )
        
        # Convert to rewards (invert if needed - IRL models typically give higher scores for better content)
        # For detoxification, we want to reward less toxic content
        if cfg.model.get('invert_irl_rewards', True):
            # Invert the rewards so higher IRL scores (less toxic) become higher rewards
            rewards = [torch.tensor(score) for score in raw_values.tolist()]
        else:
            # Use raw scores
            rewards = [torch.tensor(-score) for score in raw_values.tolist()]  # Negative for toxicity
        
        # Calculate statistics for logging
        rewards_tensor = torch.tensor([r.item() for r in rewards])
        rewards_mean = rewards_tensor.mean().item()
        rewards_std = rewards_tensor.std().item()
        
        # Store statistics
        reward_stats['epoch'].append(epoch)
        reward_stats['irl_rewards_mean'].append(rewards_mean)
        reward_stats['irl_rewards_std'].append(rewards_std)
        
        # Count NaN/Inf values
        nan_inf_count = sum(1 for x in raw_values.tolist() if not isinstance(x, (int, float)) or np.isnan(x) or np.isinf(x))
        reward_stats['nan_inf_count'].append(nan_inf_count)
        
        # Print reward stats periodically
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch} IRL reward stats:")
            print(f"  IRL Rewards - Mean: {rewards_mean:.4f}, Std: {rewards_std:.4f}")
            print(f"  NaN/Inf values replaced: {nan_inf_count}/{len(raw_values)} ({nan_inf_count/len(raw_values)*100:.1f}%)")
        
        # Run PPO update
        try:
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        except Exception as e:
            print(f"Error in PPO step: {e}")
            stats = {"error": str(e)}
        
        # Augment stats with IRL reward metrics
        stats["irl_rewards/mean"] = rewards_mean
        stats["irl_rewards/std"] = rewards_std
        stats["current_epoch"] = epoch
        
        # Log stats safely
        try:
            # Clean up stats to remove NaN/inf values
            clean_stats = {}
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    if np.isnan(v) or np.isinf(v):
                        clean_stats[k] = 0.0
                    else:
                        clean_stats[k] = v
                else:
                    clean_stats[k] = v
            
            ppo_trainer.log_stats(clean_stats, batch, rewards)
        except Exception as e:
            print(f"Error in logging stats: {e}")
        
        # Save model checkpoint
        if (epoch + 1) % cfg.training.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"irl-checkpoint-epoch-{epoch+1}")
            print(f"Saving IRL-RLHF model checkpoint to {checkpoint_path}")
            
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(checkpoint_path)
                
                # Save IRL reward stats
                reward_df = pd.DataFrame(reward_stats)
                reward_df.to_csv(os.path.join(output_dir, "irl_reward_stats.csv"), index=False)
        
        # Push checkpoint to Hub if enabled
        if cfg.output.push_to_hub and cfg.output.push_checkpoints_to_hub and (epoch + 1) % cfg.output.checkpoint_push_freq == 0:
            try:
                # Create checkpoint path if needed
                if (epoch + 1) % cfg.training.save_freq != 0:
                    temp_checkpoint_path = os.path.join(checkpoint_dir, f"temp-irl-checkpoint-epoch-{epoch+1}")
                    if ppo_trainer.accelerator.is_main_process:
                        ppo_trainer.save_pretrained(temp_checkpoint_path)
                        checkpoint_path = temp_checkpoint_path
                else:
                    checkpoint_path = os.path.join(checkpoint_dir, f"irl-checkpoint-epoch-{epoch+1}")
                
                # Determine repository name
                if cfg.output.repository_name:
                    repo_name = f"{cfg.output.repository_name}-irl"
                else:
                    model_short_name = cfg.model.name.split('/')[-1]
                    repo_name = f"{model_short_name}-irl-detox"
                
                repo_id = f"{cfg.output.organization}/{repo_name}" if cfg.output.organization else repo_name
                checkpoint_repo_name = f"{repo_name}-checkpoint-epoch-{epoch+1}"
                checkpoint_repo_id = f"{cfg.output.organization}/{checkpoint_repo_name}" if cfg.output.organization else checkpoint_repo_name
                
                print(f"Pushing IRL checkpoint to Hugging Face Hub: {checkpoint_repo_id}")
                
                # Save and push
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                with open(os.path.join(checkpoint_path, "irl_rlhf_config.yaml"), "w") as f:
                    f.write(OmegaConf.to_yaml(cfg))
                
                api = HfApi()
                if not api.repo_exists(repo_id=checkpoint_repo_id):
                    api.create_repo(repo_id=checkpoint_repo_id, private=cfg.output.private)
                
                api.upload_folder(
                    folder_path=checkpoint_path,
                    repo_id=checkpoint_repo_id,
                    commit_message=f"IRL-RLHF checkpoint after epoch {epoch+1}"
                )
                print(f"Successfully pushed IRL checkpoint to {checkpoint_repo_id}")
                
                # Clean up temporary checkpoint
                if (epoch + 1) % cfg.training.save_freq != 0 and os.path.exists(temp_checkpoint_path):
                    import shutil
                    shutil.rmtree(temp_checkpoint_path)
                    
            except Exception as e:
                print(f"Error pushing IRL checkpoint to Hub: {str(e)}")
        
        # Run evaluation
        if (epoch + 1) % cfg.training.eval_freq == 0:
            print(f"\nEvaluating IRL-RLHF at epoch {epoch+1}...")
            
            avg_toxicity, _ = evaluate_irl_toxicity(
                model=model,
                ppo_trainer=ppo_trainer,
                tokenizer=tokenizer,
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                dataset=test_dataset,
                config=cfg,
                epoch=epoch+1
            )
            
            print(f"Epoch {epoch+1}: Average IRL toxicity = {avg_toxicity:.4f}")
            
            # Save evaluation results
            with open(os.path.join(eval_dir, "irl_evaluation_results.txt"), "a") as f:
                f.write(f"Epoch {epoch+1}: Average IRL toxicity = {avg_toxicity:.4f}\n")
            
            # Log evaluation metrics
            if wandb_run:
                wandb_run.log({"eval/irl_toxicity": avg_toxicity, "eval/epoch": epoch+1})
    
    # Save final model
    final_path = os.path.join(output_dir, "final-irl-model")
    print(f"Saving final IRL-RLHF model to {final_path}")
    
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.save_pretrained(final_path)
        
        # Save final IRL reward stats
        reward_df = pd.DataFrame(reward_stats)
        reward_df.to_csv(os.path.join(output_dir, "final_irl_reward_stats.csv"), index=False)
        
        # Push to Hugging Face Hub if enabled
        if cfg.output.push_to_hub:
            if cfg.output.repository_name:
                repo_name = f"{cfg.output.repository_name}-irl"
            else:
                model_short_name = cfg.model.name.split('/')[-1]
                repo_name = f"{model_short_name}-irl-detox"
            
            repo_id = f"{cfg.output.organization}/{repo_name}" if cfg.output.organization else repo_name
            
            print(f"Pushing final IRL-RLHF model to Hugging Face Hub: {repo_id}")
            
            # Save model and tokenizer
            model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            
            # Save config file
            with open(os.path.join(final_path, "irl_rlhf_config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
            
            # Push to Hub
            try:
                api = HfApi()
                if not api.repo_exists(repo_id=repo_id):
                    api.create_repo(repo_id=repo_id, private=cfg.output.private)
                
                api.upload_folder(
                    folder_path=final_path,
                    repo_id=repo_id,
                    commit_message="Final IRL-RLHF model"
                )
                print(f"Successfully pushed IRL-RLHF model to {repo_id}")
            except Exception as e:
                print(f"Error pushing to Hugging Face Hub: {str(e)}")
    
    # Final evaluation
    final_toxicity, _ = evaluate_irl_toxicity(
        model=model,
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        dataset=test_dataset,
        config=cfg,
        epoch="final"
    )
    
    print(f"Final IRL evaluation: Average toxicity = {final_toxicity:.4f}")
    
    # Calculate total training time
    total_time = time.time() - training_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Total IRL-RLHF training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"IRL-RLHF training complete! Models and results saved to: {output_dir}")
    
    return final_toxicity


if __name__ == "__main__":
    train_irl_rlhf() 