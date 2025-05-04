"""
Main RLHF training script for detoxifying language models.
"""

import os
import time
import torch
import hydra
import argparse
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
    os.makedirs(cfg.hydra.run.dir, exist_ok=True)
    eval_dir = os.path.join(cfg.hydra.run.dir, "evaluation")
    checkpoint_dir = os.path.join(cfg.hydra.run.dir, "checkpoints")
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
    
    # Create PPO config
    ppo_config = PPOConfig(
        model_name=cfg.model.name,
        learning_rate=cfg.model.learning_rate,
        log_with="wandb" if wandb_run else None,
        ppo_epochs=cfg.model.ppo_epochs,
        mini_batch_size=cfg.model.mini_batch_size,
        batch_size=cfg.model.batch_size,
        forward_batch_size=cfg.model.forward_batch_size,
        gradient_accumulation_steps=cfg.model.gradient_accumulation_steps,
        init_kl_coef=cfg.model.init_kl_coef,
        target=cfg.model.target,
        cliprange=cfg.model.cliprange,
        cliprange_value=cfg.model.cliprange_value,
        vf_coef=cfg.model.vf_coef,
        adap_kl_ctrl=cfg.model.adap_kl_ctrl,
        use_score_norm=cfg.model.use_score_norm,
        ratio_threshold=cfg.model.ratio_threshold,
    )
    
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
                reward_df.to_csv(os.path.join(cfg.hydra.run.dir, "reward_stats.csv"), index=False)
        
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
    final_path = os.path.join(cfg.hydra.run.dir, "final-model")
    print(f"Saving final model to {final_path}")
    
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.save_pretrained(final_path)
        
        # Save final reward stats
        reward_df = pd.DataFrame(reward_stats)
        reward_df.to_csv(os.path.join(cfg.hydra.run.dir, "final_reward_stats.csv"), index=False)
        
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
            api = HfApi()
            api.upload_folder(
                folder_path=final_path,
                repo_id=repo_id,
                commit_message="Final model after RLHF training",
                create_repo=True
            )
    
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
    print(f"Training complete! Models and results saved to: {cfg.hydra.run.dir}")
    
    # Return final toxicity for potential programmatic use
    return final_toxicity


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RLHF Training for Model Detoxification")
    
    # Model selection
    parser.add_argument(
        "--model", 
        type=str,
        default="gpt_neo_125m",
        choices=["pythia_70m", "pythia_160m", "pythia_410m", "pythia_1b", "gpt_neo_125m"],
        help="Model to use"
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # Output configuration
    parser.add_argument("--save_freq", type=int, help="Checkpoint save frequency (epochs)")
    parser.add_argument("--eval_freq", type=int, help="Evaluation frequency (epochs)")
    
    # HuggingFace Hub integration
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hf_org", type=str, help="HuggingFace organization name")
    parser.add_argument("--hf_repo", type=str, help="HuggingFace repository name")
    
    # WandB configuration
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, help="Weights & Biases entity (username/team)")
    parser.add_argument("--wandb_name", type=str, help="Weights & Biases run name")
    
    # Additional overrides
    parser.add_argument("--config_file", type=str, help="Path to YAML config file with overrides")
    
    return parser.parse_args()


def build_config_overrides(args):
    """Build Hydra config overrides from command-line arguments."""
    overrides = []
    
    # Model selection
    if args.model:
        overrides.append(f"rlhf={args.model}")
    
    # Training parameters
    if args.epochs:
        overrides.append(f"training.num_train_epochs={args.epochs}")
    
    if args.batch_size:
        overrides.append(f"model.batch_size={args.batch_size}")
    
    if args.lr:
        overrides.append(f"model.learning_rate={args.lr}")
    
    if args.seed:
        overrides.append(f"training.seed={args.seed}")
    
    # Output configuration
    if args.save_freq:
        overrides.append(f"training.save_freq={args.save_freq}")
    
    if args.eval_freq:
        overrides.append(f"training.eval_freq={args.eval_freq}")
    
    # HuggingFace Hub integration
    if args.push_to_hub:
        overrides.append("output.push_to_hub=true")
    
    if args.hf_org:
        overrides.append(f"output.organization={args.hf_org}")
    
    if args.hf_repo:
        overrides.append(f"output.repository_name={args.hf_repo}")
    
    # WandB configuration
    if args.wandb_project:
        overrides.append(f"wandb.project={args.wandb_project}")
    
    if args.wandb_entity:
        overrides.append(f"wandb.entity={args.wandb_entity}")
    
    if args.wandb_name:
        overrides.append(f"wandb.name={args.wandb_name}")
    
    return overrides


if __name__ == "__main__":
    args = parse_args()
    
    # Build hydra config overrides
    overrides = build_config_overrides(args)
    
    # Load additional overrides from file if provided
    if args.config_file and os.path.exists(args.config_file):
        config = OmegaConf.load(args.config_file)
        # Convert config to overrides
        for key, value in OmegaConf.to_container(config).items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    overrides.append(f"{key}.{subkey}={subvalue}")
            else:
                overrides.append(f"{key}={value}")
    
    # Initialize and run with Hydra
    with hydra.initialize_config_module(config_module="configs", version_base=None):
        # Compose the configuration
        cfg = hydra.compose(config_name="config", overrides=overrides)
        print(f"Running with configuration:\n{OmegaConf.to_yaml(cfg)}")
        train_rlhf(cfg)