"""
Main training script for IRL detoxification.
"""

import os
import time
import torch
import hydra
import wandb
import numpy as np
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy import stats
import json
from datasets import load_dataset

from src.irl_utilities import (
    RewardModel, 
    get_loss_function, 
    plot_metrics, 
    plot_score_distribution, 
    push_to_hub,
    prepare_data as prepare_data_util
)


class IRLTrainer:
    """Trainer class for IRL detoxification."""

    def __init__(self, config: DictConfig):
        """Initialize the trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get model names for run ID
        original_model = config.dataset.original_model_name.split('/')[-1]
        detoxified_model = config.dataset.detoxified_model_name.split('/')[-1]
        
        self.run_id = f"{original_model}_to_{detoxified_model}_{self.timestamp}"
        
        # Set up output directories
        self.output_dir = os.path.join(config.output.base_dir, self.run_id)
        self.model_dir = os.path.join(self.output_dir, "models")
        self.eval_dir = os.path.join(self.output_dir, "evaluation")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Set random seeds
        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)
        
        # Initialize the reward model and tokenizer
        self._init_model_and_tokenizer()
        
        # Initialize the true reward model for evaluation
        self._init_true_reward_model()
        
        # Initialize metrics history
        self.metrics_history = []
        
    def _init_model_and_tokenizer(self):
        """Initialize the reward model and tokenizer."""
        model_name = self.config.model.reward_model_base
        
        # Check if we should use half precision
        use_half_precision = self.config.model.use_half_precision
        if use_half_precision is None:
            # Automatically detect based on model size
            large_model = any(size in model_name.lower() for size in ["2.7b", "6b", "7b", "12b", "70b", "large"])
            use_half_precision = large_model
        
        print(f"Initializing reward model based on {model_name}...")
        self.reward_model = RewardModel(
            model_name=model_name,
            use_half_precision=use_half_precision,
            device=self.device,
            num_unfrozen_layers=self.config.model.num_unfrozen_layers
        )
        
        # Enable gradient checkpointing for memory efficiency (only if we have trainable parameters)
        if hasattr(self.reward_model.model, 'gradient_checkpointing_enable'):
            # Only enable if we have unfrozen layers
            if self.config.model.num_unfrozen_layers > 0:
                self.reward_model.model.gradient_checkpointing_enable()
                print("Enabled gradient checkpointing for memory efficiency")
            else:
                print("Skipping gradient checkpointing (no unfrozen layers)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # Print memory usage
        if torch.cuda.is_available():
            print(f"GPU Memory after model load: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        
    def _init_true_reward_model(self):
        """Initialize the true reward model for evaluation."""
        print("Loading true reward model for evaluation...")
        true_reward_model_name = self.config.evaluation.true_reward_model
        self.true_reward_tokenizer = RobertaTokenizer.from_pretrained(true_reward_model_name)
        self.true_reward_model = AutoModelForSequenceClassification.from_pretrained(
            true_reward_model_name,
            torch_dtype=torch.float16
        ).to(self.device)
        
    def prepare_data(self, original_dataset_path, detoxified_dataset_path):
        """Prepare data for training by calling the utility function."""
        return prepare_data_util(
            original_dataset_path, 
            detoxified_dataset_path, 
            train_test_split=self.config.training.train_test_split
        )
    
    def data_loader(self, original_data, detoxified_data, batch_size):
        """Create batches of paired data."""
        assert len(original_data) == len(detoxified_data), "Both datasets should have the same length"
        
        indices = np.arange(len(original_data))
        np.random.shuffle(indices)
        
        # Whether to include prompt with output or just use output
        use_prompt = self.config.training.get('include_prompt', False)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_original = [original_data[idx] for idx in batch_indices]
            batch_detoxified = [detoxified_data[idx] for idx in batch_indices]
            
            yield batch_original, batch_detoxified
            
    def evaluate(self, data, split="test"):
        """Evaluate the reward model on the given data split."""
        self.reward_model.eval()
        self.true_reward_model.eval()
        
        original_outputs = []
        detoxified_outputs = []
        ground_truth_labels = []  # 1 for original (toxic), 0 for detoxified (non-toxic)
        true_rewards = []
        all_texts = []
        
        # Process in batches
        batch_size = self.config.training.batch_size
        
        # Determine whether to use prompt+output or just output based on config
        use_prompt = self.config.training.get('include_prompt', False)
        
        with torch.no_grad():
            # Process original (toxic) examples in batches
            for i in range(0, len(data['original']), batch_size):
                batch = data['original'][i:i+batch_size]
                
                # Get text based on config choice
                if use_prompt and 'prompt' in batch[0]:
                    texts = [item['prompt'] + item['output'] for item in batch]
                else:
                    texts = [item['output'] for item in batch]
                
                all_texts.extend(texts)
                
                # Tokenize
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.training.max_length
                )
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get learned rewards
                rewards = self.reward_model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.squeeze().cpu().tolist()
                
                # Handle single item case
                if not isinstance(rewards_list, list):
                    rewards_list = [rewards_list]
                
                original_outputs.extend(rewards_list)
                
                # Add ground truth labels
                ground_truth_labels.extend([1] * len(batch))
            
            # Process detoxified examples in batches
            for i in range(0, len(data['detoxified']), batch_size):
                batch = data['detoxified'][i:i+batch_size]
                
                # Get text based on config choice
                if use_prompt and 'prompt' in batch[0]:
                    texts = [item['prompt'] + item['output'] for item in batch]
                else:
                    texts = [item['output'] for item in batch]
                
                all_texts.extend(texts)
                
                # Tokenize
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.training.max_length
                )
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get learned rewards
                rewards = self.reward_model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.squeeze().cpu().tolist()
                
                # Handle single item case
                if not isinstance(rewards_list, list):
                    rewards_list = [rewards_list]
                
                detoxified_outputs.extend(rewards_list)
                
                # Add ground truth labels
                ground_truth_labels.extend([0] * len(batch))
        
        # Compute true rewards using the ground truth model
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            
            # Tokenize for the true reward model
            inputs = self.true_reward_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.training.max_length
            ).to(self.device)
            
            # Get true rewards
            logits = self.true_reward_model(**inputs).logits
            
            # Use the first logit (non-toxic) as the reward
            batch_rewards = logits[:, 0].cpu().tolist()
            true_rewards.extend(batch_rewards)
        
        # Get all outputs together
        all_outputs = original_outputs + detoxified_outputs
        
        # Compute metrics
        metrics = {}
        
        # Convert learned rewards to binary predictions
        # Higher reward should indicate less toxic (more detoxified)
        threshold = np.mean(all_outputs)  # Simple threshold
        learned_predictions = (np.array(all_outputs) > threshold).astype(int)
        learned_predictions = 1 - learned_predictions  # Invert to match ground truth (1=toxic)
        
        # Accuracy
        metrics[f'{split}_accuracy'] = accuracy_score(ground_truth_labels, learned_predictions)
        
        # F1 Score
        metrics[f'{split}_f1'] = f1_score(ground_truth_labels, learned_predictions)
        
        # AUC-ROC
        metrics[f'{split}_auc_roc'] = roc_auc_score(ground_truth_labels, [-x for x in all_outputs])  # Invert for ROC
        
        # Correlation with true rewards
        metrics[f'{split}_pearson_correlation'] = np.corrcoef([x for x in all_outputs], true_rewards)[0, 1]
        metrics[f'{split}_spearman_correlation'] = stats.spearmanr([x for x in all_outputs], true_rewards).correlation
        metrics[f'{split}_kendall_tau'] = stats.kendalltau([x for x in all_outputs], true_rewards).correlation
        
        # Average predicted rewards
        metrics[f'{split}_avg_original_reward'] = np.mean(original_outputs)
        metrics[f'{split}_avg_detoxified_reward'] = np.mean(detoxified_outputs)
        metrics[f'{split}_reward_diff'] = metrics[f'{split}_avg_detoxified_reward'] - metrics[f'{split}_avg_original_reward']
        
        # True reward model metrics
        # Convert true rewards to binary predictions
        true_threshold = np.mean(true_rewards)
        true_predictions = (np.array(true_rewards) > true_threshold).astype(int)
        true_predictions = 1 - true_predictions  # Invert to match ground truth (1=toxic)
        
        # Accuracy against true reward model predictions
        metrics[f'{split}_true_reward_accuracy'] = accuracy_score(true_predictions, learned_predictions)
        
        # F1 Score against true reward model predictions
        metrics[f'{split}_true_reward_f1'] = f1_score(true_predictions, learned_predictions)
        
        # Plot score distribution if it's the test set
        if split == "test":
            fig = plot_score_distribution(original_outputs, detoxified_outputs, self.eval_dir)
            
            # Log the plot to wandb
            if self.config.logging.use_wandb and wandb.run is not None:
                wandb.log({"score_distribution": wandb.Image(fig)})
        
        # Return metrics
        return metrics
    
    def train(self, train_data, test_data):
        """Train the reward model."""
        # Record model names for tracking
        original_model = self.config.dataset.original_model_name
        detoxified_model = self.config.dataset.detoxified_model_name
        reward_model_base = self.config.model.reward_model_base
        
        print(f"Training reward model to distinguish between outputs from:")
        print(f"  Original model: {original_model}")
        print(f"  Detoxified model: {detoxified_model}")
        print(f"Using reward model based on: {reward_model_base}")
        print(f"IRL method: {self.config.training.irl_method}")
        
        # Get loss function
        loss_fn = get_loss_function(
            method=self.config.training.irl_method,
            temperature=self.config.training.temperature,
            margin=self.config.training.margin,
            positive_penalty=self.config.training.get('positive_penalty', 1.0),
            negative_penalty=self.config.training.get('negative_penalty', 2.0),
            base_margin=self.config.training.get('base_margin', 0.1),
            confidence_factor=self.config.training.get('confidence_factor', 0.5)
        )
        
        # Initialize optimizer
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.reward_model.parameters()),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            eps=self.config.training.adam_epsilon
        )
        
        # Initialize AMP scaler for mixed precision training
        self.use_amp = getattr(self.config.training, 'use_amp', True)
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("Enabled Automatic Mixed Precision (AMP) for faster training")
        else:
            self.scaler = None
        
        # Optimize for maximum GPU utilization
        if torch.cuda.is_available():
            # Check if optimizations are disabled
            disable_optimizations = getattr(self.config.training, 'disable_gpu_optimizations', False)
            
            if disable_optimizations:
                print("GPU optimizations disabled by config")
                self.gradient_accumulation_steps = 1
            else:
                # Set memory fraction to use more GPU memory
                torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available GPU memory
                print(f"Set GPU memory fraction to 95%")
            
            # Enable memory efficient attention if available
            if hasattr(torch.backends, 'flash_attention_2'):
                torch.backends.flash_attention_2 = True
                print("Enabled Flash Attention 2 for memory efficiency")
            
            # Dynamic batch size optimization for maximum GPU utilization
            base_batch_size = self.config.training.batch_size
            max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            model_name = self.config.model.reward_model_base.lower()
            
            # Model-specific batch size scaling (some models are more memory-intensive)
            if 'llama' in model_name or 'llama-3' in model_name:
                # Llama models are more memory-intensive, be more conservative
                if max_memory >= 15:  # 15GB+ GPU
                    optimal_batch_size = min(base_batch_size * 2, 32)  # 2x instead of 4x
                elif max_memory >= 8:  # 8GB+ GPU
                    optimal_batch_size = min(base_batch_size * 1.5, 16)  # 1.5x instead of 2x
                else:
                    optimal_batch_size = base_batch_size
                print(f"Llama model detected - using conservative batch scaling")
            elif 'pythia' in model_name or 'gpt-neo' in model_name:
                # Pythia/GPT-Neo models are more memory-efficient
                if max_memory >= 15:  # 15GB+ GPU
                    optimal_batch_size = min(base_batch_size * 4, 64)  # 4x larger batch
                elif max_memory >= 8:  # 8GB+ GPU
                    optimal_batch_size = min(base_batch_size * 2, 32)  # 2x larger batch
                else:
                    optimal_batch_size = base_batch_size
                print(f"Pythia/GPT-Neo model detected - using aggressive batch scaling")
            else:
                # Default conservative scaling for unknown models
                if max_memory >= 15:  # 15GB+ GPU
                    optimal_batch_size = min(base_batch_size * 2, 32)  # Conservative 2x
                elif max_memory >= 8:  # 8GB+ GPU
                    optimal_batch_size = min(base_batch_size * 1.5, 16)  # Conservative 1.5x
                else:
                    optimal_batch_size = base_batch_size
                print(f"Unknown model type - using conservative batch scaling")
            
            print(f"GPU Memory: {max_memory:.1f}GB, Base batch size: {base_batch_size}")
            print(f"Using optimal batch size: {optimal_batch_size} for maximum GPU utilization")
            
            # Update config with optimal batch size
            self.config.training.batch_size = optimal_batch_size
            
            # Add gradient accumulation for even larger effective batch size
            if 'llama' in model_name or 'llama-3' in model_name:
                # Llama models: no accumulation (they're already memory-intensive)
                self.gradient_accumulation_steps = 1
                print(f"Llama model - no gradient accumulation (memory-intensive)")
            elif 'pythia' in model_name or 'gpt-neo' in model_name:
                # Pythia/GPT-Neo models: can use accumulation
                if max_memory >= 15:
                    self.gradient_accumulation_steps = 2  # Effective batch size = batch_size * 2
                else:
                    self.gradient_accumulation_steps = 1
                print(f"Pythia/GPT-Neo model - gradient accumulation enabled")
            else:
                # Default: conservative approach
                self.gradient_accumulation_steps = 1
                print(f"Unknown model - no gradient accumulation (conservative)")
            
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            print(f"Effective batch size: {optimal_batch_size * self.gradient_accumulation_steps}")
            
            # Print current GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory: {allocated_memory:.2f}GB / {total_memory:.2f}GB allocated")
        else:
            self.gradient_accumulation_steps = 1
        
        # Initialize wandb if needed - do this first before any logging
        if self.config.logging.use_wandb and wandb.run is not None:
            # Initialize but don't log anything yet
            pass
        
        # Evaluate at epoch 0 (before training) and log it as the first thing
        print("Evaluating at epoch 0 (before training)...")
        train_metrics = self.evaluate(train_data, split="train")
        test_metrics = self.evaluate(test_data, split="test")
        
        # Combine metrics
        metrics = {**train_metrics, **test_metrics}
        metrics['epoch'] = 0
        metrics['loss'] = 0.0  # No loss yet
        self.metrics_history.append(metrics)
        
        # Print metrics
        print(f"Metrics at epoch 0 (before training):")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
        
        # Log metrics to wandb as the very first thing
        if self.config.logging.use_wandb and wandb.run is not None:
            # Force the step counter to 0 and log
            wandb.define_metric("*", step_metric="epoch")
            wandb.log({"epoch": 0, **metrics})
        
        # Save checkpoint at epoch 0 if configured
        if self.config.output.save_checkpoints:
            print("Saving checkpoint at epoch 0...")
            self.save_checkpoint(0)
        
        # Training loop - now starting at epoch 1
        print(f"Starting training with {self.config.training.irl_method} IRL...")
        
        for epoch in range(1, self.config.training.epochs + 1):  # Changed to start at 1 and go to epochs+1
            self.reward_model.train()
            epoch_losses = []
            
            # Progress bar for batches - updated to show correct epoch
            progress_bar = tqdm(
                self.data_loader(
                    train_data['original'],
                    train_data['detoxified'],
                    self.config.training.batch_size
                ),
                desc=f"Epoch {epoch}/{self.config.training.epochs}"  # This now shows 1/30 instead of 1/30
            )
            
            # Process batches
            for batch_idx, (batch_original, batch_detoxified) in enumerate(progress_bar):
                # Zero gradients only at the start of accumulation cycle
                if batch_idx % self.gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                
                # Determine whether to use prompt+output or just output based on config
                use_prompt = self.config.training.get('include_prompt', False)
                
                # Get original outputs
                if use_prompt and 'prompt' in batch_original[0]:
                    original_texts = [item['prompt'] + item['output'] for item in batch_original]
                else:
                    original_texts = [item['output'] for item in batch_original]
                
                original_inputs = self.tokenizer(
                    original_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.training.max_length
                )
                # Move everything to the correct device
                original_inputs = {k: v.to(self.device) for k, v in original_inputs.items()}
                
                # Forward pass with AMP
                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        original_rewards = self.reward_model(**original_inputs)
                        
                        # Get detoxified outputs
                        if use_prompt and 'prompt' in batch_detoxified[0]:
                            detoxified_texts = [item['prompt'] + item['output'] for item in batch_detoxified]
                        else:
                            detoxified_texts = [item['output'] for item in batch_detoxified]
                        
                        detoxified_inputs = self.tokenizer(
                            detoxified_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=self.config.training.max_length
                        )
                        # Move everything to the correct device
                        detoxified_inputs = {k: v.to(self.device) for k, v in detoxified_inputs.items()}
                        
                        detoxified_rewards = self.reward_model(**detoxified_inputs)
                        
                        # Compute loss
                        loss = loss_fn(original_rewards, detoxified_rewards)
                else:
                    original_rewards = self.reward_model(**original_inputs)
                    
                    # Get detoxified outputs
                    if use_prompt and 'prompt' in batch_detoxified[0]:
                        detoxified_texts = [item['prompt'] + item['output'] for item in batch_detoxified]
                    else:
                        detoxified_texts = [item['output'] for item in batch_detoxified]
                    
                    detoxified_inputs = self.tokenizer(
                        detoxified_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.training.max_length
                    )
                    # Move everything to the correct device
                    detoxified_inputs = {k: v.to(self.device) for k, v in detoxified_inputs.items()}
                    
                    detoxified_rewards = self.reward_model(**detoxified_inputs)
                    
                    # Compute loss
                    loss = loss_fn(original_rewards, detoxified_rewards)
                
                # Check for NaN before backward
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN or Inf detected in loss. Skipping batch.")
                    continue
                
                # Backward pass with AMP and gradient accumulation
                if self.scaler is not None:
                    # Scale loss by accumulation steps
                    scaled_loss = loss / self.gradient_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    
                    # Only step optimizer at the end of accumulation cycle
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Add gradient clipping
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.reward_model.parameters(),
                            max_norm=self.config.training.grad_clip
                        )
                        
                        self.scaler.step(optimizer)
                        self.scaler.update()
                else:
                    # Scale loss by accumulation steps
                    scaled_loss = loss / self.gradient_accumulation_steps
                    scaled_loss.backward()
                    
                    # Only step optimizer at the end of accumulation cycle
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Add gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.reward_model.parameters(),
                            max_norm=self.config.training.grad_clip
                        )
                        
                        optimizer.step()
                
                epoch_losses.append(loss.item())
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Free up memory
                del original_inputs, detoxified_inputs
                torch.cuda.empty_cache()
            
            # Calculate average loss for epoch
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
            print(f"Epoch {epoch}/{self.config.training.epochs}, Loss: {avg_loss:.4f}")
            
            # Evaluate periodically
            if (epoch + 1) % self.config.training.eval_interval == 0 or epoch == self.config.training.epochs:
                print(f"Evaluating at epoch {epoch}...")
                train_metrics = self.evaluate(train_data, split="train")
                test_metrics = self.evaluate(test_data, split="test")
                
                # Combine metrics
                metrics = {**train_metrics, **test_metrics}
                metrics['epoch'] = epoch
                metrics['loss'] = avg_loss
                self.metrics_history.append(metrics)
                
                # Print metrics
                print(f"Metrics at epoch {epoch}:")
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                
                # Log metrics to wandb
                if self.config.logging.use_wandb and wandb.run is not None:
                    wandb.log({"epoch": epoch, **metrics})
                
                # Save checkpoint if configured (at same time as evaluation)
                if self.config.output.save_checkpoints:
                    print(f"Saving checkpoint at epoch {epoch}...")
                    self.save_checkpoint(epoch)
            else:
                # Log just the loss for non-evaluation epochs
                if self.config.logging.use_wandb and wandb.run is not None:
                    wandb.log({"loss": avg_loss, "epoch": epoch})
        
        # Save the final model
        self.save_model()
        
        return self.reward_model, self.metrics_history
        
    def save_checkpoint(self, epoch):
        """Save a model checkpoint."""
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.model_dir, f"checkpoint-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(checkpoint_dir, "model.pt")
        self.reward_model.save(model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save metrics
        metrics_path = os.path.join(checkpoint_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history[-1], f, indent=2)
        
        # Save training info
        info = {
            "model_name": self.config.model.reward_model_base,
            "original_model": self.config.dataset.original_model_name,
            "detoxified_model": self.config.dataset.detoxified_model_name,
            "irl_method": self.config.training.irl_method,
            "epoch": epoch,
            "timestamp": self.timestamp,
            "metrics": self.metrics_history[-1]
        }
        
        info_path = os.path.join(checkpoint_dir, "training_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        # Log to wandb if configured
        if self.config.logging.use_wandb and wandb.run is not None:
            # This will upload the files to wandb
            checkpoint_artifact = wandb.Artifact(
                f"model-checkpoint-{epoch}",
                type="model",
                metadata=self.metrics_history[-1]
            )
            checkpoint_artifact.add_dir(checkpoint_dir)
            wandb.log_artifact(checkpoint_artifact)
        
        # Push to hub if configured
        push_checkpoints_to_hub = getattr(self.config.output, 'push_checkpoints_to_hub', True)
        if self.config.output.push_to_hub and push_checkpoints_to_hub:
            hub_repo_id = push_to_hub(
                self.reward_model, 
                self.tokenizer, 
                self.config, 
                f"checkpoint-{epoch}"
            )
            
            if hub_repo_id:
                print(f"Checkpoint {epoch} pushed to HuggingFace: {hub_repo_id}")
        
        # Clean up local checkpoint directory
        import shutil
        shutil.rmtree(checkpoint_dir)
        print(f"Local checkpoint directory cleaned up: {checkpoint_dir}")
    
    def save_model(self):
        """Save the final model."""
        # Create model directory
        model_path = os.path.join(self.model_dir, "model.pt")
        
        # Save model
        self.reward_model.save(model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.model_dir)
        
        # Save all metrics
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Save config
        config_path = os.path.join(self.model_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.config))
        
        # Save training info
        info = {
            "model_name": self.config.model.reward_model_base,
            "original_model": self.config.dataset.original_model_name,
            "detoxified_model": self.config.dataset.detoxified_model_name,
            "irl_method": self.config.training.irl_method,
            "timestamp": self.timestamp,
            "final_metrics": self.metrics_history[-1] if self.metrics_history else None
        }
        
        info_path = os.path.join(self.model_dir, "training_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        # Plot metrics
        metrics_fig = plot_metrics(self.metrics_history, self.eval_dir)
        
        # Log to wandb if configured
        if self.config.logging.use_wandb and wandb.run is not None:
            # Log metrics plot
            wandb.log({"training_metrics": wandb.Image(metrics_fig)})
            
            # Log model as artifact
            model_artifact = wandb.Artifact(
                f"model-final",
                type="model",
                metadata=self.metrics_history[-1] if self.metrics_history else {}
            )
            model_artifact.add_dir(self.model_dir)
            wandb.log_artifact(model_artifact)
        
        # Push to HuggingFace Hub if configured
        if self.config.output.push_to_hub:
            hub_repo_id = push_to_hub(self.reward_model, self.tokenizer, self.config, "final")
            
            if hub_repo_id:
                print(f"Final model pushed to HuggingFace: {hub_repo_id}")
        
        print(f"Training complete. Model saved to {self.model_dir}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train_irl(cfg: DictConfig) -> None:
    """Main training function."""
    # Get the IRL config
    config = cfg.irl if hasattr(cfg, 'irl') else cfg
    
    print(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Set up wandb if enabled
    if config.logging.use_wandb:
        original_model = config.dataset.original_model_name.split('/')[-1]
        detoxified_model = config.dataset.detoxified_model_name.split('/')[-1]
        run_name = f"irl_{config.training.irl_method}_{original_model}_to_{detoxified_model}"
        
        wandb.init(
            project=config.logging.project_name,
            name=run_name,
            config=OmegaConf.to_container(config, resolve=True),
            mode=config.logging.wandb_mode
        )
    
    # Initialize trainer
    trainer = IRLTrainer(config)
    
    # Prepare data
    train_data, test_data = trainer.prepare_data(
        config.dataset.original_dataset_path,
        config.dataset.detoxified_dataset_path
    )
    
    # Train model
    reward_model, metrics_history = trainer.train(train_data, test_data)
    
    # Save model and metrics
    trainer.save_model()
    
    # Finish wandb run
    if config.logging.use_wandb and wandb.run is not None:
        wandb.finish()
    
    # Print final metrics
    print("\nTraining complete!")
    if metrics_history:
        print("\nFinal Metrics:")
        final_metrics = metrics_history[-1]
        for k, v in final_metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
    
    return metrics_history[-1] if metrics_history else None


if __name__ == "__main__":
    train_irl()