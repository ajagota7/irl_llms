"""
Optimized IRL training script with GPU memory optimizations and multi-device support.
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
import gc

from src.irl_utilities import (
    RewardModel, 
    get_loss_function, 
    plot_metrics, 
    plot_score_distribution, 
    push_to_hub,
    prepare_data as prepare_data_util
)


class OptimizedIRLTrainer:
    """Optimized trainer class for IRL detoxification with GPU memory optimizations."""

    def __init__(self, config: DictConfig):
        """Initialize the trainer with optimizations."""
        self.config = config
        
        # Multi-device setup
        self.devices = self._setup_devices()
        self.main_device = self.devices[0]  # Primary device for model
        
        # Memory optimization settings
        self.gradient_accumulation_steps = config.training.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.training.get('max_grad_norm', 1.0)
        self.use_amp = config.training.get('use_amp', True)  # Automatic Mixed Precision
        self.use_gradient_checkpointing = config.training.get('use_gradient_checkpointing', False)
        
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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)
        
        # Initialize the reward model and tokenizer
        self._init_model_and_tokenizer()
        
        # Initialize the true reward model for evaluation
        self._init_true_reward_model()
        
        # Initialize metrics history
        self.metrics_history = []
        
        # Initialize AMP scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"Initialized trainer with {len(self.devices)} devices: {self.devices}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Using AMP: {self.use_amp}")
        print(f"Using gradient checkpointing: {self.use_gradient_checkpointing}")
    
    def _setup_devices(self):
        """Setup available devices for training."""
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(num_devices)]
            print(f"Found {num_devices} CUDA devices: {devices}")
            
            # Print device info
            for i, device in enumerate(devices):
                props = torch.cuda.get_device_properties(i)
                print(f"Device {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f}GB")
            
            return devices
        else:
            return ["cpu"]
    
    def _init_model_and_tokenizer(self):
        """Initialize the reward model and tokenizer with optimizations."""
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
            device=self.main_device,
            num_unfrozen_layers=self.config.model.num_unfrozen_layers
        )
        
        # Enable gradient checkpointing if requested
        if self.use_gradient_checkpointing:
            self.reward_model.model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing")
        
        # Move to main device
        self.reward_model = self.reward_model.to(self.main_device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # Print model info
        total_params = sum(p.numel() for p in self.reward_model.parameters())
        trainable_params = sum(p.numel() for p in self.reward_model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        # Estimate memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def _init_true_reward_model(self):
        """Initialize the true reward model for evaluation."""
        print("Loading true reward model for evaluation...")
        true_reward_model_name = self.config.evaluation.true_reward_model
        self.true_reward_tokenizer = RobertaTokenizer.from_pretrained(true_reward_model_name)
        self.true_reward_model = AutoModelForSequenceClassification.from_pretrained(
            true_reward_model_name,
            torch_dtype=torch.float16 if self.config.model.use_half_precision else torch.float32
        ).to(self.main_device)
        
    def prepare_data(self, original_dataset_path, detoxified_dataset_path):
        """Prepare data for training by calling the utility function."""
        return prepare_data_util(
            original_dataset_path, 
            detoxified_dataset_path, 
            train_test_split=self.config.training.train_test_split
        )
    
    def data_loader(self, original_data, detoxified_data, batch_size):
        """Create batches of paired data with memory optimization."""
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
            
            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def evaluate(self, data, split="test"):
        """Evaluate the reward model on the given data split with memory optimization."""
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
                inputs = {k: v.to(self.main_device) for k, v in inputs.items()}
                
                # Get learned rewards with AMP
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        rewards = self.reward_model(**inputs)
                else:
                    rewards = self.reward_model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.squeeze().cpu().tolist()
                
                # Handle single item case
                if not isinstance(rewards_list, list):
                    rewards_list = [rewards_list]
                
                original_outputs.extend(rewards_list)
                
                # Add ground truth labels
                ground_truth_labels.extend([1] * len(batch))
                
                # Clear memory
                del inputs, rewards
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
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
                inputs = {k: v.to(self.main_device) for k, v in inputs.items()}
                
                # Get learned rewards with AMP
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        rewards = self.reward_model(**inputs)
                else:
                    rewards = self.reward_model(**inputs)
                
                # Convert to list of floats
                rewards_list = rewards.squeeze().cpu().tolist()
                
                # Handle single item case
                if not isinstance(rewards_list, list):
                    rewards_list = [rewards_list]
                
                detoxified_outputs.extend(rewards_list)
                
                # Add ground truth labels
                ground_truth_labels.extend([0] * len(batch))
                
                # Clear memory
                del inputs, rewards
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate metrics
        metrics = self._calculate_metrics(original_outputs, detoxified_outputs, ground_truth_labels, split)
        
        return metrics
    
    def _calculate_metrics(self, original_outputs, detoxified_outputs, ground_truth_labels, split):
        """Calculate evaluation metrics."""
        # Convert learned rewards to binary predictions
        all_scores = original_outputs + detoxified_outputs
        threshold = np.mean(all_scores)  # Simple threshold
        learned_predictions = (np.array(all_scores) > threshold).astype(int)
        learned_predictions = 1 - learned_predictions  # Invert to match ground truth (1=toxic)
        
        # Calculate metrics
        metrics = {
            f'{split}_accuracy': accuracy_score(ground_truth_labels, learned_predictions),
            f'{split}_f1': f1_score(ground_truth_labels, learned_predictions),
            f'{split}_auc_roc': roc_auc_score(ground_truth_labels, [-x for x in all_scores]),
            f'{split}_avg_original_reward': np.mean(original_outputs),
            f'{split}_avg_detoxified_reward': np.mean(detoxified_outputs),
            f'{split}_reward_diff': np.mean(detoxified_outputs) - np.mean(original_outputs),
            f'{split}_std_original_reward': np.std(original_outputs),
            f'{split}_std_detoxified_reward': np.std(detoxified_outputs),
            f'{split}_pearson_correlation': stats.pearsonr(original_outputs, detoxified_outputs)[0],
            f'{split}_spearman_correlation': stats.spearmanr(original_outputs, detoxified_outputs)[0],
            f'{split}_kendall_tau': stats.kendalltau(original_outputs, detoxified_outputs)[0]
        }
        
        return metrics
    
    def train(self, train_data, test_data):
        """Train the reward model with optimizations."""
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
        
        # Initialize wandb if needed
        if self.config.logging.use_wandb and wandb.run is not None:
            pass
        
        # Evaluate at epoch 0 (before training)
        print("Evaluating at epoch 0 (before training)...")
        train_metrics = self.evaluate(train_data, split="train")
        test_metrics = self.evaluate(test_data, split="test")
        
        # Combine metrics
        metrics = {**train_metrics, **test_metrics}
        metrics['epoch'] = 0
        metrics['loss'] = 0.0
        self.metrics_history.append(metrics)
        
        # Print metrics
        print(f"Metrics at epoch 0 (before training):")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
        
        # Log metrics to wandb
        if self.config.logging.use_wandb and wandb.run is not None:
            wandb.define_metric("*", step_metric="epoch")
            wandb.log({"epoch": 0, **metrics})
        
        # Training loop
        print(f"Starting training with {self.config.training.irl_method} IRL...")
        
        for epoch in range(1, self.config.training.epochs + 1):
            self.reward_model.train()
            epoch_losses = []
            
            # Progress bar for batches
            progress_bar = tqdm(
                self.data_loader(
                    train_data['original'],
                    train_data['detoxified'],
                    self.config.training.batch_size
                ),
                desc=f"Epoch {epoch}/{self.config.training.epochs}"
            )
            
            # Process batches with gradient accumulation
            optimizer.zero_grad()
            
            for batch_idx, (batch_original, batch_detoxified) in enumerate(progress_bar):
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
                original_inputs = {k: v.to(self.main_device) for k, v in original_inputs.items()}
                
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
                detoxified_inputs = {k: v.to(self.main_device) for k, v in detoxified_inputs.items()}
                
                # Forward pass with AMP
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        original_rewards = self.reward_model(**original_inputs)
                        detoxified_rewards = self.reward_model(**detoxified_inputs)
                        loss = loss_fn(original_rewards, detoxified_rewards)
                        loss = loss / self.gradient_accumulation_steps
                else:
                    original_rewards = self.reward_model(**original_inputs)
                    detoxified_rewards = self.reward_model(**detoxified_inputs)
                    loss = loss_fn(original_rewards, detoxified_rewards)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_losses.append(loss.item() * self.gradient_accumulation_steps)
                
                # Gradient accumulation step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.max_grad_norm)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Clear memory
                del original_inputs, detoxified_inputs, original_rewards, detoxified_rewards, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Final gradient step if needed
            if len(epoch_losses) % self.gradient_accumulation_steps != 0:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
            
            # Evaluate
            print(f"\nEvaluating at epoch {epoch}...")
            train_metrics = self.evaluate(train_data, split="train")
            test_metrics = self.evaluate(test_data, split="test")
            
            # Combine metrics
            metrics = {**train_metrics, **test_metrics}
            metrics['epoch'] = epoch
            metrics['loss'] = np.mean(epoch_losses)
            self.metrics_history.append(metrics)
            
            # Print metrics
            print(f"Metrics at epoch {epoch}:")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
            
            # Log to wandb
            if self.config.logging.use_wandb and wandb.run is not None:
                wandb.log({"epoch": epoch, **metrics})
            
            # Save checkpoint
            if epoch % self.config.training.save_every == 0:
                self.save_checkpoint(epoch)
            
            # Print memory usage
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f}GB reserved")
        
        # Save final model
        self.save_model()
        
        print(f"Training complete. Model saved to {self.model_dir}")
    
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
            checkpoint_artifact = wandb.Artifact(
                f"model-checkpoint-{epoch}",
                type="model",
                metadata=self.metrics_history[-1]
            )
            checkpoint_artifact.add_dir(checkpoint_dir)
            wandb.log_artifact(checkpoint_artifact)
        
        # Push to hub if configured
        if self.config.output.push_to_hub and epoch == self.config.training.epochs:
            hub_repo_id = push_to_hub(
                self.reward_model, 
                self.tokenizer, 
                self.config, 
                f"checkpoint-{epoch}"
            )
            
            if hub_repo_id:
                print(f"Checkpoint pushed to HuggingFace: {hub_repo_id}")
    
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
            wandb.log({"training_metrics": wandb.Image(metrics_fig)})
            
            model_artifact = wandb.Artifact(
                f"model-final",
                type="model",
                metadata=self.metrics_history[-1] if self.metrics_history else {}
            )
            model_artifact.add_dir(self.model_dir)
            wandb.log_artifact(model_artifact)
        
        # Push to HuggingFace Hub if configured
        if self.config.output.push_to_hub:
            hub_repo_id = push_to_hub(self.reward_model, self.tokenizer, self.config)
            
            if hub_repo_id:
                print(f"Model pushed to HuggingFace: {hub_repo_id}")
        
        print(f"Training complete. Model saved to {self.model_dir}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train_irl_optimized(cfg: DictConfig) -> None:
    """Main training function with optimizations."""
    # Initialize trainer
    trainer = OptimizedIRLTrainer(cfg)
    
    # Prepare data
    train_data, test_data = trainer.prepare_data(
        cfg.dataset.original_dataset_path,
        cfg.dataset.detoxified_dataset_path
    )
    
    # Train
    trainer.train(train_data, test_data)


if __name__ == "__main__":
    train_irl_optimized() 