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
    push_to_hub
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
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
        """Prepare data for training."""
        print(f"Loading datasets from: {original_dataset_path} and {detoxified_dataset_path}")
        
        # Check if paths are HuggingFace dataset IDs
        if not os.path.exists(original_dataset_path) and '/' in original_dataset_path:
            print(f"Loading original dataset from HuggingFace: {original_dataset_path}")
            original_data = load_dataset(original_dataset_path)
            if 'train' in original_data:
                original_data = original_data['train']
        else:
            # Load from local file
            with open(original_dataset_path, 'r') as f:
                original_data = json.load(f)
        
        if not os.path.exists(detoxified_dataset_path) and '/' in detoxified_dataset_path:
            print(f"Loading detoxified dataset from HuggingFace: {detoxified_dataset_path}")
            detoxified_data = load_dataset(detoxified_dataset_path)
            if 'train' in detoxified_data:
                detoxified_data = detoxified_data['train']
        else:
            # Load from local file
            with open(detoxified_dataset_path, 'r') as f:
                detoxified_data = json.load(f)
        
        # Verify data lengths match
        if len(original_data) != len(detoxified_data):
            print("Warning: Dataset lengths don't match!")
            # Use the smaller length
            min_len = min(len(original_data), len(detoxified_data))
            original_data = original_data[:min_len]
            detoxified_data = detoxified_data[:min_len]
            
        print(f"Loaded {len(original_data)} paired samples")
        
        # Split data into train/test sets
        train_size = int(self.config.training.train_test_split * len(original_data))
        
        train_data = {
            'original': original_data[:train_size],
            'detoxified': detoxified_data[:train_size]
        }
        
        test_data = {
            'original': original_data[train_size:],
            'detoxified': detoxified_data[train_size:]
        }
        
        print(f"Training set: {len(train_data['original'])} samples")
        print(f"Test set: {len(test_data['original'])} samples")
        
        return train_data, test_data
    
    def data_loader(self, original_data, detoxified_data, batch_size):
        """Create batches of paired data."""
        assert len(original_data) == len(detoxified_data), "Both datasets should have the same length"
        
        indices = np.arange(len(original_data))
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_original = [original_data[idx] for idx in batch_indices]
            batch_detoxified = [detoxified_data[idx] for idx in batch_indices]
            
            yield batch_original, batch_detoxified
            
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
            margin=self.config.training.margin
        )
        
        # Initialize optimizer
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.reward_model.parameters()),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            eps=self.config.training.adam_epsilon
        )
        
        # Training loop
        print(f"Starting training with {self.config.training.irl_method} IRL...")
        self.metrics_history = []
        
        for epoch in range(self.config.training.epochs):
            self.reward_model.train()
            epoch_losses = []
            
            # Progress bar for batches
            progress_bar = tqdm(
                self.data_loader(
                    train_data['original'],
                    train_data['detoxified'],
                    self.config.training.batch_size
                ),
                desc=f"Epoch {epoch+1}/{self.config.training.epochs}"
            )
            
            # Process batches
            for batch_original, batch_detoxified in progress_bar:
                optimizer.zero_grad()
                
                # Get original outputs
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
                
                original_rewards = self.reward_model(**original_inputs)
                
                # Get detoxified outputs
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
                
                # Backward pass
                loss.backward()
                
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
            print(f"Epoch {epoch+1}/{self.config.training.epochs}, Loss: {avg_loss:.4f}")
            
            # Log loss to wandb
            if self.config.logging.use_wandb and wandb.run is not None:
                wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})
            
            # Evaluate periodically
            if (epoch + 1) % self.config.training.eval_interval == 0 or epoch == self.config.training.epochs - 1:
                print(f"Evaluating at epoch {epoch+1}...")
                metrics = self.evaluate(test_data)
                
                # Add epoch and loss to metrics
                metrics['epoch'] = epoch + 1
                metrics['loss'] = avg_loss
                self.metrics_history.append(metrics)
                
                # Print metrics
                print(f"Metrics at epoch {epoch+1}:")
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                
                # Log metrics to wandb
                if self.config.logging.use_wandb and wandb.run is not None:
                    wandb.log(metrics)
                
                # Save checkpoint if configured
                if self.config.output.save_checkpoints:
                    self.save_checkpoint(epoch + 1)
        
        # Save the final model
        self.save_model()
        
        return self.reward_model, self.metrics_history
        
    def evaluate(self, test_data):
        """Evaluate the reward model."""
        self.reward_model.eval()
        self.true_reward_model.eval()
        
        original_outputs = []
        detoxified_outputs = []
        ground_truth_labels = []  # 1 for original (toxic), 0 for detoxified (non-toxic)
        
        # Process in batches
        batch_size = self.config.training.batch_size
        
        with torch.no_grad():
            # Process original (toxic) examples in batches
            for i in range(0, len(test_data['original']), batch_size):
                batch = test_data['original'][i:i+batch_size]
                texts = [item['output'] for item in batch]
                
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
            for i in range(0, len(test_data['detoxified']), batch_size):
                batch = test_data['detoxified'][i:i+batch_size]
                texts = [item['output'] for item in batch]
                
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
        true_rewards = []
        all_texts = [test_data['original'][i]['output'] for i in range(len(test_data['original']))] + \
                    [test_data['detoxified'][i]['output'] for i in range(len(test_data['detoxified']))]
        
        # Process in batches
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
        metrics['accuracy'] = accuracy_score(ground_truth_labels, learned_predictions)
        
        # F1 Score
        metrics['f1'] = f1_score(ground_truth_labels, learned_predictions)
        
        # AUC-ROC
        metrics['auc_roc'] = roc_auc_score(ground_truth_labels, [-x for x in all_outputs])  # Invert for ROC
        
        # Correlation with true rewards
        metrics['pearson_correlation'] = np.corrcoef([x for x in all_outputs], true_rewards)[0, 1]
        metrics['spearman_correlation'] = stats.spearmanr([x for x in all_outputs], true_rewards).correlation
        metrics['kendall_tau'] = stats.kendalltau([x for x in all_outputs], true_rewards).correlation
        
        # Average predicted rewards
        metrics['avg_original_reward'] = np.mean(original_outputs)
        metrics['avg_detoxified_reward'] = np.mean(detoxified_outputs)
        metrics['reward_diff'] = metrics['avg_detoxified_reward'] - metrics['avg_original_reward']
        
        # Plot score distribution
        fig = plot_score_distribution(original_outputs, detoxified_outputs, self.eval_dir)
        
        # Log the plot to wandb
        if self.config.logging.use_wandb and wandb.run is not None:
            wandb.log({"score_distribution": wandb.Image(fig)})
        
        # Return metrics
        return metrics
    
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
            hub_repo_id = push_to_hub(self.reward_model, self.tokenizer, self.config)
            
            if hub_repo_id:
                print(f"Model pushed to HuggingFace: {hub_repo_id}")
        
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