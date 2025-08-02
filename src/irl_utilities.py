"""
Utility functions and classes for IRL detoxification.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset


class RewardModel(torch.nn.Module):
    """Reward model that predicts whether text is toxic or not."""

    def __init__(self, model_name, use_half_precision=False, device="cuda", num_unfrozen_layers=1):
        """Initialize the reward model with a value head on top of a language model."""
        super().__init__()

        # Set up device and precision
        self.device = device
        self.device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        self.use_half_precision = use_half_precision
        self.model_name = model_name

        # Load the base LM with the appropriate precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_half_precision else None,
        ).to(self.device)

        # Add a value head with careful initialization
        self.v_head = torch.nn.Linear(self.model.config.hidden_size, 1, bias=False).to(self.device)
        # Initialize with small values to avoid NaN issues
        self.v_head.weight.data.normal_(mean=0.0, std=0.01)

        # Freeze the base model
        self._freeze_base_model(num_unfrozen_layers)

    def _freeze_base_model(self, num_unfrozen_layers=0):
        """Freeze the base model, except for the last few layers."""
        # First freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # For different model architectures, handle this differently
        if hasattr(self.model, 'transformer'):
            # For GPT-Neo and similar models
            layers = self.model.transformer.h
            if num_unfrozen_layers > 0:
                print(f"Unfreezing the last {num_unfrozen_layers} layers of the transformer.")
                for i in range(len(layers) - num_unfrozen_layers, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # For some newer models
            layers = self.model.model.layers
            if num_unfrozen_layers > 0:
                print(f"Unfreezing the last {num_unfrozen_layers} layers of the model.")
                for i in range(len(layers) - num_unfrozen_layers, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
        # Add Pythia model architecture support
        elif hasattr(self.model, 'gpt_neox'):
            # For Pythia models which use GPT-NeoX architecture
            layers = self.model.gpt_neox.layers
            if num_unfrozen_layers > 0:
                print(f"Unfreezing the last {num_unfrozen_layers} layers of Pythia model.")
                for i in range(len(layers) - num_unfrozen_layers, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
        else:
            print("Unsupported model architecture. All parameters frozen except value head.")
        
        # Always unfreeze the value head - FIXED THIS LINE
        for param in self.v_head.parameters():
            param.requires_grad = True
            
        # Calculate trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")
        
        # Debug: Show what's actually trainable
        if trainable_params > 0:
            print("Trainable components:")
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(f"  - {name}: {param.numel():,} parameters")
        else:
            print("WARNING: No trainable parameters found! Only the value head should be trainable.")

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model."""
        # Make sure inputs are on the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Use autocast for mixed precision if needed
        with torch.amp.autocast('cuda', enabled=self.use_half_precision):
            # Get the hidden states from the base model
            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Use the last hidden state

            # Use mean pooling for more stable representations
            if attention_mask is not None:
                # Expand attention mask to match hidden state dimensions
                expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                # Apply mask and get sum
                masked_hidden = hidden_states * expanded_mask
                sum_hidden = torch.sum(masked_hidden, dim=1)
                # Get token count (avoid division by zero)
                token_count = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1.0)
                # Mean pooling
                pooled_hidden = sum_hidden / token_count
                # Apply value head
                values = self.v_head(pooled_hidden)
            else:
                # Fallback to last token if no mask
                last_token_indices = torch.tensor([input_ids.size(1)-1] * input_ids.size(0), device=input_ids.device)
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                last_hidden_states = hidden_states[batch_indices, last_token_indices]
                values = self.v_head(last_hidden_states)

            return values
        
        # Clear intermediate tensors to save memory
        del outputs, hidden_states
        if attention_mask is not None:
            del expanded_mask, masked_hidden, sum_hidden, token_count, pooled_hidden
        else:
            del last_token_indices, batch_indices, last_hidden_states

    def save(self, path):
        """Save the model to disk."""
        state_dict = {
            'v_head': self.v_head.state_dict(),
            'config': {
                'model_name': self.model_name,
                'use_half_precision': self.use_half_precision
            }
        }
        torch.save(state_dict, path)

    @classmethod
    def load(cls, path, device="cuda"):
        """Load the model from disk."""
        state_dict = torch.load(path, map_location=device)
        config = state_dict['config']

        # Create a new model
        model = cls(config['model_name'],
                   use_half_precision=config['use_half_precision'],
                   device=device,
                   num_unfrozen_layers=0)  # Load with all frozen for inference

        # Load v_head
        model.v_head.load_state_dict(state_dict['v_head'])

        return model

    @classmethod
    def load_from_hf_model(cls, model_path, device="cuda"):
        """
        Load a reward model from a HuggingFace model directory.
        
        Args:
            model_path: Path to the model directory
            device: Device to load the model on
            
        Returns:
            RewardModel instance
        """
        import os
        from transformers import AutoModelForCausalLM
        
        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
        
        # Create a new reward model
        reward_model = cls(model_name=model_path, device=device)
        reward_model.model = model
        
        # Load the value head if it exists
        v_head_path = os.path.join(model_path, "v_head.pt")
        if os.path.exists(v_head_path):
            try:
                # Try to load the value head directly
                reward_model.v_head.load_state_dict(torch.load(v_head_path, map_location=device))
                print(f"Successfully loaded value head from {v_head_path}")
            except Exception as e:
                print(f"Error loading value head from {v_head_path}: {e}")
                # Try to load as a complete state dict
                try:
                    state_dict = torch.load(v_head_path, map_location=device)
                    reward_model.v_head = torch.nn.Linear(model.config.hidden_size, 1)
                    reward_model.v_head.to(device)
                    reward_model.v_head.weight.data = state_dict['weight']
                    reward_model.v_head.bias.data = state_dict['bias']
                    print(f"Successfully loaded value head weights directly")
                except Exception as e2:
                    print(f"Error loading value head weights: {e2}")
        
        return reward_model


# IRL method implementations
def max_margin_loss(original_rewards, detoxified_rewards, margin=0.1):
    """
    Compute max-margin loss.
    
    Args:
        original_rewards: Rewards for original (toxic) samples
        detoxified_rewards: Rewards for detoxified (non-toxic) samples
        margin: Minimum margin between rewards
        
    Returns:
        Loss value
    """
    # We want detoxified_rewards > original_rewards + margin
    reward_diff = detoxified_rewards - original_rewards
    loss = torch.clamp(margin - reward_diff, min=0)

    # Check for NaN and replace with zeros
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return loss.mean()


def asymmetric_margin_loss(original_rewards, detoxified_rewards, 
                          positive_penalty=1.0, negative_penalty=2.0):
    """
    Asymmetric max-margin loss with different penalties for each direction.
    
    Args:
        original_rewards: Rewards for original (toxic) samples
        detoxified_rewards: Rewards for detoxified (non-toxic) samples
        positive_penalty: Penalty weight when detoxified > original (good case)
        negative_penalty: Penalty weight when original > detoxified (bad case)
        
    Returns:
        Loss value
    """
    diff = detoxified_rewards - original_rewards
    
    # Apply different penalties based on direction
    loss = torch.where(
        diff > 0,
        -positive_penalty * diff,  # Good case: detoxified > original
        -negative_penalty * diff   # Bad case: original > detoxified
    )
    
    # Check for NaN and replace with zeros
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return loss.mean()


def confidence_margin_loss(original_rewards, detoxified_rewards, 
                          base_margin=0.1, confidence_factor=0.5):
    """
    Confidence-based margin loss where margin increases with prediction confidence.
    
    Args:
        original_rewards: Rewards for original (toxic) samples
        detoxified_rewards: Rewards for detoxified (non-toxic) samples
        base_margin: Base margin value
        confidence_factor: Factor to scale the dynamic margin
        
    Returns:
        Loss value
    """
    diff = detoxified_rewards - original_rewards
    # Margin increases with the magnitude of the difference
    dynamic_margin = base_margin + confidence_factor * torch.abs(diff)
    loss = torch.clamp(dynamic_margin - diff, min=0)
    
    # Check for NaN and replace with zeros
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return loss.mean()


def max_entropy_loss(original_rewards, detoxified_rewards, temperature=0.1):
    """
    Compute Maximum Entropy IRL loss.

    In MaxEnt IRL, we model P(trajectory) ∝ exp(reward(trajectory)/temperature)
    For our case with two trajectories, we want to maximize:
    P(detoxified) = exp(detoxified_reward/T) / [exp(original_reward/T) + exp(detoxified_reward/T)]

    Args:
        original_rewards: Rewards for original (toxic) samples
        detoxified_rewards: Rewards for detoxified (non-toxic) samples
        temperature: Temperature parameter for softmax (controls entropy)

    Returns:
        Loss value (negative log likelihood)
    """
    # Calculate the partition function (normalizing constant)
    # We're considering only two trajectories per prompt, so Z = exp(r_orig/T) + exp(r_detox/T)
    Z = torch.exp(original_rewards/temperature) + torch.exp(detoxified_rewards/temperature)

    # Calculate the log likelihood of the detoxified trajectory
    # log P(detox) = r_detox/T - log(Z)
    log_likelihood = detoxified_rewards/temperature - torch.log(Z)

    # Maximize log likelihood (or minimize negative log likelihood)
    loss = -log_likelihood.mean()

    # Check for NaN and replace with zeros
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("Warning: NaN or Inf detected in MaxEnt loss calculation")
        loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.tensor(1.0, device=loss.device), loss)

    return loss


def get_loss_function(method="max_margin", **kwargs):
    """
    Get the appropriate loss function based on the specified method.
    
    Args:
        method: The IRL method to use ("max_margin", "asymmetric_margin", "confidence_margin", or "max_entropy")
        **kwargs: Additional arguments to pass to the loss function
        
    Returns:
        A loss function that takes original_rewards and detoxified_rewards
    """
    if method == "asymmetric_margin":
        positive_penalty = kwargs.get("positive_penalty", 1.0)
        negative_penalty = kwargs.get("negative_penalty", 2.0)
        return lambda orig, detox: asymmetric_margin_loss(orig, detox, positive_penalty, negative_penalty)
    elif method == "confidence_margin":
        base_margin = kwargs.get("base_margin", 0.1)
        confidence_factor = kwargs.get("confidence_factor", 0.5)
        return lambda orig, detox: confidence_margin_loss(orig, detox, base_margin, confidence_factor)
    elif method == "max_entropy":
        temperature = kwargs.get("temperature", 0.1)
        return lambda orig, detox: max_entropy_loss(orig, detox, temperature)
    else:  # Default to max_margin
        margin = kwargs.get("margin", 0.1)
        return lambda orig, detox: max_margin_loss(orig, detox, margin)


# Visualization functions
def plot_metrics(metrics_history, output_dir=None):
    """Plot training metrics history."""
    if not metrics_history:
        print("No metrics to plot")
        return

    epochs_list = [m['epoch'] for m in metrics_history]

    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))

    # Plot train vs test accuracy
    ax1 = axes[0, 0]
    if 'train_accuracy' in metrics_history[0] and 'test_accuracy' in metrics_history[0]:
        ax1.plot(epochs_list, [m['train_accuracy'] for m in metrics_history], 'o-', label='Train Accuracy')
        ax1.plot(epochs_list, [m['test_accuracy'] for m in metrics_history], 's-', label='Test Accuracy')
        ax1.plot(epochs_list, [m['train_f1'] for m in metrics_history], '^-', label='Train F1')
        ax1.plot(epochs_list, [m['test_f1'] for m in metrics_history], 'v-', label='Test F1')
    else:
        # Backward compatibility
        ax1.plot(epochs_list, [m['accuracy'] for m in metrics_history], 'o-', label='Accuracy')
        ax1.plot(epochs_list, [m['f1'] for m in metrics_history], 's-', label='F1 Score')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot AUC-ROC
    ax2 = axes[0, 1]
    if 'train_auc_roc' in metrics_history[0] and 'test_auc_roc' in metrics_history[0]:
        ax2.plot(epochs_list, [m['train_auc_roc'] for m in metrics_history], 'o-', label='Train AUC-ROC')
        ax2.plot(epochs_list, [m['test_auc_roc'] for m in metrics_history], 's-', label='Test AUC-ROC')
    else:
        # Backward compatibility
        ax2.plot(epochs_list, [m['auc_roc'] for m in metrics_history], 'o-', label='AUC-ROC')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('AUC-ROC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot correlations
    ax3 = axes[1, 0]
    if 'train_pearson_correlation' in metrics_history[0]:
        ax3.plot(epochs_list, [m['train_pearson_correlation'] for m in metrics_history], 'o-', label='Train Pearson')
        ax3.plot(epochs_list, [m['test_pearson_correlation'] for m in metrics_history], 's-', label='Test Pearson')
    else:
        # Backward compatibility
        ax3.plot(epochs_list, [m['pearson_correlation'] for m in metrics_history], 'o-', label='Pearson')
        ax3.plot(epochs_list, [m['spearman_correlation'] for m in metrics_history], 's-', label='Spearman')
        ax3.plot(epochs_list, [m['kendall_tau'] for m in metrics_history], '^-', label='Kendall Tau')
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Correlation')
    ax3.set_title('Correlation with True Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot average rewards
    ax4 = axes[1, 1]
    if 'test_avg_original_reward' in metrics_history[0]:
        ax4.plot(epochs_list, [m['test_avg_original_reward'] for m in metrics_history], 'r-', label='Original (Toxic)')
        ax4.plot(epochs_list, [m['test_avg_detoxified_reward'] for m in metrics_history], 'g-', label='Detoxified')
        ax4.plot(epochs_list, [m['test_reward_diff'] for m in metrics_history], 'b--', label='Difference')
    else:
        # Backward compatibility
        ax4.plot(epochs_list, [m['avg_original_reward'] for m in metrics_history], 'r-', label='Original (Toxic)')
        ax4.plot(epochs_list, [m['avg_detoxified_reward'] for m in metrics_history], 'g-', label='Detoxified')
        ax4.plot(epochs_list, [m['reward_diff'] for m in metrics_history], 'b--', label='Difference')
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Average Predicted Rewards')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot loss
    ax5 = axes[2, 0]
    ax5.plot(epochs_list, [m['loss'] for m in metrics_history], 'o-')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Training Loss')
    ax5.grid(True, alpha=0.3)
    
    # Plot true reward model agreement
    ax6 = axes[2, 1]
    if 'train_true_reward_accuracy' in metrics_history[0]:
        ax6.plot(epochs_list, [m['train_true_reward_accuracy'] for m in metrics_history], 'o-', 
                label='Train True Reward Accuracy')
        ax6.plot(epochs_list, [m['test_true_reward_accuracy'] for m in metrics_history], 's-', 
                label='Test True Reward Accuracy')
        ax6.plot(epochs_list, [m['train_true_reward_f1'] for m in metrics_history], '^-', 
                label='Train True Reward F1')
        ax6.plot(epochs_list, [m['test_true_reward_f1'] for m in metrics_history], 'v-', 
                label='Test True Reward F1')
    else:
        ax6.text(0.5, 0.5, 'No true reward metrics available', 
                horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)
    
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Score')
    ax6.set_title('Agreement with True Reward Model')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        combined_path = os.path.join(output_dir, f'combined_metrics.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')

    # Return the figure for further use (e.g., logging to wandb)
    return fig


def plot_score_distribution(original_scores, detoxified_scores, output_dir=None):
    """Plot distribution of scores for toxic vs non-toxic content."""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot histograms
    ax.hist(original_scores, alpha=0.5, bins=20, label='Original (Toxic)', color='red')
    ax.hist(detoxified_scores, alpha=0.5, bins=20, label='Detoxified', color='green')

    # Plot means as vertical lines
    ax.axvline(np.mean(original_scores), color='red', linestyle='--',
              label=f'Mean Original: {np.mean(original_scores):.4f}')
    ax.axvline(np.mean(detoxified_scores), color='green', linestyle='--',
              label=f'Mean Detoxified: {np.mean(detoxified_scores):.4f}')

    # Add labels and title
    ax.set_xlabel('Reward Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Reward Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text with summary statistics
    diff = np.mean(detoxified_scores) - np.mean(original_scores)
    text = (f"Mean Difference: {diff:.4f}\n"
           f"Original Std: {np.std(original_scores):.4f}\n"
           f"Detoxified Std: {np.std(detoxified_scores):.4f}")
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save if path provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        dist_path = os.path.join(output_dir, f'score_distribution.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')

    # Return the figure for further use
    return fig


def push_to_hub(reward_model, tokenizer, config, checkpoint_suffix=None):
    """Push the model to the HuggingFace Hub using either specified hub_org or default username."""
    import os
    from huggingface_hub import HfApi
    from omegaconf import OmegaConf
    
    try:
        # Check if hub_org is provided in config
        hub_org = None
        if hasattr(config.output, 'hub_org') and config.output.hub_org:
            hub_org = config.output.hub_org
        
        # If hub_org is not provided, use default username
        if not hub_org:
            hub_org = "ajagota71"  # Default username if not specified
        
        # Create repository name from config or use default
        model_short_name = config.model.reward_model_base.split('/')[-1].lower()
        repo_prefix = getattr(config.output, 'repo_name_prefix', "irl-reward")
        
        if checkpoint_suffix:
            repo_name = f"{repo_prefix}-{model_short_name}-{checkpoint_suffix}"
        else:
            repo_name = f"{repo_prefix}-{model_short_name}"
        
        # Ensure repo name uses hyphens, not underscores
        repo_name = repo_name.replace('_', '-')
        
        # Construct full repository ID with username/organization
        repo_id = f"{hub_org}/{repo_name}"
        
        print(f"Pushing model to HuggingFace Hub: {repo_id}")
        
        # Create output directory
        output_dir = os.path.join(os.getcwd(), f"hf_upload_{repo_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        print(f"Saving model files to {output_dir}")
        reward_model.model.save_pretrained(output_dir)
        
        # Save the value head separately
        v_head_path = os.path.join(output_dir, "v_head.pt")
        torch.save(reward_model.v_head.state_dict(), v_head_path)
        
        # Save model configuration info
        model_info = {
            "model_type": "irl_reward_model",
            "base_model": config.model.reward_model_base,
            "value_head_size": reward_model.model.config.hidden_size
        }
        
        # Add value head info to config
        model_config_path = os.path.join(output_dir, "reward_model_config.json")
        with open(model_config_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        tokenizer.save_pretrained(output_dir)
        
        # Save config
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(config))
        
        # Create README with proper metadata
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(f"# {repo_name}\n\n")
            f.write(f"This model was trained using {config.training.irl_method} IRL to learn toxicity reward signals.\n\n")
            f.write(f"Base model: {config.model.reward_model_base}\n")
            f.write(f"Original model: {config.dataset.original_model_name}\n")
            f.write(f"Detoxified model: {config.dataset.detoxified_model_name}\n\n")
            # Proper metadata section
            f.write("---\n")
            f.write("language: en\n")
            f.write("tags:\n")
            f.write("- toxicity\n")
            f.write("- reward-model\n")
            f.write("- irl\n")
            f.write("library_name: transformers\n")
            f.write(f"base_model: {model_short_name}\n")
            f.write("pipeline_tag: text-classification\n")
            f.write("---\n")
        
        # Use HfApi to create repository and upload
        api = HfApi()
        
        # Create repository
        print(f"Creating repository: {repo_id}")
        try:
            api.create_repo(
                repo_id=repo_id,
                private=getattr(config.output, 'private', False),
                exist_ok=True
            )
            print(f"Repository created: {repo_id}")
        except Exception as e:
            print(f"Repository creation note: {e}")
        
        # Upload files
        print(f"Uploading files to {repo_id}")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            commit_message="Upload IRL reward model"
        )
        
        print(f"Successfully uploaded model to {repo_id}")
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(output_dir)
        print(f"Temporary upload directory cleaned up: {output_dir}")
        
        # Add message indicating whether it was uploaded to the explicitly set hub_org
        if hasattr(config.output, 'hub_org') and config.output.hub_org:
            print(f"Model pushed to HuggingFace: {repo_id} (using explicitly set hub_org)")
        else:
            print(f"Model pushed to HuggingFace: {repo_id} (using default username)")
            
        return repo_id
        
    except Exception as e:
        print(f"Error in push_to_hub: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_data(original_dataset_path, detoxified_dataset_path, train_test_split=0.8):
    """Prepare data for training."""
    print(f"Loading datasets from: {original_dataset_path} and {detoxified_dataset_path}")
    
    # Function to determine if a path is a HuggingFace dataset ID
    def is_hf_dataset(path):
        return '/' in path and not os.path.exists(path)
    
    # Function to load from HuggingFace with multiple fallback options
    def load_hf_dataset(dataset_path):
        """Load dataset from HuggingFace with multiple fallback options."""
        try:
            print(f"Loading dataset from HuggingFace: {dataset_path}")
            # Try loading with default format first
            ds = load_dataset(dataset_path)
            if isinstance(ds, dict) and 'train' in ds:
                ds = ds['train']
            return ds.to_pandas().to_dict('records')
        except Exception as e:
            print(f"Error loading with default format: {e}")
            try:
                # Try with just the split parameter
                print(f"Trying to load dataset with split parameter: {dataset_path}")
                ds = load_dataset(dataset_path, split='train')
                return ds.to_pandas().to_dict('records')
            except Exception as e:
                print(f"Error loading with split parameter: {e}")
                try:
                    # Try with streaming mode
                    print(f"Trying to load dataset in streaming mode: {dataset_path}")
                    ds = load_dataset(dataset_path, streaming=True)
                    if isinstance(ds, dict) and 'train' in ds:
                        ds = ds['train']
                    # Convert streaming dataset to list
                    data = list(ds.take(10000))  # Limit to 10000 samples
                    return data
                except Exception as e:
                    print(f"Error loading in streaming mode: {e}")
                    try:
                        # Last resort: try to download the dataset directly
                        print(f"Trying to download dataset directly: {dataset_path}")
                        from huggingface_hub import hf_hub_download
                        import pandas as pd
                        import os
                        
                        # Extract repo_id and filename
                        repo_id = dataset_path
                        
                        # Try to download the dataset info to find the files
                        try:
                            dataset_info = hf_hub_download(repo_id=repo_id, filename="dataset_info.json", repo_type="dataset")
                            print(f"Downloaded dataset info: {dataset_info}")
                        except:
                            pass
                        
                        # Try to download a parquet file
                        try:
                            file_path = hf_hub_download(repo_id=repo_id, filename="data/train-00000-of-00001.parquet", repo_type="dataset")
                            df = pd.read_parquet(file_path)
                            return df.to_dict('records')
                        except Exception as e1:
                            print(f"Error downloading parquet file: {e1}")
                            
                            # Try to download a JSON file
                            try:
                                file_path = hf_hub_download(repo_id=repo_id, filename="data.json", repo_type="dataset")
                                with open(file_path, 'r') as f:
                                    return json.load(f)
                            except Exception as e2:
                                print(f"Error downloading JSON file: {e2}")
                                raise Exception(f"Failed to load dataset {dataset_path} after multiple attempts")
                    except Exception as e:
                        print(f"Error downloading dataset directly: {e}")
                        raise
        except Exception as e:
            print(f"Error loading dataset from HuggingFace: {e}")
            raise
    
    # Load original dataset
    if is_hf_dataset(original_dataset_path):
        print(f"Loading original dataset from HuggingFace: {original_dataset_path}")
        original_data = load_hf_dataset(original_dataset_path)
    else:
        # Load from local file
        print(f"Loading original dataset from local file: {original_dataset_path}")
        with open(original_dataset_path, 'r') as f:
            original_data = json.load(f)
    
    # Load detoxified dataset
    if is_hf_dataset(detoxified_dataset_path):
        print(f"Loading detoxified dataset from HuggingFace: {detoxified_dataset_path}")
        detoxified_data = load_hf_dataset(detoxified_dataset_path)
    else:
        # Load from local file
        print(f"Loading detoxified dataset from local file: {detoxified_dataset_path}")
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
    train_size = int(train_test_split * len(original_data))
    
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