"""
Utility functions and classes for IRL detoxification.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo
from omegaconf import DictConfig, OmegaConf


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

    def _freeze_base_model(self, num_unfrozen_layers):
        """Freeze the base model, except for the last few layers."""
        # First freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Then unfreeze the last n layers
        if num_unfrozen_layers > 0:
            try:
                # For different model architectures, handle this differently
                if hasattr(self.model, 'transformer'):
                    # For GPT-Neo and similar models
                    layers = self.model.transformer.h
                    for i in range(1, num_unfrozen_layers + 1):
                        layer_idx = len(layers) - i
                        for param in layers[layer_idx].parameters():
                            param.requires_grad = True
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    # For some newer models
                    layers = self.model.model.layers
                    for i in range(1, num_unfrozen_layers + 1):
                        layer_idx = len(layers) - i
                        for param in layers[layer_idx].parameters():
                            param.requires_grad = True
                else:
                    print("Unsupported model architecture. All parameters frozen except value head.")
            except Exception as e:
                print(f"Error unfreezing layers: {e}")

        # Always unfreeze the value head
        for param in self.v_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model."""
        # Make sure inputs are on the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Use autocast for mixed precision if needed
        with torch.amp.autocast(device_type=self.device_type, enabled=self.use_half_precision):
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
        method: The IRL method to use ("max_margin" or "max_entropy")
        **kwargs: Additional arguments to pass to the loss function
        
    Returns:
        A loss function that takes original_rewards and detoxified_rewards
    """
    if method == "max_entropy":
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

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot accuracy and F1
    ax1 = axes[0, 0]
    ax1.plot(epochs_list, [m['accuracy'] for m in metrics_history], 'o-', label='Accuracy')
    ax1.plot(epochs_list, [m['f1'] for m in metrics_history], 's-', label='F1 Score')
    ax1.plot(epochs_list, [m['auc_roc'] for m in metrics_history], '^-', label='AUC-ROC')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot correlations
    ax2 = axes[0, 1]
    ax2.plot(epochs_list, [m['pearson_correlation'] for m in metrics_history], 'o-', label='Pearson')
    ax2.plot(epochs_list, [m['spearman_correlation'] for m in metrics_history], 's-', label='Spearman')
    ax2.plot(epochs_list, [m['kendall_tau'] for m in metrics_history], '^-', label='Kendall Tau')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation with True Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot average rewards
    ax3 = axes[1, 0]
    ax3.plot(epochs_list, [m['avg_original_reward'] for m in metrics_history], 'r-', label='Original (Toxic)')
    ax3.plot(epochs_list, [m['avg_detoxified_reward'] for m in metrics_history], 'g-', label='Detoxified')
    ax3.plot(epochs_list, [m['reward_diff'] for m in metrics_history], 'b--', label='Difference')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('Average Predicted Rewards')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot loss
    ax4 = axes[1, 1]
    ax4.plot(epochs_list, [m['loss'] for m in metrics_history], 'o-')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss')
    ax4.grid(True, alpha=0.3)

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
    """Push the model to the HuggingFace Hub using a user namespace instead of root."""
    import os
    from huggingface_hub import HfApi
    from omegaconf import OmegaConf
    
    try:
        # Use your username rather than trying to create at the root level
        # Most likely your successful uploads are in YOUR namespace
        username = "ajagota71"  # Replace with your actual HF username
        
        # Create a simple, short repo name
        model_name = config.model.reward_model_base.split('/')[-1].lower()
        
        if checkpoint_suffix:
            repo_name = f"irl-reward-{model_name}-{checkpoint_suffix}"
        else:
            repo_name = f"irl-reward-{model_name}"
        
        # Use username namespace - this is critical!
        repo_id = f"{username}/{repo_name}"
        
        print(f"Pushing model to HuggingFace Hub: {repo_id}")
        
        # Create a simple output directory
        output_dir = os.path.join(os.getcwd(), f"hf_upload_{repo_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        print(f"Saving model files to {output_dir}")
        reward_model.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save config
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(config))
        
        # Create README
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(f"# IRL Reward Model\n\n")
            f.write(f"This model was trained using {config.training.irl_method} IRL to learn toxicity reward signals.\n\n")
            f.write(f"Base model: {config.model.reward_model_base}\n")
            f.write(f"Original model: {config.dataset.original_model_name}\n")
            f.write(f"Detoxified model: {config.dataset.detoxified_model_name}\n\n")
            # Proper metadata section
            f.write("---\n")
            f.write("tags:\n")
            f.write("- toxicity\n")
            f.write("- reward-model\n")
            f.write("library_name: transformers\n")
            f.write("---\n")
        
        # Use HfApi
        api = HfApi()
        
        # Create repository with explicit username namespace
        print(f"Creating repository: {repo_id}")
        try:
            api.create_repo(
                repo_id=repo_id,
                exist_ok=True
            )
            print(f"Repository created: {repo_id}")
        except Exception as e:
            print(f"Note on repository creation: {e}")
        
        # Upload folder
        print(f"Uploading files to {repo_id}")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id
        )
        
        print(f"Successfully uploaded model to {repo_id}")
        return repo_id
        
    except Exception as e:
        print(f"Error in push_to_hub: {e}")
        import traceback
        traceback.print_exc()
        return None