"""
Extended reward model loader that can load models from HuggingFace.
"""

import os
import torch
from src.irl_utilities import RewardModel

class ExtendedRewardModel(RewardModel):
    """Extended reward model with additional loading capabilities."""
    
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
                state_dict = torch.load(v_head_path, map_location=device)
                if isinstance(state_dict, dict) and 'weight' in state_dict and 'bias' in state_dict:
                    reward_model.v_head = torch.nn.Linear(model.config.hidden_size, 1)
                    reward_model.v_head.to(device)
                    reward_model.v_head.weight.data = state_dict['weight']
                    reward_model.v_head.bias.data = state_dict['bias']
                    print(f"Successfully loaded value head weights directly")
                else:
                    print(f"Value head file has unexpected format")
            except Exception as e:
                print(f"Error loading value head from {v_head_path}: {e}")
        
        return reward_model 