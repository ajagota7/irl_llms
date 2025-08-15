#!/usr/bin/env python3
"""
Test script to verify Gemma3 model loading with the new configuration.
"""

import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable TorchDynamo compilation globally
import torch._dynamo
torch._dynamo.config.suppress_errors = True

@hydra.main(config_path="src/configs", config_name="config", version_base=None)
def test_gemma_loading(cfg: DictConfig) -> None:
    """Test loading the Gemma3 model with the new configuration."""
    
    print("Testing Gemma3 model loading...")
    print(f"Model name: {cfg.rlhf.model.name}")
    
    # Prepare model loading kwargs
    model_kwargs = {}
    if hasattr(cfg.rlhf.model, 'attn_implementation'):
        model_kwargs['attn_implementation'] = cfg.rlhf.model.attn_implementation
        print(f"Using attention implementation: {cfg.rlhf.model.attn_implementation}")
    
    if hasattr(cfg.rlhf.model, 'torch_compile') and not cfg.rlhf.model.torch_compile:
        model_kwargs['torch_compile'] = False
        print("TorchDynamo compilation disabled")
    
    if hasattr(cfg.rlhf.model, 'use_cache'):
        model_kwargs['use_cache'] = cfg.rlhf.model.use_cache
        print(f"Use cache: {cfg.rlhf.model.use_cache}")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.rlhf.model.name)
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(cfg.rlhf.model.name, **model_kwargs)
        
        print("✅ Model loaded successfully!")
        
        # Test a simple forward pass
        print("Testing forward pass...")
        test_input = tokenizer("Hello, world!", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**test_input)
        
        print("✅ Forward pass successful!")
        print(f"Output shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

if __name__ == "__main__":
    test_gemma_loading() 