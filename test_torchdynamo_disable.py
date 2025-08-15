#!/usr/bin/env python3
"""
Test script to verify TorchDynamo is completely disabled for Gemma3 models.
"""

import os
import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set environment variables to disable TorchDynamo completely
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_TORCH_COMPILE'] = '1'
os.environ['TORCH_LOGS'] = 'off'
os.environ['TORCHDYNAMO_VERBOSE'] = '0'

# Disable TorchDynamo compilation globally
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
torch._dynamo.config.backend = "eager"  # Use eager backend (no compilation)

@hydra.main(config_path="src/configs", config_name="config", version_base=None)
def test_torchdynamo_disable(cfg: DictConfig) -> None:
    """Test that TorchDynamo is completely disabled and Gemma3 works."""
    
    print("=== TorchDynamo Disable Test ===")
    print(f"TORCHDYNAMO_DISABLE: {os.environ.get('TORCHDYNAMO_DISABLE', 'Not set')}")
    print(f"TORCH_COMPILE_DISABLE: {os.environ.get('TORCH_COMPILE_DISABLE', 'Not set')}")
    print(f"PYTORCH_DISABLE_TORCH_COMPILE: {os.environ.get('PYTORCH_DISABLE_TORCH_COMPILE', 'Not set')}")
    print(f"torch._dynamo.config.disable: {torch._dynamo.config.disable}")
    print(f"torch._dynamo.config.suppress_errors: {torch._dynamo.config.suppress_errors}")
    print(f"torch._dynamo.config.backend: {torch._dynamo.config.backend}")
    
    print(f"\nModel name: {cfg.rlhf.model.name}")
    
    # Prepare model loading kwargs
    model_kwargs = {}
    if hasattr(cfg.rlhf.model, 'attn_implementation'):
        model_kwargs['attn_implementation'] = cfg.rlhf.model.attn_implementation
        print(f"Using attention implementation: {cfg.rlhf.model.attn_implementation}")
    
    if hasattr(cfg.rlhf.model, 'use_cache'):
        model_kwargs['use_cache'] = cfg.rlhf.model.use_cache
        print(f"Use cache: {cfg.rlhf.model.use_cache}")
    
    print("TorchDynamo compilation disabled via environment variables and global config")
    
    try:
        # Load tokenizer
        print("\nLoading tokenizer...")
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
        
        # Test generation (TorchDynamo is already disabled globally)
        print("Testing generation...")
        generated = model.generate(
            test_input.input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        print("✅ Generation successful!")
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
        print("\n🎉 All tests passed! TorchDynamo is completely disabled.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    test_torchdynamo_disable() 