#!/usr/bin/env python3
"""
Memory diagnostic script for SmolLM-1.7B RLHF training.
Helps identify memory fragmentation and allocation issues.
"""

import torch
import gc
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, create_reference_model

def print_memory_info():
    """Print detailed memory information."""
    print("=" * 60)
    print("MEMORY DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # System memory
    system_memory = psutil.virtual_memory()
    print(f"System RAM: {system_memory.used / 1024**3:.2f}GB / {system_memory.total / 1024**3:.2f}GB ({system_memory.percent}%)")
    
    # GPU memory
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = props.total_memory / 1024**3
        
        print(f"GPU Memory (Device {device}):")
        print(f"  Allocated: {allocated:.2f}GB")
        print(f"  Reserved:  {reserved:.2f}GB")
        print(f"  Total:     {total:.2f}GB")
        print(f"  Free:      {total - reserved:.2f}GB")
        print(f"  Usage:     {reserved/total*100:.1f}%")
        
        # Memory fragmentation info
        if hasattr(torch.cuda, 'memory_stats'):
            stats = torch.cuda.memory_stats(device)
            print(f"  Fragmentation: {stats.get('fragmentation', 'N/A')}")
            
        # Largest block
        if hasattr(torch.cuda, 'memory_summary'):
            summary = torch.cuda.memory_summary(device)
            print(f"  Largest block: {summary.split('Largest block: ')[1].split('\\n')[0] if 'Largest block:' in summary else 'N/A'}")
    else:
        print("CUDA not available")
    
    print()

def test_model_loading():
    """Test loading the SmolLM model with different configurations."""
    print("=" * 60)
    print("MODEL LOADING TESTS")
    print("=" * 60)
    
    model_name = "HuggingFaceTB/SmolLM-1.7B"
    
    # Test 1: Basic loading
    print("Test 1: Basic model loading...")
    try:
        print_memory_info()
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✓ Basic loading successful")
        print_memory_info()
        del model
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"✗ Basic loading failed: {e}")
    
    # Test 2: With half precision
    print("\nTest 2: Half precision loading...")
    try:
        print_memory_info()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print("✓ Half precision loading successful")
        print_memory_info()
        del model
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"✗ Half precision loading failed: {e}")
    
    # Test 3: With value head
    print("\nTest 3: Value head loading...")
    try:
        print_memory_info()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        print("✓ Value head loading successful")
        print_memory_info()
        del model
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"✗ Value head loading failed: {e}")

def test_batch_processing():
    """Test processing different batch sizes."""
    print("=" * 60)
    print("BATCH PROCESSING TESTS")
    print("=" * 60)
    
    model_name = "HuggingFaceTB/SmolLM-1.7B"
    
    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            try:
                print_memory_info()
                
                # Create dummy batch
                dummy_texts = ["Hello world"] * batch_size
                inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True)
                
                # Move to GPU
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(**inputs)
                
                print(f"✓ Batch size {batch_size} successful")
                print_memory_info()
                
                # Clean up
                del inputs, outputs
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"✗ Batch size {batch_size} failed: {e}")
                break
        
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"✗ Model loading for batch test failed: {e}")

def test_memory_fragmentation():
    """Test for memory fragmentation issues."""
    print("=" * 60)
    print("MEMORY FRAGMENTATION TEST")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available for fragmentation test")
        return
    
    device = torch.cuda.current_device()
    
    # Allocate and deallocate tensors of different sizes
    print("Creating memory fragmentation...")
    
    tensors = []
    sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    
    for size in sizes:
        try:
            tensor = torch.randn(size, size, device=device, dtype=torch.bfloat16)
            tensors.append(tensor)
            print(f"✓ Allocated {size}x{size} tensor")
            print_memory_info()
        except Exception as e:
            print(f"✗ Failed to allocate {size}x{size} tensor: {e}")
            break
    
    # Deallocate in random order to create fragmentation
    print("\nDeallocating tensors in random order...")
    import random
    random.shuffle(tensors)
    
    for i, tensor in enumerate(tensors):
        del tensor
        print(f"✓ Deallocated tensor {i+1}")
        print_memory_info()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\nFinal memory state:")
    print_memory_info()

def main():
    """Run all diagnostic tests."""
    print("SmolLM-1.7B Memory Diagnostic Tool")
    print("=" * 60)
    
    # Initial memory state
    print("Initial memory state:")
    print_memory_info()
    
    # Run tests
    test_model_loading()
    test_batch_processing()
    test_memory_fragmentation()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    # Final recommendations
    print("\nRECOMMENDATIONS:")
    print("1. If model loading fails, try the ultra-conservative config")
    print("2. If batch processing fails at size > 1, use batch_size=1")
    print("3. If fragmentation is high, restart the runtime")
    print("4. Consider using gradient checkpointing")
    print("5. Monitor memory usage during training")

if __name__ == "__main__":
    main() 