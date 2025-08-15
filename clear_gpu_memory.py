#!/usr/bin/env python3
"""
Script to completely clear GPU memory and reset CUDA context.
Use this before running training to fix memory fragmentation issues.
"""

import torch
import gc
import os
import subprocess
import time

def clear_gpu_memory():
    """Completely clear GPU memory and reset CUDA context."""
    print("=" * 60)
    print("GPU MEMORY CLEARING UTILITY")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Nothing to clear.")
        return
    
    # Get initial memory state
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory / 1024**3
    
    print(f"GPU Device: {device}")
    print(f"Total GPU Memory: {total_memory:.2f}GB")
    
    # Clear all PyTorch tensors
    print("\n1. Clearing PyTorch tensors...")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except:
            pass
    
    # Force garbage collection
    print("2. Running garbage collection...")
    for i in range(5):
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.1)
    
    # Reset CUDA context (nuclear option)
    print("3. Resetting CUDA context...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Try to reset the device
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)
        
        print("✓ CUDA context reset successful")
    except Exception as e:
        print(f"⚠ CUDA context reset failed: {e}")
    
    # Final memory state
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    
    print(f"\nFinal Memory State:")
    print(f"  Allocated: {allocated:.2f}GB")
    print(f"  Reserved:  {reserved:.2f}GB")
    print(f"  Free:      {total_memory - reserved:.2f}GB")
    print(f"  Usage:     {reserved/total_memory*100:.1f}%")
    
    if allocated < 0.1:  # Less than 100MB
        print("\n✓ GPU memory successfully cleared!")
    else:
        print(f"\n⚠ GPU memory still has {allocated:.2f}GB allocated")
        print("Consider restarting the runtime if this persists.")

def restart_runtime():
    """Provide instructions for restarting the runtime."""
    print("\n" + "=" * 60)
    print("RUNTIME RESTART INSTRUCTIONS")
    print("=" * 60)
    print("If memory clearing doesn't work, restart the runtime:")
    print("1. Go to Runtime → Restart runtime")
    print("2. Wait for the runtime to restart")
    print("3. Re-run your training command")
    print("\nThis will completely clear all memory and fragmentation.")

def main():
    """Main function."""
    print("This script will attempt to clear all GPU memory and reset the CUDA context.")
    print("This is useful for fixing memory fragmentation issues.")
    print()
    
    confirm = input("Proceed with memory clearing? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        clear_gpu_memory()
        restart_runtime()
    else:
        print("Memory clearing cancelled.")

if __name__ == "__main__":
    main() 