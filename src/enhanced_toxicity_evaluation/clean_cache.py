#!/usr/bin/env python3
"""
Simple script to clean up corrupted cache files.
"""

import os
import shutil
from pathlib import Path

def clean_cache():
    """Clean up cache directories and files."""
    print("🧹 Cleaning up cache files...")
    
    # List of cache directories to clean
    cache_dirs = [
        "datasets/cache",
        "modular_results",
        "modular_test_results",
        "real_model_results"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Removed cache directory: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Could not remove {cache_dir}: {e}")
        else:
            print(f"ℹ️ Cache directory does not exist: {cache_dir}")
    
    # Also clean up any .json cache files in the current directory
    current_dir = Path(".")
    for json_file in current_dir.glob("*.json"):
        if "cache" in json_file.name.lower() or "real_toxicity" in json_file.name.lower():
            try:
                json_file.unlink()
                print(f"✅ Removed cache file: {json_file}")
            except Exception as e:
                print(f"⚠️ Could not remove {json_file}: {e}")
    
    print("🎉 Cache cleanup completed!")

if __name__ == "__main__":
    clean_cache() 