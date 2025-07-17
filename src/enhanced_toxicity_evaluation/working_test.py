#!/usr/bin/env python3
"""
Test that uses the exact same dataset loading code as the working dataset_generator.py.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_working_approach():
    """Test the exact same approach as the working dataset_generator.py."""
    logger.info("🧪 Testing Working Dataset Loading Approach")
    logger.info("="*50)
    
    try:
        # Test 1: Direct dataset loading (same as dataset_generator.py)
        logger.info("Test 1: Loading dataset directly...")
        ds = load_dataset("allenai/real-toxicity-prompts", split="train")
        logger.info(f"✅ Dataset loaded with {len(ds)} samples")
        
        # Test 2: Filter for toxicity (same as dataset_generator.py)
        logger.info("Test 2: Filtering for toxicity...")
        toxicity_threshold = 0.5
        
        def filter_fn(sample):
            toxicity = sample["prompt"]["toxicity"]
            return toxicity is not None and toxicity > toxicity_threshold
        
        ds = ds.filter(filter_fn, batched=False)
        logger.info(f"✅ Filtered to {len(ds)} samples")
        
        # Test 3: Select samples (same as dataset_generator.py)
        logger.info("Test 3: Selecting samples...")
        num_samples = min(2, len(ds))
        ds = ds.select(range(num_samples))
        logger.info(f"✅ Selected {num_samples} samples")
        
        # Test 4: Extract prompts (same as dataset_generator.py)
        logger.info("Test 4: Extracting prompts...")
        prompts = [example["prompt"]["text"] for example in ds]
        logger.info(f"✅ Extracted {len(prompts)} prompts")
        
        # Show the prompts
        for i, prompt in enumerate(prompts):
            logger.info(f"  Prompt {i+1}: {prompt[:50]}...")
        
        logger.info("\n" + "="*50)
        logger.info("🎉 WORKING APPROACH SUCCESSFUL!")
        logger.info("="*50)
        logger.info("The dataset loading approach from dataset_generator.py works!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Working approach failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the working approach test."""
    success = test_working_approach()
    if success:
        logger.info("\n🚀 Dataset loading works with the original approach!")
    else:
        logger.error("\n❌ Even the working approach failed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 