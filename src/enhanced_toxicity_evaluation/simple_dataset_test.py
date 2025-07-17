#!/usr/bin/env python3
"""
Very simple dataset test that just loads the dataset and shows what we get.
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


def test_simple_dataset():
    """Test simple dataset loading."""
    logger.info("🧪 Simple Dataset Test")
    logger.info("="*30)
    
    try:
        # Just load the dataset
        logger.info("Loading dataset...")
        ds = load_dataset("allenai/real-toxicity-prompts", split="train")
        logger.info(f"✅ Dataset loaded with {len(ds)} samples")
        
        # Show first few samples
        logger.info("First few samples:")
        for i in range(min(3, len(ds))):
            sample = ds[i]
            logger.info(f"  Sample {i+1}:")
            logger.info(f"    Prompt: {sample['prompt']['text'][:50]}...")
            logger.info(f"    Toxicity: {sample['prompt']['toxicity']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the simple dataset test."""
    success = test_simple_dataset()
    if success:
        logger.info("\n🚀 Dataset loading works!")
    else:
        logger.error("\n❌ Dataset loading failed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 