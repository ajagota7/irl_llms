#!/usr/bin/env python3
"""
Simple test that only tests dataset loading to isolate the issue.
"""

import os
import sys
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from core.dataset_manager import DatasetManager

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dataset_loading():
    """Test only dataset loading."""
    logger.info("🧪 Testing Dataset Loading")
    logger.info("="*40)
    
    try:
        # Create minimal config
        config = {
            "dataset": {
                "name": "allenai/real-toxicity-prompts",
                "split": "train",
                "max_prompts": 2,
                "shuffle_seed": 42
            },
            "filtering": {
                "strategy": "toxicity_threshold",
                "toxicity_threshold": 0.5,
                "metrics": ["toxicity"]
            },
            "processing": {
                "min_prompt_length": 5,
                "max_prompt_length": 50,
                "remove_duplicates": True,
                "normalize_whitespace": True
            },
            "caching": {
                "use_cache": False
            }
        }
        
        # Initialize dataset manager
        dataset_manager = DatasetManager(OmegaConf.create(config))
        logger.info("✅ Dataset manager initialized")
        
        # Try to get prompts
        logger.info("🔄 Getting prompts...")
        prompts = dataset_manager.get_prompts()
        
        logger.info(f"✅ Got {len(prompts)} prompts")
        for i, prompt in enumerate(prompts):
            logger.info(f"  Prompt {i+1}: {prompt[:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the dataset loading test."""
    success = test_dataset_loading()
    if success:
        logger.info("\n🚀 Dataset loading works!")
    else:
        logger.error("\n❌ Dataset loading failed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 