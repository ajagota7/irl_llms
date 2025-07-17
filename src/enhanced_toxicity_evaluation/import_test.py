#!/usr/bin/env python3
"""
Simple import test to verify all modules can be imported correctly.
"""

import sys
import logging
from pathlib import Path

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all core modules can be imported."""
    logger.info("🧪 Testing Module Imports")
    logger.info("="*30)
    
    try:
        # Test 1: Core modules
        logger.info("Test 1: Importing core modules...")
        from core.model_loader import ModelLoader
        from core.classifier_manager import ClassifierManager
        from core.dataset_manager import DatasetManager
        from core.generation_engine import GenerationEngine
        from core.metrics_calculator import MetricsCalculator
        from core.evaluator import ToxicityEvaluator
        logger.info("✅ All core modules imported successfully")
        
        # Test 2: External dependencies
        logger.info("Test 2: Importing external dependencies...")
        import torch
        import transformers
        from datasets import load_dataset
        from omegaconf import OmegaConf
        import pandas as pd
        import numpy as np
        logger.info("✅ All external dependencies imported successfully")
        
        # Test 3: Check versions
        logger.info("Test 3: Checking versions...")
        logger.info(f"   PyTorch: {torch.__version__}")
        logger.info(f"   Transformers: {transformers.__version__}")
        logger.info(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   CUDA version: {torch.version.cuda}")
        
        logger.info("\n" + "="*30)
        logger.info("🎉 ALL IMPORTS SUCCESSFUL!")
        logger.info("="*30)
        logger.info("All modules can be imported correctly.")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


def main():
    """Run the import test."""
    success = test_imports()
    if success:
        logger.info("\n🚀 All imports working! Ready for component tests.")
    else:
        logger.error("\n❌ Import test failed. Please check the errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 