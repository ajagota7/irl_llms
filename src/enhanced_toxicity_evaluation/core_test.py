#!/usr/bin/env python3
"""
Core test script that only tests model and classifier loading.
No dataset loading, no generation, just the core components.
"""

import os
import sys
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from core.model_loader import ModelLoader
from core.classifier_manager import ClassifierManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_core_components():
    """Test only the core components (model and classifier loading)."""
    logger.info("🧪 Testing Core Components")
    logger.info("="*40)
    
    try:
        # Test 1: Model Loading
        logger.info("Test 1: Loading model...")
        model_config = {
            "models": [
                {
                    "name": "test_model",
                    "hf_path": "EleutherAI/pythia-70m",
                    "type": "base_model",
                    "description": "Test model"
                }
            ],
            "model_loading": {
                "device_map": "auto",
                "torch_dtype": "auto",
                "trust_remote_code": True
            }
        }
        
        model_loader = ModelLoader(OmegaConf.create(model_config))
        models = model_loader.load_models()
        logger.info(f"✅ Model loaded successfully: {list(models.keys())}")
        
        # Test 2: Classifier Loading
        logger.info("Test 2: Loading classifier...")
        classifier_config = {
            "classifiers": {
                "roberta_toxicity": {
                    "enabled": True,
                    "model": "s-nlp/roberta_toxicity_classifier",
                    "batch_size": 1,
                    "max_length": 64
                }
            },
            "evaluation": {
                "parallel": False,
                "max_workers": 1,
                "timeout": 30
            },
            "error_handling": {
                "skip_failed_classifiers": True,
                "log_errors": True,
                "fallback_to_safe": True
            }
        }
        
        classifier_manager = ClassifierManager(OmegaConf.create(classifier_config))
        classifiers = classifier_manager.load_classifiers()
        logger.info(f"✅ Classifier loaded successfully: {list(classifiers.keys())}")
        
        # Test 3: Test classifier on a simple text
        logger.info("Test 3: Testing classifier on simple text...")
        test_text = "I hate everyone and want to"
        
        results = classifier_manager.evaluate_text(test_text)
        logger.info(f"✅ Classifier evaluation completed")
        logger.info(f"   Text: {test_text}")
        for classifier_name, score in results.items():
            logger.info(f"   {classifier_name}: {score:.4f}")
        
        logger.info("\n" + "="*40)
        logger.info("🎉 CORE COMPONENTS TEST PASSED!")
        logger.info("="*40)
        logger.info("Model and classifier loading are working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Core components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the core components test."""
    success = test_core_components()
    if success:
        logger.info("\n🚀 Core components are working! Ready for full pipeline.")
    else:
        logger.error("\n❌ Core components test failed. Please check the errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 