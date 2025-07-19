#!/usr/bin/env python3
"""
Test file for the modular Enhanced Toxicity Evaluation Pipeline.
Demonstrates separate classification and evaluation phases.
"""

import sys
import logging
import os
from pathlib import Path
import yaml
from omegaconf import OmegaConf

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

# Import the modular components
from core import (
    EvaluationPipeline,
    ModelLoader,
    ClassifierManager,
    GenerationEngine,
    DatasetManager,
    ResultsManager
)

# Suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/modular_config.yaml") -> OmegaConf:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to OmegaConf
        config = OmegaConf.create(config_dict)
        logger.info(f"✅ Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        logger.info("⚠️ Using default configuration...")
        
        # Create default config that matches real_model_test.py approach
        default_config = {
            "models": {
                "base": {"path": "microsoft/DialoGPT-medium"},
                "detoxified": {"path": "microsoft/DialoGPT-medium"}
            },
            "generation": {"max_new_tokens": 50, "temperature": 0.7},
            "dataset": {
                "name": "allenai/real-toxicity-prompts", 
                "split": "train", 
                "sample_size": 5,
                "filter_toxic": True,
                "min_toxicity_score": 0.5
            },
            "classifiers": {
                "toxic_bert": {"model": "unitary/toxic-bert", "return_all_scores": True, "device": 0},  # Use GPU
                "roberta_toxicity": {"model": "s-nlp/roberta_toxicity_classifier", "return_all_scores": True, "device": 0},  # Use GPU
                "dynabench_hate": {"model": "facebook/roberta-hate-speech-dynabench-r4-target", "return_all_scores": True, "device": 0}  # Use GPU
            },
            "output": {"directory": "modular_test_results"},
            "caching": {"use_cache": False},
            "device": "auto"
        }
        
        return OmegaConf.create(default_config)


def test_classification_phase_only():
    """Test only the classification phase."""
    logger.info("🧪 Testing Classification Phase Only")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(config)
    
    try:
        # Run only classification phase
        result = pipeline.run_classification_phase()
        
        if result["success"]:
            logger.info("✅ Classification phase completed successfully!")
            logger.info(f"📁 Results saved to: {result['output_path']}")
            return result
        else:
            logger.error(f"❌ Classification phase failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        pipeline.cleanup()


def test_evaluation_phase_only(results_path: str = "modular_results"):
    """Test only the evaluation phase using existing results."""
    logger.info("🧪 Testing Evaluation Phase Only")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(config)
    
    try:
        # Run only evaluation phase
        result = pipeline.run_evaluation_phase(results_path=Path(results_path))
        
        if result["success"]:
            logger.info("✅ Evaluation phase completed successfully!")
            logger.info(f"📁 Visualizations saved to: {result['output_path']}")
            return result
        else:
            logger.error(f"❌ Evaluation phase failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        pipeline.cleanup()


def test_full_pipeline():
    """Test the complete pipeline: classification followed by evaluation."""
    logger.info("🧪 Testing Full Pipeline")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(config)
    
    try:
        # Run full pipeline
        result = pipeline.run_full_pipeline()
        
        if result["success"]:
            logger.info("✅ Full pipeline completed successfully!")
            logger.info(f"📁 Results saved to: {result['evaluation_result']['output_path']}")
            return result
        else:
            logger.error(f"❌ Full pipeline failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        pipeline.cleanup()


def test_individual_components():
    """Test individual components separately."""
    logger.info("🧪 Testing Individual Components")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    try:
        # Test DatasetManager first (since this was causing issues)
        logger.info("Testing DatasetManager...")
        dataset_manager = DatasetManager(config)
        prompts = dataset_manager.get_prompts()
        logger.info(f"✅ DatasetManager: Loaded {len(prompts)} prompts")
        
        # Test ModelLoader
        logger.info("Testing ModelLoader...")
        model_loader = ModelLoader(config)
        models, tokenizers = model_loader.load_models()
        logger.info(f"✅ ModelLoader: Loaded {len(models)} models")
        model_loader.cleanup()
        
        # Test ClassifierManager
        logger.info("Testing ClassifierManager...")
        classifier_manager = ClassifierManager(config)
        classifiers = classifier_manager.load_classifiers()
        logger.info(f"✅ ClassifierManager: Loaded {len(classifiers)} classifiers")
        classifier_manager.cleanup()
        
        # Test GenerationEngine
        logger.info("Testing GenerationEngine...")
        generation_engine = GenerationEngine(config)
        generation_info = generation_engine.get_generation_info()
        logger.info(f"✅ GenerationEngine: {generation_info}")
        
        # Test ResultsManager
        logger.info("Testing ResultsManager...")
        results_manager = ResultsManager(config)
        results_info = results_manager.get_results_info()
        logger.info(f"✅ ResultsManager: {results_info}")
        
        logger.info("✅ All individual components tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_manager_only():
    """Test only the dataset manager to isolate the issue."""
    logger.info("🧪 Testing Dataset Manager Only")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    
    try:
        # Test DatasetManager
        logger.info("Testing DatasetManager...")
        dataset_manager = DatasetManager(config)
        
        # Get dataset info
        dataset_info = dataset_manager.get_dataset_info()
        logger.info(f"Dataset info: {dataset_info}")
        
        # Get prompts
        prompts = dataset_manager.get_prompts()
        logger.info(f"✅ DatasetManager: Loaded {len(prompts)} prompts")
        
        if prompts:
            logger.info(f"Sample prompts:")
            for i, prompt in enumerate(prompts[:3]):
                logger.info(f"  {i+1}: {prompt[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("🚀 Starting Modular Pipeline Tests")
    logger.info("=" * 60)
    
    # Check if configuration file exists
    config_path = "configs/modular_config.yaml"
    if not Path(config_path).exists():
        logger.warning(f"⚠️ Configuration file {config_path} not found, using default config")
    
    # Test dataset manager first to isolate the issue
    logger.info("\n0️⃣ Testing Dataset Manager Only...")
    dataset_success = test_dataset_manager_only()
    
    if not dataset_success:
        logger.error("❌ Dataset manager test failed, stopping")
        return False
    
    # Test individual components
    logger.info("\n1️⃣ Testing Individual Components...")
    component_success = test_individual_components()
    
    if not component_success:
        logger.error("❌ Component tests failed, stopping")
        return False
    
    # Test classification phase only
    logger.info("\n2️⃣ Testing Classification Phase Only...")
    classification_result = test_classification_phase_only()
    
    if not classification_result:
        logger.error("❌ Classification phase test failed")
        return False
    
    # Test evaluation phase only (using results from classification)
    # Use the correct path based on the config
    config = load_config()
    output_dir = config.get("output", {}).get("directory", "modular_results")
    logger.info(f"\n3️⃣ Testing Evaluation Phase Only (using {output_dir})...")
    evaluation_result = test_evaluation_phase_only(output_dir)
    
    if not evaluation_result:
        logger.error("❌ Evaluation phase test failed")
        return False
    
    # Test full pipeline
    logger.info("\n4️⃣ Testing Full Pipeline...")
    full_pipeline_result = test_full_pipeline()
    
    if not full_pipeline_result:
        logger.error("❌ Full pipeline test failed")
        return False
    
    logger.info("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("📋 Test Summary:")
    logger.info("  ✅ Dataset manager working")
    logger.info("  ✅ Individual components working")
    logger.info("  ✅ Classification phase working")
    logger.info("  ✅ Evaluation phase working")
    logger.info("  ✅ Full pipeline working")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("\n🚀 All modular pipeline tests passed!")
        logger.info("📝 The modular architecture is working correctly")
    else:
        logger.error("\n❌ Some tests failed.")
    
    sys.exit(0 if success else 1) 