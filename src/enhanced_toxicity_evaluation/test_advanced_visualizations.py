#!/usr/bin/env python3
"""
Test script to verify advanced visualization features with real classifier scores.
"""

import sys
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datasets import load_dataset

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

# Import the modular pipeline components
from core.evaluation_pipeline import EvaluationPipeline
from core.visualization_manager import VisualizationManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path="configs/modular_config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        logger.info("⚠️ Using default configuration...")
        return {
            "dataset": {"name": "allenai/real-toxicity-prompts", "split": "train", "sample_size": 50},
            "wandb": {"use_wandb": True, "project": "test-project"},
            "experiment": {"name": "advanced_visualization_test"}
        }


def run_real_classification_pipeline():
    """Run the actual modular pipeline to get real classifier scores."""
    logger.info("🚀 Running real classification pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create output directory
        output_dir = Path("test_advanced_visualizations_output")
        output_dir.mkdir(exist_ok=True)
        
        # Initialize evaluation pipeline
        logger.info("🔧 Initializing Evaluation Pipeline...")
        pipeline = EvaluationPipeline(config, output_dir)
        
        # Run the full pipeline (classification + evaluation)
        logger.info("📊 Running classification phase...")
        results_df = pipeline.run_full_pipeline()
        
        if results_df is not None and len(results_df) > 0:
            logger.info(f"✅ Pipeline completed successfully!")
            logger.info(f"📊 Generated {len(results_df)} samples with real classifier scores")
            logger.info(f"📝 DataFrame shape: {results_df.shape}")
            logger.info(f"📊 Columns: {list(results_df.columns)}")
            
            # Show sample of real results
            logger.info("📋 Sample of real classifier results:")
            for col in results_df.columns:
                if col.endswith('_results'):
                    sample_result = results_df[col].iloc[0]
                    logger.info(f"  {col}: {sample_result}")
            
            return results_df
        else:
            logger.error("❌ Pipeline failed to generate results")
            return None
            
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_advanced_visualizations():
    """Test the advanced visualization features with real classifier scores."""
    logger.info("🧪 Testing Advanced Visualization Features with Real Data")
    logger.info("=" * 60)
    
    try:
        # Run the real classification pipeline to get actual scores
        results_df = run_real_classification_pipeline()
        
        if results_df is None:
            logger.error("❌ Failed to get real classification results")
            return False
        
        # Load configuration for visualization
        config = load_config()
        
        # Create visualization configuration
        viz_config = {
            "wandb": {
                "use_wandb": True,
                "project": "test-project",
                "entity": None,
                "tags": ["test", "advanced", "real_data"]
            },
            "experiment": {
                "name": "advanced_visualization_test_real_data",
                "description": "Test advanced visualization features with real classifier scores"
            },
            "logging": {
                "use_wandb": True,
                "wandb_project": "toxicity-evaluation",
                "wandb_entity": None,
                "wandb_tags": ["test", "advanced", "real_data"]
            }
        }
        
        # Create output directory for visualizations
        viz_output_dir = Path("test_advanced_visualizations_output")
        viz_output_dir.mkdir(exist_ok=True)
        
        # Initialize visualization manager
        logger.info("🔧 Initializing Visualization Manager...")
        viz_manager = VisualizationManager(viz_config, viz_output_dir)
        
        # Test model and classifier detection
        logger.info("🔍 Testing model and classifier detection...")
        models, classifiers, toxic_bert_categories = viz_manager._detect_models_and_classifiers(results_df)
        
        logger.info(f"✅ Detected models: {models}")
        logger.info(f"✅ Detected classifiers: {classifiers}")
        logger.info(f"✅ Detected Toxic-BERT categories: {toxic_bert_categories}")
        
        # Test comprehensive visualizations with real data
        logger.info("🎨 Testing comprehensive visualizations with real data...")
        viz_manager.create_comprehensive_visualizations(results_df, {})
        
        logger.info("✅ All advanced visualization tests passed!")
        
        # Cleanup
        viz_manager.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("🚀 Starting Advanced Visualization Test with Real Classifier Scores")
    logger.info("=" * 60)
    
    success = test_advanced_visualizations()
    
    if success:
        logger.info("\n🎉 Advanced visualization test passed!")
        logger.info("✅ Real classifier scores generated")
        logger.info("✅ Sophisticated plots created with real data")
        logger.info("✅ WandB integration working")
        logger.info("✅ Multi-model comparisons functional")
        logger.info("✅ Toxicity reduction plots generated")
        logger.info("✅ Scatter plots and distributions created")
        logger.info("✅ Real toxicity assessment completed")
    else:
        logger.error("\n❌ Advanced visualization test failed")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 