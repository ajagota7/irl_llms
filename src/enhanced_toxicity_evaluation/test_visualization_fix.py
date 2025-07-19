#!/usr/bin/env python3
"""
Test script to verify the visualization manager fixes work properly.
"""

import sys
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

# Import the visualization manager
from core.visualization_manager import VisualizationManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_dataframe():
    """Create a test DataFrame that mimics the modular pipeline output."""
    logger.info("📊 Creating test DataFrame...")
    
    # Create sample data that mimics the modular pipeline structure
    data = []
    
    for i in range(10):  # 10 prompts
        row = {
            'prompt': f'Test prompt {i+1}',
            'prompt_index': i,
            'output': f'Test output {i+1}',
            'full_text': f'Test prompt {i+1} Test output {i+1}',
            'model': 'base' if i % 2 == 0 else 'detox_epoch_20'
        }
        
        # Add classification results (dictionaries)
        row['prompt_toxic_bert_results'] = {
            'toxic': np.random.random() * 0.5,
            'severe_toxic': np.random.random() * 0.3,
            'obscene': np.random.random() * 0.4,
            'threat': np.random.random() * 0.2,
            'insult': np.random.random() * 0.6,
            'identity_hate': np.random.random() * 0.1
        }
        
        row['output_base_toxic_bert_results'] = {
            'toxic': np.random.random() * 0.6,
            'severe_toxic': np.random.random() * 0.4,
            'obscene': np.random.random() * 0.5,
            'threat': np.random.random() * 0.3,
            'insult': np.random.random() * 0.7,
            'identity_hate': np.random.random() * 0.2
        }
        
        row['output_detox_epoch_20_toxic_bert_results'] = {
            'toxic': np.random.random() * 0.4,  # Lower toxicity
            'severe_toxic': np.random.random() * 0.2,
            'obscene': np.random.random() * 0.3,
            'threat': np.random.random() * 0.1,
            'insult': np.random.random() * 0.5,
            'identity_hate': np.random.random() * 0.1
        }
        
        row['full_text_base_toxic_bert_results'] = {
            'toxic': np.random.random() * 0.7,
            'severe_toxic': np.random.random() * 0.5,
            'obscene': np.random.random() * 0.6,
            'threat': np.random.random() * 0.4,
            'insult': np.random.random() * 0.8,
            'identity_hate': np.random.random() * 0.3
        }
        
        row['full_text_detox_epoch_20_toxic_bert_results'] = {
            'toxic': np.random.random() * 0.5,
            'severe_toxic': np.random.random() * 0.3,
            'obscene': np.random.random() * 0.4,
            'threat': np.random.random() * 0.2,
            'insult': np.random.random() * 0.6,
            'identity_hate': np.random.random() * 0.2
        }
        
        # Add other classifiers
        row['prompt_roberta_toxicity_results'] = {'toxic': np.random.random() * 0.5, 'neutral': 1 - np.random.random() * 0.5}
        row['output_base_roberta_toxicity_results'] = {'toxic': np.random.random() * 0.6, 'neutral': 1 - np.random.random() * 0.6}
        row['output_detox_epoch_20_roberta_toxicity_results'] = {'toxic': np.random.random() * 0.4, 'neutral': 1 - np.random.random() * 0.4}
        
        row['prompt_dynabench_hate_results'] = {'hate': np.random.random() * 0.3, 'not_hate': 1 - np.random.random() * 0.3}
        row['output_base_dynabench_hate_results'] = {'hate': np.random.random() * 0.4, 'not_hate': 1 - np.random.random() * 0.4}
        row['output_detox_epoch_20_dynabench_hate_results'] = {'hate': np.random.random() * 0.2, 'not_hate': 1 - np.random.random() * 0.2}
        
        data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Created test DataFrame with shape: {df.shape}")
    logger.info(f"📝 Columns: {list(df.columns)}")
    
    return df


def test_visualization_manager():
    """Test the visualization manager with the fixed code."""
    logger.info("🧪 Testing Visualization Manager")
    logger.info("=" * 60)
    
    try:
        # Create test data
        test_df = create_test_dataframe()
        
        # Create test configuration
        config = {
            "wandb": {
                "use_wandb": False,  # Disable WandB for testing
                "project": "test-project",
                "entity": None,
                "tags": ["test"]
            },
            "experiment": {
                "name": "test_experiment",
                "description": "Test experiment"
            }
        }
        
        # Create output directory
        output_dir = Path("test_visualization_output")
        output_dir.mkdir(exist_ok=True)
        
        # Initialize visualization manager
        logger.info("🔧 Initializing Visualization Manager...")
        viz_manager = VisualizationManager(config, output_dir)
        
        # Test model and classifier detection
        logger.info("🔍 Testing model and classifier detection...")
        models, classifiers, toxic_bert_categories = viz_manager._detect_models_and_classifiers(test_df)
        
        logger.info(f"✅ Detected models: {models}")
        logger.info(f"✅ Detected classifiers: {classifiers}")
        logger.info(f"✅ Detected Toxic-BERT categories: {toxic_bert_categories}")
        
        # Test delta column creation
        logger.info("📊 Testing delta column creation...")
        df_with_deltas = viz_manager._create_delta_columns(test_df)
        
        delta_columns = [col for col in df_with_deltas.columns if col.startswith('delta_')]
        logger.info(f"✅ Created delta columns: {delta_columns}")
        
        # Test comprehensive visualizations (without WandB)
        logger.info("🎨 Testing comprehensive visualizations...")
        viz_manager.create_comprehensive_visualizations(df_with_deltas, {})
        
        logger.info("✅ All visualization tests passed!")
        
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
    logger.info("🚀 Starting Visualization Manager Fix Test")
    logger.info("=" * 60)
    
    success = test_visualization_manager()
    
    if success:
        logger.info("\n🎉 Visualization manager fix test passed!")
        logger.info("✅ The fixes are working correctly")
    else:
        logger.error("\n❌ Visualization manager fix test failed")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 