#!/usr/bin/env python3
"""
Test script to verify the visualization manager works with the actual standardized column structure.
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


def create_actual_structure_dataframe():
    """Create a test DataFrame that mimics the actual standardized column structure."""
    logger.info("📊 Creating test DataFrame with actual structure...")
    
    # Create sample data that mimics the actual modular pipeline structure
    data = []
    
    for i in range(10):  # 10 prompts
        row = {
            'prompt': f'Test prompt {i+1}',
            'prompt_index': i,
            'output': f'Test output {i+1}',
            'full_text': f'Test prompt {i+1} Test output {i+1}',
            'model': 'base' if i % 2 == 0 else 'detox_epoch_20'
        }
        
        # Add classification results with standardized column names
        # These are the same across all models - the model info comes from file names
        
        # Prompt classifications
        row['prompt_toxic_bert_results'] = {
            'toxic': np.random.random() * 0.5,
            'severe_toxic': np.random.random() * 0.3,
            'obscene': np.random.random() * 0.4,
            'threat': np.random.random() * 0.2,
            'insult': np.random.random() * 0.6,
            'identity_hate': np.random.random() * 0.1
        }
        
        row['prompt_roberta_toxicity_results'] = {
            'toxic': np.random.random() * 0.5, 
            'neutral': 1 - np.random.random() * 0.5
        }
        
        row['prompt_dynabench_hate_results'] = {
            'hate': np.random.random() * 0.3, 
            'not_hate': 1 - np.random.random() * 0.3
        }
        
        # Output classifications (standardized column names)
        row['output_toxic_bert_results'] = {
            'toxic': np.random.random() * 0.4,  # Lower toxicity
            'severe_toxic': np.random.random() * 0.2,
            'obscene': np.random.random() * 0.3,
            'threat': np.random.random() * 0.1,
            'insult': np.random.random() * 0.5,
            'identity_hate': np.random.random() * 0.1
        }
        
        row['output_roberta_toxicity_results'] = {
            'toxic': np.random.random() * 0.4, 
            'neutral': 1 - np.random.random() * 0.4
        }
        
        row['output_dynabench_hate_results'] = {
            'hate': np.random.random() * 0.2, 
            'not_hate': 1 - np.random.random() * 0.2
        }
        
        # Full text classifications (standardized column names)
        row['full_text_toxic_bert_results'] = {
            'toxic': np.random.random() * 0.5,
            'severe_toxic': np.random.random() * 0.3,
            'obscene': np.random.random() * 0.4,
            'threat': np.random.random() * 0.2,
            'insult': np.random.random() * 0.6,
            'identity_hate': np.random.random() * 0.2
        }
        
        row['full_text_roberta_toxicity_results'] = {
            'toxic': np.random.random() * 0.5, 
            'neutral': 1 - np.random.random() * 0.5
        }
        
        row['full_text_dynabench_hate_results'] = {
            'hate': np.random.random() * 0.3, 
            'not_hate': 1 - np.random.random() * 0.3
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Created test DataFrame with shape: {df.shape}")
    logger.info(f"📝 Columns: {list(df.columns)}")
    
    return df


def test_actual_structure():
    """Test the visualization manager with the actual standardized column structure."""
    logger.info("🧪 Testing Visualization Manager with Actual Structure")
    logger.info("=" * 60)
    
    try:
        # Create test data with actual structure
        test_df = create_actual_structure_dataframe()
        
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
        output_dir = Path("test_actual_structure_output")
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
    logger.info("🚀 Starting Actual Structure Test")
    logger.info("=" * 60)
    
    success = test_actual_structure()
    
    if success:
        logger.info("\n🎉 Actual structure test passed!")
        logger.info("✅ The visualization manager works with standardized column names")
    else:
        logger.error("\n❌ Actual structure test failed")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 