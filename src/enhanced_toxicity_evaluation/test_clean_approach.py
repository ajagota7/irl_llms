#!/usr/bin/env python3
"""
Test script to verify the visualization manager works with the clean approach.
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


def create_clean_dataframe():
    """Create a test DataFrame with clean structure - just raw classification results."""
    logger.info("📊 Creating clean test DataFrame...")
    
    # Create sample data with clean structure
    data = []
    
    for i in range(10):  # 10 prompts
        row = {
            'prompt': f'Test prompt {i+1}',
            'prompt_index': i,
            'output': f'Test output {i+1}',
            'full_text': f'Test prompt {i+1} Test output {i+1}',
            'model': 'base' if i % 2 == 0 else 'detox_epoch_20'
        }
        
        # Add raw classification results (no deltas, no complex calculations)
        
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
        
        # Output classifications
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
        
        # Full text classifications
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
    logger.info(f"✅ Created clean test DataFrame with shape: {df.shape}")
    logger.info(f"📝 Columns: {list(df.columns)}")
    
    return df


def test_clean_approach():
    """Test the visualization manager with the clean approach."""
    logger.info("🧪 Testing Visualization Manager with Clean Approach")
    logger.info("=" * 60)
    
    try:
        # Create test data with clean structure
        test_df = create_clean_dataframe()
        
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
        output_dir = Path("test_clean_approach_output")
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
        
        # Test delta column creation (should be minimal now)
        logger.info("📊 Testing delta column creation...")
        df_with_deltas = viz_manager._create_delta_columns(test_df)
        
        delta_columns = [col for col in df_with_deltas.columns if col.startswith('delta_')]
        logger.info(f"✅ Delta columns created: {delta_columns}")
        
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
    logger.info("🚀 Starting Clean Approach Test")
    logger.info("=" * 60)
    
    success = test_clean_approach()
    
    if success:
        logger.info("\n🎉 Clean approach test passed!")
        logger.info("✅ The visualization manager works with clean datasets")
        logger.info("✅ No complex delta calculations in main datasets")
        logger.info("✅ Model comparisons will be done during visualization")
    else:
        logger.error("\n❌ Clean approach test failed")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 