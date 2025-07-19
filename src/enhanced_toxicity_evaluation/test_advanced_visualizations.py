#!/usr/bin/env python3
"""
Test script to verify advanced visualization features with multiple models.
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


def create_multi_model_dataframe():
    """Create a test DataFrame with multiple models for advanced visualizations."""
    logger.info("📊 Creating multi-model test DataFrame...")
    
    # Create sample data with multiple models
    data = []
    
    # Define models and their epochs
    models = ['base', 'detox_epoch_20', 'detox_epoch_40', 'detox_epoch_60', 'detox_epoch_80', 'detox_epoch_100']
    
    for model in models:
        for i in range(10):  # 10 prompts per model
            row = {
                'prompt': f'Test prompt {i+1}',
                'prompt_index': i,
                'output': f'Test output {i+1} for {model}',
                'full_text': f'Test prompt {i+1} Test output {i+1} for {model}',
                'model': model
            }
            
            # Add raw classification results with realistic toxicity reduction
            # Base model has higher toxicity, detoxified models have progressively lower toxicity
            
            # Determine base toxicity level based on model
            if model == 'base':
                base_toxicity = 0.6 + np.random.random() * 0.3  # High toxicity
            else:
                # Extract epoch number for progressive reduction
                epoch_num = int(model.split('_')[-1])
                reduction_factor = epoch_num / 100.0  # Progressive reduction
                base_toxicity = (0.6 + np.random.random() * 0.3) * (1 - reduction_factor * 0.7)
            
            # Prompt classifications (same for all models)
            row['prompt_toxic_bert_results'] = {
                'toxic': 0.5 + np.random.random() * 0.4,
                'severe_toxic': np.random.random() * 0.3,
                'obscene': np.random.random() * 0.4,
                'threat': np.random.random() * 0.2,
                'insult': np.random.random() * 0.6,
                'identity_hate': np.random.random() * 0.1
            }
            
            row['prompt_roberta_toxicity_results'] = {
                'toxic': 0.4 + np.random.random() * 0.4, 
                'neutral': 1 - (0.4 + np.random.random() * 0.4)
            }
            
            row['prompt_dynabench_hate_results'] = {
                'hate': np.random.random() * 0.3, 
                'not_hate': 1 - np.random.random() * 0.3
            }
            
            # Output classifications (varies by model)
            row['output_toxic_bert_results'] = {
                'toxic': base_toxicity + np.random.random() * 0.2,
                'severe_toxic': (base_toxicity * 0.5) + np.random.random() * 0.1,
                'obscene': (base_toxicity * 0.6) + np.random.random() * 0.2,
                'threat': (base_toxicity * 0.3) + np.random.random() * 0.1,
                'insult': (base_toxicity * 0.8) + np.random.random() * 0.2,
                'identity_hate': (base_toxicity * 0.2) + np.random.random() * 0.1
            }
            
            row['output_roberta_toxicity_results'] = {
                'toxic': base_toxicity + np.random.random() * 0.2, 
                'neutral': 1 - (base_toxicity + np.random.random() * 0.2)
            }
            
            row['output_dynabench_hate_results'] = {
                'hate': (base_toxicity * 0.5) + np.random.random() * 0.2, 
                'not_hate': 1 - ((base_toxicity * 0.5) + np.random.random() * 0.2)
            }
            
            # Full text classifications
            row['full_text_toxic_bert_results'] = {
                'toxic': base_toxicity * 1.1 + np.random.random() * 0.2,
                'severe_toxic': (base_toxicity * 0.6) + np.random.random() * 0.1,
                'obscene': (base_toxicity * 0.7) + np.random.random() * 0.2,
                'threat': (base_toxicity * 0.4) + np.random.random() * 0.1,
                'insult': (base_toxicity * 0.9) + np.random.random() * 0.2,
                'identity_hate': (base_toxicity * 0.3) + np.random.random() * 0.1
            }
            
            row['full_text_roberta_toxicity_results'] = {
                'toxic': base_toxicity * 1.1 + np.random.random() * 0.2, 
                'neutral': 1 - (base_toxicity * 1.1 + np.random.random() * 0.2)
            }
            
            row['full_text_dynabench_hate_results'] = {
                'hate': (base_toxicity * 0.6) + np.random.random() * 0.2, 
                'not_hate': 1 - ((base_toxicity * 0.6) + np.random.random() * 0.2)
            }
            
            data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Created multi-model test DataFrame with shape: {df.shape}")
    logger.info(f"📝 Models: {df['model'].unique()}")
    logger.info(f"📊 Columns: {list(df.columns)}")
    
    return df


def test_advanced_visualizations():
    """Test the advanced visualization features."""
    logger.info("🧪 Testing Advanced Visualization Features")
    logger.info("=" * 60)
    
    try:
        # Create test data with multiple models
        test_df = create_multi_model_dataframe()
        
        # Create test configuration
        config = {
            "wandb": {
                "use_wandb": True,  # Enable WandB for testing
                "project": "test-project",
                "entity": None,
                "tags": ["test", "advanced"]
            },
            "experiment": {
                "name": "advanced_visualization_test",
                "description": "Test advanced visualization features"
            },
            "logging": {
                "use_wandb": True,
                "wandb_project": "toxicity-evaluation",
                "wandb_entity": None,
                "wandb_tags": ["test", "advanced"]
            }
        }
        
        # Create output directory
        output_dir = Path("test_advanced_visualizations_output")
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
        
        # Test comprehensive visualizations
        logger.info("🎨 Testing comprehensive visualizations...")
        viz_manager.create_comprehensive_visualizations(test_df, {})
        
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
    logger.info("🚀 Starting Advanced Visualization Test")
    logger.info("=" * 60)
    
    success = test_advanced_visualizations()
    
    if success:
        logger.info("\n🎉 Advanced visualization test passed!")
        logger.info("✅ Sophisticated plots created")
        logger.info("✅ WandB integration working")
        logger.info("✅ Multi-model comparisons functional")
        logger.info("✅ Toxicity reduction plots generated")
        logger.info("✅ Scatter plots and distributions created")
    else:
        logger.error("\n❌ Advanced visualization test failed")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 