#!/usr/bin/env python3
"""
Test script for enhanced Toxic-BERT category analysis.
This script demonstrates the new functionality for analyzing all Toxic-BERT categories.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the core directory to the path
sys.path.append(str(Path(__file__).parent / "core"))

from visualization_manager import VisualizationManager

def create_sample_data_with_toxic_bert_categories():
    """Create sample data with all Toxic-BERT categories for testing."""
    print("📊 Creating sample data with Toxic-BERT categories...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create base data
    data = {
        'prompt': [f"Sample prompt {i}" for i in range(n_samples)],
        'output_base': [f"Base output {i}" for i in range(n_samples)],
        'output_detox_epoch_20': [f"Detox epoch 20 output {i}" for i in range(n_samples)],
        'output_detox_epoch_40': [f"Detox epoch 40 output {i}" for i in range(n_samples)],
        'output_detox_epoch_60': [f"Detox epoch 60 output {i}" for i in range(n_samples)],
        'output_detox_epoch_80': [f"Detox epoch 80 output {i}" for i in range(n_samples)],
        'output_detox_epoch_100': [f"Detox epoch 100 output {i}" for i in range(n_samples)],
    }
    
    # Add prompt scores
    data['prompt_roberta_toxicity_score'] = np.random.beta(2, 5, n_samples)
    data['prompt_dynabench_hate_score'] = np.random.beta(1, 4, n_samples)
    
    # Add base model scores
    data['base_roberta_toxicity_score'] = np.random.beta(3, 4, n_samples)
    data['base_dynabench_hate_score'] = np.random.beta(2, 3, n_samples)
    
    # Add Toxic-BERT base scores for all categories
    toxic_bert_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for category in toxic_bert_categories:
        data[f'base_toxic_bert_{category}_score'] = np.random.beta(2, 5, n_samples)
    
    # Add detoxified model scores with progressive improvement
    models = ['detox_epoch_20', 'detox_epoch_40', 'detox_epoch_60', 'detox_epoch_80', 'detox_epoch_100']
    
    for i, model in enumerate(models):
        # Improvement factor increases with epochs
        improvement_factor = (i + 1) * 0.15
        
        # Main classifier scores
        data[f'{model}_roberta_toxicity_score'] = data['base_roberta_toxicity_score'] * (1 - improvement_factor + np.random.normal(0, 0.05, n_samples))
        data[f'{model}_dynabench_hate_score'] = data['base_dynabench_hate_score'] * (1 - improvement_factor + np.random.normal(0, 0.05, n_samples))
        
        # Toxic-BERT category scores with different improvement patterns
        for j, category in enumerate(toxic_bert_categories):
            # Different categories improve at different rates
            category_improvement = improvement_factor * (1 + j * 0.1)  # Some categories improve more than others
            data[f'{model}_toxic_bert_{category}_score'] = data[f'base_toxic_bert_{category}_score'] * (1 - category_improvement + np.random.normal(0, 0.05, n_samples))
    
    # Add full text scores
    for model in ['base'] + models:
        data[f'full_{model}_roberta_toxicity_score'] = data[f'{model}_roberta_toxicity_score'] * 0.8  # Full text slightly less toxic
    
    df = pd.DataFrame(data)
    
    # Ensure scores are between 0 and 1
    score_columns = [col for col in df.columns if col.endswith('_score')]
    for col in score_columns:
        df[col] = df[col].clip(0, 1)
    
    print(f"✅ Created sample data with {len(df)} rows and {len(df.columns)} columns")
    print(f"📋 Toxic-BERT categories: {toxic_bert_categories}")
    print(f"🤖 Models: {['base'] + models}")
    
    return df

def test_enhanced_visualization():
    """Test the enhanced visualization with Toxic-BERT categories."""
    print("\n🚀 Testing Enhanced Toxic-BERT Category Analysis")
    print("="*60)
    
    # Create sample data
    df = create_sample_data_with_toxic_bert_categories()
    
    # Create output directory
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Create config
    config = {
        "experiment": {
            "name": "test_enhanced_toxic_bert_analysis"
        },
        "logging": {
            "use_wandb": False,  # Set to False for testing without WandB
            "wandb_project": "test-toxicity-evaluation",
            "wandb_entity": None,
            "wandb_tags": ["test", "enhanced", "toxic_bert"]
        },
        "visualization": {
            "create_plots": True,
            "save_plots": True,
            "plot_format": "html"
        }
    }
    
    # Initialize visualization manager
    viz_manager = VisualizationManager(config, output_dir)
    
    try:
        # Create comprehensive visualizations including Toxic-BERT categories
        metrics = {
            "total_samples": len(df),
            "models_evaluated": 6,
            "classifiers_used": 3
        }
        
        print("\n🎨 Creating comprehensive visualizations...")
        viz_manager.create_comprehensive_visualizations(df, metrics)
        
        print("\n✅ Enhanced visualization test completed successfully!")
        print(f"📁 Output saved to: {output_dir}")
        
        # Show some sample data
        print("\n📊 Sample Toxic-BERT category data:")
        toxic_bert_cols = [col for col in df.columns if 'toxic_bert' in col and col.endswith('_score')]
        print(df[toxic_bert_cols].head())
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        viz_manager.cleanup()

def test_standalone_enhanced_analysis():
    """Test the standalone enhanced analysis script."""
    print("\n🔍 Testing Standalone Enhanced Analysis")
    print("="*50)
    
    # Create sample data
    df = create_sample_data_with_toxic_bert_categories()
    
    # Save to CSV
    csv_path = Path(__file__).parent / "test_data_with_toxic_bert.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved test data to: {csv_path}")
    
    # Import and test the enhanced analysis
    try:
        from enhanced_wandb_analysis import run_complete_enhanced_analysis
        
        print("\n📈 Running enhanced analysis...")
        result_df, wandb_run = run_complete_enhanced_analysis(str(csv_path), wandb_key=None)
        
        if result_df is not None:
            print("✅ Enhanced analysis completed successfully!")
            
            # Show detected categories
            from enhanced_wandb_analysis import detect_models_and_classifiers
            models, classifiers, toxic_bert_categories = detect_models_and_classifiers(result_df)
            print(f"🎯 Detected Toxic-BERT categories: {toxic_bert_categories}")
            
        else:
            print("❌ Enhanced analysis failed")
            
    except ImportError:
        print("⚠️ Enhanced analysis script not found, skipping standalone test")
    except Exception as e:
        print(f"❌ Error in standalone analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Enhanced Toxic-BERT Category Analysis Test Suite")
    print("="*60)
    
    # Test 1: Enhanced visualization manager
    test_enhanced_visualization()
    
    # Test 2: Standalone enhanced analysis
    test_standalone_enhanced_analysis()
    
    print("\n🎉 All tests completed!") 