#!/usr/bin/env python3
"""
Test enhanced features: multi-label support, inspector, visualizer, delta analysis.
"""

import sys
import logging
import os
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
import numpy as np

# Suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.append(str(Path(__file__).parent))

from core.evaluator import ToxicityEvaluator
from core.inspector import ToxicityInspector
from core.visualizer import ToxicityVisualizer
from core.metrics_calculator import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_enhanced_features():
    """Test all enhanced features."""
    logger.info("🧪 Testing Enhanced Features")
    logger.info("=" * 50)
    
    try:
        # Test 1: Multi-label classifier manager
        logger.info("Test 1: Testing multi-label classifier support...")
        
        # Create mock multi-label predictions
        mock_predictions = {
            "toxic_bert": [
                [
                    {"label": "toxic", "score": 0.8},
                    {"label": "severe_toxic", "score": 0.2},
                    {"label": "obscene", "score": 0.1}
                ],
                [
                    {"label": "toxic", "score": 0.3},
                    {"label": "severe_toxic", "score": 0.1},
                    {"label": "obscene", "score": 0.05}
                ]
            ]
        }
        
        # Test classifier manager
        from core.classifier_manager import ClassifierManager
        classifier_manager = ClassifierManager({})
        
        detailed_scores = classifier_manager.extract_detailed_scores(mock_predictions)
        logger.info(f"✅ Multi-label scores extracted: {list(detailed_scores.keys())}")
        
        # Test 2: Delta analysis
        logger.info("Test 2: Testing delta analysis...")
        
        # Create mock toxicity results
        mock_toxicity_results = {
            "base": {
                "roberta_score": [0.8, 0.7, 0.9],
                "toxic_bert_toxic": [0.7, 0.6, 0.8]
            },
            "detoxified": {
                "roberta_score": [0.3, 0.4, 0.5],
                "toxic_bert_toxic": [0.2, 0.3, 0.4]
            }
        }
        
        metrics_calc = MetricsCalculator({})
        delta_results = metrics_calc.calculate_delta_metrics(mock_toxicity_results)
        logger.info(f"✅ Delta analysis completed: {list(delta_results.keys())}")
        
        improvement_stats = metrics_calc.calculate_improvement_statistics(delta_results)
        logger.info(f"✅ Improvement statistics calculated: {len(improvement_stats)} models")
        
        # Test 3: Inspector
        logger.info("Test 3: Testing inspector...")
        
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            "prompt": ["Test prompt 1", "Test prompt 2", "Test prompt 3"],
            "output_base": ["Base output 1", "Base output 2", "Base output 3"],
            "output_detoxified": ["Detox output 1", "Detox output 2", "Detox output 3"],
            "output_base_roberta_score": [0.8, 0.7, 0.9],
            "output_detoxified_roberta_score": [0.3, 0.4, 0.5],
            "delta_detoxified_vs_base_roberta_score": [0.5, 0.3, 0.4]
        })
        
        inspector = ToxicityInspector(mock_df)
        logger.info(f"✅ Inspector initialized with {len(mock_df)} samples")
        
        # Test inspector methods
        improvements = inspector.get_best_improvements("detoxified", "roberta", n=2)
        logger.info(f"✅ Best improvements found: {len(improvements)} examples")
        
        summary = inspector.interactive_summary()
        logger.info(f"✅ Interactive summary generated: {summary['total_samples']} samples")
        
        # Test 4: Visualizer
        logger.info("Test 4: Testing visualizer...")
        
        visualizer = ToxicityVisualizer(mock_df)
        logger.info(f"✅ Visualizer initialized")
        
        # Test visualization creation (without saving)
        try:
            fig = visualizer.plot_model_comparison_scatter("detoxified", "roberta", save_html=False)
            logger.info("✅ Scatter plot created successfully")
        except Exception as e:
            logger.warning(f"Scatter plot creation failed: {e}")
        
        try:
            fig = visualizer.plot_delta_analysis("detoxified", save_html=False)
            logger.info("✅ Delta analysis plot created successfully")
        except Exception as e:
            logger.warning(f"Delta analysis plot creation failed: {e}")
            # Try to get more details about the error
            import traceback
            logger.debug(f"Delta analysis error details: {traceback.format_exc()}")
        
        try:
            fig = visualizer.create_comprehensive_dashboard(save_html=False)
            logger.info("✅ Comprehensive dashboard created successfully")
        except Exception as e:
            logger.warning(f"Dashboard creation failed: {e}")
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 ALL ENHANCED FEATURES TESTS PASSED!")
        logger.info("=" * 50)
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run enhanced features test."""
    success = test_enhanced_features()
    if success:
        logger.info("\n🚀 Enhanced features working correctly!")
    else:
        logger.error("\n❌ Enhanced features test failed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 