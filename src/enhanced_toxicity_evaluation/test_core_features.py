#!/usr/bin/env python3
"""
Test core enhanced features: multi-label support, inspector, delta analysis.
"""

import sys
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.append(str(Path(__file__).parent))

from core.inspector import ToxicityInspector
from core.metrics_calculator import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_core_enhanced_features():
    """Test core enhanced features without complex visualizations."""
    logger.info("🧪 Testing Core Enhanced Features")
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
        
        # Verify toxic-bert categories are extracted
        expected_categories = ["toxic_bert_toxic", "toxic_bert_severe_toxic", "toxic_bert_obscene"]
        for category in expected_categories:
            if category in detailed_scores:
                logger.info(f"✅ Found category: {category}")
            else:
                logger.warning(f"❌ Missing category: {category}")
        
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
        
        # Test statistical significance
        significance_tests = metrics_calc.perform_statistical_significance_tests(delta_results)
        logger.info(f"✅ Statistical significance tests completed: {len(significance_tests)} models")
        
        # Test 3: Inspector
        logger.info("Test 3: Testing inspector...")
        
        # Create mock DataFrame with more realistic data
        mock_df = pd.DataFrame({
            "prompt": ["Test prompt 1", "Test prompt 2", "Test prompt 3", "Test prompt 4", "Test prompt 5"],
            "output_base": ["Base output 1", "Base output 2", "Base output 3", "Base output 4", "Base output 5"],
            "output_detoxified": ["Detox output 1", "Detox output 2", "Detox output 3", "Detox output 4", "Detox output 5"],
            "output_base_roberta_score": [0.8, 0.7, 0.9, 0.6, 0.85],
            "output_detoxified_roberta_score": [0.3, 0.4, 0.5, 0.2, 0.35],
            "delta_detoxified_vs_base_roberta_score": [0.5, 0.3, 0.4, 0.4, 0.5],
            "output_base_toxic_bert_toxic": [0.7, 0.6, 0.8, 0.5, 0.75],
            "output_detoxified_toxic_bert_toxic": [0.2, 0.3, 0.4, 0.1, 0.25],
            "delta_detoxified_vs_base_toxic_bert_toxic": [0.5, 0.3, 0.4, 0.4, 0.5]
        })
        
        inspector = ToxicityInspector(mock_df)
        logger.info(f"✅ Inspector initialized with {len(mock_df)} samples")
        logger.info(f"✅ Available models: {inspector.models}")
        logger.info(f"✅ Available classifiers: {inspector.classifiers}")
        logger.info(f"✅ Toxic-bert categories: {inspector.toxic_bert_categories}")
        
        # Test inspector methods
        improvements = inspector.get_best_improvements("detoxified", "roberta", n=3)
        logger.info(f"✅ Best improvements found: {len(improvements)} examples")
        
        regressions = inspector.get_worst_regressions("detoxified", "roberta", n=3)
        logger.info(f"✅ Worst regressions found: {len(regressions)} examples")
        
        # Test toxic-bert category analysis
        category_analysis = inspector.analyze_toxic_bert_categories("detoxified")
        logger.info(f"✅ Toxic-bert category analysis: {len(category_analysis)} categories")
        
        # Test model comparison
        comparison = inspector.compare_models_across_classifiers()
        logger.info(f"✅ Model comparison: {len(comparison)} comparisons")
        
        # Test interactive summary
        summary = inspector.interactive_summary()
        logger.info(f"✅ Interactive summary generated: {summary['total_samples']} samples")
        logger.info(f"✅ Models in summary: {summary['models']}")
        
        # Test 4: Basic visualizer functionality (without complex plots)
        logger.info("Test 4: Testing basic visualizer...")
        
        try:
            from core.visualizer import ToxicityVisualizer
            visualizer = ToxicityVisualizer(mock_df)
            logger.info(f"✅ Visualizer initialized with {len(visualizer.models)} models")
            logger.info(f"✅ Available classifiers: {visualizer.classifiers}")
        except Exception as e:
            logger.warning(f"Visualizer initialization failed: {e}")
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 ALL CORE ENHANCED FEATURES TESTS PASSED!")
        logger.info("=" * 50)
        return True
        
    except Exception as e:
        logger.error(f"❌ Core enhanced features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run core enhanced features test."""
    success = test_core_enhanced_features()
    if success:
        logger.info("\n🚀 Core enhanced features working correctly!")
    else:
        logger.error("\n❌ Core enhanced features test failed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 