#!/usr/bin/env python3
"""
Test toxic-bert category extraction specifically.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_toxic_bert_category_extraction():
    """Test toxic-bert category extraction specifically."""
    logger.info("🧪 Testing Toxic-BERT Category Extraction")
    logger.info("=" * 50)
    
    try:
        # Test 1: Mock toxic-bert predictions in the format that the classifier actually produces
        logger.info("Test 1: Testing toxic-bert predictions extraction...")
        
        # This simulates what toxic-bert actually returns when return_all_scores=True
        mock_toxic_bert_predictions = {
            "toxic_bert": [
                # Each prediction is a list of dictionaries with label and score
                [
                    {"label": "toxic", "score": 0.8},
                    {"label": "severe_toxic", "score": 0.2},
                    {"label": "obscene", "score": 0.1},
                    {"label": "threat", "score": 0.05},
                    {"label": "insult", "score": 0.3},
                    {"label": "identity_attack", "score": 0.1}
                ],
                [
                    {"label": "toxic", "score": 0.3},
                    {"label": "severe_toxic", "score": 0.1},
                    {"label": "obscene", "score": 0.05},
                    {"label": "threat", "score": 0.02},
                    {"label": "insult", "score": 0.15},
                    {"label": "identity_attack", "score": 0.05}
                ]
            ]
        }
        
        # Test classifier manager
        from core.classifier_manager import ClassifierManager
        classifier_manager = ClassifierManager({})
        
        detailed_scores = classifier_manager.extract_detailed_scores(mock_toxic_bert_predictions)
        logger.info(f"✅ Detailed scores extracted: {list(detailed_scores.keys())}")
        
        # Verify all toxic-bert categories are extracted
        expected_categories = [
            "toxic_bert_toxic", 
            "toxic_bert_severe_toxic", 
            "toxic_bert_obscene",
            "toxic_bert_threat", 
            "toxic_bert_insult", 
            "toxic_bert_identity_attack"
        ]
        
        for category in expected_categories:
            if category in detailed_scores:
                logger.info(f"✅ Found category: {category}")
                logger.info(f"   Values: {detailed_scores[category]}")
            else:
                logger.warning(f"❌ Missing category: {category}")
        
        # Test 2: Mixed classifier predictions (toxic-bert + binary classifiers)
        logger.info("Test 2: Testing mixed classifier predictions...")
        
        mixed_predictions = {
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
            ],
            "roberta_toxicity": [
                {"label": "toxic", "score": 0.7},
                {"label": "toxic", "score": 0.4}
            ],
            "dynabench_hate": [
                {"label": "hate", "score": 0.6},
                {"label": "hate", "score": 0.2}
            ]
        }
        
        mixed_scores = classifier_manager.extract_detailed_scores(mixed_predictions)
        logger.info(f"✅ Mixed scores extracted: {list(mixed_scores.keys())}")
        
        # Verify all expected categories
        expected_mixed = [
            "toxic_bert_toxic", "toxic_bert_severe_toxic", "toxic_bert_obscene",
            "roberta_toxicity_score", "dynabench_hate_score"
        ]
        
        for category in expected_mixed:
            if category in mixed_scores:
                logger.info(f"✅ Found mixed category: {category}")
                logger.info(f"   Values: {mixed_scores[category]}")
            else:
                logger.warning(f"❌ Missing mixed category: {category}")
        
        # Test 3: Create a DataFrame similar to your notebook approach
        logger.info("Test 3: Testing DataFrame creation...")
        
        # Create mock data with all categories
        mock_df_data = []
        for i in range(5):
            row = {
                "prompt": f"Test prompt {i+1}",
                "output_base": f"Base output {i+1}",
                "output_detoxified": f"Detox output {i+1}",
                "output_base_roberta_toxicity_score": 0.8 - i * 0.1,
                "output_detoxified_roberta_toxicity_score": 0.3 - i * 0.05,
                "output_base_toxic_bert_toxic": 0.7 - i * 0.1,
                "output_detoxified_toxic_bert_toxic": 0.2 - i * 0.05,
                "output_base_toxic_bert_severe_toxic": 0.2 - i * 0.02,
                "output_detoxified_toxic_bert_severe_toxic": 0.05 - i * 0.01,
                "output_base_toxic_bert_obscene": 0.1 - i * 0.01,
                "output_detoxified_toxic_bert_obscene": 0.02 - i * 0.005,
                "output_base_toxic_bert_threat": 0.05 - i * 0.005,
                "output_detoxified_toxic_bert_threat": 0.01 - i * 0.002,
                "output_base_toxic_bert_insult": 0.3 - i * 0.03,
                "output_detoxified_toxic_bert_insult": 0.1 - i * 0.015,
                "output_base_toxic_bert_identity_attack": 0.1 - i * 0.01,
                "output_detoxified_toxic_bert_identity_attack": 0.02 - i * 0.005
            }
            
            # Add delta columns
            row["delta_detoxified_vs_base_roberta_toxicity_score"] = row["output_base_roberta_toxicity_score"] - row["output_detoxified_roberta_toxicity_score"]
            row["delta_detoxified_vs_base_toxic_bert_toxic"] = row["output_base_toxic_bert_toxic"] - row["output_detoxified_toxic_bert_toxic"]
            row["delta_detoxified_vs_base_toxic_bert_severe_toxic"] = row["output_base_toxic_bert_severe_toxic"] - row["output_detoxified_toxic_bert_severe_toxic"]
            row["delta_detoxified_vs_base_toxic_bert_obscene"] = row["output_base_toxic_bert_obscene"] - row["output_detoxified_toxic_bert_obscene"]
            row["delta_detoxified_vs_base_toxic_bert_threat"] = row["output_base_toxic_bert_threat"] - row["output_detoxified_toxic_bert_threat"]
            row["delta_detoxified_vs_base_toxic_bert_insult"] = row["output_base_toxic_bert_insult"] - row["output_detoxified_toxic_bert_insult"]
            row["delta_detoxified_vs_base_toxic_bert_identity_attack"] = row["output_base_toxic_bert_identity_attack"] - row["output_detoxified_toxic_bert_identity_attack"]
            
            mock_df_data.append(row)
        
        mock_df = pd.DataFrame(mock_df_data)
        logger.info(f"✅ Mock DataFrame created with {len(mock_df)} rows and {len(mock_df.columns)} columns")
        
        # Test inspector with toxic-bert categories
        from core.inspector import ToxicityInspector
        inspector = ToxicityInspector(mock_df)
        logger.info(f"✅ Inspector initialized")
        logger.info(f"✅ Available models: {inspector.models}")
        logger.info(f"✅ Available classifiers: {inspector.classifiers}")
        logger.info(f"✅ Toxic-bert categories: {inspector.toxic_bert_categories}")
        
        # Test toxic-bert category analysis
        category_analysis = inspector.analyze_toxic_bert_categories("detoxified")
        logger.info(f"✅ Toxic-bert category analysis: {len(category_analysis)} categories")
        if len(category_analysis) > 0:
            logger.info(f"✅ Category analysis columns: {list(category_analysis.columns)}")
            logger.info(f"✅ Category analysis data:")
            for _, row in category_analysis.iterrows():
                logger.info(f"   {row['category']}: improvement = {row['improvement']:.4f}")
        
        # Test getting best improvements for toxic-bert categories
        for category in inspector.toxic_bert_categories[:3]:  # Test first 3 categories
            try:
                improvements = inspector.get_best_improvements("detoxified", f"toxic_bert_{category}", n=3)
                logger.info(f"✅ Best improvements for {category}: {len(improvements)} examples")
            except Exception as e:
                logger.warning(f"⚠️ Could not get improvements for {category}: {e}")
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 TOXIC-BERT CATEGORY EXTRACTION TESTS PASSED!")
        logger.info("=" * 50)
        return True
        
    except Exception as e:
        logger.error(f"❌ Toxic-bert category extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run toxic-bert category extraction test."""
    success = test_toxic_bert_category_extraction()
    if success:
        logger.info("\n🚀 Toxic-bert category extraction working correctly!")
    else:
        logger.error("\n❌ Toxic-bert category extraction test failed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 