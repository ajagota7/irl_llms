#!/usr/bin/env python3
"""
Example usage of enhanced toxicity evaluation pipeline with new features.
"""

import logging
from pathlib import Path
from omegaconf import OmegaConf
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.evaluator import ToxicityEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_enhanced_config():
    """Create configuration with enhanced features enabled."""
    config = {
        "experiment": {
            "name": "enhanced_toxicity_eval",
            "seed": 42,
            "device": "auto"
        },
        "models": {
            "models": [
                {
                    "name": "base",
                    "hf_path": "EleutherAI/pythia-70m",
                    "type": "base_model"
                },
                {
                    "name": "detoxified",
                    "hf_path": "ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-100",
                    "type": "detoxified_model"
                }
            ]
        },
        "dataset": {
            "dataset": {
                "name": "allenai/real-toxicity-prompts",
                "max_prompts": 50
            },
            "filtering": {
                "strategy": "toxicity_threshold",
                "toxicity_threshold": 0.4
            }
        },
        "classifiers": {
            "classifiers": {
                "roberta_toxicity": {
                    "enabled": True,
                    "model": "s-nlp/roberta_toxicity_classifier",
                    "type": "binary",
                    "batch_size": 16
                },
                "toxic_bert": {
                    "enabled": True,
                    "model": "unitary/toxic-bert",
                    "type": "multi_label",
                    "batch_size": 8,
                    "return_all_scores": True
                }
            }
        },
        "evaluation": {
            "comparison": {
                "baseline_model": "base"
            }
        },
        "output": {
            "local": {
                "enabled": True,
                "enhanced_analysis": True
            }
        },
        "enhanced_analysis": {
            "enabled": True,
            "inspector": {
                "enabled": True,
                "export_reports": True
            },
            "visualizer": {
                "enabled": True,
                "save_html": True
            }
        }
    }
    
    return OmegaConf.create(config)


def main():
    """Run enhanced evaluation example."""
    logger.info("🚀 Starting Enhanced Toxicity Evaluation Example")
    
    # Create configuration
    config = create_enhanced_config()
    
    # Initialize evaluator
    evaluator = ToxicityEvaluator(config)
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Access enhanced components
    df = results['results_df']
    
    # Create inspector for interactive analysis
    inspector = evaluator.create_inspector(df)
    
    # Example inspector usage
    logger.info("\n📊 Inspector Analysis Examples:")
    
    # Get best improvements
    try:
        improvements = inspector.get_best_improvements("detoxified", "roberta", n=5)
        logger.info(f"Best improvements found: {len(improvements)}")
        if len(improvements) > 0:
            logger.info(f"Top improvement: {improvements.iloc[0]['delta_detoxified_vs_base_roberta_score']:.4f}")
    except Exception as e:
        logger.warning(f"Could not get improvements: {e}")
    
    # Analyze toxic-bert categories
    try:
        category_analysis = inspector.analyze_toxic_bert_categories("detoxified")
        logger.info(f"Toxic-bert categories analyzed: {len(category_analysis)}")
        if len(category_analysis) > 0:
            best_category = category_analysis.loc[category_analysis['improvement'].idxmax()]
            logger.info(f"Best improved category: {best_category['category']} ({best_category['improvement']:.4f})")
    except Exception as e:
        logger.warning(f"Could not analyze toxic-bert categories: {e}")
    
    # Create visualizer
    visualizer = evaluator.create_visualizer(df)
    
    # Example visualizer usage
    logger.info("\n📈 Visualizer Examples:")
    try:
        # Create comparison plot
        fig = visualizer.plot_model_comparison_scatter("detoxified", "roberta")
        logger.info("Model comparison scatter plot created")
        
        # Create delta analysis
        fig = visualizer.plot_delta_analysis("detoxified")
        logger.info("Delta analysis plot created")
        
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")
    
    # Get interactive summary
    summary = inspector.interactive_summary()
    logger.info(f"\n📋 Summary: {summary['total_samples']} samples, {len(summary['models'])} models")
    
    logger.info(f"\n✅ Enhanced evaluation complete! Results in: {results['output_dir']}")


if __name__ == "__main__":
    main() 