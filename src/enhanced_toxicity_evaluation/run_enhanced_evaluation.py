#!/usr/bin/env python3
"""
Enhanced script for running the Improved Toxicity Evaluation Pipeline.
This script provides better command-line interface and comprehensive output.
"""

import os
import sys
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from core.evaluator import ToxicityEvaluator

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for the enhanced toxicity evaluation pipeline."""
    
    # Print enhanced configuration summary
    logger.info("="*80)
    logger.info("🚀 ENHANCED TOXICITY EVALUATION PIPELINE")
    logger.info("="*80)
    
    # Print key configuration details
    logger.info("📋 Configuration Summary:")
    logger.info(f"  Experiment Name: {cfg.experiment.get('name', 'default')}")
    logger.info(f"  Models: {len(cfg.models.get('models', []))} models configured")
    logger.info(f"  Dataset: {cfg.dataset.get('dataset', {}).get('max_prompts', 'unknown')} max prompts")
    logger.info(f"  Classifiers: {len(cfg.classifiers.get('classifiers', {}))} classifiers")
    logger.info(f"  Generation: {cfg.generation.get('max_new_tokens', 'unknown')} max tokens")
    logger.info(f"  Visualization: {'Enabled' if cfg.visualization.get('enabled', True) else 'Disabled'}")
    
    # Set random seed
    import random
    import numpy as np
    import torch
    
    seed = cfg.experiment.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"🎲 Set random seed to {seed}")
    
    # Initialize evaluator
    logger.info("🔧 Initializing evaluator...")
    evaluator = ToxicityEvaluator(cfg)
    
    # Print detailed evaluation info
    eval_info = evaluator.get_evaluation_info()
    logger.info("\n📊 Evaluation Configuration:")
    for key, value in eval_info.items():
        if isinstance(value, list) and len(value) > 5:
            logger.info(f"  {key}: {len(value)} items")
        else:
            logger.info(f"  {key}: {value}")
    
    # Run evaluation
    try:
        logger.info("\n" + "="*80)
        logger.info("🔄 STARTING ENHANCED EVALUATION")
        logger.info("="*80)
        
        results = evaluator.run_evaluation()
        
        # Print comprehensive summary
        logger.info("\n" + "="*80)
        logger.info("✅ ENHANCED EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"📁 Results Directory: {results['output_dir']}")
        logger.info(f"⏱️  Duration: {results['duration']:.2f} seconds")
        logger.info(f"📊 Total Prompts: {len(results['results_df'])}")
        logger.info(f"🤖 Models Evaluated: {len([col for col in results['results_df'].columns if col.startswith('output_') and not col.endswith('_score')])}")
        logger.info(f"🔍 Classifiers Used: {len(set([col.split('_')[-2] for col in results['results_df'].columns if col.endswith('_score') and 'prompt_' not in col]))}")
        
        # Print key metrics
        metrics = results['metrics']
        if 'comparison_metrics' in metrics:
            logger.info("\n📈 Key Improvement Metrics:")
            for model_name, model_comparisons in metrics['comparison_metrics'].items():
                for classifier_name, comp_metrics in model_comparisons.items():
                    improvement = comp_metrics.get('improvement', 0.0)
                    improved_rate = comp_metrics.get('improved_rate', 0.0)
                    p_value = comp_metrics.get('p_value', 'N/A')
                    logger.info(f"  {model_name} vs base ({classifier_name}):")
                    logger.info(f"    Improvement: {improvement:.4f}")
                    logger.info(f"    Improved Rate: {improved_rate:.2%}")
                    if p_value != 'N/A':
                        logger.info(f"    P-value: {p_value:.4f}")
        
        # Print file outputs
        output_dir = Path(results['output_dir'])
        logger.info("\n📁 Generated Files:")
        
        # Main results
        if (output_dir / "toxicity_evaluation_results.csv").exists():
            logger.info(f"  📊 Main Results: toxicity_evaluation_results.csv")
        if (output_dir / "evaluation_summary.json").exists():
            logger.info(f"  📋 Summary: evaluation_summary.json")
        
        # Dataset exports
        dataset_dir = output_dir / "dataset_exports"
        if dataset_dir.exists():
            logger.info(f"  📦 Dataset Exports: {dataset_dir}")
            for file in dataset_dir.glob("*"):
                if file.is_file():
                    logger.info(f"    - {file.name}")
        
        # Plots
        plots_dir = output_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            logger.info(f"  📈 Visualizations: {len(plot_files)} plots generated")
            for plot_file in plot_files[:5]:  # Show first 5
                logger.info(f"    - {plot_file.name}")
            if len(plot_files) > 5:
                logger.info(f"    ... and {len(plot_files) - 5} more")
        
        # Reports
        if (output_dir / "evaluation_report.html").exists():
            logger.info(f"  📄 HTML Report: evaluation_report.html")
        
        # Logs
        logs_dir = output_dir / "logs"
        if logs_dir.exists():
            logger.info(f"  📝 Logs: {logs_dir}")
        
        logger.info("\n" + "="*80)
        logger.info("🎉 ENHANCED EVALUATION PIPELINE COMPLETED!")
        logger.info("="*80)
        logger.info("💡 Next Steps:")
        logger.info("  1. Check the generated plots in the plots/ directory")
        logger.info("  2. Review the HTML report for comprehensive analysis")
        logger.info("  3. Use the dataset exports for further analysis")
        logger.info("  4. Check WandB dashboard for interactive visualizations")
        
    except Exception as e:
        logger.error(f"❌ Enhanced evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 