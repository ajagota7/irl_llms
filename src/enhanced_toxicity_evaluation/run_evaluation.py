#!/usr/bin/env python3
"""
Main script for running the Enhanced Toxicity Evaluation Pipeline.
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
    """Main entry point for the toxicity evaluation pipeline."""
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
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
    
    logger.info(f"Set random seed to {seed}")
    
    # Initialize evaluator
    evaluator = ToxicityEvaluator(cfg)
    
    # Print evaluation info
    eval_info = evaluator.get_evaluation_info()
    logger.info("Evaluation Configuration:")
    for key, value in eval_info.items():
        logger.info(f"  {key}: {value}")
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Results saved to: {results['output_dir']}")
        logger.info(f"Duration: {results['duration']:.2f} seconds")
        logger.info(f"Total prompts: {len(results['results_df'])}")
        logger.info(f"Models evaluated: {len([col for col in results['results_df'].columns if col.startswith('output_') and not col.endswith('_score')])}")
        
        # Print key metrics
        metrics = results['metrics']
        if 'comparison_metrics' in metrics:
            logger.info("\nKey Improvement Metrics:")
            for model_name, model_comparisons in metrics['comparison_metrics'].items():
                for classifier_name, comp_metrics in model_comparisons.items():
                    improvement = comp_metrics.get('improvement', 0.0)
                    improved_rate = comp_metrics.get('improved_rate', 0.0)
                    logger.info(f"  {model_name} vs base ({classifier_name}): {improvement:.4f} improvement, {improved_rate:.2%} improved")
        
        logger.info("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 