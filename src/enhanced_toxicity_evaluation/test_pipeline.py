#!/usr/bin/env python3
"""
Test script for the Enhanced Toxicity Evaluation Pipeline.
This script runs a minimal evaluation to verify everything works correctly.
"""

import os
import sys
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from core.evaluator import ToxicityEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_config():
    """Create a minimal test configuration."""
    config = {
        "experiment": {
            "name": "test_evaluation",
            "seed": 42,
            "device": "auto"
        },
        "models": {
            "models": [
                {
                    "name": "base",
                    "hf_path": "EleutherAI/pythia-70m",  # Use smaller model for testing
                    "type": "base_model",
                    "description": "Test base model"
                }
            ],
            "model_loading": {
                "device_map": "auto",
                "torch_dtype": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
        },
        "dataset": {
            "dataset": {
                "name": "allenai/real-toxicity-prompts",
                "split": "train",
                "max_prompts": 10,  # Very small for testing
                "shuffle_seed": 42
            },
            "filtering": {
                "strategy": "toxicity_threshold",
                "toxicity_threshold": 0.4,
                "metrics": ["toxicity"]
            },
            "processing": {
                "min_prompt_length": 10,
                "max_prompt_length": 200,
                "remove_duplicates": True,
                "normalize_whitespace": True
            },
            "caching": {
                "use_cache": False  # Disable caching for testing
            }
        },
        "classifiers": {
            "classifiers": {
                "roberta_toxicity": {
                    "enabled": True,
                    "model": "s-nlp/roberta_toxicity_classifier",
                    "batch_size": 4,  # Small batch size for testing
                    "max_length": 256
                }
            },
            "evaluation": {
                "parallel": False,  # Disable parallel for testing
                "max_workers": 1,
                "timeout": 60
            },
            "error_handling": {
                "skip_failed_classifiers": True,
                "log_errors": True,
                "fallback_to_safe": True
            }
        },
        "evaluation": {
            "types": {
                "prompt_only": True,
                "output_only": True,
                "full_text": False  # Skip for testing
            },
            "metrics": {
                "basic": ["mean_toxicity", "std_toxicity"],
                "thresholds": ["high_toxicity_rate"]
            },
            "comparison": {
                "baseline_model": "base"
            }
        },
        "generation": {
            "max_new_tokens": 20,  # Short outputs for testing
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 1.0,
            "use_cache": True,
            "pad_token_id": 50256,
            "batch_size": 4,  # Small batch size for testing
            "max_length": 512
        },
        "logging": {
            "use_wandb": False,  # Disable WandB for testing
            "log_level": "INFO"
        },
        "output": {
            "local": {
                "enabled": True,
                "base_dir": "test_results",
                "save_csv": True,
                "save_json": True,
                "save_plots": False,
                "save_reports": False
            }
        }
    }
    
    return OmegaConf.create(config)


def test_pipeline():
    """Test the evaluation pipeline with minimal configuration."""
    logger.info("🧪 Testing Enhanced Toxicity Evaluation Pipeline")
    logger.info("="*60)
    
    try:
        # Create test configuration
        config = create_test_config()
        logger.info("✅ Test configuration created")
        
        # Initialize evaluator
        evaluator = ToxicityEvaluator(config)
        logger.info("✅ Evaluator initialized")
        
        # Get evaluation info
        eval_info = evaluator.get_evaluation_info()
        logger.info("📋 Evaluation Configuration:")
        for key, value in eval_info.items():
            logger.info(f"  {key}: {value}")
        
        # Run evaluation
        logger.info("🚀 Starting test evaluation...")
        results = evaluator.run_evaluation()
        
        # Print results summary
        logger.info("\n" + "="*60)
        logger.info("✅ TEST EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Results saved to: {results['output_dir']}")
        logger.info(f"Duration: {results['duration']:.2f} seconds")
        logger.info(f"Total prompts: {len(results['results_df'])}")
        
        # Check results
        df = results['results_df']
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # Check for toxicity scores
        toxicity_cols = [col for col in df.columns if col.endswith('_score')]
        logger.info(f"Toxicity score columns: {toxicity_cols}")
        
        if toxicity_cols:
            # Print sample scores
            for col in toxicity_cols:
                scores = df[col].dropna()
                if len(scores) > 0:
                    logger.info(f"{col}: mean={scores.mean():.4f}, std={scores.std():.4f}")
        
        logger.info("\n🎉 All tests passed! The pipeline is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1) 