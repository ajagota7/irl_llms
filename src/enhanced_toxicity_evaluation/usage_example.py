#!/usr/bin/env python3
"""
Simple usage example for the Enhanced Toxicity Evaluation Pipeline.
This example demonstrates basic usage with a small test setup.
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


def create_simple_config():
    """Create a simple configuration for testing."""
    config = {
        "experiment": {
            "name": "simple_test",
        "seed": 42,
            "device": "auto"
        },
        "models": {
            "models": [
                {
                    "name": "test_model",
                    "hf_path": "EleutherAI/pythia-70m",
                    "type": "base_model",
                    "description": "Simple test model"
                }
            ],
            "model_loading": {
                "device_map": "auto",
                "torch_dtype": "auto",
                "trust_remote_code": True
            }
        },
        "dataset": {
            "dataset": {
                "name": "allenai/real-toxicity-prompts",
                "split": "train",
                "max_prompts": 3,  # Very small for quick testing
                "shuffle_seed": 42
            },
            "filtering": {
                "strategy": "toxicity_threshold",
                "toxicity_threshold": 0.5,
                "metrics": ["toxicity"]
            },
            "processing": {
                "min_prompt_length": 5,
                "max_prompt_length": 100,
                "remove_duplicates": True,
                "normalize_whitespace": True
            },
            "caching": {
                "use_cache": False
            }
        },
        "classifiers": {
            "classifiers": {
                "roberta_toxicity": {
                    "enabled": True,
                    "model": "s-nlp/roberta_toxicity_classifier",
                    "batch_size": 2,
                    "max_length": 128
                }
            },
            "evaluation": {
                "parallel": False,
                "max_workers": 1,
                "timeout": 30
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
                "full_text": False
            },
            "metrics": {
                "basic": ["mean_toxicity", "std_toxicity"],
                "thresholds": ["high_toxicity_rate"]
            },
            "comparison": {
                "baseline_model": "test_model"
            }
        },
        "generation": {
            "max_new_tokens": 10,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 1.0,
            "use_cache": True,
            "pad_token_id": 50256,
            "batch_size": 2,
            "max_length": 256
        },
        "logging": {
            "use_wandb": False,
            "log_level": "INFO"
        },
        "output": {
            "local": {
                "enabled": True,
                "base_dir": "example_results",
                "save_csv": True,
                "save_json": True,
                "save_plots": False,
                "save_reports": False
            }
        }
    }
    
    return OmegaConf.create(config)


def main():
    """Run a simple evaluation example."""
    logger.info("🚀 Starting Simple Toxicity Evaluation Example")
    logger.info("="*50)
    
    try:
        # Create configuration
        config = create_simple_config()
        logger.info("✅ Configuration created")
        
        # Initialize evaluator
        evaluator = ToxicityEvaluator(config)
        logger.info("✅ Evaluator initialized")
        
        # Run evaluation
        logger.info("🔄 Running evaluation...")
        results = evaluator.run_evaluation()
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("✅ EVALUATION COMPLETED")
        logger.info("="*50)
        logger.info(f"Results saved to: {results['output_dir']}")
        logger.info(f"Duration: {results['duration']:.2f} seconds")
        logger.info(f"Total prompts processed: {len(results['results_df'])}")
        
        # Show sample results
        df = results['results_df']
        if len(df) > 0:
            logger.info("\n📊 Sample Results:")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Show first few rows
            logger.info("\nFirst few results:")
            for i, row in df.head(3).iterrows():
                logger.info(f"  Prompt {i+1}: {row.get('prompt', 'N/A')[:50]}...")
                toxicity_cols = [col for col in df.columns if col.endswith('_score')]
                for col in toxicity_cols:
                    if col in row and not pd.isna(row[col]):
                        logger.info(f"    {col}: {row[col]:.4f}")
        
        logger.info("\n🎉 Example completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import pandas as pd
    success = main()
    sys.exit(0 if success else 1)