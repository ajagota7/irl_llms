#!/usr/bin/env python3
"""
Simple test script that uses the exact same approach as the working dataset_generator.py.
This should work reliably since it mimics the successful implementation.
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
    """Create a configuration that mimics the working dataset_generator.py approach."""
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
                "base_dir": "simple_test_results",
                "save_csv": True,
                "save_json": True,
                "save_plots": False,
                "save_reports": False
            }
        }
    }
    
    return OmegaConf.create(config)


def main():
    """Run a simple test that should work reliably."""
    logger.info("🧪 Starting Simple Test (using working dataset_generator.py approach)")
    logger.info("="*70)
    
    try:
        # Create configuration
        config = create_simple_config()
        logger.info("✅ Configuration created")
        
        # Initialize evaluator
        evaluator = ToxicityEvaluator(config)
        logger.info("✅ Evaluator initialized")
        
        # Run evaluation
        logger.info("🔄 Running simple evaluation...")
        results = evaluator.run_evaluation()
        
        # Print results
        logger.info("\n" + "="*70)
        logger.info("✅ SIMPLE TEST COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Results saved to: {results['output_dir']}")
        logger.info(f"Duration: {results['duration']:.2f} seconds")
        logger.info(f"Total prompts processed: {len(results['results_df'])}")
        
        # Show sample results
        df = results['results_df']
        if len(df) > 0:
            logger.info("\n📊 Results Summary:")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Show toxicity scores
            toxicity_cols = [col for col in df.columns if col.endswith('_score')]
            if toxicity_cols:
                logger.info("\nToxicity Scores:")
                for col in toxicity_cols:
                    scores = df[col].dropna()
                    if len(scores) > 0:
                        logger.info(f"  {col}: mean={scores.mean():.4f}, std={scores.std():.4f}")
            
            # Show sample prompts
            logger.info("\nSample Prompts:")
            for i, row in df.head(3).iterrows():
                prompt = row.get('prompt', 'N/A')
                logger.info(f"  {i+1}: {prompt[:50]}...")
        
        logger.info("\n🎉 Simple test completed successfully!")
        logger.info("The pipeline is working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 