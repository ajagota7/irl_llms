#!/usr/bin/env python3
"""
Very basic test script that only tests core functionality.
No complex dataset loading, no baseline comparisons, just the essentials.
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


def create_basic_config():
    """Create a very basic configuration for testing core functionality."""
    config = {
        "experiment": {
            "name": "basic_test",
            "seed": 42,
            "device": "auto"
        },
        "models": {
            "models": [
                {
                    "name": "basic_model",
                    "hf_path": "EleutherAI/pythia-70m",
                    "type": "base_model",
                    "description": "Basic test model"
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
                "max_prompts": 2,  # Very small for basic testing
                "shuffle_seed": 42
            },
            "filtering": {
                "strategy": "toxicity_threshold",
                "toxicity_threshold": 0.5,
                "metrics": ["toxicity"]
            },
            "processing": {
                "min_prompt_length": 5,
                "max_prompt_length": 50,  # Very short for basic testing
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
                    "batch_size": 1,  # Very small batch size
                    "max_length": 64   # Very short sequences
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
                "basic": ["mean_toxicity"],  # Only basic metrics
                "thresholds": []  # No threshold metrics for basic test
            },
            "comparison": {
                "baseline_model": "basic_model"  # Same as the model name
            }
        },
        "generation": {
            "max_new_tokens": 5,  # Very short outputs
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 1.0,
            "use_cache": True,
            "pad_token_id": 50256,
            "batch_size": 1,  # Very small batch size
            "max_length": 128  # Very short sequences
        },
        "logging": {
            "use_wandb": False,
            "log_level": "INFO"
        },
        "output": {
            "local": {
                "enabled": True,
                "base_dir": "basic_test_results",
                "save_csv": True,
                "save_json": True,
                "save_plots": False,
                "save_reports": False
            }
        }
    }
    
    return OmegaConf.create(config)


def test_basic_functionality():
    """Test only the most basic functionality."""
    logger.info("🧪 Starting Basic Functionality Test")
    logger.info("="*50)
    
    try:
        # Test 1: Configuration creation
        logger.info("Test 1: Creating configuration...")
        config = create_basic_config()
        logger.info("✅ Configuration created successfully")
        
        # Test 2: Evaluator initialization
        logger.info("Test 2: Initializing evaluator...")
        evaluator = ToxicityEvaluator(config)
        logger.info("✅ Evaluator initialized successfully")
        
        # Test 3: Get evaluation info
        logger.info("Test 3: Getting evaluation info...")
        eval_info = evaluator.get_evaluation_info()
        logger.info(f"✅ Evaluation info: {len(eval_info)} items")
        
        # Test 4: Run basic evaluation
        logger.info("Test 4: Running basic evaluation...")
        results = evaluator.run_evaluation()
        logger.info("✅ Basic evaluation completed")
        
        # Test 5: Check results structure
        logger.info("Test 5: Checking results structure...")
        required_keys = ['results_df', 'output_dir', 'duration']
        for key in required_keys:
            if key in results:
                logger.info(f"✅ Found {key}: {type(results[key])}")
            else:
                logger.error(f"❌ Missing {key}")
                return False
        
        # Test 6: Check DataFrame
        df = results['results_df']
        logger.info(f"✅ DataFrame shape: {df.shape}")
        logger.info(f"✅ DataFrame columns: {list(df.columns)}")
        
        if len(df) > 0:
            logger.info("✅ DataFrame has data")
            
            # Show first row
            first_row = df.iloc[0]
            logger.info(f"✅ First row prompt: {first_row.get('prompt', 'N/A')[:30]}...")
            
            # Check for toxicity scores
            toxicity_cols = [col for col in df.columns if col.endswith('_score')]
            if toxicity_cols:
                logger.info(f"✅ Found toxicity columns: {toxicity_cols}")
                for col in toxicity_cols:
                    scores = df[col].dropna()
                    if len(scores) > 0:
                        logger.info(f"✅ {col}: {scores.iloc[0]:.4f}")
        else:
            logger.warning("⚠️ DataFrame is empty")
        
        logger.info("\n" + "="*50)
        logger.info("🎉 ALL BASIC TESTS PASSED!")
        logger.info("="*50)
        logger.info("The core pipeline functionality is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the basic functionality test."""
    success = test_basic_functionality()
    if success:
        logger.info("\n🚀 Ready to proceed with full evaluation!")
    else:
        logger.error("\n❌ Basic functionality test failed. Please check the errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 