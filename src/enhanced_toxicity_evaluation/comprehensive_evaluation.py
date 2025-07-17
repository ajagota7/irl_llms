#!/usr/bin/env python3
"""
Comprehensive evaluation script for detoxification models.
Tests multiple epochs with WandB logging and HuggingFace dataset output.
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


def create_comprehensive_config():
    """Create a comprehensive evaluation configuration."""
    config = {
        "experiment": {
            "name": "comprehensive_detoxification_evaluation",
            "seed": 42,
            "device": "auto"
        },
        "models": {
            "models": [
                {
                    "name": "base",
                    "hf_path": "EleutherAI/pythia-70m",
                    "type": "base_model",
                    "description": "Original Pythia-70m model"
                },
                {
                    "name": "detox_epoch_20",
                    "hf_path": "ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-20",
                    "type": "detoxified_model",
                    "description": "Detoxified Pythia-70m model - Epoch 20"
                },
                {
                    "name": "detox_epoch_40",
                    "hf_path": "ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-40",
                    "type": "detoxified_model",
                    "description": "Detoxified Pythia-70m model - Epoch 40"
                },
                {
                    "name": "detox_epoch_60",
                    "hf_path": "ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-60",
                    "type": "detoxified_model",
                    "description": "Detoxified Pythia-70m model - Epoch 60"
                },
                {
                    "name": "detox_epoch_80",
                    "hf_path": "ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-80",
                    "type": "detoxified_model",
                    "description": "Detoxified Pythia-70m model - Epoch 80"
                },
                {
                    "name": "detox_epoch_100",
                    "hf_path": "ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-100",
                    "type": "detoxified_model",
                    "description": "Detoxified Pythia-70m model - Epoch 100"
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
                "max_prompts": 2000,  # Large dataset for comprehensive evaluation
                "shuffle_seed": 42
            },
            "filtering": {
                "strategy": "toxicity_threshold",
                "toxicity_threshold": 0.2,  # Lower threshold to include more toxic prompts
                "metrics": ["toxicity"]
            },
            "processing": {
                "min_prompt_length": 10,
                "max_prompt_length": 200,
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
                    "batch_size": 16,  # Larger batch size for efficiency
                    "max_length": 256
                }
            },
            "evaluation": {
                "parallel": True,  # Enable parallel processing
                "max_workers": 4,
                "timeout": 120
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
                "full_text": True  # Include full text evaluation
            },
            "metrics": {
                "basic": ["mean_toxicity", "std_toxicity", "median_toxicity"],
                "thresholds": ["high_toxicity_rate", "medium_toxicity_rate", "low_toxicity_rate"]
            },
            "comparison": {
                "baseline_model": "base"
            }
        },
        "generation": {
            "max_new_tokens": 100,  # Longer outputs for better evaluation
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 1.0,
            "use_cache": True,
            "pad_token_id": 50256,
            "batch_size": 8,  # Larger batch size for efficiency
            "max_length": 512
        },
        "logging": {
            "use_wandb": True,  # Enable WandB logging
            "wandb_project": "detoxification-evaluation",
            "wandb_entity": None,  # Set to your username if needed
            "wandb_tags": ["detoxification", "pythia-70m", "comprehensive"],
            "log_level": "INFO"
        },
        "output": {
            "local": {
                "enabled": True,
                "base_dir": "comprehensive_results",
                "save_csv": True,
                "save_json": True,
                "save_plots": True,
                "save_reports": True
            },
            "huggingface": {
                "enabled": True,
                "dataset_name": "detoxification-evaluation-results",
                "organization": None,  # Set to your org if needed
                "private": False,
                "token": None  # Set your HF token if needed
            }
        }
    }
    
    return OmegaConf.create(config)


def run_comprehensive_evaluation():
    """Run the comprehensive evaluation."""
    logger.info("🚀 Starting Comprehensive Detoxification Evaluation")
    logger.info("="*70)
    
    try:
        # Create configuration
        config = create_comprehensive_config()
        logger.info("✅ Configuration created")
        
        # Initialize evaluator
        evaluator = ToxicityEvaluator(config)
        logger.info("✅ Evaluator initialized")
        
        # Get evaluation info
        eval_info = evaluator.get_evaluation_info()
        logger.info("📋 Evaluation Configuration:")
        for key, value in eval_info.items():
            logger.info(f"  {key}: {value}")
        
        # Run evaluation
        logger.info("🚀 Starting comprehensive evaluation...")
        results = evaluator.run_evaluation()
        
        # Print results summary
        logger.info("\n" + "="*70)
        logger.info("✅ COMPREHENSIVE EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Results saved to: {results['output_dir']}")
        logger.info(f"Duration: {results['duration']:.2f} seconds")
        logger.info(f"Total prompts: {len(results['results_df'])}")
        
        # Check results
        df = results['results_df']
        logger.info(f"DataFrame shape: {df.shape}")
        
        # Check for toxicity scores
        toxicity_cols = [col for col in df.columns if col.endswith('_score')]
        logger.info(f"Toxicity score columns: {toxicity_cols}")
        
        # Print detailed comparison results
        if 'metrics' in results and 'comparison_metrics' in results['metrics']:
            comparison = results['metrics']['comparison_metrics']
            if comparison:
                logger.info("\n📊 DETOXIFICATION EFFECTIVENESS RESULTS:")
                logger.info("-" * 50)
                
                for model_name, classifier_results in comparison.items():
                    logger.info(f"\n{model_name.upper()}:")
                    for classifier_name, metrics in classifier_results.items():
                        improvement = metrics.get('improvement', 0)
                        improved_rate = metrics.get('improved_rate', 0)
                        baseline_mean = metrics.get('baseline_mean', 0)
                        model_mean = metrics.get('model_mean', 0)
                        
                        logger.info(f"  {classifier_name}:")
                        logger.info(f"    Toxicity Improvement: {improvement:.4f}")
                        logger.info(f"    Improved Samples: {improved_rate:.2%}")
                        logger.info(f"    Baseline Mean: {baseline_mean:.4f}")
                        logger.info(f"    Model Mean: {model_mean:.4f}")
        
        # Print model performance summary
        if 'metrics' in results and 'model_metrics' in results['metrics']:
            model_metrics = results['metrics']['model_metrics']
            logger.info("\n📈 MODEL PERFORMANCE SUMMARY:")
            logger.info("-" * 50)
            
            for model_name, metrics in model_metrics.items():
                mean_toxicity = metrics.get('mean', 0)
                std_toxicity = metrics.get('std', 0)
                logger.info(f"{model_name}: mean={mean_toxicity:.4f}, std={std_toxicity:.4f}")
        
        logger.info("\n🎉 Comprehensive evaluation completed successfully!")
        logger.info("📊 Check WandB dashboard for detailed metrics and visualizations")
        logger.info("📁 Results saved locally and to HuggingFace dataset")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_evaluation()
    sys.exit(0 if success else 1) 