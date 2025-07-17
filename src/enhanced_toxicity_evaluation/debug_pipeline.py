#!/usr/bin/env python3
"""
Debug script for the Enhanced Toxicity Evaluation Pipeline.
This script runs with detailed logging to identify issues.
"""

import os
import sys
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from core.evaluator import ToxicityEvaluator

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_debug_config():
    """Create a minimal debug configuration."""
    config = {
        "experiment": {
            "name": "debug_evaluation",
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
                    "name": "detoxified",
                    "hf_path": "ajagota71/pythia-70m-s-nlp-detox-checkpoint-epoch-100",
                    "type": "detoxified_model",
                    "description": "Detoxified Pythia-70m model"
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
                "max_prompts": 2,  # Very small for debugging
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
                "use_cache": False
            }
        },
        "classifiers": {
            "classifiers": {
                "roberta_toxicity": {
                    "enabled": True,
                    "model": "s-nlp/roberta_toxicity_classifier",
                    "batch_size": 2,  # Very small batch size for debugging
                    "max_length": 256
                }
            },
            "evaluation": {
                "parallel": False,
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
                "full_text": False
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
            "max_new_tokens": 10,  # Very short outputs for debugging
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 1.0,
            "use_cache": True,
            "pad_token_id": 50256,
            "batch_size": 2,
            "max_length": 512
        },
        "logging": {
            "use_wandb": False,
            "log_level": "DEBUG"
        },
        "output": {
            "local": {
                "enabled": True,
                "base_dir": "debug_results",
                "save_csv": True,
                "save_json": True,
                "save_plots": False,
                "save_reports": False
            }
        }
    }
    
    return OmegaConf.create(config)


def debug_pipeline():
    """Debug the evaluation pipeline with detailed logging."""
    logger.info("🐛 DEBUGGING Enhanced Toxicity Evaluation Pipeline")
    logger.info("="*60)
    
    try:
        # Create debug configuration
        config = create_debug_config()
        logger.info("✅ Debug configuration created")
        
        # Initialize evaluator
        logger.info("🔧 Initializing evaluator...")
        evaluator = ToxicityEvaluator(config)
        logger.info("✅ Evaluator initialized")
        
        # Get evaluation info
        eval_info = evaluator.get_evaluation_info()
        logger.info("📋 Evaluation Configuration:")
        for key, value in eval_info.items():
            logger.info(f"  {key}: {value}")
        
        # Run evaluation step by step
        logger.info("🚀 Starting step-by-step evaluation...")
        
        # Step 1: Load models
        logger.info("📥 Step 1: Loading models...")
        models = evaluator.model_loader.load_all_models()
        logger.info(f"✅ Loaded {len(models)} models: {list(models.keys())}")
        
        # Step 2: Load dataset
        logger.info("📊 Step 2: Loading dataset...")
        prompts = evaluator.dataset_manager.get_prompts()
        logger.info(f"✅ Loaded {len(prompts)} prompts")
        for i, prompt in enumerate(prompts):
            logger.info(f"  Prompt {i+1}: {prompt[:50]}...")
        
        # Step 3: Load classifiers
        logger.info("🔍 Step 3: Loading classifiers...")
        classifiers = evaluator.classifier_manager.load_classifiers()
        logger.info(f"✅ Loaded {len(classifiers)} classifiers: {list(classifiers.keys())}")
        
        # Step 4: Generate completions
        logger.info("🔄 Step 4: Generating completions...")
        completions = evaluator.generation_engine.generate_all(models, prompts)
        logger.info(f"✅ Generated completions for {len(completions)} models")
        for model_name, model_completions in completions.items():
            logger.info(f"  {model_name}: {len(model_completions)} completions")
            for i, completion in enumerate(model_completions):
                logger.info(f"    Completion {i+1}: {completion[:50]}...")
        
        # Step 5: Generate full texts
        logger.info("📝 Step 5: Generating full texts...")
        full_texts = evaluator.generation_engine.generate_full_texts(models, prompts)
        logger.info(f"✅ Generated full texts for {len(full_texts)} models")
        
        # Step 6: Evaluate toxicity
        logger.info("🔍 Step 6: Evaluating toxicity...")
        toxicity_results = evaluator._evaluate_all_toxicity(prompts, completions, full_texts)
        logger.info(f"✅ Evaluated toxicity for {len(toxicity_results)} text types")
        for text_type, classifier_scores in toxicity_results.items():
            logger.info(f"  {text_type}: {list(classifier_scores.keys())}")
            for classifier_name, scores in classifier_scores.items():
                logger.info(f"    {classifier_name}: {len(scores)} scores, mean={sum(scores)/len(scores):.4f}")
        
        # Step 7: Calculate metrics
        logger.info("📊 Step 7: Calculating metrics...")
        metrics = evaluator.metrics_calculator.calculate_comprehensive_metrics(toxicity_results)
        logger.info("✅ Metrics calculated")
        logger.info(f"  Model metrics: {list(metrics.get('model_metrics', {}).keys())}")
        logger.info(f"  Comparison metrics: {list(metrics.get('comparison_metrics', {}).keys())}")
        
        # Step 8: Create results DataFrame
        logger.info("📋 Step 8: Creating results DataFrame...")
        results_df = evaluator._create_results_dataframe(prompts, completions, full_texts, toxicity_results)
        logger.info(f"✅ DataFrame created with shape {results_df.shape}")
        logger.info(f"  Columns: {list(results_df.columns)}")
        
        # Step 9: Save results
        logger.info("💾 Step 9: Saving results...")
        evaluator._save_results(results_df, metrics, toxicity_results)
        logger.info("✅ Results saved")
        
        # Print final results
        logger.info("\n" + "="*60)
        logger.info("✅ DEBUG EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Results saved to: {evaluator.output_dir}")
        logger.info(f"DataFrame shape: {results_df.shape}")
        
        # Check for toxicity scores
        toxicity_cols = [col for col in results_df.columns if col.endswith('_score')]
        logger.info(f"Toxicity score columns: {toxicity_cols}")
        
        if toxicity_cols:
            for col in toxicity_cols:
                scores = results_df[col].dropna()
                if len(scores) > 0:
                    logger.info(f"{col}: mean={scores.mean():.4f}, std={scores.std():.4f}")
        
        # Print comparison results
        if 'metrics' in locals() and 'comparison_metrics' in metrics:
            comparison = metrics['comparison_metrics']
            if comparison:
                logger.info("\n📊 COMPARISON RESULTS:")
                for model_name, classifier_results in comparison.items():
                    logger.info(f"  {model_name}:")
                    for classifier_name, comp_metrics in classifier_results.items():
                        improvement = comp_metrics.get('improvement', 0)
                        improved_rate = comp_metrics.get('improved_rate', 0)
                        logger.info(f"    {classifier_name}: improvement={improvement:.4f}, improved_rate={improved_rate:.2%}")
        
        logger.info("\n🎉 All debug steps completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_pipeline()
    sys.exit(0 if success else 1) 