"""
Main evaluator class for the Enhanced Toxicity Evaluation Pipeline.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import wandb
import numpy as np

from .model_loader import ModelLoader
from .dataset_manager import DatasetManager
from .classifier_manager import ClassifierManager
from .generation_engine import GenerationEngine
from .metrics_calculator import MetricsCalculator
from .visualizer import ToxicityVisualizer

logger = logging.getLogger(__name__)


class ToxicityEvaluator:
    """Main evaluator class that orchestrates the entire toxicity evaluation pipeline."""
    
    def __init__(self, config: DictConfig):
        """Initialize the evaluator with configuration."""
        self.config = config
        self.experiment_config = config.get("experiment", {})
        self.logging_config = config.get("logging", {})
        self.output_config = config.get("output", {})
        
        # Setup output directory first
        self.output_dir = self._setup_output_directory()
        
        # Initialize components
        self.model_loader = ModelLoader(config.get("models", {}))
        self.dataset_manager = DatasetManager(config.get("dataset", {}))
        self.classifier_manager = ClassifierManager(config.get("classifiers", {}))
        self.generation_engine = GenerationEngine(config)
        self.metrics_calculator = MetricsCalculator(config.get("evaluation", {}))
        self.visualizer = ToxicityVisualizer(self.output_dir, config.get("visualization", {}))
        
        # Setup logging
        self._setup_logging()
        
        # Initialize wandb if enabled
        self.wandb_run = None
        if self.logging_config.get("use_wandb", True):
            self._setup_wandb()
        
        logger.info(f"ToxicityEvaluator initialized with output_dir: {self.output_dir}")
    
    def _setup_output_directory(self) -> Path:
        """Setup output directory for results."""
        output_config = self.output_config.get("local", {})
        base_dir = output_config.get("base_dir", "results/${experiment.name}")
        
        # Replace placeholders
        base_dir = base_dir.replace("${experiment.name}", self.experiment_config.get("name", "default"))
        
        output_dir = Path(base_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["plots", "reports", "logs"]:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        return output_dir
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.logging_config.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / "logs" / "evaluation.log")
            ]
        )
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            self.wandb_run = wandb.init(
                project=self.logging_config.get("wandb_project", "toxicity-evaluation"),
                entity=self.logging_config.get("wandb_entity"),
                name=self.experiment_config.get("name"),
                config=OmegaConf.to_container(self.config, resolve=True),
                tags=self.logging_config.get("wandb_tags", [])
            )
            logger.info(f"WandB initialized: {self.wandb_run.get_url()}")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_run = None
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete toxicity evaluation pipeline."""
        start_time = time.time()
        logger.info("🚀 Starting toxicity evaluation pipeline")
        
        try:
            # Step 1: Load models
            logger.info("📥 Loading models...")
            models = self.model_loader.load_all_models()
            if not models:
                raise ValueError("No models loaded successfully")
            
            # Step 2: Load dataset
            logger.info("📊 Loading dataset...")
            prompts = self.dataset_manager.get_prompts()
            if not prompts:
                raise ValueError("No prompts loaded successfully")
            
            # Step 3: Load classifiers
            logger.info("🔍 Loading toxicity classifiers...")
            classifiers = self.classifier_manager.load_classifiers()
            
            # Step 4: Generate completions
            logger.info("🔄 Generating completions...")
            completions = self.generation_engine.generate_all(models, prompts)
            
            # Step 5: Generate full texts
            logger.info("📝 Generating full texts...")
            full_texts = self.generation_engine.generate_full_texts(models, prompts, completions)
            
            # Step 6: Evaluate toxicity
            logger.info("🔍 Evaluating toxicity...")
            toxicity_results = self._evaluate_all_toxicity(prompts, completions, full_texts)
            
            # Step 7: Calculate metrics
            logger.info("📊 Calculating metrics...")
            metrics = self.metrics_calculator.calculate_comprehensive_metrics(toxicity_results)
            
            # Step 8: Create results DataFrame
            logger.info("📋 Creating results DataFrame...")
            results_df = self._create_results_dataframe(prompts, completions, full_texts, toxicity_results)
            
            # Step 9: Save results
            logger.info("💾 Saving results...")
            self._save_results(results_df, metrics, toxicity_results)
            
            # Step 10: Create visualizations
            logger.info("📊 Creating visualizations...")
            plot_paths = self._create_visualizations(results_df)
            
            # Step 11: Log to WandB
            if self.wandb_run:
                logger.info("📈 Logging to WandB...")
                self._log_to_wandb(results_df, metrics, toxicity_results, plot_paths)
            
            duration = time.time() - start_time
            logger.info(f"✅ Evaluation completed in {duration:.2f} seconds")
            
            return {
                "results_df": results_df,
                "metrics": metrics,
                "toxicity_results": toxicity_results,
                "duration": duration,
                "output_dir": str(self.output_dir)
            }
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
        finally:
            # Cleanup
            self._cleanup()
    
    def _evaluate_all_toxicity(self, prompts: List[str], completions: Dict[str, List[str]], 
                              full_texts: Dict[str, List[str]]) -> Dict[str, Dict[str, List[float]]]:
        """Evaluate toxicity for prompts, outputs, and full texts."""
        evaluation_config = self.config.get("evaluation", {})
        evaluation_types = evaluation_config.get("types", {})
        
        toxicity_results = {}
        
        # Evaluate prompt toxicity
        if evaluation_types.get("prompt_only", True):
            logger.info("Evaluating prompt toxicity...")
            prompt_toxicity = self.classifier_manager.evaluate_texts(prompts, "prompt")
            toxicity_results["prompt"] = self.classifier_manager.extract_detailed_scores(prompt_toxicity)
        
        # Evaluate output toxicity for each model
        if evaluation_types.get("output_only", True):
            logger.info("Evaluating output toxicity...")
            for model_name, model_outputs in completions.items():
                output_toxicity = self.classifier_manager.evaluate_texts(model_outputs, f"output_{model_name}")
                toxicity_results[model_name] = self.classifier_manager.extract_detailed_scores(output_toxicity)
        
        # Evaluate full text toxicity for each model
        if evaluation_types.get("full_text", True):
            logger.info("Evaluating full text toxicity...")
            for model_name, model_full_texts in full_texts.items():
                full_toxicity = self.classifier_manager.evaluate_texts(model_full_texts, f"full_{model_name}")
                toxicity_results[f"full_{model_name}"] = self.classifier_manager.extract_detailed_scores(full_toxicity)
        
        return toxicity_results
    
    def _create_visualizations(self, results_df: pd.DataFrame) -> Dict[str, str]:
        """Create comprehensive visualizations for the evaluation results."""
        try:
            baseline_model = self.config.get("evaluation", {}).get("comparison", {}).get("baseline_model", "base")
            plot_paths = self.visualizer.create_all_plots(results_df, baseline_model)
            logger.info(f"Created {len(plot_paths)} visualization plots")
            return plot_paths
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return {}
    
    def _create_results_dataframe(self, prompts: List[str], completions: Dict[str, List[str]], 
                                 full_texts: Dict[str, List[str]], 
                                 toxicity_results: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
        """Create a comprehensive results DataFrame."""
        logger.info("Creating comprehensive results DataFrame")
        
        # Start with prompts
        data = {
            "prompt": prompts,
            "prompt_index": range(len(prompts))
        }
        
        # Add completions for each model
        for model_name, model_completions in completions.items():
            data[f"output_{model_name}"] = model_completions
            data[f"full_text_{model_name}"] = full_texts.get(model_name, [])
        
        # Add toxicity scores
        for text_type, classifier_scores in toxicity_results.items():
            for classifier_name, scores in classifier_scores.items():
                if text_type == "prompt":
                    column_name = f"prompt_{classifier_name}_score"
                elif text_type.startswith("output_"):
                    model_name = text_type.replace("output_", "")
                    column_name = f"output_{model_name}_{classifier_name}_score"
                elif text_type.startswith("full_"):
                    model_name = text_type.replace("full_", "")
                    column_name = f"full_{model_name}_{classifier_name}_score"
                else:
                    column_name = f"{text_type}_{classifier_name}_score"
                data[column_name] = scores
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate deltas between base model and others
        self._calculate_deltas(df, toxicity_results)
        
        return df
    
    def _calculate_deltas(self, df: pd.DataFrame, toxicity_results: Dict[str, Dict[str, List[float]]]):
        """Calculate delta scores between base model and other models."""
        baseline_model = self.config.get("evaluation", {}).get("comparison", {}).get("baseline_model", "base")
        
        # Find output columns for baseline model
        baseline_output_cols = [col for col in df.columns if col.startswith(f"output_{baseline_model}_") and col.endswith("_score")]
        
        for col in baseline_output_cols:
            classifier_name = col.replace(f"output_{baseline_model}_", "").replace("_score", "")
            
            # Find corresponding columns for other models
            for model_name in [m for m in toxicity_results.keys() if m != baseline_model and not m.startswith("prompt")]:
                other_col = f"output_{model_name}_{classifier_name}_score"
                
                if other_col in df.columns:
                    delta_col = f"delta_{model_name}_vs_{baseline_model}_{classifier_name}_score"
                    df[delta_col] = df[col] - df[other_col]
    
    def _save_results(self, results_df: pd.DataFrame, metrics: Dict[str, Any], 
                     toxicity_results: Dict[str, Dict[str, List[float]]]):
        """Save all results to disk and optionally to HuggingFace."""
        output_config = self.output_config.get("local", {})
        hf_config = self.output_config.get("huggingface", {})
        
        # Save main results DataFrame
        if output_config.get("save_csv", True):
            csv_path = self.output_dir / output_config.get("naming", {}).get("results_file", "toxicity_evaluation_results.csv")
            results_df.to_csv(csv_path, index=False)
            logger.info(f"Saved results to {csv_path}")
        
        # Save metrics summary
        if output_config.get("save_json", True):
            summary_path = self.output_dir / output_config.get("naming", {}).get("summary_file", "evaluation_summary.json")
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                else:
                    return obj
            
            serializable_metrics = convert_numpy_types(metrics)
            
            with open(summary_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2, default=str)
            logger.info(f"Saved summary to {summary_path}")
        
        # Save raw toxicity results
        raw_results_path = self.output_dir / "raw_toxicity_results.json"
        with open(raw_results_path, 'w') as f:
            json.dump(convert_numpy_types(toxicity_results), f, indent=2, default=str)
        logger.info(f"Saved raw toxicity results to {raw_results_path}")
        
        # Save comprehensive dataset export
        self._save_comprehensive_dataset(results_df, metrics, toxicity_results)
        
        # Push to HuggingFace if enabled
        if hf_config.get("enabled", False):
            self._push_to_huggingface(results_df, metrics, toxicity_results, hf_config)
    
    def _push_to_huggingface(self, results_df: pd.DataFrame, metrics: Dict[str, Any], 
                           toxicity_results: Dict[str, Dict[str, List[float]]], hf_config: Dict):
        """Push results to HuggingFace Hub as a dataset."""
        try:
            from datasets import Dataset
            from huggingface_hub import HfApi
            
            dataset_name = hf_config.get("dataset_name", "detoxification-evaluation-results")
            organization = hf_config.get("organization")
            private = hf_config.get("private", False)
            token = hf_config.get("token")
            
            # Create dataset from DataFrame
            hf_dataset = Dataset.from_pandas(results_df)
            
            # Add metadata
            metadata = {
                "description": "Detoxification evaluation results",
                "metrics": metrics,
                "experiment_name": self.experiment_config.get("name"),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            hf_dataset.info.metadata = metadata
            
            # Determine repo ID
            if organization:
                repo_id = f"{organization}/{dataset_name}"
            else:
                repo_id = dataset_name
            
            # Push to HuggingFace
            logger.info(f"Pushing dataset to HuggingFace: {repo_id}")
            hf_dataset.push_to_hub(
                repo_id,
                private=private,
                token=token
            )
            
            logger.info(f"✅ Dataset successfully pushed to HuggingFace: {repo_id}")
        
    except ImportError:
        logger.warning("huggingface_hub not installed. Skipping HuggingFace upload.")
    except Exception as e:
        logger.error(f"Failed to push to HuggingFace: {e}")
    
    def _save_comprehensive_dataset(self, results_df: pd.DataFrame, metrics: Dict[str, Any], 
                                  toxicity_results: Dict[str, Dict[str, List[float]]]):
        """Save comprehensive dataset exports in multiple formats."""
        try:
            # Create dataset directory
            dataset_dir = self.output_dir / "dataset_exports"
            dataset_dir.mkdir(exist_ok=True)
            
            # Save main results in multiple formats
            results_df.to_csv(dataset_dir / "toxicity_evaluation_results.csv", index=False)
            results_df.to_parquet(dataset_dir / "toxicity_evaluation_results.parquet", index=False)
            results_df.to_json(dataset_dir / "toxicity_evaluation_results.json", orient='records', indent=2)
            
            # Save summary statistics
            summary_stats = {}
            for col in results_df.columns:
                if col.endswith('_score'):
                    scores = results_df[col].dropna()
                    if len(scores) > 0:
                        summary_stats[col] = {
                            'count': len(scores),
                            'mean': float(scores.mean()),
                            'std': float(scores.std()),
                            'min': float(scores.min()),
                            'max': float(scores.max()),
                            'median': float(scores.median()),
                            'q25': float(scores.quantile(0.25)),
                            'q75': float(scores.quantile(0.75))
                        }
            
            with open(dataset_dir / "summary_statistics.json", 'w') as f:
                json.dump(summary_stats, f, indent=2)
            
            # Save model comparison data
            delta_cols = [col for col in results_df.columns if col.startswith('delta_')]
            if delta_cols:
                comparison_df = results_df[['prompt'] + delta_cols].copy()
                comparison_df.to_csv(dataset_dir / "model_comparisons.csv", index=False)
                comparison_df.to_json(dataset_dir / "model_comparisons.json", orient='records', indent=2)
            
            # Save prompt analysis
            prompt_analysis = {
                'total_prompts': len(results_df),
                'prompt_lengths': [len(p) for p in results_df['prompt']],
                'unique_prompts': len(results_df['prompt'].unique()),
                'prompt_samples': results_df['prompt'].head(100).tolist()
            }
            
            with open(dataset_dir / "prompt_analysis.json", 'w') as f:
                json.dump(prompt_analysis, f, indent=2)
            
            # Save configuration snapshot
            config_snapshot = {
                'experiment_name': self.experiment_config.get("name"),
                'models': list(self.model_loader.models.keys()) if hasattr(self.model_loader, 'models') else [],
                'classifiers': list(self.classifier_manager.classifiers.keys()) if hasattr(self.classifier_manager, 'classifiers') else [],
                'generation_params': self.generation_engine.get_generation_info(),
                'dataset_info': self.dataset_manager.get_dataset_info(),
                'evaluation_config': self.config.get("evaluation", {}),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(dataset_dir / "experiment_config.json", 'w') as f:
                json.dump(config_snapshot, f, indent=2)
            
            # Create README for the dataset
            readme_content = f"""# Toxicity Evaluation Dataset

## Overview
This dataset contains comprehensive toxicity evaluation results for multiple language models.

## Files
- `toxicity_evaluation_results.csv` - Main results with all scores and comparisons
- `toxicity_evaluation_results.parquet` - Same data in Parquet format (more efficient)
- `toxicity_evaluation_results.json` - Same data in JSON format
- `summary_statistics.json` - Statistical summary of all toxicity scores
- `model_comparisons.csv` - Direct model comparison data
- `prompt_analysis.json` - Analysis of the prompts used
- `experiment_config.json` - Complete experiment configuration

## Experiment Details
- **Experiment Name**: {self.experiment_config.get("name")}
- **Total Prompts**: {len(results_df)}
- **Models Evaluated**: {len([col for col in results_df.columns if col.startswith('output_') and not col.endswith('_score')])}
- **Classifiers Used**: {len(set([col.split('_')[-2] for col in results_df.columns if col.endswith('_score') and 'prompt_' not in col]))}

## Usage
Load the main results:
```python
import pandas as pd
df = pd.read_csv('toxicity_evaluation_results.csv')
```

## Citation
If you use this dataset, please cite the original RealToxicityPrompts paper and this evaluation framework.
"""
            
            with open(dataset_dir / "README.md", 'w') as f:
                f.write(readme_content)
            
            logger.info(f"Saved comprehensive dataset exports to {dataset_dir}")
            
        except Exception as e:
            logger.error(f"Error saving comprehensive dataset: {e}")
    
    def _log_to_wandb(self, results_df: pd.DataFrame, metrics: Dict[str, Any], 
                     toxicity_results: Dict[str, Dict[str, List[float]]], 
                     plot_paths: Dict[str, str] = None):
        """Log results to Weights & Biases."""
        if not self.wandb_run:
            return
        
        try:
            # Log basic metrics
            wandb.log({
                "total_prompts": len(results_df),
                "total_models": len([col for col in results_df.columns if col.startswith("output_") and not col.endswith("_score")]),
                "total_classifiers": len([col for col in results_df.columns if col.endswith("_score")])
            })
            
            # Log model performance metrics
            for model_name, model_metrics in metrics.get("model_metrics", {}).items():
                wandb.log({
                    f"{model_name}_mean_toxicity": model_metrics.get("mean", 0.0),
                    f"{model_name}_std_toxicity": model_metrics.get("std", 0.0),
                    f"{model_name}_median_toxicity": model_metrics.get("median", 0.0)
                })
            
            # Log comparison metrics
            for model_name, comparison_metrics in metrics.get("comparison_metrics", {}).items():
                for classifier_name, comp_metrics in comparison_metrics.items():
                    wandb.log({
                        f"{model_name}_vs_base_{classifier_name}_improvement": comp_metrics.get("improvement", 0.0),
                        f"{model_name}_vs_base_{classifier_name}_improved_rate": comp_metrics.get("improved_rate", 0.0),
                        f"{model_name}_vs_base_{classifier_name}_baseline_mean": comp_metrics.get("baseline_mean", 0.0),
                        f"{model_name}_vs_base_{classifier_name}_model_mean": comp_metrics.get("model_mean", 0.0)
                    })
            
            # Log toxicity score distributions
            toxicity_cols = [col for col in results_df.columns if col.endswith('_score')]
            for col in toxicity_cols:
                scores = results_df[col].dropna()
                if len(scores) > 0:
                    wandb.log({
                        f"{col}_histogram": wandb.Histogram(scores),
                        f"{col}_mean": scores.mean(),
                        f"{col}_std": scores.std()
                    })
            
            # Log sample results table
            sample_df = results_df.head(100)  # Log first 100 rows
            wandb.log({"sample_results": wandb.Table(dataframe=sample_df)})
            
            # Log configuration
            wandb.config.update({
                "experiment_name": self.experiment_config.get("name"),
                "models": list(self.model_loader.models.keys()) if hasattr(self.model_loader, 'models') else [],
                "classifiers": list(self.classifier_manager.classifiers.keys()) if hasattr(self.classifier_manager, 'classifiers') else []
            })
            
            # Save files as artifacts
            wandb.save(str(self.output_dir / "*.csv"))
            wandb.save(str(self.output_dir / "*.json"))
            
            # Log plots if available
            if plot_paths:
                for plot_name, plot_path in plot_paths.items():
                    if plot_path.endswith('.png'):
                        wandb.log({f"plots/{plot_name}": wandb.Image(plot_path)})
            
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")
    
    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        # Cleanup models
        self.model_loader.cleanup()
        
        # Cleanup classifiers
        self.classifier_manager.cleanup()
        
        # Finish WandB run
        if self.wandb_run:
            wandb.finish()
        
        logger.info("Cleanup completed")
    
    def get_evaluation_info(self) -> Dict[str, Any]:
        """Get information about the evaluation configuration."""
        return {
            "experiment_name": self.experiment_config.get("name"),
            "output_dir": str(self.output_dir),
            "models": list(self.model_loader.models.keys()) if hasattr(self.model_loader, 'models') else [],
            "classifiers": list(self.classifier_manager.classifiers.keys()) if hasattr(self.classifier_manager, 'classifiers') else [],
            "generation_params": self.generation_engine.get_generation_info(),
            "dataset_info": self.dataset_manager.get_dataset_info()
        } 