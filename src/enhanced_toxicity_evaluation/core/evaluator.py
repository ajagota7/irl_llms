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
from .visualization_manager import VisualizationManager
from .inspector import ToxicityInspector
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
        self.visualization_manager = VisualizationManager(config, self.output_dir)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize wandb if enabled (now handled by visualization manager)
        self.wandb_run = None
        
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
        """Setup Weights & Biases logging (deprecated - now handled by visualization manager)."""
        logger.warning("_setup_wandb is deprecated. WandB is now handled by VisualizationManager.")
        pass
    
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
            
            # Step 10: Create comprehensive visualizations and log to WandB
            logger.info("📈 Creating visualizations and logging to WandB...")
            self.visualization_manager.create_comprehensive_visualizations(results_df, metrics)
            
            # Step 11: Generate enhanced outputs
            if self.output_config.get("local", {}).get("enhanced_analysis", True):
                logger.info("🔍 Generating enhanced analysis outputs...")
                self.generate_enhanced_outputs(results_df, metrics)
            
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
    
    def _create_results_dataframe(self, prompts: List[str], completions: Dict[str, List[str]], 
                                 full_texts: Dict[str, List[str]], 
                                 toxicity_results: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
        """Create a comprehensive results DataFrame with delta analysis."""
        # Create base DataFrame
        data = {
            "prompt": prompts,
            "prompt_index": range(len(prompts))
        }
        
        # Add completions
        for model_name, model_completions in completions.items():
            data[f"output_{model_name}"] = model_completions
        
        # Add full texts
        for model_name, model_full_texts in full_texts.items():
            data[f"full_text_{model_name}"] = model_full_texts
        
        df = pd.DataFrame(data)
        
        # Add toxicity scores
        for text_type, classifier_scores in toxicity_results.items():
            for classifier_name, scores in classifier_scores.items():
                column_name = f"{text_type}_{classifier_name}"
                if len(scores) == len(df):
                    df[column_name] = scores
                else:
                    logger.warning(f"Score length mismatch for {column_name}: {len(scores)} vs {len(df)}")
        
        # Calculate and add delta columns
        baseline_model = self.config.get("evaluation", {}).get("comparison", {}).get("baseline_model", "base")
        delta_results = self.metrics_calculator.calculate_delta_metrics(toxicity_results, baseline_model)
        
        for model_name, model_deltas in delta_results.items():
            for metric_name, deltas in model_deltas.items():
                if len(deltas) == len(df):
                    df[metric_name] = deltas
                else:
                    logger.warning(f"Delta length mismatch for {metric_name}: {len(deltas)} vs {len(df)}")
        
        return df

    def create_inspector(self, results_df: pd.DataFrame) -> ToxicityInspector:
        """Create inspector instance for interactive analysis."""
        inspector = ToxicityInspector(results_df, self.output_dir / "inspection")
        logger.info("Inspector created for interactive analysis")
        return inspector

    def create_visualizer(self, results_df: pd.DataFrame) -> ToxicityVisualizer:
        """Create visualizer instance for enhanced plotting."""
        visualizer = ToxicityVisualizer(results_df, self.output_dir / "visualizations")
        logger.info("Visualizer created for enhanced plotting")
        return visualizer

    def generate_enhanced_outputs(self, results_df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        """Generate enhanced outputs using inspector and visualizer."""
        try:
            # Create inspector and visualizer
            inspector = self.create_inspector(results_df)
            visualizer = self.create_visualizer(results_df)
            
            # Generate inspector analysis
            logger.info("Generating inspector analysis...")
            inspector.export_analysis_report()
            
            # Generate visualizations
            logger.info("Generating enhanced visualizations...")
            visualizer.plot_toxicity_distributions()
            visualizer.create_comprehensive_dashboard()
            
            # Generate model-specific analyses
            models = [m for m in inspector.models if m != "base"]
            for model in models:
                try:
                    visualizer.plot_model_comparison_scatter(model)
                    visualizer.plot_delta_analysis(model)
                    
                    # Generate toxic-bert analysis if available
                    if inspector.toxic_bert_categories:
                        visualizer.plot_toxic_bert_categories(model)
                except Exception as e:
                    logger.warning(f"Failed to generate analysis for model {model}: {e}")
            
            logger.info("Enhanced outputs generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced outputs: {e}")
    
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
    
    def _log_to_wandb(self, results_df: pd.DataFrame, metrics: Dict[str, Any], 
                     toxicity_results: Dict[str, Dict[str, List[float]]]):
        """Log results to Weights & Biases (deprecated - now handled by visualization manager)."""
        logger.warning("_log_to_wandb is deprecated. WandB logging is now handled by VisualizationManager.")
        pass
    
    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        # Cleanup models
        self.model_loader.cleanup()
        
        # Cleanup classifiers
        self.classifier_manager.cleanup()
        
        # Cleanup visualization manager (includes WandB cleanup)
        self.visualization_manager.cleanup()
        
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