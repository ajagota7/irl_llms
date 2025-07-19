"""
Evaluation pipeline for the Enhanced Toxicity Evaluation Pipeline.
Orchestrates classification and evaluation processes separately.
"""

import logging
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig
from pathlib import Path
import yaml

from .model_loader import ModelLoader
from .classifier_manager import ClassifierManager
from .generation_engine import GenerationEngine
from .dataset_manager import DatasetManager
from .results_manager import ResultsManager
from .visualization_manager import VisualizationManager

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Orchestrates the complete evaluation pipeline with separate classification and evaluation phases."""
    
    def __init__(self, config: DictConfig):
        """Initialize the evaluation pipeline with configuration."""
        self.config = config
        self.model_loader = ModelLoader(config)
        self.classifier_manager = ClassifierManager(config)
        self.generation_engine = GenerationEngine(config)
        self.dataset_manager = DatasetManager(config)
        self.results_manager = ResultsManager(config)
        
        logger.info("EvaluationPipeline initialized")
    
    def run_classification_phase(self) -> Dict[str, Any]:
        """Run the classification phase: load models, generate outputs, and classify everything."""
        logger.info("🚀 Starting Classification Phase")
        logger.info("=" * 60)
        
        # Check GPU availability
        import torch
        if torch.cuda.is_available():
            logger.info(f"🔥 GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"🔥 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.info("💻 Using CPU for computation")
        
        try:
            # Load dataset
            logger.info("📥 Loading dataset...")
            prompts = self.dataset_manager.get_prompts()
            logger.info(f"✅ Loaded {len(prompts)} prompts")
            
            # Load models and tokenizers
            logger.info("🔧 Loading models...")
            models, tokenizers = self.model_loader.load_models()
            logger.info(f"✅ Loaded {len(models)} models")
            
            # Load classifiers
            logger.info("🔧 Loading classifiers...")
            classifiers = self.classifier_manager.load_classifiers()
            logger.info(f"✅ Loaded {len(classifiers)} classifiers")
            
            # Generate outputs
            logger.info("🔄 Generating outputs...")
            max_new_tokens = self.config.get("generation", {}).get("max_new_tokens", 50)
            model_outputs = self.generation_engine.generate_outputs(
                models, tokenizers, prompts, max_new_tokens
            )
            logger.info(f"✅ Generated outputs for {len(model_outputs)} models")
            
            # Create comprehensive results
            logger.info("📊 Creating comprehensive results...")
            model_dfs = self.results_manager.create_comprehensive_results(
                prompts, model_outputs, classifiers
            )
            logger.info(f"✅ Created results for {len(model_dfs)} models")
            
            # Save results
            output_dir = self.config.get("output", {}).get("directory", "real_model_results")
            output_path = self.results_manager.save_results(model_dfs, model_outputs, output_dir)
            logger.info(f"✅ Saved results to {output_path}")
            
            # Print classifier summary for first model
            first_model_df = list(model_dfs.values())[0]
            self.results_manager.print_classifier_summary(first_model_df)
            
            # Display summary
            logger.info("\n📋 CLASSIFICATION PHASE SUMMARY:")
            logger.info("=" * 30)
            
            for model_name, outputs in model_outputs.items():
                model_num = list(model_outputs.keys()).index(model_name) + 1
                logger.info(f"\nMODEL {model_num} ({model_name.upper()}):")
                logger.info(f"  Generated {len(outputs)} outputs")
                if outputs:
                    logger.info(f"  Sample output: {outputs[0][:100]}...")
            
            # Show classification results for first sample
            first_row = first_model_df.iloc[0]
            logger.info(f"\nFIRST SAMPLE CLASSIFICATIONS:")
            logger.info(f"  Prompt: {first_row['prompt'][:50]}...")
            
            # Show toxic-bert results specifically
            logger.info(f"\nTOXIC-BERT CLASSIFICATIONS (First Sample):")
            for col in first_model_df.columns:
                if col.endswith('_results') and isinstance(first_row[col], dict) and 'toxic_bert' in col:
                    logger.info(f"  {col}: {first_row[col]}")
            
            # Show other classifier results
            logger.info(f"\nOTHER CLASSIFIER RESULTS (First Sample):")
            for col in first_model_df.columns:
                if col.endswith('_results') and isinstance(first_row[col], dict) and 'toxic_bert' not in col:
                    logger.info(f"  {col}: {first_row[col]}")
            
            logger.info(f"\n📁 All results saved to: {output_path}")
            logger.info("🎉 CLASSIFICATION PHASE COMPLETED!")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "model_dfs": model_dfs,
                "model_outputs": model_outputs,
                "prompts": prompts,
                "output_path": output_path,
                "classifiers": classifiers
            }
            
        except Exception as e:
            logger.error(f"❌ Classification phase failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_evaluation_phase(self, results_data: Optional[Dict[str, Any]] = None, 
                           results_path: Optional[Path] = None) -> Dict[str, Any]:
        """Run the evaluation phase: create visualizations and insights from classified data."""
        logger.info("🚀 Starting Evaluation Phase")
        logger.info("=" * 60)
        
        try:
            # Load results if not provided
            if results_data is None:
                if results_path is None:
                    results_path = Path(self.config.get("output", {}).get("directory", "real_model_results"))
                
                logger.info(f"📥 Loading results from {results_path}")
                results_data = self._load_results_from_path(results_path)
            
            model_dfs = results_data["model_dfs"]
            output_path = results_data["output_path"]
            
            # Initialize visualization manager
            viz_manager = VisualizationManager(self.config, output_path)
            
            # Create visualizations
            logger.info("📊 Creating visualizations...")
            
            # Create toxicity reduction plots
            self._create_toxicity_plots(model_dfs, output_path)
            
            # Create prompt comparison plots
            self._create_prompt_comparison_plots(model_dfs, output_path)
            
            # Create interactive plots
            self._create_interactive_plots(model_dfs, output_path)
            
            # Create comprehensive visualizations using visualization manager
            comprehensive_df = self._create_comprehensive_dataframe(model_dfs)
            viz_manager.create_comprehensive_visualizations(comprehensive_df, {})
            
            logger.info("✅ All visualizations created!")
            
            # Display evaluation summary
            logger.info("\n📋 EVALUATION PHASE SUMMARY:")
            logger.info("=" * 30)
            logger.info(f"📊 Created visualizations for {len(model_dfs)} models")
            logger.info(f"📁 Visualizations saved to: {output_path}")
            logger.info("🎉 EVALUATION PHASE COMPLETED!")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "output_path": output_path,
                "visualizations_created": True
            }
            
        except Exception as e:
            logger.error(f"❌ Evaluation phase failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline: classification followed by evaluation."""
        logger.info("🚀 Starting Full Evaluation Pipeline")
        logger.info("=" * 60)
        
        # Run classification phase
        classification_result = self.run_classification_phase()
        
        if not classification_result["success"]:
            logger.error("❌ Classification phase failed, stopping pipeline")
            return classification_result
        
        # Run evaluation phase
        evaluation_result = self.run_evaluation_phase(classification_result)
        
        if not evaluation_result["success"]:
            logger.error("❌ Evaluation phase failed")
            return evaluation_result
        
        logger.info("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        return {
            "success": True,
            "classification_result": classification_result,
            "evaluation_result": evaluation_result
        }
    
    def _load_results_from_path(self, results_path: Path) -> Dict[str, Any]:
        """Load results from a saved path."""
        # Load comprehensive results
        comprehensive_path = results_path / "comprehensive_results.json"
        if comprehensive_path.exists():
            import pandas as pd
            comprehensive_df = pd.read_json(comprehensive_path)
            
            # Reconstruct model_dfs
            model_dfs = {}
            for model_name in comprehensive_df['model'].unique():
                model_data = comprehensive_df[comprehensive_df['model'] == model_name]
                model_dfs[model_name] = model_data
            
            return {
                "model_dfs": model_dfs,
                "output_path": results_path
            }
        else:
            raise FileNotFoundError(f"Results not found at {results_path}")
    
    def _create_comprehensive_dataframe(self, model_dfs: Dict[str, Any]) -> Any:
        """Create a comprehensive DataFrame from model results."""
        import pandas as pd
        
        comprehensive_data = []
        for model_name, model_df in model_dfs.items():
            for _, row in model_df.iterrows():
                comprehensive_row = {
                    "model": model_name,
                    "prompt": row["prompt"],
                    "prompt_index": row["prompt_index"],
                    "output": row["output"],
                    "full_text": row["full_text"]
                }
                # Add all classification results
                for col in model_df.columns:
                    if col.endswith('_results'):
                        comprehensive_row[col] = row[col]
                comprehensive_data.append(comprehensive_row)
        
        return pd.DataFrame(comprehensive_data)
    
    def _create_toxicity_plots(self, model_dfs: Dict[str, Any], output_path: Path):
        """Create toxicity reduction plots."""
        logger.info("📊 Creating toxicity reduction plots...")
        
        # Import plotting functions from real_model_test.py
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Extract epoch numbers and create mapping
        model_epochs = {}
        for model_name in model_dfs.keys():
            if model_name == "base":
                model_epochs[model_name] = 0  # Base model = epoch 0
            elif "epoch" in model_name:
                # Extract epoch number from name like "detox_epoch_20"
                epoch_num = int(model_name.split("_")[-1])
                model_epochs[model_name] = epoch_num
            else:
                # For any other models, assign a default epoch
                model_epochs[model_name] = 100
        
        # Sort models by epoch
        sorted_models = sorted(model_epochs.items(), key=lambda x: x[1])
        model_names = [name for name, _ in sorted_models]
        epochs = [epoch for _, epoch in sorted_models]
        
        # Create plots for each classifier and text type
        classifiers = ["toxic_bert", "roberta_toxicity", "dynabench_hate"]
        text_types = ["output", "full_text"]
        
        for classifier in classifiers:
            for text_type in text_types:
                logger.info(f"  Creating plot for {classifier} - {text_type}")
                
                # Create figure
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f'Toxicity Reduction: {classifier} - {text_type}', fontsize=16, fontweight='bold')
                
                # Plot 1: Mean toxicity across epochs
                mean_scores = []
                for model_name in model_names:
                    if model_name in model_dfs:
                        df = model_dfs[model_name]
                        col_name = f"{text_type}_{classifier}_results"
                        
                        if col_name in df.columns:
                            # Extract toxicity scores
                            scores = []
                            for _, row in df.iterrows():
                                if isinstance(row[col_name], dict):
                                    if classifier == "toxic_bert":
                                        # Use 'toxic' category for toxic-bert
                                        scores.append(row[col_name].get('toxic', 0.0))
                                    elif classifier == "roberta_toxicity":
                                        # Use 'toxic' category for roberta
                                        scores.append(row[col_name].get('toxic', 0.0))
                                    elif classifier == "dynabench_hate":
                                        # Use 'hate' category for dynabench
                                        scores.append(row[col_name].get('hate', 0.0))
                            
                            if scores:
                                mean_scores.append(np.mean(scores))
                            else:
                                mean_scores.append(0.0)
                        else:
                            mean_scores.append(0.0)
                    else:
                        mean_scores.append(0.0)
                
                # Plot mean toxicity progression
                axes[0].plot(epochs, mean_scores, 'o-', linewidth=2, markersize=8, label='Mean Toxicity')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Mean Toxicity Score')
                axes[0].set_title(f'Mean Toxicity Across Epochs')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
                
                # Plot 2: Scatter plot of individual prompt toxicity
                all_scores = []
                all_epochs = []
                all_prompts = []
                
                for i, model_name in enumerate(model_names):
                    if model_name in model_dfs:
                        df = model_dfs[model_name]
                        col_name = f"{text_type}_{classifier}_results"
                        
                        if col_name in df.columns:
                            for j, (_, row) in enumerate(df.iterrows()):
                                if isinstance(row[col_name], dict):
                                    if classifier == "toxic_bert":
                                        score = row[col_name].get('toxic', 0.0)
                                    elif classifier == "roberta_toxicity":
                                        score = row[col_name].get('toxic', 0.0)
                                    elif classifier == "dynabench_hate":
                                        score = row[col_name].get('hate', 0.0)
                                    else:
                                        score = 0.0
                                    
                                    all_scores.append(score)
                                    all_epochs.append(epochs[i])
                                    all_prompts.append(j)
                
                # Create scatter plot
                scatter = axes[1].scatter(all_epochs, all_scores, c=all_prompts, cmap='viridis', alpha=0.6, s=30)
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Toxicity Score')
                axes[1].set_title(f'Individual Prompt Toxicity')
                axes[1].grid(True, alpha=0.3)
                
                # Add colorbar for prompt indices
                cbar = plt.colorbar(scatter, ax=axes[1])
                cbar.set_label('Prompt Index')
                
                # Add improvement annotations
                if len(mean_scores) > 1:
                    improvement = mean_scores[0] - mean_scores[-1]  # Base - Final
                    improvement_pct = (improvement / mean_scores[0]) * 100 if mean_scores[0] > 0 else 0
                    axes[0].text(0.02, 0.98, f'Total Reduction: {improvement:.3f} ({improvement_pct:.1f}%)', 
                               transform=axes[0].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                
                # Save plot
                plot_path = output_path / f"toxicity_reduction_{classifier}_{text_type}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"    Saved plot to {plot_path}")
                plt.close()
    
    def _create_prompt_comparison_plots(self, model_dfs: Dict[str, Any], output_path: Path):
        """Create prompt comparison plots."""
        logger.info("📊 Creating prompt comparison plots...")
        # Implementation would be similar to the toxicity plots
        # This is a placeholder for the full implementation
        pass
    
    def _create_interactive_plots(self, model_dfs: Dict[str, Any], output_path: Path):
        """Create interactive plots."""
        logger.info("📊 Creating interactive plots...")
        # Implementation would use plotly for interactive visualizations
        # This is a placeholder for the full implementation
        pass
    
    def cleanup(self):
        """Clean up all components."""
        logger.info("Cleaning up evaluation pipeline...")
        
        self.model_loader.cleanup()
        self.classifier_manager.cleanup()
        
        logger.info("Evaluation pipeline cleanup completed") 